"""Real, token-clock rollouts with explicit player and reset boundaries.

Only policy-safe A04 observations enter PolicyValueNet.  Privileged ownership
is collected separately, and only when a centralized Q critic was supplied.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, replace
import hashlib
import json
from time import monotonic
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from ..encoding import encode_observation
from ..env_batch import GuandanEnvBatch
from ..environment_types import BatchObservation, Observation
from ..model import PolicyValueNet
from ..state import TEAM_OF
from .estimators import expand_team_values
from .runtime import BudgetExceeded, Deadline, TRAIN_SEED_BASE, EVALUATION_SEED_BASE


class IncompleteTrainingError(RuntimeError):
    """A fixed target was not reached; ``diagnostic`` is machine-readable."""

    def __init__(self, diagnostic: Mapping[str, Any]):
        self.diagnostic = {"status": "incomplete", **diagnostic}
        super().__init__(str(self.diagnostic))


def check_deadline(deadline: Any) -> None:
    """Accept runtime.Deadline or its check() callback, with no target scaling."""
    if deadline is not None:
        check = getattr(deadline, "check", deadline)
        if not callable(check):
            raise TypeError("deadline must be callable or expose check()")
        check()


def policy_inputs(observation: BatchObservation, device: torch.device | str) -> tuple[Tensor, ...]:
    """Drop padding only, not history or legal actions, before a model forward."""
    tokens = observation.observation_tokens
    candidates = observation.legal_next_tokens
    mask = observation.legal_next_mask
    token_columns = np.flatnonzero(np.any(tokens != 0, axis=0))
    legal_columns = np.flatnonzero(np.any(mask, axis=0))
    history_width = int(token_columns[-1]) + 1 if len(token_columns) else 1
    legal_width = int(legal_columns[-1]) + 1 if len(legal_columns) else 1
    return (
        torch.as_tensor(tokens[:, :history_width], dtype=torch.long, device=device),
        torch.as_tensor(observation.state_channels, dtype=torch.float32, device=device),
        torch.as_tensor(candidates[:, :legal_width], dtype=torch.long, device=device),
        torch.as_tensor(mask[:, :legal_width], dtype=torch.bool, device=device),
    )


def _replace_observation_row(batch: BatchObservation, index: int, row: Observation) -> None:
    # The destination is an owned copy, never an environment-owned allocation.
    for field in fields(Observation):
        if field.name != "private_hand_card_ids":
            getattr(batch, field.name)[index] = getattr(row, field.name)
    batch.present[index] = True


def _take_observations(observation: BatchObservation, indices: np.ndarray) -> BatchObservation:
    return replace(observation, **{
        field.name: getattr(observation, field.name)[indices].copy()
        for field in fields(BatchObservation)
        if isinstance(getattr(observation, field.name), np.ndarray)
    })


def concatenate_observations(observations: Sequence[BatchObservation]) -> BatchObservation:
    if not observations:
        raise ValueError("observations must not be empty")
    return replace(observations[0], **{
        field.name: np.concatenate([getattr(obs, field.name) for obs in observations])
        for field in fields(BatchObservation)
        if isinstance(getattr(observations[0], field.name), np.ndarray)
    })


def truncation_bootstrap_observation(batch: GuandanEnvBatch, index: int) -> Observation:
    """Encode a pre-reset time-limit state without stepping or changing it.

    A04 intentionally returns no observation for truncated slots.  Its public
    clone is isolated; clearing ONLY the cloned protocol's truncation flag
    permits the regular policy-safe encoder.  The token limit, history, token
    clock, committed clock and pending action prefix are left intact.  This is
    the sole private-protocol access needed until A04 exposes a bootstrap API.
    """
    env = batch.envs[index]
    if not env.truncated or env.done:
        raise ValueError("bootstrap snapshot requires truncation, not true termination")
    clone = env.clone()
    protocol = clone._protocol
    if protocol is None:
        raise RuntimeError("truncated slot has no protocol")
    protocol.truncated = False
    return encode_observation(protocol, clone.observation_spec)


def critic_state(batch: GuandanEnvBatch, observation: BatchObservation) -> Tensor:
    """[B,688]: absolute seat/card ownership (4*108), then actor channels.

    The privileged tensor is never merged into an observation or actor input.
    Terminal rows remain zero and are never evaluated by the critic.
    """
    result = np.zeros((len(batch.envs), 688), dtype=np.float32)
    for index, env in enumerate(batch.envs):
        if not observation.present[index]:
            continue
        hands = env.full_state()["round_state"]["hands"]
        for player, cards in enumerate(hands):
            for card_id in cards:
                result[index, player * 108 + int(card_id)] = 1.0
        result[index, 432:] = observation.state_channels[index]
    return torch.from_numpy(result)


@dataclass(frozen=True)
class Transition:
    observation: BatchObservation
    next_observation: BatchObservation  # raw step result, BEFORE any reset
    bootstrap_observation: BatchObservation  # truncation rows safely encoded
    episode_indices: Tensor
    player_ids: Tensor
    team_ids: Tensor
    token_step_before: Tensor
    token_step_after: Tensor
    committed_step_before: Tensor
    committed_step_after: Tensor
    actions: Tensor  # actual token IDs, not legal-row indices
    legal_mask: Tensor
    old_log_probs: Tensor
    values: Tensor  # [B,4], fixed-seat perspective
    next_values: Tensor  # [B,4]; true terminal=0, truncation bootstraps
    rewards: Tensor  # [B,4], untouched environment reward vector
    acting_player_rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    is_commit: Tensor
    info: tuple[dict, ...]
    critic_states: Tensor | None = None
    selected_q: Tensor | None = None

    @property
    def value(self) -> Tensor:
        return self.values.gather(1, self.player_ids[:, None]).squeeze(1)

    @property
    def done(self) -> Tensor:
        return self.terminated

    @property
    def action_tokens(self) -> Tensor:
        return self.actions


@dataclass(frozen=True)
class Rollout:
    transitions: tuple[Transition, ...]

    @property
    def ticks(self) -> int:
        return len(self.transitions)

    @property
    def batch_size(self) -> int:
        return len(self.transitions[0].actions)

    @property
    def token_transitions(self) -> int:
        return self.ticks * self.batch_size

    def stack(self, name: str) -> Tensor:
        values = [getattr(row, name) for row in self.transitions]
        if any(value is None for value in values):
            raise ValueError(f"rollout field {name!r} is unavailable")
        return torch.stack(values)

    @property
    def observations(self) -> BatchObservation:
        return concatenate_observations([row.observation for row in self.transitions])

    @property
    def rewards(self) -> Tensor:
        return self.stack("rewards")

    @property
    def values(self) -> Tensor:
        return self.stack("values")

    @property
    def next_values(self) -> Tensor:
        return self.stack("next_values")

    @property
    def terminated(self) -> Tensor:
        return self.stack("terminated")

    @property
    def truncated(self) -> Tensor:
        return self.stack("truncated")

    def digest(self) -> str:
        """SHA256 of named, shaped, time-major transition arrays (not targets)."""
        digest = hashlib.sha256()
        for name in ("episode_indices", "player_ids", "team_ids", "actions", "rewards",
                     "token_step_before", "token_step_after", "committed_step_before",
                     "committed_step_after", "terminated", "truncated"):
            array = self.stack(name).cpu().numpy()
            digest.update(json.dumps([name, str(array.dtype), array.shape]).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def terminal_snapshots(self) -> list[dict[str, Any]]:
        records = []
        for tick, row in enumerate(self.transitions):
            for slot in torch.nonzero(row.terminated, as_tuple=False).flatten().tolist():
                records.append({
                    "tick": tick, "env_index": slot,
                    "episode_index": int(row.episode_indices[slot]),
                    "actor": int(row.player_ids[slot]), "team": int(row.team_ids[slot]),
                    "token_step": int(row.token_step_after[slot]),
                    "committed_step": int(row.committed_step_after[slot]),
                    "ranking": list(row.info[slot].get("ranking", ())),
                    "winner_team": row.info[slot].get("winner_team"),
                    "outcome_class": row.info[slot].get("outcome_class"),
                    "rewards": row.rewards[slot].tolist(),
                    "acting_player_reward": float(row.acting_player_rewards[slot]),
                })
        return records


class RolloutCollector:
    """One real ``batch.step`` per tick, all live slots, post-capture reset.

    Pass an already initialized batch.  Batch auto-reset is rejected because it
    discards the pre-reset observation needed for time-limit bootstrapping.
    Optional initial states are ONLY for dedicated boundary tests, not counted
    acceptance rollouts.  The training engine never supplies such fixtures.
    """

    def __init__(
        self, batch: GuandanEnvBatch, model: PolicyValueNet, *, seed: int = 0,
        critic: torch.nn.Module | None = None,
        expected_values: Callable[..., Tensor] | None = None,
        initial_states: Sequence[bytes | None] | None = None,
        reset_seed_base: int = TRAIN_SEED_BASE,
    ):
        if batch.auto_reset:
            raise ValueError("collector requires batch.auto_reset=False; it resets after capture")
        if (critic is None) != (expected_values is None):
            raise ValueError("critic and masked_expected_values provider must be supplied together")
        if type(reset_seed_base) is not int or not TRAIN_SEED_BASE <= reset_seed_base < EVALUATION_SEED_BASE:
            raise ValueError("reset_seed_base must lie in the reserved training namespace")
        self.reset_seed_base = reset_seed_base
        self.batch, self.model, self.critic = batch, model, critic
        self.expected_values = expected_values
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.initial_states = tuple(initial_states) if initial_states is not None else None
        if self.initial_states is not None and len(self.initial_states) != len(batch.envs):
            raise ValueError("initial_states must match batch size")
        self.observation = batch.observe()
        self._check_live(self.observation)
        self.total_token_steps = 0
        self.total_committed_steps = 0
        self.ticks_per_env = np.zeros(len(batch.envs), dtype=np.int64)
        self.reset_counts = np.zeros(len(batch.envs), dtype=np.int64)
        self.terminal_count = 0
        self.truncation_count = 0
        self.incomplete = False
        self.deal_records = [self._deal_record(i, env.config.seed) for i, env in enumerate(batch.envs)]

    def _deal_record(self, index: int, seed: int | None) -> dict[str, Any]:
        hands = self.batch.envs[index].full_state()["round_state"]["hands"]
        # Only a digest and cardinalities are exposed in evidence; hidden hands
        # themselves are never added to a policy observation.
        canonical = json.dumps([sorted(hand) for hand in hands], separators=(",", ":"))
        return {"env_index": index, "episode_index": int(self.reset_counts[index]),
                "seed": seed, "hand_sizes": [len(hand) for hand in hands],
                "deal_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest()}

    @staticmethod
    def _check_live(observation: BatchObservation) -> None:
        if observation.present is None or not np.all(observation.present):
            raise ValueError("every collector slot must have a live observation")
        if np.any((observation.player_id < 0) | (observation.player_id > 3)):
            raise ValueError("invalid acting player in batch observation")
        expected_teams = np.asarray(TEAM_OF)[observation.player_id]
        if not np.array_equal(observation.team_id, expected_teams):
            raise ValueError("observation player/team perspective mismatch")
        if not observation.legal_next_mask.any(axis=1).all():
            raise ValueError("live observation has no legal token")

    @torch.no_grad()
    def _evaluate(self, observation: BatchObservation, state: Tensor | None = None):
        inputs = policy_inputs(observation, self.device)
        logits, scalar_values = self.model(*inputs)
        probabilities = torch.softmax(logits, dim=-1)
        q = None
        if self.critic is None:
            values = expand_team_values(
                scalar_values, torch.as_tensor(observation.player_id, dtype=torch.long, device=self.device)
            )
        else:
            if state is None:
                raise ValueError("centralized critic requires a separate critic state")
            q = self.critic(state.to(self.device), inputs[2])
            values = self.expected_values(q, probabilities, inputs[3])
        if (not torch.isfinite(probabilities).all() or not torch.isfinite(values).all()
                or (q is not None and not torch.isfinite(q).all())):
            raise FloatingPointError("non-finite rollout policy/value")
        if not torch.allclose(probabilities.sum(-1), torch.ones_like(scalar_values), atol=1e-6):
            raise FloatingPointError("legal probabilities do not normalize")
        if torch.any(probabilities[~inputs[3]] != 0):
            raise FloatingPointError("illegal token has nonzero probability")
        return inputs, logits, probabilities, values, q

    def collect(self, token_steps: int, *, deadline: Deadline | None = None) -> Rollout:
        batch_size = len(self.batch.envs)
        if type(token_steps) is not int or token_steps <= 0 or token_steps % batch_size:
            raise ValueError("aggregate token_steps must be positive and divisible by batch size")
        if self.incomplete:
            raise IncompleteTrainingError({"reason": "previous_rollout_incomplete", "actual_token_steps": self.total_token_steps})
        target_ticks = token_steps // batch_size
        rows: list[Transition] = []
        start_steps = self.total_token_steps
        started = monotonic()
        try:
            for tick in range(target_ticks):
                check_deadline(deadline)
                budget = getattr(deadline, "__self__", deadline)
                if tick and isinstance(budget, Deadline):
                    elapsed = monotonic() - started
                    remaining_estimate = elapsed / tick * (target_ticks - tick)
                    if remaining_estimate > budget.seconds - budget.elapsed:
                        raise BudgetExceeded("measured rollout throughput cannot finish the fixed target in the remaining budget")
                obs = deepcopy(self.observation)
                self._check_live(obs)
                state = critic_state(self.batch, obs) if self.critic is not None else None
                inputs, logits, probabilities, values, q = self._evaluate(obs, state)
                indices = torch.multinomial(probabilities.cpu(), 1, generator=self.generator).to(self.device)
                actions = inputs[2].gather(1, indices).squeeze(1)
                log_probs = self.model.log_prob(logits, actions, legal_next_tokens=inputs[2])
                selected_q = q.gather(1, indices[..., None].expand(-1, 1, 4)).squeeze(1) if q is not None else None
                result = self.batch.step(actions.detach().cpu().numpy().astype(np.int32))
                # Counters describe completed real transitions even if later work fails.
                self.total_token_steps += batch_size
                self.ticks_per_env += 1
                self.total_committed_steps += int(result.is_commit.sum())
                if not np.array_equal(result.token_step, obs.token_step + 1):
                    raise RuntimeError("batch did not advance exactly one real token per slot")
                if not np.array_equal(result.committed_step, obs.committed_step + result.is_commit):
                    raise RuntimeError("committed clock disagrees with commit receipts")
                if np.any(result.done & result.truncated):
                    raise RuntimeError("a transition cannot both terminate and truncate")
                raw_next = deepcopy(result.observation)
                # Fixed-width strings in inactive rows may be shorter than live phases.
                bootstrap = replace(deepcopy(raw_next), phase=raw_next.phase.astype("U32"))
                for index in np.flatnonzero(result.truncated):
                    _replace_observation_row(bootstrap, int(index), truncation_bootstrap_observation(self.batch, int(index)))
                next_values = torch.zeros((batch_size, 4), dtype=torch.float32)
                live = np.flatnonzero(~result.done)
                if len(live):
                    boot_state = critic_state(self.batch, bootstrap) if self.critic is not None else None
                    live_state = boot_state[live] if boot_state is not None else None
                    _, _, _, boot_values, _ = self._evaluate(_take_observations(bootstrap, live), live_state)
                    next_values[live] = boot_values.detach().cpu()
                rewards = torch.from_numpy(result.rewards.copy())
                players = torch.from_numpy(obs.player_id.astype(np.int64))
                row = Transition(
                    observation=obs, next_observation=raw_next, bootstrap_observation=bootstrap,
                    episode_indices=torch.from_numpy(self.reset_counts.copy()),
                    player_ids=players, team_ids=torch.from_numpy(obs.team_id.astype(np.int64)),
                    token_step_before=torch.from_numpy(obs.token_step.copy()),
                    token_step_after=torch.from_numpy(result.token_step.copy()),
                    committed_step_before=torch.from_numpy(obs.committed_step.copy()),
                    committed_step_after=torch.from_numpy(result.committed_step.copy()),
                    actions=actions.detach().cpu(), legal_mask=torch.from_numpy(obs.legal_next_mask.copy()),
                    old_log_probs=log_probs.detach().cpu(), values=values.detach().cpu(), next_values=next_values,
                    rewards=rewards, acting_player_rewards=rewards.gather(1, players[:, None]).squeeze(1),
                    terminated=torch.from_numpy(result.done.copy()), truncated=torch.from_numpy(result.truncated.copy()),
                    is_commit=torch.from_numpy(result.is_commit.copy()), info=deepcopy(result.info),
                    critic_states=state, selected_q=selected_q.detach().cpu() if selected_q is not None else None,
                )
                rows.append(row)  # Capture everything BEFORE the first reset.
                self.terminal_count += int(result.done.sum())
                self.truncation_count += int(result.truncated.sum())
                ended = np.flatnonzero(result.done | result.truncated)
                next_observation = replace(deepcopy(raw_next), phase=raw_next.phase.astype("U32"))
                for index in ended:
                    initial = self.initial_states[index] if self.initial_states is not None else None
                    reset_seed = self.reset_seed_base + batch_size * (1 + int(self.reset_counts[index])) + int(index)
                    if initial is None and reset_seed >= EVALUATION_SEED_BASE:
                        raise RuntimeError("training reset seed namespace exhausted; refusing evaluation seed overlap")
                    reset_obs = (self.batch.reset_at(int(index), initial_state=initial) if initial is not None
                                 else self.batch.reset_at(int(index), seed=reset_seed))
                    _replace_observation_row(next_observation, int(index), reset_obs)
                    self.reset_counts[index] += 1
                    self.deal_records.append(self._deal_record(int(index), reset_seed if initial is None else None))
                self.observation = next_observation
            check_deadline(deadline)
        except Exception as exc:
            self.incomplete = True
            if isinstance(exc, BudgetExceeded):
                exc.diagnostic = {
                    "status": "incomplete", "reason": "deadline", "stage": "rollout",
                    "target_token_steps": token_steps,
                    "actual_token_steps": self.total_token_steps - start_steps,
                    "lifetime_token_steps": self.total_token_steps,
                }
            raise
        return Rollout(tuple(rows))

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1, "envs": self.batch.serialize(), "rng_state": self.generator.get_state().clone(),
            "initial_states": self.initial_states, "reset_seed_base": self.reset_seed_base,
            "deal_records": deepcopy(self.deal_records), "total_token_steps": self.total_token_steps,
            "total_committed_steps": self.total_committed_steps, "ticks_per_env": self.ticks_per_env.tolist(),
            "reset_counts": self.reset_counts.tolist(), "terminal_count": self.terminal_count,
            "truncation_count": self.truncation_count, "incomplete": self.incomplete,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1 or state.get("incomplete"):
            raise ValueError("unsupported or incomplete collector checkpoint")
        size = len(self.batch.envs)
        if state.get("reset_seed_base") != self.reset_seed_base:
            raise ValueError("checkpoint reset seed namespace differs")
        if state.get("initial_states") != self.initial_states:
            raise ValueError("checkpoint fixture/reset mode differs")
        ticks = np.asarray(state["ticks_per_env"], dtype=np.int64)
        resets = np.asarray(state["reset_counts"], dtype=np.int64)
        if ticks.shape != (size,) or resets.shape != (size,) or np.any(ticks < 0) or np.any(resets < 0):
            raise ValueError("invalid collector counter shapes/values")
        if not np.all(ticks == ticks[0]) or int(ticks.sum()) != state["total_token_steps"]:
            raise ValueError("collector clocks disagree with real aggregate transitions")
        candidate = GuandanEnvBatch([env.config for env in self.batch.envs], observation_spec=self.batch.observation_spec)
        candidate.load_serialized(state["envs"])
        observation = candidate.observe()
        self._check_live(observation)
        generator = torch.Generator(device="cpu")
        generator.set_state(state["rng_state"].cpu())
        self.batch.load_serialized(state["envs"])
        self.observation, self.generator = observation, generator
        self.initial_states = state.get("initial_states")
        self.deal_records = deepcopy(state["deal_records"])
        self.total_token_steps = int(state["total_token_steps"])
        self.total_committed_steps = int(state["total_committed_steps"])
        self.ticks_per_env, self.reset_counts = ticks.copy(), resets.copy()
        self.terminal_count = int(state["terminal_count"])
        self.truncation_count = int(state["truncation_count"])
        self.incomplete = False


__all__ = ["Deadline", "IncompleteTrainingError", "Transition", "Rollout", "RolloutCollector",
           "policy_inputs", "critic_state", "truncation_bootstrap_observation"]
