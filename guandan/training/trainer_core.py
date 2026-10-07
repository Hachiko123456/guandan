"""Reusable on-policy training engine for A05 profile rollouts.

The parent runner owns profile selection, checkpoints, evidence, and the
outer update/resume loop.  This module owns exactly one real rollout and one
logical optimizer update per ``collect_and_update`` call.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from time import perf_counter as monotonic
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from ..env_batch import GuandanEnvBatch
from ..model import PolicyValueNet
from .collector import Rollout, RolloutCollector, check_deadline, policy_inputs
from .runtime import BudgetExceeded
from . import estimators as _default_estimators
from . import objectives as _default_objectives


class TrainingEngineError(ValueError):
    """Invalid low-level engine configuration or incompatible state."""


def _provider_get(provider: Any, name: str, default: Any = None) -> Any:
    if provider is None:
        return default
    if isinstance(provider, Mapping):
        return provider.get(name, default)
    return getattr(provider, name, default)


def _as_float_metrics(metrics: Mapping[str, Tensor | float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, Tensor):
            value = value.detach().cpu().item()
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite training metric: {name}")
        result[name] = value
    return result


@dataclass(frozen=True)
class _Targets:
    advantages: Tensor
    targets: Tensor
    kind: str


class TrainingEngine:
    """One-batch-at-a-time IPPO/VRPO low-level trainer.

    ``envs`` must be a normal A04 ``GuandanEnvBatch`` with four-player full
    deals created by the caller.  Auto-reset is deliberately disallowed: the
    collector captures raw terminal/truncated transitions and resets only
    afterwards.  ``objective_provider`` is dependency injection for the
    estimator/objective worker; by default the public local APIs are used.
    """

    def __init__(
        self,
        algorithm: str,
        envs: GuandanEnvBatch,
        actor: PolicyValueNet,
        optimizer: torch.optim.Optimizer,
        *,
        critic: torch.nn.Module | None = None,
        device: str | torch.device = "cpu",
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        epochs: int = 2,
        max_grad_norm: float = 1.0,
        seed: int = 10_000,
        objective_provider: Any = None,
        execution_mode: str = "reference",
        compute_chunk_ticks: int = 16,
    ) -> None:
        algorithm = str(algorithm).lower()
        if algorithm not in {"ippo", "vrpo"}:
            raise TrainingEngineError("algorithm must be ippo or vrpo")
        if not isinstance(envs, GuandanEnvBatch):
            raise TypeError("envs must be GuandanEnvBatch")
        if envs.auto_reset:
            raise TrainingEngineError("engine requires auto_reset=False")
        if not isinstance(actor, PolicyValueNet):
            raise TypeError("actor must be PolicyValueNet")
        if type(epochs) is not int or epochs <= 0:
            raise TrainingEngineError("epochs must be a positive integer")
        if isinstance(max_grad_norm, bool) or not math.isfinite(max_grad_norm) or float(max_grad_norm) <= 0:
            raise TrainingEngineError("max_grad_norm must be positive")
        if isinstance(gamma, bool) or not 0 <= float(gamma) <= 1:
            raise TrainingEngineError("gamma must be in [0,1]")
        if isinstance(gae_lambda, bool) or not 0 <= float(gae_lambda) <= 1:
            raise TrainingEngineError("gae_lambda must be in [0,1]")
        if algorithm == "vrpo" and critic is None:
            raise TrainingEngineError("vrpo requires a separate centralized critic")
        if algorithm == "ippo" and critic is not None:
            raise TrainingEngineError("IPPO uses the decentralized actor value head, not a Q critic")
        if critic is not None and not isinstance(critic, torch.nn.Module):
            raise TypeError("critic must be a torch module")
        if execution_mode not in {"reference", "batched"}:
            raise TrainingEngineError("execution_mode must be reference or batched")

        if type(compute_chunk_ticks) is not int or compute_chunk_ticks <= 0:
            raise TrainingEngineError("compute_chunk_ticks must be a positive integer")
        # Compute/memory tuning only: excluded from checkpoint semantic settings.
        self.compute_chunk_ticks = compute_chunk_ticks
        self.algorithm = algorithm
        self.envs = envs
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.actor.to(self.device)
        if self.critic is not None:
            self.critic.to(self.device)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.epochs = epochs
        self.max_grad_norm = float(max_grad_norm)
        self.seed = int(seed)
        self.objective_provider = objective_provider
        self.execution_mode = execution_mode
        self.update_count = 0
        self.optimizer_steps = 0
        self.incomplete = False
        self._pending_rollout: Rollout | None = None

        self._estimators = {
            name: _provider_get(objective_provider, name, getattr(_default_estimators, name))
            for name in ("compute_gae", "compute_q_boost", "acting_player_values", "masked_expected_values")
        }
        self._objectives = {
            name: _provider_get(objective_provider, name, getattr(_default_objectives, name))
            for name in ("ippo_loss", "vrpo_loss")
        }
        for name in ("compute_gae", "acting_player_values"):
            if _provider_get(self._estimators, name) is None:
                raise TrainingEngineError(f"estimator provider is missing {name}()")
        if algorithm == "vrpo":
            for name in ("compute_q_boost", "masked_expected_values"):
                if _provider_get(self._estimators, name) is None:
                    raise TrainingEngineError(f"VRPO estimator provider is missing {name}()")
            if _provider_get(self._objectives, "vrpo_loss") is None:
                raise TrainingEngineError("VRPO objective provider is missing vrpo_loss()")
        elif _provider_get(self._objectives, "ippo_loss") is None:
            raise TrainingEngineError("IPPO objective provider is missing ippo_loss()")

        self.collector = RolloutCollector(
            envs, actor, seed=self.seed,
            critic=critic,
            expected_values=_provider_get(self._estimators, "masked_expected_values")
            if algorithm == "vrpo" else None,
        )
        self._parameters = list(actor.parameters()) + (list(critic.parameters()) if critic is not None else [])
        if len({id(p) for p in self._parameters}) != len(self._parameters):
            raise TrainingEngineError("centralized critic must not share actor parameters")
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        required_ids = {id(p) for p in self._parameters if p.requires_grad}
        if not required_ids.issubset(optimizer_ids):
            raise TrainingEngineError("optimizer must include all actor and centralized critic parameters")

    @property
    def total_token_steps(self) -> int:
        return self.collector.total_token_steps

    @property
    def total_committed_steps(self) -> int:
        return self.collector.total_committed_steps

    @property
    def reset_counts(self) -> np.ndarray:
        return self.collector.reset_counts.copy()

    def collect(self, token_steps_per_update: int, *, deadline: Any = None) -> Rollout:
        """Collect once; ``update(rollout)`` must consume it before another collection."""
        if self.incomplete or self.collector.incomplete:
            raise TrainingEngineError("previous update incomplete; restore a complete checkpoint")
        if self._pending_rollout is not None:
            raise TrainingEngineError("an unconsumed rollout is pending; call update() first")
        # An expired preflight has not mutated anything and can safely be retried.
        check_deadline(deadline)
        try:
            rollout = self.collector.collect(token_steps_per_update, deadline=deadline)
        except Exception:
            self.incomplete = True
            raise
        self._pending_rollout = rollout
        return rollout

    def _targets(self, rollout: Rollout) -> _Targets:
        rewards = rollout.rewards.to(self.device, dtype=torch.float32)
        values = rollout.values.to(self.device, dtype=torch.float32)
        next_values = rollout.next_values.to(self.device, dtype=torch.float32)
        terminated = rollout.terminated.to(self.device)
        truncated = rollout.truncated.to(self.device)
        acting = _provider_get(self._estimators, "acting_player_values")
        if self.algorithm == "ippo":
            advantages, returns = _provider_get(self._estimators, "compute_gae")(
                rewards, values, next_values, terminated, truncated,
                gamma=self.gamma, gae_lambda=self.gae_lambda,
            )
            return _Targets(acting(advantages, rollout.stack("player_ids").to(self.device)),
                            acting(returns, rollout.stack("player_ids").to(self.device)), "gae")
        selected_q = rollout.stack("selected_q").to(self.device, dtype=torch.float32)
        advantages, q_targets = _provider_get(self._estimators, "compute_q_boost")(
            rewards, selected_q, values, next_values, terminated, truncated,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )
        player_ids = rollout.stack("player_ids").to(self.device)
        return _Targets(acting(advantages, player_ids), acting(q_targets, player_ids), "q_boost")

    def _loss(self, row: Any, advantage: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        # One timestep graph at a time; privilege is confined to the Q critic.
        inputs = policy_inputs(row.observation, self.device)
        logits, scalar_values = self.actor(*inputs)
        actions = row.actions.to(self.device)
        if self.algorithm == "ippo":
            prediction = scalar_values
            objective = _provider_get(self._objectives, "ippo_loss")
        else:
            q = self.critic(row.critic_states.to(self.device), inputs[2])
            matches = (inputs[2] == actions[:, None]) & inputs[3]
            if not torch.all(matches.sum(-1) == 1):
                raise TrainingEngineError("recorded action does not identify exactly one legal row")
            indices = matches.long().argmax(-1)
            selected_q = q.gather(1, indices[:, None, None].expand(-1, 1, 4)).squeeze(1)
            prediction = _provider_get(self._estimators, "acting_player_values")(
                selected_q, row.player_ids.to(self.device)
            )
            objective = _provider_get(self._objectives, "vrpo_loss")
        return objective(
            logits, prediction, actions, row.old_log_probs.to(self.device, dtype=logits.dtype),
            advantage.detach(), target.detach(), inputs[2],
        )

    def _batched_loss(self, rollout: Rollout, targets: _Targets) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute one mean loss over all T*B transitions.

        This keeps policy sampling in the collector and only batches the
        learner-side forward/backward. Reference mode remains available for
        focused tests and exact behavioral comparisons.
        """
        observations = rollout.observations
        inputs = policy_inputs(observations, self.device)
        logits, scalar_values = self.actor(*inputs)
        actions = rollout.stack("actions").reshape(-1).to(self.device)
        old_log_probs = rollout.stack("old_log_probs").reshape(-1).to(self.device, dtype=logits.dtype)
        advantages = targets.advantages.reshape(-1).detach()
        target_values = targets.targets.reshape(-1).detach()
        if self.algorithm == "ippo":
            return _provider_get(self._objectives, "ippo_loss")(
                logits, scalar_values, actions, old_log_probs, advantages, target_values, inputs[2]
            )
        states = torch.cat([row.critic_states for row in rollout.transitions], dim=0).to(self.device)
        q = self.critic(states, inputs[2])
        matches = (inputs[2] == actions[:, None]) & inputs[3]
        if not torch.all(matches.sum(-1) == 1):
            raise TrainingEngineError("recorded action does not identify exactly one legal row")
        indices = matches.long().argmax(-1)
        selected_q = q.gather(1, indices[:, None, None].expand(-1, 1, 4)).squeeze(1)
        player_ids = rollout.stack("player_ids").reshape(-1).to(self.device)
        prediction = _provider_get(self._estimators, "acting_player_values")(selected_q, player_ids)
        return _provider_get(self._objectives, "vrpo_loss")(
            logits, prediction, actions, old_log_probs, advantages, target_values, inputs[2]
        )

    def collect_and_update(self, token_steps_per_update: int, *, deadline: Any = None) -> dict[str, Any]:
        """Exactly one rollout and one logical update, never another outer loop."""
        rollout = self.collect(token_steps_per_update, deadline=deadline)
        return self.update(rollout, deadline=deadline)

    def update(self, rollout: Rollout | None = None, *, deadline: Any = None) -> dict[str, Any]:
        """Consume this engine's pending rollout exactly once (epochs are explicit)."""
        if self.incomplete or self.collector.incomplete:
            raise TrainingEngineError("previous update incomplete; restore a complete checkpoint")
        rollout = self._pending_rollout if rollout is None else rollout
        if rollout is None or rollout is not self._pending_rollout:
            raise TrainingEngineError("rollout is foreign, absent, or already consumed")
        started = monotonic()
        epoch_metrics: list[dict[str, float]] = []
        gradient_norms: list[float] = []
        timings = dict.fromkeys(("targets_seconds", "forward_loss_seconds",
                                 "backward_seconds", "optimizer_seconds"), 0.0)
        try:
            check_deadline(deadline)
            phase_started = monotonic()
            targets = self._targets(rollout)
            timings["targets_seconds"] = monotonic() - phase_started
            self.actor.train()
            if self.critic is not None:
                self.critic.train()
            for _ in range(self.epochs):
                check_deadline(deadline)
                self.optimizer.zero_grad(set_to_none=True)
                averaged: dict[str, float] = {}
                chunk_ticks = self.compute_chunk_ticks if self.execution_mode == "batched" else 1
                for start in range(0, rollout.ticks, chunk_ticks):
                    check_deadline(deadline)
                    stop = min(start + chunk_ticks, rollout.ticks)
                    weight = (stop - start) / rollout.ticks
                    phase_started = monotonic()
                    if self.execution_mode == "batched":
                        chunk = Rollout(rollout.transitions[start:stop])
                        chunk_targets = _Targets(targets.advantages[start:stop],
                                                 targets.targets[start:stop], targets.kind)
                        loss, metrics = self._batched_loss(chunk, chunk_targets)
                    else:
                        loss, metrics = self._loss(rollout.transitions[start],
                                                   targets.advantages[start], targets.targets[start])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite A05 loss")
                    timings["forward_loss_seconds"] += monotonic() - phase_started
                    phase_started = monotonic()
                    if self.execution_mode == "batched":
                        (loss * weight).backward()
                    else:
                        (loss / rollout.ticks).backward()
                    timings["backward_seconds"] += monotonic() - phase_started
                    for name, value in _as_float_metrics(metrics).items():
                        weighted = value * weight if self.execution_mode == "batched" else value / rollout.ticks
                        averaged[name] = averaged.get(name, 0.0) + weighted
                    # Release the graph before constructing the next chunk.
                    del loss, metrics
                phase_started = monotonic()
                norm = torch.nn.utils.clip_grad_norm_(
                    self._parameters, self.max_grad_norm, error_if_nonfinite=True,
                )
                gradient_norms.append(float(norm.detach().cpu()))
                epoch_metrics.append(averaged)
                check_deadline(deadline)
                self.optimizer.step()
                self.optimizer_steps += 1
                timings["optimizer_seconds"] += monotonic() - phase_started
                if any(not torch.isfinite(p).all() for p in self._parameters):
                    raise FloatingPointError("non-finite updated model parameter")
            self.update_count += 1
            self._pending_rollout = None
            check_deadline(deadline)
        except Exception as exc:
            self.incomplete = True
            if isinstance(exc, BudgetExceeded):
                exc.diagnostic = {
                    "status": "incomplete", "reason": "deadline", "stage": "update",
                    "target_token_steps": rollout.token_transitions,
                    "actual_token_steps": rollout.token_transitions,
                    "lifetime_token_steps": self.total_token_steps,
                    "update_count": self.update_count, "optimizer_steps": self.optimizer_steps,
                }
            raise
        result = {
            "status": "complete", "algorithm": self.algorithm,
            "update": self.update_count, "update_count": self.update_count,
            "rollout_ticks": rollout.ticks, "ticks_per_env": rollout.ticks,
            "rollout_envs": rollout.batch_size,
            "token_steps": rollout.token_transitions,
            "token_steps_per_update": rollout.token_transitions,
            "total_token_steps": self.total_token_steps,
            "committed_steps": int(rollout.stack("is_commit").sum()),
            "total_committed_steps": self.total_committed_steps,
            "optimizer_steps": self.optimizer_steps, "epochs": self.epochs,
            "gradient_norm": max(gradient_norms), "grad_norm": max(gradient_norms),
            "target_kind": targets.kind,
            "terminated": int(rollout.terminated.sum()),
            "truncated": int(rollout.truncated.sum()),
            "reset_counts": self.reset_counts.tolist(),
            "per_env_ticks": [rollout.ticks] * rollout.batch_size,
            "lifetime_ticks_per_env": self.collector.ticks_per_env.tolist(),
            "terminal_snapshots": rollout.terminal_snapshots(),
            "rollout_sha256": rollout.digest(),
            "deal_records": deepcopy(self.collector.deal_records),
            "update_seconds": monotonic() - started,
            "timing_kind": "host_wall_no_extra_cuda_sync",
            "collector_timings": dict(self.collector.last_timings),
            "update_timings": timings,
            "compute_chunk_ticks": self.compute_chunk_ticks,
        }
        for key in epoch_metrics[-1]:
            result[key] = float(np.mean([row[key] for row in epoch_metrics]))
        return result

    def _settings(self) -> dict[str, Any]:
        return {"gamma": self.gamma, "gae_lambda": self.gae_lambda, "epochs": self.epochs,
                "max_grad_norm": self.max_grad_norm, "batch_size": len(self.envs.envs),
                "execution_mode": self.execution_mode}

    def extra_state(self) -> dict[str, Any]:
        if self.incomplete or self._pending_rollout is not None:
            raise TrainingEngineError("cannot checkpoint an incomplete/unconsumed update")
        numpy_rng = np.random.get_state()
        return {
            "version": 1, "algorithm": self.algorithm, "seed": self.seed,
            "settings": self._settings(),
            "update_count": self.update_count, "optimizer_steps": self.optimizer_steps,
            "torch_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            "numpy_rng_state": {
                "algorithm": numpy_rng[0], "keys": numpy_rng[1].tolist(),
                "position": int(numpy_rng[2]), "has_gauss": int(numpy_rng[3]),
                "cached_gaussian": float(numpy_rng[4]),
            },
            "collector": self.collector.state_dict(),
        }

    def _validate_extra(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1 or state.get("algorithm") != self.algorithm:
            raise TrainingEngineError("incompatible engine extra state")
        if state.get("settings") != self._settings():
            raise TrainingEngineError("engine settings differ from checkpoint")
        count, steps = state.get("update_count"), state.get("optimizer_steps")
        if type(count) is not int or count < 0 or type(steps) is not int or steps != count * self.epochs:
            raise TrainingEngineError("checkpoint has incomplete or invalid update counters")

    def load_extra_state(self, state: Mapping[str, Any]) -> None:
        """Restore counters/RNG/envs after loading the corresponding model/optimizer."""
        self._validate_extra(state)
        self.collector.load_state_dict(state["collector"])
        self.seed = int(state["seed"])
        self.update_count = state["update_count"]
        self.optimizer_steps = state["optimizer_steps"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        numpy_rng = state["numpy_rng_state"]
        np.random.set_state((numpy_rng["algorithm"], np.asarray(numpy_rng["keys"], dtype=np.uint32),
                             numpy_rng["position"], numpy_rng["has_gauss"], numpy_rng["cached_gaussian"]))
        if self.device.type == "cuda" and state.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng_states"]])
        self.incomplete = False
        self._pending_rollout = None

    def state_dict(self) -> dict[str, Any]:
        # Deep copy makes a snapshot stable even if the live optimizer runs again.
        return deepcopy({
            "version": 1, "algorithm": self.algorithm,
            "actor": self.actor.state_dict(),
            "critic": None if self.critic is None else self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "extra_state": self.extra_state(),
        })

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1 or state.get("algorithm") != self.algorithm:
            raise TrainingEngineError("incompatible engine state")
        if (state.get("critic") is None) != (self.critic is None):
            raise TrainingEngineError("engine and checkpoint critic presence differ")
        self._validate_extra(state["extra_state"])
        self.actor.load_state_dict(state["actor"])
        if self.critic is not None:
            self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.load_extra_state(state["extra_state"])


__all__ = ["TrainingEngine", "TrainingEngineError"]
