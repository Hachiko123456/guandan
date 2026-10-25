# SPDX-License-Identifier: MIT
"""DMC ε-greedy self-play sampler (stage 4).

Data-generation contract (``dmc_v1``).  This docstring *is* the behavioural
contract referenced by the locked migration table — there is no other knob:

* The behavioural policy is exactly **ε-greedy over complete legal
  candidates**: with probability ε one uniformly random legal candidate, else
  the Q-argmax candidate.  There is no temperature, no top-k, no importance
  ratio and no on-policy contract: DMC is off-policy Q regression and the
  replay stores the executed action regardless of how it was chosen.
* **Single-candidate decisions are submitted directly** (FableDan's
  ``auto_single`` rule): no training sample, no ε draw.
* **Only multi-candidate decisions enter the replay buffer.**
* ε anneals linearly from ``epsilon_start`` to ``epsilon_final`` over
  ``epsilon_decay_samples`` samples pushed to the replay (locked table:
  0.10 -> 0.02 over 5,000,000), never from wall-clock time.
* All four seats share one network.  The acting seat is stored on the sample
  as provenance only; Q is computed with the neutral (seat-free) player
  embedding exactly like ``DMCLearner._forward``, so sampled and trained Q
  values are the same function.
* ``z`` is the acting seat's terminal team rank point (raw ±3/±2/±1) divided
  by three (``TRAINING_REWARD_SCALE``), with ``gamma = 1`` and no discounting:
  the final-outcome regression target of Deep Monte-Carlo.

Environment driving reuses the shipped Python-reference V2 adapter
(``guandan_marvel.game.EnvBatch`` with the explicit ``python_reference``
backend — never a silent switch) and the accepted stage-2 enumerator
(``dmc.candidates.enumerate_candidates``) on the live protocol.  Chosen
candidates are committed through the adapter's legal action rows one token per
batch step, exactly like the FableDan guest driver.  Per-seat observation
token rows are the *cumulative* event stream of the acting seat within the
current game (the V2 event vocabulary, one event token per position, tail-
windowed to ``max_obs_tokens``); observation channels are the adapter's
512-wide actor channels for the acting seat.
"""

from __future__ import annotations

import hashlib
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from guandan.action_state import ProtocolError as V1ProtocolError
from guandan_marvel.backends import get_game_backend
from guandan_marvel.context_schedule import (
    expand_context_schedule,
    validate_context_schedule,
)
from guandan_marvel.dmc.candidates import (
    ACTION_FEATURE_DIM,
    CandidateAction,
    encode_action_features,
    enumerate_candidates,
)
from guandan_marvel.dmc.learner import DMCNonFiniteError
from guandan_marvel.dmc.replay import DMCReplay
from guandan_marvel.reward import TRAINING_REWARD_SCALE
from guandan_marvel.evaluation.agents import ACTION_TOKEN_COLUMN

SAMPLER_STATE_SCHEMA = "guandan-marvel-dmc-sampler-state-v1"

#: A single event row is at most ``4 + max_action_tokens + 1`` tokens and the
#: adapter's own observation capacity is 48; a window below that would clip
#: even one event, so it is rejected instead of silently losing state.
MIN_OBS_WINDOW = 48

_GAME_RECORD_HISTORY = 4096


class DMCSamplerFailedStateError(RuntimeError):
    """The sampler observed a non-transactional failure and cannot continue."""


class DMCSampleSink(ABC):
    """Push target for finished games' sample batches.

    A sink exposes the same read-side layout attributes as :class:`DMCReplay`
    (``channel_dim``, ``action_feature_dim``, ``pad_token``, ``num_tokens``,
    ``num_players``, ``max_obs_tokens``) so the sampler's layout checks run
    unchanged, plus ``push_batch`` and ``__len__``.  The default sink is the
    replay itself; the parallel trainer (``dmc.parallel``) forwards batches to
    the learner process through one instead of keeping a second full buffer
    in the collector.  Rejection semantics are unchanged either way: a batch
    the sink cannot accept fails the collection loop exactly like a replay
    push would.
    """

    channel_dim: int
    action_feature_dim: int
    pad_token: int
    num_tokens: int | None
    num_players: int
    max_obs_tokens: int

    @abstractmethod
    def push_batch(self, samples) -> int:
        """Accept one finished game's batch; return the accepted row count."""

    @abstractmethod
    def __len__(self) -> int:
        """Rows currently accepted (for ``health`` reporting)."""


def linear_epsilon(samples_pushed: int, *, start: float, final: float, decay_samples: int) -> float:
    """Linear anneal from ``start`` to ``final`` over ``decay_samples`` samples.

    A pure function of the pushed-sample count: the same count always yields
    the same ε, so a restored sampler anneals exactly where it stopped.
    """
    samples = int(samples_pushed)
    start = float(start)
    final = float(final)
    decay = int(decay_samples)
    if decay <= 0:
        raise ValueError("decay_samples must be positive")
    if not 0.0 <= final <= start <= 1.0:
        raise ValueError("epsilon must satisfy 0 <= final <= start <= 1")
    if samples <= 0:
        return start
    fraction = min(1.0, samples / decay)
    return float(start - (start - final) * fraction)


def select_candidate_index(
    q_values: torch.Tensor, rng: np.random.Generator, epsilon: float,
) -> tuple[int, bool]:
    """ε-greedy choice for one multi-candidate decision.

    Returns ``(index, chosen_is_argmax)``.  ``torch.argmax`` tie-breaks to the
    first index, which keeps the choice deterministic for a given Q row and
    RNG state.  The ε draw happens for every multi-candidate decision (one
    ``rng.random()`` call) so the RNG stream does not depend on the ε value.
    """
    if q_values.dim() != 1 or int(q_values.shape[0]) < 1:
        raise ValueError("q_values must be a 1-D non-empty tensor")
    if not bool(torch.isfinite(q_values).all()):
        raise DMCNonFiniteError("non-finite Q value in candidate selection")
    epsilon = float(epsilon)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    argmax = int(torch.argmax(q_values).item())
    index = argmax
    if rng.random() < epsilon:
        index = int(rng.integers(int(q_values.shape[0])))
    return index, index == argmax


@dataclass(frozen=True)
class DMCSamplerConfig:
    """Explicit sampler configuration; every field is validated."""

    num_envs: int
    device: str = "cpu"
    env_seed: int = 0
    seed: int | None = None
    backend: str = "python_reference"
    max_horizon: int | None = None
    max_obs_tokens: int = 1024
    epsilon_start: float = 0.10
    epsilon_final: float = 0.02
    epsilon_decay_samples: int = 5_000_000
    context_schedule: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.num_envs, bool) or int(self.num_envs) <= 0:
            raise ValueError("num_envs must be a positive integer")
        if self.seed is not None and isinstance(self.seed, bool):
            raise TypeError("seed must be an integer or None")
        if self.max_horizon is not None and int(self.max_horizon) <= 0:
            raise ValueError("max_horizon must be positive when provided")
        if int(self.max_obs_tokens) < MIN_OBS_WINDOW:
            raise ValueError(
                f"max_obs_tokens must be >= {MIN_OBS_WINDOW} so one event row always fits"
            )
        if int(self.epsilon_decay_samples) <= 0:
            raise ValueError("epsilon_decay_samples must be positive")
        if not 0.0 <= float(self.epsilon_final) <= float(self.epsilon_start) <= 1.0:
            raise ValueError("epsilon must satisfy 0 <= epsilon_final <= epsilon_start <= 1")
        if self.context_schedule is not None and not isinstance(self.context_schedule, Mapping):
            raise TypeError("context_schedule must be a mapping or None")


@dataclass
class _DecisionRequest:
    """One multi-candidate decision queued for the batched Q forward."""

    slot_index: int
    seat: int
    candidates: tuple[CandidateAction, ...]
    obs_tokens: list[int]
    channels: np.ndarray
    action_feats: np.ndarray


@dataclass
class _SlotState:
    """Per-environment in-flight game state (event streams + pending candidate)."""

    game_id: int
    streams: list[list[int]] = field(default_factory=lambda: [[], [], [], []])
    pending: list[int] = field(default_factory=list)
    decisions: int = 0
    samples: int = 0
    buffer: list[dict[str, Any]] = field(default_factory=list)
    action_tokens: list[int] = field(default_factory=list)


class DMCSampler:
    """Collect ε-greedy self-play games into a :class:`DMCReplay`."""

    def __init__(
        self,
        *,
        q_agent: torch.nn.Module,
        replay: DMCReplay,
        config: DMCSamplerConfig,
    ) -> None:
        if not isinstance(q_agent, torch.nn.Module) or not hasattr(q_agent, "q_values"):
            raise TypeError("q_agent must be a torch.nn.Module exposing q_values(...)")
        if not isinstance(replay, (DMCReplay, DMCSampleSink)):
            raise TypeError("replay must be a DMCReplay or a DMCSampleSink")
        if not isinstance(config, DMCSamplerConfig):
            raise TypeError("config must be a DMCSamplerConfig")
        self._config = config
        self._game = get_game_backend(config.backend)
        if config.backend != "python_reference":
            # Stage 4 wires the Python oracle only.  A C++ DMC path needs its
            # own protocol mirror for candidate enumeration and must be added
            # explicitly, never by silently swapping the backend here.
            raise ValueError(
                "the DMC sampler supports backend='python_reference' only; "
                f"got {config.backend!r}"
            )

        # The replay must match the real shipped game exactly.  The game's
        # channel width is the single source of truth (512 for the Guandan
        # adapter) — a replay built with a default width fails here instead of
        # silently training on mis-shaped observations.
        if replay.channel_dim != int(self._game.num_channels):
            raise ValueError(
                f"replay.channel_dim={replay.channel_dim} does not match "
                f"guandan_marvel.game.num_channels={int(self._game.num_channels)}; build the "
                "DMCReplay with the game adapter's channel width, never a module default"
            )
        if replay.action_feature_dim != ACTION_FEATURE_DIM:
            raise ValueError("replay.action_feature_dim must be the GD-DMC-ACTION-1 width")
        if replay.pad_token != int(self._game.pad_token):
            raise ValueError("replay.pad_token must match the game adapter pad token")
        if replay.num_tokens is not None and replay.num_tokens < int(self._game.num_tokens):
            raise ValueError("replay.num_tokens must cover the game token vocabulary")
        if replay.num_players != int(self._game.num_players):
            raise ValueError("replay.num_players must match the game adapter")
        if config.max_obs_tokens != replay.max_obs_tokens:
            raise ValueError(
                "config.max_obs_tokens must equal replay.max_obs_tokens so the sampler "
                "window and the buffer layout cannot drift apart"
            )

        self._q_agent = q_agent
        self._device = torch.device(config.device)
        self._q_agent.to(self._device)
        self._q_agent.eval()
        self._amp = self._device.type == "cuda"
        self._replay = replay
        self._max_obs_tokens = int(config.max_obs_tokens)
        self._max_horizon = (
            int(self._game.max_game_horizon)
            if config.max_horizon is None
            else int(config.max_horizon)
        )
        if self._max_horizon > int(self._game.max_game_horizon):
            raise ValueError("max_horizon cannot exceed the adapter's max_game_horizon")
        self._rng = np.random.default_rng(config.seed)

        kwargs: dict[str, Any] = {
            "backend": "python_reference",
            "auto_reset": True,
            "max_horizon": self._max_horizon,
        }
        if config.context_schedule is not None:
            schedule = validate_context_schedule(dict(config.context_schedule))
            kwargs["configs"] = None
            self._context_schedule = schedule
        else:
            self._context_schedule = None
        self.envs = self._game.EnvBatch(int(config.num_envs), **kwargs)
        if self._context_schedule is not None:
            expanded = expand_context_schedule(self._context_schedule)
            self.envs.configure_context_schedule(expanded, offset=0)

        # Cumulative health/counters (poison-on-failure style, mirroring
        # runtime/sampler.py).
        self._games_started = 0
        self.games_completed = 0
        self.samples_pushed = 0
        self.decisions_total = 0
        self.multi_decisions_total = 0
        self.single_decisions_total = 0
        self.argmax_hits = 0
        self.illegal_count = 0
        self.nonfinite_count = 0
        self.obs_window_truncations = 0
        self.truncated_count = 0
        self._failed = False
        self._failure_info: dict[str, Any] | None = None
        self._game_records: deque[dict[str, Any]] = deque(maxlen=_GAME_RECORD_HISTORY)
        self._slots: list[_SlotState] = []
        self._current_step = None
        self.reset()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def failure_info(self) -> dict[str, Any] | None:
        return None if self._failure_info is None else dict(self._failure_info)

    def epsilon(self) -> float:
        """Current ε, a pure function of samples pushed to the replay."""
        return linear_epsilon(
            self.samples_pushed,
            start=float(self._config.epsilon_start),
            final=float(self._config.epsilon_final),
            decay_samples=int(self._config.epsilon_decay_samples),
        )

    def health(self) -> dict[str, Any]:
        """Cumulative health counters (locked stage-4 counter set)."""
        multi = self.multi_decisions_total
        return {
            "games_started": self._games_started,
            "games_completed": self.games_completed,
            "decisions": self.decisions_total,
            "single_candidate_decisions": self.single_decisions_total,
            "multi_candidate_decisions": multi,
            "samples_pushed": self.samples_pushed,
            "argmax_hit_rate": (self.argmax_hits / multi) if multi else None,
            "epsilon": self.epsilon(),
            "illegal": self.illegal_count,
            "truncated": self.truncated_count,
            "nonfinite": self.nonfinite_count,
            "obs_window_truncations": self.obs_window_truncations,
            "replay_size": len(self._replay),
            "failed": self._failed,
        }

    def game_records(self) -> tuple[dict[str, Any], ...]:
        """Per-completed-game records (bounded deque, oldest dropped)."""
        return tuple(dict(record) for record in self._game_records)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Deal the first game in every slot from the configured env seed."""
        self._require_healthy()
        step = self.envs.reset(int(self._config.env_seed))
        self._current_step = step
        self._slots = []
        for batch_index in range(int(self._config.num_envs)):
            slot = _SlotState(game_id=self._next_game_id())
            self._install_reset_streams(slot, step.observations.tokens[batch_index])
            self._slots.append(slot)

    def _next_game_id(self) -> int:
        game_id = self._games_started
        self._games_started += 1
        return game_id

    @staticmethod
    def _event_row(row: np.ndarray) -> list[int]:
        """Extract the true (non-padded) event tokens of one obs row."""
        tokens = [int(token) for token in row.tolist() if token != 0]
        return tokens

    def _install_reset_streams(self, slot: _SlotState, token_rows: np.ndarray) -> None:
        for seat in range(4):
            slot.streams[seat] = self._event_row(token_rows[seat])

    def _append_events(self, slot: _SlotState, token_rows: np.ndarray) -> None:
        for seat in range(4):
            stream = slot.streams[seat]
            stream.extend(self._event_row(token_rows[seat]))
            if len(stream) > self._max_obs_tokens:
                # Keep the most recent window; the drop is counted, never
                # silent, and the stored sample stays inside the buffer layout.
                del stream[: len(stream) - self._max_obs_tokens]
                self.obs_window_truncations += 1

    def _require_healthy(self) -> None:
        if self._failed:
            info = self._failure_info or {}
            raise DMCSamplerFailedStateError(
                "DMC sampler is unusable after a failed iteration; restore from a "
                f"checkpoint (failure={info})"
            )

    def _mark_failed(self, *, stage: str, error: BaseException) -> None:
        self._failed = True
        self._failure_info = {
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "games_completed": self.games_completed,
        }

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def collect_games(
        self, count: int, *, progress_callback=None,
    ) -> dict[str, Any]:
        """Run lockstep steps until ``count`` further games complete.

        Returns the per-cycle statistics consumed by ``train.py``; the sampler
        pushes finished games' samples into the replay itself.  When given,
        ``progress_callback`` is called once per lockstep iteration with a
        read-only progress mapping; it must not mutate sampler state or RNG.
        """
        self._require_healthy()
        value = int(count)
        if value <= 0:
            raise ValueError("count must be positive")
        target = self.games_completed + value
        started = time.perf_counter()
        games_at_start = self.games_completed
        samples_at_start = self.samples_pushed
        decisions_at_start = self.decisions_total
        epsilon_at_start = self.epsilon()
        while self.games_completed < target:
            self._iteration()
            if progress_callback is not None:
                progress_callback({
                    "games_completed": self.games_completed,
                    "target_games": target,
                    "samples_pushed": self.samples_pushed,
                })
        elapsed = time.perf_counter() - started
        games = self.games_completed - games_at_start
        samples = self.samples_pushed - samples_at_start
        decisions = self.decisions_total - decisions_at_start
        return {
            "games": games,
            "samples_pushed": samples,
            "decisions": decisions,
            "epsilon_start": epsilon_at_start,
            "epsilon_end": self.epsilon(),
            "wall_seconds": elapsed,
            "games_per_second": (games / elapsed) if elapsed > 0 else None,
            "samples_per_second": (samples / elapsed) if elapsed > 0 else None,
            "health": self.health(),
        }

    def _iteration(self) -> None:
        try:
            self._iteration_inner()
        except Exception as error:  # noqa: BLE001 - poison on any failure
            # Fail closed: any unexpected failure poisons the sampler exactly
            # like runtime/sampler.py, after recording the stage.
            self._mark_failed(stage="lockstep_iteration", error=error)
            raise

    def _iteration_inner(self) -> None:
        num_envs = int(self._config.num_envs)
        requests: list[_DecisionRequest] = []
        for batch_index, slot in enumerate(self._slots):
            if not slot.pending:
                self._decide(batch_index, slot, requests)
        if requests:
            self._score_and_select(requests)

        action_width = int(self._current_step.available_actions.tokens.shape[-1])
        rows = np.zeros((num_envs, action_width), dtype=np.int64)
        for batch_index, slot in enumerate(self._slots):
            rows[batch_index] = self._action_row_for(batch_index, slot.pending[0])

        step = self.envs.step(rows)
        self._current_step = step
        for batch_index, slot in enumerate(self._slots):
            token = slot.pending.pop(0)
            slot.action_tokens.append(int(token))
            if bool(step.is_terminal[batch_index]):
                self._flush_terminal_slot(batch_index, slot, step)
            else:
                self._append_events(slot, step.observations.tokens[batch_index])

    def _decide(
        self,
        batch_index: int,
        slot: _SlotState,
        requests: list[_DecisionRequest],
    ) -> None:
        """Handle one fresh decision point; queues multi-candidate forwards."""
        protocol = self.envs.envs[batch_index]._require_protocol()
        candidates = enumerate_candidates(protocol)
        if not candidates:
            raise V1ProtocolError(
                f"slot {batch_index} reached a decision point with no legal candidate"
            )
        seat = int(protocol.round_state.active_seat)
        slot.decisions += 1
        self.decisions_total += 1
        if len(candidates) == 1:
            # Locked rule: single-candidate decisions are submitted directly —
            # no training sample and no ε draw.
            self.single_decisions_total += 1
            slot.pending = list(candidates[0].tokens)
            return
        self.multi_decisions_total += 1
        obs_tokens = list(slot.streams[seat])[-self._max_obs_tokens:]
        channels = np.array(
            self._current_step.observations.channels[batch_index, seat], dtype=np.uint8, copy=True,
        )
        feats = np.stack(
            [encode_action_features(candidate, protocol.round_state) for candidate in candidates],
        ).astype(np.float32)
        requests.append(_DecisionRequest(
            slot_index=batch_index,
            seat=seat,
            candidates=candidates,
            obs_tokens=obs_tokens,
            channels=channels,
            action_feats=feats,
        ))

    def _score_and_select(self, requests: list[_DecisionRequest]) -> None:
        batch = len(requests)
        max_tokens = max(len(request.obs_tokens) for request in requests)
        max_candidates = max(len(request.candidates) for request in requests)
        pad = int(self._replay.pad_token)
        obs_tokens = np.full((batch, max_tokens), pad, dtype=np.int64)
        token_lengths = np.zeros(batch, dtype=np.int64)
        channels = np.zeros((batch, self._replay.channel_dim), dtype=np.uint8)
        feats = np.zeros((batch, max_candidates, ACTION_FEATURE_DIM), dtype=np.float32)
        masks = np.zeros((batch, max_candidates), dtype=bool)
        for index, request in enumerate(requests):
            length = len(request.obs_tokens)
            obs_tokens[index, :length] = request.obs_tokens
            token_lengths[index] = length
            channels[index] = request.channels
            feats[index, : len(request.candidates)] = request.action_feats
            masks[index, : len(request.candidates)] = True

        tokens = torch.from_numpy(obs_tokens).to(self._device)
        lengths = torch.from_numpy(token_lengths).to(self._device)
        channels_t = torch.from_numpy(channels).to(self._device)
        feats_t = torch.from_numpy(feats).to(self._device)
        masks_t = torch.from_numpy(masks).to(self._device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._amp
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.inference_mode(), autocast:
            q = self._q_agent.q_values(
                tokens, channels_t, feats_t, masks_t, token_lengths=lengths,
            )
        valid = q[masks_t]
        nonfinite = int((~torch.isfinite(valid)).sum().item())
        if nonfinite:
            self.nonfinite_count += nonfinite
            raise DMCNonFiniteError(
                f"non-finite Q value(s) at a DMC decision point: count={nonfinite}"
            )
        epsilon = self.epsilon()
        q_cpu = q.detach().to("cpu")
        for index, request in enumerate(requests):
            count = len(request.candidates)
            q_row = q_cpu[index, :count]
            choice, is_argmax = select_candidate_index(q_row, self._rng, epsilon)
            if is_argmax:
                self.argmax_hits += 1
            chosen = request.candidates[choice]
            slot = self._slots[request.slot_index]
            slot.pending = list(chosen.tokens)
            slot.samples += 1
            slot.buffer.append({
                "obs_tokens": list(request.obs_tokens),
                "obs_channels": request.channels,
                "action_feats": np.array(
                    request.action_feats[choice], dtype=np.float32, copy=True,
                ),
                "seat": request.seat,
                "game_id": slot.game_id,
            })

    def _action_row_for(self, batch_index: int, token: int) -> np.ndarray:
        """Locate the unique current legal V2 row carrying ``token``."""
        step = self._current_step
        rows = step.available_actions.tokens[batch_index]
        legal = step.available_actions.weights[batch_index] > 0
        matches = np.flatnonzero(legal & (rows[:, ACTION_TOKEN_COLUMN] == int(token)))
        if len(matches) != 1:
            raise V1ProtocolError(
                f"token {token} does not map to exactly one legal row in slot {batch_index}"
            )
        return np.array(rows[matches[0]], dtype=np.int64, copy=True)

    def _flush_terminal_slot(
        self, batch_index: int, slot: _SlotState, step: Any,
    ) -> None:
        rewards = np.asarray(step.rewards[batch_index], dtype=np.float64)
        if slot.pending:
            raise V1ProtocolError(
                "a game ended with an uncommitted candidate prefix; token protocol diverged"
            )
        if slot.buffer:
            rows = len(slot.buffer)
            lengths = np.array(
                [len(sample["obs_tokens"]) for sample in slot.buffer], dtype=np.int32,
            )
            max_length = max(int(lengths.max()), 1)
            tokens = np.full((rows, max_length), int(self._replay.pad_token), dtype=np.int32)
            for index, sample in enumerate(slot.buffer):
                tokens[index, : len(sample["obs_tokens"])] = sample["obs_tokens"]
            push = {
                "obs_tokens": tokens,
                "obs_token_lengths": lengths,
                "obs_channels": np.stack([sample["obs_channels"] for sample in slot.buffer]),
                "action_feats": np.stack([sample["action_feats"] for sample in slot.buffer]),
                # z is the acting seat's terminal team rank point divided by
                # three (TRAINING_REWARD_SCALE), gamma = 1, no discounting.
                "z": np.array(
                    [
                        float(rewards[sample["seat"]]) * TRAINING_REWARD_SCALE
                        for sample in slot.buffer
                    ],
                    dtype=np.float32,
                ),
                "seat": np.array([sample["seat"] for sample in slot.buffer], dtype=np.int8),
                "game_id": np.full(rows, slot.game_id, dtype=np.int64),
            }
            pushed = self._replay.push_batch(push)
            self.samples_pushed += pushed
        self.games_completed += 1
        digest = hashlib.sha256(
            b",".join(str(token).encode("ascii") for token in slot.action_tokens),
        ).hexdigest()
        self._game_records.append({
            "game_id": slot.game_id,
            "decisions": slot.decisions,
            "samples": slot.samples,
            "action_sha256": digest,
        })
        fresh = _SlotState(game_id=self._next_game_id())
        # The terminal step's observations already belong to the auto-reset
        # deal, so the streams restart from its reset events.
        self._install_reset_streams(fresh, step.observations.tokens[batch_index])
        self._slots[batch_index] = fresh

    # ------------------------------------------------------------------
    # Exact state transfer
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        self._require_healthy()
        return {
            "schema": SAMPLER_STATE_SCHEMA,
            "num_envs": int(self._config.num_envs),
            "device_type": self._device.type,
            "env_seed": int(self._config.env_seed),
            "backend": str(self._config.backend),
            "max_horizon": int(self._max_horizon),
            "max_obs_tokens": int(self._max_obs_tokens),
            "epsilon_start": float(self._config.epsilon_start),
            "epsilon_final": float(self._config.epsilon_final),
            "epsilon_decay_samples": int(self._config.epsilon_decay_samples),
            "context_schedule": (
                None if self._context_schedule is None else dict(self._context_schedule)
            ),
            "envs": self.envs.serialize(),
            "rng": self._rng.bit_generator.state,
            "games_started": self._games_started,
            "games_completed": self.games_completed,
            "samples_pushed": self.samples_pushed,
            "decisions_total": self.decisions_total,
            "single_decisions_total": self.single_decisions_total,
            "multi_decisions_total": self.multi_decisions_total,
            "argmax_hits": self.argmax_hits,
            "illegal_count": self.illegal_count,
            "nonfinite_count": self.nonfinite_count,
            "obs_window_truncations": self.obs_window_truncations,
            "truncated_count": self.truncated_count,
            "slots": [
                {
                    "game_id": slot.game_id,
                    "streams": [list(stream) for stream in slot.streams],
                    "pending": list(slot.pending),
                    "decisions": slot.decisions,
                    "samples": slot.samples,
                    "action_tokens": list(slot.action_tokens),
                    "buffer": [
                        {
                            "obs_tokens": list(sample["obs_tokens"]),
                            "obs_channels": np.array(sample["obs_channels"], copy=True),
                            "action_feats": np.array(sample["action_feats"], copy=True),
                            "seat": int(sample["seat"]),
                            "game_id": int(sample["game_id"]),
                        }
                        for sample in slot.buffer
                    ],
                }
                for slot in self._slots
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != SAMPLER_STATE_SCHEMA:
            raise ValueError(
                f"unsupported DMC sampler state schema {state.get('schema')!r}; "
                f"expected {SAMPLER_STATE_SCHEMA!r}"
            )
        expected = {
            "num_envs": int(self._config.num_envs),
            "device_type": self._device.type,
            "env_seed": int(self._config.env_seed),
            "backend": str(self._config.backend),
            "max_horizon": int(self._max_horizon),
            "max_obs_tokens": int(self._max_obs_tokens),
            "epsilon_start": float(self._config.epsilon_start),
            "epsilon_final": float(self._config.epsilon_final),
            "epsilon_decay_samples": int(self._config.epsilon_decay_samples),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"DMC sampler state {key} mismatch")
        raw_schedule = state.get("context_schedule")
        if (raw_schedule is None) != (self._context_schedule is None):
            raise ValueError("DMC sampler state context_schedule mismatch")
        if raw_schedule is not None and dict(raw_schedule) != dict(self._context_schedule):
            raise ValueError("DMC sampler state context_schedule mismatch")
        envs = self._game.EnvBatch.deserialize(bytes(state["envs"]))
        raw_slots = state.get("slots")
        if not isinstance(raw_slots, list) or len(raw_slots) != int(self._config.num_envs):
            raise ValueError("DMC sampler state slots mismatch")
        slots = []
        for raw in raw_slots:
            buffer = []
            for sample in raw["buffer"]:
                channels = np.asarray(sample["obs_channels"])
                if channels.shape != (self._replay.channel_dim,):
                    raise ValueError("restored sample channel width mismatch")
                buffer.append({
                    "obs_tokens": [int(token) for token in sample["obs_tokens"]],
                    "obs_channels": channels.astype(np.uint8, copy=True),
                    "action_feats": np.asarray(sample["action_feats"], dtype=np.float32),
                    "seat": int(sample["seat"]),
                    "game_id": int(sample["game_id"]),
                })
            slot = _SlotState(game_id=int(raw["game_id"]))
            slot.streams = [[int(token) for token in stream] for stream in raw["streams"]]
            slot.pending = [int(token) for token in raw["pending"]]
            slot.decisions = int(raw["decisions"])
            slot.samples = int(raw["samples"])
            slot.action_tokens = [int(token) for token in raw.get("action_tokens", [])]
            slot.buffer = buffer
            slots.append(slot)
        self.envs = envs
        # observe() rebuilds the current information sets (including legal
        # rows) from the restored environments without mutating them.
        self._current_step = envs.observe()
        self._slots = slots
        self._rng.bit_generator.state = state["rng"]
        self._games_started = int(state["games_started"])
        self.games_completed = int(state["games_completed"])
        self.samples_pushed = int(state["samples_pushed"])
        self.decisions_total = int(state["decisions_total"])
        self.single_decisions_total = int(state["single_decisions_total"])
        self.multi_decisions_total = int(state["multi_decisions_total"])
        self.argmax_hits = int(state["argmax_hits"])
        self.illegal_count = int(state["illegal_count"])
        self.nonfinite_count = int(state["nonfinite_count"])
        self.obs_window_truncations = int(state["obs_window_truncations"])
        self.truncated_count = int(state["truncated_count"])
        self._failed = False
        self._failure_info = None


__all__ = [
    "DMCSampleSink",
    "DMCSampler",
    "DMCSamplerConfig",
    "DMCSamplerFailedStateError",
    "SAMPLER_STATE_SCHEMA",
    "linear_epsilon",
    "select_candidate_index",
]
