# SPDX-License-Identifier: MIT
"""Stage 07C: DMC-only thin environment driver.

The 07A profile proved that 53-66% of a DMC collection second is spent inside
:class:`guandan_marvel.game.EnvBatch.step`, and that the cost is the adapter's
*validation* contract, not candidate enumeration: every environment token step
clones the protocol transactionally, re-enumerates the legal action set to
rebuild all V2 action rows twice, validates eagerly, and encodes the 512-wide
actor channels for all four seats even though DMC only ever reads the acting
seat's row.

This module is a **DMC-only** driver with the same observable behaviour.  It
never edits ``guandan/`` or ``EnvBatch``; it drives the accepted protocol
directly and reuses the adapter's own encoders (``_encode_channels``,
``_pad_event``, ``_reset_events``, ``_step_events``, ``_action_row``) so the
byte-level vocabulary has exactly one source of truth.

Three levers, each one a removal of redundant work — never a semantic change:

1. **Acting-seat channels only.**  ``_build_output`` encodes the 512-wide
   actor channels for the seat that acts at the *next* decision (the post-step
   active seat), which is the only row ``DMCSampler``/``evaluate`` read.  The
   other three rows stay zero; nothing consumes them.
2. **Candidate rows are built once (or not at all).**  The DMC sampler already
   holds the complete candidate token sequence from the enumerator, so the
   collection path submits **tokens** (``step_tokens``) and never rebuilds all
   legal rows to match one.  ``expose_action_rows=True`` restores the row
   surface for the evaluation lanes whose rule/random opponents select by row
   index — the rows are then built exactly like the reference.
3. **No per-token transactional re-validation.**  A token sequence handed to
   this driver is the canonical sequence of a legal action produced by the same
   enumerator the adapter validates against, so re-proving every prefix token
   is redundant.  The driver validates at the **sequence boundaries** instead:
   the first token of a sequence must be a live legal-next token, the complete
   sequence must be a live canonical action, ``apply_action`` runs the A02
   invariant check on the commit, a nonterminal commit must leave a live legal
   prefix, and a terminal commit runs the adapter's full
   ``_validate_protocol`` once.  Any violation raises immediately and poisons
   the driver (fail-closed, never a silent skip).

Everything else is inherited verbatim: terminal rewards/ranking/outcome come
from ``guandan.round.apply_action`` + ``RoundState``; event rows come from the
adapter's ``_reset_events``/``_step_events``; ``serialize`` emits the *same*
payload as ``EnvBatch.serialize`` (byte-identical, asserted by the differential
suite), so a checkpoint written by either driver restores under either.

The equivalence contract is enforced by ``tests/dmc/test_env_fast_differential``
(suite id :data:`DIFFERENTIAL_SUITE`): at least 200 random games driven through
both drivers in lockstep must agree byte-for-byte on every acting-seat
observation token row, every acting-seat channel row, every legal action
sequence, the public state digest and the full protocol digest after each
action, and the terminal rewards/ranking/outcome.  There is no tolerance and no
sampling of states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from guandan.action_state import COMMIT_TOKEN, PASS_TOKEN, ProtocolError
from guandan.environment import GuandanEnv
from guandan.environment_types import GameConfig
from guandan.round import apply_action
from guandan.state import EpisodeTerminatedError, Phase
from guandan_marvel.game import (
    ACTION_VERSION, AdapterCapacityError, AvailableActionBatch, BACKEND,
    ENCODING_VERSION, ENV_VERSION, EnvBatch, EnvStep, ObservationBatch,
    SERIALIZATION_SCHEMA, TrajectoryHorizonError, _action_row, _encode_channels,
    _pad_event, _reset_events, _step_events, max_action_tokens,
    max_available_actions, max_game_horizon, max_observation_tokens,
    num_channels, num_players,
)

#: Version of this driver.  A behaviour-affecting change must bump it together
#: with :data:`DIFFERENTIAL_SUITE`, because the Gate L plan declares both.
DRIVER_VERSION = "dmc-env-fast-v1"

#: Differential-suite identifier.  The Gate L plan (``plan.json``) declares this
#: exact string, so a delivery can never claim an equivalence it did not run.
DIFFERENTIAL_SUITE = "dmc-env-fast-differential-v1"

#: Column of ``ACTION_TOKEN_COLUMN`` in a V2 action row (``ACTION_SCHEMA``:
#: ROW_BEGIN, PHASE, PREFIX_STAGE, KIND, PAYLOAD_MARKER, V1_TOKEN_ID, ...).
_TOKEN_COLUMN = 5


@dataclass(frozen=True, slots=True)
class _RoundView:
    """The three pre-step round fields ``_step_events`` reads."""

    phase: Phase
    active_seat: int
    current_winning: object


@dataclass(frozen=True, slots=True)
class _ProtocolView:
    """Pre-step snapshot passed to the adapter's event builder."""

    round_state: _RoundView
    _prefix_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _StepFlag:
    """The single ``ProtocolStepResult`` field ``_step_events`` reads."""

    is_commit: bool


_COMMIT = _StepFlag(True)
_PREFIX = _StepFlag(False)


def _pre_step_view(protocol) -> _ProtocolView:
    state = protocol.round_state
    return _ProtocolView(
        _RoundView(state.phase, int(state.active_seat), state.current_winning),
        protocol._prefix_tokens,
    )


class DMCEnvBatch:
    """DMC-only batch driver; same observations, far less redundant validation.

    ``expose_action_rows=False`` (collection) leaves
    ``available_actions.tokens`` with zero width and expects submissions through
    :meth:`step_tokens`.  ``expose_action_rows=True`` (evaluation) builds the
    complete legal-row surface exactly like :class:`EnvBatch`, so row-selecting
    opponents (``RuleV2Agent``/``RandomV2Agent``) keep working unchanged.
    """

    #: Marks the token-submission surface for ``DMCSampler``.
    accepts_token_submission = True

    def __init__(
        self,
        batch_size: int,
        *,
        configs: Sequence[GameConfig] | None = None,
        auto_reset: bool = True,
        max_horizon: int = max_game_horizon,
        action_capacity: int = max_available_actions,
        observation_capacity: int = max_observation_tokens,
        expose_action_rows: bool = False,
        boundary_validation: bool = True,
    ) -> None:
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(auto_reset, bool):
            raise TypeError("auto_reset must be bool")
        if isinstance(max_horizon, bool) or not isinstance(max_horizon, Integral) or int(max_horizon) <= 0:
            raise ValueError("max_horizon must be a positive integer")
        if not isinstance(expose_action_rows, bool):
            raise TypeError("expose_action_rows must be bool")
        if not isinstance(boundary_validation, bool):
            raise TypeError("boundary_validation must be bool")
        self.batch_size = int(batch_size)
        self.auto_reset = auto_reset
        self.max_horizon = int(max_horizon)
        self.action_capacity = int(action_capacity)
        self.observation_capacity = int(observation_capacity)
        self.expose_action_rows = expose_action_rows
        self.boundary_validation = boundary_validation
        if self.max_horizon > max_game_horizon:
            raise ValueError(f"max_horizon cannot exceed {max_game_horizon}")
        if self.action_capacity > max_available_actions or self.observation_capacity > max_observation_tokens:
            raise ValueError("configured capacity exceeds the V2 schema")
        if self.action_capacity <= 0 or self.observation_capacity <= 0:
            raise ValueError("configured capacities must be positive")

        if configs is None:
            configs = tuple(GameConfig(max_token_steps=self.max_horizon) for _ in range(self.batch_size))
        else:
            configs = tuple(configs)
            if len(configs) != self.batch_size or any(not isinstance(config, GameConfig) for config in configs):
                raise ValueError("configs must contain one GameConfig per slot")
            configs = tuple(GameConfig(
                level_rank=config.level_rank, seed=config.seed,
                leader_seat=config.leader_seat, previous_result=config.previous_result,
                max_token_steps=self.max_horizon,
            ) for config in configs)
        self._envs = tuple(GuandanEnv(config) for config in configs)
        self._events: tuple[tuple[tuple[int, ...], ...], ...] = tuple(
            ((),) * num_players for _ in range(self.batch_size)
        )
        self._trajectory_steps = np.zeros(self.batch_size, dtype=np.int64)
        self._is_reset = np.zeros(self.batch_size, dtype=np.bool_)
        self._context_schedule: tuple[Mapping[str, object], ...] = ()
        self._context_offset = 0
        self._context_deal_counter = 0
        #: Fail-closed latch: a mid-batch exception leaves some slots advanced
        #: and others not, so the driver refuses further use instead of
        #: pretending the batch is consistent.
        self._poisoned = False
        self._poison_reason: str | None = None

    # ------------------------------------------------------------------
    # Introspection / lifecycle helpers
    # ------------------------------------------------------------------
    @property
    def envs(self) -> tuple[GuandanEnv, ...]:
        return self._envs

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _poison(self, error: BaseException) -> None:
        self._poisoned = True
        self._poison_reason = f"{type(error).__name__}: {error}"

    def _require_usable(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "the fast environment driver is unusable after a failed step; "
                f"restore from a checkpoint (failure={self._poison_reason})"
            )

    def configure_context_schedule(
        self, configs: Sequence[Mapping[str, object]], *, offset: int = 0,
    ) -> None:
        """Install the ordered context list and the iteration offset.

        Same contract as ``EnvBatch.configure_context_schedule``: consecutive
        slots consume consecutive positions, the per-deal counter advances by
        one and restarts at ``offset`` exactly once per collection.
        """
        rows = [dict(row) for row in configs]
        if not rows:
            raise ValueError("context schedule must be non-empty")
        if int(offset) < 0:
            raise ValueError("context offset must be non-negative")
        self._context_schedule = tuple(rows)
        self._context_offset = int(offset)
        self._context_deal_counter = 0

    def _next_context_row(self) -> Mapping[str, object] | None:
        if not self._context_schedule:
            return None
        index = (self._context_offset + self._context_deal_counter) % len(self._context_schedule)
        self._context_deal_counter += 1
        return self._context_schedule[index]

    # ------------------------------------------------------------------
    # Protocol-level fast paths
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_paths(protocol):
        """The protocol's memoized canonical sequences (never re-enumerated here)."""
        return protocol._candidate_sequences()

    def _commit_action(self, protocol, token: int):
        """Resolve the canonical action of ``prefix + (token,)`` (sequence boundary)."""
        prefix = protocol._prefix_tokens
        sequence = prefix + (token,) if prefix else (token,)
        for candidate, action in self._canonical_paths(protocol):
            if candidate == sequence:
                return action
        raise ProtocolError(
            "the fast driver received a token sequence that is not a live canonical "
            f"action (prefix={list(prefix)}, token={token})"
        )

    def _extend_prefix(self, protocol, token: int) -> None:
        """Append one known-legal prefix token without re-validating the prefix."""
        if not protocol._prefix_tokens:
            legal_first = {candidate[0] for candidate, _action in self._canonical_paths(protocol)}
            if token not in legal_first:
                raise ProtocolError(
                    f"token {token} is not a legal next token for the current state"
                )
        protocol.token_step += 1
        protocol._prefix_tokens = protocol._prefix_tokens + (token,)
        if protocol.token_step >= protocol.max_token_steps:
            protocol.truncated = True
            raise TrajectoryHorizonError("V1 token horizon reached before terminal")

    def _commit(self, env: GuandanEnv, protocol, action) -> tuple[np.ndarray, bool]:
        """Apply one canonical action; validate the resulting state once."""
        protocol.token_step += 1
        apply_action(protocol.round_state, action)  # clones + A02 invariant check
        protocol._prefix_tokens = ()
        done = bool(protocol.round_state.done)
        if done:
            # Token clock belongs to A03/A04, never to A02's atomic approximation
            # (mirrors GuandanEnv.step).
            protocol.round_state.finish_token_step = protocol.token_step
        if protocol.token_step >= protocol.max_token_steps and not done:
            protocol.truncated = True
            raise TrajectoryHorizonError("V1 token horizon reached before terminal")
        if done:
            # Full adapter validation exactly once per game (no legal enumeration
            # on a terminal state).
            env._validate_protocol(protocol)
            reward = protocol.round_state.rewards.astype(np.float32, copy=True)
            return reward, True
        if self.boundary_validation:
            state = protocol.round_state
            if protocol.committed_step != len(state.history):
                raise ProtocolError("committed_step must match committed history length")
            if protocol.committed_step > protocol.token_step:
                raise ProtocolError("committed_step exceeds token_step")
            if not self._canonical_paths(protocol):
                raise ProtocolError("nonterminal state has no live legal prefix")
        return np.zeros(num_players, dtype=np.float32), False

    @staticmethod
    def _base_info() -> dict:
        """The adapter's per-step provenance keys (``EnvBatch.step``)."""
        return {
            "termination_reason": None,
            "backend": BACKEND,
            "action_version": ACTION_VERSION,
            "environment_version": ENV_VERSION,
            "encoding_version": ENCODING_VERSION,
        }

    def _terminal_info(self, protocol, before_phase: Phase) -> dict:
        """Terminal metadata, captured before any auto-reset.

        Same fields as the adapter's ``info`` except ``terminal_serialized_v1``:
        the fast driver never serializes a whole game per terminal step (no
        consumer reads that key; ``env.serialize()`` remains available).
        """
        state = protocol.round_state
        info = self._base_info()
        info.update(
            ranking=tuple(state.ranking),
            winner_team=state.winner_team,
            outcome_class=state.outcome_class,
            team_reward=tuple(float(value) for value in state.team_rewards),
            finish_token_step=protocol.token_step,
            finish_committed_step=protocol.committed_step,
        )
        if before_phase in (Phase.TRIBUTE, Phase.RETURN) and state.phase is Phase.PLAY:
            info["resolved_transfers"] = {
                "tribute": tuple((donor, recipient, card.card_id)
                                 for donor, recipient, card in state.tribute_transfers),
                "return": tuple((recipient, donor, card.card_id)
                                for recipient, donor, card in state.return_transfers),
            }
        return info

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------
    def _candidate_actions(self, protocol):
        legal = tuple(int(token) for token in protocol.legal_next_tokens().tolist())
        if len(legal) > self.action_capacity:
            raise AdapterCapacityError(
                f"legal action count {len(legal)} exceeds capacity {self.action_capacity}"
            )
        actions = tuple(_action_row(protocol, token) for token in legal)
        if len({action.row for action in actions}) != len(actions):
            raise ProtocolError("V2 action rows are not unique")
        return actions

    def _build_output(self, events, *, rewards=None, terminal=None, infos=None) -> EnvStep:
        size = len(self._envs)
        obs_tokens = np.zeros((size, num_players, self.observation_capacity), dtype=np.int64)
        obs_channels = np.zeros((size, num_players, num_channels), dtype=np.uint8)
        row_width = self.action_capacity if self.expose_action_rows else 0
        action_tokens = np.zeros((size, row_width, max_action_tokens), dtype=np.int64)
        action_weights = np.zeros((size, row_width), dtype=np.float32)
        action_commit = np.zeros((size, row_width), dtype=np.bool_)
        player_ids = np.full(size, -1, dtype=np.int64)
        for batch_index, (env, seat_events) in enumerate(zip(self._envs, events, strict=True)):
            if env.done or env.truncated:
                continue
            protocol = env._require_protocol()
            active_seat = int(protocol.round_state.active_seat)
            player_ids[batch_index] = active_seat
            for observer in range(num_players):
                obs_tokens[batch_index, observer] = _pad_event(
                    seat_events[observer], self.observation_capacity,
                )
            # Lever 1: only the seat that acts at the next decision is encoded.
            obs_channels[batch_index, active_seat] = _encode_channels(protocol, active_seat)
            if self.expose_action_rows:
                for action_index, action in enumerate(self._candidate_actions(protocol)):
                    action_tokens[batch_index, action_index] = action.row
                    action_weights[batch_index, action_index] = 1.0
                    action_commit[batch_index, action_index] = action.is_commit
        rewards_array = np.zeros((size, num_players), np.float32) if rewards is None else np.asarray(rewards, np.float32)
        terminal_array = np.zeros(size, np.bool_) if terminal is None else np.asarray(terminal, np.bool_)
        if rewards_array.shape != (size, num_players) or terminal_array.shape != (size,):
            raise ValueError("reward or terminal shape mismatch")
        info_tuple = tuple({} for _ in range(size)) if infos is None else tuple(dict(info) for info in infos)
        return EnvStep(
            rewards_array.copy(), terminal_array.copy(), player_ids,
            ObservationBatch(obs_tokens, obs_channels),
            AvailableActionBatch(action_tokens, action_weights, action_commit), info_tuple,
        )

    def observe(self) -> EnvStep:
        """Return the current information sets without mutation."""
        self._require_usable()
        if not bool(np.all(self._is_reset)):
            raise RuntimeError("every slot must be reset before observe")
        return self._build_output(self._events)

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------
    def reset(self, seed: int = 0) -> EnvStep:
        self._require_usable()
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        try:
            if self._context_schedule:
                self._context_deal_counter = 0
            events = []
            for index, env in enumerate(self._envs):
                if self._context_schedule:
                    EnvBatch._apply_context(env, self._next_context_row())
                env.reset(seed=int(seed) + index)
                events.append(_reset_events(env._require_protocol()))
            output = self._build_output(events)
        except BaseException as error:  # noqa: BLE001 - poison on any failure
            self._poison(error)
            raise
        self._events = tuple(events)
        self._trajectory_steps.fill(0)
        self._is_reset.fill(True)
        return output

    def step(self, actions) -> EnvStep:
        """Compatibility entry point: submit V2 action rows (token column only)."""
        if isinstance(actions, np.ndarray):
            array = actions
            if array.ndim != 2 or array.shape[1] != max_action_tokens:
                raise ValueError(f"rows must have shape [B,{max_action_tokens}]")
            if array.shape[0] != self.batch_size:
                raise ValueError("rows must contain one action row per slot")
            if array.dtype.kind not in "iu":
                raise ValueError("rows must use an integer dtype")
            return self.step_tokens(array[:, _TOKEN_COLUMN])
        raise TypeError(
            "the fast driver accepts V2 action rows (np.ndarray) or tokens via "
            "step_tokens(); it never re-matches V2Action objects"
        )

    def step_tokens(self, tokens: Sequence[int] | np.ndarray) -> EnvStep:
        """Advance every slot by one token (DMC's native submission surface)."""
        self._require_usable()
        array = np.asarray(tokens)
        if array.shape != (self.batch_size,) or array.dtype.kind not in "iu":
            raise ValueError(f"tokens must be an integer vector with shape ({self.batch_size},)")
        if not bool(np.all(self._is_reset)):
            raise RuntimeError("every slot must be reset before step")
        rewards = np.zeros((self.batch_size, num_players), np.float32)
        terminal = np.zeros(self.batch_size, np.bool_)
        infos: list[dict] = []
        events = list(self._events)
        steps = self._trajectory_steps
        try:
            for index, env in enumerate(self._envs):
                if env.done or env.truncated:
                    raise EpisodeTerminatedError("inactive V2 slot cannot be stepped")
                if int(steps[index]) >= self.max_horizon:
                    raise TrajectoryHorizonError("trajectory horizon exceeded before terminal")
                token = int(array[index])
                protocol = env._require_protocol()
                view = _pre_step_view(protocol)
                if token == PASS_TOKEN or token == COMMIT_TOKEN:
                    action = self._commit_action(protocol, token)
                    rewards[index], done = self._commit(env, protocol, action)
                    events[index] = _step_events(view, _COMMIT, token, protocol)
                    steps[index] += 1
                    terminal[index] = done
                    if done:
                        info = self._terminal_info(protocol, view.round_state.phase)
                        if self.auto_reset:
                            if self._context_schedule:
                                EnvBatch._apply_context(env, self._next_context_row())
                            env.reset()
                            events[index] = _reset_events(env._require_protocol())
                            steps[index] = 0
                        else:
                            events[index] = ((),) * num_players
                        infos.append(info)
                    else:
                        infos.append(self._base_info())
                else:
                    self._extend_prefix(protocol, token)
                    events[index] = _step_events(view, _PREFIX, token, protocol)
                    steps[index] += 1
                    infos.append(self._base_info())
            output = self._build_output(events, rewards=rewards, terminal=terminal, infos=infos)
        except BaseException as error:  # noqa: BLE001 - poison on any failure
            self._poison(error)
            raise
        self._events = tuple(events)
        return output

    # ------------------------------------------------------------------
    # Serialization: byte-identical to EnvBatch (one transport, two drivers)
    # ------------------------------------------------------------------
    def serialize(self) -> bytes:
        self._require_usable()
        if not bool(np.all(self._is_reset)):
            raise RuntimeError("cannot serialize before reset")
        payload = {
            "schema": SERIALIZATION_SCHEMA, "backend": BACKEND,
            "action_version": ACTION_VERSION, "environment_version": ENV_VERSION,
            "encoding_version": ENCODING_VERSION, "auto_reset": self.auto_reset,
            "max_horizon": self.max_horizon, "action_capacity": self.action_capacity,
            "observation_capacity": self.observation_capacity,
            "trajectory_steps": self._trajectory_steps.tolist(),
            "events": [[[int(token) for token in event] for event in seat_events] for seat_events in self._events],
            "context_schedule": [dict(row) for row in self._context_schedule],
            "context_offset": int(self._context_offset),
            "context_deal_counter": int(self._context_deal_counter),
            "envs": [json.loads(env.serialize()) for env in self._envs],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def deserialize(
        cls, payload: bytes, *, expose_action_rows: bool = False,
        boundary_validation: bool = True,
    ) -> "DMCEnvBatch":
        """Restore a payload written by either driver.

        ``EnvBatch.deserialize`` owns the transport validation (versions, event
        shape, counters); this method only adopts the validated state, so a
        reference checkpoint and a fast checkpoint are interchangeable.
        """
        reference = EnvBatch.deserialize(payload)
        result = cls(
            reference.batch_size, auto_reset=reference.auto_reset,
            max_horizon=reference.max_horizon,
            action_capacity=reference.action_capacity,
            observation_capacity=reference.observation_capacity,
            expose_action_rows=expose_action_rows,
            boundary_validation=boundary_validation,
        )
        result._envs = reference.envs
        result._events = reference._events
        result._trajectory_steps = reference._trajectory_steps.copy()
        result._context_schedule = reference._context_schedule
        result._context_offset = reference._context_offset
        result._context_deal_counter = reference._context_deal_counter
        result._is_reset.fill(True)
        return result


__all__ = [
    "DIFFERENTIAL_SUITE",
    "DMCEnvBatch",
    "DRIVER_VERSION",
]
