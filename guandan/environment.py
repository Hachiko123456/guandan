"""GD-ENV-0.1 policy-safe single environment; no training or rule redefinition."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from numbers import Integral

import numpy as np

from .action_state import ProtocolError, StepwiseActionState, canonical_tokens
from .cards import Card, Rank
from .encoding import encode_observation
from .environment_types import (
    ACTION_VERSION, ENCODING_VERSION, ENV_VERSION, RULES_VERSION,
    SERIALIZATION_VERSION, BatchObservation, BatchStepResult, CommittedAction,
    GameConfig, Observation, ObservationSpec, StepResult, strict_int,
)
from .state import EpisodeTerminatedError, IllegalActionError, Phase, PreviousHandResult, RoundState, StateInvariantError, TEAM_OF
from .round import enumerate_legal_actions
from .combos import CombinationKind, recognize_combinations


class _CachedProtocol(StepwiseActionState):
    """Per-instance memoization of A03 paths, keyed by the ENTIRE round snapshot.

    No legal action is added/removed. Prefix-only steps reuse immutable paths;
    any round mutation (including test/debug mutation) invalidates the cache.
    Caches are neither policy-visible nor serialized, and are never global.
    """
    def _candidate_sequences(self):
        key = json.dumps(self.round_state.serialize(), sort_keys=True, separators=(",", ":"))
        key = (key, self.max_action_tokens, self.max_legal_next_tokens)
        if getattr(self, "_cache_key", None) != key:
            paths = {}
            for action in enumerate_legal_actions(self.round_state):
                sequence = canonical_tokens(action)
                if len(sequence) > self.max_action_tokens:
                    raise ProtocolError("canonical action exceeds max_action_tokens")
                previous = paths.get(sequence)
                if previous is not None:
                    # A01 distinguishes natural-vs-substitute metadata even
                    # when the v1 token sequence is identical. A03's accepted
                    # canonical policy selects the natural declaration once.
                    left, right = previous.declaration, action.declaration
                    same_semantics = (
                        previous.player_id == action.player_id
                        and previous.kind == action.kind
                        and previous.cards == action.cards
                        and left is not None and right is not None
                        and left.kind == right.kind
                        and left.cards == right.cards
                        and left.level_rank == right.level_rank
                        and left.comparison_rank == right.comparison_rank
                        and left.comparison_key == right.comparison_key
                        and left._wild_assignment_items == right._wild_assignment_items
                    )
                    if not same_semantics:
                        raise ProtocolError("two distinct committed actions share one canonical token sequence")
                    differing_natural = set(left.natural_wild_ids) ^ set(right.natural_wild_ids)
                    assignments = left.wild_assignments
                    if any(
                        not Card(card_id).is_level_wild(left.level_rank)
                        or assignments.get(card_id) != (Card(card_id).rank, Card(card_id).suit)
                        for card_id in differing_natural
                    ):
                        raise ProtocolError("canonical collision is not natural-wild equivalence")
                    # Multiple wilds have independent flags: prefer ALL natural
                    # identities, not merely the first declaration with any flag.
                    if len(right.natural_wild_ids) > len(left.natural_wild_ids):
                        paths[sequence] = action
                else:
                    paths[sequence] = action
            self._cache_paths = tuple(sorted(paths.items()))
            self._cache_key = key
        return [(sequence, action) for sequence, action in self._cache_paths]

    def clone(self):
        result = _CachedProtocol(
            self.round_state.clone(), max_action_tokens=self.max_action_tokens,
            max_legal_next_tokens=self.max_legal_next_tokens, max_token_steps=self.max_token_steps,
        )
        result.token_step = self.token_step
        result.truncated = self.truncated
        result._prefix_tokens = self._prefix_tokens
        # A03 paths reference only immutable actions/cards/declarations.
        if hasattr(self, "_cache_key"):
            result._cache_key = self._cache_key
            result._cache_paths = self._cache_paths
        return result


def _config_dict(config: GameConfig) -> dict:
    return asdict(config)


def _load_config(data: dict) -> GameConfig:
    data = dict(data)
    previous = data.get("previous_result")
    if previous is not None:
        data["previous_result"] = PreviousHandResult(**previous)
    return GameConfig(**data)


def _restore_protocol(data: dict) -> _CachedProtocol:
    """Reject malformed checkpoint metadata before A03's permissive coercions.

    This is transport validation, not an alternate rules evaluator.
    """
    if not isinstance(data, dict):
        raise ProtocolError("serialized protocol must be an object")
    for key in ("token_step", "max_action_tokens", "max_legal_next_tokens", "max_token_steps"):
        strict_int(data[key], key)
    if not isinstance(data["truncated"], bool):
        raise ProtocolError("truncated must be boolean")
    for token in data["prefix_tokens"]:
        strict_int(token, "prefix token", minimum=1, maximum=255)
    state = data["round_state"]
    for key in ("active_seat", "leader_seat"):
        strict_int(state[key], key, maximum=3)
    strict_int(state["committed_step"], "committed_step")
    strict_int(state["consecutive_passes"], "consecutive_passes", maximum=3)
    if not isinstance(state["done"], bool):
        raise ProtocolError("done must be boolean")
    protocol = _CachedProtocol.deserialize(json.dumps(data).encode("utf-8"))
    restored = protocol.round_state
    if tuple(state["ranking"]) != restored.ranking:
        raise ProtocolError("serialized rank indexes disagree with finish order")
    if restored.current_winning is None:
        if restored.current_winner_seat is not None:
            raise ProtocolError("trick winner exists without a winning declaration")
    elif restored.current_winner_seat not in range(4):
        raise ProtocolError("invalid trick winner seat")
    for action in restored.history:
        strict_int(action.player_id, "history actor", maximum=3)
        if isinstance(action.kind, CombinationKind):
            declaration = action.declaration
            if declaration is None or declaration.kind != action.kind or declaration.cards != action.cards:
                raise ProtocolError("history play declaration does not match physical cards")
            if declaration.level_rank != restored.level_rank or declaration not in recognize_combinations(action.cards, level_rank=restored.level_rank):
                raise ProtocolError("invalid serialized play declaration")
        elif action.kind not in {"pass", "tribute", "return"} or action.declaration is not None:
            raise ProtocolError("invalid serialized control action")
        elif (action.kind == "pass" and action.cards) or (action.kind != "pass" and len(action.cards) != 1):
            raise ProtocolError("invalid serialized control-action cards")
    if restored.current_winning is not None:
        latest_play = next((action for action in reversed(restored.history)
                            if isinstance(action.kind, CombinationKind)), None)
        if latest_play is None or latest_play.declaration != restored.current_winning or latest_play.player_id != restored.current_winner_seat:
            raise ProtocolError("current trick is not the latest committed play")
    return protocol


class GuandanEnv:
    """One explicit-reset slot. Policy observation and full-state APIs are separate."""
    def __init__(self, config: GameConfig, *, observation_spec: ObservationSpec | None = None):
        if not isinstance(config, GameConfig):
            raise TypeError("config must be GameConfig")
        if observation_spec is not None and not isinstance(observation_spec, ObservationSpec):
            raise TypeError("observation_spec must be ObservationSpec")
        self.config = config
        self.observation_spec = observation_spec or ObservationSpec()
        self._protocol: _CachedProtocol | None = None
        self._rng = np.random.default_rng(config.seed)
        self._reset_count = 0

    @property
    def done(self) -> bool:
        return self._protocol is not None and self._protocol.done

    @property
    def truncated(self) -> bool:
        return self._protocol is not None and self._protocol.truncated

    @property
    def token_step(self) -> int:
        return 0 if self._protocol is None else self._protocol.token_step

    @property
    def committed_step(self) -> int:
        return 0 if self._protocol is None else self._protocol.committed_step

    def _require_protocol(self) -> _CachedProtocol:
        if self._protocol is None:
            raise RuntimeError("environment must be reset before use")
        return self._protocol

    def _validate_protocol(self, protocol: StepwiseActionState) -> None:
        spec = self.observation_spec
        if (protocol.max_action_tokens, protocol.max_legal_next_tokens, protocol.max_token_steps) != (
            spec.max_action_tokens, spec.max_legal_next_tokens, self.config.max_token_steps
        ):
            raise ProtocolError("serialized protocol capacities/configuration mismatch")
        if protocol.round_state.level_rank != Rank(self.config.level_rank):
            raise ProtocolError("serialized level does not match configuration")
        if protocol.round_state.previous_result != self.config.previous_result:
            raise ProtocolError("serialized previous result does not match configuration")
        strict_int(protocol.token_step, "token_step")
        strict_int(protocol.committed_step, "committed_step")
        if protocol.committed_step > protocol.token_step:
            raise ProtocolError("committed_step exceeds token_step")
        if protocol.committed_step != len(protocol.round_state.history):
            raise ProtocolError("committed_step must match committed history length")
        if protocol.token_step > protocol.max_token_steps:
            raise ProtocolError("token_step exceeds episode limit")
        if protocol.truncated != (protocol.token_step == protocol.max_token_steps and not protocol.done):
            raise ProtocolError("truncation flag does not match episode limit")
        if protocol.done and protocol.truncated:
            raise ProtocolError("hand cannot be both terminal and truncated")
        protocol.round_state.check_invariants()
        if not protocol.done:
            state = protocol.round_state
            if state.phase is Phase.PLAY and (
                state.active_seat in state.finished_ranks or not state.hands[state.active_seat]
            ):
                raise StateInvariantError("nonterminal acting seat must be live")
        if not protocol.done and not protocol.truncated:
            legal = protocol.legal_next_tokens()
            if len(legal) == 0:
                raise ProtocolError("nonterminal state has no live legal prefix")
        elif protocol.done and protocol._prefix_tokens:
            raise ProtocolError("terminal state must have empty prefix")
        if protocol.truncated:
            prefix = protocol._prefix_tokens
            if not any(sequence[:len(prefix)] == prefix and len(sequence) > len(prefix)
                       for sequence, _action in protocol._candidate_sequences()):
                raise ProtocolError("truncated snapshot contains a dead prefix")
        state = protocol.round_state
        if state.done:
            expected = PreviousHandResult(state.ranking, state.winner_team, state.outcome_class)
            if (state.finish_token_step != protocol.token_step
                    or state.finish_committed_step != protocol.committed_step
                    or tuple(state.team_rewards) != tuple(1.0 if t == expected.winner_team else -1.0 for t in range(2))):
                raise StateInvariantError("terminal metadata/rewards disagree with episode clocks")

    def reset(self, *, seed: int | None = None, initial_state: bytes | None = None) -> Observation:
        """New episode; initial_state is a fresh A03 or A04 snapshot, not resume.

        deserialize() is the resume API and retains counters, prefix and flags.
        A failed reset (including encoder overflow) retains state AND RNG.
        """
        if seed is not None and initial_state is not None:
            raise ValueError("provide seed or initial_state, not both")
        candidate_rng = np.random.default_rng()
        candidate_rng.bit_generator.state = deepcopy(self._rng.bit_generator.state)
        if initial_state is not None:
            if not isinstance(initial_state, bytes):
                raise TypeError("initial_state must be bytes")
            data = json.loads(initial_state)
            if "serialization_version" in data:
                restored = self.deserialize(initial_state)
                if restored.config != self.config or restored.observation_spec != self.observation_spec:
                    raise ProtocolError("initial environment snapshot configuration differs")
                protocol = restored._require_protocol().clone()
            else:
                protocol = _restore_protocol(data)
                protocol.max_action_tokens = self.observation_spec.max_action_tokens
                protocol.max_legal_next_tokens = self.observation_spec.max_legal_next_tokens
                protocol.max_token_steps = self.config.max_token_steps
            if protocol.token_step or protocol.committed_step or protocol._prefix_tokens or protocol.done or protocol.truncated:
                raise ProtocolError("reset initial_state must be fresh with zero counters and empty prefix; use deserialize for resume")
        else:
            if seed is not None:
                deal_seed = strict_int(seed, "seed")
                candidate_rng = np.random.default_rng(deal_seed)
            elif self._reset_count == 0 and self.config.seed is not None:
                deal_seed = self.config.seed
            else:
                deal_seed = int(candidate_rng.integers(0, 2**63 - 1))
            state = RoundState.deal(
                seed=deal_seed, level_rank=Rank(self.config.level_rank),
                leader_seat=self.config.leader_seat, previous_result=self.config.previous_result,
            )
            protocol = _CachedProtocol(
                state, max_action_tokens=self.observation_spec.max_action_tokens,
                max_legal_next_tokens=self.observation_spec.max_legal_next_tokens,
                max_token_steps=self.config.max_token_steps,
            )
        self._validate_protocol(protocol)
        observation = encode_observation(protocol, self.observation_spec)
        self._protocol = protocol
        self._rng = candidate_rng
        self._reset_count += 1
        return observation

    def observe(self) -> Observation | None:
        if self._protocol is None or self.done or self.truncated:
            return None
        return encode_observation(self._protocol, self.observation_spec)

    def legal_tokens(self) -> np.ndarray:
        protocol = self._require_protocol()
        return protocol.legal_next_tokens().copy()

    @staticmethod
    def _receipt(action, *, unresolved_exchange: bool) -> CommittedAction | None:
        if action is None:
            return None
        redacted = unresolved_exchange and action.kind in {"TRIBUTE", "RETURN"}
        return CommittedAction(
            action.actor, action.phase, action.kind, action.family,
            () if redacted else tuple(action.card_ids), tuple(action.declared_ranks),
            action.declared_suit, tuple(action.wild_assignments), action.public_index,
        )

    def step(self, token_id: int) -> StepResult:
        protocol = self._require_protocol()
        if self.done or self.truncated:
            raise EpisodeTerminatedError("environment is terminal or truncated")
        if isinstance(token_id, (bool, np.bool_)) or not isinstance(token_id, Integral):
            raise IllegalActionError("token_id must be an integer, not a coerced value")
        trial = protocol.clone()
        before_phase = trial.round_state.phase
        result = trial.step(int(token_id))
        if trial.done:
            # Token clock belongs to A03/A04, never to A02's atomic approximation.
            trial.round_state.finish_token_step = trial.token_step
        self._validate_protocol(trial)
        observation = None if trial.done or trial.truncated else encode_observation(trial, self.observation_spec)
        state = trial.round_state
        info = {"termination_reason": "max_token_steps" if trial.truncated else None}
        if trial.done:
            info.update(
                ranking=tuple(state.ranking), winner_team=state.winner_team,
                outcome_class=state.outcome_class, team_reward=tuple(float(x) for x in state.team_rewards),
                finish_token_step=trial.token_step, finish_committed_step=trial.committed_step,
            )
        if before_phase in {Phase.TRIBUTE, Phase.RETURN} and state.phase is Phase.PLAY:
            info["resolved_transfers"] = {
                "tribute": tuple((d, r, c.card_id) for d, r, c in state.tribute_transfers),
                "return": tuple((r, d, c.card_id) for r, d, c in state.return_transfers),
            }
        output = StepResult(
            observation, result.rewards.astype(np.float32, copy=True), trial.done, trial.truncated,
            result.is_commit, self._receipt(result.committed_action, unresolved_exchange=state.phase in {Phase.TRIBUTE, Phase.RETURN}),
            trial.token_step, trial.committed_step, info,
        )
        self._protocol = trial  # Last operation: encoding/validation failures publish nothing.
        return output

    def clone(self) -> "GuandanEnv":
        result = GuandanEnv(self.config, observation_spec=self.observation_spec)
        result._rng.bit_generator.state = deepcopy(self._rng.bit_generator.state)
        result._reset_count = self._reset_count
        result._protocol = None if self._protocol is None else self._protocol.clone()
        return result

    def full_state(self) -> dict:
        """Explicit privileged debug/critic API. Never attached to policy results."""
        protocol = self._require_protocol()
        data = json.loads(protocol.serialize())
        data["committed_step"] = protocol.committed_step
        data["done"] = protocol.done
        data["rng_state"] = deepcopy(self._rng.bit_generator.state)
        return {"protocol": data, "round_state": data["round_state"], "config": _config_dict(self.config), "observation_spec": asdict(self.observation_spec), "rng_state": deepcopy(self._rng.bit_generator.state)}

    def serialize(self) -> bytes:
        protocol = self._require_protocol()
        data = dict(
            serialization_version=SERIALIZATION_VERSION, rules_version=RULES_VERSION,
            environment_version=ENV_VERSION, action_version=ACTION_VERSION, encoding_version=ENCODING_VERSION,
            config=_config_dict(self.config), observation_spec=asdict(self.observation_spec),
            protocol=json.loads(protocol.serialize()), rng_state=self._rng.bit_generator.state,
            reset_count=self._reset_count,
        )
        return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def deserialize(cls, payload: bytes) -> "GuandanEnv":
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        data = json.loads(payload)
        versions = dict(serialization_version=SERIALIZATION_VERSION, rules_version=RULES_VERSION,
                        environment_version=ENV_VERSION, action_version=ACTION_VERSION, encoding_version=ENCODING_VERSION)
        if any(data.get(key) != value for key, value in versions.items()):
            raise ProtocolError("unsupported serialized environment/rules/action/encoding version")
        env = cls(_load_config(data["config"]), observation_spec=ObservationSpec(**data["observation_spec"]))
        raw = data["protocol"]
        protocol = _restore_protocol(raw)
        env._validate_protocol(protocol)
        env._protocol = protocol
        env._rng.bit_generator.state = deepcopy(data["rng_state"])
        env._reset_count = strict_int(data["reset_count"], "reset_count", minimum=1)
        env.observe()  # Reject snapshots which cannot be encoded, before returning an env.
        return env


def __getattr__(name):
    # Lazy re-export avoids a circular import with env_batch's use of GuandanEnv.
    if name == "GuandanEnvBatch":
        from .env_batch import GuandanEnvBatch
        return GuandanEnvBatch
    raise AttributeError(name)


__all__ = ["GameConfig", "PreviousHandResult", "ObservationSpec", "Observation", "CommittedAction", "StepResult",
           "BatchObservation", "BatchStepResult", "GuandanEnv", "GuandanEnvBatch", "ProtocolError",
           "IllegalActionError", "EpisodeTerminatedError", "StateInvariantError"]
