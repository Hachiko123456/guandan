"""Versioned MARVEL V2 Python reference adapter for the accepted GuanDan rules."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from guandan.action_state import (
    BOS_TOKEN, CARD_TOKEN_BASE, COMMIT_TOKEN, FAMILY_ORDER, FAMILY_TOKEN_BASE,
    FOLLOW_PLAY_TOKEN, LEAD_PLAY_TOKEN, LENGTH_TOKEN_BASE, PASS_TOKEN,
    RANK_ORDER, RANK_TOKEN_BASE, RETURN_TOKEN, SUIT_TOKEN_BASE, TRIBUTE_TOKEN,
    WILD_TOKEN_BASE, ProtocolError, TokenCodec,
)
from guandan.cards import PRINTED_RANKS, Suit
from guandan.environment import GuandanEnv
from guandan.environment_types import GameConfig
from guandan.state import EpisodeTerminatedError, Phase, TEAM_OF
from guandan_marvel.config import GameSpec

ACTION_VERSION = "GD-ACTION-0.2-MARVEL"
ENV_VERSION = "GD-ENV-0.2-MARVEL"
ENCODING_VERSION = "GD-ENCODING-0.2-MARVEL"
SERIALIZATION_SCHEMA = "guandan-marvel-python-adapter-v1"
BACKEND = "python_reference"
GAME_IDENTIFIER = "GuandanMARVELV2PythonReference"

num_players = 4
num_tokens = 320
num_channels = 512
pad_token = 0
max_game_horizon = 4096
max_available_actions = 256
max_observation_tokens = 48
max_action_tokens = 8
NUM_PLAYERS = num_players
NUM_TOKENS = num_tokens
NUM_CHANNELS = num_channels
PAD_TOKEN = pad_token
MAX_GAME_HORIZON = max_game_horizon
MAX_AVAILABLE_ACTIONS = max_available_actions
MAX_OBSERVATION_TOKENS = max_observation_tokens
MAX_ACTION_TOKENS = max_action_tokens
protocol_version = f"{ACTION_VERSION}|{ENV_VERSION}|{ENCODING_VERSION}"
identifier = GAME_IDENTIFIER

# V2-only vocabulary. V1 semantic tokens remain unchanged in 0..255.
EVENT_RESET = 256
EVENT_PRIVATE_PREFIX = 257
EVENT_PUBLIC_COMMIT = 258
EVENT_PRIVATE_EXCHANGE_COMMIT = 259
EVENT_EXCHANGE_RESOLVED = 260
EVENT_HIDDEN_PRIVATE_STEP = 261
ACTION_ROW_BEGIN = 262
ACTION_STAGE_BASE = 263  # 263..295 encode prefix lengths 0..32.
ACTION_KIND_BASE = 296   # 296..303 encode the eight legal-row kinds.
ACTION_PAYLOAD = 304
ACTION_NON_COMMIT = 305
ACTION_COMMIT = 306
ACTION_ROW_END = 307
EVENT_END = 308

_KIND_NAMES = ("PASS", "COMMIT", "FAMILY", "RANK", "SUIT", "LENGTH", "CARD", "WILD_ASSIGNMENT")
_KIND_TO_CODE = {name: index for index, name in enumerate(_KIND_NAMES)}
_PHASE_INDEX = {Phase.PLAY: 0, Phase.TRIBUTE: 1, Phase.RETURN: 2, Phase.TERMINAL: 3}
_OUTCOME_INDEX = {"DOUBLE_DOWN": 0, "HEAD_THIRD": 1, "HEAD_LAST": 2}

# Actor-information channel layout. No slice contains other-seat ownership.
SELF_HAND_SLICE = slice(0, 108)
CURRENT_WINNING_CARDS_SLICE = slice(108, 216)
SELF_EXCHANGE_CARDS_SLICE = slice(216, 324)
SELF_PREFIX_TOKENS_SLICE = slice(324, 356)
REMAINING_COUNTS_SLICE = slice(356, 360)
FINISHING_RANKS_SLICE = slice(360, 364)
OBSERVER_ONE_HOT_SLICE = slice(364, 368)
ACTIVE_ONE_HOT_SLICE = slice(368, 372)
LEADER_ONE_HOT_SLICE = slice(372, 376)
WINNER_ONE_HOT_SLICE = slice(376, 380)
EMPTIED_ONE_HOT_SLICE = slice(380, 384)
PHASE_ONE_HOT_SLICE = slice(384, 388)
LEVEL_ONE_HOT_SLICE = slice(388, 401)
PASS_COUNT_ONE_HOT_SLICE = slice(401, 405)
TEAM_ONE_HOT_SLICE = slice(405, 407)
PREVIOUS_WINNER_TEAM_SLICE = slice(407, 409)
PREVIOUS_OUTCOME_SLICE = slice(409, 412)
PREVIOUS_RANKING_SLICE = slice(412, 416)
CURRENT_FAMILY_SLICE = slice(416, 426)
CURRENT_RANK_SLICE = slice(426, 441)
CURRENT_SIZE_SLICE = slice(441, 452)
TRIBUTE_DONOR_SLICE = slice(452, 456)
TRIBUTE_ORDER_SLICE = slice(456, 460)
TOKEN_STEP_BITS_SLICE = slice(460, 473)
COMMITTED_STEP_BITS_SLICE = slice(473, 486)

ACTION_SCHEMA: Mapping[str, object] = {
    "version": ACTION_VERSION,
    "row_length": max_action_tokens,
    "row": ["ROW_BEGIN", "PHASE", "PREFIX_STAGE", "KIND", "PAYLOAD_MARKER", "V1_TOKEN_ID", "IS_COMMIT", "ROW_END"],
    "execution": "The complete row must exactly match one current V1 legal-next row.",
}
ENV_SCHEMA: Mapping[str, object] = {
    "version": ENV_VERSION,
    "backend": BACKEND,
    "reset_step_fields": {
        "rewards": ["B", 4], "is_terminal": ["B"], "player_ids": ["B"],
        "observations.tokens": ["B", 4, max_observation_tokens],
        "observations.channels": ["B", 4, num_channels],
        "available_actions.tokens": ["B", max_available_actions, max_action_tokens],
        "available_actions.weights": ["B", max_available_actions],
        "available_actions.is_commit": ["B", max_available_actions],
    },
    "capacity_policy": "hard_fail_no_truncation",
    "critic_privileged_api": "EnvBatch.critic_state (never attached to EnvStep)",
}
ENCODING_SCHEMA: Mapping[str, object] = {
    "version": ENCODING_VERSION,
    "history": "one incremental event row per environment token step",
    "state": "public state plus observer hand/private prefix/private unresolved exchange selection",
    "privacy": "no other-seat hand identity and no unresolved other-seat exchange card",
    "channel_dtype": "uint8",
}

GAME_SPEC = GameSpec(
    identifier=GAME_IDENTIFIER, num_players=num_players, num_tokens=num_tokens,
    pad_token=pad_token, num_channels=num_channels,
    max_game_horizon=max_game_horizon,
    max_observation_tokens=max_observation_tokens,
    max_action_tokens=max_action_tokens,
    max_available_actions=max_available_actions,
    protocol_version=protocol_version,
)


class AdapterCapacityError(ProtocolError):
    """A V2 capacity was exceeded; the adapter never truncates silently."""


class TrajectoryHorizonError(AdapterCapacityError):
    """A nonterminal game reached the complete-trajectory horizon."""


@dataclass(frozen=True, slots=True)
class V2Action:
    token_id: int
    row: tuple[int, ...]
    phase_token: int
    prefix_stage: int
    kind: str
    is_commit: bool


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    tokens: np.ndarray
    channels: np.ndarray


@dataclass(frozen=True, slots=True)
class AvailableActionBatch:
    tokens: np.ndarray
    weights: np.ndarray
    is_commit: np.ndarray


@dataclass(frozen=True, slots=True)
class PrivilegedCriticBatch:
    """Explicit training-only ownership labels, never attached to EnvStep."""

    ownership: np.ndarray
    escrow: np.ndarray
    played: np.ndarray


@dataclass(frozen=True, slots=True)
class EnvStep:
    rewards: np.ndarray
    is_terminal: np.ndarray
    player_ids: np.ndarray
    observations: ObservationBatch
    available_actions: AvailableActionBatch
    infos: tuple[dict, ...]


def _strict_positive(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _phase_token(protocol) -> int:
    state = protocol.round_state
    if state.phase is Phase.TRIBUTE:
        return TRIBUTE_TOKEN
    if state.phase is Phase.RETURN:
        return RETURN_TOKEN
    if state.phase is Phase.PLAY:
        return LEAD_PLAY_TOKEN if state.current_winning is None else FOLLOW_PLAY_TOKEN
    raise EpisodeTerminatedError("terminal states have no V2 action phase")


def _token_kind(token_id: int) -> str:
    if token_id == PASS_TOKEN:
        return "PASS"
    if token_id == COMMIT_TOKEN:
        return "COMMIT"
    if FAMILY_TOKEN_BASE <= token_id < FAMILY_TOKEN_BASE + len(FAMILY_ORDER):
        return "FAMILY"
    if RANK_TOKEN_BASE <= token_id < RANK_TOKEN_BASE + len(RANK_ORDER):
        return "RANK"
    if SUIT_TOKEN_BASE <= token_id <= SUIT_TOKEN_BASE + int(Suit.JOKER):
        return "SUIT"
    if LENGTH_TOKEN_BASE <= token_id <= LENGTH_TOKEN_BASE + 10:
        return "LENGTH"
    if CARD_TOKEN_BASE <= token_id < CARD_TOKEN_BASE + 108:
        return "CARD"
    if WILD_TOKEN_BASE <= token_id < WILD_TOKEN_BASE + 52:
        return "WILD_ASSIGNMENT"
    raise ProtocolError(f"V1 exposed unsupported legal token {token_id}")


def _action_row(protocol, token_id: int) -> V2Action:
    token_id = int(token_id)
    phase = _phase_token(protocol)
    stage = len(protocol._prefix_tokens)
    if not 0 <= stage <= 32:
        raise AdapterCapacityError("V1 prefix exceeds V2 structured-row stage capacity")
    kind = _token_kind(token_id)
    is_commit = token_id in (PASS_TOKEN, COMMIT_TOKEN)
    row = (
        ACTION_ROW_BEGIN, phase, ACTION_STAGE_BASE + stage,
        ACTION_KIND_BASE + _KIND_TO_CODE[kind], ACTION_PAYLOAD, token_id,
        ACTION_COMMIT if is_commit else ACTION_NON_COMMIT, ACTION_ROW_END,
    )
    return V2Action(token_id, row, phase, stage, kind, is_commit)


def _parse_action_row(row: np.ndarray | Sequence[int]) -> V2Action:
    array = np.asarray(row)
    if array.shape != (max_action_tokens,) or array.dtype.kind not in "iu":
        raise ValueError(f"action row must be an integer vector with shape ({max_action_tokens},)")
    values = tuple(int(value) for value in array.tolist())
    if values[0] != ACTION_ROW_BEGIN or values[4] != ACTION_PAYLOAD or values[7] != ACTION_ROW_END:
        raise ProtocolError("action row framing is invalid")
    phase = values[1]
    if phase not in (LEAD_PLAY_TOKEN, FOLLOW_PLAY_TOKEN, TRIBUTE_TOKEN, RETURN_TOKEN):
        raise ProtocolError("action row phase is invalid")
    stage = values[2] - ACTION_STAGE_BASE
    kind_code = values[3] - ACTION_KIND_BASE
    if not 0 <= stage <= 32 or not 0 <= kind_code < len(_KIND_NAMES):
        raise ProtocolError("action row stage or kind is invalid")
    token_id = values[5]
    kind = _KIND_NAMES[kind_code]
    if _token_kind(token_id) != kind:
        raise ProtocolError("action row kind does not match its V1 token")
    if values[6] not in (ACTION_NON_COMMIT, ACTION_COMMIT):
        raise ProtocolError("action row commit flag is invalid")
    is_commit = token_id in (PASS_TOKEN, COMMIT_TOKEN)
    if (values[6] == ACTION_COMMIT) != is_commit:
        raise ProtocolError("action row commit flag does not match its V1 token")
    if any(not 0 <= value < num_tokens for value in values):
        raise ProtocolError("action row token is outside the V2 vocabulary")
    return V2Action(token_id, values, phase, stage, kind, is_commit)


def actions_from_np(rows: np.ndarray) -> tuple[V2Action, ...]:
    """Decode rows while retaining all metadata for current-state validation."""
    if not isinstance(rows, np.ndarray):
        raise TypeError("rows must be a NumPy array")
    if rows.dtype.kind not in "iu":
        raise ValueError("rows must use an integer dtype")
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != max_action_tokens:
        raise ValueError(f"rows must have shape [B,{max_action_tokens}]")
    return tuple(_parse_action_row(row) for row in rows)


def _binary_bits(value: int, width: int) -> list[int]:
    if value < 0 or value >= 1 << width:
        raise AdapterCapacityError(f"counter {value} exceeds {width}-bit channel capacity")
    return [(value >> bit) & 1 for bit in range(width)]


def _encode_channels(protocol, observer: int) -> np.ndarray:
    state = protocol.round_state
    channels = np.zeros(num_channels, dtype=np.uint8)
    visible_hand_ids = {card.card_id for card in state.hands[observer]}
    if state.phase is Phase.RETURN:
        # A02 transfers a returned card into the donor's physical hand
        # immediately, but A04 does not publish return identities until the
        # entire exchange resolves. Keep those incoming cards out of the
        # donor's actor information set until PLAY begins.
        visible_hand_ids.difference_update(
            card.card_id for _source, target, card in state.return_transfers
            if target == observer
        )
    for card_id in visible_hand_ids:
        channels[SELF_HAND_SLICE.start + card_id] = 1
    if state.current_winning is not None:
        for card in state.current_winning.cards:
            channels[CURRENT_WINNING_CARDS_SLICE.start + card.card_id] = 1
    if state.phase in (Phase.TRIBUTE, Phase.RETURN):
        own_cards = []
        selected = state.tribute_selections.get(observer)
        if selected is not None:
            own_cards.append(selected)
        own_cards.extend(card for source, _target, card in state.return_transfers if source == observer)
        for card in own_cards:
            channels[SELF_EXCHANGE_CARDS_SLICE.start + card.card_id] = 1
    prefix = tuple(protocol._prefix_tokens) if observer == state.active_seat else ()
    if len(prefix) > 32:
        raise AdapterCapacityError("private prefix exceeds V2 channel capacity")
    channels[SELF_PREFIX_TOKENS_SLICE.start: SELF_PREFIX_TOKENS_SLICE.start + len(prefix)] = prefix
    channels[REMAINING_COUNTS_SLICE] = [len(hand) for hand in state.hands]
    channels[FINISHING_RANKS_SLICE] = state.ranking
    channels[OBSERVER_ONE_HOT_SLICE.start + observer] = 1
    channels[ACTIVE_ONE_HOT_SLICE.start + state.active_seat] = 1
    channels[LEADER_ONE_HOT_SLICE.start + state.leader_seat] = 1
    if state.current_winner_seat is not None:
        channels[WINNER_ONE_HOT_SLICE.start + state.current_winner_seat] = 1
    for seat in state.emptied_seats:
        channels[EMPTIED_ONE_HOT_SLICE.start + seat] = 1
    channels[PHASE_ONE_HOT_SLICE.start + _PHASE_INDEX[state.phase]] = 1
    channels[LEVEL_ONE_HOT_SLICE.start + PRINTED_RANKS.index(state.level_rank)] = 1
    channels[PASS_COUNT_ONE_HOT_SLICE.start + min(state.consecutive_passes, 3)] = 1
    channels[TEAM_ONE_HOT_SLICE.start + TEAM_OF[observer]] = 1
    previous = state.previous_result
    if previous is not None:
        channels[PREVIOUS_WINNER_TEAM_SLICE.start + previous.winner_team] = 1
        channels[PREVIOUS_OUTCOME_SLICE.start + _OUTCOME_INDEX[previous.outcome_class]] = 1
        channels[PREVIOUS_RANKING_SLICE] = previous.ranking
        channels[486] = int(previous.tribute_cancelled)
    if state.current_winning is not None:
        channels[CURRENT_FAMILY_SLICE.start + FAMILY_ORDER.index(state.current_winning.kind)] = 1
        channels[CURRENT_RANK_SLICE.start + RANK_ORDER.index(state.current_winning.comparison_rank)] = 1
        size = len(state.current_winning.cards)
        if not 0 <= size <= 10:
            raise AdapterCapacityError("current combination exceeds size channel capacity")
        channels[CURRENT_SIZE_SLICE.start + size] = 1
    for seat in state.tribute_donors:
        channels[TRIBUTE_DONOR_SLICE.start + seat] = 1
    if state.phase in (Phase.PLAY, Phase.TERMINAL):
        for index, seat in enumerate(state.tribute_order[:4]):
            channels[TRIBUTE_ORDER_SLICE.start + index] = seat + 1
    if observer == state.active_seat:
        channels[TOKEN_STEP_BITS_SLICE] = _binary_bits(protocol.token_step, 13)
    channels[COMMITTED_STEP_BITS_SLICE] = _binary_bits(protocol.committed_step, 13)
    channels[487] = int(state.phase in (Phase.PLAY, Phase.TERMINAL))
    channels[488] = int(state.anti_tribute)
    channels[489] = int(state.current_winning is not None)
    channels[490] = len(prefix)
    channels[491] = int(observer == state.active_seat)
    channels[492] = int(observer in state.finished_ranks)
    channels[493] = int(observer in state.emptied_seats)
    channels[494] = int(bool(state.return_obligations))
    channels[495] = int(observer in state.tribute_selections)
    channels[496] = int(any(source == observer for source, _target, _card in state.return_transfers))
    channels[497] = int(state.previous_result is not None)
    channels[498] = int(state.current_winner_seat == observer)
    channels[499] = int(state.leader_seat == observer)
    channels[500] = int(TEAM_OF[state.active_seat] == TEAM_OF[observer])
    channels[511] = 1
    return channels


def _pad_event(tokens: Sequence[int], capacity: int) -> np.ndarray:
    if len(tokens) > capacity:
        raise AdapterCapacityError(f"event requires {len(tokens)} tokens, capacity is {capacity}")
    if any(not 0 < int(token) < num_tokens for token in tokens):
        raise ProtocolError("event contains PAD or an out-of-vocabulary token")
    result = np.zeros(capacity, dtype=np.int64)
    result[: len(tokens)] = tokens
    return result


def _reset_events(protocol) -> tuple[tuple[int, ...], ...]:
    event = (
        BOS_TOKEN, EVENT_RESET, _phase_token(protocol),
        TokenCodec.rank(protocol.round_state.level_rank),
        TokenCodec.length(protocol.round_state.active_seat), EVENT_END,
    )
    return (event,) * num_players


def _step_events(before_protocol, result, token_id: int, after_protocol) -> tuple[tuple[int, ...], ...]:
    actor = before_protocol.round_state.active_seat
    phase = _phase_token(before_protocol)
    seat_token = TokenCodec.length(actor)
    if not result.is_commit:
        private = (BOS_TOKEN, EVENT_PRIVATE_PREFIX, phase, seat_token, token_id, EVENT_END)
        hidden: tuple[int, ...] = ()
        return tuple(private if observer == actor else hidden for observer in range(num_players))
    sequence = (PASS_TOKEN,) if token_id == PASS_TOKEN else tuple(before_protocol._prefix_tokens) + (token_id,)
    before_state, after_state = before_protocol.round_state, after_protocol.round_state
    if before_state.phase in (Phase.TRIBUTE, Phase.RETURN) and after_state.phase in (Phase.TRIBUTE, Phase.RETURN):
        private = (BOS_TOKEN, EVENT_PRIVATE_EXCHANGE_COMMIT, phase, seat_token, *sequence, EVENT_END)
        hidden: tuple[int, ...] = ()
        return tuple(private if observer == actor else hidden for observer in range(num_players))
    if before_state.phase in (Phase.TRIBUTE, Phase.RETURN) and after_state.phase is Phase.PLAY:
        records: list[int] = []
        for source, target, card in after_state.tribute_transfers:
            records.extend((TRIBUTE_TOKEN, TokenCodec.length(source), TokenCodec.length(target), TokenCodec.card(card.card_id), COMMIT_TOKEN))
        for source, target, card in after_state.return_transfers:
            records.extend((RETURN_TOKEN, TokenCodec.length(source), TokenCodec.length(target), TokenCodec.card(card.card_id), COMMIT_TOKEN))
        event = (BOS_TOKEN, EVENT_EXCHANGE_RESOLVED, phase, seat_token, *records, EVENT_END)
        return (event,) * num_players
    event = (BOS_TOKEN, EVENT_PUBLIC_COMMIT, phase, seat_token, *sequence, EVENT_END)
    return (event,) * num_players


class EnvBatch:
    """Transactional batch with the explicit python_reference backend."""

    def __init__(
        self, batch_size: int, *, backend: str = BACKEND,
        configs: Sequence[GameConfig] | None = None, auto_reset: bool = True,
        max_horizon: int = max_game_horizon,
        action_capacity: int = max_available_actions,
        observation_capacity: int = max_observation_tokens,
    ) -> None:
        self.batch_size = _strict_positive(batch_size, "batch_size")
        if backend != BACKEND:
            raise ValueError(f"unsupported backend {backend!r}; no fallback is available")
        if not isinstance(auto_reset, bool):
            raise TypeError("auto_reset must be bool")
        self.backend, self.auto_reset = backend, auto_reset
        self.max_horizon = _strict_positive(max_horizon, "max_horizon")
        self.action_capacity = _strict_positive(action_capacity, "action_capacity")
        self.observation_capacity = _strict_positive(observation_capacity, "observation_capacity")
        if self.max_horizon > max_game_horizon:
            raise ValueError(f"max_horizon cannot exceed {max_game_horizon}")
        if self.action_capacity > max_available_actions or self.observation_capacity > max_observation_tokens:
            raise ValueError("configured capacity exceeds the V2 schema")
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
        self._events = tuple(((),) * num_players for _ in range(self.batch_size))
        self._trajectory_steps = np.zeros(self.batch_size, dtype=np.int64)
        self._is_reset = np.zeros(self.batch_size, dtype=np.bool_)

    @property
    def envs(self) -> tuple[GuandanEnv, ...]:
        return self._envs

    @classmethod
    def from_envs(cls, envs: Sequence[GuandanEnv], *, auto_reset: bool = False, **kwargs) -> "EnvBatch":
        envs = tuple(envs)
        if not envs or any(not isinstance(env, GuandanEnv) for env in envs):
            raise TypeError("envs must contain GuandanEnv instances")
        result = cls(len(envs), configs=[env.config for env in envs], auto_reset=auto_reset, **kwargs)
        result._envs = tuple(env.clone() for env in envs)
        result._events = tuple(
            _reset_events(env._require_protocol()) if not env.done and not env.truncated else ((),) * num_players
            for env in result._envs
        )
        result._trajectory_steps = np.asarray([env.token_step for env in result._envs], dtype=np.int64)
        result._is_reset.fill(True)
        return result

    def _candidate_actions(self, env: GuandanEnv) -> tuple[V2Action, ...]:
        protocol = env._require_protocol()
        legal = tuple(int(token) for token in protocol.legal_next_tokens().tolist())
        if len(legal) > self.action_capacity:
            raise AdapterCapacityError(f"legal action count {len(legal)} exceeds capacity {self.action_capacity}")
        actions = tuple(_action_row(protocol, token) for token in legal)
        if len({action.row for action in actions}) != len(actions):
            raise ProtocolError("V2 action rows are not unique")
        return actions

    def _build_output(self, envs, events, *, rewards=None, terminal=None, infos=None) -> EnvStep:
        size = len(envs)
        obs_tokens = np.zeros((size, num_players, self.observation_capacity), dtype=np.int64)
        obs_channels = np.zeros((size, num_players, num_channels), dtype=np.uint8)
        action_tokens = np.zeros((size, self.action_capacity, max_action_tokens), dtype=np.int64)
        action_weights = np.zeros((size, self.action_capacity), dtype=np.float32)
        action_commit = np.zeros((size, self.action_capacity), dtype=np.bool_)
        player_ids = np.full(size, -1, dtype=np.int64)
        for batch_index, (env, seat_events) in enumerate(zip(envs, events, strict=True)):
            if env.done or env.truncated:
                continue
            protocol = env._require_protocol()
            player_ids[batch_index] = protocol.round_state.active_seat
            for observer in range(num_players):
                obs_tokens[batch_index, observer] = _pad_event(seat_events[observer], self.observation_capacity)
                obs_channels[batch_index, observer] = _encode_channels(protocol, observer)
            for action_index, action in enumerate(self._candidate_actions(env)):
                action_tokens[batch_index, action_index] = action.row
                action_weights[batch_index, action_index] = 1.0
                action_commit[batch_index, action_index] = action.is_commit
        rewards_array = np.zeros((size, num_players), np.float32) if rewards is None else np.asarray(rewards, np.float32)
        terminal_array = np.zeros(size, np.bool_) if terminal is None else np.asarray(terminal, np.bool_)
        if rewards_array.shape != (size, num_players) or terminal_array.shape != (size,):
            raise ValueError("reward or terminal shape mismatch")
        info_tuple = tuple({} for _ in range(size)) if infos is None else tuple(deepcopy(info) for info in infos)
        return EnvStep(
            rewards_array.copy(), terminal_array.copy(), player_ids,
            ObservationBatch(obs_tokens, obs_channels),
            AvailableActionBatch(action_tokens, action_weights, action_commit), info_tuple,
        )

    def observe(self) -> EnvStep:
        """Return the current four-seat information sets without mutation."""
        if not bool(np.all(self._is_reset)):
            raise RuntimeError("every slot must be reset before observe")
        return self._build_output(self._envs, self._events)

    def critic_state(self) -> PrivilegedCriticBatch:
        """Return explicit training-only full ownership labels.

        This API is deliberately separate from reset/step/observe so actor
        tensors and shared caches cannot receive privileged ownership by
        accident.
        """
        if not bool(np.all(self._is_reset)):
            raise RuntimeError("every slot must be reset before critic_state")
        ownership = np.zeros((self.batch_size, num_players, 108), dtype=np.uint8)
        escrow = np.zeros((self.batch_size, 108), dtype=np.uint8)
        played = np.zeros((self.batch_size, 108), dtype=np.uint8)
        for batch_index, env in enumerate(self._envs):
            state = env._require_protocol().round_state
            for seat, hand in enumerate(state.hands):
                for card in hand:
                    ownership[batch_index, seat, card.card_id] = 1
            for card in state.tribute_escrow:
                escrow[batch_index, card.card_id] = 1
            for action in state.history:
                if action.kind not in ("pass", "tribute", "return"):
                    for card in action.cards:
                        played[batch_index, card.card_id] = 1
        if np.any(ownership.sum(axis=1) + escrow + played != 1):
            raise ProtocolError("privileged ownership labels do not conserve 108 physical cards")
        return PrivilegedCriticBatch(ownership, escrow, played)

    def reset(self, seed: int = 0) -> EnvStep:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        trials = tuple(env.clone() for env in self._envs)
        trial_events = []
        for index, env in enumerate(trials):
            env.reset(seed=int(seed) + index)
            trial_events.append(_reset_events(env._require_protocol()))
        output = self._build_output(trials, trial_events)
        self._envs, self._events = trials, tuple(trial_events)
        self._trajectory_steps.fill(0)
        self._is_reset.fill(True)
        return output

    def step(self, actions: Sequence[V2Action] | np.ndarray) -> EnvStep:
        actions = actions_from_np(actions) if isinstance(actions, np.ndarray) else tuple(actions)
        if len(actions) != self.batch_size or any(not isinstance(action, V2Action) for action in actions):
            raise ValueError("actions must contain one V2Action per slot")
        if not bool(np.all(self._is_reset)):
            raise RuntimeError("every slot must be reset before step")
        trials = tuple(env.clone() for env in self._envs)
        trial_events = list(self._events)
        trial_steps = self._trajectory_steps.copy()
        rewards = np.zeros((self.batch_size, num_players), np.float32)
        terminal = np.zeros(self.batch_size, np.bool_)
        infos: list[dict] = []
        for index, (env, action) in enumerate(zip(trials, actions, strict=True)):
            if env.done or env.truncated:
                raise EpisodeTerminatedError("inactive V2 slot cannot be stepped")
            if trial_steps[index] >= self.max_horizon:
                raise TrajectoryHorizonError("trajectory horizon exceeded before terminal")
            matches = [candidate for candidate in self._candidate_actions(env) if candidate.row == action.row]
            if len(matches) != 1 or matches[0] != action:
                raise ProtocolError("action row is stale, forged, or not uniquely executable")
            before = env._require_protocol().clone()
            result = env.step(action.token_id)
            if result.truncated:
                raise TrajectoryHorizonError("V1 token horizon reached before terminal")
            after = env._require_protocol()
            trial_steps[index] += 1
            rewards[index], terminal[index] = result.rewards, result.done
            info = deepcopy(result.info)
            info.update(backend=BACKEND, action_version=ACTION_VERSION,
                        environment_version=ENV_VERSION, encoding_version=ENCODING_VERSION)
            if result.done:
                info["terminal_serialized_v1"] = env.serialize().decode("utf-8")
                if self.auto_reset:
                    env.reset()
                    trial_events[index] = _reset_events(env._require_protocol())
                    trial_steps[index] = 0
                else:
                    trial_events[index] = ((),) * num_players
            else:
                trial_events[index] = _step_events(before, result, action.token_id, after)
            infos.append(info)
        output = self._build_output(trials, trial_events, rewards=rewards, terminal=terminal, infos=infos)
        self._envs, self._events, self._trajectory_steps = trials, tuple(trial_events), trial_steps
        return output

    def serialize(self) -> bytes:
        if not bool(np.all(self._is_reset)):
            raise RuntimeError("cannot serialize before reset")
        payload = {
            "schema": SERIALIZATION_SCHEMA, "backend": self.backend,
            "action_version": ACTION_VERSION, "environment_version": ENV_VERSION,
            "encoding_version": ENCODING_VERSION, "auto_reset": self.auto_reset,
            "max_horizon": self.max_horizon, "action_capacity": self.action_capacity,
            "observation_capacity": self.observation_capacity,
            "trajectory_steps": self._trajectory_steps.tolist(),
            "events": [[[int(token) for token in event] for event in seat_events] for seat_events in self._events],
            "envs": [json.loads(env.serialize()) for env in self._envs],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def deserialize(cls, payload: bytes) -> "EnvBatch":
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        data = json.loads(payload)
        expected = {"schema": SERIALIZATION_SCHEMA, "backend": BACKEND,
                    "action_version": ACTION_VERSION, "environment_version": ENV_VERSION,
                    "encoding_version": ENCODING_VERSION}
        if any(data.get(key) != value for key, value in expected.items()):
            raise ProtocolError("unsupported V2 serialization or protocol version")
        raw_envs = data.get("envs")
        if not isinstance(raw_envs, list) or not raw_envs:
            raise ProtocolError("serialized batch must contain environments")
        envs = tuple(GuandanEnv.deserialize(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ) for raw in raw_envs)
        result = cls(
            len(envs), configs=[env.config for env in envs],
            auto_reset=bool(data["auto_reset"]), max_horizon=int(data["max_horizon"]),
            action_capacity=int(data["action_capacity"]),
            observation_capacity=int(data["observation_capacity"]),
        )
        result._envs = envs
        events = tuple(tuple(tuple(int(token) for token in event) for event in seat_events)
                       for seat_events in data["events"])
        if len(events) != len(envs) or any(len(seat_events) != num_players for seat_events in events):
            raise ProtocolError("serialized event shape is invalid")
        steps = np.asarray(data["trajectory_steps"], dtype=np.int64)
        if steps.shape != (len(envs),) or bool(np.any(steps < 0)):
            raise ProtocolError("serialized trajectory counters are invalid")
        result._events, result._trajectory_steps = events, steps
        result._is_reset.fill(True)
        result._build_output(result._envs, result._events)
        return result


__all__ = [
    "ACTION_SCHEMA", "ACTION_VERSION", "AdapterCapacityError", "AvailableActionBatch", "BACKEND",
    "ENCODING_SCHEMA", "ENCODING_VERSION", "ENV_SCHEMA", "ENV_VERSION", "EnvBatch", "EnvStep",
    "GAME_IDENTIFIER", "GAME_SPEC", "ObservationBatch", "PrivilegedCriticBatch", "SERIALIZATION_SCHEMA", "TrajectoryHorizonError",
    "V2Action", "actions_from_np", "identifier", "max_action_tokens", "max_available_actions",
    "max_game_horizon", "max_observation_tokens", "num_channels", "num_players", "num_tokens",
    "pad_token", "protocol_version",
]
