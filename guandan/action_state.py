"""Scheme-B stepwise action protocol for A03.

The protocol is intentionally a thin layer over the accepted A02 atomic rules
engine.  It enumerates canonical token paths from the complete A02 action set;
public round state is changed only when COMMIT (or PASS) is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Iterable

import numpy as np

from .cards import Card, PRINTED_RANKS, Rank, Suit, WINDOW_RANKS
from .combos import Combination, CombinationKind
from .round import enumerate_legal_actions, apply_action
from .state import CommittedAction as AtomicCommittedAction, EpisodeTerminatedError, GuandanError, IllegalActionError, Phase, RoundState


class ProtocolError(GuandanError):
    """A protocol configuration or capacity error, never a legal-action error."""


PAD_TOKEN = 0
BOS_TOKEN = 1
LEAD_PLAY_TOKEN = 2
FOLLOW_PLAY_TOKEN = 3
TRIBUTE_TOKEN = 4
RETURN_TOKEN = 5
PASS_TOKEN = 6
COMMIT_TOKEN = 7
FAMILY_TOKEN_BASE = 8
RANK_TOKEN_BASE = 18
SUIT_TOKEN_BASE = 33
LENGTH_TOKEN_BASE = 38
CARD_TOKEN_BASE = 64
WILD_TOKEN_BASE = 172
TOKEN_VOCAB_SIZE = 256
MAX_ACTION_TOKENS = 32
MAX_LEGAL_NEXT_TOKENS = 256

FAMILY_ORDER = (
    CombinationKind.SINGLE,
    CombinationKind.PAIR,
    CombinationKind.TRIPLE,
    CombinationKind.FULL_HOUSE,
    CombinationKind.STRAIGHT,
    CombinationKind.PAIR_SEQUENCE,
    CombinationKind.TRIPLE_SEQUENCE,
    CombinationKind.RANK_BOMB,
    CombinationKind.STRAIGHT_FLUSH,
    CombinationKind.FOUR_KINGS,
)
RANK_ORDER = (*PRINTED_RANKS, Rank.SMALL_JOKER, Rank.BIG_JOKER)


class TokenCodec:
    """Deterministic v1 token vocabulary codec."""

    @staticmethod
    def family(kind: CombinationKind | str) -> int:
        return FAMILY_TOKEN_BASE + FAMILY_ORDER.index(CombinationKind(kind))

    @staticmethod
    def family_from_token(token: int) -> CombinationKind:
        index = int(token) - FAMILY_TOKEN_BASE
        if not 0 <= index < len(FAMILY_ORDER):
            raise ValueError("token is not a family token")
        return FAMILY_ORDER[index]

    @staticmethod
    def rank(rank: Rank | int) -> int:
        return RANK_TOKEN_BASE + RANK_ORDER.index(Rank(rank))

    @staticmethod
    def rank_from_token(token: int) -> Rank:
        index = int(token) - RANK_TOKEN_BASE
        if not 0 <= index < len(RANK_ORDER):
            raise ValueError("token is not a rank token")
        return RANK_ORDER[index]

    @staticmethod
    def suit(suit: Suit | int) -> int:
        return SUIT_TOKEN_BASE + int(Suit(suit))

    @staticmethod
    def suit_from_token(token: int) -> Suit:
        index = int(token) - SUIT_TOKEN_BASE
        if not 0 <= index <= int(Suit.JOKER):
            raise ValueError("token is not a suit token")
        return Suit(index)

    @staticmethod
    def length(length: int) -> int:
        if not 0 <= int(length) <= 10:
            raise ValueError("length token supports values 0..10")
        return LENGTH_TOKEN_BASE + int(length)

    @staticmethod
    def length_from_token(token: int) -> int:
        value = int(token) - LENGTH_TOKEN_BASE
        if not 0 <= value <= 10:
            raise ValueError("token is not a length token")
        return value

    @staticmethod
    def card(card_id: int) -> int:
        if not 0 <= int(card_id) < 108:
            raise ValueError("card id must be in 0..107")
        return CARD_TOKEN_BASE + int(card_id)

    @staticmethod
    def card_from_token(token: int) -> int:
        card_id = int(token) - CARD_TOKEN_BASE
        if not 0 <= card_id < 108:
            raise ValueError("token is not a card token")
        return card_id

    @staticmethod
    def wild(rank: Rank | int, suit: Suit | int) -> int:
        rank = Rank(rank)
        suit = Suit(suit)
        if rank in {Rank.SMALL_JOKER, Rank.BIG_JOKER} or suit is Suit.JOKER:
            raise ValueError("wild assignment must target a non-joker rank and suit")
        return WILD_TOKEN_BASE + RANK_ORDER.index(rank) * 4 + int(suit)

    @staticmethod
    def wild_from_token(token: int) -> tuple[Rank, Suit]:
        index = int(token) - WILD_TOKEN_BASE
        if not 0 <= index < 13 * 4:
            raise ValueError("token is not a wild-assignment token")
        return RANK_ORDER[index // 4], Suit(index % 4)


@dataclass(frozen=True, slots=True)
class ActionPrefix:
    phase: str
    family: str | None
    declared_ranks: tuple[int, ...]
    declared_length: int | None
    declared_suit: int | None
    selected_card_ids: tuple[int, ...]
    wild_assignments: tuple[tuple[int, int, int | None], ...]
    next_expected_kind: str
    complete: bool
    tokens: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ProtocolCommittedAction:
    actor: int
    phase: str
    kind: str
    family: str | None
    card_ids: tuple[int, ...]
    declared_ranks: tuple[int, ...]
    declared_suit: int | None
    wild_assignments: tuple[tuple[int, int, int | None], ...]
    public_index: int
    atomic_action: AtomicCommittedAction = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class LegalTokenRow:
    token_id: int
    kind: str
    value: int
    is_commit: bool


@dataclass(frozen=True, slots=True)
class ProtocolStepResult:
    observation: ActionPrefix
    rewards: np.ndarray
    done: bool
    truncated: bool
    is_commit: bool
    committed_action: ProtocolCommittedAction | None
    token_step: int
    committed_step: int
    info: dict


StepResult = ProtocolStepResult
CommittedAction = ProtocolCommittedAction


def _phase_name(state: RoundState) -> str:
    if state.phase is Phase.TRIBUTE:
        return "TRIBUTE"
    if state.phase is Phase.RETURN:
        return "RETURN"
    if state.current_winning is None:
        return "LEAD_PLAY"
    return "FOLLOW_PLAY"


def _phase_expected(state: RoundState) -> str:
    return _phase_name(state)


def _sequence_windows(length: int) -> tuple[tuple[Rank, ...], ...]:
    windows = tuple(tuple(WINDOW_RANKS[i : i + length]) for i in range(len(WINDOW_RANKS) - length + 1))
    extras = {5: (Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE), 3: (Rank.ACE, Rank.TWO, Rank.THREE), 2: (Rank.ACE, Rank.TWO)}
    return windows + (extras[length],)


def _represented_rank(card_id: int, declaration: Combination) -> Rank:
    for physical_id, (rank, _suit) in declaration.wild_assignments.items():
        if physical_id == card_id:
            return rank
    return next(card.rank for card in declaration.cards if card.card_id == card_id)


def _window_for(declaration: Combination) -> tuple[Rank, ...]:
    if declaration.kind is CombinationKind.STRAIGHT:
        length = 5
    elif declaration.kind is CombinationKind.PAIR_SEQUENCE:
        length = 3
    elif declaration.kind is CombinationKind.TRIPLE_SEQUENCE:
        length = 2
    elif declaration.kind is CombinationKind.STRAIGHT_FLUSH:
        length = 5
    else:
        raise ValueError("declaration is not a window family")
    candidates = [window for window in _sequence_windows(length) if window[-1] is declaration.comparison_rank]
    represented = {_represented_rank(card.card_id, declaration) for card in declaration.cards}
    for window in candidates:
        if represented.issubset(set(window)):
            return window
    if candidates:
        return candidates[0]
    raise ValueError("cannot derive canonical sequence window")


def _declaration_fields(declaration: Combination) -> tuple[list[int], int | None, int | None]:
    """Return declaration rank tokens, length value, and suit value."""
    kind = declaration.kind
    if kind in {CombinationKind.PAIR, CombinationKind.TRIPLE, CombinationKind.RANK_BOMB}:
        return [TokenCodec.rank(declaration.comparison_rank)], len(declaration.cards) if kind is CombinationKind.RANK_BOMB else None, None
    if kind is CombinationKind.FULL_HOUSE:
        counts: dict[Rank, int] = {}
        for card in declaration.cards:
            rank = _represented_rank(card.card_id, declaration)
            counts[rank] = counts.get(rank, 0) + 1
        pair_rank = next((rank for rank, count in counts.items() if count == 2), declaration.comparison_rank)
        return [TokenCodec.rank(declaration.comparison_rank), TokenCodec.rank(pair_rank)], None, None
    if kind in {CombinationKind.STRAIGHT, CombinationKind.PAIR_SEQUENCE, CombinationKind.TRIPLE_SEQUENCE}:
        window = _window_for(declaration)
        length = None if kind is CombinationKind.STRAIGHT else len(window)
        return [TokenCodec.rank(window[0])], length, None
    if kind is CombinationKind.STRAIGHT_FLUSH:
        window = _window_for(declaration)
        natural_suits = [card.suit for card in declaration.cards if not card.is_joker and not card.is_level_wild(declaration.level_rank)]
        suit = natural_suits[0] if natural_suits else next(iter(declaration.wild_assignments.values()))[1]
        return [TokenCodec.rank(window[0])], None, int(suit)
    return [], None, None


def _declared_rank_for_card(card_id: int, declaration: Combination) -> Rank:
    return _represented_rank(card_id, declaration)


def _ordered_action_cards(action: AtomicCommittedAction) -> tuple:
    """Canonical group ordering required by GD-ACTION-0.1."""
    cards = tuple(action.cards)
    declaration = action.declaration
    if declaration is None or not isinstance(action.kind, CombinationKind):
        return tuple(sorted(cards, key=lambda card: card.card_id))
    if declaration.kind is CombinationKind.FULL_HOUSE:
        triple = declaration.comparison_rank
        groups = {triple: [], "pair": []}
        counts: dict[Rank, int] = {}
        for card in cards:
            rank = _declared_rank_for_card(card.card_id, declaration)
            counts[rank] = counts.get(rank, 0) + 1
        pair_rank = next((rank for rank, count in counts.items() if rank is not triple and count == 2), None)
        if pair_rank is None:
            return tuple(sorted(cards, key=lambda card: card.card_id))
        ordered = []
        for target in (triple, pair_rank):
            ordered.extend(sorted((card for card in cards if _declared_rank_for_card(card.card_id, declaration) is target), key=lambda card: card.card_id))
        return tuple(ordered)
    if declaration.kind in {CombinationKind.STRAIGHT, CombinationKind.STRAIGHT_FLUSH, CombinationKind.PAIR_SEQUENCE, CombinationKind.TRIPLE_SEQUENCE}:
        window = _window_for(declaration)
        rank_order = {rank: index for index, rank in enumerate(window)}
        return tuple(sorted(cards, key=lambda card: (rank_order.get(_declared_rank_for_card(card.card_id, declaration), 999), card.card_id)))
    return tuple(sorted(cards, key=lambda card: card.card_id))


def canonical_tokens(action: AtomicCommittedAction) -> tuple[int, ...]:
    """Encode one A02 committed action as the unique v1 token sequence."""
    if action.kind == "pass":
        return (PASS_TOKEN,)
    if action.kind in {"tribute", "return"}:
        if len(action.cards) != 1:
            raise ProtocolError("tribute/return action must contain one card")
        return (TokenCodec.card(action.cards[0].card_id), COMMIT_TOKEN)
    if not isinstance(action.kind, CombinationKind) or action.declaration is None:
        raise ProtocolError("play action requires a declaration")
    declaration = action.declaration
    tokens: list[int] = [TokenCodec.family(declaration.kind)]
    ranks, length, suit = _declaration_fields(declaration)
    if declaration.kind is CombinationKind.STRAIGHT_FLUSH:
        tokens.append(TokenCodec.suit(suit))
        tokens.extend(ranks)
    else:
        tokens.extend(ranks)
    if length is not None:
        tokens.append(TokenCodec.length(length))
    tokens.extend(TokenCodec.card(card.card_id) for card in _ordered_action_cards(action))
    for card_id, (rank, represented_suit) in sorted(declaration.wild_assignments.items()):
        tokens.append(TokenCodec.wild(rank, represented_suit))
    tokens.append(COMMIT_TOKEN)
    if len(tokens) > MAX_ACTION_TOKENS:
        raise ProtocolError("canonical action exceeds max_action_tokens")
    return tuple(tokens)


def _atomic_action_key(action: AtomicCommittedAction) -> tuple:
    declaration = action.declaration
    declaration_key = None if declaration is None else (declaration.kind.value, declaration.physical_ids(), declaration.comparison_rank, declaration.comparison_key, tuple(sorted((card_id, int(rank), int(suit)) for card_id, (rank, suit) in declaration.wild_assignments.items())), declaration.natural_wild_ids)
    return (action.player_id, str(action.kind), tuple(card.card_id for card in action.cards), declaration_key)



class StepwiseActionState:
    """Private-prefix A03 state wrapped around an A02 RoundState."""

    def __init__(self, round_state: RoundState, *, max_action_tokens: int = MAX_ACTION_TOKENS, max_legal_next_tokens: int = MAX_LEGAL_NEXT_TOKENS, max_token_steps: int = 4096) -> None:
        self.round_state = round_state
        self.max_action_tokens = int(max_action_tokens)
        self.max_legal_next_tokens = int(max_legal_next_tokens)
        self.max_token_steps = int(max_token_steps)
        if self.max_action_tokens <= 0 or self.max_legal_next_tokens <= 0 or self.max_token_steps <= 0:
            raise ValueError("protocol capacities must be positive")
        self.token_step = 0
        self.truncated = False
        self._prefix_tokens: tuple[int, ...] = ()

    @classmethod
    def deal(cls, *, seed: int = 0, level_rank: Rank = Rank.TWO, leader_seat: int = 0, previous_result=None, max_token_steps: int = 4096) -> "StepwiseActionState":
        return cls(RoundState.deal(seed=seed, level_rank=level_rank, leader_seat=leader_seat, previous_result=previous_result), max_token_steps=max_token_steps)

    @property
    def done(self) -> bool:
        return self.round_state.done

    @property
    def committed_step(self) -> int:
        return self.round_state.committed_step

    @property
    def prefix(self) -> ActionPrefix:
        sequences = self._candidate_sequences()
        next_kind = "COMMIT" if self._prefix_tokens and any(sequence[: len(self._prefix_tokens)] == self._prefix_tokens and len(sequence) == len(self._prefix_tokens) + 1 and sequence[-1] == COMMIT_TOKEN for sequence, _ in sequences) else "PREFIX"
        family = None
        ranks: list[int] = []
        length = None
        suit = None
        selected: list[int] = []
        wilds: list[tuple[int, int, int | None]] = []
        for token in self._prefix_tokens:
            if FAMILY_TOKEN_BASE <= token < FAMILY_TOKEN_BASE + len(FAMILY_ORDER):
                family = TokenCodec.family_from_token(token).value
            elif RANK_TOKEN_BASE <= token < RANK_TOKEN_BASE + len(RANK_ORDER):
                ranks.append(int(TokenCodec.rank_from_token(token)))
            elif LENGTH_TOKEN_BASE <= token <= LENGTH_TOKEN_BASE + 10:
                length = TokenCodec.length_from_token(token)
            elif SUIT_TOKEN_BASE <= token <= SUIT_TOKEN_BASE + 4:
                suit = int(TokenCodec.suit_from_token(token))
            elif CARD_TOKEN_BASE <= token < CARD_TOKEN_BASE + 108:
                selected.append(TokenCodec.card_from_token(token))
            elif WILD_TOKEN_BASE <= token < WILD_TOKEN_BASE + 52:
                rank, target_suit = TokenCodec.wild_from_token(token)
                wilds.append((-1, int(rank), int(target_suit)))
        selected_wilds = sorted(card_id for card_id in selected if Card(card_id).is_level_wild(self.round_state.level_rank))
        wilds = [(selected_wilds[index] if index < len(selected_wilds) else -1, rank, target_suit) for index, (_unused, rank, target_suit) in enumerate(wilds)]
        complete = any(sequence[:-1] == self._prefix_tokens and sequence[-1] == COMMIT_TOKEN for sequence, _ in sequences)
        return ActionPrefix(_phase_name(self.round_state), family, tuple(ranks), length, suit, tuple(selected), tuple(wilds), next_kind, complete, self._prefix_tokens)

    def _candidate_sequences(self) -> list[tuple[tuple[int, ...], CommittedAction]]:
        if self.round_state.done or self.round_state.phase is Phase.TERMINAL:
            return []
        result: dict[tuple[int, ...], CommittedAction] = {}
        for action in enumerate_legal_actions(self.round_state):
            sequence = canonical_tokens(action)
            if len(sequence) > self.max_action_tokens:
                raise ProtocolError("canonical action exceeds configured max_action_tokens")
            previous = result.get(sequence)
            if previous is not None and _atomic_action_key(previous) != _atomic_action_key(action):
                raise ProtocolError("two distinct committed actions share one canonical token sequence")
            result[sequence] = action
        return [(sequence, result[sequence]) for sequence in sorted(result)]

    def legal_next_tokens(self) -> np.ndarray:
        if self.done or self.truncated:
            return np.zeros((0,), dtype=np.int32)
        next_tokens = sorted({sequence[len(self._prefix_tokens)] for sequence, _ in self._candidate_sequences() if sequence[: len(self._prefix_tokens)] == self._prefix_tokens and len(sequence) > len(self._prefix_tokens)})
        if len(next_tokens) > self.max_legal_next_tokens:
            raise ProtocolError("max_legal_next_tokens capacity overflow")
        return np.asarray(next_tokens, dtype=np.int32)

    def legal_tokens(self) -> np.ndarray:
        return self.legal_next_tokens()

    def legal_token_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows = self.legal_token_rows()
        tokens = np.zeros(self.max_legal_next_tokens, dtype=np.int32)
        mask = np.zeros(self.max_legal_next_tokens, dtype=np.bool_)
        is_commit = np.zeros(self.max_legal_next_tokens, dtype=np.bool_)
        kinds = np.zeros(self.max_legal_next_tokens, dtype=np.int32)
        values = np.zeros(self.max_legal_next_tokens, dtype=np.int32)
        for index, row in enumerate(rows):
            tokens[index] = row.token_id
            mask[index] = True
            is_commit[index] = row.is_commit
            kinds[index] = {"PASS": 1, "COMMIT": 2, "FAMILY": 3, "RANK": 4, "SUIT": 5, "LENGTH": 6, "CARD": 7, "WILD_ASSIGNMENT": 8}.get(row.kind, 0)
            values[index] = row.value if isinstance(row.value, int) else 0
        return tokens, mask, is_commit, kinds, values

    def _public_action(self, action: AtomicCommittedAction, phase: str | None = None) -> ProtocolCommittedAction:
        declaration = action.declaration
        family = declaration.kind.value.upper() if declaration is not None else None
        declared_ranks: tuple[int, ...] = ()
        declared_suit = None
        wild_assignments: tuple[tuple[int, int, int | None], ...] = ()
        if declaration is not None:
            ranks, _length, suit = _declaration_fields(declaration)
            declared_ranks = tuple(TokenCodec.rank_from_token(token).value for token in ranks)
            declared_suit = suit
            wild_assignments = tuple((int(card_id), int(rank), int(suit)) for card_id, (rank, suit) in sorted(declaration.wild_assignments.items()))
        return ProtocolCommittedAction(action.player_id, phase or _phase_name(self.round_state), str(action.kind.value if isinstance(action.kind, CombinationKind) else action.kind).upper(), family, tuple(card.card_id for card in action.cards), declared_ranks, declared_suit, wild_assignments, max(0, len(self.round_state.history) - 1), action)

    def legal_token_rows(self) -> tuple[LegalTokenRow, ...]:
        rows = []
        for token in self.legal_next_tokens().tolist():
            token = int(token)
            if token == PASS_TOKEN:
                rows.append(LegalTokenRow(token, "PASS", 0, True))
            elif token == COMMIT_TOKEN:
                rows.append(LegalTokenRow(token, "COMMIT", 0, True))
            elif FAMILY_TOKEN_BASE <= token < FAMILY_TOKEN_BASE + len(FAMILY_ORDER):
                rows.append(LegalTokenRow(token, "FAMILY", token - FAMILY_TOKEN_BASE, False))
            elif RANK_TOKEN_BASE <= token < RANK_TOKEN_BASE + len(RANK_ORDER):
                rows.append(LegalTokenRow(token, "RANK", int(TokenCodec.rank_from_token(token)), False))
            elif SUIT_TOKEN_BASE <= token <= SUIT_TOKEN_BASE + 4:
                rows.append(LegalTokenRow(token, "SUIT", int(TokenCodec.suit_from_token(token)), False))
            elif LENGTH_TOKEN_BASE <= token <= LENGTH_TOKEN_BASE + 10:
                rows.append(LegalTokenRow(token, "LENGTH", TokenCodec.length_from_token(token), False))
            elif CARD_TOKEN_BASE <= token < CARD_TOKEN_BASE + 108:
                rows.append(LegalTokenRow(token, "CARD", TokenCodec.card_from_token(token), False))
            elif WILD_TOKEN_BASE <= token < WILD_TOKEN_BASE + 52:
                rank, suit = TokenCodec.wild_from_token(token)
                rows.append(LegalTokenRow(token, "WILD_ASSIGNMENT", int(rank) * 10 + int(suit), False))
            else:
                rows.append(LegalTokenRow(token, "TOKEN", token, False))
        return tuple(rows)

    def _result(self, *, is_commit: bool, committed_action: AtomicCommittedAction | None, reward: np.ndarray, info: dict, action_phase: str | None = None) -> ProtocolStepResult:
        public_action = None if committed_action is None else self._public_action(committed_action, action_phase)
        return ProtocolStepResult(self.prefix, reward.astype(np.float32, copy=False), self.done, self.truncated, is_commit, public_action, self.token_step, self.committed_step, info)

    def step(self, token_id: int) -> ProtocolStepResult:
        if self.done or self.truncated:
            raise EpisodeTerminatedError("stepwise protocol is terminal or truncated")
        token_id = int(token_id)
        legal = set(int(token) for token in self.legal_next_tokens())
        if token_id not in legal:
            raise IllegalActionError("token is not in legal-next-token set")
        before_round = self.round_state.clone()
        before_prefix = self._prefix_tokens
        before_token_step = self.token_step
        before_truncated = self.truncated
        action_phase = _phase_name(self.round_state)
        try:
            matching = [(sequence, action) for sequence, action in self._candidate_sequences() if sequence[: len(self._prefix_tokens)] == self._prefix_tokens and len(sequence) > len(self._prefix_tokens) and sequence[len(self._prefix_tokens)] == token_id]
            if not matching:
                raise IllegalActionError("token does not extend a live canonical prefix")
            self.token_step += 1
            if token_id == PASS_TOKEN:
                action = matching[0][1]
                apply_action(self.round_state, action)
                self._prefix_tokens = ()
                if not self.done:
                    self.legal_next_tokens()
                if self.token_step >= self.max_token_steps and not self.done:
                    self.truncated = True
                    reason = "max_token_steps"
                else:
                    reason = None
                reward = self.round_state.rewards.copy() if self.done else np.zeros(4, dtype=np.float32)
                return self._result(is_commit=True, committed_action=action, reward=reward, info={"termination_reason": reason}, action_phase=action_phase)
            if token_id == COMMIT_TOKEN:
                complete = [(sequence, action) for sequence, action in matching if len(sequence) == len(self._prefix_tokens) + 1 and sequence[-1] == COMMIT_TOKEN]
                if len(complete) != 1:
                    raise ProtocolError("COMMIT does not identify exactly one canonical action")
                action = complete[0][1]
                apply_action(self.round_state, action)
                self._prefix_tokens = ()
                if not self.done:
                    self.legal_next_tokens()
                if self.token_step >= self.max_token_steps and not self.done:
                    self.truncated = True
                    reason = "max_token_steps"
                else:
                    reason = None
                reward = self.round_state.rewards.copy() if self.done else np.zeros(4, dtype=np.float32)
                return self._result(is_commit=True, committed_action=action, reward=reward, info={"termination_reason": reason}, action_phase=action_phase)
            self._prefix_tokens = self._prefix_tokens + (token_id,)
            self.legal_next_tokens()
            if self.token_step >= self.max_token_steps:
                self.truncated = True
                info = {"termination_reason": "max_token_steps"}
            else:
                info = {"termination_reason": None}
            if self.round_state.serialize() != before_round.serialize():
                raise ProtocolError("non-commit token mutated public round state")
            return self._result(is_commit=False, committed_action=None, reward=np.zeros(4, dtype=np.float32), info=info)
        except Exception:
            self.round_state.overwrite_from(before_round)
            self._prefix_tokens = before_prefix
            self.token_step = before_token_step
            self.truncated = before_truncated
            raise

    def clone(self) -> "StepwiseActionState":
        other = StepwiseActionState(self.round_state.clone(), max_action_tokens=self.max_action_tokens, max_legal_next_tokens=self.max_legal_next_tokens, max_token_steps=self.max_token_steps)
        other.token_step = self.token_step
        other.truncated = self.truncated
        other._prefix_tokens = self._prefix_tokens
        return other

    def serialize(self) -> bytes:
        payload = {"round_state": self.round_state.serialize(), "prefix_tokens": list(self._prefix_tokens), "token_step": self.token_step, "truncated": self.truncated, "max_action_tokens": self.max_action_tokens, "max_legal_next_tokens": self.max_legal_next_tokens, "max_token_steps": self.max_token_steps}
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def deserialize(cls, payload: bytes) -> "StepwiseActionState":
        data = json.loads(payload.decode("utf-8"))
        state = cls(RoundState.from_serialized(data["round_state"]), max_action_tokens=data["max_action_tokens"], max_legal_next_tokens=data["max_legal_next_tokens"], max_token_steps=data["max_token_steps"])
        state._prefix_tokens = tuple(int(token) for token in data["prefix_tokens"])
        state.token_step = int(data["token_step"])
        state.truncated = bool(data["truncated"])
        return state


    @classmethod
    def from_serialized(cls, payload: bytes) -> "StepwiseActionState":
        return cls.deserialize(payload)


__all__ = [
    "ActionPrefix",
    "CommittedAction",
    "LegalTokenRow",
    "ProtocolCommittedAction",
    "BOS_TOKEN",
    "CARD_TOKEN_BASE",
    "COMMIT_TOKEN",
    "FAMILY_TOKEN_BASE",
    "LEAD_PLAY_TOKEN",
    "MAX_ACTION_TOKENS",
    "MAX_LEGAL_NEXT_TOKENS",
    "PAD_TOKEN",
    "PASS_TOKEN",
    "ProtocolError",
    "ProtocolStepResult",
    "RETURN_TOKEN",
    "StepResult",
    "StepwiseActionState",
    "TokenCodec",
]
