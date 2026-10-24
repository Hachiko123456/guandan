# SPDX-License-Identifier: MIT
"""FableDan guest policy for the MARVEL C++ arena.

FableDan (pure NumPy inference) selects one *complete* move per decision,
while MARVEL uses a stepwise token prefix protocol.  The guest session:

1. mirrors the round through the legacy ``StepwiseActionState`` (already kept
   in sync by the driver),
2. enumerates the engine's complete legal actions with canonical token
   sequences,
3. converts every engine action into FableDan's ``Move`` representation and
   intersects it with ``fabledan_vendor.combos.gen_moves`` outputs, so the
   guest only ever chooses actions it could have generated natively,
4. scores the intersection with the FableDan Q-network and returns the token
   sequence of the chosen action, which the driver replays row-by-row.

The vendored FableDan modules live in the delivery overlay package
``fabledan_vendor`` (not part of this repository).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from guandan.action_state import ProtocolCommittedAction, StepwiseActionState
from guandan.cards import Card
from guandan.combos import CombinationKind

from .cards_map import marvel_to_fd_card, marvel_to_fd_level, fd_claim_card

try:  # The vendor package exists only in the Kaggle comparison overlay.
    from fabledan_vendor import combos as fd_combos
    from fabledan_vendor import cards as fd_cards
    from fabledan_vendor.encode import hand_action_features, tokenize
    from fabledan_vendor.model_np import NumpyModel
except ImportError:  # pragma: no cover - exercised only without the overlay
    fd_combos = None
    fd_cards = None
    hand_action_features = None
    tokenize = None
    NumpyModel = None

FABLEDAN_VENDOR_AVAILABLE = fd_combos is not None

FAMILY_TO_TYPE: dict[str, int] = {} if not FABLEDAN_VENDOR_AVAILABLE else {
    "SINGLE": fd_combos.SINGLE,
    "PAIR": fd_combos.PAIR,
    "TRIPLE": fd_combos.TRIPLE,
    "FULL_HOUSE": fd_combos.FULL,
    "STRAIGHT": fd_combos.STRAIGHT,
    "PAIR_SEQUENCE": fd_combos.PLATE,
    "TRIPLE_SEQUENCE": fd_combos.TUBE,
    "RANK_BOMB": fd_combos.BOMB,
    "STRAIGHT_FLUSH": fd_combos.SFLUSH,
    "FOUR_KINGS": fd_combos.ROCKET,
}
SEQUENCE_SPEC = {} if not FABLEDAN_VENDOR_AVAILABLE else {
    fd_combos.STRAIGHT: (5, 1),
    fd_combos.SFLUSH: (5, 1),
    fd_combos.PLATE: (3, 2),
    fd_combos.TUBE: (2, 3),
}


def move_key(move) -> tuple:
    return (
        int(move.type),
        tuple(sorted(int(r) for r in move.claim_ranks)),
        tuple(sorted(int(c) for c in move.cards)),
    )


def _claim_cards_from(marvel_card_ids: Sequence[int], wild_assignments: Mapping[int, tuple[int, int]]) -> list[int]:
    claims = []
    for card_id in marvel_card_ids:
        card_id = int(card_id)
        if card_id in wild_assignments:
            rank, suit = wild_assignments[card_id]
            claims.append(fd_claim_card(int(rank), int(suit)))
        else:
            card = Card(card_id)
            claims.append(fd_claim_card(int(card.rank), int(card.suit)))
    return claims


def _classify(cards_fd: list[int], claims: list[int], level: int):
    return fd_combos.classify_claim(cards_fd, claims, level)


def _fallback_move(family: str, cards_fd: list[int], claims: list[int], level: int):
    name = str(family or "").upper()
    mtype = FAMILY_TO_TYPE.get(name)
    if mtype is None:
        raise ValueError(f"unknown family for fallback move: {family!r}")
    ranks = sorted(fd_cards.rank_of(c) for c in claims)
    if mtype == fd_combos.ROCKET:
        key = 0
    elif mtype in SEQUENCE_SPEC:
        length, mult = SEQUENCE_SPEC[mtype]
        low = fd_combos._find_seq_low(ranks, length, mult)
        key = low if low is not None else (min(ranks) if ranks else 0)
    elif mtype == fd_combos.FULL:
        counts: dict[int, int] = {}
        for rank in ranks:
            counts[rank] = counts.get(rank, 0) + 1
        trip = next((rank for rank, count in counts.items() if count == 3), ranks[-1] if ranks else 0)
        key = fd_cards.order_of(trip, level)
    else:
        key = fd_cards.order_of(ranks[0] if ranks else 0, level)
    return fd_combos.Move(mtype, key, list(cards_fd), ranks)


def resolve_action_move(action: ProtocolCommittedAction, level: int) -> tuple[object, bool]:
    """Return ``(fd Move, used_fallback)`` for an engine public action."""
    if action.kind == "PASS":
        return fd_combos.PASS_MOVE, False
    if action.kind in {"TRIBUTE", "RETURN"}:
        raise RuntimeError("tribute/return commits are out of scope for this comparison")
    wild_map = {int(card_id): (int(rank), int(suit)) for card_id, rank, suit in action.wild_assignments}
    cards_fd = [marvel_to_fd_card(card_id) for card_id in action.card_ids]
    claims = _claim_cards_from(action.card_ids, wild_map)
    try:
        return _classify(cards_fd, claims, level), False
    except ValueError:
        return _fallback_move(str(action.family or action.kind), cards_fd, claims, level), True


def resolve_combination_move(combination, level: int) -> tuple[object, bool]:
    """Return ``(fd Move, used_fallback)`` for a MARVEL ``Combination``."""
    wild_map = {
        int(card_id): (int(rank), int(suit))
        for card_id, (rank, suit) in combination.wild_assignments.items()
    }
    card_ids = [card.card_id for card in combination.cards]
    cards_fd = [marvel_to_fd_card(card_id) for card_id in card_ids]
    claims = _claim_cards_from(card_ids, wild_map)
    try:
        return _classify(cards_fd, claims, level), False
    except ValueError:
        return _fallback_move(combination.kind.value.upper(), cards_fd, claims, level), True


def atomic_to_public(action) -> ProtocolCommittedAction:
    """Wrap an atomic ``CommittedAction`` from candidate enumeration."""
    kind = action.kind.value.upper() if isinstance(action.kind, CombinationKind) else str(action.kind).upper()
    family = None
    wilds: tuple[tuple[int, int, int], ...] = ()
    if action.declaration is not None:
        family = action.declaration.kind.value.upper()
        wilds = tuple(
            (int(card_id), int(rank), int(suit))
            for card_id, (rank, suit) in sorted(action.declaration.wild_assignments.items())
        )
    return ProtocolCommittedAction(
        actor=int(action.player_id), phase="", kind=kind, family=family,
        card_ids=tuple(int(card.card_id) for card in action.cards),
        declared_ranks=(), declared_suit=None, wild_assignments=wilds,
        public_index=0, atomic_action=action,
    )


def classify_candidate(action: ProtocolCommittedAction, level: int):
    """Strict conversion of a candidate action; ``None`` when unclassifiable."""
    if action.kind == "PASS":
        return fd_combos.PASS_MOVE
    if action.family is None:
        return None
    wild_map = {int(card_id): (int(rank), int(suit)) for card_id, rank, suit in action.wild_assignments}
    cards_fd = [marvel_to_fd_card(card_id) for card_id in action.card_ids]
    claims = _claim_cards_from(action.card_ids, wild_map)
    try:
        return _classify(cards_fd, claims, level)
    except ValueError:
        return None


class FableDanGuestSession:
    """Per-game guest state: event mirror, plan execution, diagnostics."""

    def __init__(self, model: NumpyModel, level: int):
        if not FABLEDAN_VENDOR_AVAILABLE:
            raise RuntimeError(
                "fabledan_vendor is required for FableDan guest games; "
                "run this arm from the Kaggle comparison overlay delivery"
            )
        self.model = model
        self.level = int(level)
        self.events: list[tuple] = []
        self._plan: tuple[int, ...] | None = None
        self._plan_pos = 0
        self.diagnostics = {
            "decisions": 0,
            "model_decisions": 0,
            "auto_single": 0,
            "engine_candidates": 0,
            "fd_candidates": 0,
            "intersection": 0,
            "unsupported_candidates": 0,
            "empty_intersection_fallbacks": 0,
            "classification_fallbacks": 0,
        }

    # ------------------------------------------------------------------
    def record_commit(self, action: ProtocolCommittedAction | None) -> None:
        if action is None:
            return
        actor = int(action.actor)
        if action.kind == "PASS":
            self.events.append(("pass", actor))
            return
        move, used_fallback = resolve_action_move(action, self.level)
        self.diagnostics["classification_fallbacks"] += int(used_fallback)
        self.events.append(("play", actor, move))

    # ------------------------------------------------------------------
    def next_token(self, protocol: StepwiseActionState) -> int:
        if self._plan is None:
            if tuple(protocol._prefix_tokens):
                raise RuntimeError("guest session asked mid-prefix without an active plan")
            self._plan = self._decide(protocol)
            self._plan_pos = 0
        token = int(self._plan[self._plan_pos])
        self._plan_pos += 1
        if self._plan_pos >= len(self._plan):
            self._plan = None
            self._plan_pos = 0
        return token

    # ------------------------------------------------------------------
    def _decide(self, protocol: StepwiseActionState) -> tuple[int, ...]:
        state = protocol.round_state
        actor = int(state.active_seat)
        sequences = protocol._candidate_sequences()
        if not sequences:
            raise RuntimeError("guest session reached a decision point with no legal action")

        level = self.level
        entries: list[tuple[tuple[int, ...], ProtocolCommittedAction, object | None]] = []
        unsupported = 0
        for tokens, atomic in sequences:
            public = atomic_to_public(atomic)
            fd_move = classify_candidate(public, level)
            if fd_move is None:
                unsupported += 1
            entries.append((tokens, public, fd_move))

        engine_index: dict[tuple, int] = {}
        for index, (_tokens, _action, fd_move) in enumerate(entries):
            if fd_move is None:
                continue
            engine_index.setdefault(move_key(fd_move), index)

        hand_fd = [marvel_to_fd_card(card.card_id) for card in state.hands[actor]]
        left = [len(hand) for hand in state.hands]
        done = [seat in state.finished_ranks for seat in range(4)]
        lead = None
        if state.current_winning is not None:
            lead, used_fallback = resolve_combination_move(state.current_winning, level)
            self.diagnostics["classification_fallbacks"] += int(used_fallback)
        fd_moves = fd_combos.gen_moves(hand_fd, level, lead)

        intersect: list[tuple[int, int]] = []
        for fd_index, fd_move in enumerate(fd_moves):
            index = engine_index.get(move_key(fd_move))
            if index is not None:
                intersect.append((fd_index, index))

        self.diagnostics["decisions"] += 1
        self.diagnostics["engine_candidates"] += len(entries)
        self.diagnostics["fd_candidates"] += len(fd_moves)
        self.diagnostics["intersection"] += len(intersect)
        self.diagnostics["unsupported_candidates"] += unsupported

        if not intersect:
            self.diagnostics["empty_intersection_fallbacks"] += 1
            return entries[0][0]
        if len(intersect) == 1:
            self.diagnostics["auto_single"] += 1
            return entries[intersect[0][1]][0]

        obs = {
            "player": actor,
            "level": level,
            "hand": hand_fd,
            "legal": fd_moves,
            "lead": lead,
            "lead_owner": state.current_winner_seat,
            "events": self.events,
            "done": done,
            "left": left,
        }
        history = tokenize(self.events, actor, level)
        features = np.stack([hand_action_features(obs, fd_moves[fd_index]) for fd_index, _ in intersect])
        q_values = np.asarray(self.model.q_values(history, features), dtype=np.float64)
        best = int(np.argmax(q_values))
        self.diagnostics["model_decisions"] += 1
        return entries[intersect[best][1]][0]


__all__ = [
    "FableDanGuestSession", "classify_candidate", "move_key",
    "resolve_action_move", "resolve_combination_move",
]
