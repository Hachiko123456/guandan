"""Atomic single-hand GuanDan round engine.

Scheme B is deliberately not implemented here. This module validates and
applies already committed actions; the later action-state layer will construct
these actions token by token.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np

from .cards import Card, Rank, cards_for_rank, sort_cards
from .combos import Combination, CombinationKind, can_beat, recognize_combinations, require_combination
from .state import (
    CommittedAction, EpisodeTerminatedError, GuandanError, IllegalActionError,
    Phase, RoundState, StateInvariantError, TEAM_OF,
)


def _next_active(state: RoundState, seat: int) -> int:
    for offset in range(1, 5):
        candidate = (seat + offset) % 4
        if candidate not in state.finished_ranks:
            return candidate
    raise StateInvariantError("no active seat remains")


def _finish_if_needed(state: RoundState, just_finished: int) -> None:
    if just_finished not in state.finished_ranks:
        state.finished_ranks.append(just_finished)
    # A two-player same-team finish is already a double-down result.
    if len(state.finished_ranks) >= 2 and TEAM_OF[state.finished_ranks[0]] == TEAM_OF[state.finished_ranks[1]]:
        remaining = [seat for seat in range(4) if seat not in state.finished_ranks]
        state.finished_ranks.extend(sorted(remaining))
    elif len(state.finished_ranks) == 3:
        remaining = next(seat for seat in range(4) if seat not in state.finished_ranks)
        state.finished_ranks.append(remaining)
    if len(state.finished_ranks) == 4:
        state.done = True
        state.phase = Phase.TERMINAL
        winner = TEAM_OF[state.finished_ranks[0]]
        state.rewards = np.asarray([1.0 if TEAM_OF[seat] == winner else -1.0 for seat in range(4)], dtype=np.float32)


def _validate_cards_in_hand(state: RoundState, action: CommittedAction) -> None:
    held = {card.card_id for card in state.hands[action.player_id]}
    selected = [card.card_id for card in action.cards]
    if len(selected) != len(set(selected)) or not set(selected).issubset(held):
        raise IllegalActionError("action selects a missing or duplicated physical card")


def apply_action(state: RoundState, action: CommittedAction) -> RoundState:
    """Apply one already committed play/pass and return the same mutated state."""
    if state.done or state.phase is Phase.TERMINAL:
        raise EpisodeTerminatedError("round is already terminal")
    if state.phase is not Phase.PLAY:
        raise NotImplementedError("tribute/return atomic actions are reserved for the next slice")
    if action.player_id != state.active_seat:
        raise IllegalActionError("action is from a non-active seat")
    if action.player_id in state.finished_ranks:
        raise IllegalActionError("finished seat cannot act")

    if action.is_pass:
        if state.current_winning is None:
            raise IllegalActionError("pass is illegal while leading")
        _validate_cards_in_hand(state, action) if action.cards else None
        state.history.append(action)
        state.committed_step += 1
        state.consecutive_passes += 1
        active_count = 4 - len(state.finished_ranks)
        if state.consecutive_passes >= max(1, active_count - 1):
            leader = state.current_winner_seat
            if leader is None:
                raise StateInvariantError("missing current winner")
            state.current_winning = None
            state.current_winner_seat = None
            state.consecutive_passes = 0
            state.leader_seat = leader
            state.active_seat = _next_active(state, leader)
        else:
            state.active_seat = _next_active(state, state.active_seat)
        state.check_invariants()
        return state

    if not isinstance(action.kind, CombinationKind):
        raise IllegalActionError("non-pass action must use a combination kind")
    _validate_cards_in_hand(state, action)
    declaration = action.declaration or require_combination(action.cards, level_rank=state.level_rank, kind=action.kind)
    if declaration.kind is not action.kind:
        raise IllegalActionError("action kind does not match declaration")
    if state.current_winning is not None and not can_beat(declaration, state.current_winning):
        raise IllegalActionError("play does not beat current winning play")

    old_winner = state.current_winner_seat
    state.hands[action.player_id] = [card for card in state.hands[action.player_id]
                                     if card.card_id not in {item.card_id for item in action.cards}]
    committed = CommittedAction(action.player_id, action.kind, tuple(sort_cards(action.cards)), declaration)
    state.history.append(committed)
    state.committed_step += 1
    state.current_winning = declaration
    state.current_winner_seat = action.player_id
    state.consecutive_passes = 0
    if not state.hands[action.player_id]:
        _finish_if_needed(state, action.player_id)
        if state.done:
            state.check_invariants()
            return state
    # Next active player after a committed play. If the winner has just emptied,
    # the partner receives the lead when the trick is later reset (接风).
    state.active_seat = _next_active(state, action.player_id)
    state.check_invariants()
    return state


def legal_follow(state: RoundState, cards: Iterable[Card], *, kind: CombinationKind) -> bool:
    """Validate a candidate atomic play without mutating state."""
    trial = state.clone()
    apply_action(trial, CommittedAction(state.active_seat, kind, tuple(cards)))
    return True


__all__ = ["apply_action", "legal_follow", "RoundState", "CommittedAction"]
