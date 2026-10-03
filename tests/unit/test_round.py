from __future__ import annotations

import random

import pytest

from guandan.cards import Rank, cards_for_rank
from guandan.combos import CombinationKind
from guandan.round import apply_action
from guandan.state import CommittedAction, EpisodeTerminatedError, IllegalActionError, Phase, RoundState


def test_deterministic_deal_and_card_conservation() -> None:
    left = RoundState.deal(seed=123, level_rank=Rank.FIVE)
    right = RoundState.deal(seed=123, level_rank=Rank.FIVE)
    assert [[c.card_id for c in hand] for hand in left.hands] == [[c.card_id for c in hand] for hand in right.hands]
    assert sorted(c.card_id for hand in left.hands for c in hand) == list(range(108))
    assert [len(hand) for hand in left.hands] == [27, 27, 27, 27]


def test_lead_play_changes_turn_and_consumes_cards() -> None:
    state = RoundState.deal(seed=2)
    card = state.hands[0][0]
    apply_action(state, CommittedAction(0, CombinationKind.SINGLE, (card,)))
    assert state.active_seat == 1
    assert len(state.hands[0]) == 26
    assert state.current_winner_seat == 0
    assert state.committed_step == 1


def test_pass_resets_trick_after_all_other_active_players_pass() -> None:
    state = RoundState.deal(seed=3)
    card = state.hands[0][0]
    apply_action(state, CommittedAction(0, CombinationKind.SINGLE, (card,)))
    apply_action(state, CommittedAction(1, "pass"))
    apply_action(state, CommittedAction(2, "pass"))
    apply_action(state, CommittedAction(3, "pass"))
    assert state.current_winning is None
    assert state.current_winner_seat is None
    assert state.active_seat == 0
    assert state.leader_seat == 0


def test_invalid_action_preserves_state() -> None:
    state = RoundState.deal(seed=4)
    before = state.serialize()
    missing = next(card for card in state.hands[1] if card.card_id not in {item.card_id for item in state.hands[0]})
    with pytest.raises(IllegalActionError):
        apply_action(state, CommittedAction(0, CombinationKind.SINGLE, (missing,)))
    assert state.serialize() == before


def test_fixture_requires_full_physical_deck_partition() -> None:
    state = RoundState.deal(seed=5)
    assert sum(len(hand) for hand in state.hands) == 108
    with pytest.raises(Exception):
        RoundState.from_hands([[state.hands[0][0]], [], [], []])


def test_terminal_state_rejects_future_actions() -> None:
    state = RoundState.deal(seed=6)
    state.done = True
    state.phase = Phase.TERMINAL
    with pytest.raises(EpisodeTerminatedError):
        apply_action(state, CommittedAction(0, "pass"))
