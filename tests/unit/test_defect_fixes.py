from __future__ import annotations

import pytest

from guandan.cards import ALL_CARDS, Card, Rank, Suit, cards_for_rank, full_deck
from guandan.round import apply_action, return_cards_for, tribute_card_for
from guandan.state import CommittedAction, IllegalActionError, Phase, PreviousHandResult, RoundState

PREVIOUS_HEAD_THIRD = PreviousHandResult((1, 2, 3, 4), 0, "HEAD_THIRD")


def _card(rank: Rank, suit: Suit, deck: int = 0) -> Card:
    return next(card for card in cards_for_rank(rank) if card.suit is suit and card.deck_index == deck)


def _hands_with_donor(donor_cards: list[Card]) -> list[list[Card]]:
    chosen = {card.card_id for card in donor_cards}
    rest = [card for card in full_deck() if card.card_id not in chosen]
    hands = [[], [], [], list(donor_cards)]
    for index, card in enumerate(rest):
        seat = index % 3
        if len(hands[seat]) < 27:
            hands[seat].append(card)
    assert all(len(hand) == 27 for hand in hands)
    return hands


def _partition_donor(donor_cards: list[Card], *, level_rank: Rank) -> RoundState:
    return RoundState.from_hands(_hands_with_donor(donor_cards), level_rank=level_rank, previous_result=PREVIOUS_HEAD_THIRD)


def test_tribute_candidate_includes_jokers_and_excludes_level_wild() -> None:
    state = RoundState.deal(seed=0, level_rank=Rank.SIX, previous_result=PREVIOUS_HEAD_THIRD)
    selected = tribute_card_for(state, 3)
    assert selected is not None and selected.rank is Rank.BIG_JOKER and selected.card_id == 107
    assert not selected.is_level_wild(state.level_rank)


def test_tribute_uses_level_strength_and_preserves_duplicate_entity_identity() -> None:
    donor = [card for rank in (Rank.THREE, Rank.FOUR, Rank.FIVE) for card in cards_for_rank(rank)][:24]
    donor.extend([_card(Rank.SIX, Suit.CLUBS), _card(Rank.SIX, Suit.DIAMONDS), _card(Rank.SIX, Suit.HEARTS)])
    state = _partition_donor(donor, level_rank=Rank.SIX)
    selected = tribute_card_for(state, 3)
    assert selected is not None and selected.rank is Rank.SIX and selected.suit is not Suit.HEARTS

    duplicate_donor = [card for rank in (Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN) for card in cards_for_rank(rank)][:25]
    duplicate_donor.extend([_card(Rank.KING, Suit.CLUBS, 0), _card(Rank.KING, Suit.CLUBS, 1)])
    duplicate_state = _partition_donor(duplicate_donor, level_rank=Rank.TWO)
    duplicate_selected = tribute_card_for(duplicate_state, 3)
    assert duplicate_selected is not None and duplicate_selected.rank is Rank.KING and duplicate_selected.card_id == 40
    assert len({card.card_id for card in duplicate_state.hands[3]}) == 27


def test_tribute_return_round_trip_excludes_every_current_level_card_and_preserves_cards() -> None:
    state = RoundState.deal(seed=0, level_rank=Rank.SIX, previous_result=PREVIOUS_HEAD_THIRD)
    tribute = tribute_card_for(state, 3)
    assert tribute is not None
    apply_action(state, CommittedAction(3, "tribute", (tribute,)))
    assert state.phase is Phase.RETURN and state.pending_return == (0, 3)
    candidates = return_cards_for(state, 0)
    assert candidates and all(card.rank is not Rank.SIX for card in candidates)
    assert all(not card.is_joker and not card.is_level_wild(Rank.SIX) for card in candidates)

    illegal_level_card = next(card for card in state.hands[0] if card.rank is Rank.SIX)
    before = state.serialize()
    with pytest.raises(IllegalActionError, match="return card"):
        apply_action(state, CommittedAction(0, "return", (illegal_level_card,)))
    assert state.serialize() == before

    returned = candidates[0]
    apply_action(state, CommittedAction(0, "return", (returned,)))
    assert state.phase is Phase.PLAY
    assert state.tribute_transfers == [(3, 0, tribute)]
    assert state.return_transfers == [(0, 3, returned)]
    state.check_invariants()
    assert len({card.card_id for hand in state.hands for card in hand}) == 108


def test_non_level_two_rank_is_also_excluded_from_returns() -> None:
    donor = [card for card in ALL_CARDS if card.rank not in {Rank.ACE, Rank.TWO, Rank.SMALL_JOKER, Rank.BIG_JOKER}][:27]
    state = _partition_donor(donor, level_rank=Rank.FIVE)
    tribute = tribute_card_for(state, 3)
    assert tribute is not None
    apply_action(state, CommittedAction(3, "tribute", (tribute,)))
    assert state.phase is Phase.RETURN
    assert all(card.rank is not Rank.FIVE for card in return_cards_for(state, 0))
