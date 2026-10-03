"""Demand-driven A01 acceptance tests for cards and combinations.

These tests are intentionally independent of the implementation's currently
returned enum set. They encode the v1 project rules and reject the old
provisional two-joker/variable-length behavior.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from guandan.cards import ALL_CARDS, Rank, Suit, TOTAL_CARDS, cards_for_rank
from guandan.combos import (
    CombinationKind,
    LevelContextMismatchError,
    can_beat,
    compare_combinations,
    recognize_combinations,
    require_combination,
)


@pytest.mark.acceptance
def test_two_decks_preserve_108_physical_cards() -> None:
    assert len(ALL_CARDS) == TOTAL_CARDS == 108
    assert len({card.card_id for card in ALL_CARDS}) == 108
    assert len(cards_for_rank(Rank.THREE)) == 8
    assert len(cards_for_rank(Rank.SMALL_JOKER)) == 2
    assert len(cards_for_rank(Rank.BIG_JOKER)) == 2


def card(rank: Rank, suit: Suit = Suit.CLUBS, deck: int = 0):
    return next(
        item for item in cards_for_rank(rank)
        if item.suit is suit and item.deck_index == deck
    )


def rank_cards(rank: Rank, count: int, *, exclude_suit: Suit | None = None):
    values = [item for item in cards_for_rank(rank) if item.suit is not exclude_suit]
    return tuple(values[:count])


@pytest.mark.acceptance
def test_all_ten_approved_families_are_demanded() -> None:
    cases = {
        CombinationKind.SINGLE: (card(Rank.THREE),),
        CombinationKind.PAIR: rank_cards(Rank.FOUR, 2),
        CombinationKind.TRIPLE: rank_cards(Rank.FIVE, 3),
        CombinationKind.FULL_HOUSE: rank_cards(Rank.SIX, 3) + rank_cards(Rank.SEVEN, 2),
        CombinationKind.STRAIGHT: tuple(card(rank) for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX)),
        CombinationKind.PAIR_SEQUENCE: rank_cards(Rank.SEVEN, 2) + rank_cards(Rank.EIGHT, 2) + rank_cards(Rank.NINE, 2),
        CombinationKind.TRIPLE_SEQUENCE: rank_cards(Rank.SEVEN, 3) + rank_cards(Rank.EIGHT, 3),
        CombinationKind.RANK_BOMB: rank_cards(Rank.NINE, 4),
        CombinationKind.STRAIGHT_FLUSH: tuple(card(rank, Suit.HEARTS) for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX)),
        CombinationKind.FOUR_KINGS: (ALL_CARDS[52], ALL_CARDS[53], ALL_CARDS[106], ALL_CARDS[107]),
    }
    for kind, cards in cases.items():
        declaration = require_combination(cards, kind=kind)
        assert declaration.kind is kind
        assert len({item.card_id for item in declaration.cards}) == len(cards)


@pytest.mark.acceptance
def test_four_kings_is_exactly_four_jokers_not_a_two_joker_rocket() -> None:
    four_kings = (ALL_CARDS[52], ALL_CARDS[53], ALL_CARDS[106], ALL_CARDS[107])
    assert require_combination(four_kings, kind=CombinationKind.FOUR_KINGS).kind is CombinationKind.FOUR_KINGS
    assert recognize_combinations((ALL_CARDS[52], ALL_CARDS[53])) == ()
    assert recognize_combinations((ALL_CARDS[52], ALL_CARDS[107])) == ()


@pytest.mark.acceptance
def test_fixed_sequence_lengths_and_windows() -> None:
    five = tuple(card(rank) for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX))
    low_ace = tuple(card(rank) for rank in (Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE))
    high_ace = tuple(card(rank) for rank in (Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE))
    invalid_six = tuple(card(rank) for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN))
    invalid_wrap = tuple(card(rank) for rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE, Rank.TWO))
    assert require_combination(five, kind=CombinationKind.STRAIGHT)
    assert require_combination(low_ace, kind=CombinationKind.STRAIGHT)
    assert require_combination(high_ace, kind=CombinationKind.STRAIGHT)
    assert recognize_combinations(invalid_six) == ()
    assert recognize_combinations(invalid_wrap) == ()
    assert recognize_combinations(rank_cards(Rank.TWO, 2) + rank_cards(Rank.THREE, 2) + rank_cards(Rank.FOUR, 2) + rank_cards(Rank.FIVE, 2)) == ()
    assert recognize_combinations(rank_cards(Rank.TWO, 3) + rank_cards(Rank.THREE, 3) + rank_cards(Rank.FOUR, 3)) == ()


@pytest.mark.acceptance
def test_straight_flush_is_suited_and_bomb_order_is_fixed() -> None:
    flush_cards = tuple(card(rank, Suit.SPADES) for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX))
    mixed = tuple(card(rank, Suit.SPADES if index % 2 else Suit.CLUBS) for index, rank in enumerate((Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX)))
    flush = require_combination(flush_cards, kind=CombinationKind.STRAIGHT_FLUSH)
    assert not any(item.kind is CombinationKind.STRAIGHT_FLUSH for item in recognize_combinations(mixed))
    bomb4 = require_combination(rank_cards(Rank.THREE, 4), kind=CombinationKind.RANK_BOMB)
    bomb5 = require_combination(rank_cards(Rank.THREE, 5), kind=CombinationKind.RANK_BOMB)
    bomb6 = require_combination(rank_cards(Rank.THREE, 6), kind=CombinationKind.RANK_BOMB)
    assert compare_combinations(bomb5, bomb4) > 0
    assert compare_combinations(flush, bomb5) > 0
    assert compare_combinations(bomb6, flush) > 0


@pytest.mark.acceptance
def test_level_and_wild_declarations_are_explicit_and_non_joker() -> None:
    level = Rank.FIVE
    wild = card(Rank.FIVE, Suit.HEARTS)
    cards = (card(Rank.TWO), card(Rank.THREE), card(Rank.FOUR), card(Rank.SIX), wild)
    declarations = recognize_combinations(cards, level_rank=level)
    straights = [item for item in declarations if item.kind is CombinationKind.STRAIGHT]
    assert straights
    assert any(wild.card_id in item.wild_assignments for item in straights)
    assert all(rank not in (Rank.SMALL_JOKER, Rank.BIG_JOKER) for item in straights for rank, _suit in item.wild_assignments.values())
    standalone = require_combination((wild,), level_rank=level, kind=CombinationKind.SINGLE)
    assert standalone.wild_assignments[wild.card_id] == (Rank.FIVE, Suit.HEARTS)
    assert standalone.natural_wild_ids == (wild.card_id,)


@pytest.mark.acceptance
def test_comparison_checks_level_context_and_rejects_forged_metadata() -> None:
    left = require_combination(rank_cards(Rank.THREE, 2), level_rank=Rank.FIVE, kind=CombinationKind.PAIR)
    right = require_combination(rank_cards(Rank.FOUR, 2), level_rank=Rank.SIX, kind=CombinationKind.PAIR)
    with pytest.raises(LevelContextMismatchError):
        compare_combinations(left, right)
    forged = replace(left, comparison_key=(999,))
    with pytest.raises(ValueError, match="declaration"):
        compare_combinations(forged, left)


@pytest.mark.acceptance
def test_non_bomb_families_do_not_cross_compare() -> None:
    single = require_combination((card(Rank.THREE),), kind=CombinationKind.SINGLE)
    pair = require_combination(rank_cards(Rank.FOUR, 2), kind=CombinationKind.PAIR)
    with pytest.raises(ValueError):
        compare_combinations(single, pair)
    assert not can_beat(single, pair)

