"""Unit tests for the ten-family GuanDan declaration layer."""

from dataclasses import replace

import pytest

from guandan.cards import ALL_CARDS, Rank, Suit, cards_for_rank
from guandan.combos import (
    CombinationKind,
    IncomparableCombinationsError,
    LevelContextMismatchError,
    can_beat,
    compare_combinations,
    recognize_combination,
    recognize_combinations,
    require_combination,
)


def cards_of_rank(rank: Rank, count: int, *, exclude_suit: Suit | None = None):
    cards = [card for card in cards_for_rank(rank) if card.suit is not exclude_suit]
    return tuple(cards[:count])


def card(rank: Rank, suit: Suit = Suit.CLUBS, deck: int = 0):
    return next(
        item
        for item in cards_for_rank(rank)
        if item.suit is suit and item.deck_index == deck
    )


def wilds(level_rank: Rank = Rank.FIVE):
    return tuple(item for item in cards_for_rank(level_rank) if item.suit is Suit.HEARTS)


def kinds(cards, *, level_rank: Rank = Rank.FIVE):
    return {declaration.kind for declaration in recognize_combinations(cards, level_rank=level_rank)}


def test_all_ten_family_names_are_present() -> None:
    assert {
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
    } == set(CombinationKind)


def test_basic_families_and_same_joker_pairs() -> None:
    assert require_combination((card(Rank.THREE),), kind=CombinationKind.SINGLE).kind is CombinationKind.SINGLE
    assert require_combination(cards_of_rank(Rank.FOUR, 2), kind=CombinationKind.PAIR).kind is CombinationKind.PAIR
    assert require_combination(cards_of_rank(Rank.SIX, 3), kind=CombinationKind.TRIPLE).kind is CombinationKind.TRIPLE
    assert require_combination(
        cards_of_rank(Rank.SIX, 3) + cards_of_rank(Rank.SEVEN, 2), kind=CombinationKind.FULL_HOUSE
    ).kind is CombinationKind.FULL_HOUSE

    small_pair = (ALL_CARDS[52], ALL_CARDS[106])
    big_pair = (ALL_CARDS[53], ALL_CARDS[107])
    mixed_jokers = (ALL_CARDS[52], ALL_CARDS[53])
    assert require_combination(small_pair, kind=CombinationKind.PAIR).comparison_rank is Rank.SMALL_JOKER
    assert require_combination(big_pair, kind=CombinationKind.PAIR).comparison_rank is Rank.BIG_JOKER
    assert recognize_combinations(mixed_jokers) == ()


def test_four_kings_requires_all_four_jokers() -> None:
    four_kings = (ALL_CARDS[52], ALL_CARDS[53], ALL_CARDS[106], ALL_CARDS[107])
    declaration = require_combination(four_kings, kind=CombinationKind.FOUR_KINGS)
    assert declaration.kind is CombinationKind.FOUR_KINGS
    assert kinds((ALL_CARDS[52], ALL_CARDS[53])) == set()


def test_exact_five_straights_and_allowed_low_windows() -> None:
    straight_23456 = tuple(card(rank) for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX))
    straight_a2345 = tuple(card(rank) for rank in (Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE))
    straight_10jqka = tuple(card(rank) for rank in (Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE))
    invalid_six = tuple(card(rank) for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN))
    invalid_jqka2 = tuple(card(rank) for rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE, Rank.TWO))
    invalid_ka234 = tuple(card(rank) for rank in (Rank.KING, Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR))

    assert require_combination(straight_23456, kind=CombinationKind.STRAIGHT).comparison_rank is Rank.SIX
    assert require_combination(straight_a2345, kind=CombinationKind.STRAIGHT).comparison_rank is Rank.FIVE
    assert require_combination(straight_10jqka, kind=CombinationKind.STRAIGHT).comparison_rank is Rank.ACE
    assert recognize_combinations(invalid_six) == ()
    assert recognize_combinations(invalid_jqka2) == ()
    assert recognize_combinations(invalid_ka234) == ()


def test_pair_and_triple_sequence_low_windows() -> None:
    pair_a23 = cards_of_rank(Rank.ACE, 2) + cards_of_rank(Rank.TWO, 2) + cards_of_rank(Rank.THREE, 2)
    triple_a2 = cards_of_rank(Rank.ACE, 3) + cards_of_rank(Rank.TWO, 3)
    pair_ka2 = cards_of_rank(Rank.KING, 2) + cards_of_rank(Rank.ACE, 2) + cards_of_rank(Rank.TWO, 2)
    triple_ka2 = cards_of_rank(Rank.KING, 3) + cards_of_rank(Rank.ACE, 3) + cards_of_rank(Rank.TWO, 3)

    assert require_combination(pair_a23, kind=CombinationKind.PAIR_SEQUENCE).comparison_rank is Rank.THREE
    assert require_combination(triple_a2, kind=CombinationKind.TRIPLE_SEQUENCE).comparison_rank is Rank.TWO
    assert recognize_combinations(pair_ka2) == ()
    assert recognize_combinations(triple_ka2) == ()

    four_pairs = cards_of_rank(Rank.TWO, 2) + cards_of_rank(Rank.THREE, 2) + cards_of_rank(Rank.FOUR, 2) + cards_of_rank(Rank.FIVE, 2)
    three_triples = cards_of_rank(Rank.TWO, 3) + cards_of_rank(Rank.THREE, 3) + cards_of_rank(Rank.FOUR, 3)
    assert not any(item.kind is CombinationKind.PAIR_SEQUENCE for item in recognize_combinations(four_pairs))
    assert not any(item.kind is CombinationKind.TRIPLE_SEQUENCE for item in recognize_combinations(three_triples))


def test_same_suit_five_cards_have_two_explicit_declarations() -> None:
    cards = tuple(card(rank, Suit.CLUBS) for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX))
    declarations = recognize_combinations(cards, level_rank=Rank.FIVE)
    declared_kinds = {declaration.kind for declaration in declarations}
    assert CombinationKind.STRAIGHT in declared_kinds
    assert CombinationKind.STRAIGHT_FLUSH in declared_kinds
    assert require_combination(cards, level_rank=Rank.FIVE, kind=CombinationKind.STRAIGHT).kind is CombinationKind.STRAIGHT
    assert require_combination(cards, level_rank=Rank.FIVE, kind=CombinationKind.STRAIGHT_FLUSH).kind is CombinationKind.STRAIGHT_FLUSH
    assert recognize_combination(cards, level_rank=Rank.FIVE).kind is CombinationKind.STRAIGHT


def test_wild_substitution_is_explicit_and_standalone_wild_is_natural_only() -> None:
    level_wilds = wilds()
    bomb_cards = cards_of_rank(Rank.THREE, 3) + level_wilds
    bomb_declarations = [
        declaration
        for declaration in recognize_combinations(bomb_cards, level_rank=Rank.FIVE)
        if declaration.kind is CombinationKind.RANK_BOMB
    ]
    assert bomb_declarations
    assert all(set(declaration.wild_assignments) == {item.card_id for item in level_wilds} for declaration in bomb_declarations)
    assert all(all(value[0] is Rank.THREE for value in declaration.wild_assignments.values()) for declaration in bomb_declarations)

    standalone = require_combination((level_wilds[0],), level_rank=Rank.FIVE, kind=CombinationKind.SINGLE)
    assert standalone.wild_assignments[level_wilds[0].card_id] == (Rank.FIVE, Suit.HEARTS)
    assert standalone.natural_wild_ids == (level_wilds[0].card_id,)

    assert recognize_combinations((level_wilds[0], ALL_CARDS[52]), level_rank=Rank.FIVE) == ()


def test_wild_can_fill_straight_and_straight_flush_suit_is_constrained() -> None:
    level_wild = wilds()[0]
    cards = (card(Rank.TWO), card(Rank.THREE), card(Rank.FOUR), card(Rank.SIX), level_wild)
    declarations = recognize_combinations(cards, level_rank=Rank.FIVE)
    straight_declarations = [item for item in declarations if item.kind is CombinationKind.STRAIGHT]
    assert straight_declarations
    assert any(level_wild.card_id in item.wild_assignments for item in straight_declarations)
    assert all(value[0] is not Rank.SMALL_JOKER and value[0] is not Rank.BIG_JOKER for item in straight_declarations for value in item.wild_assignments.values())

    hearts = (card(Rank.TWO, Suit.HEARTS), card(Rank.THREE, Suit.HEARTS), card(Rank.FOUR, Suit.HEARTS), card(Rank.SIX, Suit.HEARTS), level_wild)
    flushes = [item for item in recognize_combinations(hearts, level_rank=Rank.FIVE) if item.kind is CombinationKind.STRAIGHT_FLUSH]
    assert flushes
    assert all(value[1] is Suit.HEARTS for item in flushes for value in item.wild_assignments.values())


def test_rank_bomb_limits_and_bomb_class_order() -> None:
    bomb4 = require_combination(cards_of_rank(Rank.THREE, 4), level_rank=Rank.FIVE, kind=CombinationKind.RANK_BOMB)
    bomb5 = require_combination(cards_of_rank(Rank.THREE, 5), level_rank=Rank.FIVE, kind=CombinationKind.RANK_BOMB)
    straight_flush = require_combination(
        tuple(card(rank, Suit.CLUBS) for rank in (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX)),
        level_rank=Rank.FIVE,
        kind=CombinationKind.STRAIGHT_FLUSH,
    )
    bomb6 = require_combination(cards_of_rank(Rank.THREE, 6), level_rank=Rank.FIVE, kind=CombinationKind.RANK_BOMB)
    four_kings = require_combination((ALL_CARDS[52], ALL_CARDS[53], ALL_CARDS[106], ALL_CARDS[107]), level_rank=Rank.FIVE, kind=CombinationKind.FOUR_KINGS)

    assert compare_combinations(bomb5, bomb4) > 0
    assert compare_combinations(straight_flush, bomb5) > 0
    assert compare_combinations(bomb6, straight_flush) > 0
    assert compare_combinations(four_kings, bomb6) > 0
    assert can_beat(bomb6, bomb5)


def test_level_rank_controls_pair_strength_and_level_bomb_rank() -> None:
    low_two = require_combination(cards_of_rank(Rank.TWO, 2), level_rank=Rank.FIVE, kind=CombinationKind.PAIR)
    three = require_combination(cards_of_rank(Rank.THREE, 2), level_rank=Rank.FIVE, kind=CombinationKind.PAIR)
    ace = require_combination(cards_of_rank(Rank.ACE, 2), level_rank=Rank.FIVE, kind=CombinationKind.PAIR)
    level_pair = require_combination(cards_of_rank(Rank.FIVE, 2, exclude_suit=Suit.HEARTS), level_rank=Rank.FIVE, kind=CombinationKind.PAIR)
    assert compare_combinations(three, low_two) > 0
    assert compare_combinations(ace, three) > 0
    assert compare_combinations(level_pair, ace) > 0

    level_bomb = require_combination(cards_of_rank(Rank.FIVE, 4, exclude_suit=Suit.HEARTS), level_rank=Rank.FIVE, kind=CombinationKind.RANK_BOMB)
    ace_bomb = require_combination(cards_of_rank(Rank.ACE, 4), level_rank=Rank.FIVE, kind=CombinationKind.RANK_BOMB)
    assert compare_combinations(level_bomb, ace_bomb) > 0


def test_comparison_rejects_different_levels_and_forged_metadata() -> None:
    left = require_combination(cards_of_rank(Rank.THREE, 2), level_rank=Rank.FIVE, kind=CombinationKind.PAIR)
    right = require_combination(cards_of_rank(Rank.FOUR, 2), level_rank=Rank.SIX, kind=CombinationKind.PAIR)
    with pytest.raises(LevelContextMismatchError):
        compare_combinations(left, right)

    forged = replace(left, comparison_key=(999,))
    with pytest.raises(ValueError, match="declaration"):
        compare_combinations(forged, left)


def test_invalid_shapes_and_duplicate_physical_ids_are_rejected() -> None:
    assert recognize_combinations(()) == ()
    assert recognize_combinations((card(Rank.THREE), card(Rank.FOUR))) == ()
    repeated = card(Rank.SEVEN)
    assert recognize_combinations((repeated, repeated)) == ()
    assert recognize_combinations((ALL_CARDS[52], ALL_CARDS[53])) == ()
    assert recognize_combinations(cards_of_rank(Rank.THREE, 3) + cards_of_rank(Rank.FOUR, 2) + (card(Rank.FIVE),)) == ()


def test_legacy_aliases_remain_available() -> None:
    assert CombinationKind.TRIPLE_WITH_PAIR is CombinationKind.FULL_HOUSE
    assert CombinationKind.BOMB is CombinationKind.RANK_BOMB
    assert CombinationKind.KING_BOMB is CombinationKind.FOUR_KINGS
    assert CombinationKind.CONSECUTIVE_PAIRS is CombinationKind.PAIR_SEQUENCE
    assert CombinationKind.STEEL_PLATE is CombinationKind.TRIPLE_SEQUENCE


def test_level_two_strength_is_shared_by_pairs_and_bombs() -> None:
    ace_pair = require_combination(cards_of_rank(Rank.ACE, 2), level_rank=Rank.TWO, kind=CombinationKind.PAIR)
    two_pair = require_combination(cards_of_rank(Rank.TWO, 2), level_rank=Rank.TWO, kind=CombinationKind.PAIR)
    assert compare_combinations(two_pair, ace_pair) > 0

    ace_bomb = require_combination(cards_of_rank(Rank.ACE, 4), level_rank=Rank.TWO, kind=CombinationKind.RANK_BOMB)
    two_bomb = require_combination(cards_of_rank(Rank.TWO, 4), level_rank=Rank.TWO, kind=CombinationKind.RANK_BOMB)
    assert compare_combinations(two_bomb, ace_bomb) > 0


def test_same_heart_level_declaration_is_canonicalized_to_natural_use() -> None:
    level = Rank.SIX
    wild = card(level, Suit.HEARTS)
    cards = (card(Rank.FIVE, Suit.HEARTS), wild, card(Rank.SEVEN, Suit.HEARTS), card(Rank.EIGHT, Suit.HEARTS), card(Rank.NINE, Suit.HEARTS))
    declarations = [item for item in recognize_combinations(cards, level_rank=level) if item.kind is CombinationKind.STRAIGHT_FLUSH]
    assert len(declarations) == 1
    assert declarations[0].wild_assignments[wild.card_id] == (level, Suit.HEARTS)
    assert declarations[0].natural_wild_ids == (wild.card_id,)
