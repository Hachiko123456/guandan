"""Unit tests for the independent two-deck physical-card layer."""

from collections import Counter

import pytest

from guandan.cards import (
    ALL_CARDS,
    NORMAL_RANKS,
    Rank,
    Suit,
    TOTAL_CARDS,
    Card,
    cards_for_rank,
    decode_cards,
    effective_rank_order,
    effective_rank_value,
    encode_cards,
    full_deck,
    rank_counts,
    suit_counts,
)


def test_full_two_deck_size_and_rank_counts() -> None:
    cards = full_deck()
    assert len(cards) == TOTAL_CARDS == 108
    assert len({card.card_id for card in cards}) == 108
    assert Counter(card.rank for card in cards) == Counter(
        {rank: 8 for rank in NORMAL_RANKS}
    ) + Counter({Rank.SMALL_JOKER: 2, Rank.BIG_JOKER: 2})


def test_suit_counts_and_two_physical_copies() -> None:
    cards = full_deck()
    assert suit_counts(cards) == Counter(
        {
            Suit.CLUBS: 26,
            Suit.DIAMONDS: 26,
            Suit.HEARTS: 26,
            Suit.SPADES: 26,
            Suit.JOKER: 4,
        }
    )
    sevens = cards_for_rank(Rank.SEVEN)
    assert len(sevens) == 8
    assert len({card.card_id for card in sevens}) == 8
    assert len({(card.suit, card.deck_index) for card in sevens}) == 8


def test_encode_decode_is_lossless_and_order_preserving() -> None:
    selected = (ALL_CARDS[0], ALL_CARDS[53], ALL_CARDS[54], ALL_CARDS[-1])
    ids = encode_cards(selected)
    assert ids == (0, 53, 54, 107)
    assert decode_cards(ids) == selected


def test_duplicate_rank_suit_from_two_decks_is_valid_but_duplicate_id_is_not() -> None:
    cards = cards_for_rank(Rank.THREE)
    same_rank_suit = (cards[0], cards[4])
    assert same_rank_suit[0].rank is same_rank_suit[1].rank
    assert same_rank_suit[0].suit is same_rank_suit[1].suit
    assert same_rank_suit[0] != same_rank_suit[1]
    assert encode_cards(same_rank_suit) == (cards[0].card_id, cards[4].card_id)

    with pytest.raises(ValueError, match="physical card ID"):
        encode_cards((cards[0], cards[0]))
    with pytest.raises(ValueError, match="physical card ID"):
        decode_cards((cards[0].card_id, cards[0].card_id))


def test_card_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError):
        Card(-1)
    with pytest.raises(ValueError):
        Card(TOTAL_CARDS)
    with pytest.raises(TypeError):
        Card(True)  # type: ignore[arg-type]


def test_effective_order_is_not_doudizhu_order() -> None:
    order = effective_rank_order(Rank.FIVE)
    assert order[:4] == (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.SIX)
    assert order[-4:] == (Rank.ACE, Rank.FIVE, Rank.SMALL_JOKER, Rank.BIG_JOKER)
    assert effective_rank_value(Rank.TWO, Rank.FIVE) < effective_rank_value(Rank.THREE, Rank.FIVE)
    assert effective_rank_value(Rank.ACE, Rank.FIVE) < effective_rank_value(Rank.FIVE, Rank.FIVE)
    assert effective_rank_value(Rank.FIVE, Rank.FIVE) < effective_rank_value(Rank.SMALL_JOKER, Rank.FIVE)


def test_level_wild_identity_is_heart_only_and_never_a_joker() -> None:
    heart_fives = tuple(card for card in cards_for_rank(Rank.FIVE) if card.suit is Suit.HEARTS)
    club_five = next(card for card in cards_for_rank(Rank.FIVE) if card.suit is Suit.CLUBS)
    assert all(card.is_level_wild(Rank.FIVE) for card in heart_fives)
    assert not club_five.is_level_wild(Rank.FIVE)
    assert not ALL_CARDS[52].is_level_wild(Rank.FIVE)
    assert not ALL_CARDS[53].is_level_wild(Rank.FIVE)


def test_joker_identity_and_properties() -> None:
    small = ALL_CARDS[52]
    big = ALL_CARDS[53]
    assert small.rank is Rank.SMALL_JOKER
    assert big.rank is Rank.BIG_JOKER
    assert small.suit is Suit.JOKER
    assert big.suit is Suit.JOKER
    assert small.is_joker and big.is_joker
    assert rank_counts((small, big)) == Counter(
        {Rank.SMALL_JOKER: 1, Rank.BIG_JOKER: 1}
    )


def test_effective_order_treats_two_as_the_level_when_level_is_two() -> None:
    order = effective_rank_order(Rank.TWO)
    assert order == (
        Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT,
        Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
        Rank.TWO, Rank.SMALL_JOKER, Rank.BIG_JOKER,
    )
    assert effective_rank_value(Rank.ACE, Rank.TWO) < effective_rank_value(Rank.TWO, Rank.TWO)
    assert effective_rank_value(Rank.TWO, Rank.TWO) < effective_rank_value(Rank.SMALL_JOKER, Rank.TWO)


def test_effective_order_has_exactly_one_level_position_for_every_level() -> None:
    for level_rank in NORMAL_RANKS:
        order = effective_rank_order(level_rank)
        assert len(order) == len(set(order)) == 15
        assert order[-2:] == (Rank.SMALL_JOKER, Rank.BIG_JOKER)
        assert effective_rank_value(level_rank, level_rank) == 12
