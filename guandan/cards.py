"""Physical cards and rank strength utilities for the independent GuanDan project.

The two-deck identity model is deliberately independent from combination
semantics.  Card IDs remain 0..107 (two 54-card decks), while rank strength is
computed separately with a caller-supplied ``level_rank``:

    2 < 3 < ... < A < level < small joker < big joker

When ``level_rank`` is ``Rank.TWO``, the printed 2 occupies the single level
position above A: ``3 < ... < A < 2 < small joker < big joker``.  There is no
second level position, and the printed rank of a card never changes.  A heart
card whose printed rank equals the
level is a wild card for declarations, but its physical identity and printed
rank remain unchanged.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class Suit(IntEnum):
    """The four standard suits plus the marker used by jokers."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3
    JOKER = 4


class Rank(IntEnum):
    """Printed rank labels used by the two physical decks."""

    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    TWO = 15
    SMALL_JOKER = 16
    BIG_JOKER = 17


DECKS = 2
CARDS_PER_DECK = 54
NORMAL_CARDS_PER_DECK = 52
TOTAL_CARDS = DECKS * CARDS_PER_DECK

# Printed order is intentionally separate from enum integer values.  The
# enum values preserve the existing card-ID layout; this tuple drives windows.
PRINTED_RANKS: tuple[Rank, ...] = (
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.TEN,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.ACE,
    Rank.TWO,
)
# Window order is a game-strength order, not the physical card-ID layout.
WINDOW_RANKS: tuple[Rank, ...] = (Rank.TWO, *PRINTED_RANKS[:-1])
NORMAL_RANKS = PRINTED_RANKS
NON_JOKER_RANKS = PRINTED_RANKS
JOKER_RANKS: tuple[Rank, ...] = (Rank.SMALL_JOKER, Rank.BIG_JOKER)


@dataclass(frozen=True, slots=True)
class Card:
    """A physical card represented by a stable identity.

    Equality is physical identity equality.  The two copies of the same
    printed rank and suit therefore compare unequal because their ``card_id``
    values differ.
    """

    card_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.card_id, int) or isinstance(self.card_id, bool):
            raise TypeError("card_id must be an integer")
        if not 0 <= self.card_id < TOTAL_CARDS:
            raise ValueError(f"card_id must be in [0, {TOTAL_CARDS}), got {self.card_id}")

    @property
    def deck_index(self) -> int:
        return self.card_id // CARDS_PER_DECK

    @property
    def deck_offset(self) -> int:
        return self.card_id % CARDS_PER_DECK

    @property
    def rank(self) -> Rank:
        offset = self.deck_offset
        if offset < NORMAL_CARDS_PER_DECK:
            rank_index, _ = divmod(offset, 4)
            return PRINTED_RANKS[rank_index]
        return JOKER_RANKS[offset - NORMAL_CARDS_PER_DECK]

    @property
    def suit(self) -> Suit:
        offset = self.deck_offset
        if offset < NORMAL_CARDS_PER_DECK:
            _, suit_index = divmod(offset, 4)
            return Suit(suit_index)
        return Suit.JOKER

    @property
    def is_joker(self) -> bool:
        return self.suit is Suit.JOKER

    def is_level_wild(self, level_rank: Rank) -> bool:
        """Return whether this physical card is the heart-level wild."""

        level_rank = validate_level_rank(level_rank)
        return self.suit is Suit.HEARTS and self.rank is level_rank

    def __str__(self) -> str:
        return f"{self.rank.name}@{self.suit.name}[d{self.deck_index}]"


ALL_CARDS: tuple[Card, ...] = tuple(Card(card_id) for card_id in range(TOTAL_CARDS))


def validate_level_rank(level_rank: Rank) -> Rank:
    """Validate and normalize a non-joker level rank."""

    try:
        normalized = Rank(level_rank)
    except (TypeError, ValueError) as exc:
        raise ValueError("level_rank must be a printed non-joker rank") from exc
    if normalized in JOKER_RANKS:
        raise ValueError("a joker cannot be the level rank")
    return normalized


def effective_rank_order(level_rank: Rank = Rank.TWO) -> tuple[Rank, ...]:
    """Return low-to-high ordinary strength followed by jokers.

    The level rank is moved above A, rather than making printed 2 the usual
    high card.  ``Rank.TWO`` is the special starting level and is not
    duplicated in the order.
    """

    level_rank = validate_level_rank(level_rank)
    non_two = [rank for rank in PRINTED_RANKS if rank is not Rank.TWO]
    if level_rank is Rank.TWO:
        # The starting level is a real level position, not the low natural 2.
        # There is only one printed rank TWO, so it is moved above A exactly
        # once and remains below the jokers.
        ordinary = [*non_two, Rank.TWO]
    else:
        ordinary = [Rank.TWO, *non_two]
        ordinary.remove(level_rank)
        ordinary.append(level_rank)
    return (*ordinary, *JOKER_RANKS)


def effective_rank_value(rank: Rank, level_rank: Rank = Rank.TWO) -> int:
    """Return a comparable strength value under ``level_rank``."""

    rank = Rank(rank)
    return effective_rank_order(level_rank).index(rank)


def card_from_id(card_id: int) -> Card:
    return Card(card_id)


def encode_card(card: Card) -> int:
    if not isinstance(card, Card):
        raise TypeError(f"expected Card, got {type(card).__name__}")
    return card.card_id


def decode_cards(card_ids: Iterable[int], *, reject_duplicates: bool = True) -> tuple[Card, ...]:
    ids = tuple(card_ids)
    if reject_duplicates and len(set(ids)) != len(ids):
        raise ValueError("a physical card ID occurs more than once")
    return tuple(card_from_id(card_id) for card_id in ids)


def encode_cards(cards: Iterable[Card], *, reject_duplicates: bool = True) -> tuple[int, ...]:
    cards_tuple = tuple(cards)
    ids = tuple(encode_card(card) for card in cards_tuple)
    if reject_duplicates and len(set(ids)) != len(ids):
        raise ValueError("a physical card ID occurs more than once")
    return ids


def full_deck() -> tuple[Card, ...]:
    return ALL_CARDS


def rank_order_value(rank: Rank) -> int:
    """Return a stable printed-order value for canonical serialization."""

    return (*PRINTED_RANKS, *JOKER_RANKS).index(Rank(rank))


def card_sort_key(card: Card) -> tuple[int, int, int, int]:
    if not isinstance(card, Card):
        raise TypeError(f"expected Card, got {type(card).__name__}")
    return (rank_order_value(card.rank), int(card.suit), card.deck_index, card.card_id)


def sort_cards(cards: Iterable[Card]) -> tuple[Card, ...]:
    return tuple(sorted(cards, key=card_sort_key))


def rank_counts(cards: Iterable[Card]) -> Counter[Rank]:
    return Counter(card.rank for card in cards)


def suit_counts(cards: Iterable[Card]) -> Counter[Suit]:
    return Counter(card.suit for card in cards)


def validate_physical_cards(cards: Iterable[Card]) -> tuple[Card, ...]:
    cards_tuple = tuple(cards)
    encode_cards(cards_tuple)
    return cards_tuple


def cards_for_rank(rank: Rank) -> tuple[Card, ...]:
    rank = Rank(rank)
    return tuple(card for card in ALL_CARDS if card.rank is rank)


__all__ = [
    "ALL_CARDS",
    "CARDS_PER_DECK",
    "DECKS",
    "JOKER_RANKS",
    "NORMAL_CARDS_PER_DECK",
    "NORMAL_RANKS",
    "NON_JOKER_RANKS",
    "PRINTED_RANKS",
    "Rank",
    "Suit",
    "TOTAL_CARDS",
    "WINDOW_RANKS",
    "Card",
    "card_from_id",
    "card_sort_key",
    "cards_for_rank",
    "decode_cards",
    "effective_rank_order",
    "effective_rank_value",
    "encode_card",
    "encode_cards",
    "full_deck",
    "rank_counts",
    "rank_order_value",
    "sort_cards",
    "suit_counts",
    "validate_level_rank",
    "validate_physical_cards",
]
