# SPDX-License-Identifier: MIT
"""Card-id mapping between the MARVEL engine and FableDan's Botzone encoding.

MARVEL physical ids (guandan.cards): ``id % 54`` offset 0..51 maps to
``PRINTED_RANKS[offset // 4]`` (THREE..ACE,TWO) with
``Suit(offset % 4)`` (CLUBS, DIAMONDS, HEARTS, SPADES); offsets 52/53 are
small/big joker.

FableDan physical ids: ``id % 54`` offset 0..51 maps to rank index 0..12
(A,2,...,K) with suit 0..3 (hearts, diamonds, spades, clubs); offsets 52/53
are small/big joker.  Both encodings use two identical decks (54 cards each).
"""
from __future__ import annotations

from guandan.cards import Rank, Suit

MARVEL_RANK_TO_FD: dict[int, int] = {
    int(Rank.THREE): 2, int(Rank.FOUR): 3, int(Rank.FIVE): 4, int(Rank.SIX): 5,
    int(Rank.SEVEN): 6, int(Rank.EIGHT): 7, int(Rank.NINE): 8, int(Rank.TEN): 9,
    int(Rank.JACK): 10, int(Rank.QUEEN): 11, int(Rank.KING): 12, int(Rank.ACE): 0,
    int(Rank.TWO): 1, int(Rank.SMALL_JOKER): 13, int(Rank.BIG_JOKER): 14,
}

MARVEL_SUIT_TO_FD: dict[int, int] = {
    int(Suit.CLUBS): 3, int(Suit.DIAMONDS): 1, int(Suit.HEARTS): 0, int(Suit.SPADES): 2,
}

FD_RANK_SLOT_TO_MARVEL = (11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
FD_SUIT_TO_MARVEL = (2, 1, 3, 0)

FD_SMALL_JOKER = 52
FD_BIG_JOKER = 53


def _offset(card_id: int) -> tuple[int, int]:
    return divmod(int(card_id), 54)


def marvel_to_fd_card(card_id: int) -> int:
    deck, offset = _offset(card_id)
    if offset >= 52:
        return deck * 54 + offset
    fd_rank = MARVEL_RANK_TO_FD[3 + offset // 4]
    fd_suit = MARVEL_SUIT_TO_FD[offset % 4]
    return deck * 54 + fd_rank * 4 + fd_suit


def fd_to_marvel_card(card_id: int) -> int:
    deck, offset = _offset(card_id)
    if offset >= 52:
        return deck * 54 + offset
    slot = FD_RANK_SLOT_TO_MARVEL[offset // 4]
    suit = FD_SUIT_TO_MARVEL[offset % 4]
    return deck * 54 + slot * 4 + suit


def marvel_to_fd_level(level_rank: int) -> int:
    return MARVEL_RANK_TO_FD[int(level_rank)]


def fd_claim_card(marvel_rank: int, marvel_suit: int) -> int:
    """Physical claim id (deck 0) for a represented rank/suit."""
    fd_rank = MARVEL_RANK_TO_FD[int(marvel_rank)]
    if fd_rank >= 13:
        return FD_SMALL_JOKER if fd_rank == 13 else FD_BIG_JOKER
    fd_suit = MARVEL_SUIT_TO_FD[int(marvel_suit)]
    return fd_rank * 4 + fd_suit


__all__ = [
    "MARVEL_RANK_TO_FD", "MARVEL_SUIT_TO_FD",
    "marvel_to_fd_card", "fd_to_marvel_card", "marvel_to_fd_level", "fd_claim_card",
]
