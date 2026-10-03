"""Property-style conservation checks for the physical-card layer."""

from collections import Counter
import random

from guandan.cards import ALL_CARDS, TOTAL_CARDS, decode_cards, encode_cards, full_deck


def test_full_deck_round_trip_preserves_every_physical_card() -> None:
    cards = full_deck()
    assert decode_cards(encode_cards(cards)) == cards
    assert len({card.card_id for card in cards}) == TOTAL_CARDS


def test_random_hand_round_trip_and_conservation() -> None:
    rng = random.Random(20261003)
    deck_ids = [card.card_id for card in ALL_CARDS]
    for hand_size in range(0, 109):
        for _ in range(3):
            selected_ids = rng.sample(deck_ids, hand_size)
            decoded = decode_cards(selected_ids)
            assert encode_cards(decoded) == tuple(selected_ids)
            assert len(decoded) + len(set(deck_ids) - set(selected_ids)) == TOTAL_CARDS
            assert Counter(card.card_id for card in decoded) == Counter(selected_ids)


def test_partitioning_the_deck_into_four_guandan_hands_preserves_identity() -> None:
    rng = random.Random(7)
    shuffled = list(ALL_CARDS)
    rng.shuffle(shuffled)
    # A two-deck GuanDan deal starts with four 27-card hands.
    partitions = [shuffled[index : index + 27] for index in range(0, TOTAL_CARDS, 27)]
    assert len(partitions) == 4
    assert [len(part) for part in partitions] == [27, 27, 27, 27]

    flattened = [card for part in partitions for card in part]
    assert set(card.card_id for card in flattened) == set(card.card_id for card in ALL_CARDS)
    assert len(flattened) == TOTAL_CARDS
    assert len({card.card_id for card in flattened}) == TOTAL_CARDS
