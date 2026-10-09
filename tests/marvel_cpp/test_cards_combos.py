from __future__ import annotations

import pytest

from guandan.cards import Card, Rank, Suit, effective_rank_value as py_effective_rank_value
from guandan.combos import compare_combinations as py_compare
from guandan.combos import recognize_combinations as py_recognize
from guandan_marvel.testing.complex_actions import DIRECTED_ACTION_CASES
from nanopokerlib import guandan as cpp


def _py_signature(declaration):
    return (
        declaration.kind.value,
        tuple(card.card_id for card in declaration.cards),
        int(declaration.comparison_rank),
        tuple(declaration.comparison_key),
        tuple(
            (card_id, int(rank), int(suit), card_id in declaration.natural_wild_ids)
            for card_id, (rank, suit) in sorted(declaration.wild_assignments.items())
        ),
        tuple(declaration.natural_wild_ids),
    )


def _cpp_signature(declaration):
    return (
        declaration.kind,
        tuple(declaration.card_ids),
        declaration.comparison_rank,
        tuple(declaration.comparison_key),
        tuple(
            (item.card_id, int(item.rank), int(item.suit), item.natural)
            for item in declaration.wild_assignments
        ),
        tuple(declaration.natural_wild_ids),
    )


def _cpp_decl(card_ids, kind, level=Rank.FOUR):
    matches = [item for item in cpp.recognize_combinations(card_ids, int(level)) if item.kind == kind]
    assert matches, (card_ids, kind)
    return matches[0]


def test_all_108_physical_cards_and_all_13_level_contexts_match_oracle():
    for level in tuple(Rank(value) for value in range(int(Rank.THREE), int(Rank.TWO) + 1)):
        for card_id in range(108):
            py_card = Card(card_id)
            info = cpp.card_info(card_id, int(level))
            assert info == {
                "card_id": card_id,
                "deck_index": py_card.deck_index,
                "deck_offset": py_card.deck_offset,
                "rank": int(py_card.rank),
                "suit": int(py_card.suit),
                "is_joker": py_card.is_joker,
                "is_level_wild": py_card.is_level_wild(level),
                "effective_rank_value": py_effective_rank_value(py_card.rank, level),
            }
        for rank in Rank:
            assert cpp.effective_rank_value(int(rank), int(level)) == py_effective_rank_value(rank, level)


@pytest.mark.parametrize("case", DIRECTED_ACTION_CASES, ids=lambda case: case.name)
def test_all_ten_families_and_wild_declarations_match_oracle(case):
    py_values = {
        _py_signature(item)
        for item in py_recognize(tuple(Card(card_id) for card_id in case.card_ids), level_rank=Rank.FOUR)
    }
    cpp_values = {
        _cpp_signature(item)
        for item in cpp.recognize_combinations(list(case.card_ids), int(Rank.FOUR))
    }
    assert cpp_values == py_values


def test_bomb_order_is_four_five_straight_flush_six_four_kings():
    values = [
        _cpp_decl([0, 1, 2, 3], "rank_bomb"),
        _cpp_decl([0, 1, 2, 3, 54], "rank_bomb"),
        _cpp_decl([0, 4, 8, 12, 16], "straight_flush"),
        _cpp_decl([0, 1, 2, 3, 54, 55], "rank_bomb"),
        _cpp_decl([52, 53, 106, 107], "four_kings"),
    ]
    for left, right in zip(values, values[1:]):
        assert cpp.compare_combinations(left, right) < 0
        assert cpp.compare_combinations(right, left) > 0


def test_ordinary_family_boundary_and_comparison_match_oracle():
    cpp_pair = _cpp_decl([0, 54], "pair")
    cpp_triple = _cpp_decl([4, 5, 58], "triple")
    with pytest.raises(ValueError):
        cpp.compare_combinations(cpp_pair, cpp_triple)

    py_pair = next(item for item in py_recognize((Card(0), Card(54)), level_rank=Rank.FOUR) if item.kind.value == "pair")
    py_pair_high = next(item for item in py_recognize((Card(4), Card(58)), level_rank=Rank.FOUR) if item.kind.value == "pair")
    cpp_pair_high = _cpp_decl([4, 58], "pair")
    assert cpp.compare_combinations(cpp_pair, cpp_pair_high) == py_compare(py_pair, py_pair_high) == -1


def test_duplicate_and_out_of_range_physical_ids_are_rejected_without_coercion():
    assert cpp.recognize_combinations([0, 0], int(Rank.TWO)) == []
    assert cpp.recognize_combinations([-1], int(Rank.TWO)) == []
    assert cpp.recognize_combinations([108], int(Rank.TWO)) == []
