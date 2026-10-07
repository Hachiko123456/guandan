"""Enumeration optimizations cannot prune legal physical-card declarations."""
import importlib.util
from pathlib import Path
import random
import pytest
from guandan.cards import ALL_CARDS,Rank,Suit
from guandan.combos import CombinationKind,recognize_combinations
from guandan.round import enumerate_plays,_dk

p=Path(__file__).parents[1]/'reference/round_enumeration.py'
s=importlib.util.spec_from_file_location('round_oracle',p)
oracle=importlib.util.module_from_spec(s);s.loader.exec_module(oracle)


def signatures(ds):
    return [_dk(x) for x in ds]


@pytest.mark.parametrize('level',[Rank.TWO,Rank.FIVE,Rank.ACE])
def test_enumeration_reference_random_hands_and_current(level):
    rng=random.Random(927)
    current_cards=[ALL_CARDS[0],ALL_CARDS[4],ALL_CARDS[8],ALL_CARDS[12],ALL_CARDS[16]]
    targets=list(recognize_combinations([ALL_CARDS[20]],level_rank=level))
    targets+=list(recognize_combinations(current_cards,level_rank=level))
    for size in [9,14,27]:
        for _ in range(3):
            hand=rng.sample(list(ALL_CARDS),size)
            for current in [None,*targets]:
                actual=enumerate_plays(hand,level_rank=level,current=current)
                expected=oracle.enumerate_plays(hand,level_rank=level,current=current)
                assert signatures(actual)==signatures(expected)


@pytest.mark.parametrize('level',[Rank.FIVE,Rank.TWO])
def test_two_wildcards_and_all_suits_keep_substituted_flushes(level):
    wilds=[c for c in ALL_CARDS if c.is_level_wild(level)]
    natural=[c for c in ALL_CARDS if c.rank in [Rank.THREE,Rank.FOUR,Rank.SIX] and c.deck_index==0]
    hand=list(dict.fromkeys(wilds+natural))
    current=recognize_combinations([ALL_CARDS[20]],level_rank=level)[0]
    assert signatures(enumerate_plays(hand,level_rank=level,current=current))==signatures(oracle.enumerate_plays(hand,level_rank=level,current=current))
