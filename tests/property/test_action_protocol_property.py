from __future__ import annotations

from guandan.action_state import StepwiseActionState, canonical_tokens
from guandan.cards import Rank, cards_for_rank, full_deck
from guandan.round import enumerate_legal_actions
from guandan.state import RoundState


def partition(hand):
    used = {card.card_id for card in hand}
    rest = [card for card in full_deck() if card.card_id not in used]
    hands = [list(hand), [], [], []]
    for index, card in enumerate(rest):
        hands[1 + index % 3].append(card)
    return hands


def test_property_all_reduced_actions_have_live_prefixes():
    hand = list(cards_for_rank(Rank.THREE)[:2]) + list(cards_for_rank(Rank.FOUR)[:3]) + list(cards_for_rank(Rank.FIVE)[:2])
    state = RoundState.from_hands(partition(hand), level_rank=Rank.SIX)
    for action in enumerate_legal_actions(state):
        protocol = StepwiseActionState(state.clone())
        for token in canonical_tokens(action):
            assert int(token) in set(int(value) for value in protocol.legal_tokens())
            protocol.step(int(token))
        assert protocol.committed_step == 1


def test_property_prefix_commit_does_not_duplicate_actions():
    hand = list(cards_for_rank(Rank.THREE)[:2]) + list(cards_for_rank(Rank.FOUR)[:3])
    state = RoundState.from_hands(partition(hand))
    sequences = [canonical_tokens(action) for action in enumerate_legal_actions(state)]
    assert len(sequences) == len(set(sequences))
