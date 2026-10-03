from __future__ import annotations

import random

from guandan.cards import Rank
from guandan.combos import CombinationKind
from guandan.round import apply_action
from guandan.state import CommittedAction, RoundState


def test_random_valid_leads_preserve_card_partition() -> None:
    rng = random.Random(20261003)
    for seed in range(10):
        state = RoundState.deal(seed=seed, level_rank=Rank.FIVE)
        player = state.active_seat
        card = state.hands[player][rng.randrange(len(state.hands[player]))]
        apply_action(state, CommittedAction(player, CombinationKind.SINGLE, (card,)))
        ids = [c.card_id for hand in state.hands for c in hand]
        played = [c.card_id for action in state.history for c in action.cards]
        assert len(ids) == len(set(ids))
        assert len(played) == len(set(played))
        assert set(ids) | set(played) == set(range(108))


def test_ten_thousand_seeded_bounded_legal_rounds() -> None:
    """10,000 seeded legal one-trick trajectories preserve card conservation."""
    for seed in range(10_000):
        state = RoundState.deal(seed=seed, level_rank=Rank.FIVE)
        player = state.active_seat
        card = state.hands[player][seed % len(state.hands[player])]
        apply_action(state, CommittedAction(player, CombinationKind.SINGLE, (card,)))
        for _ in range(3):
            if state.done:
                break
            apply_action(state, CommittedAction(state.active_seat, "pass"))
        state.check_invariants()
