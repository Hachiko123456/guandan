from __future__ import annotations

import random

import numpy as np

from guandan.action_state import StepwiseActionState
from guandan.cards import Rank


def test_randomized_canonical_prefixes_are_unique() -> None:
    rng = random.Random(20261004)
    levels = (Rank.TWO, Rank.FIVE, Rank.SIX, Rank.KING)
    for _ in range(4):
        seed = rng.randrange(10_000)
        level_rank = levels[rng.randrange(len(levels))]
        protocol = StepwiseActionState.deal(seed=seed, level_rank=level_rank)
        candidates = protocol._candidate_sequences()
        sequences = [sequence for sequence, _action in candidates]
        assert sequences and len(sequences) == len(set(sequences))
        assert protocol.legal_next_tokens().dtype == np.int32
