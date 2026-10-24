# SPDX-License-Identifier: MIT
"""DMC stage 2: complete candidate actions and GD-DMC-ACTION-1 features.

The enumerator is a thin, read-only wrapper over the accepted A03 stepwise
protocol (``StepwiseActionState._candidate_sequences``), which itself wraps the
A02 rules engine.  No rules, token layout or candidate ordering is redefined
here: ``guandan.action_state`` / ``guandan.round`` stay the single source of
truth, and each DMC decision is one complete candidate action (a play, a pass,
or an exchange card) rather than a token sequence.

This module contains no model, sampling or training logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from guandan.action_state import COMMIT_TOKEN, FAMILY_ORDER, PASS_TOKEN, StepwiseActionState
from guandan.cards import Card, PRINTED_RANKS, Rank, Suit
from guandan.combos import CombinationKind
from guandan_marvel.dmc import DMC_ACTION_FEATURE_VERSION

# One source of truth for the versioned layout: ``DMCRuntimeConfig`` defaults
# to the same constant, so a feature-version bump cannot drift between them.
ACTION_FEATURE_VERSION = DMC_ACTION_FEATURE_VERSION
ACTION_FEATURE_DIM = 64

#: Printed-rank order used by both 15-bin histograms: 13 printed ranks followed
#: by the two jokers (identical to ``guandan.action_state.RANK_ORDER``).
RANK_HISTOGRAM_ORDER: tuple[Rank, ...] = (*PRINTED_RANKS, Rank.SMALL_JOKER, Rank.BIG_JOKER)

_FAMILY_COUNT = len(FAMILY_ORDER)  # 10 combination families
_SUIT_COUNT = len(Suit)  # CLUBS, DIAMONDS, HEARTS, SPADES, JOKER


@dataclass(frozen=True, slots=True)
class CandidateAction:
    """One complete legal action at a DMC decision point.

    ``tokens`` is the unique A03 canonical token sequence for the action; it is
    kept so a candidate can be committed through the existing stepwise
    protocol (``StepwiseActionState.step``) without any token-level decision.

    ``family`` uses the token-codec family order: ``0`` for a non-combination
    action (PASS, tribute or return) and ``1..10`` for
    ``guandan.action_state.FAMILY_ORDER`` (single, pair, triple, full house,
    straight, pair sequence, triple sequence, rank bomb, straight flush, four
    kings).  ``is_pass`` separates a PASS from exchange actions, which share
    family slot 0.

    ``comparison_rank`` is the declaration's ``comparison_rank`` as an integer
    (``Rank`` value); it is ``0`` for PASS/tribute/return.  ``card_ids`` are the
    physical card IDs in ascending order.  ``wild_assignments`` is the sorted
    ``(card_id, represented_rank, represented_suit)`` tuple for every level
    wild used by the declaration, including wilds used naturally.
    """

    tokens: tuple[int, ...]
    family: int
    comparison_rank: int
    card_ids: tuple[int, ...]
    wild_assignments: tuple[tuple[int, int, int], ...]
    is_pass: bool


def _candidate_from_action(tokens: tuple[int, ...], action: Any) -> CandidateAction:
    kind = action.kind
    declaration = action.declaration
    is_pass = kind == "pass"
    if isinstance(kind, CombinationKind):
        family = FAMILY_ORDER.index(kind) + 1
    else:
        family = 0
    if declaration is None:
        comparison_rank = 0
        wild_assignments: tuple[tuple[int, int, int], ...] = ()
    else:
        comparison_rank = int(declaration.comparison_rank)
        wild_assignments = tuple(
            (int(card_id), int(rank), int(suit))
            for card_id, (rank, suit) in sorted(declaration.wild_assignments.items())
        )
    card_ids = tuple(sorted(int(card.card_id) for card in action.cards))
    return CandidateAction(
        tokens=tuple(int(token) for token in tokens),
        family=family,
        comparison_rank=comparison_rank,
        card_ids=card_ids,
        wild_assignments=wild_assignments,
        is_pass=is_pass,
    )


def enumerate_candidates(stepwise_state: StepwiseActionState) -> tuple[CandidateAction, ...]:
    """Enumerate every legal complete action, stably sorted by token sequence.

    A pure, read-only projection of ``stepwise_state._candidate_sequences()``:
    the same accepted enumerator used by the stepwise protocol, including its
    duplicate/ambiguity checks and its per-instance memoization (when the state
    is a cached protocol).  The function does not mutate the round state, the
    prefix or the RNG, and two calls on an unchanged state return equal tuples.
    """
    if not isinstance(stepwise_state, StepwiseActionState):
        raise TypeError("stepwise_state must be a StepwiseActionState")
    candidates = [
        _candidate_from_action(tokens, action)
        for tokens, action in stepwise_state._candidate_sequences()
    ]
    candidates.sort(key=lambda candidate: candidate.tokens)
    return tuple(candidates)


def _context_fields(context: Any) -> tuple[Rank, Any]:
    if isinstance(context, Mapping):
        if "level_rank" not in context:
            raise KeyError("context must provide level_rank")
        level_rank = context["level_rank"]
        current_winning = context.get("current_winning")
    else:
        if not hasattr(context, "level_rank"):
            raise TypeError("context must provide level_rank and current_winning")
        level_rank = context.level_rank
        current_winning = getattr(context, "current_winning", None)
    try:
        level = level_rank if isinstance(level_rank, Rank) else Rank(int(level_rank))
    except (TypeError, ValueError) as exc:
        raise ValueError("context.level_rank must be a non-joker Rank") from exc
    if level in {Rank.SMALL_JOKER, Rank.BIG_JOKER}:
        raise ValueError("context.level_rank must be a non-joker Rank")
    return level, current_winning


def encode_action_features(candidate: CandidateAction, context: Any) -> np.ndarray:
    """Encode one candidate as the 64-dimensional GD-DMC-ACTION-1 vector.

    ``context`` must expose ``level_rank`` and ``current_winning`` (an object
    such as ``RoundState``, or a mapping with those keys).  The layout is
    specified dimension by dimension in ``docs/dmc_action_encoding.md``.
    """
    if not isinstance(candidate, CandidateAction):
        raise TypeError("candidate must be a CandidateAction")
    if not 0 <= candidate.family <= _FAMILY_COUNT:
        raise ValueError("candidate.family must lie in 0..10")
    level_rank, current_winning = _context_fields(context)

    features = np.zeros(ACTION_FEATURE_DIM, dtype=np.float32)
    features[0] = 1.0 if candidate.is_pass else 0.0
    features[1 + candidate.family] = 1.0

    wild_by_id: dict[int, Rank] = {}
    for card_id, rank, _suit in candidate.wild_assignments:
        wild_by_id[int(card_id)] = Rank(int(rank))

    claimed_counts = np.zeros(len(RANK_HISTOGRAM_ORDER), dtype=np.float32)
    printed_counts = np.zeros(len(RANK_HISTOGRAM_ORDER), dtype=np.float32)
    suit_counts = np.zeros(_SUIT_COUNT, dtype=np.float32)
    wild_count = 0
    for card_id in candidate.card_ids:
        card = Card(int(card_id))
        printed_counts[RANK_HISTOGRAM_ORDER.index(card.rank)] += 1.0
        claimed_counts[RANK_HISTOGRAM_ORDER.index(wild_by_id.get(card.card_id, card.rank))] += 1.0
        suit_counts[int(card.suit)] += 1.0
        wild_count += int(card.is_level_wild(level_rank))

    # The locked table reserves indices 12..27 for the 15-bin claim histogram;
    # index 27 is block padding and stays zero (64-dim layout is fixed).
    features[12:27] = claimed_counts / 4.0
    features[28] = len(candidate.card_ids) / 27.0
    features[29] = wild_count / 2.0
    features[30] = candidate.comparison_rank / 15.0
    # Legal candidates returned by the engine only contain plays that strictly
    # beat a live trick; a PASS or a lead play (no current winning declaration)
    # is 0.  See the documented invariant in docs/dmc_action_encoding.md.
    features[31] = 1.0 if current_winning is not None and 1 <= candidate.family <= _FAMILY_COUNT else 0.0
    features[32:47] = printed_counts / 4.0
    features[47:52] = suit_counts / 27.0
    features[52] = 0.5 if wild_count > 0 else 0.0
    # 53..63 stay zero: reserved version-alignment slots.
    return features


__all__ = [
    "ACTION_FEATURE_DIM",
    "ACTION_FEATURE_VERSION",
    "CandidateAction",
    "RANK_HISTOGRAM_ORDER",
    "encode_action_features",
    "enumerate_candidates",
]
