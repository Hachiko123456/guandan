"""Phase-5 aligned policy diagnostics with hand-computed references."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from guandan_marvel.phase5.diagnostics import (
    ACTION_UNIT_FULL_MACRO, ACTION_UNIT_RESTRICTED, aggregate_diagnostic,
    aligned_policy_comparison, bucket_key, decide_augmentation, positions_rank,
    restricted_candidate_kl,
)


def _log(probs):
    return torch.log(torch.tensor(probs, dtype=torch.float64))


def test_aligned_kl_matches_hand_computed_reference_and_requires_alignment():
    base = _log([0.5, 0.3, 0.2])
    other = _log([0.2, 0.5, 0.3])
    permutation = (1, 0, 2)  # q(a_i) = pi(P_T(a_i)) = other[perm[i]] = [0.5,0.2,0.3]
    expected = 0.5 * math.log(0.5 / 0.5) + 0.3 * math.log(0.3 / 0.2) + 0.2 * math.log(0.2 / 0.3)
    comparison = aligned_policy_comparison(base, other, permutation)
    assert comparison.action_unit == ACTION_UNIT_RESTRICTED
    assert comparison.kl == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert comparison.max_prob_diff == pytest.approx(0.1, abs=1e-12)
    assert comparison.mean_prob_diff == pytest.approx(0.2 / 3.0, abs=1e-12)
    assert comparison.argmax_agreement == 1
    assert comparison.candidate_count == 3
    # A naive unaligned comparison is a different quantity; the permutation is
    # load-bearing, not decorative.
    misaligned = aligned_policy_comparison(base, other, (0, 1, 2))
    assert misaligned.kl != pytest.approx(expected, abs=1e-9)
    assert restricted_candidate_kl(base, other, permutation) == pytest.approx(expected)


def test_argmax_tie_rule_is_declared_and_deterministic():
    base = _log([0.4, 0.4, 0.2])
    other = _log([0.4, 0.4, 0.2])
    canonical = aligned_policy_comparison(base, other, (1, 0, 2))
    assert canonical.argmax_agreement == 1
    alternate = aligned_policy_comparison(
        base, other, (1, 0, 2), tie_rule_index=(1, 0, 2),
    )
    assert alternate.argmax_agreement == 1
    disagree = aligned_policy_comparison(_log([0.6, 0.4]), _log([0.4, 0.6]), (0, 1))
    assert disagree.argmax_agreement == 0
    assert positions_rank((1, 0, 2), 0) == 1


def test_padding_and_illegal_candidates_do_not_participate():
    base = _log([0.7, 0.2, 0.1])
    other = _log([0.7, 0.2, 0.1])
    comparison = aligned_policy_comparison(
        base, other, (0, 1, 2), legal=torch.tensor([True, True, False]),
    )
    assert comparison.candidate_count == 2
    assert comparison.kl == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        aligned_policy_comparison(
            base, other, (0, 1, 2), legal=torch.tensor([False, False, False]),
        )
    with pytest.raises(ValueError):
        aligned_policy_comparison(base, other, (0, 0, 2))
    with pytest.raises(ValueError):
        aligned_policy_comparison(base, other, (0, 1))
    with pytest.raises(ValueError):
        aligned_policy_comparison(base, other, (0, 1, 2), action_unit="next_token")


def test_zero_probability_is_fail_closed_not_silently_clamped():
    with pytest.raises(ValueError):
        aligned_policy_comparison(
            torch.tensor([0.0, float("-inf")]),
            torch.tensor([0.0, float("-inf")]),
            (0, 1),
        )


def test_action_unit_labels_are_explicit():
    base = _log([0.6, 0.4])
    other = _log([0.4, 0.6])
    comparison = aligned_policy_comparison(
        base, other, (1, 0), action_unit=ACTION_UNIT_FULL_MACRO,
    )
    assert comparison.action_unit == ACTION_UNIT_FULL_MACRO
    restricted = aligned_policy_comparison(base, other, (1, 0))
    assert restricted.action_unit == ACTION_UNIT_RESTRICTED


def _record(kl, agreement=1, deal="deal-1", phase="PLAY"):
    return {
        "kl": float(kl),
        "argmax_agreement": int(agreement),
        "mean_prob_diff": 0.01,
        "max_prob_diff": 0.02,
        "action_unit": ACTION_UNIT_RESTRICTED,
        "deal_id": deal,
        "bucket": bucket_key(
            phase=phase, action_kind=phase, level_rank=7,
            leader_seat=1, tribute_mode="none",
        ),
    }


def test_aggregate_and_decision_thresholds_are_pre_registered():
    records = [_record(0.1), _record(0.1, deal="deal-2")]
    aggregate = aggregate_diagnostic(records)
    assert aggregate["sample_count"] == 2
    assert aggregate["deal_count"] == 2
    assert aggregate["mean_kl"] == pytest.approx(0.1)
    assert aggregate["buckets"]["phase"]["PLAY"]["count"] == 2
    significance = {
        "min_macro_samples": 2,
        "min_deals": 2,
        "mean_restricted_kl_threshold": 0.05,
        "argmax_disagreement_threshold": 0.1,
    }
    assert decide_augmentation(aggregate, significance=significance)["status"] == "SIGNIFICANT"
    low = aggregate_diagnostic([_record(0.01), _record(0.01, deal="deal-2")])
    assert decide_augmentation(low, significance=significance)["status"] == "NOT_SIGNIFICANT"
    insufficient = aggregate_diagnostic([_record(0.5)])
    decision = decide_augmentation(insufficient, significance=significance)
    assert decision["status"] == "INCONCLUSIVE"
    assert "insufficient_macro_samples" in decision["reasons"]


def test_non_finite_kl_is_reported_as_inconclusive_not_laundered():
    records = [_record(0.1), _record(float("inf"), deal="deal-2")]
    aggregate = aggregate_diagnostic(records)
    assert aggregate["non_finite_kl_count"] == 1
    decision = decide_augmentation(
        aggregate,
        significance={
            "min_macro_samples": 1, "min_deals": 1,
            "mean_restricted_kl_threshold": 0.0,
            "argmax_disagreement_threshold": 0.0,
        },
    )
    assert decision["status"] == "INCONCLUSIVE"
    assert "non_finite_kl" in decision["reasons"]
