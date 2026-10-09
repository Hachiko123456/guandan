from __future__ import annotations

import math

from guandan_marvel.evaluation.stats import clustered_mean_ci95, summarize_records


def _record(opponent, deal, rotation, win):
    return {
        "status": "complete", "opponent_id": opponent, "deal_id": deal,
        "seat_rotation": rotation, "base_deal_sha256": f"{int(deal[-1]):064x}",
        "candidate_team": rotation % 2, "candidate_win": win,
        "candidate_team_reward": 1.0 if win else -1.0,
        "outcome_class": ("DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST")[rotation % 3],
        "illegal_count": 0, "truncated_count": 0, "runtime_seconds": 1.0,
    }


def test_cluster_ci_uses_deals_not_four_rotations_as_independent_samples():
    ci = clustered_mean_ci95([0.0, 0.5, 0.5, 1.0])
    expected_se = math.sqrt((0.25 + 0 + 0 + 0.25) / 3) / 2
    assert ci["clusters"] == 4
    assert ci["mean"] == 0.5
    assert math.isclose(ci["standard_error"], expected_se)
    assert math.isclose(ci["low"], max(0.0, 0.5 - 3.182 * expected_se), rel_tol=1e-12)
    assert math.isclose(ci["high"], min(1.0, 0.5 + 3.182 * expected_se), rel_tol=1e-12)


def test_single_deal_cluster_is_inconclusive_not_fake_n4_ci():
    records = [_record("champion", "deal0", rotation, rotation < 2) for rotation in range(4)]
    summary = summarize_records(records, expected_opponents=["champion"])
    result = summary["per_opponent"]["champion"]
    assert result["eval/win_rate"] == 0.5
    assert result["eval/cluster_count"] == 1
    assert result["eval/clustered_ci95_low"] is None
    assert result["clustered_ci95"]["conclusive"] is False


def test_rotation_coverage_detects_missing_rotation():
    records = [_record("champion", "deal0", rotation, True) for rotation in range(3)]
    summary = summarize_records(records, expected_opponents=["champion"])
    assert summary["overall"]["complete"] is False
    assert summary["seat_rotation_coverage"]["complete"] is False
    assert "rotations" in summary["seat_rotation_coverage"]["errors"][0]
