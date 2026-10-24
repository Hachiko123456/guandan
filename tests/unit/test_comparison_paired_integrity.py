"""Paired-endpoint integrity: rotation completeness, duplicates, mismatches, invalid records."""
from __future__ import annotations

from guandan_marvel.comparison.plan import (
    BOOTCAMP_FINAL_CANDIDATE_ARM, BOOTCAMP_FINAL_MIRROR_ARM, build_comparison_deals,
)
from guandan_marvel.comparison.stats import (
    build_summary, paired_delta, paired_primary_decision,
)

A = BOOTCAMP_FINAL_CANDIDATE_ARM
B = BOOTCAMP_FINAL_MIRROR_ARM


def _record(
    arm: str, deal_id: str, rotation: int, win: bool = True, *,
    status: str = "complete", illegal: int = 0, truncated: int = 0,
) -> dict[str, object]:
    return {
        "status": status, "arm_id": arm, "deal_id": deal_id, "seat_rotation": rotation,
        "candidate_win": win, "illegal_count": illegal, "truncated_count": truncated,
        "candidate_team_reward": 1.0 if win else -1.0, "runtime_seconds": 1.0,
    }


def _full(arm: str, deal_id: str, wins=(True, True, True, False)) -> list[dict[str, object]]:
    return [_record(arm, deal_id, rotation, wins[rotation]) for rotation in range(4)]


def _summary(records, *, planned: int, manifest_deals=None):
    deals = build_comparison_deals(count=planned) if manifest_deals is None else manifest_deals
    control = {
        "schema": "guandan-marvel-comparison-control-v1", "hash": "0" * 64,
        "comparison_id": "paired-integrity",
        "arms": [
            {"id": A, "priority": 0, "deals": planned, "candidate_kind": "bootcamp_final",
             "opponent_kind": "m09", "candidate_label": "new", "opponent_label": "m09"},
            {"id": B, "priority": 0, "deals": planned, "candidate_kind": "m09",
             "opponent_kind": "bootcamp_final", "candidate_label": "m09", "opponent_label": "new"},
        ],
    }
    return build_summary(control, records, {}, deals)


def test_one_of_four_vs_one_of_four_is_incomplete_evidence():
    deals = build_comparison_deals(count=1)
    deal_id = str(deals["datasets"]["final"][0]["deal_id"])
    records = [_record(A, deal_id, 0), _record(B, deal_id, 0)]
    paired = paired_delta(records[:1], records[1:], label="x", expected_deals=[deal_id])
    assert paired["paired_deals"] == 0
    assert paired["complete"] is False
    assert paired["problems"]["incomplete_deals"] == 1
    assert paired["mean_delta"] is None
    assert paired_primary_decision(paired)["verdict"] == "incomplete_evidence"
    summary = _summary(records, planned=1, manifest_deals=deals)
    assert summary["primary_endpoint"] == A
    assert summary["primary"]["verdict"] == "incomplete_evidence"
    assert summary["primary"]["mean_delta"] is None
    assert summary["primary"]["ci95_low"] is None


def test_three_of_four_vs_four_of_four_is_mismatched():
    records = [
        *[_record(A, "d0", rotation) for rotation in (0, 1, 2)],
        *_full(B, "d0"),
    ]
    paired = paired_delta(records[:3], records[3:], label="x", expected_deals=["d0"])
    assert paired["paired_deals"] == 0
    assert paired["complete"] is False
    assert paired["problems"]["mismatched_deals"] == 1
    assert paired_primary_decision(paired)["verdict"] == "incomplete_evidence"


def test_duplicate_rotation_is_rejected():
    arm_a = [_record(A, "d0", 0), _record(A, "d0", 0), _record(A, "d0", 1),
             _record(A, "d0", 2), _record(A, "d0", 3)]
    arm_b = _full(B, "d0")
    paired = paired_delta(arm_a, arm_b, label="x", expected_deals=["d0"])
    assert paired["paired_deals"] == 0
    assert paired["complete"] is False
    assert paired["record_counts"]["duplicate_records"] == 1
    assert paired["problems"]["duplicate_rotation_deals"] == 1
    assert paired["problems"]["invalid_deals"] == 1
    assert paired_primary_decision(paired)["verdict"] == "incomplete_evidence"


def test_rotation_sets_must_match_between_arms():
    arm_a = _full(A, "d0")
    arm_b = [_record(B, "d0", 0), _record(B, "d0", 0), _record(B, "d0", 1), _record(B, "d0", 2)]
    paired = paired_delta(arm_a, arm_b, label="x", expected_deals=["d0"])
    assert paired["paired_deals"] == 0
    assert paired["complete"] is False
    assert paired["problems"]["mismatched_deals"] == 1
    assert paired["record_counts"]["duplicate_records"] == 1
    assert paired_primary_decision(paired)["verdict"] == "incomplete_evidence"


def test_error_and_incomplete_records_never_enter_the_delta():
    arm_a = _full(A, "d0")
    arm_b = [
        _record(B, "d0", 0), _record(B, "d0", 1, status="error"),
        _record(B, "d0", 2), _record(B, "d0", 3, status="incomplete"),
    ]
    paired = paired_delta(arm_a, arm_b, label="x", expected_deals=["d0"])
    assert paired["paired_deals"] == 0
    assert paired["mean_delta"] is None
    assert paired["complete"] is False
    assert paired["record_counts"]["error_records"] == 1
    assert paired["record_counts"]["incomplete_records"] == 1
    assert paired_primary_decision(paired)["verdict"] == "incomplete_evidence"


def test_fewer_paired_deals_than_planned_is_incomplete_evidence():
    deals = build_comparison_deals(count=2)
    deal_ids = [str(row["deal_id"]) for row in deals["datasets"]["final"]]
    records = [*_full(A, deal_ids[0]), *_full(B, deal_ids[0])]
    control = {
        "schema": "guandan-marvel-comparison-control-v1", "hash": "0" * 64,
        "comparison_id": "paired-integrity",
        "arms": [
            {"id": A, "priority": 0, "deals": 2, "candidate_kind": "bootcamp_final",
             "opponent_kind": "m09", "candidate_label": "new", "opponent_label": "m09"},
            {"id": B, "priority": 0, "deals": 2, "candidate_kind": "m09",
             "opponent_kind": "bootcamp_final", "candidate_label": "m09", "opponent_label": "new"},
        ],
    }
    summary = build_summary(control, records, {A: 4}, deals)
    endpoint = summary["paired"][f"{A}_minus_{B}"]
    assert endpoint["planned_deals"] == 2
    assert endpoint["paired_deals"] == 1
    assert endpoint["problems"]["missing_deals"] == 1
    assert endpoint["complete"] is False
    assert summary["arms"][A]["skipped_due_to_time"] == 4
    assert summary["primary"]["verdict"] == "incomplete_evidence"


def test_illegal_or_truncated_games_are_excluded():
    arm_a = _full(A, "d0")
    arm_b = _full(B, "d0")
    arm_a[1]["illegal_count"] = 1
    arm_b[2]["truncated_count"] = 1
    paired = paired_delta(arm_a, arm_b, label="x", expected_deals=["d0"])
    assert paired["paired_deals"] == 0
    assert paired["mean_delta"] is None
    assert paired["complete"] is False
    assert paired["record_counts"]["illegal_records"] == 1
    assert paired["record_counts"]["truncated_records"] == 1
    assert paired["problems"]["invalid_deals"] == 1
    assert paired_primary_decision(paired)["verdict"] == "incomplete_evidence"


def test_full_256_deal_plan_yields_a_complete_positive_endpoint():
    deals = build_comparison_deals(count=256)
    deal_ids = [str(row["deal_id"]) for row in deals["datasets"]["final"]]
    records = []
    for deal_id in deal_ids:
        records.extend(_full(A, deal_id, wins=(True, True, True, False)))
        records.extend(_full(B, deal_id, wins=(True, False, True, False)))
    summary = _summary(records, planned=256, manifest_deals=deals)
    endpoint = summary["paired"][f"{A}_minus_{B}"]
    assert endpoint["planned_deals"] == 256
    assert endpoint["paired_deals"] == 256
    assert endpoint["complete"] is True
    assert all(int(value) == 0 for value in endpoint["problems"].values())
    assert all(int(value) == 0 for value in endpoint["record_counts"].values())
    assert endpoint["mean_delta"] == 0.25
    assert endpoint["bootstrap"]["clusters"] == 256
    assert summary["primary"]["verdict"] == "bootcamp_final_stronger"
    assert summary["primary"]["ci95_low"] > 0.0
    assert summary["primary"]["ci95_high"] == 0.25
