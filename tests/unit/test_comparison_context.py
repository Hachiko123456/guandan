"""Comparison context slicing (F5) and paired bootcamp-endpoint (F6)."""
from __future__ import annotations

from guandan_marvel.comparison.plan import (
    BOOTCAMP_FINAL_CANDIDATE_ARM, BOOTCAMP_FINAL_MIRROR_ARM, build_comparison_deals,
    default_arms, enumerate_tasks, expected_games,
)
from guandan_marvel.comparison.stats import (
    build_summary, context_slices, deal_contexts,
)
from guandan_marvel.evaluation.manifests import build_deals_manifest

DEALS = build_deals_manifest(
    counts={"training": 0, "selection": 0, "final": 4},
    bases={"training": 0, "selection": 0, "final": 1600000000},
    level_ranks=(15, 15, 4, 9),
    leader_seats=(0, 2, 0, 3),
    previous_results=(None,),
)
DEAL_IDS = [str(row["deal_id"]) for row in DEALS["datasets"]["final"]]

WIN_TABLE = {
    BOOTCAMP_FINAL_CANDIDATE_ARM: {
        DEAL_IDS[0]: [True, True, True, True], DEAL_IDS[1]: [True, True, True, False],
        DEAL_IDS[2]: [True, True, False, False], DEAL_IDS[3]: [True, False, False, False],
    },
    BOOTCAMP_FINAL_MIRROR_ARM: {
        DEAL_IDS[0]: [False, False, False, False], DEAL_IDS[1]: [True, True, False, False],
        DEAL_IDS[2]: [True, True, False, False], DEAL_IDS[3]: [True, True, False, False],
    },
}


def _records(arm_id: str) -> list[dict[str, object]]:
    rows = []
    for deal_id, wins in WIN_TABLE[arm_id].items():
        for rotation, win in enumerate(wins):
            rows.append({
                "status": "complete", "arm_id": arm_id, "deal_id": deal_id,
                "seat_rotation": rotation, "candidate_win": win,
                "candidate_team_reward": 1.0 if win else -1.0,
                "runtime_seconds": 10.0, "illegal_count": 0, "truncated_count": 0,
                "outcome_class": "DOUBLE_DOWN",
            })
    return rows


def _control() -> dict[str, object]:
    arms = []
    for arm in default_arms(with_bootcamp_final=True):
        arm = dict(arm)
        arm["deals"] = 4
        arms.append(arm)
    return {
        "schema": "guandan-marvel-comparison-control-v1",
        "hash": "0" * 64,
        "comparison_id": "test",
        "arms": arms,
    }


def test_deal_contexts_maps_rows_and_rejects_duplicates():
    contexts = deal_contexts(DEALS)
    assert contexts[DEAL_IDS[0]] == {"level_rank": 15, "leader_seat": 0}
    assert contexts[DEAL_IDS[3]] == {"level_rank": 9, "leader_seat": 3}
    duplicated = {"datasets": {"final": DEALS["datasets"]["final"] + [DEALS["datasets"]["final"][0]]}}
    try:
        deal_contexts(duplicated)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:  # pragma: no cover
        raise AssertionError("duplicate deal ids must be rejected")


def test_context_slices_split_full_level15_indistribution_and_ood():
    rows = _records(BOOTCAMP_FINAL_CANDIDATE_ARM)
    slices = context_slices(rows, deal_contexts(DEALS))
    assert slices["full"]["games"] == 16 and slices["full"]["win_rate"] == 10 / 16
    assert slices["level_rank_15"]["games"] == 8 and slices["level_rank_15"]["win_rate"] == 7 / 8
    assert slices["level_rank_15_leader_0"]["games"] == 4 and slices["level_rank_15_leader_0"]["win_rate"] == 1.0
    assert slices["out_of_distribution"]["games"] == 12 and slices["out_of_distribution"]["win_rate"] == 6 / 12
    assert slices["full"]["clusters"] == 4
    assert slices["level_rank_15_leader_0"]["clusters"] == 1
    assert slices["out_of_distribution"]["clusters"] == 3


def test_build_summary_primary_endpoint_is_paired_bootcamp_arm():
    control = _control()
    records = _records(BOOTCAMP_FINAL_CANDIDATE_ARM) + _records(BOOTCAMP_FINAL_MIRROR_ARM)
    summary = build_summary(control, records, {}, DEALS)
    arm = summary["arms"][BOOTCAMP_FINAL_CANDIDATE_ARM]
    assert arm["by_context"]["level_rank_15_leader_0"]["win_rate"] == 1.0
    paired = summary["paired"][f"{BOOTCAMP_FINAL_CANDIDATE_ARM}_minus_{BOOTCAMP_FINAL_MIRROR_ARM}"]
    assert paired["paired_deals"] == 4
    assert paired["planned_deals"] == 4
    assert paired["complete"] is True
    assert all(int(value) == 0 for value in paired["problems"].values())
    assert all(int(value) == 0 for value in paired["record_counts"].values())
    assert paired["mean_delta"] is not None and abs(paired["mean_delta"] - 0.25) < 1e-12
    assert paired["bootstrap"]["low"] is not None and paired["bootstrap"]["high"] is not None
    assert summary["primary_endpoint"] == BOOTCAMP_FINAL_CANDIDATE_ARM
    assert summary["primary"]["verdict"] in {"bootcamp_final_stronger", "inconclusive_at_this_sample_size"}
    assert summary["primary"]["ci95_low"] is not None
    assert summary["primary"]["ci95_high"] is not None


def test_default_arms_without_bootcamp_checkpoint_stays_legacy():
    legacy = default_arms()
    assert BOOTCAMP_FINAL_CANDIDATE_ARM not in {arm["id"] for arm in legacy}
    with_bootcamp = default_arms(with_bootcamp_final=True)
    ids = [arm["id"] for arm in with_bootcamp]
    assert ids[:2] == [BOOTCAMP_FINAL_CANDIDATE_ARM, BOOTCAMP_FINAL_MIRROR_ARM]
    assert expected_games({"arms": with_bootcamp}) == 5632
    tasks = enumerate_tasks({"arms": with_bootcamp}, build_comparison_deals(count=256))
    assert len(tasks) == 5632
