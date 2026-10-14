from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from guandan_marvel.evaluation.common import payload_hash
from guandan_marvel.evaluation.dual_gpu import (
    DeterministicWorkerGroup,
    FormalWorkerFailure,
    WorkerCommand,
    compare_formal_outputs,
    deterministic_task_shards,
)
from guandan_marvel.evaluation.resumable import (
    IncompleteEvaluationError,
    ResumableEvaluationStore,
    build_resumable_evaluation_plan,
)
from guandan_marvel.kaggle.unified_pipeline import (
    AUTO_CONTINUATION_SCHEMA,
    AutoContinuationConfig,
    continuation_gate,
)


def _plan():
    candidate = {
        "checkpoint_sha256": "a" * 64,
        "backend_identity_hash": "b" * 64,
        "protocol_version": "GD-ACTION-0.2-MARVEL|GD-ENV-0.2-MARVEL",
    }
    matrix = [
        {"id": "champion", "kind": "v2", "checkpoint_sha256": "c" * 64},
        {"id": "rule", "kind": "rule", "checkpoint_sha256": None},
        {"id": "random", "kind": "random", "checkpoint_sha256": None},
        {"id": "legacy", "kind": "legacy_v1", "checkpoint_sha256": "d" * 64},
        {"id": "opponent_pool", "kind": "v2", "checkpoint_sha256": "c" * 64},
    ]
    body = {
        "schema": "guandan-marvel-evaluation-plan-v2",
        "mode": "selection",
        "deal_domain": "selection",
        "evaluation_id": "dual-test",
        "deals_manifest_hash": "e" * 64,
        "deal_dataset_hash": "f" * 64,
        "deal_ids": ["selection-000000", "selection-000001"],
        "candidate": candidate,
        "current_champion": {"checkpoint_sha256": "c" * 64},
        "legacy_anchor": {"checkpoint_sha256": "d" * 64},
        "opponent_pool_sample": {"checkpoint_sha256": "c" * 64},
        "matrix": matrix,
        "seat_rotations": [0, 1, 2, 3],
        "temperature": 0.0,
        "backend": "cpp",
        "data_isolation": {"formal_selection_eligible": True},
        "promotion_policy": {"mode": "formal"},
        "promotion_allowed": True,
    }
    base = {**body, "hash": payload_hash(body)}
    return build_resumable_evaluation_plan(
        base_plan=base,
        materialized_deals=[
            {"deal_id": "selection-000000", "deal_sha256": "1" * 64},
            {"deal_id": "selection-000001", "deal_sha256": "2" * 64},
        ],
    )


def _raw(task):
    team = int(task["seat_rotation"]) % 2
    win = int(task["task_index"]) % 2 == 0
    winner = team if win else 1 - team
    return {
        "opponent_id": task["opponent_id"],
        "opponent_kind": task["opponent_kind"],
        "deal_id": task["deal_id"],
        "seat_rotation": task["seat_rotation"],
        "candidate_team": team,
        "winner_team": winner,
        "candidate_win": win,
        "ranking": [1, 2, 3, 4] if winner == 0 else [2, 1, 4, 3],
        "outcome_class": "HEAD_LAST",
        "candidate_team_reward": 1.0 if win else -1.0,
        "token_steps": 12 + int(task["task_index"]),
        "committed_steps": 5,
        "illegal_count": 0,
        "truncated_count": 0,
        "action_trace_sha256": f"{10000 + int(task['task_index']):064x}",
    }


def test_deterministic_dual_gpu_sharding_is_stable_and_exhaustive():
    plan = _plan()
    shards = deterministic_task_shards(plan, 2)
    assert shards[0] == list(range(0, plan["task_count"], 2))
    assert shards[1] == list(range(1, plan["task_count"], 2))
    assert sorted(shards[0] + shards[1]) == list(range(plan["task_count"]))


def test_single_and_dual_worker_records_merge_and_summary_are_identical(tmp_path):
    plan = _plan()
    single = ResumableEvaluationStore(tmp_path / "single", plan)
    single.run_missing(_raw, worker_index=0, world_size=1)
    single_result = single.finalize()

    dual = ResumableEvaluationStore(tmp_path / "dual", plan)
    dual.run_missing(_raw, worker_index=0, world_size=2)
    dual.run_missing(_raw, worker_index=1, world_size=2)
    dual_result = dual.finalize()

    assert (tmp_path / "single/games.jsonl").read_bytes() == (tmp_path / "dual/games.jsonl").read_bytes()
    assert single_result["evaluation_hash"] == dual_result["evaluation_hash"]
    comparison = compare_formal_outputs(tmp_path / "single", tmp_path / "dual")
    assert comparison["complete"] is True
    assert comparison["records_identical"] is True
    assert comparison["summary_hash_identical"] is True
    assert comparison["promotion_input_identical"] is True


def test_worker_failure_blocks_merge_and_preserves_legal_shards_for_resume(tmp_path):
    plan = _plan()
    store = ResumableEvaluationStore(tmp_path / "dual", plan)
    store.run_missing(_raw, worker_index=0, world_size=2)
    with pytest.raises(RuntimeError, match="controlled"):
        store.run_missing(_raw, worker_index=1, world_size=2, fail_after=2)
    before = (tmp_path / "dual/games.gpu0.jsonl").read_bytes()
    with pytest.raises(IncompleteEvaluationError, match="unresolved_worker_failures"):
        store.finalize()
    assert not (tmp_path / "dual/evaluation_summary.json").exists()
    store.run_missing(_raw, worker_index=1, world_size=2)
    assert store.finalize()["complete"] is True
    assert (tmp_path / "dual/games.gpu0.jsonl").read_bytes() == before


def test_worker_group_binds_fixed_cuda_devices_and_forwards_logs(tmp_path):
    plan = _plan()
    snippet = (
        "import os; print(os.environ['GUANDAN_FORMAL_WORKER_INDEX'], "
        "os.environ['GUANDAN_FORMAL_WORLD_SIZE'], os.environ['CUDA_VISIBLE_DEVICES'])"
    )
    group = DeterministicWorkerGroup(
        output_dir=tmp_path / "workers",
        plan=plan,
        commands=[
            WorkerCommand(0, 0, (sys.executable, "-c", snippet)),
            WorkerCommand(1, 1, (sys.executable, "-c", snippet)),
        ],
        heartbeat_seconds=0.01,
    )
    result = group.run()
    assert result["status"] == "complete"
    assert (tmp_path / "workers/worker0.log").read_text().strip() == "0 2 0"
    assert (tmp_path / "workers/worker1.log").read_text().strip() == "1 2 1"


def test_nonzero_worker_exit_blocks_publication(tmp_path):
    plan = _plan()
    group = DeterministicWorkerGroup(
        output_dir=tmp_path / "workers",
        plan=plan,
        commands=[
            WorkerCommand(0, 0, (sys.executable, "-c", "raise SystemExit(0)")),
            WorkerCommand(1, 1, (sys.executable, "-c", "raise SystemExit(7)")),
        ],
        heartbeat_seconds=0.01,
    )
    with pytest.raises(FormalWorkerFailure, match="blocks merge/promotion"):
        group.run()
    assert json.loads((tmp_path / "workers/worker_group_result.json").read_text())["status"] == "failed"


def test_auto_continuation_is_versioned_default_off_and_gate_bound():
    default = AutoContinuationConfig.from_mapping(None)
    assert default.enabled is False and default.final_holdout_allowed is False
    blocked = continuation_gate(
        config=default,
        segments_completed_this_run=0,
        evaluation={"complete": True, "identity_verified": True},
        promotion={"status": "inconclusive", "promote": False},
        health={"illegal": 0, "truncated": 0, "nonfinite": 0},
        floors={"rule": True, "random": True},
    )
    assert blocked["allowed"] is False and "disabled" in blocked["reasons"][0]

    enabled = AutoContinuationConfig.from_mapping({
        "schema": AUTO_CONTINUATION_SCHEMA,
        "enabled": True,
        "max_segments_per_run": 2,
        "final_holdout_allowed": False,
    })
    allowed = continuation_gate(
        config=enabled,
        segments_completed_this_run=1,
        evaluation={"complete": True, "identity_verified": True},
        promotion={"status": "inconclusive", "promote": False},
        health={"illegal": 0, "truncated": 0, "nonfinite": 0},
        floors={"rule": True, "random": True},
    )
    assert allowed["allowed"] is True and allowed["retain_existing_champion"] is True
    floor_failure = continuation_gate(
        config=enabled,
        segments_completed_this_run=1,
        evaluation={"complete": True, "identity_verified": True},
        promotion={"status": "inconclusive", "promote": False},
        health={"illegal": 0},
        floors={"rule": False, "random": True},
    )
    assert floor_failure["allowed"] is False and floor_failure["manual_review"] is True
    with pytest.raises(ValueError, match="never authorize final holdout"):
        AutoContinuationConfig.from_mapping({
            "schema": AUTO_CONTINUATION_SCHEMA,
            "enabled": True,
            "max_segments_per_run": 2,
            "final_holdout_allowed": True,
        })
