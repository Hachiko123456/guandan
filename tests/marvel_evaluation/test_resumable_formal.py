from __future__ import annotations

import json
from pathlib import Path

import pytest

from guandan_marvel.evaluation.common import payload_hash
from guandan_marvel.evaluation.resumable import (
    IncompleteEvaluationError,
    PromotionTransaction,
    RecordConflictError,
    ResumableEvaluationStore,
    bind_formal_record,
    build_resumable_evaluation_plan,
    task_key,
)


def _base_plan(deals=("selection-000000", "selection-000001")):
    candidate = {
        "schema": "guandan-marvel-policy-identity-v1",
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
        "evaluation_id": "formal-test",
        "deals_manifest_hash": "e" * 64,
        "deal_dataset_hash": "f" * 64,
        "deal_ids": list(deals),
        "candidate": candidate,
        "current_champion": {"checkpoint_sha256": "c" * 64},
        "legacy_anchor": {"checkpoint_sha256": "d" * 64},
        "opponent_pool_sample": {"checkpoint_sha256": "c" * 64},
        "matrix": matrix,
        "seat_rotations": [0, 1, 2, 3],
        "temperature": 0.0,
        "backend": "cpp",
        "data_isolation": {"formal_selection_eligible": True},
        "promotion_policy": {"schema": "policy", "mode": "formal"},
        "promotion_allowed": True,
        "created_at": "2026-10-14T00:00:00+00:00",
        "identity_hash": "1" * 64,
    }
    return {**body, "hash": payload_hash(body)}


def _plan(deals=("selection-000000", "selection-000001")):
    materialized = [
        {"deal_id": deal_id, "deal_sha256": f"{index + 1:064x}"}
        for index, deal_id in enumerate(deals)
    ]
    return build_resumable_evaluation_plan(base_plan=_base_plan(deals), materialized_deals=materialized)


def _raw(task):
    candidate_team = int(task["seat_rotation"]) % 2
    candidate_win = (int(task["task_index"]) % 3) != 0
    winner = candidate_team if candidate_win else 1 - candidate_team
    return {
        "schema": "guandan-marvel-evaluation-game-v1",
        "status": "complete",
        "opponent_id": task["opponent_id"],
        "opponent_kind": task["opponent_kind"],
        "deal_id": task["deal_id"],
        "deal_seed": 100 + int(task["task_index"]),
        "seat_rotation": task["seat_rotation"],
        "candidate_team": candidate_team,
        "winner_team": winner,
        "candidate_win": candidate_win,
        "ranking": [1, 2, 3, 4] if winner == 0 else [2, 1, 4, 3],
        "outcome_class": "HEAD_LAST",
        "candidate_team_reward": 1.0 if candidate_win else -1.0,
        "token_steps": 10,
        "committed_steps": 4,
        "illegal_count": 0,
        "truncated_count": 0,
        "runtime_seconds": 9.9,
        "actions_per_second": 1.01,
        "action_trace_sha256": f"{int(task['task_index']) + 1000:064x}",
        "candidate_checkpoint_sha256": "foreign-will-be-overwritten",
        "opponent_checkpoint_sha256": "foreign-will-be-overwritten",
        "backend": "cpp",
        "protocol_version": "GD-ACTION-0.2-MARVEL|GD-ENV-0.2-MARVEL",
    }


def test_partial_evaluation_resume_only_runs_missing_tasks(tmp_path):
    plan = _plan()
    store = ResumableEvaluationStore(tmp_path / "evaluation", plan)
    calls = []

    def executor(task):
        calls.append(int(task["task_index"]))
        return _raw(task)

    with pytest.raises(RuntimeError, match="controlled"):
        store.run_missing(executor, fail_after=3)
    assert len(store.load_records()[0]) == 3
    progress = store.run_missing(executor)
    assert progress["complete"] is True
    assert calls[:3] == [0, 1, 2]
    assert calls[3:] == list(range(3, int(plan["task_count"])))
    result = store.finalize()
    assert result["complete"] is True
    assert result["games"] == plan["task_count"]
    assert json.loads((tmp_path / "evaluation/evaluation_summary.json").read_text())["deterministic"] is True


def test_formal_records_bind_plan_candidate_opponent_deal_backend_and_policy():
    plan = _plan(("selection-000000",))
    task = plan["tasks"][0]
    record = bind_formal_record(_raw(task), plan=plan, task=task)
    binding = record["formal_binding"]
    assert binding["evaluation_plan_hash"] == plan["hash"]
    assert binding["candidate_checkpoint_sha256"] == "a" * 64
    assert binding["opponent_checkpoint_sha256"] == "c" * 64
    assert binding["deal_sha256"] == task["deal_sha256"]
    assert binding["backend_identity_hash"] == "b" * 64
    assert binding["deterministic_policy_hash"] == plan["deterministic_policy_hash"]
    assert record["runtime_seconds"] == 0.0 and record["actions_per_second"] is None


def test_corrupted_and_foreign_records_are_rejected(tmp_path):
    plan = _plan(("selection-000000",))
    store = ResumableEvaluationStore(tmp_path / "evaluation", plan)
    path = store.shard_path(0)
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted JSONL"):
        store.load_records()
    path.unlink()
    task = plan["tasks"][0]
    record = bind_formal_record(_raw(task), plan=plan, task=task)
    record["formal_binding"]["evaluation_plan_hash"] = "0" * 64
    record["formal_binding"]["hash"] = __import__("hashlib").sha256(
        json.dumps(
            {key: value for key, value in record["formal_binding"].items() if key != "hash"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    record["formal_record_hash"] = __import__("hashlib").sha256(
        json.dumps(
            {key: value for key, value in record.items() if key != "formal_record_hash"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from plan"):
        store.load_records()


def test_identical_duplicate_is_deduplicated_and_conflict_rejected(tmp_path):
    plan = _plan(("selection-000000",))
    store = ResumableEvaluationStore(tmp_path / "evaluation", plan)
    task = plan["tasks"][0]
    record = bind_formal_record(_raw(task), plan=plan, task=task)
    path0 = store.shard_path(0)
    path1 = store.shard_path(1)
    path0.write_text(json.dumps(record) + "\n", encoding="utf-8")
    path1.write_text(json.dumps(record) + "\n", encoding="utf-8")
    records, stats = store.load_records()
    assert len(records) == 1 and stats["identical_duplicates"] == 1
    conflict = dict(record)
    conflict["candidate_win"] = not conflict["candidate_win"]
    conflict["formal_record_hash"] = __import__("hashlib").sha256(
        json.dumps(
            {key: value for key, value in conflict.items() if key != "formal_record_hash"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path1.write_text(json.dumps(conflict) + "\n", encoding="utf-8")
    with pytest.raises(RecordConflictError, match="conflicting duplicate"):
        store.load_records()


def test_summary_and_promotion_are_blocked_until_matrix_complete(tmp_path):
    plan = _plan(("selection-000000",))
    store = ResumableEvaluationStore(tmp_path / "evaluation", plan)
    store.run_missing(lambda task: _raw(task), fail_after=None, worker_index=0, world_size=2)
    with pytest.raises(IncompleteEvaluationError, match="incomplete"):
        store.finalize()
    assert not (tmp_path / "evaluation/evaluation_summary.json").exists()
    assert not (tmp_path / "evaluation/evaluation_result.json").exists()


def test_promotion_transaction_resume_is_idempotent(tmp_path):
    transaction = PromotionTransaction(tmp_path / "promotion_transaction.json")
    calls = {"champion": 0, "pool": 0}

    def publish_champion():
        calls["champion"] += 1
        return {"generation": 2, "manifest_hash": "g" * 64}

    fail_once = {"value": True}

    def publish_pool(champion):
        calls["pool"] += 1
        if fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("controlled pool failure")
        return {"revision": 2, "manifest_hash": "p" * 64, "champion": champion}

    with pytest.raises(RuntimeError, match="controlled"):
        transaction.execute(
            decision_hash="d" * 64,
            candidate_checkpoint_sha256="a" * 64,
            publish_champion=publish_champion,
            publish_pool=publish_pool,
        )
    failed = json.loads((tmp_path / "promotion_transaction.json").read_text())
    assert failed["state"] == "partial_failure"
    completed = transaction.execute(
        decision_hash="d" * 64,
        candidate_checkpoint_sha256="a" * 64,
        publish_champion=publish_champion,
        publish_pool=publish_pool,
    )
    replay = transaction.execute(
        decision_hash="d" * 64,
        candidate_checkpoint_sha256="a" * 64,
        publish_champion=publish_champion,
        publish_pool=publish_pool,
    )
    assert completed["state"] == "completed" and replay == completed
    assert calls == {"champion": 1, "pool": 2}
