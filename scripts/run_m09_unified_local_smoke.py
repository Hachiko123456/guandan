"""Run a deterministic local fault/resume smoke for the M09 Unified Pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.evaluation.common import atomic_write_json, payload_hash
from guandan_marvel.evaluation.dual_gpu import compare_formal_outputs
from guandan_marvel.evaluation.resumable import ResumableEvaluationStore, build_resumable_evaluation_plan
from guandan_marvel.kaggle.common import atomic_json, canonical_hash, sha256_file
from guandan_marvel.kaggle.unified_pipeline import (
    DEFAULT_SEGMENT_SCHEDULE, PIPELINE_STATE_NAME, PipelineHooks, SegmentTarget,
    TrainingArtifacts, UnifiedSegmentPipeline, create_continuation_bundle,
)

SOURCE = {"source_commit": "local-smoke", "source_archive_sha256": "a" * 64, "source_content_manifest_hash": "b" * 64}
CONTROL_HASH = "c" * 64


def _write_hashed(path: Path, body: dict[str, object], field: str = "hash") -> dict[str, object]:
    payload = {**body, field: canonical_hash(body)}
    atomic_json(path, payload)
    return payload


def _recovery(root: Path, target: SegmentTarget) -> TrainingArtifacts:
    recovery = root / "recovery"
    (recovery / "checkpoint").mkdir(parents=True)
    (recovery / "artifacts").mkdir()
    rows = []
    for rank in (0, 1):
        shard = recovery / "checkpoint" / f"rank{rank:03d}.pt"
        shard.write_bytes(f"local-smoke-rank-{rank}".encode())
        rows.append({"rank": rank, "shard": shard.name, "shard_sha256": sha256_file(shard)})
    checkpoint = _write_hashed(recovery / "checkpoint/manifest.json", {"schema": "guandan-marvel-ddp-checkpoint-v1", "iteration": target.target_iteration, "complete": True, "world_size": 2, "rank_shards": rows}, "manifest_hash")
    atomic_json(recovery / "artifacts/progress.json", {"schema": "guandan-marvel-progress-v2", "iteration": target.target_iteration, "completed_games": target.target_completed_games, "complete": True})
    files = [{"path": path.relative_to(recovery).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(recovery.rglob("*")) if path.is_file()]
    _write_hashed(recovery / "recovery_manifest.json", {"schema": "guandan-marvel-recovery-manifest-v3", "package_id": "local-smoke", "status": "complete", "iteration": target.target_iteration, "checkpoint": "checkpoint/manifest.json", "checkpoint_manifest_hash": checkpoint["manifest_hash"], **SOURCE, "files": files}, "manifest_hash")
    archive = root / "recovery.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(recovery.rglob("*")):
            if path.is_file(): bundle.write(path, path.relative_to(recovery).as_posix())
    return TrainingArtifacts(archive, recovery)


def _plan():
    matrix = [
        {"id": "champion", "kind": "v2", "checkpoint_sha256": "d" * 64},
        {"id": "rule", "kind": "rule", "checkpoint_sha256": None},
        {"id": "random", "kind": "random", "checkpoint_sha256": None},
        {"id": "legacy", "kind": "legacy_v1", "checkpoint_sha256": "e" * 64},
        {"id": "opponent_pool", "kind": "v2", "checkpoint_sha256": "d" * 64},
    ]
    body = {"schema": "guandan-marvel-evaluation-plan-v2", "mode": "selection", "deal_domain": "selection", "evaluation_id": "local-smoke", "deals_manifest_hash": "f" * 64, "deal_dataset_hash": "1" * 64, "deal_ids": ["selection-000000"], "candidate": {"checkpoint_sha256": "2" * 64, "backend_identity_hash": "3" * 64, "protocol_version": "GD-ACTION-0.2-MARVEL|GD-ENV-0.2-MARVEL"}, "current_champion": {"checkpoint_sha256": "d" * 64}, "legacy_anchor": {"checkpoint_sha256": "e" * 64}, "opponent_pool_sample": {"checkpoint_sha256": "d" * 64}, "matrix": matrix, "seat_rotations": [0, 1, 2, 3], "temperature": 0.0, "backend": "cpp", "data_isolation": {"formal_selection_eligible": True}, "promotion_policy": {"mode": "formal"}, "promotion_allowed": True}
    base = {**body, "hash": payload_hash(body)}
    return build_resumable_evaluation_plan(base_plan=base, materialized_deals=[{"deal_id": "selection-000000", "deal_sha256": "4" * 64}])


def _record(task):
    team = int(task["seat_rotation"]) % 2
    win = int(task["task_index"]) % 2 == 0
    winner = team if win else 1 - team
    return {"opponent_id": task["opponent_id"], "opponent_kind": task["opponent_kind"], "deal_id": task["deal_id"], "seat_rotation": task["seat_rotation"], "candidate_team": team, "winner_team": winner, "candidate_win": win, "ranking": [1, 2, 3, 4] if winner == 0 else [2, 1, 4, 3], "outcome_class": "HEAD_LAST", "candidate_team_reward": 1.0 if win else -1.0, "token_steps": 20 + int(task["task_index"]), "committed_steps": 8, "illegal_count": 0, "truncated_count": 0, "action_trace_sha256": f"{1000 + int(task['task_index']):064x}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m09-unified-smoke-") as temporary:
        root = Path(temporary)
        pipeline_root = root / "pipeline"
        target = DEFAULT_SEGMENT_SCHEDULE[1]
        plan = _plan()
        calls = {"training": 0, "evaluation": 0, "promotion": 0}
        partial = {"records": 0}

        def train(selected, out):
            calls["training"] += 1
            return _recovery(root / "training", selected)

        def evaluate(selected, commit, out):
            calls["evaluation"] += 1
            store = ResumableEvaluationStore(out / "evaluation" / selected.segment_id, plan)
            if calls["evaluation"] == 1:
                try: store.run_missing(_record, fail_after=3)
                finally: partial["records"] = len(store.load_records()[0])
            else:
                store.run_missing(_record)
                result = store.finalize()
                return {**result, "complete": True, "identity_verified": True}

        def promote(selected, evaluation, out):
            calls["promotion"] += 1
            return {"status": "inconclusive", "promote": False}

        pipeline = UnifiedSegmentPipeline(output_root=pipeline_root, source_identity=SOURCE, control_hash=CONTROL_HASH, hooks=PipelineHooks(train=train, evaluate=evaluate, promote=promote))
        failure = None
        try:
            pipeline.run(iteration=500, completed_games=2000)
        except RuntimeError as error:
            failure = str(error)
        committed = pipeline_root / "committed_training" / target.segment_id / "training_recovery.committed.zip"
        committed_before = sha256_file(committed)
        second = pipeline.run(iteration=1000, completed_games=4000, pending_target=target)
        committed_after = sha256_file(committed)

        dual = ResumableEvaluationStore(root / "dual", plan)
        dual.run_missing(_record, worker_index=0, world_size=2)
        dual.run_missing(_record, worker_index=1, world_size=2)
        dual_result = dual.finalize()
        consistency = compare_formal_outputs(pipeline_root / "evaluation" / target.segment_id, root / "dual")

        champion = _write_hashed(root / "champion.json", {"schema": "local-champion", "generation": 1})
        pool = _write_hashed(root / "pool.json", {"schema": "local-pool", "revision": 1})
        continuation = create_continuation_bundle(output_root=root / "handoff", training_commit_path=pipeline_root / "committed_training" / target.segment_id / "training_commit.json", pipeline_state_path=pipeline_root / PIPELINE_STATE_NAME, champion_manifest_path=root / "champion.json", opponent_pool_manifest_path=root / "pool.json", promotion_transaction_path=None, evaluation_paths=[pipeline_root / "evaluation" / target.segment_id / "evaluation_summary.json"], next_target=DEFAULT_SEGMENT_SCHEDULE[2], source_identity=SOURCE, control_hash=CONTROL_HASH, plan_hash=plan["hash"])
        evidence = {
            "schema": "guandan-marvel-m09-unified-local-smoke-v1",
            "status": "passed",
            "local_only": True,
            "kaggle_acceptance": False,
            "controlled_evaluation_failure": failure,
            "partial_records_before_retry": partial["records"],
            "training_calls": calls["training"],
            "evaluation_calls": calls["evaluation"],
            "promotion_calls": calls["promotion"],
            "rerun_skipped_training": calls["training"] == 1,
            "training_recovery_sha256_before_failure_retry": committed_before,
            "training_recovery_sha256_after_retry": committed_after,
            "training_recovery_preserved": committed_before == committed_after,
            "pipeline_final_phase": second["phase"],
            "single_evaluation_hash": json.loads((pipeline_root / "evaluation" / target.segment_id / "evaluation_result.json").read_text())["evaluation_hash"],
            "dual_evaluation_hash": dual_result["evaluation_hash"],
            "single_dual_consistency": consistency,
            "continuation_sha256": continuation["archive_sha256"],
            "continuation_next_target": continuation["manifest"]["next_target"],
            "final_holdout_generated": False,
        }
        evidence["hash"] = canonical_hash(evidence)
        atomic_json(output, evidence)
        print(json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
