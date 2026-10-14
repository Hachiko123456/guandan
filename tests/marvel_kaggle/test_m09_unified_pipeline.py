from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile

import pytest

from guandan_marvel.kaggle.common import atomic_json, canonical_hash, sha256_file
from guandan_marvel.kaggle.unified_pipeline import (
    CONTINUATION_MANIFEST_NAME,
    CONTINUATION_SCHEMA,
    DEFAULT_SEGMENT_SCHEDULE,
    PIPELINE_STATE_NAME,
    PipelineHooks,
    SegmentTarget,
    TRAINING_COMMIT_NAME,
    TrainingArtifacts,
    UnifiedSegmentPipeline,
    commit_training_artifacts,
    create_continuation_bundle,
    discover_previous_continuation,
    load_training_commit,
    select_next_segment,
)

SOURCE = {
    "source_commit": "source-commit",
    "source_archive_sha256": "a" * 64,
    "source_content_manifest_hash": "b" * 64,
}
CONTROL_HASH = "c" * 64


def _write_hashed(path: Path, body: dict[str, object], *, field: str = "hash") -> dict[str, object]:
    payload = {**body, field: canonical_hash(body)}
    atomic_json(path, payload)
    return payload


def _fake_recovery(tmp_path: Path, target: SegmentTarget) -> TrainingArtifacts:
    root = tmp_path / f"recovery-{target.segment_id}"
    (root / "checkpoint").mkdir(parents=True)
    (root / "artifacts").mkdir()
    for rank in (0, 1):
        (root / "checkpoint" / f"rank{rank:03d}.pt").write_bytes(f"rank-{rank}".encode())
    checkpoint_body = {
        "schema": "guandan-marvel-ddp-checkpoint-v1",
        "iteration": target.target_iteration,
        "complete": True,
        "world_size": 2,
        "rank_shards": [
            {
                "rank": rank,
                "shard": f"rank{rank:03d}.pt",
                "shard_sha256": sha256_file(root / "checkpoint" / f"rank{rank:03d}.pt"),
            }
            for rank in (0, 1)
        ],
    }
    checkpoint = _write_hashed(root / "checkpoint" / "manifest.json", checkpoint_body, field="manifest_hash")
    atomic_json(root / "artifacts" / "progress.json", {
        "schema": "guandan-marvel-progress-v2",
        "iteration": target.target_iteration,
        "completed_games": target.target_completed_games,
        "complete": True,
    })
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    recovery_body = {
        "schema": "guandan-marvel-recovery-manifest-v3",
        "package_id": "test",
        "status": "complete",
        "iteration": target.target_iteration,
        "checkpoint": "checkpoint/manifest.json",
        "checkpoint_manifest_hash": checkpoint["manifest_hash"],
        **SOURCE,
        "files": files,
    }
    _write_hashed(root / "recovery_manifest.json", recovery_body, field="manifest_hash")
    archive = tmp_path / f"{target.segment_id}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(root).as_posix())
    return TrainingArtifacts(archive, root)


def test_auto_segment_selection_and_counter_drift_rejection():
    assert select_next_segment(500, 2_000) == DEFAULT_SEGMENT_SCHEDULE[1]
    assert select_next_segment(1_000, 4_000) == DEFAULT_SEGMENT_SCHEDULE[2]
    assert select_next_segment(2_000, 8_000) is None
    with pytest.raises(ValueError, match="inconsistently"):
        select_next_segment(1_000, 2_000)
    with pytest.raises(ValueError, match="exceeds"):
        select_next_segment(2_001, 8_004)


def test_previous_continuation_discovery_is_identity_bound(tmp_path):
    root = tmp_path / "input" / "prior"
    root.mkdir(parents=True)
    body = {
        "schema": CONTINUATION_SCHEMA,
        "version": 1,
        "control_hash": CONTROL_HASH,
        "source_identity": SOURCE,
        "plan_hash": "p" * 64,
        "training_commit_hash": "t" * 64,
        "next_target": DEFAULT_SEGMENT_SCHEDULE[1].identity(),
        "final_holdout_generated": False,
        "files": [],
        "created_at": "2026-10-14T00:00:00+00:00",
    }
    payload = {**body, "hash": canonical_hash(body)}
    atomic_json(root / CONTINUATION_MANIFEST_NAME, payload)
    found = discover_previous_continuation(
        tmp_path / "input", expected_control_hash=CONTROL_HASH, expected_source_identity=SOURCE,
    )
    assert found is not None and found[0] == (root / CONTINUATION_MANIFEST_NAME).resolve()
    payload["control_hash"] = "foreign"
    payload["hash"] = canonical_hash({key: value for key, value in payload.items() if key != "hash"})
    atomic_json(root / CONTINUATION_MANIFEST_NAME, payload)
    with pytest.raises(RuntimeError, match="none match"):
        discover_previous_continuation(
            tmp_path / "input", expected_control_hash=CONTROL_HASH, expected_source_identity=SOURCE,
        )


def test_training_commit_validates_budget_ranks_crc_and_immutable_copy(tmp_path):
    target = DEFAULT_SEGMENT_SCHEDULE[1]
    artifacts = _fake_recovery(tmp_path, target)
    commit = commit_training_artifacts(
        target=target,
        artifacts=artifacts,
        committed_root=tmp_path / "committed",
        source_identity=SOURCE,
        control_hash=CONTROL_HASH,
    )
    path = tmp_path / "committed" / target.segment_id / TRAINING_COMMIT_NAME
    assert load_training_commit(path)["hash"] == commit["hash"]
    assert commit["immutable"] is True
    assert [row["rank"] for row in commit["rank_shards"]] == [0, 1]
    artifacts.recovery_zip.write_bytes(b"corrupt")
    assert load_training_commit(path)["hash"] == commit["hash"]


def test_training_commits_before_evaluation_and_eval_failure_preserves_recovery(tmp_path):
    target = DEFAULT_SEGMENT_SCHEDULE[1]
    calls = {"train": 0, "evaluate": 0, "promote": 0}

    def train(selected, output):
        assert selected == target
        calls["train"] += 1
        return _fake_recovery(tmp_path / f"train-{calls['train']}", selected)

    fail_once = {"value": True}

    def evaluate(selected, commit, output):
        calls["evaluate"] += 1
        committed = output / "committed_training" / selected.segment_id / TRAINING_COMMIT_NAME
        assert committed.is_file(), "evaluation started before durable training commit"
        load_training_commit(committed)
        if fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("controlled evaluation failure")
        return {"complete": True, "evaluation_hash": "e" * 64}

    def promote(selected, evaluation, output):
        calls["promote"] += 1
        return {"status": "inconclusive", "promote": False}

    pipeline = UnifiedSegmentPipeline(
        output_root=tmp_path / "pipeline",
        source_identity=SOURCE,
        control_hash=CONTROL_HASH,
        hooks=PipelineHooks(train=train, evaluate=evaluate, promote=promote),
    )
    with pytest.raises(RuntimeError, match="controlled"):
        pipeline.run(iteration=500, completed_games=2_000)
    committed_dir = tmp_path / "pipeline" / "committed_training" / target.segment_id
    committed_zip = committed_dir / "training_recovery.committed.zip"
    before = sha256_file(committed_zip)
    state = json.loads((tmp_path / "pipeline" / PIPELINE_STATE_NAME).read_text(encoding="utf-8"))
    assert state["phase"] == "TRAINING_COMPLETE_EVALUATION_FAILED"

    result = pipeline.run(iteration=500, completed_games=2_000)
    assert result["phase"] == "PROMOTION_COMMITTED"
    assert calls == {"train": 1, "evaluate": 2, "promote": 1}
    assert sha256_file(committed_zip) == before
    assert result["details"]["promotion"]["promote"] is False


def test_incomplete_evaluation_is_resumable_without_retraining(tmp_path):
    target = DEFAULT_SEGMENT_SCHEDULE[1]
    calls = {"train": 0, "evaluate": 0}

    def train(selected, output):
        calls["train"] += 1
        return _fake_recovery(tmp_path / "training", selected)

    def evaluate(selected, commit, output):
        calls["evaluate"] += 1
        return {"complete": calls["evaluate"] > 1, "completed": calls["evaluate"], "total": 2}

    pipeline = UnifiedSegmentPipeline(
        output_root=tmp_path / "pipeline",
        source_identity=SOURCE,
        control_hash=CONTROL_HASH,
        hooks=PipelineHooks(
            train=train,
            evaluate=evaluate,
            promote=lambda *_: {"status": "inconclusive", "promote": False},
        ),
    )
    first = pipeline.run(iteration=500, completed_games=2_000)
    assert first["phase"] == "EVALUATION_INCOMPLETE_RESUMABLE"
    second = pipeline.run(iteration=500, completed_games=2_000)
    assert second["phase"] == "PROMOTION_COMMITTED"
    assert calls == {"train": 1, "evaluate": 2}


def test_continuation_bundle_contains_only_non_holdout_handoff_state(tmp_path):
    target = DEFAULT_SEGMENT_SCHEDULE[1]
    artifacts = _fake_recovery(tmp_path / "source", target)
    commit = commit_training_artifacts(
        target=target,
        artifacts=artifacts,
        committed_root=tmp_path / "committed",
        source_identity=SOURCE,
        control_hash=CONTROL_HASH,
    )
    commit_path = tmp_path / "committed" / target.segment_id / TRAINING_COMMIT_NAME
    state_body = {
        "schema": "guandan-marvel-m09-unified-pipeline-state-v1",
        "version": 1,
        "phase": "PACKAGING",
        "segment": target.identity(),
        "details": {},
        "error": None,
        "updated_at": "2026-10-14T00:00:00+00:00",
        "history": [],
    }
    atomic_json(tmp_path / "state.json", {**state_body, "hash": canonical_hash(state_body)})
    champion = _write_hashed(tmp_path / "champion.json", {"schema": "champion", "generation": 1})
    pool = _write_hashed(tmp_path / "pool.json", {"schema": "pool", "revision": 1})
    evaluation = _write_hashed(tmp_path / "evaluation.json", {"schema": "evaluation", "complete": True})
    result = create_continuation_bundle(
        output_root=tmp_path / "output",
        training_commit_path=commit_path,
        pipeline_state_path=tmp_path / "state.json",
        champion_manifest_path=tmp_path / "champion.json",
        opponent_pool_manifest_path=tmp_path / "pool.json",
        promotion_transaction_path=None,
        evaluation_paths=[tmp_path / "evaluation.json"],
        next_target=DEFAULT_SEGMENT_SCHEDULE[2],
        source_identity=SOURCE,
        control_hash=CONTROL_HASH,
        plan_hash="p" * 64,
    )
    assert result["manifest"]["training_commit_hash"] == commit["hash"]
    assert result["manifest"]["final_holdout_generated"] is False
    with zipfile.ZipFile(result["archive"]) as bundle:
        names = set(bundle.namelist())
    assert CONTINUATION_MANIFEST_NAME in names
    assert all("holdout" not in name.lower() for name in names)
