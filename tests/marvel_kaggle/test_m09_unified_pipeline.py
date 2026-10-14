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


def test_unified_dataset_notebook_delivery_validation(tmp_path, monkeypatch):
    from guandan_marvel.kaggle import m09_unified_delivery as delivery

    repo_root = Path(__file__).resolve().parents[2]
    source_dataset = Path(r"D:\project\kaggle_train\guandan-marvel-m09-main-ippo-segmented-20261013\datasets\segment1000")
    recovery = repo_root / "runs/M09_full_training/remote/r2_main_segment0500_20261013T135603Z/downloaded/guandan-marvel-recovery.zip"
    champion = Path(r"D:\project\kaggle_train\guandan-marvel-m09-algorithm-selection-20261013\dataset\ippo_iter000100.pt")
    legacy = repo_root / "runs/a07probe/local_fast/ippo/ippo_update_000001_90453d2696194fc2b755820939dcbed2_periodic.pt"
    assert source_dataset.is_dir() and recovery.is_file() and champion.is_file() and legacy.is_file()
    original_copy2 = delivery.shutil.copy2

    def link_large(source, destination, *args, **kwargs):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.is_dir():
            destination_path = destination_path / source_path.name
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.stat().st_size > 10_000_000:
            destination_path.unlink(missing_ok=True)
            destination_path.hardlink_to(source_path)
            return str(destination_path)
        return original_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(delivery.shutil, "copy2", link_large)
    output = repo_root / "runs/.m09-unified-delivery-tests" / tmp_path.name / delivery.UNIFIED_DELIVERY_ID
    if output.parent.exists():
        shutil.rmtree(output.parent)
    manifest = delivery.build_unified_delivery(
        repo_root,
        output,
        source_dataset_root=source_dataset,
        initial_recovery=recovery,
        champion_seed=champion,
        legacy_checkpoint=legacy,
    )
    validation = delivery.validate_unified_delivery(output)
    assert manifest["delivery_id"] == delivery.UNIFIED_DELIVERY_ID
    assert validation["valid"] is True
    assert validation["dataset_count"] == validation["notebook_count"] == 1
    assert validation["formal_workers_default"] == 1
    assert validation["production_enabled"] is False
    assert validation["auto_continuation_enabled"] is False
    assert validation["final_holdout_generated"] is False
    notebook_text = (output / delivery.UNIFIED_NOTEBOOK_NAME).read_text(encoding="utf-8")
    assert "unified_bootstrap_heartbeat" in notebook_text
    assert "copy_directory=False" in notebook_text
    assert "ignored_runtime_manifests" in notebook_text
    assert "CONTROL_PATH.parent/'guandan_marvel_dataset.json'" in notebook_text
    assert "REMOTE_B_EXPECTED_FAILURE_COMPLETE" in notebook_text
    assert "EXPECTED_REMOTE_B_FAULT" in notebook_text
    assert (output / "dataset/control_overlay/guandan_marvel/evaluation/provenance.py").read_bytes() == (repo_root / "guandan_marvel/evaluation/provenance.py").read_bytes()
    assert not list(output.rglob("*final_holdout*"))
    shutil.rmtree(output.parent)


def test_latest_recovery_prefers_previous_output_over_static_seed(tmp_path):
    from guandan_marvel.kaggle.unified_pipeline import discover_latest_recovery, materialize_recovery_candidate

    input_root = tmp_path / "input"
    input_root.mkdir()
    seed = _fake_recovery(tmp_path / "seed", DEFAULT_SEGMENT_SCHEDULE[0])
    previous = _fake_recovery(tmp_path / "previous", DEFAULT_SEGMENT_SCHEDULE[1])
    shutil.copy2(seed.recovery_zip, input_root / "m09_initial_iter0500_recovery.seed.zip")
    prior_dir = input_root / "previous-output/continuation/training"
    prior_dir.mkdir(parents=True)
    shutil.copy2(previous.recovery_zip, prior_dir / "training_recovery.committed.zip")
    selected = discover_latest_recovery(input_root, source_identity=SOURCE)
    assert int(selected.manifest["iteration"]) == 1000
    root = materialize_recovery_candidate(selected, tmp_path / "materialized")
    assert json.loads((root / "recovery_manifest.json").read_text())["iteration"] == 1000
    directory_candidate = type(selected)("directory", root, json.loads((root / "recovery_manifest.json").read_text()), None)
    reused = materialize_recovery_candidate(directory_candidate, tmp_path / "unused", copy_directory=False)
    assert reused == root.resolve() and not (tmp_path / "unused").exists()


def test_retry_with_target_recovery_uses_pending_segment_and_skips_training(tmp_path):
    target = DEFAULT_SEGMENT_SCHEDULE[1]
    calls = {"train": 0, "evaluate": 0}

    def train(selected, output):
        calls["train"] += 1
        return _fake_recovery(tmp_path / "training", selected)

    def evaluate(selected, commit, output):
        calls["evaluate"] += 1
        if calls["evaluate"] == 1:
            raise RuntimeError("controlled")
        return {"complete": True, "evaluation_hash": "e" * 64}

    pipeline = UnifiedSegmentPipeline(
        output_root=tmp_path / "pipeline",
        source_identity=SOURCE,
        control_hash=CONTROL_HASH,
        hooks=PipelineHooks(train=train, evaluate=evaluate, promote=lambda *_: {"status": "inconclusive", "promote": False}),
    )
    with pytest.raises(RuntimeError):
        pipeline.run(iteration=500, completed_games=2000)
    result = pipeline.run(iteration=1000, completed_games=4000, pending_target=target)
    assert result["phase"] == "PROMOTION_COMMITTED"
    assert calls == {"train": 1, "evaluate": 2}


def test_unified_remote_scripts_keep_training_and_evaluation_in_separate_processes():
    repo_root = Path(__file__).resolve().parents[2]
    pipeline = (repo_root / "scripts/m09_unified_pipeline.py").read_text(encoding="utf-8")
    worker = (repo_root / "scripts/m09_unified_formal_worker.py").read_text(encoding="utf-8")
    compile(pipeline, "m09_unified_pipeline.py", "exec")
    compile(worker, "m09_unified_formal_worker.py", "exec")
    assert '"-m", "guandan_marvel.kaggle.runner"' in pipeline
    assert "DeterministicWorkerGroup" in pipeline and "worker_script" in pipeline
    assert 'source_root / "configs/acceptance_profiles.json"' in pipeline
    assert "acceptance_profiles_materialized" in pipeline
    assert "remote_validation_defaulted" in pipeline
    assert 'validation_control.get("default_mode", "A")' in pipeline
    assert 'validation_mode == "C"' in pipeline
    assert pipeline.index('"-m", "guandan_marvel.kaggle.runner"') < pipeline.index("def evaluate(")
    assert "final_holdout" not in worker


def test_remote_validation_target_override_is_small_and_does_not_select_segment1000(tmp_path):
    target = SegmentTarget("remote_a_smoke0502", 502, 2008)
    observed = []

    def train(selected, output):
        observed.append(selected)
        return _fake_recovery(tmp_path / "smoke-training", selected)

    pipeline = UnifiedSegmentPipeline(
        output_root=tmp_path / "pipeline",
        source_identity=SOURCE,
        control_hash=CONTROL_HASH,
        hooks=PipelineHooks(
            train=train,
            evaluate=lambda *_: {"complete": True, "evaluation_hash": "e" * 64},
            promote=lambda *_: {"status": "inconclusive", "promote": False},
        ),
    )
    result = pipeline.run(iteration=500, completed_games=2000, target_override=target)
    assert result["segment"] == target.identity()
    assert observed == [target]


def test_runtime_dataset_preserves_retained_or_auto_unpacked_source(tmp_path):
    import importlib.util
    import zipfile
    from guandan_marvel.kaggle.bootstrap import resolve_dataset_input

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("m09_unified_pipeline_runtime_dataset", repo_root / "scripts/m09_unified_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    delivery_dataset = Path(r"D:\project\kaggle_train\guandan-marvel-m09-unified-pipeline-v1\dataset")
    target = SegmentTarget("remote_a_smoke0502", 502, 2008)

    retained = module._runtime_dataset(
        dataset_manifest_path=delivery_dataset / "guandan_marvel_dataset.json",
        target=target,
        runtime_root=tmp_path / "retained-runtime",
    )
    assert resolve_dataset_input(retained).source.kind == "zip"

    unpacked_dataset = tmp_path / "auto-unpacked-dataset"
    unpacked_dataset.mkdir()
    for name in (
        "guandan_marvel_dataset.json",
        "source_manifest.json",
        "marvel_m09_main_training_plan.json",
        "marvel_m08_locked_config.json",
    ):
        shutil.copy2(delivery_dataset / name, unpacked_dataset / name)
    with zipfile.ZipFile(delivery_dataset / "guandan-marvel-source.zip") as archive:
        archive.extractall(unpacked_dataset)
    assert not (unpacked_dataset / "guandan-marvel-source.zip").exists()
    unpacked = module._runtime_dataset(
        dataset_manifest_path=unpacked_dataset / "guandan_marvel_dataset.json",
        target=target,
        runtime_root=tmp_path / "unpacked-runtime",
    )
    assert resolve_dataset_input(unpacked).source.kind == "directory"
