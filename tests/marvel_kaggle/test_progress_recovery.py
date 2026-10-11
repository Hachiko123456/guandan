from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import guandan_marvel.kaggle.checkpoints as checkpoints
from guandan_marvel.kaggle.common import atomic_json, canonical_hash, sha256_file
from guandan_marvel.kaggle.progress import LiveProgressHeartbeat, format_console_summary
from guandan_marvel.kaggle.runner import _build_output_zip
from guandan_marvel.kaggle.worker import (
    _publish_evaluation_rendezvous, _wait_evaluation_rendezvous, shared_budget_decision,
)


def test_progress_eta_uses_session_iterations_not_historical(monkeypatch, tmp_path):
    monkeypatch.setattr("guandan_marvel.kaggle.progress.gpu_snapshot", lambda: [
        {"index": 0, "status": "ok", "utilization_percent": 50, "memory_used_mib": 100, "memory_total_mib": 200},
        {"index": 1, "status": "unavailable"},
    ])
    heartbeat = LiveProgressHeartbeat(
        run_root=tmp_path, rank_root=tmp_path / "ranks/rank000", mode="resume", rank=0,
        world_size=2, target_iterations=12, session_start_iteration=10,
        interval_seconds=10, is_primary=True,
    )
    heartbeat.started -= 20
    heartbeat.update(iteration=11, phase="learner", metrics={
        "statistics/completed_games_total": 4,
        "statistics/env_steps_total": 100,
        "statistics/game_steps_total": 40,
        "perf/env_steps_per_second": 5.0,
        "perf/game_steps_per_second": 2.0,
    })
    progress = heartbeat.snapshot()
    assert progress["session_completed_iterations"] == 1
    assert 19 <= progress["eta_seconds"] <= 21
    text = format_console_summary(progress)
    for token in ("mode=resume", "phase=learner", "iteration=11/12", "gpu0=", "gpu1=unavailable", "r0:", "r1:"):
        assert token in text


def test_first_iteration_progress_uses_unknown_not_fake_zero(monkeypatch, tmp_path):
    monkeypatch.setattr("guandan_marvel.kaggle.progress.gpu_snapshot", lambda: [])
    heartbeat = LiveProgressHeartbeat(
        run_root=tmp_path, rank_root=tmp_path / "ranks/rank000", mode="pilot", rank=0,
        world_size=2, target_iterations=2, session_start_iteration=0,
        interval_seconds=10, is_primary=True,
    )
    progress = heartbeat.snapshot()
    assert progress["completed_games"] is None
    assert progress["env_steps"] is None
    assert progress["eta_seconds"] is None
    assert "games=unknown" in format_console_summary(progress)


def test_checkpoint_retention_prunes_old_artifacts_and_updates_index(tmp_path):
    root = tmp_path / "run"
    checkpoint_root = root / "checkpoints"
    evaluation_root = root / "evaluation_checkpoints"
    checkpoint_root.mkdir(parents=True)
    evaluation_root.mkdir(parents=True)
    history = []
    for iteration in range(1, 5):
        directory = checkpoint_root / f"iter{iteration:06d}"
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"iteration": iteration}), encoding="utf-8")
        (directory / f"rank000.pt").write_bytes(f"rank-{iteration}".encode())
        evaluation = evaluation_root / f"iter{iteration:06d}.pt"
        evaluation.write_bytes(f"eval-{iteration}".encode())
        history.append({"iteration": iteration, "checkpoint": directory.relative_to(root).as_posix()})
    atomic_json(root / "checkpoint_manifest.json", {
        "schema": checkpoints.CHECKPOINT_MANIFEST_SCHEMA,
        "latest": history[-1], "history": history,
    })
    retention = checkpoints.prune_run_artifacts(
        root, checkpoint_keep_last=2, evaluation_keep_last=1,
    )
    assert retention["deleted_checkpoints"] == ["checkpoints/iter000001", "checkpoints/iter000002"]
    assert retention["deleted_evaluations"] == ["evaluation_checkpoints/iter000001.pt", "evaluation_checkpoints/iter000002.pt", "evaluation_checkpoints/iter000003.pt"]
    assert not (root / "checkpoints/iter000001").exists()
    assert (root / "checkpoints/iter000003").exists()
    updated = json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert [row["iteration"] for row in updated["history"]] == [3, 4]
    assert updated["retention"]["schema"] == "guandan-marvel-artifact-retention-v2"


def test_checkpoint_retention_refuses_unknown_paths(tmp_path):
    root = tmp_path / "run"
    checkpoint = root / "checkpoints/iter000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "checkpoints/user-notes").mkdir()
    atomic_json(root / "checkpoint_manifest.json", {
        "schema": checkpoints.CHECKPOINT_MANIFEST_SCHEMA,
        "latest": {"iteration": 1, "checkpoint": "checkpoints/iter000001"},
        "history": [{"iteration": 1, "checkpoint": "checkpoints/iter000001"}],
    })
    with pytest.raises(ValueError, match="unexpected checkpoint directory"):
        checkpoints.prune_run_artifacts(
            root, checkpoint_keep_last=1, evaluation_keep_last=0,
        )
    assert (root / "checkpoints/user-notes").is_dir()


def test_checkpoint_retention_publishes_safe_index_before_delete_failure(monkeypatch, tmp_path):
    root = tmp_path / "run"
    history = []
    for iteration in range(1, 3):
        checkpoint = root / "checkpoints" / f"iter{iteration:06d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
        history.append({"iteration": iteration, "checkpoint": checkpoint.relative_to(root).as_posix()})
    atomic_json(root / "checkpoint_manifest.json", {
        "schema": checkpoints.CHECKPOINT_MANIFEST_SCHEMA,
        "latest": history[-1], "history": history,
    })
    monkeypatch.setattr(checkpoints.shutil, "rmtree", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("delete failed")))
    with pytest.raises(OSError, match="delete failed"):
        checkpoints.prune_run_artifacts(
            root, checkpoint_keep_last=1, evaluation_keep_last=0,
        )
    updated = json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert [row["iteration"] for row in updated["history"]] == [2]
    assert updated["latest"]["iteration"] == 2


def test_recovery_bundle_replace_failure_preserves_previous_bundle(monkeypatch, tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/iter000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
    (checkpoint / "rank000.pt").write_bytes(b"rank0")
    atomic_json(run / "checkpoint_manifest.json", {
        "schema": checkpoints.CHECKPOINT_MANIFEST_SCHEMA,
        "latest": {"iteration": 1, "checkpoint": "checkpoints/iter000001"},
        "history": [],
    })
    destination = run / "guandan-marvel-recovery.zip"
    destination.write_bytes(b"previous-valid-bundle")
    previous_hash = sha256_file(destination)
    monkeypatch.setattr(checkpoints, "load_ddp_manifest", lambda *args, **kwargs: {"manifest_hash": "a" * 64})
    def fail_replace(source, target):
        raise OSError("injected replace failure")
    monkeypatch.setattr(checkpoints.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        checkpoints.build_recovery_bundle(
            run_root=run,
            destination=destination,
            source_commit="f" * 40,
            source_archive_sha256="e" * 64,
            source_content_manifest_hash="d" * 64,
            package_id="test",
            status="partial",
            native_extension=None,
        )
    assert sha256_file(destination) == previous_hash
    assert not destination.with_suffix(".zip.tmp").exists()

def test_budget_skew_is_reduced_to_one_shared_decision():
    class Context:
        def all_gather_object(self, local):
            return [
                {"rank": 0, "elapsed_seconds": 1.0, "budget_reached": False},
                {"rank": 1, "elapsed_seconds": 101.0, "budget_reached": True},
            ]
    decision, votes = shared_budget_decision(
        Context(), rank=0, local_elapsed=1.0, max_wall_seconds=100.0,
        iteration=1, target_iterations=3,
    )
    assert decision is True
    assert [row["budget_reached"] for row in votes] == [False, True]


def test_worker_publishes_recovery_before_entering_evaluation():
    source = Path("guandan_marvel/kaggle/worker.py").read_text(encoding="utf-8")
    checkpoint = source.index("save_distributed_checkpoint(")
    index = source.index("update_checkpoint_index(", checkpoint)
    bundle = source.index("build_recovery_bundle(", index)
    evaluation = source.index("run_smoke_evaluation_isolated(", bundle)
    assert checkpoint < index < bundle < evaluation


def test_output_zip_excludes_training_shards_and_nested_recovery(tmp_path):
    import zipfile
    run = tmp_path / "run"
    (run / "checkpoints/iter000001").mkdir(parents=True)
    (run / "checkpoints/iter000001/rank000.pt").write_bytes(b"large-shard")
    (run / "guandan-marvel-recovery.zip").write_bytes(b"nested-recovery")
    (run / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "output.zip"
    _build_output_zip(run, destination)
    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == ["metrics.jsonl"]

def test_long_evaluation_uses_filesystem_rendezvous_not_pending_collective(tmp_path):
    path = tmp_path / "evaluation_rendezvous_iter000100.json"
    expected = {"status": "complete", "evaluation_status": "complete"}

    def publish():
        time.sleep(0.05)
        _publish_evaluation_rendezvous(path, iteration=100, event=expected)

    thread = threading.Thread(target=publish)
    thread.start()
    observed = _wait_evaluation_rendezvous(
        path, iteration=100, timeout_seconds=2.0, poll_seconds=0.01,
    )
    thread.join(timeout=1.0)
    assert observed == expected
    source = Path("guandan_marvel/kaggle/worker.py").read_text(encoding="utf-8")
    assert "all_gather_object(eval_event" not in source
    assert source.index("run_smoke_evaluation_isolated(") < source.index("_publish_evaluation_rendezvous(", source.index("run_smoke_evaluation_isolated("))


def test_evaluation_rendezvous_rejects_wrong_iteration(tmp_path):
    path = tmp_path / "evaluation_rendezvous.json"
    _publish_evaluation_rendezvous(
        path, iteration=99, event={"status": "complete"},
    )
    with pytest.raises(TimeoutError, match="iteration mismatch"):
        _wait_evaluation_rendezvous(
            path, iteration=100, timeout_seconds=0.05, poll_seconds=0.01,
        )
