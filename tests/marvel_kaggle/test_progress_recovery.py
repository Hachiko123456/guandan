from __future__ import annotations

import json
from pathlib import Path

import pytest

import guandan_marvel.kaggle.checkpoints as checkpoints
from guandan_marvel.kaggle.common import atomic_json, canonical_hash, sha256_file
from guandan_marvel.kaggle.progress import LiveProgressHeartbeat, format_console_summary
from guandan_marvel.kaggle.worker import shared_budget_decision


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
