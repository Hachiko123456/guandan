"""Phase-3 v1.4 + AMP durable checkpoint -> new-process resume verification."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests" / "marvel_training" / "_fuse_recovery_worker.py"
SPLIT_AT = 2
TOTAL_ITERS = 4


def _run(mode: str, *, iters: int, checkpoint: Path, run_dir: Path, out: Path, shrink_at=None) -> dict:
    command = [
        sys.executable, str(WORKER), "--mode", mode, "--iters", str(iters),
        "--seed", "4242", "--checkpoint", str(checkpoint), "--run-dir", str(run_dir),
        "--out", str(out),
    ]
    if shrink_at is not None:
        command += ["--shrink-at", str(shrink_at)]
    result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_v14_amp_durable_checkpoint_new_process_resume_matches_control(tmp_path: Path):
    control = _run(
        "train", iters=TOTAL_ITERS, checkpoint=tmp_path / "control.pt",
        run_dir=tmp_path / "control-dir", out=tmp_path / "control.json", shrink_at=SPLIT_AT,
    )
    assert control["checkpoint_exists"] is True

    split_checkpoint = tmp_path / "split.pt"
    first = _run(
        "train", iters=SPLIT_AT, checkpoint=split_checkpoint,
        run_dir=tmp_path / "split-a", out=tmp_path / "split-a.json", shrink_at=SPLIT_AT,
    )
    assert first["state"]["iteration"] == SPLIT_AT
    resumed = _run(
        "resume", iters=TOTAL_ITERS - SPLIT_AT, checkpoint=split_checkpoint,
        run_dir=tmp_path / "split-b", out=tmp_path / "split-b.json",
    )
    # A genuinely new process loaded the durable checkpoint (strict classification).
    assert resumed["mode"] == "resume"
    assert resumed["resume_kind"] in {"exact_resume", "compatible_resume"}
    assert resumed["state_on_load"]["iteration"] == SPLIT_AT
    assert resumed["state_on_load"]["training_semantics_profile"] == "macro_decision_v1"

    control_state = control["state"]
    resumed_state = resumed["state"]
    assert resumed_state["iteration"] == control_state["iteration"] == TOTAL_ITERS
    assert resumed_state["durable_updates"] == control_state["durable_updates"] == TOTAL_ITERS
    assert resumed_state["learner_updates"] == control_state["learner_updates"]
    assert resumed_state["durable_learner_updates"] == control_state["durable_learner_updates"]
    assert resumed_state["completed_games_total"] == control_state["completed_games_total"]
    assert resumed_state["env_steps"] == control_state["env_steps"]
    assert resumed_state["game_steps"] == control_state["game_steps"]
    # Model, optimizer, RNG and sampler continuity are verified bitwise.
    assert resumed_state["agent_resume_digest"] == control_state["agent_resume_digest"]
    assert resumed_state["agent_merged_digest"] == control_state["agent_merged_digest"]
    assert resumed_state["optimizer_digest"] == control_state["optimizer_digest"]
    assert resumed_state["rng_digest"] == control_state["rng_digest"]

    # --- v1.4 + AMP specific invariants -------------------------------------
    assert resumed_state["architecture"] == "v1.4-macro-fuse"
    assert resumed_state["has_fuse"] is True
    assert resumed_state["amp_enabled"] is True
    assert resumed_state["amp_autocast_dtype"] == "torch.bfloat16"
    assert resumed_state["scaler_enabled"] is True
    assert resumed_state["output_proj_nonzero"] > 0
    # The GradScaler state was checkpointed and restored exactly.
    assert resumed["state_on_load"]["scaler_scale"] == 1024.0
    assert resumed_state["scaler_scale"] == control_state["scaler_scale"] == 1024.0
    assert resumed_state["scaler_state_digest"] == control_state["scaler_state_digest"]
    assert resumed_state["amp_skipped_steps"] == control_state["amp_skipped_steps"] == 0
    for key, value in control_state["last_metrics"].items():
        assert resumed_state["last_metrics"][key] == pytest.approx(value, abs=0.0), key
    # The durable checkpoint carries the architecture it was written for.
    assert first["state"]["architecture"] == "v1.4-macro-fuse"
