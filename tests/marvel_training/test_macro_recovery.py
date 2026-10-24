"""Phase-2 Profile C durable checkpoint -> new-process resume verification."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests" / "marvel_training" / "_macro_recovery_worker.py"


def _run(mode: str, *, iters: int, checkpoint: Path, run_dir: Path, out: Path) -> dict:
    result = subprocess.run(
        [
            sys.executable, str(WORKER), "--mode", mode, "--iters", str(iters),
            "--seed", "4242", "--checkpoint", str(checkpoint), "--run-dir", str(run_dir),
            "--out", str(out),
        ],
        cwd=str(ROOT), capture_output=True, text=True, timeout=3600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_profile_c_durable_checkpoint_new_process_resume_matches_control(tmp_path: Path):
    control_checkpoint = tmp_path / "control.pt"
    control = _run(
        "train", iters=4, checkpoint=control_checkpoint,
        run_dir=tmp_path / "control-dir", out=tmp_path / "control.json",
    )
    assert control["checkpoint_exists"] is True

    split_checkpoint = tmp_path / "split.pt"
    first = _run(
        "train", iters=2, checkpoint=split_checkpoint,
        run_dir=tmp_path / "split-a", out=tmp_path / "split-a.json",
    )
    assert first["state"]["iteration"] == 2
    resumed = _run(
        "resume", iters=2, checkpoint=split_checkpoint,
        run_dir=tmp_path / "split-b", out=tmp_path / "split-b.json",
    )
    # A genuinely new process loaded the durable checkpoint (strict resume classification).
    assert resumed["mode"] == "resume"
    assert resumed["resume_kind"] in {"exact_resume", "compatible_resume"}
    assert resumed["state_on_load"]["iteration"] == 2
    assert resumed["state_on_load"]["training_semantics_profile"] == "macro_decision_v1"
    assert resumed["state"]["training_semantics_profile"] == "macro_decision_v1"
    assert resumed["state"]["policy_regularizer_mode"] == "legacy_token_kl_uniform_v1"

    control_state = control["state"]
    resumed_state = resumed["state"]
    assert resumed_state["iteration"] == control_state["iteration"] == 4
    assert resumed_state["durable_updates"] == control_state["durable_updates"] == 4
    assert resumed_state["learner_updates"] == control_state["learner_updates"]
    assert resumed_state["durable_learner_updates"] == control_state["durable_learner_updates"]
    assert resumed_state["completed_games_total"] == control_state["completed_games_total"]
    assert resumed_state["env_steps"] == control_state["env_steps"]
    assert resumed_state["game_steps"] == control_state["game_steps"]
    # Model, optimizer and RNG restoration are verified by bitwise continuation equality.
    assert resumed_state["agent_resume_digest"] == control_state["agent_resume_digest"]
    assert resumed_state["agent_merged_digest"] == control_state["agent_merged_digest"]
    assert resumed_state["optimizer_digest"] == control_state["optimizer_digest"]
    assert resumed_state["rng_digest"] == control_state["rng_digest"]
    for key, value in control_state["last_metrics"].items():
        assert resumed_state["last_metrics"][key] == pytest.approx(value, abs=0.0), key
    # The durable checkpoint must carry the profile and be reloadable again.
    assert first["state"]["training_semantics_profile"] == "macro_decision_v1"
