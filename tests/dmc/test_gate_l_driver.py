"""Gate L driver tests (stage 5): tiny end-to-end notebook run + import hygiene.

The decisive checks:

* :func:`dmc_gate_l.evaluate.play_eval_games` plays real ε=0 greedy games
  against the shipped rule/random row agents with paired rotations, clean
  health counters and per-game determinism;
* :func:`dmc_gate_l.runner.run_notebook` performs a full tiny Gate L run on
  CPU — budgeted training cycle loop, interval checkpoints, mini + final
  evaluations, gate evaluation, hashed RESULT.json / RESULTS_MANIFEST.json and
  the results zip;
* the whole driver import chain never touches a native/C++ module (Gate L
  runs the explicit ``python_reference`` backend on Kaggle; there is no C++
  fallback to silently depend on).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from guandan_marvel.backends import get_game_backend
from guandan_marvel.config import ModelConfig, PPOConfig, TrainConfig
from guandan_marvel.dmc.q_agent import DMCAgent
from scripts.build_dmc_gate_l_delivery import BOUNDARIES, GATES, build_plan
from tools.dmc_gate_l.evaluate import play_eval_games
from tools.dmc_gate_l.runner import (
    PLAN_SCHEMA,
    RESULTS_ZIP_NAME,
    canonical,
    progress_extrapolation,
    run_notebook,
)

GAME = get_game_backend("python_reference")

TINY_MODEL = {
    "history_dim": 16,
    "feature_dim": 24,
    "decoder_layers": 1,
    "decoder_heads": 4,
    "state_layers": 1,
    "action_layers": 1,
    "fuse_layers": 1,
}


def _tiny_agent() -> DMCAgent:
    torch.manual_seed(0)
    return DMCAgent(TrainConfig(
        game=GAME,
        algo_name="dmc",
        device="cpu",
        ppo=PPOConfig(training_semantics_profile="dmc_v1"),
        model=ModelConfig(**TINY_MODEL),
    ))


def _tiny_dataset(
    tmp_path: Path, *, total_games: int = 6, games_per_cycle: int = 2,
    early_stop: dict | None = None,
) -> Path:
    config = {
        "schema": "guandan-marvel-dmc-train-config-v1",
        "run_name": "dmc_gate_l_tiny",
        "output_dir": None,
        "device": "cpu",
        "seed": 0,
        "env_seed": 0,
        "num_envs": 2,
        "total_games": total_games,
        "games_per_cycle": games_per_cycle,
        "gradient_replay_ratio": 1.0,
        "batch_size": 64,
        "buffer_capacity": 8192,
        "max_obs_tokens": 1024,
        "epsilon_start": 0.10,
        "epsilon_final": 0.02,
        "epsilon_decay_samples": 5000000,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "checkpoint_interval_cycles": 1,
        "max_wall_seconds": None,
        "heartbeat_seconds": 5.0,
        "context_schedule": None,
        "model": dict(TINY_MODEL),
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "budgets": {
            "total_seconds": 3600,
            "train_seconds": 3600,
            "eval_reserve_seconds": 600,
            "platform_timeout_seconds": 15000,
        },
        "schedule": {
            "checkpoint_interval_games": games_per_cycle,
            # fraction 0.0 makes the mini evaluation fire right after the first
            # cycle so the embedded-eval path is actually exercised.
            "mini_eval": {"fraction_of_train_budget": 0.0, "vs_rule_games": 2},
            "final_eval": {"vs_rule_games": 2, "vs_random_games": 2},
            "eval_seed_base": {
                "mini_vs_rule": 1000,
                "final_vs_rule": 2000,
                "final_vs_random": 3000,
            },
        },
        "gates": json.loads(json.dumps(GATES)),
        "boundaries": dict(BOUNDARIES),
        "deviations": ["tiny test plan"],
    }
    if early_stop is not None:
        plan["early_stop"] = dict(early_stop)
    plan["plan_hash"] = canonical({key: value for key, value in plan.items() if key != "plan_hash"})
    dataset = tmp_path / "input"
    dataset.mkdir()
    (dataset / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (dataset / "gate_l_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return dataset


# ---------------------------------------------------------------------------
# Stage 07A corrected early stop (progress extrapolation)
# ---------------------------------------------------------------------------
def test_progress_extrapolation_is_the_throughput_gate_line() -> None:
    """The stop line is the throughput gate itself: 12,000 games in 14,400 s."""
    below = progress_extrapolation(
        games_completed=1_000, elapsed_seconds=1_250.0, budget_seconds=14_400.0,
    )
    assert below["rate_games_per_second"] == pytest.approx(0.8)
    assert below["extrapolated_games"] == pytest.approx(11_520.0)
    assert below["extrapolated_games"] < 12_000

    on_track = progress_extrapolation(
        games_completed=1_000, elapsed_seconds=1_000.0, budget_seconds=14_400.0,
    )
    assert on_track["rate_games_per_second"] == pytest.approx(1.0)
    assert on_track["extrapolated_games"] == pytest.approx(14_400.0)
    assert on_track["extrapolated_games"] >= 12_000

    # 0.90 games/s projects to 12,960 games: 07A's intent is to keep running,
    # which is exactly what the floor does (the void 1.2 line would have stopped).
    keeps_running = progress_extrapolation(
        games_completed=900, elapsed_seconds=1_000.0, budget_seconds=14_400.0,
    )
    assert keeps_running["extrapolated_games"] == pytest.approx(12_960.0)
    assert keeps_running["extrapolated_games"] >= 12_000

    for bad in (
        {"games_completed": 0, "elapsed_seconds": 1.0, "budget_seconds": 1.0},
        {"games_completed": 10, "elapsed_seconds": 0.0, "budget_seconds": 1.0},
        {"games_completed": 10, "elapsed_seconds": 1.0, "budget_seconds": 0.0},
    ):
        with pytest.raises(ValueError):
            progress_extrapolation(**bad)


def test_delivery_plan_declares_driver_and_early_stop() -> None:
    repo = Path(__file__).resolve().parents[2]
    plan = build_plan(repo, "TESTCOMMIT")
    driver = plan["env_driver"]
    assert driver["status"] == "ENABLED"
    assert driver["version"] == "dmc-env-fast-v1"
    assert driver["differential_suite"] == "dmc-env-fast-differential-v1"
    assert driver["differential_games"] >= 200
    early = plan["early_stop"]
    assert early["rule"] == "progress_extrapolation"
    assert early["after_games"] == 1_000
    assert early["min_extrapolated_games"] == 12_000
    assert early["extrapolation_budget_seconds"] == plan["gates"]["throughput"]["budget_seconds"]
    assert early["min_extrapolated_games"] == plan["gates"]["throughput"]["min_games_in_budget"]
    assert "1.2" in early["supersedes"]  # the void line is named as superseded
    assert plan["plan_hash"] == canonical({k: v for k, v in plan.items() if k != "plan_hash"})


# ---------------------------------------------------------------------------
# Evaluation lanes
# ---------------------------------------------------------------------------
def test_play_eval_games_rows_rotations_and_determinism() -> None:
    agent = _tiny_agent()
    rule = play_eval_games(agent, opponent_kind="rule", games=2, base_seed=100, device="cpu")
    assert rule["summary"]["games"] == 2
    assert rule["summary"]["independent_deals"] == 1
    assert rule["summary"]["illegal"] == 0
    assert rule["summary"]["truncated"] == 0
    assert rule["summary"]["nonfinite"] == 0
    rows = rule["rows"]
    assert [row["rotation"] for row in rows] == [0, 1]
    assert rows[0]["seed"] == rows[1]["seed"]  # one deal, both partnerships
    assert rows[0]["dmc_seats"] == [0, 2]
    assert rows[1]["dmc_seats"] == [1, 3]
    for row in rows:
        assert row["status"] == "complete"
        assert row["steps"] > 0
        assert abs(row["dmc_team_reward"]) in (1.0, 2.0, 3.0)
        assert isinstance(row["win"], bool)
        assert row["win"] == (row["dmc_team_reward"] > 0)
        assert len(row["action_sha256"]) == 64
    again = play_eval_games(agent, opponent_kind="rule", games=2, base_seed=100, device="cpu")
    assert [row["action_sha256"] for row in again["rows"]] == [row["action_sha256"] for row in rows]
    random_arm = play_eval_games(agent, opponent_kind="random", games=2, base_seed=900, device="cpu")
    assert random_arm["summary"]["games"] == 2
    assert random_arm["rows"][0]["dmc_seats"] == [0, 2]


def test_play_eval_games_rejects_unknown_opponent() -> None:
    agent = _tiny_agent()
    try:
        play_eval_games(agent, opponent_kind="fabledan", games=1, base_seed=0, device="cpu")
    except ValueError as error:
        assert "opponent_kind" in str(error)
    else:
        raise AssertionError("unknown opponent_kind must be rejected")


# ---------------------------------------------------------------------------
# Full tiny run
# ---------------------------------------------------------------------------
def test_run_notebook_end_to_end_tiny(tmp_path: Path) -> None:
    dataset = _tiny_dataset(tmp_path)
    run_root = tmp_path / "run_root"
    result = run_notebook(dataset=dataset, run_root=run_root, log=lambda text: None)

    assert result["status"] == "complete"
    assert result["training"]["games_completed"] == 6
    assert result["training"]["health"]["illegal"] == 0
    assert result["training"]["health"]["nonfinite"] == 0
    assert result["training"]["health"]["truncated"] == 0
    assert result["evaluations"]["mini_vs_rule"]["games"] == 2
    assert result["evaluations"]["final_vs_rule"]["games"] == 2
    assert result["evaluations"]["final_vs_random"]["games"] == 2
    gates = result["gates"]
    assert gates["overall_verdict"] in ("PASS", "FAIL")
    for key in ("health", "q_calibration", "vs_rule", "vs_random", "throughput"):
        assert key in gates
        assert isinstance(gates[key]["pass"], bool)
    assert gates["health"]["pass"] is True

    stored = json.loads((run_root / "artifacts" / "RESULT.json").read_text(encoding="utf-8"))
    assert stored["hash"] == result["hash"]
    assert canonical({key: value for key, value in stored.items() if key != "hash"}) == stored["hash"]

    metrics = [
        json.loads(line)
        for line in (run_root / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metrics) == 3
    assert all(row["schema"] == "guandan-marvel-dmc-metrics-v1" for row in metrics)

    checkpoints = sorted((run_root / "run").glob("checkpoint_*.pt"))
    assert len(checkpoints) == 4  # interval 1 -> cycles 1/2/3 + final

    manifest = json.loads((run_root / "RESULTS_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["evidence_complete"] is True
    assert manifest["mode"] == "full"
    assert canonical({key: value for key, value in manifest.items() if key != "hash"}) == manifest["hash"]
    with zipfile.ZipFile(run_root / RESULTS_ZIP_NAME) as bundle:
        names = set(bundle.namelist())
        assert "RESULTS_MANIFEST.json" in names
        assert "artifacts/RESULT.json" in names
        assert names - {"RESULTS_MANIFEST.json"} == {row["path"] for row in manifest["files"]}
        for row in manifest["files"]:
            assert hashlib.sha256(bundle.read(row["path"])).hexdigest() == row["sha256"]

    recovery = json.loads((run_root / "artifacts" / "recovery_index.json").read_text(encoding="utf-8"))
    assert recovery["replay_reset_expected"] is True
    assert any(Path(row["path"]).name == "checkpoint_final.pt" for row in recovery["checkpoints"])
    # No early_stop block in the plan -> the stage-07A check stays inert.
    assert result["early_stop"] is None


def test_run_notebook_honours_the_early_stop_rule(tmp_path: Path) -> None:
    """A plan whose projected games miss the floor stops training, keeps evals.

    ``min_extrapolated_games`` is set absurdly high so the rule must fire on the
    first check: this pins the shipped runner behaviour, not the arithmetic
    (``test_progress_extrapolation_is_the_throughput_gate_line`` covers that).
    """
    dataset = _tiny_dataset(
        tmp_path, total_games=6, games_per_cycle=2,
        early_stop={
            "rule": "progress_extrapolation",
            "after_games": 1,
            "min_extrapolated_games": 10**9,
            "extrapolation_budget_seconds": 14400,
        },
    )
    run_root = tmp_path / "run_root"
    result = run_notebook(dataset=dataset, run_root=run_root, log=lambda text: None)

    assert result["status"] == "early_stop_extrapolation"
    record = result["early_stop"]
    assert record["triggered"] is True
    assert record["rule"] == "progress_extrapolation"
    assert record["games_completed"] >= 1
    assert record["threshold_games"] == 10**9
    assert record["extrapolated_games"] < record["threshold_games"]
    # Training stopped at the check cycle boundary instead of running to 6 games.
    assert result["training"]["games_completed"] == 2
    assert result["training"]["cycles"] == 1
    # The pre-declared evaluations still ran and the gates were still scored.
    assert result["evaluations"]["final_vs_rule"]["games"] == 2
    assert result["evaluations"]["final_vs_random"]["games"] == 2
    assert "throughput" in result["gates"]
    stored = json.loads((run_root / "artifacts" / "RESULT.json").read_text(encoding="utf-8"))
    assert stored["early_stop"] == record
    assert stored["status"] == "early_stop_extrapolation"
    early_stop_file = json.loads((run_root / "run" / "early_stop.json").read_text(encoding="utf-8"))
    assert early_stop_file["triggered"] is True


# ---------------------------------------------------------------------------
# Import hygiene: the Kaggle runtime has no native extension at all
# ---------------------------------------------------------------------------
def test_driver_import_chain_never_touches_cpp() -> None:
    repo = Path(__file__).resolve().parents[2]
    script = (
        "import sys\n"
        "class _Blocker:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if (fullname.startswith('guandan_marvel.cpp')\n"
        "                or fullname.startswith('guandan.cpp')\n"
        "                or fullname.startswith('nanopokerlib') or fullname == 'cpp_game'):\n"
        "            raise ImportError('native chain blocked: ' + fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "import dmc_gate_l.runner\n"
        "import dmc_gate_l.evaluate\n"
        "print('driver-import-ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "tools")])
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "driver-import-ok" in completed.stdout
