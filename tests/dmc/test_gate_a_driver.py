"""Gate A driver tests (stage 6): lane matrix, gates, resume chain, tiny run.

The decisive checks:

* the four-lane matrix is the one ``docs/dmc_prompts/06_gate_a_kaggle.md``
  pre-declares (rule / m09 / random / legacy, 512 games each, 2/4/2/2
  rotations, declared drivers) and the plan pins the frozen m09 identity;
* :func:`dmc_gate_a.lanes.play_lane_games` plays real ε=0 greedy games against
  the shipped rule/random row agents with the Gate L pairing, clean health
  counters and per-game determinism;
* :func:`dmc_gate_a.runner.evaluate_gates` implements the pre-declared
  PASS/ABORT arithmetic (including "must beat Gate L's 65.2%") and never
  upgrades a skipped lane into a PASS;
* :func:`dmc_gate_a.runner.run_notebook` performs a full tiny Gate A run on
  CPU — parallel trainer, interval checkpoints, per-lane evidence, hashed
  RESULT.json / RESULTS_MANIFEST.json, the results zip and the resume copy;
* a second session resumes from the emitted ``dmc_gate_a_resume.pt``;
* the plan/config coupling (wall budget, checkpoint cadence) is enforced.
"""

from __future__ import annotations

import dataclasses
import json
import zipfile
from pathlib import Path

import pytest
import torch

from guandan_marvel.backends import get_game_backend
from guandan_marvel.config import ModelConfig, PPOConfig, TrainConfig
from guandan_marvel.dmc.q_agent import DMCAgent
from scripts.build_dmc_gate_a_delivery import (
    BOUNDARIES,
    BUDGETS,
    GATES,
    SCHEDULE,
    TOPOLOGY,
    build_plan,
)
from tools.dmc_gate_a import m09 as m09_module
from tools.dmc_gate_a.lanes import (
    LANE_DRIVERS,
    LANE_ROTATIONS,
    ROTATION_SEATS_2,
    ROTATION_SEATS_4,
    play_lane_games,
)
from tools.dmc_gate_a.runner import (
    PLAN_SCHEMA,
    RESULTS_ZIP_NAME,
    RESUME_FILENAME,
    canonical,
    evaluate_gates,
    run_notebook,
    stage7_projection,
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


# ---------------------------------------------------------------------------
# Declared shape
# ---------------------------------------------------------------------------
def test_lane_matrix_is_the_pre_declared_four_arms() -> None:
    lanes = {row["kind"]: row for row in SCHEDULE["lanes"]}
    assert list(lanes) == ["rule", "m09", "random", "legacy"]
    assert SCHEDULE["eval_order"] == ["rule", "m09", "random", "legacy"]
    for row in lanes.values():
        assert row["games"] == 512
    assert lanes["m09"]["rotations"] == 4  # 128 deals x 4 rotations
    assert lanes["rule"]["rotations"] == 2
    assert lanes["m09"]["driver"] == "env_batch_reference"
    assert lanes["rule"]["driver"] == "dmc_env_fast"
    assert LANE_ROTATIONS["m09"] == ROTATION_SEATS_4
    assert LANE_ROTATIONS["rule"] == ROTATION_SEATS_2
    assert LANE_DRIVERS["m09"] == "env_batch_reference"
    # The declared estimates must fit inside the declared eval reserve.
    assert sum(row["expected_seconds"] for row in lanes.values()) <= BUDGETS["eval_reserve_seconds"]


def test_budget_arithmetic_is_a_nine_hour_session() -> None:
    assert BUDGETS["platform_timeout_seconds"] == 32_400  # 9 h
    assert BUDGETS["total_seconds"] + 300 <= BUDGETS["platform_timeout_seconds"]
    assert (
        BUDGETS["train_seconds"] + BUDGETS["eval_reserve_seconds"]
        == BUDGETS["total_seconds"]
    )
    assert TOPOLOGY["learner_device"] == "cuda:0"
    assert TOPOLOGY["collector_device"] == "cuda:1"
    assert TOPOLOGY["required_gpu_count"] == 2


def test_delivery_plan_pins_the_gate_a_shape() -> None:
    repo = Path(__file__).resolve().parents[2]
    m09_block = {
        "loader": "official",
        "frozen_candidate": {"sha256": "x", "iteration": 2000},
        "runtime": {"source_root": "/kaggle/working/guandan-marvel-source"},
        "legacy_anchor": {"sha256": "y"},
    }
    plan = build_plan(repo, "TESTCOMMIT", m09_block)
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["topology"] == TOPOLOGY
    assert plan["budgets"] == BUDGETS
    assert plan["gates"] == GATES
    assert plan["boundaries"] == BOUNDARIES
    assert plan["env_driver"]["lane_exception"]["lane"] == "m09"
    assert plan["resume"]["filename"] == RESUME_FILENAME
    assert plan["m09_lane"] == m09_block
    assert plan["plan_hash"] == canonical({k: v for k, v in plan.items() if k != "plan_hash"})


# ---------------------------------------------------------------------------
# Gate arithmetic
# ---------------------------------------------------------------------------
def _lane(rate: float | None, games: int = 512) -> dict:
    return {"win_rate": rate, "games": games, "games_completed": games,
            "illegal": 0, "truncated": 0, "nonfinite": 0}


CLEAN = {"illegal": 0, "truncated": 0, "nonfinite": 0}


def test_evaluate_gates_pass_requires_all_four_arms() -> None:
    result = evaluate_gates(
        GATES,
        lanes={"rule": _lane(0.78), "m09": _lane(0.62), "random": _lane(0.93),
               "legacy": _lane(0.55)},
        train_health=CLEAN,
        eval_health=CLEAN,
        skipped=[],
    )
    assert result["overall_verdict"] == "PASS"
    assert result["vs_rule"]["above_gate_l_reference"] is True
    assert result["vs_rule"]["pass"] is True
    assert result["h2h_m09"]["pass"] is True


def test_evaluate_gates_rule_must_exceed_gate_l_reference() -> None:
    """Standing still at Gate L's 65.2% is an ABORT, not a pass.

    The pre-declared table makes the two conditions nested (the 68% abort line
    is stricter than the 65.2% reference), so this asserts both are reported
    and that a non-rising run can never pass.
    """
    result = evaluate_gates(
        GATES,
        lanes={"rule": _lane(0.652), "m09": _lane(0.62), "random": _lane(0.93),
               "legacy": _lane(0.5)},
        train_health=CLEAN,
        eval_health=CLEAN,
        skipped=[],
    )
    assert result["vs_rule"]["above_gate_l_reference"] is False
    assert result["vs_rule"]["pass"] is False
    assert result["vs_rule"]["abort"] is True
    assert result["vs_rule"]["disposition"] == "ABORT"
    assert result["overall_verdict"] == "ABORT"


def test_evaluate_gates_abort_lines() -> None:
    result = evaluate_gates(
        GATES,
        lanes={"rule": _lane(0.66), "m09": _lane(0.60), "random": _lane(0.93),
               "legacy": _lane(0.5)},
        train_health=CLEAN,
        eval_health=CLEAN,
        skipped=[],
    )
    assert result["vs_rule"]["abort"] is True
    assert result["overall_verdict"] == "ABORT"
    assert "differential-test" in result["abort_disposition"]

    h2h = evaluate_gates(
        GATES,
        lanes={"rule": _lane(0.80), "m09": _lane(0.54), "random": _lane(0.93),
               "legacy": _lane(0.5)},
        train_health=CLEAN,
        eval_health=CLEAN,
        skipped=[],
    )
    assert h2h["h2h_m09"]["abort"] is True
    assert h2h["overall_verdict"] == "ABORT"


def test_evaluate_gates_never_upgrades_a_skipped_lane() -> None:
    result = evaluate_gates(
        GATES,
        lanes={"rule": _lane(0.80), "random": _lane(0.93), "legacy": _lane(0.5)},
        train_health=CLEAN,
        eval_health=CLEAN,
        skipped=["m09"],
    )
    assert result["h2h_m09"]["measured"] is None
    assert result["h2h_m09"]["pass"] is False
    assert result["overall_verdict"] == "INCONCLUSIVE"


def test_evaluate_gates_health_violation_blocks_pass() -> None:
    result = evaluate_gates(
        GATES,
        lanes={"rule": _lane(0.80), "m09": _lane(0.62), "random": _lane(0.93),
               "legacy": _lane(0.5)},
        train_health={**CLEAN, "truncated": 1},
        eval_health=CLEAN,
        skipped=[],
    )
    assert result["health"]["pass"] is False
    assert result["overall_verdict"] == "FAIL"


def test_stage7_projection_is_arithmetic_only() -> None:
    projection = stage7_projection(
        games_completed=10_000, samples_pushed=800_000, train_seconds=10_000.0,
    )
    assert projection["games_per_second"] == pytest.approx(1.0)
    assert projection["samples_per_game"] == pytest.approx(80.0)
    assert projection["games_to_target"] == pytest.approx(3_125_000.0)
    assert projection["days_to_target_at_2_gpu"] == pytest.approx(
        250_000_000 / 80.0 / 86_400.0,
    )
    assert stage7_projection(games_completed=0, samples_pushed=0, train_seconds=1.0)[
        "status"
    ] == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------
def test_play_lane_games_rule_pairing_and_determinism() -> None:
    agent = _tiny_agent()
    first = play_lane_games(agent, kind="rule", games=2, base_seed=100, device="cpu")
    summary = first["summary"]
    assert summary["games"] == 2
    assert summary["independent_deals"] == 1
    assert summary["illegal"] == 0
    assert summary["truncated"] == 0
    assert summary["nonfinite"] == 0
    assert summary["driver"] == "dmc_env_fast"
    rows = first["rows"]
    assert [row["rotation"] for row in rows] == [0, 1]
    assert rows[0]["deal_seed"] == rows[1]["deal_seed"]
    assert rows[0]["dmc_seats"] == [0, 2]
    assert rows[1]["dmc_seats"] == [1, 3]

    again = play_lane_games(agent, kind="rule", games=2, base_seed=100, device="cpu")
    assert [row["action_sha256"] for row in again["rows"]] == [
        row["action_sha256"] for row in rows
    ]


def test_play_lane_games_random_health() -> None:
    agent = _tiny_agent()
    result = play_lane_games(agent, kind="random", games=2, base_seed=200, device="cpu")
    summary = result["summary"]
    assert summary["games_completed"] == 2
    assert summary["illegal"] == 0
    assert summary["truncated"] == 0
    assert summary["nonfinite"] == 0
    assert summary["win_rate"] is not None


def test_lane_requires_its_frozen_artifact() -> None:
    agent = _tiny_agent()
    with pytest.raises(Exception):
        play_lane_games(agent, kind="m09", games=2, base_seed=300, device="cpu")
    with pytest.raises(Exception):
        play_lane_games(agent, kind="legacy", games=2, base_seed=400, device="cpu")


# ---------------------------------------------------------------------------
# Tiny end-to-end run (CPU, injected loaders)
# ---------------------------------------------------------------------------
class _ShimArtifact:
    def __init__(self, sha256: str, iteration: int) -> None:
        self.sha256 = sha256
        self.iteration = iteration
        self.architecture = "v1.3"
        self.algorithm = "ippo"
        self.protocol_versions = {"action": "GD-ACTION-0.1", "environment": "GD-ENV-0.1"}
        self.path = Path("shim.pt")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tiny_plan(
    tmp_path: Path, *, total_games: int = 4, train_seconds: int = 900,
) -> Path:
    dataset = tmp_path / "input"
    dataset.mkdir()
    native_rows = []
    for relative in (
        "nanopokerlib/cpp/bindings.cpp",
        "nanopokerlib/cpp/core/types.hpp",
        "nanopokerlib/cpp/guandan/guandan.cpp",
        "nanopokerlib/cpp/guandan/guandan.hpp",
    ):
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8") + b"\n")
        native_rows.append({"manifest_path": relative, "payload_path": relative,
                            "sha256": _sha256(path), "bytes": path.stat().st_size})
    binary = dataset / "nanopokerlib/_guandan_cpp.cpython-313-x86_64-linux-gnu.so"
    binary.write_bytes(b"fake-native-runtime\n")
    candidate = dataset / "m09/ippo_iter002000_final_candidate.pt"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"fake-candidate\n")
    anchor = dataset / "m09/legacy_v1_anchor.pt"
    anchor.write_bytes(b"fake-anchor\n")

    manifest_hash = m09_module.native_source_manifest(dataset, native_rows)["hash"]
    config = {
        "schema": "guandan-marvel-dmc-train-config-v1",
        "run_name": "dmc_gate_a_tiny",
        "output_dir": None,
        "device": "cpu",
        "seed": 0,
        "env_seed": 0,
        "num_envs": 2,
        "total_games": total_games,
        "games_per_cycle": 2,
        "gradient_replay_ratio": 1.0,
        "batch_size": 64,
        "buffer_capacity": 8192,
        "max_obs_tokens": 1024,
        "epsilon_start": 0.10,
        "epsilon_final": 0.02,
        "epsilon_decay_samples": 5_000_000,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "checkpoint_interval_cycles": 1,
        "max_wall_seconds": train_seconds,
        "heartbeat_seconds": 5.0,
        "context_schedule": None,
        "model": dict(TINY_MODEL),
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "budgets": {"total_seconds": 3600, "train_seconds": train_seconds,
                    "eval_reserve_seconds": 600, "platform_timeout_seconds": 4200},
        "topology": dict(TOPOLOGY),
        "schedule": {
            "checkpoint_interval_games": 2,
            "eval_order": ["rule", "random"],
            "lanes": [
                {"kind": "rule", "games": 2, "seed_base": 1000, "rotations": 2,
                 "driver": "dmc_env_fast", "expected_seconds": 60},
                {"kind": "random", "games": 2, "seed_base": 2000, "rotations": 2,
                 "driver": "dmc_env_fast", "expected_seconds": 60},
            ],
        },
        "gates": json.loads(json.dumps(GATES)),
        "boundaries": dict(BOUNDARIES),
        "env_driver": {"name": "dmc_env_fast", "version": "dmc-env-fast-v1"},
        "resume": {"mode": "optional_input", "input_root": "/kaggle/input",
                   "filename": RESUME_FILENAME, "requirement": "optional"},
        "m09_lane": {
            "loader": "official",
            "frozen_candidate": {
                "path": "m09/ippo_iter002000_final_candidate.pt",
                "sha256": _sha256(candidate),
                "iteration": 2000,
                "algorithm": "ippo",
                "extension_sha256": _sha256(binary),
                "extension_filename": binary.name,
                "embedded_source_sha256": manifest_hash,
                "backend_identity_hash": "test-identity",
                "python_version": "3.13.15",
                "python_cache_tag": "cpython-313",
            },
            "runtime": {"source_root": "/kaggle/working/guandan-marvel-source",
                        "binary": f"nanopokerlib/{binary.name}",
                        "cpp_sources": native_rows},
            "legacy_anchor": {"path": "m09/legacy_v1_anchor.pt",
                              "sha256": _sha256(anchor)},
        },
        "deviations": ["tiny test plan"],
    }
    plan["plan_hash"] = canonical({k: v for k, v in plan.items() if k != "plan_hash"})
    (dataset / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (dataset / "gate_a_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return dataset


def _shim_loader(dataset, plan, *, device="cpu"):
    candidate = plan["m09_lane"]["frozen_candidate"]
    return _ShimArtifact(candidate["sha256"], int(candidate["iteration"]))


def _shim_legacy_loader(dataset, plan):
    return _ShimArtifact(plan["m09_lane"]["legacy_anchor"]["sha256"], 0)


def test_run_notebook_tiny_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        m09_module, "check_runtime_identity",
        lambda plan: {"identity_hash": "test-identity", "extension_path": "shim",
                      "python_version": "3.13.15", "python_cache_tag": "cpython-313",
                      "compiler": "test"},
    )
    dataset = _tiny_plan(tmp_path)
    run_root = tmp_path / "run_root"
    result = run_notebook(
        dataset=dataset,
        run_root=run_root,
        m09_loader=_shim_loader,
        legacy_loader=_shim_legacy_loader,
        resume_input_root=tmp_path / "no-input",
        log=lambda _message: None,
    )
    assert result["schema"] == "guandan-dmc-gate-a-result-v1"
    assert result["m09_lane"]["loader"] == "injected"
    assert result["session"]["games_completed_this_session"] == 4
    assert result["resume"]["status"] == "FRESH"
    assert set(result["lanes"]) == {"rule", "random"}
    # Mechanics, not luck: a random-init tiny agent cannot pass a 75% gate, and
    # the two lanes this tiny plan omits must never be scored as passes.
    assert result["gates"]["lanes_skipped"] == []
    assert result["gates"]["h2h_m09"]["measured"] is None
    assert result["gates"]["overall_verdict"] != "PASS"
    assert result["stage7_projection"]["status"] == "RECORD_ONLY"
    assert result["hash"] == canonical({k: v for k, v in result.items() if k != "hash"})

    artifacts = run_root / "artifacts"
    assert (artifacts / "RESULT.json").is_file()
    assert (artifacts / "recovery_index.json").is_file()
    assert (run_root / "RESULTS_MANIFEST.json").is_file()
    assert (run_root / RESUME_FILENAME).is_file()
    assert (run_root / "run" / "checkpoint_final.pt").is_file()

    manifest = json.loads((run_root / "RESULTS_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["plan_hash"] == result["plan_hash"]
    # "complete" is relative to the plan's own declared matrix (this tiny plan
    # declares two lanes and both ran their full game count).
    assert manifest["evidence_complete"] is True
    assert manifest["omissions"] == []
    assert manifest["hash"] == canonical({k: v for k, v in manifest.items() if k != "hash"})
    members = {row["path"] for row in manifest["files"]}
    assert RESUME_FILENAME in members
    assert "artifacts/eval/rule.summary.json" in members

    with zipfile.ZipFile(run_root / RESULTS_ZIP_NAME) as bundle:
        names = set(bundle.namelist())
    assert names == {"RESULTS_MANIFEST.json", *members}


def test_second_session_resumes_from_the_emitted_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        m09_module, "check_runtime_identity",
        lambda plan: {"identity_hash": "test-identity", "extension_path": "shim",
                      "python_version": "3.13.15", "python_cache_tag": "cpython-313",
                      "compiler": "test"},
    )
    dataset = _tiny_plan(tmp_path, total_games=8, train_seconds=1)
    first_root = tmp_path / "session1"
    run_notebook(
        dataset=dataset, run_root=first_root, m09_loader=_shim_loader,
        legacy_loader=_shim_legacy_loader, resume_input_root=tmp_path / "no-input",
        log=lambda _message: None,
    )
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / RESUME_FILENAME).write_bytes((first_root / RESUME_FILENAME).read_bytes())

    second_root = tmp_path / "session2"
    result = run_notebook(
        dataset=dataset, run_root=second_root, m09_loader=_shim_loader,
        legacy_loader=_shim_legacy_loader, resume_input_root=upload,
        log=lambda _message: None,
    )
    first = json.loads((first_root / "artifacts" / "RESULT.json").read_text(encoding="utf-8"))
    assert first["status"] == "budget_stopped"
    # One coordinator cycle before the budget stop, plus the cycle the
    # collector had already started: the session summary and the final
    # checkpoint both carry 4 games (documented in-flight behaviour), while
    # ``cycles`` counts only the cycles the coordinator closed.
    assert first["session"]["games_completed_this_session"] == 4
    assert first["session"]["games_completed_cumulative"] == 4
    assert first["session"]["cycles"] == 1
    assert result["resume"]["status"] == "RESUMED"
    assert result["resume"]["games_completed"] == 4
    assert result["resume"]["cycles"] == 1
    assert result["session"]["games_completed_this_session"] == 4
    assert result["session"]["games_completed_cumulative"] == 8
    # One coordinator cycle on top of the resumed count; the second cycle was
    # already in flight when the budget stop landed (same behaviour as above).
    assert result["session"]["cycles"] == 2
    assert result["session"]["stop_reason"] == "budget_stopped"


def test_plan_and_config_coupling_is_enforced(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        m09_module, "check_runtime_identity",
        lambda plan: {"identity_hash": "test-identity", "extension_path": "shim",
                      "python_version": "3.13.15", "python_cache_tag": "cpython-313",
                      "compiler": "test"},
    )
    dataset = _tiny_plan(tmp_path)
    plan = json.loads((dataset / "plan.json").read_text(encoding="utf-8"))
    plan["budgets"]["train_seconds"] = 1234
    plan["plan_hash"] = canonical({k: v for k, v in plan.items() if k != "plan_hash"})
    (dataset / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(Exception):
        run_notebook(
            dataset=dataset, run_root=tmp_path / "bad", m09_loader=_shim_loader,
            legacy_loader=_shim_legacy_loader, resume_input_root=tmp_path / "no-input",
            log=lambda _message: None,
        )
