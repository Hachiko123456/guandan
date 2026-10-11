from __future__ import annotations

import json
from pathlib import Path

from guandan_marvel.kaggle.benchmark import compare_benchmark_summaries
from guandan_marvel.kaggle.common import canonical_hash
from guandan_marvel.kaggle.profiles import build_train_config
from guandan_marvel.kaggle.worker import _write_benchmark_summary
from guandan_marvel.training.smoke import PythonReferenceSmokeGame


def _profiles() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "configs/marvel_kaggle_profiles.json").read_text(encoding="utf-8"))["profiles"]


def test_single_and_dual_controls_share_one_global_workload_contract() -> None:
    profiles = _profiles()
    single = profiles["single_gpu_control"]
    dual = profiles["dual_gpu_control"]
    assert single["training"] == dual["training"]
    assert single["algorithm"] == dual["algorithm"] == "ippo"
    assert single["ddp"]["world_size"] == 1
    assert dual["ddp"]["world_size"] == 2
    assert single["benchmark"]["comparison_id"] == dual["benchmark"]["comparison_id"]
    assert single["benchmark"]["workload_hash"] == dual["benchmark"]["workload_hash"]
    expected = canonical_hash({"algorithm": single["algorithm"], "training": single["training"]})
    assert single["benchmark"]["workload_hash"] == expected
    assert single["benchmark"]["warmup_iterations"] == 1
    assert single["benchmark"]["measured_iterations"] == 2
    assert single["training"]["target_iterations"] == 3
    assert single["training"]["evaluation_interval"] == 0
    assert single["training"]["checkpoint_interval"] == 3
    assert single["benchmark"]["persist_checkpoint"] is False
    assert dual["benchmark"]["persist_checkpoint"] is False
    assert single["benchmark"]["emit_recovery"] is False
    assert dual["benchmark"]["emit_recovery"] is False


def test_control_configs_partition_the_same_initial_environment_stream(tmp_path: Path) -> None:
    profiles = _profiles()
    game = PythonReferenceSmokeGame(512)
    single = build_train_config(
        profiles["single_gpu_control"], game=game, run_root=tmp_path / "single", local_cpu_test=True,
    )
    dual = build_train_config(
        profiles["dual_gpu_control"], game=game, run_root=tmp_path / "dual", local_cpu_test=True,
    )
    assert single.rollout_batch_size == dual.rollout_batch_size == 4
    assert single.collect_batch_size == dual.collect_batch_size == 4
    assert single.environment_rank_seed_stride == dual.environment_rank_seed_stride == 2
    assert single.training_data_generator_version == dual.training_data_generator_version == "guandan-marvel-m08-partitioned-prng-v1"
    assert single.ddp.enabled is False
    assert dual.ddp.enabled is True


def test_benchmark_summary_excludes_warmup_and_reports_core_throughput(tmp_path: Path) -> None:
    profile = _profiles()["single_gpu_control"]
    rows = [
        {
            "statistics/iterations": 1, "statistics/completed_games_total": 4,
            "statistics/env_steps_total": 1000, "statistics/game_steps_total": 400,
            "perf/rollout_seconds": 4.0, "perf/learner_seconds": 2.0,
        },
        {
            "statistics/iterations": 2, "statistics/completed_games_total": 8,
            "statistics/env_steps_total": 2100, "statistics/game_steps_total": 840,
            "perf/rollout_seconds": 3.0, "perf/learner_seconds": 2.0,
        },
        {
            "statistics/iterations": 3, "statistics/completed_games_total": 12,
            "statistics/env_steps_total": 3300, "statistics/game_steps_total": 1320,
            "perf/rollout_seconds": 2.5, "perf/learner_seconds": 1.5,
        },
    ]
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    path = _write_benchmark_summary(
        tmp_path, profile=profile, world_size=1, session_start_iteration=0,
        target_iterations=3, session_elapsed_seconds=20.0,
    )
    assert path is not None
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["measured_iterations"] == [2, 3]
    assert summary["training_core_seconds"] == 9.0
    assert summary["measured_completed_games"] == 8
    assert summary["measured_env_steps"] == 2300
    assert summary["measured_game_steps"] == 920
    assert summary["world_size"] == 1


def test_benchmark_comparison_requires_matching_workload_and_target() -> None:
    base = {
        "schema": "guandan-marvel-m08-benchmark-summary-v1",
        "comparison_id": "x", "workload_hash": "a" * 64,
        "warmup_iterations": 1, "measured_iterations": [2, 3],
        "measured_completed_games": 8,
        "session_elapsed_seconds": 20.0, "training_core_seconds": 10.0,
        "core_completed_games_per_second": 0.8,
        "core_env_steps_per_second": 100.0, "core_game_steps_per_second": 40.0,
        "health_totals": {"health/illegal_actions": 0.0},
        "benchmark_contract": {"speedup_target": 1.5},
    }
    single = {**base, "control": "single_gpu", "world_size": 1}
    dual = {
        **base, "control": "dual_gpu_ddp", "world_size": 2,
        "session_elapsed_seconds": 10.0, "training_core_seconds": 5.0,
        "core_completed_games_per_second": 1.6,
        "core_env_steps_per_second": 200.0, "core_game_steps_per_second": 80.0,
    }
    result = compare_benchmark_summaries(single, dual)
    assert result["session_speedup"] == 2.0
    assert result["training_core_speedup"] == 2.0
    assert result["passes"] is True


def test_full_profile_has_bounded_checkpoint_and_evaluation_retention() -> None:
    full = _profiles()["full"]["training"]
    assert full["checkpoint_keep_last"] == 2
    assert full["evaluation_keep_last"] == 2
    assert full["checkpoint_interval"] == 25
    assert full["evaluation_interval"] == 100
