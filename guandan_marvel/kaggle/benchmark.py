# SPDX-License-Identifier: MIT
"""M08 single/dual benchmark comparison and acceptance math."""
from __future__ import annotations

from collections.abc import Mapping

BENCHMARK_SUMMARY_SCHEMA = "guandan-marvel-m08-benchmark-summary-v1"
BENCHMARK_COMPARISON_SCHEMA = "guandan-marvel-m08-benchmark-comparison-v1"


def compare_benchmark_summaries(
    single: Mapping[str, object], dual: Mapping[str, object],
) -> dict[str, object]:
    for name, payload in (("single", single), ("dual", dual)):
        if payload.get("schema") != BENCHMARK_SUMMARY_SCHEMA:
            raise ValueError(f"{name} benchmark summary schema mismatch")
    if single.get("control") != "single_gpu" or int(single.get("world_size", 0)) != 1:
        raise ValueError("single benchmark is not the world-size-1 control")
    if dual.get("control") != "dual_gpu_ddp" or int(dual.get("world_size", 0)) != 2:
        raise ValueError("dual benchmark is not the world-size-2 DDP control")
    for key in ("comparison_id", "workload_hash", "warmup_iterations", "measured_iterations"):
        if single.get(key) != dual.get(key):
            raise ValueError(f"single/dual benchmark mismatch for {key}")
    if single.get("measured_completed_games") != dual.get("measured_completed_games"):
        raise ValueError("single/dual measured completed-game workload differs")
    target = float((single.get("benchmark_contract") or {}).get("speedup_target", 1.5))
    if target != float((dual.get("benchmark_contract") or {}).get("speedup_target", target)):
        raise ValueError("single/dual speedup targets differ")

    def ratio(single_key: str, dual_key: str, *, inverse: bool = False) -> float:
        left = float(single[single_key])
        right = float(dual[dual_key])
        if left <= 0.0 or right <= 0.0:
            raise ValueError(f"benchmark values must be positive: {single_key}/{dual_key}")
        return left / right if not inverse else right / left

    session_speedup = ratio("session_elapsed_seconds", "session_elapsed_seconds")
    training_core_speedup = ratio("training_core_seconds", "training_core_seconds")
    games_throughput_speedup = ratio(
        "core_completed_games_per_second", "core_completed_games_per_second", inverse=True,
    )
    env_throughput_speedup = ratio("core_env_steps_per_second", "core_env_steps_per_second", inverse=True)
    game_step_throughput_speedup = ratio(
        "core_game_steps_per_second", "core_game_steps_per_second", inverse=True,
    )
    health = {
        "single": dict(single.get("health_totals") or {}),
        "dual": dict(dual.get("health_totals") or {}),
    }
    health_ok = all(float(value) == 0.0 for row in health.values() for value in row.values())
    passes = bool(health_ok and session_speedup >= target and training_core_speedup >= target)
    return {
        "schema": BENCHMARK_COMPARISON_SCHEMA,
        "comparison_id": single["comparison_id"],
        "workload_hash": single["workload_hash"],
        "measured_completed_games": single["measured_completed_games"],
        "speedup_target": target,
        "session_speedup": session_speedup,
        "training_core_speedup": training_core_speedup,
        "games_throughput_speedup": games_throughput_speedup,
        "env_throughput_speedup": env_throughput_speedup,
        "game_step_throughput_speedup": game_step_throughput_speedup,
        "scaling_efficiency": session_speedup / 2.0,
        "health": health,
        "health_ok": health_ok,
        "passes": passes,
        "remote_verified": False,
    }


__all__ = [
    "BENCHMARK_COMPARISON_SCHEMA", "BENCHMARK_SUMMARY_SCHEMA",
    "compare_benchmark_summaries",
]
