# SPDX-License-Identifier: MIT
"""Exact completed-game budget contracts for controlled Kaggle training runs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .common import canonical_hash

COMPLETED_GAME_BUDGET_SCHEMA = "guandan-marvel-completed-game-budget-v1"
SUPPORTED_COMPARISON_ALGORITHMS = ("ippo", "vrpo")


def build_completed_game_budget_contract(
    *,
    comparison_id: str,
    algorithm: str,
    target_completed_games: int,
    global_collect_batch_size: int,
    expected_iterations: int,
    selection_dataset_hash: str,
    peer_algorithms: Sequence[str] = SUPPORTED_COMPARISON_ALGORITHMS,
) -> dict[str, object]:
    body = {
        "schema": COMPLETED_GAME_BUDGET_SCHEMA,
        "comparison_id": str(comparison_id),
        "unit": "completed_games",
        "algorithm": str(algorithm),
        "peer_algorithms": [str(value) for value in peer_algorithms],
        "target_completed_games": int(target_completed_games),
        "global_collect_batch_size": int(global_collect_batch_size),
        "expected_iterations": int(expected_iterations),
        "selection_dataset_hash": str(selection_dataset_hash),
        "exact_budget_required": True,
        "wall_time_is_safety_stop_only": True,
    }
    contract = {**body, "contract_hash": canonical_hash(body)}
    validate_completed_game_budget_contract(contract)
    return contract


def validate_completed_game_budget_contract(
    payload: Mapping[str, object],
    *,
    expected_algorithm: str | None = None,
    global_collect_batch_size: int | None = None,
    target_iterations: int | None = None,
) -> dict[str, object]:
    if payload.get("schema") != COMPLETED_GAME_BUDGET_SCHEMA:
        raise ValueError("unsupported completed-game budget schema")
    declared = str(payload.get("contract_hash", ""))
    actual = canonical_hash({key: value for key, value in payload.items() if key != "contract_hash"})
    if not declared or declared != actual:
        raise ValueError("completed-game budget contract hash mismatch")
    algorithm = str(payload.get("algorithm", ""))
    peers = tuple(str(value) for value in payload.get("peer_algorithms", ()))
    if algorithm not in SUPPORTED_COMPARISON_ALGORITHMS:
        raise ValueError(f"unsupported budget algorithm {algorithm!r}")
    if set(peers) != set(SUPPORTED_COMPARISON_ALGORITHMS) or len(peers) != len(SUPPORTED_COMPARISON_ALGORITHMS):
        raise ValueError("completed-game comparison must name IPPO and VRPO exactly once")
    if expected_algorithm is not None and algorithm != str(expected_algorithm):
        raise ValueError("completed-game budget algorithm/profile mismatch")
    if payload.get("unit") != "completed_games" or payload.get("exact_budget_required") is not True:
        raise ValueError("comparison budget must require exact completed games")
    if payload.get("wall_time_is_safety_stop_only") is not True:
        raise ValueError("wall time must be a safety stop, not the comparison unit")
    comparison_id = str(payload.get("comparison_id", ""))
    selection_hash = str(payload.get("selection_dataset_hash", ""))
    if not comparison_id or len(selection_hash) != 64:
        raise ValueError("comparison id and selection dataset hash are required")
    target_games = int(payload.get("target_completed_games", 0))
    collect = int(payload.get("global_collect_batch_size", 0))
    iterations = int(payload.get("expected_iterations", 0))
    if target_games <= 0 or collect <= 0 or iterations <= 0:
        raise ValueError("completed-game budget values must be positive")
    if target_games != collect * iterations:
        raise ValueError("completed-game target must equal global collect batch times expected iterations")
    if global_collect_batch_size is not None and collect != int(global_collect_batch_size):
        raise ValueError("completed-game contract collect batch differs from training config")
    if target_iterations is not None and iterations != int(target_iterations):
        raise ValueError("completed-game contract iteration target differs from run control")
    return dict(payload)


def observe_completed_game_budget(
    contract: Mapping[str, object],
    *,
    iteration: int,
    completed_games: int,
) -> dict[str, object]:
    normalized = validate_completed_game_budget_contract(contract)
    iteration = int(iteration)
    completed_games = int(completed_games)
    expected_games = iteration * int(normalized["global_collect_batch_size"])
    target_games = int(normalized["target_completed_games"])
    target_iterations = int(normalized["expected_iterations"])
    if iteration < 0 or completed_games < 0:
        raise ValueError("iteration/completed games cannot be negative")
    if iteration > target_iterations:
        raise RuntimeError("completed-game run exceeded its iteration budget")
    if completed_games != expected_games:
        raise RuntimeError(
            "completed-game counter is not exact for the configured global collect batch: "
            f"iteration={iteration}, completed={completed_games}, expected={expected_games}"
        )
    if completed_games > target_games:
        raise RuntimeError("completed-game run overshot the controlled comparison budget")
    return {
        "schema": "guandan-marvel-completed-game-budget-observation-v1",
        "comparison_id": normalized["comparison_id"],
        "contract_hash": normalized["contract_hash"],
        "algorithm": normalized["algorithm"],
        "iteration": iteration,
        "completed_games": completed_games,
        "target_iterations": target_iterations,
        "target_completed_games": target_games,
        "remaining_completed_games": target_games - completed_games,
        "complete": iteration == target_iterations and completed_games == target_games,
    }


__all__ = [
    "COMPLETED_GAME_BUDGET_SCHEMA", "SUPPORTED_COMPARISON_ALGORITHMS",
    "build_completed_game_budget_contract", "observe_completed_game_budget",
    "validate_completed_game_budget_contract",
]
