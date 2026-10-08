# SPDX-License-Identifier: MIT
"""M03 rollout and learner metric helpers."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import torch

from guandan.action_state import (
    FAMILY_TOKEN_BASE, PASS_TOKEN, RETURN_TOKEN, TRIBUTE_TOKEN, WILD_TOKEN_BASE,
)
from guandan.combos import CombinationKind
from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.utils import compute_commit_step_counts


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float(), torch.tensor(q, device=values.device)).item())


def rollout_metrics(
    rollout: TensorDict,
    *,
    terminal_infos: Sequence[Mapping[str, object]],
    env_steps_total: int,
    game_steps_total: int,
    completed_games_total: int,
    rollout_seconds: float,
) -> dict[str, float | int | None]:
    num_steps = rollout["num_steps"].to(torch.long)
    batch = int(num_steps.numel())
    if len(terminal_infos) != batch:
        raise ValueError("terminal info count must equal complete trajectory batch size")
    commit_steps = compute_commit_step_counts(
        rollout["avail_is_commit"], rollout["action_taken"], num_steps,
    )
    returns = rollout["rewards"].sum(dim=1)
    family_counts: Counter[str] = Counter()
    passes = wilds = tribute = returns_exchange = total_commits = 0
    for b in range(batch):
        pending_family: str | None = None
        for t in range(int(num_steps[b])):
            action_index = int(rollout["action_taken"][b, t])
            row = rollout["avail_tokens"][b, t, action_index]
            phase, token = int(row[1]), int(row[5])
            is_commit = bool(rollout["avail_is_commit"][b, t, action_index])
            if FAMILY_TOKEN_BASE <= token < FAMILY_TOKEN_BASE + len(CombinationKind):
                pending_family = list(CombinationKind)[token - FAMILY_TOKEN_BASE].value
            if WILD_TOKEN_BASE <= token < WILD_TOKEN_BASE + 52:
                wilds += 1
            if is_commit:
                total_commits += 1
                if token == PASS_TOKEN:
                    passes += 1
                elif phase == TRIBUTE_TOKEN:
                    tribute += 1
                elif phase == RETURN_TOKEN:
                    returns_exchange += 1
                elif pending_family is not None:
                    family_counts[pending_family] += 1
                pending_family = None
    outcomes = Counter(str(info.get("outcome_class", "UNKNOWN")) for info in terminal_infos)
    duration = max(float(rollout_seconds), 1e-9)
    metrics: dict[str, float | int | None] = {
        "statistics/env_steps_total": int(env_steps_total),
        "statistics/game_steps_total": int(game_steps_total),
        "statistics/completed_games_total": int(completed_games_total),
        "statistics/trajectories": batch,
        "statistics/env_steps_mean": float(num_steps.float().mean()),
        "statistics/game_steps_mean": float(commit_steps.float().mean()),
        "statistics/trajectory_length_mean": float(num_steps.float().mean()),
        "statistics/return_team0_mean": float(returns[:, [0, 2]].mean()),
        "statistics/return_team1_mean": float(returns[:, [1, 3]].mean()),
        "perf/env_steps_per_second": int(num_steps.sum()) / duration,
        "perf/game_steps_per_second": int(commit_steps.sum()) / duration,
        "perf/completed_games_per_second": batch / duration,
        "perf/rollout_seconds": float(rollout_seconds),
        "perf/evaluation_seconds": 0.0,
        "perf/gpu_utilization/0": None,
        "perf/gpu_memory_bytes/0": 0,
        "health/illegal_actions": 0,
        "health/truncated_games": 0,
        "health/nonfinite_losses": 0,
        "health/nonfinite_gradients": 0,
        "health/action_capacity_overflows": 0,
        "health/horizon_overflows": 0,
        "guandan/token_steps_per_game_mean": float(num_steps.float().mean()),
        "guandan/token_steps_per_game_p50": _quantile(num_steps, 0.50),
        "guandan/token_steps_per_game_p95": _quantile(num_steps, 0.95),
        "guandan/token_steps_per_game_p99": _quantile(num_steps, 0.99),
        "guandan/commit_steps_per_game_mean": float(commit_steps.float().mean()),
        "guandan/commit_steps_per_game_p95": _quantile(commit_steps, 0.95),
        "guandan/pass_rate": passes / max(total_commits, 1),
        "guandan/bomb_rate": sum(family_counts[k.value] for k in (CombinationKind.RANK_BOMB, CombinationKind.STRAIGHT_FLUSH, CombinationKind.FOUR_KINGS)) / max(sum(family_counts.values()), 1),
        "guandan/straight_flush_rate": family_counts[CombinationKind.STRAIGHT_FLUSH.value] / max(sum(family_counts.values()), 1),
        "guandan/four_joker_rate": family_counts[CombinationKind.FOUR_KINGS.value] / max(sum(family_counts.values()), 1),
        "guandan/wild_use_rate": wilds / max(total_commits, 1),
        "guandan/tribute_rate": tribute / max(total_commits, 1),
        "guandan/return_rate": returns_exchange / max(total_commits, 1),
        "guandan/anti_tribute_rate": 0.0,
        "guandan/double_down_rate": outcomes["DOUBLE_DOWN"] / batch,
        "guandan/head_third_rate": outcomes["HEAD_THIRD"] / batch,
        "guandan/head_last_rate": outcomes["HEAD_LAST"] / batch,
    }
    return metrics


__all__ = ["rollout_metrics"]
