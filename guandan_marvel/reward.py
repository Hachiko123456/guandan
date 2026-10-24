# SPDX-License-Identifier: MIT
"""MARVEL v14 phase-4 reward semantics.

The environment exposes the raw GuanDan *rank-point* reward: the winning team
receives ``+3`` for a double down, ``+2`` for a head-third finish and ``+1`` for
a head-last finish; the losing team receives the negation, and every seat
receives the reward of the team it belongs to.  Training consumes the normalised
reward ``R_train = R_raw / 3`` so that the win criterion is unchanged
(``sign(R_raw)``) while the value scale is damped by a factor of three relative
to the previous ``+/-1`` objective.

The C++ runtime (``nanopokerlib``) and the Python oracle (``guandan``) both emit
the raw rank points.  This module owns the *canonical* semantics that the MARVEL
training layer (config, checkpoints, metrics, delivery manifests) records and
enforces; it is deliberately versioned so an incompatible change is a semantic
break rather than a silent drift.
"""
from __future__ import annotations

# Versioned reward-semantics identifier recorded in configs/checkpoints/deliveries.
REWARD_SEMANTICS = "guandan_rank_points"

# Team outcome -> raw rank points paid to the winning team.
RANK_POINTS: dict[str, int] = {"DOUBLE_DOWN": 3, "HEAD_THIRD": 2, "HEAD_LAST": 1}
OUTCOME_CLASSES: tuple[str, ...] = ("DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST")

# Closed range of a single seat's raw rank-point reward.
RAW_REWARD_MIN = -float(max(RANK_POINTS.values()))
RAW_REWARD_MAX = float(max(RANK_POINTS.values()))

# Training normalisation factor: R_train = R_raw / 3.
TRAINING_REWARD_SCALE = 1.0 / 3.0


def rank_point_reward(outcome_class: str, winner_team: int) -> tuple[float, float]:
    """Return ``(team0_raw, team1_raw)`` for one terminal outcome.

    This mirrors ``guandan.state.rank_point_reward`` and the C++
    ``outcome_rank_points`` helper; it is the single Python reference used by the
    MARVEL layer (the independent cross-engine acceptance test hand-writes the
    expected table instead of importing this helper).
    """
    points = RANK_POINTS[str(outcome_class)]
    team = int(winner_team)
    if team not in (0, 1):
        raise ValueError("winner_team must be 0 or 1")
    return (
        float(points) if team == 0 else -float(points),
        float(points) if team == 1 else -float(points),
    )


def scale_reward(raw_reward: float) -> float:
    """Apply the fixed training normalisation ``R_train = R_raw / 3``."""
    return float(raw_reward) * TRAINING_REWARD_SCALE


def reward_semantics_block() -> dict[str, object]:
    """Canonical, hashable description of the active reward semantics."""
    return {
        "schema": "guandan-marvel-reward-semantics-v1",
        "semantics": REWARD_SEMANTICS,
        "rank_points": {name: int(RANK_POINTS[name]) for name in OUTCOME_CLASSES},
        "raw_range": [RAW_REWARD_MIN, RAW_REWARD_MAX],
        "training_scale": TRAINING_REWARD_SCALE,
        "train_relation": "R_train = R_raw * training_scale",
        "win_criterion": "sign(R_raw) == sign(R_train)",
    }


__all__ = [
    "OUTCOME_CLASSES", "RANK_POINTS", "RAW_REWARD_MAX", "RAW_REWARD_MIN",
    "REWARD_SEMANTICS", "TRAINING_REWARD_SCALE", "rank_point_reward",
    "reward_semantics_block", "scale_reward",
]
