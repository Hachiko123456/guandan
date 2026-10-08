# SPDX-License-Identifier: MIT
# Reworked from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/training/trainer.py.
from __future__ import annotations

from dataclasses import dataclass

from guandan_marvel.config import TrainConfig


def compute_stable_decay_scale(*, stable_iterations: int, iteration: int, decay_power: float) -> float:
    if iteration <= 0:
        raise ValueError("iteration must be positive.")
    if stable_iterations < 0:
        raise ValueError("stable_iterations must be non-negative.")
    if decay_power < 0:
        raise ValueError("decay_power must be non-negative.")
    return min(stable_iterations / iteration, 1.0) ** decay_power


@dataclass(frozen=True)
class ScheduleState:
    lr_scale: float
    critic_lr_scale: float
    clip_coef_scale: float
    agent_lr: float
    clip_coef: float
    reg_coef: float
    reg_scale: float


def compute_agent_schedule(cfg: TrainConfig, *, iteration: int) -> ScheduleState:
    lr_scale = compute_stable_decay_scale(stable_iterations=cfg.lr_stable_iterations, iteration=iteration, decay_power=cfg.schedule.agent_lr_decay_power)
    critic_lr_scale = compute_stable_decay_scale(stable_iterations=cfg.lr_stable_iterations, iteration=iteration, decay_power=cfg.schedule.critic_lr_decay_power)
    clip_scale = compute_stable_decay_scale(stable_iterations=cfg.lr_stable_iterations, iteration=iteration, decay_power=cfg.schedule.clip_coef_decay_power)
    reg_scale = compute_stable_decay_scale(stable_iterations=cfg.reg_stable_iterations, iteration=iteration, decay_power=cfg.schedule.reg_coef_decay_power)
    return ScheduleState(
        lr_scale=lr_scale,
        critic_lr_scale=critic_lr_scale,
        clip_coef_scale=clip_scale,
        agent_lr=cfg.ppo.base_lr * lr_scale,
        clip_coef=cfg.ppo.base_clip_coef * clip_scale,
        reg_coef=cfg.ppo.base_reg_coef * reg_scale,
        reg_scale=reg_scale,
    )


__all__ = ["ScheduleState", "compute_agent_schedule", "compute_stable_decay_scale"]
