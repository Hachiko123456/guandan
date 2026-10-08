# SPDX-License-Identifier: MIT
"""Fail-closed validation for complete MARVEL training trajectories."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from guandan_marvel.model.tensordict import TensorDict

TRAJECTORY_TRAINING_SCHEMA = "guandan-marvel-complete-trajectory-v1"
REQUIRED_TRAJECTORY_FIELDS = frozenset({
    "num_steps", "player_ids", "obs_tokens", "obs_channels",
    "avail_tokens", "avail_weights", "avail_is_commit", "policy_probs",
    "behavior_action_prob", "on_policy_valid", "action_taken", "rewards",
    "is_terminal",
})
CRITIC_ONLY_FIELDS = frozenset({"ownership", "escrow", "played", "critic_state"})


@dataclass(frozen=True, slots=True)
class TrajectoryValidation:
    batch_size: int
    total_steps: int
    min_steps: int
    max_steps: int
    completed_games: int


def _require_finite(name: str, value: torch.Tensor) -> None:
    if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"trajectory field {name!r} contains non-finite values")


def validate_complete_trajectories(
    rollout: TensorDict,
    *,
    require_on_policy: bool,
) -> TrajectoryValidation:
    missing = sorted(REQUIRED_TRAJECTORY_FIELDS - set(rollout.keys()))
    if missing:
        raise KeyError(f"trajectory is missing required fields: {missing}")
    leaked = sorted(CRITIC_ONLY_FIELDS & set(rollout.keys()))
    if leaked:
        raise ValueError(f"actor trajectory contains critic-only fields: {leaked}")
    batch = int(rollout.batch_size[0])
    if batch <= 0:
        raise ValueError("trajectory batch must be non-empty")
    num_steps = rollout["num_steps"].to(dtype=torch.long)
    terminal = rollout["is_terminal"].to(dtype=torch.bool)
    if num_steps.shape != (batch,) or terminal.dim() != 2 or terminal.shape[0] != batch:
        raise ValueError("trajectory num_steps/is_terminal shapes are invalid")
    horizon = terminal.shape[1]
    if bool(((num_steps <= 0) | (num_steps > horizon)).any()):
        raise ValueError("trajectory num_steps is outside 1..horizon")
    time = torch.arange(horizon, device=num_steps.device).unsqueeze(0)
    active = time < num_steps.unsqueeze(1)
    final_index = (num_steps - 1).unsqueeze(1)
    final_terminal = terminal.gather(1, final_index).squeeze(1)
    if not bool(final_terminal.all()):
        raise ValueError("every training trajectory must end at a real terminal")
    if bool((terminal & active).sum(dim=1).ne(1).any()):
        raise ValueError("each complete trajectory must contain exactly one terminal step")
    if bool((terminal & ~active).any()):
        raise ValueError("terminal markers outside num_steps are forbidden")
    if require_on_policy:
        player_mask = active & (rollout["player_ids"] >= 0)
        if not bool(rollout["on_policy_valid"].to(torch.bool)[player_mask].all()):
            raise ValueError("training trajectory contains non-on-policy actor steps")
    action_taken = rollout["action_taken"].to(dtype=torch.long)
    avail_weights = rollout["avail_weights"]
    if action_taken.shape != terminal.shape:
        raise ValueError("action_taken must have shape [B,T]")
    selected_weight = avail_weights.gather(2, action_taken.unsqueeze(-1)).squeeze(-1)
    if bool((selected_weight[active] <= 0).any()):
        raise ValueError("trajectory selected an unavailable action")
    behavior = rollout["behavior_action_prob"]
    if bool((behavior[active] <= 0).any()) or not bool(torch.isfinite(behavior[active]).all()):
        raise ValueError("behavior_action_prob must be finite and positive on active steps")
    for name in ("avail_weights", "policy_probs", "behavior_action_prob", "rewards"):
        _require_finite(name, rollout[name])
    return TrajectoryValidation(
        batch_size=batch,
        total_steps=int(num_steps.sum().item()),
        min_steps=int(num_steps.min().item()),
        max_steps=int(num_steps.max().item()),
        completed_games=batch,
    )


def ensure_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def ensure_finite_gradients(module: torch.nn.Module, *, name: str) -> None:
    for parameter_name, parameter in module.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"{name}.{parameter_name} gradient contains non-finite values")


__all__ = [
    "CRITIC_ONLY_FIELDS", "REQUIRED_TRAJECTORY_FIELDS", "TRAJECTORY_TRAINING_SCHEMA",
    "TrajectoryValidation", "ensure_finite_gradients", "ensure_finite_tensor",
    "validate_complete_trajectories",
]
