"""Fixed-seat GAE and Q-boost targets for token-level on-policy rollouts.

All trajectory values are [time, batch, seat], never the value of whichever
player happens to act on that row. Targets are deliberately detached. A
truncation bootstraps from the *pre-reset* final state but cuts the trace.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor


NUM_PLAYERS = 4


def _unit_interval(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _floating(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def _same_layout(name: str, value: Tensor, reference: Tensor) -> None:
    _floating(name, value)
    if value.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name} must share the reference device and dtype")


def _trajectory_inputs(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[float, float]:
    gamma = _unit_interval("gamma", gamma)
    gae_lambda = _unit_interval("gae_lambda", gae_lambda)
    _floating("rewards", rewards)
    if rewards.ndim != 3 or rewards.shape[-1] != NUM_PLAYERS:
        raise ValueError("rewards must have shape [T, B, 4]")
    if rewards.shape[1] == 0:
        raise ValueError("the batch dimension must be nonempty")
    for name, tensor in (("values", values), ("next_values", next_values)):
        _same_layout(name, tensor, rewards)
    for name, flag in (("terminated", terminated), ("truncated", truncated)):
        if not isinstance(flag, Tensor) or flag.dtype != torch.bool:
            raise TypeError(f"{name} must be a boolean tensor")
        if flag.shape != rewards.shape[:2] or flag.device != rewards.device:
            raise ValueError(f"{name} must have shape [T, B] on the rewards device")
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():
        raise ValueError("rewards and values must be finite")
    # A terminal successor does not exist. Mask it before arithmetic, so even
    # a NaN placeholder there cannot poison an otherwise valid target.
    if not torch.isfinite(next_values[~terminated]).all():
        raise ValueError("nonterminal next_values must be finite (including truncations)")
    return gamma, gae_lambda


def _backward_trace(residuals: Tensor, cuts: Tensor, discount: float) -> Tensor:
    trace = torch.zeros_like(residuals)
    carry = residuals.new_zeros(residuals.shape[1:])
    for t in range(residuals.shape[0] - 1, -1, -1):
        carry = residuals[t] + discount * torch.where(
            cuts[t].unsqueeze(-1), torch.zeros_like(carry), carry
        )
        trace[t] = carry
    return trace


@torch.no_grad()
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Return detached ``(advantages, returns)`` with shape [T, B, 4].

    delta = r + gamma * (not terminated) * next_V - V
    A[t] = delta[t] + gamma * lambda * (not (terminated | truncated)) * A[t+1]
    returns = V + A

    ``next_values[t]`` describes the actual successor of transition t, not an
    auto-reset observation. Ordinary rollout boundaries still bootstrap but
    have zero trace beyond the collected horizon. Empty time axes are allowed.
    """
    gamma, gae_lambda = _trajectory_inputs(
        rewards, values, next_values, terminated, truncated, gamma, gae_lambda
    )
    bootstrap = next_values.masked_fill(terminated.unsqueeze(-1), 0.0)
    residuals = rewards + gamma * bootstrap - values
    advantages = _backward_trace(residuals, terminated | truncated, gamma * gae_lambda)
    return advantages, values + advantages


@torch.no_grad()
def compute_q_boost(
    rewards: Tensor,
    selected_q: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Return detached ``(advantages, q_targets)`` with shape [T, B, 4].

    Implements arXiv:2605.19235v1, section 3.2, equations (3.2)-(3.3):
    delta_plus = r + gamma * (not terminated) * next_V - selected_Q
    trace[t] = delta_plus[t] + gamma * lambda * (not (terminated | truncated)) * trace[t+1]
    advantages = selected_Q - V + trace; q_targets = selected_Q + trace.

    V and next_V must be policy expectations of the centralized Q critic
    under the player acting at each respective state, in fixed seat order.
    """
    gamma, gae_lambda = _trajectory_inputs(
        rewards, values, next_values, terminated, truncated, gamma, gae_lambda
    )
    _same_layout("selected_q", selected_q, rewards)
    if not torch.isfinite(selected_q).all():
        raise ValueError("selected_q must be finite")
    bootstrap = next_values.masked_fill(terminated.unsqueeze(-1), 0.0)
    residuals = rewards + gamma * bootstrap - selected_q
    trace = _backward_trace(residuals, terminated | truncated, gamma * gae_lambda)
    return selected_q - values + trace, selected_q + trace


def masked_expected_values(
    q_values: Tensor, probabilities: Tensor, legal_mask: Tensor
) -> Tensor:
    """Return V[B,4] = sum_a pi(a | o_active) Q(s,a) over legal actions.

    Invalid slots are zeroed *before* multiplication, including nonfinite
    padding. Legal probabilities are renormalized after masking; every row
    must have positive finite legal mass. Already masked softmax probabilities
    are unchanged except for roundoff. Gradients flow only through legal slots.
    """
    _floating("q_values", q_values)
    _floating("probabilities", probabilities)
    if q_values.ndim != 3 or q_values.shape[-1] != NUM_PLAYERS:
        raise ValueError("q_values must have shape [B, A, 4]")
    if q_values.shape[0] == 0 or q_values.shape[1] == 0:
        raise ValueError("q_values must have nonempty batch and action dimensions")
    if probabilities.shape != q_values.shape[:2]:
        raise ValueError("probabilities must have shape [B, A]")
    if probabilities.device != q_values.device or probabilities.dtype != q_values.dtype:
        raise ValueError("probabilities and q_values must share device and dtype")
    if not isinstance(legal_mask, Tensor) or legal_mask.dtype != torch.bool:
        raise TypeError("legal_mask must be a boolean tensor")
    if legal_mask.shape != probabilities.shape or legal_mask.device != q_values.device:
        raise ValueError("legal_mask must have shape [B, A] on the q_values device")
    legal_probabilities = probabilities.masked_fill(~legal_mask, 0.0)
    legal_q = q_values.masked_fill(~legal_mask.unsqueeze(-1), 0.0)
    if not torch.isfinite(legal_probabilities).all() or (legal_probabilities < 0).any():
        raise ValueError("legal probabilities must be finite and nonnegative")
    if not torch.isfinite(legal_q).all():
        raise ValueError("legal q_values must be finite")
    mass = legal_probabilities.sum(dim=-1, keepdim=True)
    if not torch.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError("each row must have positive finite legal probability mass")
    policy = legal_probabilities / mass
    return (legal_q * policy.unsqueeze(-1)).sum(dim=1)


def _player_ids(player_ids: Tensor, leading_shape: torch.Size, device: torch.device) -> None:
    if not isinstance(player_ids, Tensor) or player_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("player_ids must be an int32 or int64 tensor")
    if player_ids.shape != leading_shape or player_ids.device != device:
        raise ValueError("player_ids must match the leading shape and device")
    if ((player_ids < 0) | (player_ids >= NUM_PLAYERS)).any():
        raise ValueError("player_ids must be in [0, 3]")


def acting_player_values(values: Tensor, player_ids: Tensor) -> Tensor:
    """Gather [*leading,4] into [*leading] using the actor ID of each row."""
    _floating("values", values)
    if values.ndim < 1 or values.shape[-1] != NUM_PLAYERS:
        raise ValueError("values must have a trailing dimension of four players")
    _player_ids(player_ids, values.shape[:-1], values.device)
    return values.gather(-1, player_ids.long().unsqueeze(-1)).squeeze(-1)


def expand_team_values(actor_values: Tensor, player_ids: Tensor) -> Tensor:
    """Expand shared IPPO actor-seat scalars to fixed-seat [...,4] values.

    Seats 0/2 form team 0 and 1/3 team 1. The current actor's team gets +v;
    the other team gets -v. This is an approximate decentralized baseline,
    not four independently observed player values and not a centralized actor.
    """
    _floating("actor_values", actor_values)
    _player_ids(player_ids, actor_values.shape, actor_values.device)
    if not torch.isfinite(actor_values).all():
        raise ValueError("actor_values must be finite")
    seat_teams = torch.arange(NUM_PLAYERS, device=actor_values.device).remainder(2)
    same_team = player_ids.remainder(2).unsqueeze(-1).eq(seat_teams)
    return torch.where(same_team, actor_values.unsqueeze(-1), -actor_values.unsqueeze(-1))


__all__ = [
    "compute_gae", "compute_q_boost", "masked_expected_values",
    "acting_player_values", "expand_team_values",
]
