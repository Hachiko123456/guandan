# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/training/gae.py.
"""Generalized advantage estimation for MARVEL training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from guandan_marvel.model.tensordict import TensorDict


@dataclass(frozen = True)
class GAEFromValueOutput:
    player_advantage: torch.Tensor  # [B, T]
    player_state_value: torch.Tensor  # [B, T]


@dataclass(frozen = True)
class GAEFromQValueOutput:
    player_advantage: torch.Tensor  # [B, T]
    qvalue_lbd_true: torch.Tensor  # [B, T, P]
    qvalue_td1_true: torch.Tensor  # [B, T, P]


def compute_gae_from_value(
    rollout: TensorDict,
    state_values: torch.Tensor,
    *,
    discount_gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> GAEFromValueOutput:
    """Compute GAE advantages/value targets from rollout rewards and predicted values.

    Parameters
    ----------
    rollout:
        Rollout TensorDict containing ``num_steps``, ``player_ids``, and ``rewards``.
    state_values:
        Value estimates shaped like ``rollout['rewards']`` (typically
        ``agent_model(...).state_values``).
    discount_gamma:
        Discount factor for bootstrapping.
    gae_lambda:
        Lambda parameter for generalized advantage estimation.
    """
    num_steps = rollout["num_steps"]
    player_ids = rollout["player_ids"]
    rewards = rollout["rewards"]

    B, T, P = rewards.shape
    device = rewards.device

    if state_values.shape != (B, T, P):
        raise ValueError(
            f"state_values must have shape {(B, T, P)}, received {tuple(state_values.shape)}."
        )
    if player_ids.shape != (B, T):
        raise ValueError(
            f"player_ids must have shape {(B, T)}, received {tuple(player_ids.shape)}."
        )
    if player_ids.device != device:
        player_ids = player_ids.to(device)

    time_range = torch.arange(T, device = device).unsqueeze(0)
    history_mask = time_range < num_steps.unsqueeze(1)
    player_mask = history_mask & (player_ids >= 0)

    history_mask_f = history_mask.to(rewards.dtype).unsqueeze(-1)  # [B, T, 1]
    rewards = rewards * history_mask_f
    state_values = state_values * history_mask_f

    player_advantage = torch.zeros((B, T), device = device, dtype = state_values.dtype)
    player_state_value = torch.zeros((B, T), device = device, dtype = state_values.dtype)

    reward_buffer = torch.zeros(B, P, device = device, dtype = rewards.dtype)
    discounted_next_value = torch.zeros(B, P, device = device, dtype = rewards.dtype)
    gae = torch.zeros(B, P, device = device, dtype = rewards.dtype)

    gamma_tensor = torch.as_tensor(discount_gamma, dtype = rewards.dtype, device = device)
    gae_lambda_tensor = torch.as_tensor(gae_lambda, dtype = rewards.dtype, device = device)
    player_range = torch.arange(P, device = device).view(1, P)

    for t in range(T - 1, -1, -1):
        reward_t = rewards[:, t]
        value_t = state_values[:, t]

        reward_buffer = reward_t + gamma_tensor * reward_buffer
        discounted_next_value = gamma_tensor * discounted_next_value
        gae = gamma_tensor * gae_lambda_tensor * gae

        acting_player_t = player_ids[:, t]
        acting_mask_t = (acting_player_t.unsqueeze(-1) == player_range) & (
            acting_player_t.unsqueeze(-1) >= 0
        )
        acting_mask_t = acting_mask_t & history_mask[:, t].unsqueeze(-1)
        mask_f_t = acting_mask_t.to(rewards.dtype)

        delta = reward_buffer + discounted_next_value - value_t
        updated_gae = delta + gae
        gae = torch.where(acting_mask_t, updated_gae, gae)

        masked_gae = gae * mask_f_t
        player_advantage[:, t] = masked_gae.sum(dim = -1)
        player_state_value[:, t] = ((value_t + masked_gae) * mask_f_t).sum(dim = -1)

        reward_buffer = torch.where(
            acting_mask_t,
            torch.zeros_like(reward_buffer),
            reward_buffer,
        )
        discounted_next_value = torch.where(acting_mask_t, value_t, discounted_next_value)

    player_mask_f = player_mask.to(player_advantage.dtype)
    player_advantage = player_advantage * player_mask_f
    player_state_value = player_state_value * player_mask_f

    return GAEFromValueOutput(
        player_advantage = player_advantage,
        player_state_value = player_state_value,
    )


def compute_gae_from_qvalue(
    rollout: TensorDict,
    state_qvalues: torch.Tensor,
    *,
    avail_probs_new: torch.Tensor | None = None,
    discount_gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> GAEFromQValueOutput:
    """Compute GAE-style lambda advantages from rollout rewards and predicted Q-values.

    This helper is used by the VRPO Q-critic trainer:

    - `V_pi(s)` is computed as the on-policy expectation `E_a[Q(s, a)]`
      using `avail_probs_new`.
    - `Q_lambda(s, a)` is computed via TD(lambda) bootstrapping using
      `V_pi(s_{t+1})`.
    - `A_pi(s, a) = Q_lambda(s, a) - V_pi(s)` is gathered for the acting
      player at each step.

    Parameters
    ----------
    rollout:
        Rollout TensorDict containing ``num_steps``, ``player_ids``, ``rewards``,
        ``action_taken``, and optionally ``avail_probs``.
    state_qvalues:
        Q-value estimates shaped ``[B, T, A, P]`` (typically critic_model(...).qvalues).
    avail_probs_new:
        Action probabilities shaped ``[B, T, A]`` used to form `V_pi(s)` and
        bootstrap targets. If omitted, the rollout-time `avail_probs` are used.
    discount_gamma:
        Discount factor for bootstrapping.
    gae_lambda:
        Lambda parameter for TD(lambda).
    """
    num_steps = rollout["num_steps"]
    player_ids = rollout["player_ids"]
    rewards = rollout["rewards"].to(dtype = state_qvalues.dtype)
    action_taken = rollout["action_taken"].to(dtype = torch.long)
    if avail_probs_new is None:
        avail_probs_new = rollout["avail_probs"]

    B, T = player_ids.shape
    if state_qvalues.shape[:2] != (B, T):
        raise ValueError("state_qvalues leading dims must match rollout [B, T].")
    if avail_probs_new.shape[:2] != (B, T):
        raise ValueError("avail_probs_new leading dims must match rollout [B, T].")
    if action_taken.shape != (B, T):
        raise ValueError("rollout['action_taken'] must have shape [B, T].")

    A = avail_probs_new.shape[-1]
    if state_qvalues.shape[2] != A:
        raise ValueError("state_qvalues action dim must match avail_probs_new.")

    P = rewards.shape[-1]
    if state_qvalues.shape[-1] != P:
        raise ValueError("state_qvalues player dim must match rollout rewards.")

    time_range = torch.arange(T, device = player_ids.device).unsqueeze(0)
    history_mask = time_range < num_steps.unsqueeze(1)
    history_mask_f = history_mask.to(state_qvalues.dtype).unsqueeze(-1)  # [B, T, 1]

    value = (state_qvalues * avail_probs_new.unsqueeze(-1)).sum(dim = 2)  # [B, T, P]
    value = value * history_mask_f

    action_index = action_taken.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, P)
    qvalue_pred = state_qvalues.gather(dim = 2, index = action_index).squeeze(2)  # [B, T, P]
    qvalue_pred = qvalue_pred * history_mask_f

    rewards = rewards * history_mask_f
    next_qvalue = torch.zeros(B, P, device = state_qvalues.device)
    next_qerror = torch.zeros(B, P, device = state_qvalues.device)
    qvalue_lbd_true = torch.zeros(B, T, P, device = state_qvalues.device)
    qvalue_td1_true = torch.zeros(B, T, P, device = state_qvalues.device)

    for t in range(T - 1, -1, -1):
        next_qvalue = next_qvalue * discount_gamma + rewards[:, t]
        next_qerror = next_qerror * discount_gamma

        qvalue_td1_true[:, t] = next_qvalue

        delta = next_qvalue - qvalue_pred[:, t]
        next_qerror = delta + next_qerror * gae_lambda
        qvalue_lbd_true[:, t] = qvalue_pred[:, t] + next_qerror

        next_qvalue = value[:, t]

    advantages = qvalue_lbd_true - value
    player_mask = history_mask & (player_ids >= 0)
    acting_players = player_ids.clamp(min = 0)
    acting_idx = acting_players.unsqueeze(-1)
    player_advantage = advantages.gather(dim = 2, index = acting_idx).squeeze(-1)
    player_advantage = player_advantage * player_mask.to(player_advantage.dtype)

    return GAEFromQValueOutput(
        player_advantage = player_advantage,
        qvalue_lbd_true = qvalue_lbd_true,
        qvalue_td1_true = qvalue_td1_true,
    )


__all__ = [
    "GAEFromValueOutput",
    "GAEFromQValueOutput",
    "compute_gae_from_value",
    "compute_gae_from_qvalue",
]
