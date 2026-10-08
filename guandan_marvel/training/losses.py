# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/training/losses.py.
"""Loss computation functions for MARVEL training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from guandan_marvel.model.tensordict import TensorDict


@dataclass(frozen = True)
class ClippedPGLossOutput:
    pg_loss_mean: torch.Tensor
    clip_frac: torch.Tensor
    kl_unif_mean: torch.Tensor
    kl_ref_mean: torch.Tensor
    adv_mean: torch.Tensor
    adv_std: torch.Tensor


@dataclass(frozen = True)
class ValueLossOutput:
    value_loss_mean: torch.Tensor


@dataclass(frozen = True)
class QValueLossOutput:
    qv_lbd_loss_mean: torch.Tensor
    qv_td1_loss_mean: torch.Tensor


def compute_clipped_pg_loss(
    rollout: TensorDict,
    *,
    clip_coef: float,
    avail_probs_new: torch.Tensor,
    player_advantage: torch.Tensor,
) -> ClippedPGLossOutput:
    """Compute the PPO clipped policy gradient objective terms.

    Derives `action_prob_new`/`action_prob_ref` from the rollout's taken actions, reference
    probabilities from `rollout['avail_probs']`, and the regularization KL term from
    `rollout['avail_weights']`.

    Loss terms
    ----------
    pg_loss_mean:
        PPO clipped surrogate objective, averaged over valid acting-player timesteps:
        ``mean(max(-A_t * r_t, -A_t * clip(r_t, 1-eps, 1+eps)))``.
    clip_frac:
        Fraction of valid samples where ``|r_t - 1| > clip_coef``.
    kl_unif_mean:
        Batch-local-normalized mean ``KL(pi_new || pi_unif_on_legal_actions)``.
    kl_ref_mean:
        Batch-local-normalized mean ``KL(pi_new || pi_ref)``.
    adv_mean / adv_std:
        Mean/std of masked advantages before the internal z-score normalization used by PPO.
    """
    player_ids = rollout["player_ids"]
    num_steps = rollout["num_steps"]
    action_taken = rollout["action_taken"].to(dtype = torch.long)
    avail_weights = rollout["avail_weights"]
    avail_probs_ref = rollout["avail_probs"].to(dtype = avail_probs_new.dtype)

    B, T = player_ids.shape
    if avail_probs_new.shape[:2] != (B, T):
        raise ValueError("avail_probs_new leading dims must match rollout [B, T].")
    if avail_probs_ref.shape != avail_probs_new.shape:
        raise ValueError("avail_probs_ref must have the same shape as avail_probs_new.")
    if player_advantage.shape != (B, T):
        raise ValueError("player_advantage must have shape [B, T].")
    if action_taken.shape != (B, T):
        raise ValueError("rollout['action_taken'] must have shape [B, T].")

    A = avail_probs_new.shape[-1]
    if avail_weights.shape != (B, T, A):
        raise ValueError("rollout['avail_weights'] must have shape [B, T, A].")

    time_range = torch.arange(T, device = player_ids.device).unsqueeze(0)
    history_mask = time_range < num_steps.unsqueeze(1)
    player_mask = history_mask & (player_ids >= 0)

    batch_idx_range = torch.arange(B, device = player_ids.device).unsqueeze(1)
    time_idx_range = torch.arange(T, device = player_ids.device).unsqueeze(0)
    action_prob_new = avail_probs_new[batch_idx_range, time_idx_range, action_taken]
    action_prob_ref = avail_probs_ref[batch_idx_range, time_idx_range, action_taken]

    avail_probs_unif = avail_weights / avail_weights.sum(dim = -1, keepdim = True).clamp(min = 1e-8)

    flat_player_mask = player_mask.reshape(-1)
    mask_avail_probs_new = avail_probs_new.reshape(-1, A)[flat_player_mask]
    mask_avail_probs_ref = avail_probs_ref.reshape(-1, A)[flat_player_mask]
    mask_avail_probs_unif = avail_probs_unif.reshape(-1, A)[flat_player_mask]
    mask_action_prob_ref = action_prob_ref.reshape(-1)[flat_player_mask]
    mask_action_prob_new = action_prob_new.reshape(-1)[flat_player_mask]

    mask_logprob_ref = mask_action_prob_ref.log()
    mask_logprob_new = mask_action_prob_new.log()
    mask_logratio = mask_logprob_new - mask_logprob_ref
    mask_ratio = mask_logratio.exp()

    eps = 1e-9
    mask_avail_probs_new_safe = mask_avail_probs_new + eps
    mask_kl_unif = (
        mask_avail_probs_new_safe
        * (mask_avail_probs_new_safe.log() - (mask_avail_probs_unif + eps).log())
    ).sum(dim = -1)

    mask_kl_ref = (
        mask_avail_probs_new_safe
        * (mask_avail_probs_new_safe.log() - (mask_avail_probs_ref + eps).log())
    ).sum(dim = -1)

    kl_unif_mean = mask_kl_unif.mean()
    kl_ref_mean = mask_kl_ref.mean()

    mask_clipped = (mask_ratio - 1.0).abs() > clip_coef
    clip_frac = mask_clipped.float().mean()

    mask_player_advantage = player_advantage.reshape(-1)[flat_player_mask]
    adv_mean = mask_player_advantage.mean()
    adv_std = mask_player_advantage.std()
    mask_player_advantage = (mask_player_advantage - adv_mean) / (adv_std + 1e-8)

    pg_loss1 = -mask_player_advantage * mask_ratio
    pg_loss2 = -mask_player_advantage * torch.clamp(
        mask_ratio,
        1.0 - clip_coef,
        1.0 + clip_coef,
    )
    pg_loss_mean = torch.max(pg_loss1, pg_loss2).mean()

    return ClippedPGLossOutput(
        pg_loss_mean = pg_loss_mean,
        clip_frac = clip_frac,
        kl_unif_mean = kl_unif_mean,
        kl_ref_mean = kl_ref_mean,
        adv_mean = adv_mean,
        adv_std = adv_std,
    )


def compute_value_loss(
    rollout: TensorDict,
    *,
    player_state_value_pred: torch.Tensor,
    player_state_value_target: torch.Tensor,
) -> ValueLossOutput:
    """Compute value-function regression loss for acting-player states.

    Loss terms
    ----------
    value_loss_mean:
        Mean Huber loss of ``V_pred`` vs ``V_target`` over valid acting-player timesteps.
    """
    player_ids = rollout["player_ids"]
    num_steps = rollout["num_steps"]

    B, T = player_ids.shape
    if player_state_value_pred.shape != (B, T):
        raise ValueError("player_state_value_pred must have shape [B, T].")
    if player_state_value_target.shape != (B, T):
        raise ValueError("player_state_value_target must have shape [B, T].")

    time_range = torch.arange(T, device = player_ids.device).unsqueeze(0)
    history_mask = time_range < num_steps.unsqueeze(1)
    player_mask = history_mask & (player_ids >= 0)

    flat_player_mask = player_mask.reshape(-1)
    mask_pred = player_state_value_pred.reshape(-1)[flat_player_mask]
    mask_target = player_state_value_target.reshape(-1)[flat_player_mask]
    value_loss_mean = F.huber_loss(mask_pred, mask_target, reduction = "mean", delta = 10.0)

    return ValueLossOutput(value_loss_mean = value_loss_mean)


def compute_qvalue_loss(
    rollout: TensorDict,
    state_qvalues: torch.Tensor,
    *,
    qvalue_lbd_true: torch.Tensor,
    qvalue_td1_true: torch.Tensor,
) -> QValueLossOutput:
    """Compute critic regression losses for taken-action Q-values.

    Loss terms
    ----------
    qv_lbd_loss_mean:
        Mean Huber loss of Q_pred vs Q_lambda_target over valid critic-training timesteps.
    qv_td1_loss_mean:
        Mean Huber loss of Q_pred vs Q_td1_target over the same mask.
    """
    num_steps = rollout["num_steps"]
    player_ids = rollout["player_ids"]
    action_taken = rollout["action_taken"].to(dtype = torch.long)

    B, T = player_ids.shape
    if state_qvalues.shape[:2] != (B, T):
        raise ValueError("state_qvalues leading dims must match rollout [B, T].")
    if action_taken.shape != (B, T):
        raise ValueError("rollout['action_taken'] must have shape [B, T].")

    P = state_qvalues.shape[3]
    if qvalue_lbd_true.shape != (B, T, P):
        raise ValueError("qvalue_lbd_true must have shape [B, T, P].")
    if qvalue_td1_true.shape != (B, T, P):
        raise ValueError("qvalue_td1_true must have shape [B, T, P].")

    action_index = action_taken.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, P)
    qvalue_pred = state_qvalues.gather(dim = 2, index = action_index).squeeze(2)  # [B, T, P]

    time_range = torch.arange(T, device = player_ids.device).unsqueeze(0)
    history_mask = time_range < num_steps.unsqueeze(1)
    # Include trainer-controlled chance (-1) and acting players (>=0), but exclude
    # environment-controlled prefix chance (-2) from critic loss.
    train_mask = history_mask & (player_ids >= -1)

    flat_train_mask = train_mask.reshape(-1)
    mask_qvalue_pred = qvalue_pred.reshape(-1, P)[flat_train_mask]
    mask_qvalue_lbd_true = qvalue_lbd_true.reshape(-1, P)[flat_train_mask]
    mask_qvalue_td1_true = qvalue_td1_true.reshape(-1, P)[flat_train_mask]
    qv_lbd_loss_mean = F.huber_loss(mask_qvalue_pred, mask_qvalue_lbd_true, reduction = "mean", delta = 10.0)
    qv_td1_loss_mean = F.huber_loss(mask_qvalue_pred, mask_qvalue_td1_true, reduction = "mean", delta = 10.0)

    return QValueLossOutput(
        qv_lbd_loss_mean = qv_lbd_loss_mean,
        qv_td1_loss_mean = qv_td1_loss_mean,
    )

__all__ = [
    "ClippedPGLossOutput",
    "ValueLossOutput",
    "QValueLossOutput",
    "compute_clipped_pg_loss",
    "compute_value_loss",
    "compute_qvalue_loss",
]
