"""PPO actor objectives with distinct IPPO V and VRPO selected-Q regression."""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F

from ..model import PolicyValueNet


def _coefficient(name: str, value: float, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its finite nonnegative range")
    return value


def _ppo_loss(
    logits: Tensor,
    predictions: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    targets: Tensor,
    legal_tokens: Tensor,
    *,
    clip: float,
    entropy_coef: float,
    value_coef: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    clip = _coefficient("clip", clip, maximum=1.0)
    entropy_coef = _coefficient("entropy_coef", entropy_coef)
    value_coef = _coefficient("value_coef", value_coef)
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have nonempty shape [N, A]")
    # Invalid candidates are -inf, not a large finite negative sentinel.
    PolicyValueNet._require_legal(logits)
    for name, tensor in (
        ("predictions", predictions), ("old_log_probs", old_log_probs),
        ("advantages", advantages), ("targets", targets),
    ):
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if tensor.shape != logits.shape[:1] or tensor.device != logits.device:
            raise ValueError(f"{name} must have shape [N] on the logits device")
        if tensor.dtype != logits.dtype:
            raise ValueError(f"{name} must have the same dtype as logits")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")
    for name, tensor, shape in (
        ("actions", actions, logits.shape[:1]),
        ("legal_tokens", legal_tokens, logits.shape),
    ):
        if not isinstance(tensor, Tensor) or tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be an int32 or int64 tensor")
        if tensor.shape != shape or tensor.device != logits.device:
            raise ValueError(f"{name} has the wrong shape or device")
    # Actions are actual token IDs, not indices. Ignore duplicate padding IDs
    # at invalid positions, but reject ambiguous or absent legal action IDs.
    matches = legal_tokens.eq(actions.unsqueeze(-1)) & torch.isfinite(logits)
    if (matches.sum(dim=-1) != 1).any():
        raise ValueError("each action must identify exactly one legal token")
    indices = matches.long().argmax(dim=-1)
    new_log_probs = F.log_softmax(logits, dim=-1).gather(1, indices.unsqueeze(-1)).squeeze(-1)
    log_ratio = new_log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    frozen_advantages = advantages.detach()
    policy_loss = -torch.minimum(
        ratio * frozen_advantages,
        ratio.clamp(1.0 - clip, 1.0 + clip) * frozen_advantages,
    ).mean()
    value_loss = F.mse_loss(predictions, targets.detach())
    entropy = PolicyValueNet.entropy(logits).mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    metrics = {
        "loss": total.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "approx_kl": ((ratio - 1.0) - log_ratio).mean().detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip).to(logits.dtype).mean().detach(),
    }
    if not all(torch.isfinite(value).all() for value in metrics.values()):
        raise FloatingPointError("non-finite PPO loss or diagnostics")
    return total, metrics


def ippo_loss(
    logits: Tensor,
    new_values: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    legal_tokens: Tensor,
    *,
    clip: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Clipped PPO plus MSE of acting-seat V[N] against detached GAE returns.

    No implicit advantage normalization or value clipping is performed. All
    rollout labels, including old log probabilities, are detached internally.
    """
    return _ppo_loss(
        logits, new_values, actions, old_log_probs, advantages, returns, legal_tokens,
        clip=clip, entropy_coef=entropy_coef, value_coef=value_coef,
    )


def vrpo_loss(
    logits: Tensor,
    q_selected: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    q_targets: Tensor,
    legal_tokens: Tensor,
    *,
    clip: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Clipped PPO plus MSE of acting-seat selected Q[N], never expected V.

    Pass advantages and Q targets from compute_q_boost, gathered only after
    completing the fixed-player traces. Metrics are detached scalar tensors;
    ``q_loss`` explicitly names the regression (also reported as value_loss).
    """
    total, metrics = _ppo_loss(
        logits, q_selected, actions, old_log_probs, advantages, q_targets, legal_tokens,
        clip=clip, entropy_coef=entropy_coef, value_coef=value_coef,
    )
    metrics["q_loss"] = metrics["value_loss"]
    return total, metrics


__all__ = ["ippo_loss", "vrpo_loss"]
