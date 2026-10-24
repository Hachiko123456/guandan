# SPDX-License-Identifier: MIT
"""Production macro-sequence PPO for MARVEL v1.4 phase 2.

Formulas implemented here (see ``docs/marvel_v14_prompts/02_...md``):

- sequence log-probability ``log pi(m|S) = sum_{t=s_m}^{e_m} log pi(a_t|h_t)``
  accumulated in the log domain (never by multiplying token probabilities);
- one PPO clip on the *complete macro* ratio
  ``r_m = exp(new_logprob_m - old_logprob_m)``:
  ``L_PG,m = max(-r_m A_m, -clip(r_m, 1-eps, 1+eps) A_m)``;
- one policy sample per macro: ``L_PG = mean_m L_PG,m`` (a long macro never
  gets more policy weight than a short one).  With a macro of length one the
  sequence ratio/clip/loss numerically coincide with token PPO;
- profile B uses the *production legacy token GAE* gathered at macro starts;
  profile C uses :mod:`guandan_marvel.training.macro_gae`;
- macro advantage normalization uses the population variance of the valid
  trainable macros of the current policy mini-batch only
  (``unbiased=False``), with DDP count/sum/sumsq all-reduce so every rank sees
  the same global statistics.  ``L_k = (world_size / N) * sum_{i in k} l_i`` so
  that the DDP gradient average equals the global per-macro mean.

All phase-2 macro math is forced to FP32, and a finite log-ratio beyond the
FP32 ``exp`` safety range fails fast instead of being silently clamped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.distributed as dist

from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.training.macro_actions import MacroSegmentation

MACRO_PPO_SCHEMA = "guandan-marvel-macro-sequence-ppo-v1"
MACRO_ADVANTAGE_EPS = 1e-8
MACRO_LOG_RATIO_SAFE_MAX = float(math.log(torch.finfo(torch.float32).max)) - 1.0
MACRO_OLD_LOGPROB_CROSSCHECK_TOLERANCE = 1e-3


class MacroPPOError(ValueError):
    """Raised when a macro-sequence PPO input violates the training contract."""


@dataclass(frozen=True)
class MacroBatchView:
    """Per-mini-batch slice of the phase-1 macro segmentation.

    Re-indexing/mini-batching must be exactly ``full_batch_arrays[indices]``;
    this view is the only supported way to build macro mini-batches so that a
    mini-batch result can never drift from the full-batch gather.
    """

    max_macros: int
    token_to_macro_id: torch.Tensor  # [b, T] long
    macro_start_index: torch.Tensor  # [b, M] long
    macro_end_index: torch.Tensor  # [b, M] long
    macro_length: torch.Tensor  # [b, M] long
    macro_player: torch.Tensor  # [b, M] long
    macro_valid: torch.Tensor  # [b, M] bool
    macro_is_terminal: torch.Tensor  # [b, M] bool
    macro_is_last_for_player: torch.Tensor  # [b, M] bool
    macro_tau: torch.Tensor  # [b, M] long
    macro_old_logprob: torch.Tensor | None  # [b, M] float64, diagnostic only
    macro_train_mask: torch.Tensor | None = None  # [b, M] bool, current-owned complete macros
    macro_policy_owner: torch.Tensor | None = None  # [b, M] long, -1 when unspecified
    incomplete_count: int = 0
    player_switch_violation_count: int = 0


def build_macro_batch_view(segmentation: MacroSegmentation, indices: torch.Tensor) -> MacroBatchView:
    """Slice the full-batch segmentation by trajectory indices (index gather)."""
    if not all(hasattr(segmentation, name) for name in (
        "batch_size", "max_macros", "token_to_macro_id", "macro_start_index",
        "macro_end_index", "macro_length", "macro_player", "macro_valid",
        "macro_is_terminal", "macro_is_last_for_player", "macro_tau",
        "macro_old_logprob", "incomplete_count", "player_switch_violation_count",
    )):
        raise MacroPPOError("macro batch view requires a phase-1 MacroSegmentation")
    if not isinstance(indices, torch.Tensor) or indices.dim() != 1:
        raise MacroPPOError("mini-batch indices must be a 1-D tensor")
    indices = indices.to(dtype=torch.long)
    batch = segmentation.batch_size
    if bool((indices < 0).any()) or bool((indices >= batch).any()):
        raise MacroPPOError("mini-batch indices are outside the rollout batch")

    def take(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.index_select(0, indices.to(value.device))

    return MacroBatchView(
        max_macros=segmentation.max_macros,
        token_to_macro_id=take(segmentation.token_to_macro_id),
        macro_start_index=take(segmentation.macro_start_index),
        macro_end_index=take(segmentation.macro_end_index),
        macro_length=take(segmentation.macro_length),
        macro_player=take(segmentation.macro_player),
        macro_valid=take(segmentation.macro_valid),
        macro_is_terminal=take(segmentation.macro_is_terminal),
        macro_is_last_for_player=take(segmentation.macro_is_last_for_player),
        macro_tau=take(segmentation.macro_tau),
        macro_old_logprob=take(segmentation.macro_old_logprob),
        macro_train_mask=take(getattr(segmentation, "macro_train_mask", None)),
        macro_policy_owner=take(getattr(segmentation, "macro_policy_owner", None)),
        incomplete_count=int(segmentation.incomplete_count),
        player_switch_violation_count=int(segmentation.player_switch_violation_count),
    )


def _field(rollout: TensorDict | Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    value = rollout.get(name) if isinstance(rollout, TensorDict) else rollout.get(name)
    if not isinstance(value, torch.Tensor):
        raise MacroPPOError(f"macro PPO requires rollout field {name!r} as a tensor")
    return value


def require_macro_training_ready(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    view: MacroBatchView,
) -> None:
    """Hard gates required before any macro loss math runs."""
    if int(view.incomplete_count) != 0:
        raise MacroPPOError(
            f"macro training requires incomplete_count == 0, got {view.incomplete_count}"
        )
    if int(view.player_switch_violation_count) != 0:
        raise MacroPPOError(
            "macro training requires player_switch_violation_count == 0, got "
            f"{view.player_switch_violation_count}"
        )
    num_steps = _field(rollout, "num_steps").to(torch.long)
    is_terminal = _field(rollout, "is_terminal").to(torch.bool)
    final_index = (num_steps - 1).clamp(min=0).reshape(-1, 1)
    final_terminal = is_terminal.reshape(num_steps.numel(), -1).gather(1, final_index).reshape(-1)
    if not bool(final_terminal.all()):
        raise MacroPPOError(
            "macro training rejects truncated/incomplete trajectories: every trajectory "
            "must end on a real terminal step"
        )


def macro_trainable_mask(view: MacroBatchView) -> torch.Tensor:
    """Complete macros that may enter a current-policy loss.

    Legacy views (no owner mask) keep ``macro_valid & (end >= 0)`` exactly.
    A phase-6 view adds the current-owner train mask, so opponent macros are
    never counted, normalized over, or used as a value target sample.
    """
    trainable = view.macro_valid & (view.macro_end_index >= 0)
    mask = getattr(view, "macro_train_mask", None)
    if mask is not None:
        if tuple(mask.shape) != tuple(trainable.shape):
            raise MacroPPOError("macro_train_mask shape does not match macro slots")
        trainable = trainable & mask.to(torch.bool)
    return trainable


def _token_trainable_mask(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    view: MacroBatchView,
) -> torch.Tensor | None:
    """Token-level current-sample mask, or ``None`` on the legacy path."""
    player_ids = _field(rollout, "player_ids")
    mask = rollout.get("policy_train_mask") if isinstance(rollout, TensorDict) else rollout.get("policy_train_mask")
    if isinstance(mask, torch.Tensor):
        if tuple(mask.shape) != tuple(player_ids.shape):
            raise MacroPPOError("policy_train_mask must have shape [B, T]")
        if mask.dtype != torch.bool:
            raise TypeError("policy_train_mask must be bool")
        return mask
    macro_mask = getattr(view, "macro_train_mask", None)
    if macro_mask is None:
        return None
    token_to_macro = view.token_to_macro_id
    if token_to_macro.shape != tuple(player_ids.shape):
        raise MacroPPOError("segmentation does not match the rollout shape")
    flat = token_to_macro.clamp(min=0)
    expanded = macro_mask.to(torch.bool).gather(1, flat)
    return expanded & (token_to_macro >= 0)


def macro_active_token_mask(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    view: MacroBatchView,
    *,
    trainable_only: bool = True,
) -> torch.Tensor:
    """``history_mask & (player_ids >= 0) & macro_token_mask`` (spec 2.4.1).

    With phase-6 owner fields present, ``trainable_only=True`` additionally
    restricts the mask to current-owned tokens; opponent tokens remain in the
    timeline and in macro boundaries but never enter a current loss.
    """
    player_ids = _field(rollout, "player_ids")
    num_steps = _field(rollout, "num_steps").to(torch.long)
    b, t = player_ids.shape
    if view.token_to_macro_id.shape != (b, t):
        raise MacroPPOError("segmentation does not match the rollout shape")
    time = torch.arange(t, device=player_ids.device).unsqueeze(0)
    history = time < num_steps.unsqueeze(1)
    active = history & (player_ids >= 0) & (view.token_to_macro_id >= 0)
    if trainable_only:
        token_mask = _token_trainable_mask(rollout, view)
        if token_mask is not None:
            active = active & token_mask
    return active


def validate_active_macro_token_contract(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    *,
    active: torch.Tensor,
    trainable: torch.Tensor | None = None,
) -> None:
    """Spec 2.4.3-2.4.7 gather-safety and reference-policy contract checks.

    ``active`` is the complete set of acting macro tokens (including opponent
    tokens); ``trainable`` is the current-owned subset.  Gather safety and
    positive behaviour probabilities are checked on every acting token, while
    ``on_policy_valid`` and behaviour==reference equality are required only on
    trainable tokens.  On a pooled trajectory ``on_policy_valid`` must equal
    the owner mask on all acting tokens, so an opponent cannot pose as current.
    """
    player_ids = _field(rollout, "player_ids")
    b, t = player_ids.shape
    action_taken = _field(rollout, "action_taken").to(torch.long)
    avail_weights = _field(rollout, "avail_weights")
    behavior = _field(rollout, "behavior_action_prob").to(torch.float32)
    on_policy_valid = _field(rollout, "on_policy_valid").to(torch.bool)
    policy_probs = _field(rollout, "policy_probs").to(torch.float32)
    if action_taken.shape != (b, t):
        raise MacroPPOError("rollout['action_taken'] must have shape [B, T]")
    if avail_weights.dim() != 3 or avail_weights.shape[:2] != (b, t):
        raise MacroPPOError("rollout['avail_weights'] must have shape [B, T, A]")
    a_count = int(avail_weights.shape[-1])
    if behavior.shape != (b, t) or on_policy_valid.shape != (b, t):
        raise MacroPPOError("behavior_action_prob/on_policy_valid must have shape [B, T]")
    if policy_probs.shape != avail_weights.shape:
        raise MacroPPOError("rollout['policy_probs'] must have shape [B, T, A]")
    if trainable is None:
        trainable = active
    if tuple(active.shape) != (b, t) or tuple(trainable.shape) != (b, t):
        raise MacroPPOError("active/trainable masks must have shape [B, T]")
    if bool((trainable & ~active).any()):
        raise MacroPPOError("trainable tokens must be a subset of acting macro tokens")
    if bool((action_taken[active] < 0).any()) or bool((action_taken[active] >= a_count).any()):
        raise MacroPPOError(
            "active macro token selected an action index outside the available-action range"
        )
    safe_action = torch.where(active, action_taken, torch.zeros_like(action_taken))
    selected_weight = avail_weights.gather(-1, safe_action.unsqueeze(-1)).squeeze(-1)
    if bool((selected_weight[active] <= 0).any()):
        raise MacroPPOError("active macro token selected an illegal/unavailable action")
    if bool((behavior[active] <= 0).any()) or not bool(torch.isfinite(behavior[active]).all()):
        raise MacroPPOError("behavior_action_prob must be finite and positive on active macro tokens")
    policy_mask = rollout.get("policy_train_mask") if isinstance(rollout, TensorDict) else rollout.get("policy_train_mask")
    if isinstance(policy_mask, torch.Tensor):
        policy_mask_b = policy_mask.to(torch.bool)
        if bool((on_policy_valid[active] != policy_mask_b[active]).any()):
            raise MacroPPOError(
                "on_policy_valid must equal policy_train_mask on acting tokens; "
                "frozen-opponent behaviour is never a current on-policy sample"
            )
        if bool((policy_mask_b[active] & ~trainable[active]).any()):
            raise MacroPPOError(
                "a current-owned acting token was excluded from the trainable sample set"
            )
    if not bool(on_policy_valid[trainable].all()):
        raise MacroPPOError("all trainable macro tokens must be on-policy (on_policy_valid=True)")
    reference = policy_probs.gather(-1, safe_action.unsqueeze(-1)).squeeze(-1)
    if bool(trainable.any()) and not torch.allclose(
        behavior[trainable], reference[trainable], rtol=1e-5, atol=1e-7,
    ):
        raise MacroPPOError(
            "behavior selected probability must match the rollout reference policy probability "
            "on trainable macro tokens"
        )


def _scatter_token_sums(
    token_values: torch.Tensor,
    active: torch.Tensor,
    view: MacroBatchView,
) -> torch.Tensor:
    b, t = token_values.shape
    m = view.max_macros
    macro_ids = view.token_to_macro_id
    if macro_ids.shape != (b, t):
        raise MacroPPOError("segmentation does not match the rollout shape")
    sums = torch.zeros((b, m), dtype=token_values.dtype, device=token_values.device)
    if m == 0:
        return sums
    safe_ids = torch.where(active, macro_ids, torch.zeros_like(macro_ids))
    contributions = torch.where(active, token_values, torch.zeros_like(token_values))
    sums.scatter_add_(1, safe_ids, contributions)
    return sums


def macro_sequence_logprob(
    token_log_probs: torch.Tensor,
    rollout: TensorDict | Mapping[str, torch.Tensor],
    view: MacroBatchView,
    *,
    active: torch.Tensor,
) -> torch.Tensor:
    """Sum token log-probabilities into one value per complete macro."""
    b, t = _field(rollout, "player_ids").shape
    if token_log_probs.shape != (b, t):
        raise MacroPPOError("token log-probabilities must have shape [B, T]")
    sums = _scatter_token_sums(token_log_probs, active, view)
    valid = view.macro_valid & (view.macro_end_index >= 0)
    if not bool(torch.isfinite(sums[valid]).all()):
        raise MacroPPOError("macro log-probability contains non-finite values")
    return sums


def selected_token_log_prob(
    avail_log_probs_new: torch.Tensor,
    action_taken: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Gather the FP32 selected new log-probability without touching garbage."""
    if avail_log_probs_new.dim() != 3:
        raise MacroPPOError("avail_log_probs_new must have shape [B, T, A]")
    b, t, a = avail_log_probs_new.shape
    if action_taken.shape != (b, t) or active.shape != (b, t):
        raise MacroPPOError("action_taken/active must have shape [B, T]")
    if bool((action_taken[active] < 0).any()) or bool((action_taken[active] >= a).any()):
        raise MacroPPOError("active action index is outside the action range")
    safe_action = torch.where(active, action_taken, torch.zeros_like(action_taken))
    return avail_log_probs_new.gather(-1, safe_action.unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True)
class MacroPGLossOutput:
    pg_loss_mean: torch.Tensor
    clip_frac: float
    macro_count_local: int
    macro_count_global: int
    zero_variance_count: int
    empty_local_batch: bool
    # Rank-consistent DDP aggregates derived from the all-reduced statistics; they are
    # identical on every rank and must not be re-averaged by downstream aggregation.
    adv_raw_mean_global: float
    adv_raw_std_global: float
    adv_norm_mean_global: float
    adv_norm_std_global: float
    adv_count_global: int
    # Local (per-rank) diagnostics; pick up a rank-specific sampling slice only.
    adv_raw_mean: float
    adv_raw_std: float
    adv_raw_min: float
    adv_raw_max: float
    adv_norm_mean: float
    adv_norm_std: float
    adv_norm_min: float
    adv_norm_max: float
    log_ratio_mean: float
    log_ratio_std: float
    log_ratio_min: float
    log_ratio_max: float
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    old_logprob_mean: float
    new_logprob_mean: float
    macro_length_mean: float
    macro_length_max: int
    clip_frac_by_length: Mapping[str, float] = field(default_factory=dict)
    old_logprob_crosscheck_max_abs_diff: float = 0.0
    ratio: torch.Tensor | None = None  # [b, M] detached, valid slots only meaningful
    log_ratio: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    duration: torch.Tensor | None = None


def _all_reduce_vector(vector: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)
    return vector


def connected_zero_like(reference: torch.Tensor) -> torch.Tensor:
    """Return an exact zero that stays connected to ``reference``'s autograd graph.

    Used when a DDP rank holds no local macro samples in a mini-batch while the
    global macro count is positive: that rank must still contribute a
    differentiable (zero) term so the DDP gradient average and unused-parameter
    accounting stay consistent, instead of an empty-set ``mean`` that would
    produce NaN.

    Caveat: this only stays connected if ``reference`` itself is part of the
    model graph.  When a rank holds no macros at all the macro loss is a fresh
    constant (``requires_grad=False``) and ``reference * 0.0`` is disconnected;
    use :func:`model_connected_zero` instead for that case.
    """
    return reference * 0.0


def model_connected_zero(*tensors: torch.Tensor | None) -> torch.Tensor:
    """Return an exact zero anchored to *real model outputs* for autograd.

    An empty DDP rank (local macro count ``0`` while the global count is
    positive) must contribute a differentiable zero to the learner loss so that
    (a) the DDP gradient average stays numerically consistent and (b)
    ``find_unused_parameters=False`` never observes a parameter that received no
    gradient.  Every argument must be a live tensor produced by the current
    ``agent_model(...)`` forward pass (for example ``avail_log_probs`` and
    ``player_state_value``); ``None`` entries are skipped.  The returned value is
    exactly ``0.0``, finite, and ``requires_grad=True`` whenever the anchored
    model outputs do.
    """
    anchor: torch.Tensor | None = None
    for tensor in tensors:
        if tensor is None:
            continue
        # Multiply by zero *before* reducing: masked logits legitimately hold very
        # large finite magnitudes, so ``sum() * 0.0`` can overflow to ``+/-inf``
        # and then to NaN.  ``(x * 0).sum()`` is exactly 0 and stays differentiable.
        term = (tensor.float() * 0.0).sum()
        anchor = term if anchor is None else anchor + term
    if anchor is None:
        raise ValueError("model_connected_zero requires at least one model tensor")
    return anchor


def _global_macro_statistics(
    advantage: torch.Tensor,
    valid: torch.Tensor,
    ratio: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, int, int]:
    """All-reduce (count, sum, sumsq, clip_count, ratio_sum) across ranks."""
    mask_f = valid.to(advantage.dtype)
    local = torch.stack((
        mask_f.sum(),
        (advantage * mask_f).sum(),
        (advantage * advantage * mask_f).sum(),
        ((ratio - 1.0).abs() > clip_coef).to(advantage.dtype).mul(mask_f).sum(),
        (ratio * mask_f).sum(),
    )).to(dtype=torch.float32, device=advantage.device)
    local = local.detach()
    statistics = _all_reduce_vector(local)
    count = int(round(float(statistics[0].item())))
    clip_count = int(round(float(statistics[3].item())))
    return statistics, count, clip_count


def _macro_length_buckets(lengths: torch.Tensor) -> Mapping[str, tuple[float, ...]]:
    return {
        "1": (1, 1),
        "2-4": (2, 4),
        "5-8": (5, 8),
        "9+": (9, None),
    }


# ---------------------------------------------------------------------------
# Optional macro-sample approximate-KL control (``ppo.target_kl``)
# ---------------------------------------------------------------------------
MACRO_TARGET_KL_SCHEMA = "guandan-marvel-macro-target-kl-v1"
# Bounded per-rank sample used only for *approximate* global quantiles.  A mean
# of per-rank quantiles is never reported as the global quantile.
MACRO_KL_SAMPLE_CAPACITY_PER_RANK = 256
MACRO_KL_QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.99)


@dataclass(frozen=True)
class MacroKLControlDecision:
    """Rank-consistent macro sample-KL statistic and stop decision.

    Math (spec section 4): for a valid current-owned macro ``m`` the sampled
    scalar is ``kl_m = exp(log_ratio_m) - 1 - log_ratio_m`` (``expm1`` form).
    The reported statistic is the **global** ``sum(kl_m) / count`` across all
    DDP ranks, never the mean of per-rank means, so unequal macro counts cannot
    bias the decision.
    """

    schema: str
    local_sum: float
    local_count: int
    local_max: float
    global_sum: float
    global_count: int
    global_mean: float | None
    global_max: float
    length_bucket_sums: Mapping[str, float]
    length_bucket_counts: Mapping[str, int]
    quantiles: Mapping[str, float]
    quantile_sample_size: int
    quantiles_approximate: bool
    target_kl: float | None
    stop: bool

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "local_sum": self.local_sum,
            "local_count": self.local_count,
            "local_max": self.local_max,
            "global_sum": self.global_sum,
            "global_count": self.global_count,
            "global_mean": self.global_mean,
            "global_max": self.global_max,
            "length_bucket_sums": dict(self.length_bucket_sums),
            "length_bucket_counts": dict(self.length_bucket_counts),
            "quantiles": dict(self.quantiles),
            "quantile_sample_size": self.quantile_sample_size,
            "quantiles_approximate": self.quantiles_approximate,
            "target_kl": self.target_kl,
            "stop": self.stop,
        }


def macro_sample_kl_values(log_ratio: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Sampled approximate KL ``expm1(lr) - lr`` on valid macros (FP64).

    Invalid slots are excluded *before* the arithmetic so padding/hostile values
    can never leak in.  A non-finite value fails fast instead of being clamped or
    zero-filled.
    """
    if log_ratio.shape != valid.shape:
        raise MacroPPOError("log_ratio and valid must share the macro slot shape")
    mask = valid.to(torch.bool)
    if int(mask.sum().item()) == 0:
        return torch.zeros(0, dtype=torch.float64, device=log_ratio.device)
    selected = log_ratio[mask].detach().to(torch.float64)
    if not bool(torch.isfinite(selected).all()):
        raise MacroPPOError("macro sample KL input contains a non-finite log-ratio")
    values = torch.expm1(selected) - selected
    if not bool(torch.isfinite(values).all()):
        raise MacroPPOError(
            "macro sample KL is non-finite; refusing to clamp or mask an invalid estimator"
        )
    return values


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile of an empty sample is undefined")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def macro_sample_kl_control(
    log_ratio: torch.Tensor,
    valid: torch.Tensor,
    macro_length: torch.Tensor,
    *,
    world_size: int = 1,
    target_kl: float | None = None,
    sample_capacity_per_rank: int = MACRO_KL_SAMPLE_CAPACITY_PER_RANK,
) -> MacroKLControlDecision:
    """Global macro sample KL and the rank-consistent early-stop decision.

    The two only-legal collective calls are an unconditional all-reduce of the
    scalar statistics and an unconditional (world_size>1) object gather of the
    bounded quantile sample; both run on every rank in every minibatch so the
    stop decision cannot deadlock or diverge between ranks.  An empty local rank
    contributes zeros.  ``global_count == 0`` yields ``global_mean=None`` and
    never stops (there is nothing to measure), which is distinct from a finite
    zero KL.
    """
    if world_size < 1:
        raise MacroPPOError("world_size must be positive")
    if target_kl is not None:
        value = float(target_kl)
        if not math.isfinite(value) or value <= 0.0:
            raise MacroPPOError("target_kl must be None or a finite positive number")
    if macro_length.shape != valid.shape:
        raise MacroPPOError("macro_length and valid must share the macro slot shape")
    values = macro_sample_kl_values(log_ratio, valid)
    local_count = int(values.numel())
    local_sum = float(values.sum().item()) if local_count else 0.0
    local_max = float(values.max().item()) if local_count else 0.0

    # Per-length-bucket sums/counts, all-reduced together with the scalar stats.
    lengths = macro_length[valid.to(torch.bool)].detach().to(torch.float64)
    bucket_spec = _macro_length_buckets(lengths)
    bucket_names = list(bucket_spec)
    local_buckets: list[float] = []
    for name in bucket_names:
        low, high = bucket_spec[name]
        bucket = (lengths >= low) & (lengths <= high) if high is not None else (lengths >= low)
        local_buckets.append(float(values[bucket].sum().item()) if local_count else 0.0)
        local_buckets.append(float(bucket.sum().item()) if local_count else 0.0)

    statistics = torch.tensor(
        [local_sum, float(local_count), local_max, *local_buckets],
        dtype=torch.float64, device=log_ratio.device,
    ).detach()
    if dist.is_available() and dist.is_initialized() and int(world_size) > 1:
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    global_sum = float(statistics[0].item())
    global_count = int(round(float(statistics[1].item())))
    global_max = float(statistics[2].item())
    bucket_sums: dict[str, float] = {}
    bucket_counts: dict[str, int] = {}
    for index, name in enumerate(bucket_names):
        bucket_sums[name] = float(statistics[3 + 2 * index].item())
        bucket_counts[name] = int(round(float(statistics[4 + 2 * index].item())))
    global_mean = (global_sum / global_count) if global_count > 0 else None

    # Bounded approximate global quantile sample.  The gather is a collective and
    # must therefore be executed by every rank even when a rank is empty.
    sample = values[:sample_capacity_per_rank].tolist() if local_count else []
    gathered: list[list[float]] = [[]]
    if dist.is_available() and dist.is_initialized() and int(world_size) > 1:
        gathered = [None] * int(world_size)  # type: ignore[list-item]
        dist.all_gather_object(gathered, sample)
    else:
        gathered = [sample]
    flat = [float(value) for chunk in gathered for value in (chunk or [])]
    flat.sort()
    quantiles = {
        f"p{int(round(q * 100)):02d}": _quantile(flat, q) for q in MACRO_KL_QUANTILES
    } if flat else {}
    approximate = len(flat) < global_count

    stop = bool(
        target_kl is not None
        and global_count > 0
        and global_mean is not None
        and global_mean > float(target_kl)
    )
    return MacroKLControlDecision(
        schema=MACRO_TARGET_KL_SCHEMA,
        local_sum=local_sum,
        local_count=local_count,
        local_max=local_max,
        global_sum=global_sum,
        global_count=global_count,
        global_mean=global_mean,
        global_max=global_max,
        length_bucket_sums=bucket_sums,
        length_bucket_counts=bucket_counts,
        quantiles=quantiles,
        quantile_sample_size=len(flat),
        quantiles_approximate=approximate,
        target_kl=None if target_kl is None else float(target_kl),
        stop=stop,
    )


def compute_macro_pg_loss(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    *,
    view: MacroBatchView,
    avail_log_probs_new: torch.Tensor,
    macro_advantage: torch.Tensor,
    clip_coef: float,
    world_size: int = 1,
) -> MacroPGLossOutput:
    """Complete macro-sequence PPO policy loss with one clip per macro."""
    require_macro_training_ready(rollout, view)
    active_all = macro_active_token_mask(rollout, view, trainable_only=False)
    active = macro_active_token_mask(rollout, view, trainable_only=True)
    validate_active_macro_token_contract(rollout, active=active_all, trainable=active)

    player_ids = _field(rollout, "player_ids")
    b, t = player_ids.shape
    if avail_log_probs_new.shape[:2] != (b, t):
        raise MacroPPOError("avail_log_probs_new must have shape [B, T, A]")
    if avail_log_probs_new.dtype != torch.float32:
        avail_log_probs_new = avail_log_probs_new.float()
    action_taken = _field(rollout, "action_taken").to(torch.long)
    behavior = _field(rollout, "behavior_action_prob").to(torch.float32)

    if macro_advantage.dim() != 2 or macro_advantage.shape[0] != b:
        raise MacroPPOError("macro_advantage must have shape [b, M] matching the mini-batch")
    macro_advantage = macro_advantage.to(dtype=torch.float32, device=avail_log_probs_new.device).detach()
    if macro_advantage.shape != view.macro_valid.shape:
        raise MacroPPOError("macro_advantage must have shape [b, M] matching the segmentation view")

    new_tok = selected_token_log_prob(avail_log_probs_new, action_taken, active_all)
    if not bool(torch.isfinite(new_tok[active]).all()):
        raise MacroPPOError("active new selected log-probability is non-finite")
    new_macro = macro_sequence_logprob(new_tok, rollout, view, active=active)

    old_tok = torch.where(
        active_all, behavior.clamp(min=torch.finfo(torch.float32).tiny).log(),
        torch.zeros_like(behavior),
    )
    old_macro = macro_sequence_logprob(old_tok, rollout, view, active=active)

    crosscheck_diff = 0.0
    if view.macro_old_logprob is not None:
        reference = view.macro_old_logprob.to(dtype=torch.float32, device=old_macro.device)
        valid = macro_trainable_mask(view)
        diff = (old_macro - reference).abs()
        crosscheck_diff = float(diff[valid].max().item()) if bool(valid.any()) else 0.0
        if crosscheck_diff > MACRO_OLD_LOGPROB_CROSSCHECK_TOLERANCE:
            raise MacroPPOError(
                "production macro old log-probability disagrees with the phase-1 diagnostic "
                f"segmentation field (max abs diff {crosscheck_diff:.6g})"
            )

    trainable = macro_trainable_mask(view)
    if not bool(torch.isfinite(macro_advantage[view.macro_valid]).all()):
        raise MacroPPOError("macro advantage contains non-finite values")

    log_ratio = new_macro - old_macro
    if not bool(torch.isfinite(log_ratio[trainable]).all()):
        raise MacroPPOError("macro log-ratio is non-finite; refusing to train on an invalid estimator")
    if bool(trainable.any()) and bool((log_ratio[trainable] > MACRO_LOG_RATIO_SAFE_MAX).any()):
        index = torch.nonzero(trainable & (log_ratio > MACRO_LOG_RATIO_SAFE_MAX))
        b_index, m_index = int(index[0, 0]), int(index[0, 1])
        raise MacroPPOError(
            "macro log-ratio exceeds the FP32 exp safety range on sample "
            f"batch={b_index} macro={m_index} length={int(view.macro_length[b_index, m_index])} "
            f"log_ratio={float(log_ratio[b_index, m_index]):.6g} "
            f"old_logprob={float(old_macro[b_index, m_index]):.6g} "
            f"new_logprob={float(new_macro[b_index, m_index]):.6g}; no silent clamp is applied"
        )
    ratio = torch.exp(log_ratio)
    if not bool(torch.isfinite(ratio[trainable]).all()):
        raise MacroPPOError("macro ratio is non-finite; refusing to train on an invalid estimator")

    statistics, global_count, global_clip_count = _global_macro_statistics(
        macro_advantage, trainable, ratio, float(clip_coef),
    )
    if world_size > 1 and int(world_size) <= 0:
        raise MacroPPOError("world_size must be positive")
    if global_count == 0:
        if getattr(view, "macro_train_mask", None) is None:
            raise MacroPPOError("no valid trainable macros in the current policy mini-batch")
        # Pool path: this mini-batch legitimately contains only opponent macros
        # on every rank.  Emit a model-connected exact zero so every parameter
        # still participates in the DDP graph, and never average an empty set.
        zero = model_connected_zero(avail_log_probs_new)
        shape = view.macro_valid.shape
        empty = torch.zeros(shape, dtype=torch.float32, device=avail_log_probs_new.device)
        return MacroPGLossOutput(
            pg_loss_mean=zero,
            clip_frac=0.0,
            macro_count_local=int(trainable.sum().item()),
            macro_count_global=0,
            zero_variance_count=0,
            empty_local_batch=True,
            adv_raw_mean_global=0.0,
            adv_raw_std_global=0.0,
            adv_norm_mean_global=0.0,
            adv_norm_std_global=0.0,
            adv_count_global=0,
            adv_raw_mean=0.0,
            adv_raw_std=0.0,
            adv_raw_min=0.0,
            adv_raw_max=0.0,
            adv_norm_mean=0.0,
            adv_norm_std=0.0,
            adv_norm_min=0.0,
            adv_norm_max=0.0,
            log_ratio_mean=0.0,
            log_ratio_std=0.0,
            log_ratio_min=0.0,
            log_ratio_max=0.0,
            ratio_mean=0.0,
            ratio_min=0.0,
            ratio_max=0.0,
            old_logprob_mean=0.0,
            new_logprob_mean=0.0,
            macro_length_mean=0.0,
            macro_length_max=0,
            clip_frac_by_length={},
            old_logprob_crosscheck_max_abs_diff=crosscheck_diff,
            ratio=empty,
            log_ratio=empty,
            valid=torch.zeros(shape, dtype=torch.bool, device=empty.device),
            duration=view.macro_tau.detach(),
        )
    global_mean = statistics[1] / statistics[0]
    variance = torch.clamp(statistics[2] / statistics[0] - global_mean * global_mean, min=0.0)
    global_std = torch.sqrt(variance)
    zero_variance = bool(float(variance.item()) <= 0.0) or global_count <= 1
    # Rank-consistent global aggregates: every rank must report the same numbers so that
    # downstream metrics aggregation can take them as-is instead of averaging rank means.
    global_mean_value = float(global_mean.item())
    global_std_value = float(global_std.item())
    adv_norm = (macro_advantage - global_mean) / (global_std + MACRO_ADVANTAGE_EPS)
    adv_norm = torch.where(trainable, adv_norm, torch.zeros_like(adv_norm))

    pg_loss1 = -adv_norm * ratio
    pg_loss2 = -adv_norm * torch.clamp(ratio, 1.0 - float(clip_coef), 1.0 + float(clip_coef))
    per_macro = torch.max(pg_loss1, pg_loss2)
    per_macro = torch.where(trainable, per_macro, torch.zeros_like(per_macro))
    local_loss = per_macro.sum() * (float(world_size) / float(global_count))

    if global_count > 0:
        clip_fraction = float(global_clip_count) / float(global_count)
    else:  # pragma: no cover - global_count == 0 already failed fast
        clip_fraction = 0.0

    valid_adv = macro_advantage[trainable]
    valid_norm = adv_norm[trainable]
    valid_log_ratio = log_ratio[trainable]
    valid_ratio = ratio[trainable]
    lengths = view.macro_length[trainable]
    clip_buckets: dict[str, float] = {}
    for name, (low, high) in _macro_length_buckets(lengths).items():
        bucket = (lengths >= low) & (lengths <= high) if high is not None else (lengths >= low)
        if bool(bucket.any()):
            clipped = ((ratio[trainable] - 1.0).abs() > float(clip_coef))[bucket]
            clip_buckets[name] = float(clipped.float().mean().item())

    local_count = int(trainable.sum().item())
    return MacroPGLossOutput(
        pg_loss_mean=local_loss,
        clip_frac=clip_fraction,
        macro_count_local=local_count,
        macro_count_global=global_count,
        zero_variance_count=1 if zero_variance else 0,
        empty_local_batch=local_count == 0,
        adv_raw_mean_global=global_mean_value,
        adv_raw_std_global=global_std_value,
        adv_norm_mean_global=0.0,
        adv_norm_std_global=global_std_value / (global_std_value + MACRO_ADVANTAGE_EPS),
        adv_count_global=global_count,
        adv_raw_mean=float(valid_adv.mean().item()) if local_count else 0.0,
        adv_raw_std=float(valid_adv.std(unbiased=False).item()) if local_count else 0.0,
        adv_raw_min=float(valid_adv.min().item()) if local_count else 0.0,
        adv_raw_max=float(valid_adv.max().item()) if local_count else 0.0,
        adv_norm_mean=float(valid_norm.mean().item()) if local_count else 0.0,
        adv_norm_std=float(valid_norm.std(unbiased=False).item()) if local_count else 0.0,
        adv_norm_min=float(valid_norm.min().item()) if local_count else 0.0,
        adv_norm_max=float(valid_norm.max().item()) if local_count else 0.0,
        log_ratio_mean=float(valid_log_ratio.mean().item()) if local_count else 0.0,
        log_ratio_std=float(valid_log_ratio.std(unbiased=False).item()) if local_count else 0.0,
        log_ratio_min=float(valid_log_ratio.min().item()) if local_count else 0.0,
        log_ratio_max=float(valid_log_ratio.max().item()) if local_count else 0.0,
        ratio_mean=float(valid_ratio.mean().item()) if local_count else 0.0,
        ratio_min=float(valid_ratio.min().item()) if local_count else 0.0,
        ratio_max=float(valid_ratio.max().item()) if local_count else 0.0,
        old_logprob_mean=float(old_macro[trainable].mean().item()) if local_count else 0.0,
        new_logprob_mean=float(new_macro[trainable].mean().item()) if local_count else 0.0,
        macro_length_mean=float(lengths.float().mean().item()) if local_count else 0.0,
        macro_length_max=int(lengths.max().item()) if local_count else 0,
        clip_frac_by_length=clip_buckets,
        old_logprob_crosscheck_max_abs_diff=crosscheck_diff,
        ratio=ratio.detach(),
        log_ratio=log_ratio.detach(),
        valid=trainable.detach(),
        duration=view.macro_tau.detach(),
    )


@dataclass(frozen=True)
class MacroValueLossOutput:
    macro_value_loss_mean: torch.Tensor
    macro_count_local: int
    macro_count_global: int


def compute_macro_value_loss(
    *,
    player_state_value_pred: torch.Tensor,
    view: MacroBatchView,
    macro_value_target: torch.Tensor,
    world_size: int = 1,
) -> MacroValueLossOutput:
    """Huber value loss at macro starts, once per valid macro (Profile C)."""
    if player_state_value_pred.dim() != 2:
        raise MacroPPOError("player_state_value_pred must have shape [b, T]")
    b, _t = player_state_value_pred.shape
    if view.macro_start_index.shape[0] != b:
        raise MacroPPOError("value loss view does not match the mini-batch")
    if macro_value_target.shape != view.macro_valid.shape:
        raise MacroPPOError("macro_value_target must have shape [b, M]")
    trainable = macro_trainable_mask(view)
    count_tensor = trainable.to(torch.float32).sum().detach()
    count_tensor = _all_reduce_vector(count_tensor)
    global_count = int(round(float(count_tensor.item())))
    if global_count == 0:
        if getattr(view, "macro_train_mask", None) is None:
            raise MacroPPOError("no valid macros for the macro value loss")
        # A globally opponent-only mini-batch carries no current value sample;
        # stay graph-connected with an exact zero instead of an empty-set mean.
        return MacroValueLossOutput(
            macro_value_loss_mean=model_connected_zero(player_state_value_pred),
            macro_count_local=0,
            macro_count_global=0,
        )
    if not bool(torch.isfinite(macro_value_target[trainable]).all()):
        raise MacroPPOError("macro value target contains non-finite values")

    indices = view.macro_start_index.clamp(min=0)
    prediction = player_state_value_pred.gather(1, indices)
    elementwise = torch.nn.functional.huber_loss(
        prediction, macro_value_target.to(prediction.dtype), reduction="none", delta=10.0,
    )
    masked = torch.where(trainable, elementwise, torch.zeros_like(elementwise))
    weighted = masked.sum() * (float(world_size) / float(global_count))
    return MacroValueLossOutput(
        macro_value_loss_mean=weighted,
        macro_count_local=int(trainable.sum().item()),
        macro_count_global=global_count,
    )


def explain_macro_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    degenerate_threshold: float = 1e-12,
) -> tuple[float | None, int]:
    """Macro explained variance; degenerate target variance is reported undefined."""
    mask = valid.to(torch.bool)
    if int(mask.sum().item()) == 0:
        return None, 0
    y = target[mask].to(torch.float64)
    y_hat = prediction[mask].to(torch.float64)
    variance = float(y.var(unbiased=False).item())
    if not math.isfinite(variance) or variance < float(degenerate_threshold):
        return None, 1
    residual = float((y - y_hat).var(unbiased=False).item())
    return 1.0 - residual / variance, 0


# ---------------------------------------------------------------------------
# Regularizer (independent ablation switch)
# ---------------------------------------------------------------------------


def _token_kl_uniform(
    avail_probs_new: torch.Tensor,
    avail_weights: torch.Tensor,
) -> torch.Tensor:
    """Legacy per-token ``KL(pi || U_legal)`` formula (eps offset included)."""
    eps = 1e-9
    probs = avail_probs_new + eps
    uniform = avail_weights / avail_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return (probs * (probs.log() - (uniform + eps).log())).sum(dim=-1)


def compute_macro_normalized_kl_mean(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    *,
    view: MacroBatchView,
    avail_probs_new: torch.Tensor,
    world_size: int = 1,
) -> torch.Tensor:
    """Differentiable ``mean_m (1/L_m) sum_t KL(pi_t || U_t)`` (per-macro equal weight)."""
    require_macro_training_ready(rollout, view)
    active_all = macro_active_token_mask(rollout, view, trainable_only=False)
    active = macro_active_token_mask(rollout, view, trainable_only=True)
    validate_active_macro_token_contract(rollout, active=active_all, trainable=active)
    avail_weights = _field(rollout, "avail_weights").to(torch.float32)
    if avail_probs_new.shape != avail_weights.shape:
        raise MacroPPOError("avail_probs_new must match rollout['avail_weights'] shape")
    token_kl = _token_kl_uniform(avail_probs_new.float(), avail_weights)
    # ``_scatter_token_sums`` masks with ``where(active, ...)`` so padding rows can
    # never contribute (or divide) - even if they carry hostile values.
    sums = _scatter_token_sums(token_kl, active, view)
    active_f = active.to(torch.float32)
    counts = _scatter_token_sums(active_f, active, view)
    trainable = macro_trainable_mask(view)
    lengths = torch.where(trainable, counts.clamp(min=1.0), torch.ones_like(counts))
    per_macro = sums / lengths
    per_macro = torch.where(trainable, per_macro, torch.zeros_like(per_macro))
    count_tensor = trainable.to(torch.float32).sum().detach()
    count_tensor = _all_reduce_vector(count_tensor)
    global_count = int(round(float(count_tensor.item())))
    if global_count == 0:
        if getattr(view, "macro_train_mask", None) is None:
            raise MacroPPOError("no valid macros for the macro regularizer")
        return model_connected_zero(avail_probs_new)
    weighted = per_macro.sum() * (float(world_size) / float(global_count))
    return weighted


def macro_regularizer_diagnostics(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    *,
    view: MacroBatchView,
    avail_probs_new: torch.Tensor,
    avail_weights: torch.Tensor | None = None,
) -> Mapping[str, float]:
    """Dual-track diagnostics: token KL, macro KL, raw and normalized entropy."""
    active_all = macro_active_token_mask(rollout, view, trainable_only=False)
    active = macro_active_token_mask(rollout, view, trainable_only=True)
    opponent_token_count = int((active_all & ~active).sum().item())
    if avail_weights is None:
        avail_weights = _field(rollout, "avail_weights").to(torch.float32)
    probs = avail_probs_new.detach().float()
    weights = avail_weights.detach().float()
    with torch.no_grad():
        token_kl = _token_kl_uniform(probs, weights)
        active_f = active.to(torch.float32)
        legal_counts = (weights > 0).sum(dim=-1).to(torch.float32)
        probs_safe = torch.where(probs > 0, probs, torch.zeros_like(probs))
        entropy = -(probs_safe * torch.where(probs_safe > 0, probs_safe.log(), torch.zeros_like(probs_safe))).sum(dim=-1)
        log_legal = torch.where(legal_counts > 1, legal_counts.log(), torch.ones_like(legal_counts))
        normalized = torch.where(legal_counts > 1, entropy / log_legal, torch.zeros_like(entropy))
        token_kl_masked = torch.where(active, token_kl, torch.zeros_like(token_kl))
        entropy_masked = torch.where(active, entropy, torch.zeros_like(entropy))
        normalized_masked = torch.where(active, normalized, torch.zeros_like(normalized))
        token_count = float(active_f.sum().item())
        trainable = macro_trainable_mask(view)
        counts = _scatter_token_sums(active_f, active, view)
        lengths = torch.where(trainable, counts.clamp(min=1.0), torch.ones_like(counts))
        macro_kl = torch.where(
            trainable, _scatter_token_sums(token_kl_masked, active, view) / lengths, torch.zeros_like(counts),
        )
        macro_entropy = torch.where(
            trainable, _scatter_token_sums(entropy_masked, active, view) / lengths, torch.zeros_like(counts),
        )
        macro_normalized = torch.where(
            trainable, _scatter_token_sums(normalized_masked, active, view) / lengths, torch.zeros_like(counts),
        )
        macro_count = int(trainable.sum().item())
        # Forced (single-legal-action) decisions are recorded as zero entropy by
        # construction.  Keep the all-token mean and the non-forced mean apart so
        # a change in the forced-action mix cannot masquerade as entropy change.
        non_forced = active & (legal_counts > 1)
        non_forced_count = float(non_forced.to(torch.float32).sum().item())
        non_forced_entropy = float(
            (normalized_masked * non_forced.to(normalized_masked.dtype)).sum().item()
        )
        return {
            "legacy_token_kl_mean": float(token_kl_masked.sum().item()) / token_count if token_count else 0.0,
            "macro_kl_mean": float(macro_kl[trainable].mean().item()) if macro_count else 0.0,
            "raw_entropy_mean": float(entropy_masked.sum().item()) / token_count if token_count else 0.0,
            "normalized_entropy_mean": (
                float(macro_normalized[trainable].mean().item()) if macro_count else 0.0
            ),
            "non_forced_normalized_entropy_mean": (
                non_forced_entropy / non_forced_count if non_forced_count else 0.0
            ),
            "non_forced_token_count": non_forced_count,
            "token_count": token_count,
            "opponent_token_count": opponent_token_count,
            "macro_count": macro_count,
            "single_legal_token_count": int((active & (legal_counts <= 1)).sum().item()),
        }


def parameter_group_gradient_norms(module: torch.nn.Module) -> Mapping[str, float]:
    """Per-parameter-group gradient norms (diagnostics only, no math)."""
    groups = {
        "decoder": ("decoder.",),
        "state": ("state_tokens_embed.", "state_player_embed.", "state_mlp."),
        "action": ("action_tokens_embed.", "action_mlp."),
        "logit": ("logit_head.",),
        "value": ("value_head.",),
    }
    totals: dict[str, float] = {}
    for name, prefixes in groups.items():
        squared = 0.0
        for parameter_name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            if any(parameter_name.startswith(prefix) for prefix in prefixes):
                squared += float(parameter.grad.detach().float().pow(2).sum().item())
        totals[name] = math.sqrt(squared)
    return totals


__all__ = [
    "MACRO_ADVANTAGE_EPS", "MACRO_KL_QUANTILES", "MACRO_KL_SAMPLE_CAPACITY_PER_RANK",
    "MACRO_LOG_RATIO_SAFE_MAX", "MACRO_OLD_LOGPROB_CROSSCHECK_TOLERANCE",
    "MACRO_PPO_SCHEMA", "MACRO_TARGET_KL_SCHEMA",
    "MacroBatchView", "MacroKLControlDecision", "MacroPGLossOutput", "MacroPPOError",
    "MacroValueLossOutput", "build_macro_batch_view", "compute_macro_normalized_kl_mean",
    "connected_zero_like", "compute_macro_pg_loss", "compute_macro_value_loss", "explain_macro_value",
    "macro_active_token_mask", "macro_sample_kl_control", "macro_sample_kl_values",
    "macro_trainable_mask", "macro_regularizer_diagnostics",
    "macro_sequence_logprob",
    "model_connected_zero", "parameter_group_gradient_norms", "require_macro_training_ready",
    "selected_token_log_prob", "validate_active_macro_token_contract",
]
