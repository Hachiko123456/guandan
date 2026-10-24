# SPDX-License-Identifier: MIT
"""Phase-1 legacy token-GAE equivalence oracle.

The production training path computes token deltas and token advantages on the
global token timeline (``guandan_marvel.training.gae.compute_gae_from_value``).
This module provides an independent, diagnosis-only reconstruction of that
exact quantity at macro-action starts:

1. ``compute_legacy_token_advantages`` calls the *current* production GAE
   function, so it is the ground truth for regression.
2. ``compute_legacy_token_deltas`` re-derives the per-token TD residual with an
   independent forward scan over the same rollout contract.
3. ``expand_macro_start_advantages`` compresses those token deltas into the
   legacy macro-start advantage using real token indices::

       A_old_macro_start(m)
           = sum_i (gamma*lambda)^(offset_i) * delta_token(m, i)
             + (gamma*lambda)^tau_m * A_old_macro_start(m + 1)

   where ``offset_i = t_i - start_m`` for the real token indices of macro ``m``
   and ``tau_m`` is the global token distance from the start of macro ``m`` to
   the start of the *same player's* next macro.  No contiguity assumption is
   made; chance steps and other players' tokens inside ``tau_m`` are counted.

The oracle exists only to prove that macro segmentation, reward aggregation and
value bootstrapping reproduce the legacy token GAE; it is never used to produce
training advantages.  A single macro length-1 step does **not** make the
plan's ``gamma^tau * lambda`` macro clock equivalent to this legacy form: only
``(gamma*lambda)^tau`` reproduces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch

from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.training.gae import compute_gae_from_value
from guandan_marvel.training.macro_actions import MacroSegmentation, segment_rollout_macros


@dataclass(frozen=True)
class LegacyMacroGAEOracleOutput:
    """Legacy token GAE versus its independent macro-start compression."""

    segmentation: MacroSegmentation
    token_advantage: torch.Tensor  # [B, T] production token GAE
    token_delta: torch.Tensor  # [B, T] independent token TD residual
    legacy_at_macro_start: torch.Tensor  # [B, M] token GAE gathered at starts
    expanded_at_macro_start: torch.Tensor  # [B, M] macro compression of deltas
    macro_abs_diff: torch.Tensor  # [B, M] |gathered - expanded| on valid macros
    max_abs_diff: float


def compute_legacy_token_advantages(
    rollout: TensorDict,
    state_values: torch.Tensor,
    *,
    discount_gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> torch.Tensor:
    """Return the production token-level advantage for every token step."""
    output = compute_gae_from_value(
        rollout,
        state_values,
        discount_gamma=discount_gamma,
        gae_lambda=gae_lambda,
    )
    return output.player_advantage


def compute_legacy_token_deltas(
    rollout: TensorDict,
    state_values: torch.Tensor,
    *,
    discount_gamma: float = 1.0,
) -> torch.Tensor:
    """Re-derive the legacy per-token TD residual with an explicit reference loop.

    For a player step at time ``t`` whose next acting step is ``t_next``::

        delta(t) = sum_{j=t}^{t_next - 1} gamma^(j - t) * reward_j
                   + gamma^(t_next - t) * V_{t_next} - V_t

    When the player never acts again the window runs to that trajectory's own
    ``num_steps`` and there is no bootstrap term::

        delta(t) = sum_{j=t}^{n - 1} gamma^(j - t) * reward_j - V_t

    Rewards are therefore consumed by the player's *preceding* decision point
    and the bootstrap value is taken at the *next* acting step, mirroring
    ``compute_gae_from_value`` exactly.  This is a deliberately simple,
    independent reference implementation (per batch, per player, explicit
    windows enumerated from the real token indices) - it never calls or copies
    the production backward scan.

    ``discount_gamma`` may lie anywhere in ``[0, 1]``.  At ``gamma = 0`` the
    mathematical convention ``0 ** 0 == 1`` applies, so only the reward at the
    decision token itself survives and all later rewards/bootstrap vanish.
    Non-acting positions stay zero and padding (``t >= num_steps``) is never
    read.
    """
    num_steps = rollout["num_steps"]
    player_ids = rollout["player_ids"]
    rewards = rollout["rewards"]
    if rewards.dim() != 3:
        raise ValueError("legacy oracle: rewards must have shape [B, T, P]")
    if state_values.shape != rewards.shape:
        raise ValueError("legacy oracle: state_values must match rewards [B, T, P]")
    if player_ids.shape != rewards.shape[:2]:
        raise ValueError("legacy oracle: player_ids must have shape [B, T]")
    B, T, P = rewards.shape
    if num_steps.shape != (B,):
        raise ValueError("legacy oracle: num_steps must have shape [B]")
    gamma = float(discount_gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("legacy oracle: discount_gamma must lie in [0, 1]")

    keep = rewards.dtype
    device = rewards.device
    rewards = rewards.to(keep)
    values = state_values.to(keep)
    player_ids = player_ids.to(device)
    num_steps = num_steps.to(device)
    delta = torch.zeros((B, T), dtype=keep, device=device)
    zero = torch.zeros((), dtype=keep, device=device)

    for b in range(B):
        n = int(num_steps[b].item())
        if n < 0 or n > T:
            raise ValueError("legacy oracle: num_steps must lie within 0..horizon")
        for p in range(P):
            times = [t for t in range(n) if int(player_ids[b, t].item()) == p]
            for i, start in enumerate(times):
                has_next = i + 1 < len(times)
                stop = times[i + 1] if has_next else n
                reward_sum = zero
                for j in range(start, stop):
                    reward_sum = reward_sum + (gamma ** (j - start)) * rewards[b, j, p]
                bootstrap = zero
                if has_next:
                    bootstrap = (gamma ** (stop - start)) * values[b, stop, p]
                delta[b, start] = reward_sum + bootstrap - values[b, start, p]
    return delta


def expand_macro_start_advantages(
    token_delta: torch.Tensor,
    segmentation: MacroSegmentation,
    *,
    discount_gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> torch.Tensor:
    """Compress token deltas into legacy-equivalent macro-start advantages.

    Implements ``A_m = sum_i (gamma*lambda)^(t_i - start_m) * delta(t_i)
    + (gamma*lambda)^tau_m * A_{m+1}`` using the real token indices stored in
    ``segmentation``; macros are processed in reverse start order so that each
    same-player successor is already expanded.
    """
    B, T = token_delta.shape
    if segmentation.token_to_macro_id.shape != (B, T):
        raise ValueError("legacy oracle: segmentation does not match token_delta shape")
    M = segmentation.max_macros
    decay = float(discount_gamma) * float(gae_lambda)
    if not (0.0 <= decay <= 1.0):
        raise ValueError("legacy oracle: discount_gamma * gae_lambda must lie in [0, 1]")
    if M == 0:
        return torch.zeros((B, 0), dtype=token_delta.dtype, device=token_delta.device)

    device = token_delta.device
    macro_ids = segmentation.token_to_macro_id
    valid = macro_ids >= 0
    positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    starts = segmentation.macro_start_index.gather(1, macro_ids.clamp(min=0))
    offsets = torch.where(valid, positions - starts, torch.zeros_like(positions))
    if bool((offsets < 0).any()):
        raise ValueError("legacy oracle: token precedes its macro start")
    offset_decay = torch.pow(
        torch.as_tensor(decay, dtype=token_delta.dtype, device=device),
        offsets.to(token_delta.dtype),
    )
    contributions = torch.where(valid, token_delta * offset_decay, torch.zeros_like(token_delta))

    expanded = torch.zeros((B, M), dtype=token_delta.dtype, device=device)
    decay_tensor = torch.as_tensor(decay, dtype=token_delta.dtype, device=device)
    for macro in range(M - 1, -1, -1):
        segment_sum = (contributions * (macro_ids == macro).to(token_delta.dtype)).sum(dim=1)
        successor = segmentation.macro_next_index[:, macro]
        successor_value = expanded.gather(1, successor.clamp(min=0).unsqueeze(1)).squeeze(1)
        tau = segmentation.macro_tau[:, macro].to(token_delta.dtype)
        successor_term = torch.where(
            successor >= 0,
            torch.pow(decay_tensor, tau) * successor_value,
            torch.zeros_like(successor_value),
        )
        expanded[:, macro] = segment_sum + successor_term
    return expanded


def legacy_gae_macro_oracle(
    rollout: TensorDict,
    state_values: torch.Tensor,
    *,
    discount_gamma: float = 1.0,
    gae_lambda: float = 0.95,
    segmentation: MacroSegmentation | None = None,
) -> LegacyMacroGAEOracleOutput:
    """Compare the production token GAE with the macro-start expansion.

    Both sides must agree numerically: the production token advantage gathered
    at macro starts versus the independently expanded token deltas.
    """
    if segmentation is None:
        segmentation = segment_rollout_macros(rollout)
    token_advantage = compute_legacy_token_advantages(
        rollout,
        state_values,
        discount_gamma=discount_gamma,
        gae_lambda=gae_lambda,
    )
    token_delta = compute_legacy_token_deltas(
        rollout,
        state_values,
        discount_gamma=discount_gamma,
    ).to(token_advantage.dtype)
    expanded = expand_macro_start_advantages(
        token_delta,
        segmentation,
        discount_gamma=discount_gamma,
        gae_lambda=gae_lambda,
    )
    start = segmentation.macro_start_index
    gathered = token_advantage.gather(1, start.clamp(min=0))
    valid = start >= 0
    macro_abs_diff = torch.where(valid, (gathered - expanded).abs(), torch.zeros_like(gathered))
    max_abs_diff = float(macro_abs_diff.max().item()) if bool(macro_abs_diff.numel()) else 0.0
    return LegacyMacroGAEOracleOutput(
        segmentation=segmentation,
        token_advantage=token_advantage,
        token_delta=token_delta,
        legacy_at_macro_start=gathered,
        expanded_at_macro_start=expanded,
        macro_abs_diff=macro_abs_diff,
        max_abs_diff=max_abs_diff,
    )


__all__ = [
    "LegacyMacroGAEOracleOutput",
    "compute_legacy_token_advantages",
    "compute_legacy_token_deltas",
    "expand_macro_start_advantages",
    "legacy_gae_macro_oracle",
]
