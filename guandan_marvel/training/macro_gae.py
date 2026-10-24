# SPDX-License-Identifier: MIT
"""Player macro-decision GAE for MARVEL v1.4 phase-2 Profile C.

Semantics (deliberately *not* equivalent to legacy token GAE)
-------------------------------------------------------------
For player ``p``'s macro decision ``m`` starting at token ``s = s_m^p``:

- ``u_m = t_{m+1}^p`` (that player's next macro start) and ``c_m = 1`` when a
  successor exists;
- ``u_m = num_steps[b]`` and ``c_m = 0`` for the player's last macro.  Phase-1
  keeps ``macro_tau = 0`` for that last macro; production macro GAE must use
  the explicit ``num_steps - start`` window and must never treat ``tau = 0`` as
  the terminal duration.
- ``Delta_m = u_m - s_m`` counts *base tokens*: chance / environment tokens,
  other players' tokens and the tokens inside the macro all count, padding
  never counts.
- ``R_m = sum_{j=0}^{Delta-1} gamma^j * r_p[s+j]`` (the acting player's reward
  channel only; padding rewards are never read).
- ``delta_m = R_m + gamma^{Delta_m} * c_m * V_p(S_{m+1}) - V_p(S_m)``.
- ``A_m = delta_m + gamma^{Delta_m} * lambda_macro * c_m * A_{m+1}``, expanded
  as ``A_m = sum_l lambda^l * gamma^{sum Delta} * delta_{m+l}`` until the
  player's last macro decision.

The environment discount clock stays on base tokens (``gamma^{Delta_m}``);
``lambda_macro`` decays once per *same-player macro decision*.  The legacy
token trace is reproduced by ``(gamma * lambda_token)^{Delta_m}`` and only
degenerates to the macro clock when ``Delta_m = 1``, ``lambda = 0`` or
``lambda = 1`` -- a macro of length one does **not** imply a same-player
``Delta`` of one (with four rotating players ``Delta`` is usually larger).

``continue_value_target``/``advantage`` are produced under ``torch.no_grad()``
and returned detached: they are training targets, never differentiable paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.training.macro_actions import MacroSegmentation, segment_rollout_macros

MACRO_GAE_SCHEMA = "guandan-marvel-macro-decision-gae-v1"


class MacroGAEError(ValueError):
    """Raised when a macro-decision GAE input violates the training contract."""


@dataclass(frozen=True)
class MacroDecisionGAEOutput:
    schema: str
    macro_lambda: float
    valid: torch.Tensor  # [B, M] bool: complete, on-policy policy macros
    advantage: torch.Tensor  # [B, M] float32, detached
    value_target: torch.Tensor  # [B, M] float32, detached
    delta: torch.Tensor  # [B, M] float32 semi-MDP TD residual
    duration: torch.Tensor  # [B, M] long: base-token Delta_m
    continuation: torch.Tensor  # [B, M] bool: same-player successor exists
    reward_sum: torch.Tensor  # [B, M] float32 window reward
    next_index: torch.Tensor  # [B, M] long: same-player next macro slot or -1
    is_last_for_player: torch.Tensor  # [B, M] bool: terminal/last-for-player


def _field(rollout: TensorDict | Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    value = rollout.get(name) if isinstance(rollout, TensorDict) else rollout.get(name)
    if not isinstance(value, torch.Tensor):
        raise MacroGAEError(f"macro GAE requires rollout field {name!r} as a tensor")
    return value


def require_macro_rollout_ready(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    segmentation: MacroSegmentation,
) -> None:
    """Fail fast on incomplete or player-switch-violating macro trajectories.

    A truncated/incomplete rollout must never be treated as a normal terminal
    zero-bootstrap episode, so both counters are hard gates and every
    trajectory must end on a real terminal step.
    """
    if int(segmentation.incomplete_count) != 0:
        raise MacroGAEError(
            "macro training requires incomplete_count == 0, got "
            f"{segmentation.incomplete_count}"
        )
    if int(segmentation.player_switch_violation_count) != 0:
        raise MacroGAEError(
            "macro training requires player_switch_violation_count == 0, got "
            f"{segmentation.player_switch_violation_count}"
        )
    num_steps = _field(rollout, "num_steps").to(torch.long)
    is_terminal = _field(rollout, "is_terminal").to(torch.bool)
    if is_terminal.shape[:1] != num_steps.shape:
        raise MacroGAEError("rollout is_terminal must align with num_steps")
    final_index = (num_steps - 1).clamp(min=0).reshape(-1, 1)
    final_terminal = is_terminal.reshape(num_steps.numel(), -1).gather(1, final_index).reshape(-1)
    if not bool(final_terminal.all()):
        raise MacroGAEError(
            "macro training rejects truncated/incomplete trajectories: every trajectory "
            "must end on a real terminal step"
        )


def compute_macro_decision_gae(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    state_values: torch.Tensor,
    *,
    segmentation: MacroSegmentation | None = None,
    discount_gamma: float = 1.0,
    macro_lambda: float = 0.95,
) -> MacroDecisionGAEOutput:
    """Compute same-player macro-decision GAE advantages and value targets."""
    gamma = float(discount_gamma)
    lam = float(macro_lambda)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise MacroGAEError("discount_gamma must lie within [0, 1]")
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise MacroGAEError("macro_lambda must lie within [0, 1]")

    num_steps = _field(rollout, "num_steps").to(torch.long)
    player_ids = _field(rollout, "player_ids").to(torch.long)
    rewards = _field(rollout, "rewards")
    if rewards.dim() != 3:
        raise MacroGAEError("rewards must have shape [B, T, P]")
    B, T, P = rewards.shape
    if player_ids.shape != (B, T):
        raise MacroGAEError("player_ids must have shape [B, T]")
    if num_steps.shape != (B,):
        raise MacroGAEError("num_steps must have shape [B]")
    if state_values.shape != (B, T, P):
        raise MacroGAEError("state_values must have shape [B, T, P]")
    if segmentation is None:
        segmentation = segment_rollout_macros(rollout)
    if segmentation.batch_size != B or segmentation.horizon != T:
        raise MacroGAEError("segmentation shape does not match the rollout")
    M = segmentation.max_macros

    with torch.no_grad():
        require_macro_rollout_ready(rollout, segmentation)

        device = rewards.device
        rewards = rewards.detach().to(dtype=torch.float32, device=device)
        values = state_values.detach().to(dtype=torch.float32, device=device)
        top = torch.full((B, M), float(T), dtype=torch.long, device=device)
        start = segmentation.macro_start_index.to(device)
        valid = (start >= 0) & (segmentation.macro_end_index.to(device) >= 0)
        next_index = segmentation.macro_next_index.to(device)
        player = segmentation.macro_player.to(device)

        # Base-token history mask: padding is never read (rewards and values
        # outside num_steps cannot influence any macro).
        time = torch.arange(T, device=device).unsqueeze(0)
        history = time < num_steps.unsqueeze(1)
        history_f = history.to(torch.float32).unsqueeze(-1)
        rewards_masked = rewards * history_f
        values_masked = values * history_f

        # F[b, t, p] = sum_{j=t}^{n-1} gamma^{j-t} r[b, j, p]; F[:, n] = 0.
        discounted = torch.zeros((B, T + 1, P), dtype=torch.float32, device=device)
        for t in range(T - 1, -1, -1):
            carry = gamma * discounted[:, t + 1]
            discounted[:, t] = torch.where(
                history[:, t].unsqueeze(-1), rewards_masked[:, t] + carry, torch.zeros_like(carry),
            )

        flat_batch = torch.arange(B, device=device).unsqueeze(1).expand(B, M)
        safe_start = start.clamp(min=0, max=max(T - 1, 0))
        safe_next = next_index.clamp(min=0, max=max(M - 1, 0))
        safe_player = player.clamp(min=0, max=max(P - 1, 0))
        successor_start = start.gather(1, safe_next)
        has_successor = (next_index >= 0) & valid
        # No successor: the window runs to this trajectory's real num_steps.
        end = torch.where(has_successor, successor_start, num_steps.unsqueeze(1).expand(B, M))
        duration = end - start
        if bool((duration[valid] < 1).any()):
            raise MacroGAEError("macro decision interval must be at least one base token")
        expected_tau = segmentation.macro_tau.to(device)
        if M > 0 and bool(valid.any()):
            mismatch = valid & has_successor & (duration != expected_tau)
            if bool(mismatch.any()):
                raise MacroGAEError(
                    "macro duration must equal the phase-1 macro_tau whenever a same-player "
                    "successor exists"
                )
            order = valid & has_successor & (successor_start <= start)
            if bool(order.any()):
                raise MacroGAEError(
                    "same-player successor must start after the current macro start"
                )

        v_start = values_masked[flat_batch, safe_start, safe_player]
        v_end = values_masked[flat_batch, end.clamp(min=0, max=max(T - 1, 0)), safe_player]
        discounted_start = discounted[flat_batch, safe_start, safe_player]
        discounted_end = discounted[flat_batch, end.clamp(min=0, max=T), safe_player]
        decay_end = torch.pow(
            torch.as_tensor(gamma, dtype=torch.float32, device=device),
            duration.to(torch.float32),
        )
        reward_sum = discounted_start - decay_end * discounted_end
        continuation = has_successor
        bootstrap = torch.where(
            continuation,
            decay_end * v_end,
            torch.zeros_like(v_end),
        )
        delta = reward_sum + bootstrap - v_start

        advantage = torch.zeros((B, M), dtype=torch.float32, device=device)
        decay_tensor = torch.as_tensor(gamma, dtype=torch.float32, device=device)
        lambda_tensor = torch.as_tensor(lam, dtype=torch.float32, device=device)
        for macro in range(M - 1, -1, -1):
            successor_adv = advantage.gather(
                1, next_index[:, macro].clamp(min=0, max=max(M - 1, 0)).unsqueeze(1),
            ).squeeze(1)
            recursion = torch.where(
                continuation[:, macro],
                torch.pow(decay_tensor, duration[:, macro].to(torch.float32))
                * lambda_tensor
                * successor_adv,
                torch.zeros_like(successor_adv),
            )
            advantage[:, macro] = delta[:, macro] + recursion

        if not bool(torch.isfinite(advantage[valid]).all()):
            raise MacroGAEError("macro advantage contains non-finite values")
        if not bool(torch.isfinite(delta[valid]).all()):
            raise MacroGAEError("macro delta contains non-finite values")
        if not bool(torch.isfinite(reward_sum[valid]).all()):
            raise MacroGAEError("macro reward window contains non-finite values")

        advantage = torch.where(valid, advantage, torch.zeros_like(advantage))
        delta = torch.where(valid, delta, torch.zeros_like(delta))
        reward_sum = torch.where(valid, reward_sum, torch.zeros_like(reward_sum))
        duration = torch.where(valid, duration, torch.ones_like(duration))
        continuation = continuation & valid
        value_target = torch.where(valid, v_start + advantage, torch.zeros_like(advantage))

        return MacroDecisionGAEOutput(
            schema=MACRO_GAE_SCHEMA,
            macro_lambda=lam,
            valid=valid,
            advantage=advantage.detach(),
            value_target=value_target.detach(),
            delta=delta.detach(),
            duration=duration.detach(),
            continuation=continuation.detach(),
            reward_sum=reward_sum.detach(),
            next_index=next_index.detach(),
            is_last_for_player=(valid & (next_index < 0)).detach(),
        )


__all__ = [
    "MACRO_GAE_SCHEMA", "MacroDecisionGAEOutput", "MacroGAEError",
    "compute_macro_decision_gae", "require_macro_rollout_ready",
]
