# SPDX-License-Identifier: MIT
"""Phase-1 macro-action segmentation and diagnostics for MARVEL training.

One environment step is one protocol token (FAMILY / RANK / SUIT / LENGTH /
CARD / WILD_ASSIGNMENT / COMMIT / PASS).  A *macro action* is the contiguous
token segment in which one acting player starts from an empty action prefix
and ends by selecting COMMIT or PASS.

This module recovers macro actions from recorded rollouts without changing the
legacy token-level training path.  It is diagnostic-only in phase 1: the
trainer does not call it, and its outputs never enter a loss.

Definitions used here
---------------------
- Only steps with ``player_ids >= 0`` (policy steps) can belong to a macro.
  Trainer-chance (``-1``) and environment-controlled prefix (``-2``) steps are
  never part of a policy macro.
- Every array index derived from a player id first goes through ``safe_player``
  (active tokens keep their real non-negative id, everything else becomes 0).
  Positive garbage ids on inactive/chance/padding positions (``3``/``99``/
  ``999``) therefore can never overflow the ``(player, ordinal)`` lookup, and
  padding cannot influence ``num_players``, ``macro_player`` or the
  next-player check.  Active illegal player ids are still not clamped away.
- ``chosen_is_commit[b, t] = avail_is_commit[b, t, action_taken[b, t]]``.  A
  macro ends at its commit token; PASS is a length-1 commit macro.
- A macro starts at the first token of a player's prefix or at the first token
  of a player's next action taken after their previous commit.
- ``tau`` is the global token distance between the start of one macro and the
  start of the *same player's* next macro.  It includes any chance or other
  players' tokens in between and is therefore not the macro length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from guandan_marvel.model.tensordict import TensorDict

MACRO_SEGMENTATION_SCHEMA = "guandan-marvel-macro-segmentation-v1"

# GD-ACTION-0.2-MARVEL fixed row layout shared by both backends
# (rows are [ROW_BEGIN, PHASE, PREFIX_STAGE, KIND, PAYLOAD_MARKER, V1_TOKEN_ID,
#            IS_COMMIT, ROW_END]).
ACTION_ROW_BEGIN = 262
ACTION_KIND_BASE = 296
ACTION_PAYLOAD = 304
ACTION_ROW_END = 307
ACTION_KIND_NAMES = ("PASS", "COMMIT", "FAMILY", "RANK", "SUIT", "LENGTH", "CARD", "WILD_ASSIGNMENT")
KIND_PASS = ACTION_KIND_NAMES.index("PASS")
KIND_COMMIT = ACTION_KIND_NAMES.index("COMMIT")

COMMIT_KIND_NONE = 0
COMMIT_KIND_PASS = 1
COMMIT_KIND_COMMIT = 2

PLAYER_CHANCE_TRAINER = -1
PLAYER_CHANCE_ENVIRONMENT = -2


def _as_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"macros: {name} must be a torch.Tensor")
    return value


@dataclass(frozen=True)
class MacroSegmentation:
    """Macro boundaries recovered from one batch of complete trajectories.

    Tensors are shaped ``[B, T]`` (tokens) or ``[B, M]`` (macros), where ``M``
    is the maximum macro count per trajectory in the batch.
    """

    schema: str
    batch_size: int
    horizon: int
    max_macros: int
    num_players: int
    num_macros: torch.Tensor  # [B] long
    token_to_macro_id: torch.Tensor  # [B, T] long, -1 outside macros
    chosen_is_commit: torch.Tensor  # [B, T] bool
    macro_start_index: torch.Tensor  # [B, M] long, -1 for padding
    macro_end_index: torch.Tensor  # [B, M] long, -1 for incomplete macros
    macro_last_index: torch.Tensor  # [B, M] long, last token of the macro
    macro_length: torch.Tensor  # [B, M] long
    macro_player: torch.Tensor  # [B, M] long
    macro_valid: torch.Tensor  # [B, M] bool, False for padding macro slots
    macro_is_terminal: torch.Tensor  # [B, M] bool
    macro_is_last_for_player: torch.Tensor  # [B, M] bool
    macro_next_index: torch.Tensor  # [B, M] long, same player's next macro or -1
    macro_tau: torch.Tensor  # [B, M] long, same-player start distance or 0
    macro_start_mask: torch.Tensor  # [B, T] bool
    macro_token_mask: torch.Tensor  # [B, T] bool
    macro_commit_kind: torch.Tensor  # [B, M] long
    macro_old_logprob: torch.Tensor | None  # [B, M] float64
    incomplete_count: int
    player_switch_violation_count: int


def _safe_action_index(action_taken: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Zero out non-acting positions before any gather.

    Inactive, chance and padding positions may carry arbitrary garbage in
    ``action_taken``; they must never reach a gather.  Active positions are
    validated separately by the callers.
    """
    return torch.where(active, action_taken, torch.zeros_like(action_taken))


def _selected_commit_flags(
    action_taken: torch.Tensor,
    avail_is_commit: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    if avail_is_commit.dim() != 3:
        raise ValueError("macros: avail_is_commit must have shape [B, T, A]")
    if action_taken.dim() != 2:
        raise ValueError("macros: action_taken must have shape [B, T]")
    if action_taken.shape != active.shape:
        raise ValueError("macros: action_taken must have shape [B, T]")
    if avail_is_commit.shape[:2] != action_taken.shape:
        raise ValueError("macros: avail_is_commit leading dims must match action_taken")
    action_count = avail_is_commit.shape[-1]
    if bool((action_taken[active] < 0).any()) or bool((action_taken[active] >= action_count).any()):
        raise ValueError("macros: action_taken is outside the available-action range on active steps")
    safe_action = _safe_action_index(action_taken, active)
    gathered = avail_is_commit.to(torch.bool).gather(
        2, safe_action.unsqueeze(-1)
    ).squeeze(-1)
    return gathered & active


def _decode_selected_kinds(
    avail_tokens: torch.Tensor,
    action_taken: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    if avail_tokens.dim() != 4:
        raise ValueError("macros: avail_tokens must have shape [B, T, A, L]")
    if action_taken.dim() != 2:
        raise ValueError("macros: action_taken must have shape [B, T]")
    if avail_tokens.shape[:2] != action_taken.shape:
        raise ValueError("macros: avail_tokens leading dims must match action_taken")
    row_length = avail_tokens.shape[-1]
    if row_length < 8:
        raise ValueError("macros: avail_tokens rows must expose the 8-slot GD-ACTION-0.2 layout")
    action_count = avail_tokens.shape[2]
    if bool((action_taken[active] < 0).any()) or bool((action_taken[active] >= action_count).any()):
        raise ValueError("macros: action_taken is outside the available-action range on active steps")
    safe_action = _safe_action_index(action_taken, active)
    index = safe_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, row_length)
    rows = avail_tokens.gather(2, index).squeeze(2)
    kind = rows[..., 3] - ACTION_KIND_BASE
    framing_ok = (
        (rows[..., 0] == ACTION_ROW_BEGIN)
        & (rows[..., 4] == ACTION_PAYLOAD)
        & (rows[..., 7] == ACTION_ROW_END)
        & (kind >= 0)
        & (kind < len(ACTION_KIND_NAMES))
    )
    if bool(active.any()) and not bool(framing_ok[active].all()):
        raise ValueError("macros: selected action row does not match the GD-ACTION-0.2 layout")
    return torch.where(active, kind, torch.full_like(kind, -1))


def segment_macro_actions(
    *,
    num_steps: torch.Tensor,
    player_ids: torch.Tensor,
    action_taken: torch.Tensor,
    avail_is_commit: torch.Tensor,
    behavior_action_prob: torch.Tensor | None = None,
    avail_tokens: torch.Tensor | None = None,
    is_terminal: torch.Tensor | None = None,
) -> MacroSegmentation:
    """Recover macro actions and their boundaries from recorded token steps.

    ``num_steps`` is ``[B]``; ``player_ids``/``action_taken`` are ``[B, T]``;
    ``avail_is_commit`` is ``[B, T, A]``.  Optional ``behavior_action_prob``
    (``[B, T]``) yields the macro old log-probability sum; optional
    ``avail_tokens`` (``[B, T, A, L]``) yields strict PASS/COMMIT decoding;
    optional ``is_terminal`` (``[B, T]``) marks the terminal macro.
    """
    num_steps = _as_tensor(num_steps, name="num_steps")
    player_ids = _as_tensor(player_ids, name="player_ids")
    action_taken = _as_tensor(action_taken, name="action_taken")
    avail_is_commit = _as_tensor(avail_is_commit, name="avail_is_commit")

    if player_ids.dim() != 2:
        raise ValueError("macros: player_ids must have shape [B, T]")
    B, T = player_ids.shape
    if num_steps.shape != (B,):
        raise ValueError("macros: num_steps must have shape [B]")
    if action_taken.shape != (B, T):
        raise ValueError("macros: action_taken must have shape [B, T]")
    if avail_is_commit.shape[:2] != (B, T):
        raise ValueError("macros: avail_is_commit must have shape [B, T, A]")
    device = player_ids.device
    time = torch.arange(T, device=device).unsqueeze(0)
    history = time < num_steps.unsqueeze(1)
    active = history & (player_ids >= 0)
    if bool((num_steps < 0).any()) or bool((num_steps > T).any()):
        raise ValueError("macros: num_steps must be within 0..horizon")

    if behavior_action_prob is not None:
        behavior_action_prob = _as_tensor(behavior_action_prob, name="behavior_action_prob")
        if behavior_action_prob.shape != (B, T):
            raise ValueError("macros: behavior_action_prob must have shape [B, T]")
        if bool((behavior_action_prob[active] <= 0).any()) or not bool(
            torch.isfinite(behavior_action_prob[active]).all()
        ):
            raise ValueError("macros: behavior_action_prob must be finite and positive on active steps")
    if is_terminal is not None:
        is_terminal = _as_tensor(is_terminal, name="is_terminal")
        if is_terminal.shape != (B, T):
            raise ValueError("macros: is_terminal must have shape [B, T]")

    chosen_is_commit = _selected_commit_flags(action_taken, avail_is_commit, active)
    if avail_tokens is not None:
        avail_tokens = _as_tensor(avail_tokens, name="avail_tokens")
        kinds = _decode_selected_kinds(avail_tokens, action_taken, active)
        decoded_commit = (kinds == KIND_PASS) | (kinds == KIND_COMMIT)
        if not bool((decoded_commit[active] == chosen_is_commit[active]).all()):
            raise ValueError(
                "macros: avail_is_commit disagrees with the decoded GD-ACTION-0.2 kind"
            )
    else:
        kinds = None

    # Any use of player ids as an index or as a (player, ordinal) key must be
    # safe on inactive/chance/padding positions: they keep arbitrary garbage
    # ids (e.g. 3/99/999) whose contract is to be *ignored*.  ``safe_player``
    # preserves the real non-negative id on active steps and forces 0
    # everywhere else; it never masks away an illegal *active* id.
    safe_player = torch.where(active, player_ids.clamp(min=0), torch.zeros_like(player_ids))
    num_players = int(safe_player[active].max().item()) + 1 if bool(active.any()) else 0
    num_players = max(num_players, 0)

    commit = active & chosen_is_commit
    macro_start = torch.zeros((B, T), dtype=torch.bool, device=device)
    ordinal_by_token = torch.zeros((B, T), dtype=torch.long, device=device)
    for player in range(num_players):
        mask = active & (safe_player == player)
        commit_row = commit & mask
        commits_before_or_at = commit_row.to(torch.long).cumsum(dim=1)
        ordinal = commits_before_or_at - commit_row.to(torch.long)
        position = torch.where(
            mask, time.expand(B, T), torch.full((B, T), -1, dtype=torch.long, device=device)
        )
        last_inclusive = torch.cummax(position, dim=1).values
        previous = last_inclusive.roll(1, dims=1)
        previous[:, 0] = -1
        previous_ordinal = ordinal.gather(1, previous.clamp(min=0))
        starts_here = mask & ((previous < 0) | (previous_ordinal != ordinal))
        macro_start = macro_start | starts_here
        ordinal_by_token = torch.where(mask, ordinal, ordinal_by_token)

    # Macro ids are ordered by start index, but a macro owns exactly the tokens
    # of one (player, ordinal) pair.  Assigning ids by a global cumsum over
    # start positions would collide when a player's macro interleaves with
    # another player's macro start (which happens exactly when the
    # player-switch invariant is violated); map each (player, ordinal) to its
    # own start-ordered id instead.
    num_macros = macro_start.sum(dim=1)
    max_macros = int(num_macros.max().item()) if B > 0 else 0
    if max_macros > 0 and bool(active.any()):
        max_ordinal = int(ordinal_by_token[active].max().item()) + 1
        lookup = torch.full(
            (B, num_players, max_ordinal), -1, dtype=torch.long, device=device
        )
        start_rows_pre, start_cols_pre = macro_start.nonzero(as_tuple=True)
        batch_base_pre = num_macros.cumsum(dim=0) - num_macros
        start_ids_pre = torch.arange(start_rows_pre.numel(), device=device) - batch_base_pre[start_rows_pre]
        lookup[
            start_rows_pre,
            safe_player[start_rows_pre, start_cols_pre],
            ordinal_by_token[start_rows_pre, start_cols_pre],
        ] = start_ids_pre
        flat_batch = torch.arange(B, device=device).unsqueeze(1)
        mapped = lookup[
            flat_batch.expand(B, T),
            safe_player,
            ordinal_by_token.clamp(max=max_ordinal - 1),
        ]
        token_to_macro_id = torch.where(
            active, mapped, torch.full_like(mapped, -1)
        )
    else:
        token_to_macro_id = torch.full((B, T), -1, dtype=torch.long, device=device)

    macro_start_index = torch.full((B, max_macros), -1, dtype=torch.long, device=device)
    macro_end_index = torch.full((B, max_macros), -1, dtype=torch.long, device=device)
    macro_last_index = torch.full((B, max_macros), -1, dtype=torch.long, device=device)
    macro_length = torch.zeros((B, max_macros), dtype=torch.long, device=device)
    macro_player = torch.full((B, max_macros), -1, dtype=torch.long, device=device)

    if max_macros > 0:
        start_rows, start_cols = macro_start.nonzero(as_tuple=True)
        batch_base = num_macros.cumsum(dim=0) - num_macros
        start_ids = torch.arange(start_rows.numel(), device=device) - batch_base[start_rows]
        macro_start_index[start_rows, start_ids] = start_cols
        macro_player[start_rows, start_ids] = safe_player[start_rows, start_cols]

        end_rows, end_cols = commit.nonzero(as_tuple=True)
        end_ids = token_to_macro_id[end_rows, end_cols]
        macro_end_index[end_rows, end_ids] = end_cols

        positions = time.expand(B, T)
        valid_macro = token_to_macro_id >= 0
        scatter_index = torch.where(
            valid_macro, token_to_macro_id, torch.zeros_like(token_to_macro_id)
        )
        scatter_source = torch.where(
            valid_macro, positions, torch.full_like(positions, -1)
        )
        macro_last_index.scatter_reduce_(
            1, scatter_index, scatter_source, reduce="amax", include_self=True
        )
        macro_length.scatter_add_(
            1, scatter_index, valid_macro.to(torch.long)
        )
        macro_last_index = torch.where(
            torch.arange(max_macros, device=device).unsqueeze(0) < num_macros.unsqueeze(1),
            macro_last_index,
            torch.full_like(macro_last_index, -1),
        )
        macro_length = torch.where(
            torch.arange(max_macros, device=device).unsqueeze(0) < num_macros.unsqueeze(1),
            macro_length,
            torch.zeros_like(macro_length),
        )

    macro_next_index = torch.full((B, max_macros), -1, dtype=torch.long, device=device)
    if max_macros > 0:
        macro_ids = torch.arange(max_macros, device=device).unsqueeze(0).expand(B, max_macros)
        for player in range(num_players):
            is_player = macro_player == player
            sequence_position = is_player.to(torch.long).cumsum(dim=1) - 1
            lookup = torch.full((B, max_macros + 1), -1, dtype=torch.long, device=device)
            lookup.scatter_(
                1,
                torch.where(is_player, sequence_position, torch.full_like(sequence_position, max_macros)),
                torch.where(is_player, macro_ids, torch.full_like(macro_ids, -1)),
            )
            successor = lookup.gather(1, (sequence_position + 1).clamp(max=max_macros))
            macro_next_index = torch.where(is_player, successor, macro_next_index)

    macro_valid = macro_start_index >= 0
    next_start = macro_start_index.gather(1, macro_next_index.clamp(min=0))
    macro_tau = torch.where(
        macro_valid & (macro_next_index >= 0),
        next_start - macro_start_index,
        torch.zeros_like(macro_start_index),
    )
    macro_is_last_for_player = macro_valid & (macro_next_index < 0)

    if is_terminal is not None:
        end_terminal = is_terminal.to(torch.bool).gather(1, macro_end_index.clamp(min=0))
        macro_is_terminal = (macro_end_index >= 0) & end_terminal
    else:
        macro_is_terminal = (macro_end_index >= 0) & (
            macro_end_index == (num_steps - 1).unsqueeze(1)
        )
    macro_is_terminal = macro_is_terminal & macro_valid

    if kinds is not None:
        commit_kind = torch.zeros((B, max_macros), dtype=torch.long, device=device)
        if max_macros > 0:
            end_rows, end_cols = commit.nonzero(as_tuple=True)
            end_ids = token_to_macro_id[end_rows, end_cols]
            kind_values = kinds[end_rows, end_cols]
            decoded = torch.where(
                kind_values == KIND_PASS,
                torch.full_like(kind_values, COMMIT_KIND_PASS),
                torch.full_like(kind_values, COMMIT_KIND_COMMIT),
            )
            commit_kind[end_rows, end_ids] = decoded
    else:
        commit_kind = torch.where(
            macro_end_index >= 0,
            torch.where(
                macro_length == 1,
                torch.full_like(macro_length, COMMIT_KIND_PASS),
                torch.full_like(macro_length, COMMIT_KIND_COMMIT),
            ),
            torch.zeros_like(macro_length),
        )

    macro_old_logprob: torch.Tensor | None = None
    if behavior_action_prob is not None:
        log_prob = behavior_action_prob.to(torch.float64).log()
        log_prob = torch.where(active, log_prob, torch.zeros_like(log_prob))
        prefix = log_prob.cumsum(dim=1)
        prefix_before = prefix - log_prob
        flat_index = macro_last_index.clamp(min=0)
        end_prefix = prefix.gather(1, flat_index)
        start_prefix_before = prefix_before.gather(1, macro_start_index.clamp(min=0))
        macro_old_logprob = torch.where(
            (macro_start_index >= 0) & (macro_last_index >= 0),
            end_prefix - start_prefix_before,
            torch.zeros_like(end_prefix),
        )

    incomplete_count = int(((macro_start_index >= 0) & (macro_end_index < 0)).sum().item())

    next_active = torch.full((B, T), T, dtype=torch.long, device=device)
    if T > 1:
        following_position = torch.where(
            active, torch.arange(T, device=device).expand(B, T), torch.full((B, T), T, device=device)
        )
        reversed_min = torch.flip(
            torch.cummin(torch.flip(following_position, dims=[1]), dim=1).values, dims=[1]
        )
        next_active = reversed_min.roll(-1, dims=1)
        next_active[:, -1] = T
    has_next = next_active < T
    next_player = safe_player.gather(1, next_active.clamp(min=0, max=max(T - 1, 0)))
    player_switch_violation_count = int(
        (
            active
            & has_next
            & (next_player != safe_player)
            & (~chosen_is_commit)
        ).sum().item()
    )

    return MacroSegmentation(
        schema=MACRO_SEGMENTATION_SCHEMA,
        batch_size=int(B),
        horizon=int(T),
        max_macros=max_macros,
        num_players=num_players,
        num_macros=num_macros,
        token_to_macro_id=token_to_macro_id,
        chosen_is_commit=chosen_is_commit,
        macro_start_index=macro_start_index,
        macro_end_index=macro_end_index,
        macro_last_index=macro_last_index,
        macro_length=macro_length,
        macro_player=macro_player,
        macro_valid=macro_valid,
        macro_is_terminal=macro_is_terminal,
        macro_is_last_for_player=macro_is_last_for_player,
        macro_next_index=macro_next_index,
        macro_tau=macro_tau,
        macro_start_mask=macro_start,
        macro_token_mask=token_to_macro_id >= 0,
        macro_commit_kind=commit_kind,
        macro_old_logprob=macro_old_logprob,
        incomplete_count=incomplete_count,
        player_switch_violation_count=player_switch_violation_count,
    )


def segment_rollout_macros(rollout: TensorDict | Mapping[str, torch.Tensor]) -> MacroSegmentation:
    """Segment macro actions from a rollout TensorDict or mapping of tensors."""

    def _field(name: str):
        if isinstance(rollout, TensorDict):
            return rollout.get(name)
        return rollout.get(name) if name in rollout else None

    return segment_macro_actions(
        num_steps=_field("num_steps"),
        player_ids=_field("player_ids"),
        action_taken=_field("action_taken"),
        avail_is_commit=_field("avail_is_commit"),
        behavior_action_prob=_field("behavior_action_prob"),
        avail_tokens=_field("avail_tokens"),
        is_terminal=_field("is_terminal"),
    )


def _quantile(values: torch.Tensor, q: float) -> float | None:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.float(), torch.tensor(q, device=values.device)).item())


def macro_diagnostics(
    segmentation: MacroSegmentation,
    *,
    engine_committed_steps: int | None = None,
    prefix: str = "macro/",
) -> dict[str, float | int | None]:
    """Diagnostic macro metrics.  These never participate in any loss."""
    start = segmentation.macro_start_index
    player = segmentation.macro_player
    valid = start >= 0
    count = int(valid.sum().item())

    metrics: dict[str, float | int | None] = {
        f"{prefix}schema": segmentation.schema,
        f"{prefix}count": count,
        f"{prefix}count_mean_per_trajectory": (
            float(segmentation.num_macros.float().mean()) if segmentation.batch_size else None
        ),
        f"{prefix}committed_count": int((segmentation.macro_end_index >= 0).sum().item()),
    }
    for index in range(segmentation.num_players):
        metrics[f"{prefix}count_by_player/{index}"] = int((valid & (player == index)).sum().item())

    lengths = segmentation.macro_length[valid]
    metrics.update({
        f"{prefix}length_mean": float(lengths.float().mean().item()) if lengths.numel() else None,
        f"{prefix}length_p50": _quantile(lengths, 0.50),
        f"{prefix}length_p95": _quantile(lengths, 0.95),
        f"{prefix}length_max": int(lengths.max().item()) if lengths.numel() else None,
    })

    tau_mask = valid & (segmentation.macro_next_index >= 0)
    taus = segmentation.macro_tau[tau_mask]
    metrics.update({
        f"{prefix}tau_mean": float(taus.float().mean().item()) if taus.numel() else None,
        f"{prefix}tau_p50": _quantile(taus, 0.50),
        f"{prefix}tau_p95": _quantile(taus, 0.95),
        f"{prefix}tau_max": int(taus.max().item()) if taus.numel() else None,
    })

    kinds = segmentation.macro_commit_kind
    metrics.update({
        f"{prefix}commit_count": int(((kinds == COMMIT_KIND_COMMIT) & valid).sum().item()),
        f"{prefix}pass_count": int(((kinds == COMMIT_KIND_PASS) & valid).sum().item()),
        f"{prefix}incomplete_count": segmentation.incomplete_count,
        f"{prefix}player_switch_violation_count": segmentation.player_switch_violation_count,
    })
    if segmentation.macro_old_logprob is not None:
        logprobs = segmentation.macro_old_logprob[valid]
        metrics[f"{prefix}old_logprob_mean"] = (
            float(logprobs.mean().item()) if logprobs.numel() else None
        )
    else:
        metrics[f"{prefix}old_logprob_mean"] = None

    if engine_committed_steps is not None:
        metrics[f"{prefix}engine_committed_steps"] = int(engine_committed_steps)
        metrics[f"{prefix}commit_step_match"] = int(
            int((segmentation.macro_end_index >= 0).sum().item()) == int(engine_committed_steps)
        )
    return metrics


__all__ = [
    "ACTION_KIND_BASE", "ACTION_KIND_NAMES", "ACTION_PAYLOAD", "ACTION_ROW_BEGIN",
    "ACTION_ROW_END", "COMMIT_KIND_COMMIT", "COMMIT_KIND_NONE", "COMMIT_KIND_PASS",
    "KIND_COMMIT", "KIND_PASS", "MACRO_SEGMENTATION_SCHEMA", "MacroSegmentation",
    "PLAYER_CHANCE_ENVIRONMENT", "PLAYER_CHANCE_TRAINER",
    "macro_diagnostics", "segment_macro_actions", "segment_rollout_macros",
]
