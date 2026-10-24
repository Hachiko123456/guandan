# SPDX-License-Identifier: MIT
"""Behaviour attribution for the phase-6 opponent pool.

The training trajectory keeps one record per acting token that identifies the
policy which *behaved* there.  Trainable current-policy samples and frozen
opponent behaviour are separated by construction:

- ``policy_train_mask`` is True only where ``policy_owner == 0`` (current);
- ``on_policy_valid`` must equal ``policy_train_mask`` on acting tokens, so an
  opponent step can never be smuggled into current PPO;
- the actual behaviour probability stays in the sampler's
  ``behavior_action_prob`` field and is always produced by the policy that
  sampled the action;
- a macro must have exactly one owner and one train mask; anything mixed fails
  closed before any loss math.

Chance/padding positions carry owner ``-1`` and mask ``False`` and never enter
a macro or a loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.training.macro_actions import (
    MacroSegmentation, segment_macro_actions,
)

from .design import (
    CURRENT_POLICY_OWNER_ID, NO_POLICY_OWNER_ID, POOL_FIELD_NAMES,
)

POLICY_OWNER_SCHEMA = "guandan-marvel-phase6-policy-owner-v1"


@dataclass(frozen=True)
class PooledMacroSegmentation:
    """Base phase-1 segmentation plus the phase-6 owner/trainer masks.

    The frozen ``macro_actions`` module is not modified: pooling is layered on
    top as a wrapper so the phase-1 snapshot input stays byte-identical.
    """

    base: MacroSegmentation
    macro_train_mask: torch.Tensor
    macro_policy_owner: torch.Tensor

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def segment_pooled_macros(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    *,
    current_owner_id: int = CURRENT_POLICY_OWNER_ID,
) -> PooledMacroSegmentation:
    """Recover macro boundaries, then enforce the phase-6 ownership contract.

    Opponent tokens remain part of the environment timeline and of macro
    boundaries, but a macro that mixes owners or train masks fails closed and
    only current-owned complete macros become trainable.
    """
    def _field(name: str):
        value = rollout.get(name) if isinstance(rollout, TensorDict) else rollout.get(name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"pooled rollout requires tensor field {name!r}")
        return value

    base = segment_macro_actions(
        num_steps=_field("num_steps"),
        player_ids=_field("player_ids"),
        action_taken=_field("action_taken"),
        avail_is_commit=_field("avail_is_commit"),
        behavior_action_prob=_field("behavior_action_prob"),
        avail_tokens=(
            rollout.get("avail_tokens")
            if isinstance(rollout.get("avail_tokens"), torch.Tensor) else None
        ),
        is_terminal=_field("is_terminal"),
    )
    owner = _field("policy_owner")
    train = _field("policy_train_mask")
    player_ids = _field("player_ids")
    num_steps = _field("num_steps").to(torch.long)
    if tuple(owner.shape) != tuple(player_ids.shape) or tuple(train.shape) != tuple(player_ids.shape):
        raise ValueError("policy_owner/policy_train_mask must have shape [B, T]")
    if train.dtype != torch.bool:
        raise TypeError("policy_train_mask must be bool")
    if owner.dtype not in (torch.int16, torch.int32, torch.int64):
        raise TypeError("policy_owner must be an integer tensor")
    time = torch.arange(player_ids.shape[1], device=player_ids.device).unsqueeze(0)
    history = time < num_steps.unsqueeze(1)
    acting = history & (player_ids >= 0)
    owner_i = owner.to(torch.long)
    train_b = train.to(torch.bool)
    if bool((owner_i[acting] < 0).any()):
        raise ValueError("every acting token must have a non-negative policy owner")
    if bool((owner_i[history & ~acting] != NO_POLICY_OWNER_ID).any()):
        raise ValueError("chance/environment tokens must carry policy_owner=-1")
    if bool((train_b[acting] != (owner_i[acting] == int(current_owner_id))).any()):
        raise ValueError(
            "policy_train_mask must be True exactly on current-owned acting tokens"
        )
    if bool(train_b[~history].any()):
        raise ValueError("padding tokens must carry a False policy_train_mask")

    batch, _length = player_ids.shape
    macros = base.max_macros
    device = player_ids.device
    owner_min = torch.full((batch, macros), 10**9, dtype=torch.long, device=device)
    owner_max = torch.full((batch, macros), -1, dtype=torch.long, device=device)
    train_min = torch.ones((batch, macros), dtype=torch.long, device=device)
    train_max = torch.zeros((batch, macros), dtype=torch.long, device=device)
    if macros > 0:
        valid_token = base.token_to_macro_id >= 0
        safe_ids = torch.where(
            valid_token, base.token_to_macro_id, torch.zeros_like(base.token_to_macro_id),
        )
        owner_min.scatter_reduce_(
            1, safe_ids,
            torch.where(valid_token, owner_i, torch.full_like(owner_i, 10**9)),
            reduce="amin", include_self=True,
        )
        owner_max.scatter_reduce_(
            1, safe_ids,
            torch.where(valid_token, owner_i, torch.full_like(owner_i, -1)),
            reduce="amax", include_self=True,
        )
        train_long = train_b.to(torch.long)
        train_min.scatter_reduce_(
            1, safe_ids,
            torch.where(valid_token, train_long, torch.ones_like(train_long)),
            reduce="amin", include_self=True,
        )
        train_max.scatter_reduce_(
            1, safe_ids,
            torch.where(valid_token, train_long, torch.zeros_like(train_long)),
            reduce="amax", include_self=True,
        )
    valid_macro = base.macro_start_index >= 0
    mixed_owner = valid_macro & (owner_min != owner_max)
    if bool(mixed_owner.any()):
        index = torch.nonzero(mixed_owner)[0]
        raise ValueError(
            "one macro mixes policy owners and must fail closed "
            f"(batch={int(index[0])} macro={int(index[1])} "
            f"owners=[{int(owner_min[index[0], index[1]])},"
            f"{int(owner_max[index[0], index[1]])}])"
        )
    mixed_mask = valid_macro & (train_min == 0) & (train_max == 1)
    if bool(mixed_mask.any()):
        index = torch.nonzero(mixed_mask)[0]
        raise ValueError(
            "one macro mixes policy_train_mask values and must fail closed "
            f"(batch={int(index[0])} macro={int(index[1])})"
        )
    macro_train_mask = (
        (base.macro_start_index >= 0)
        & (base.macro_end_index >= 0)
        & (train_min == 1)
        & (owner_min == int(current_owner_id))
    )
    macro_policy_owner = torch.where(
        base.macro_start_index >= 0,
        owner_min,
        torch.full_like(owner_min, NO_POLICY_OWNER_ID),
    )
    return PooledMacroSegmentation(
        base=base,
        macro_train_mask=macro_train_mask,
        macro_policy_owner=macro_policy_owner,
    )


def pool_field_specs(*, max_game_horizon: int, num_players: int = 4) -> dict[str, object]:
    """Buffer field factories registered only when the pool is enabled.

    The default (pool off) trajectory field set is untouched, so the legacy
    buffer schema, sampler state and loss path stay exactly as before.  The
    factories return full trajectory storage ``[B, T]`` / ``[B, T, P]``; the
    sampler writes one step at a time through ``TrajectoryBuffer.add_step``.
    """
    players = int(num_players)
    horizon = int(max_game_horizon)
    if players <= 0 or horizon <= 0:
        raise ValueError("num_players and max_game_horizon must be positive")
    return {
        "policy_owner": lambda b, device: torch.full(
            (b, horizon), NO_POLICY_OWNER_ID, dtype=torch.long, device=device,
        ),
        "policy_train_mask": lambda b, device: torch.zeros(
            (b, horizon), dtype=torch.bool, device=device,
        ),
        "opponent_checkpoint_id": lambda b, device: torch.full(
            (b, horizon), -1, dtype=torch.long, device=device,
        ),
        "opponent_policy_version": lambda b, device: torch.full(
            (b, horizon), -1, dtype=torch.long, device=device,
        ),
        "assignment_id": lambda b, device: torch.full(
            (b, horizon), -1, dtype=torch.long, device=device,
        ),
        "assigned_seats": lambda b, device: torch.full(
            (b, horizon, players), -1, dtype=torch.long, device=device,
        ),
    }


POOL_STEP_FIELDS: tuple[str, ...] = POOL_FIELD_NAMES


@dataclass(frozen=True)
class PooledTrajectoryStats:
    """Counts used by trainer metrics; never part of a loss."""

    current_token_count: int
    opponent_token_count: int
    current_assignment_count: int
    opponent_owner_ids: tuple[int, ...]
    current_macro_count: int
    opponent_macro_count: int


def _field(rollout: TensorDict | Mapping[str, torch.Tensor], name: str) -> torch.Tensor | None:
    value = rollout.get(name) if isinstance(rollout, TensorDict) else rollout.get(name)
    return value if isinstance(value, torch.Tensor) else None


def validate_pooled_trajectory(
    rollout: TensorDict | Mapping[str, torch.Tensor],
    *,
    num_players: int = 4,
    require_assignment_constant: bool = True,
) -> PooledTrajectoryStats:
    """Fail-closed validation of the pool attribution contract.

    This is intentionally independent of the loss functions: a trainer calls it
    immediately after collection, so a malformed owner/mask combination cannot
    reach PPO.  ``on_policy_valid`` is *not* allowed to claim an opponent step.
    """
    required = set(POOL_FIELD_NAMES) | {
        "num_steps", "player_ids", "behavior_action_prob", "on_policy_valid",
    }
    missing = sorted(required - set(rollout.keys()))
    if missing:
        raise KeyError(f"pooled trajectory is missing required fields: {missing}")
    player_ids = _field(rollout, "player_ids")
    num_steps = _field(rollout, "num_steps")
    owner = _field(rollout, "policy_owner")
    train_mask = _field(rollout, "policy_train_mask")
    checkpoint_id = _field(rollout, "opponent_checkpoint_id")
    policy_version = _field(rollout, "opponent_policy_version")
    assignment_id = _field(rollout, "assignment_id")
    assigned_seats = _field(rollout, "assigned_seats")
    behavior = _field(rollout, "behavior_action_prob")
    on_policy = _field(rollout, "on_policy_valid")
    for name, tensor in (
        ("player_ids", player_ids), ("num_steps", num_steps), ("policy_owner", owner),
        ("policy_train_mask", train_mask), ("opponent_checkpoint_id", checkpoint_id),
        ("opponent_policy_version", policy_version), ("assignment_id", assignment_id),
        ("assigned_seats", assigned_seats), ("behavior_action_prob", behavior),
        ("on_policy_valid", on_policy),
    ):
        if tensor is None:
            raise TypeError(f"pooled trajectory field {name!r} is not a tensor")
    B, T = player_ids.shape
    for name, tensor, shapes in (
        ("num_steps", num_steps, ((B,),)),
        ("policy_owner", owner, ((B, T),)),
        ("policy_train_mask", train_mask, ((B, T),)),
        ("opponent_checkpoint_id", checkpoint_id, ((B, T),)),
        ("opponent_policy_version", policy_version, ((B, T),)),
        ("assignment_id", assignment_id, ((B, T),)),
        ("assigned_seats", assigned_seats, ((B, T, int(num_players)),)),
        ("behavior_action_prob", behavior, ((B, T),)),
        ("on_policy_valid", on_policy, ((B, T),)),
    ):
        if tuple(tensor.shape) != shapes[0]:
            raise ValueError(
                f"pooled trajectory field {name!r} must have shape {shapes[0]}, "
                f"got {tuple(tensor.shape)}"
            )
    if owner.dtype not in (torch.int16, torch.int32, torch.int64):
        raise TypeError("policy_owner must be an integer tensor")
    if train_mask.dtype != torch.bool or on_policy.dtype != torch.bool:
        raise TypeError("policy_train_mask/on_policy_valid must be bool tensors")
    time = torch.arange(T, device=player_ids.device).unsqueeze(0)
    history = time < num_steps.unsqueeze(1)
    acting = history & (player_ids >= 0)
    chance = history & (player_ids < 0)

    owner_i = owner.to(torch.long)
    mask_b = train_mask.to(torch.bool)
    on_policy_b = on_policy.to(torch.bool)
    if bool((owner_i[acting] < 0).any()):
        raise ValueError("an acting token must have a non-negative policy owner")
    if bool((mask_b[acting] != (owner_i[acting] == CURRENT_POLICY_OWNER_ID)).any()):
        raise ValueError(
            "policy_train_mask must be True exactly on current-owned acting tokens "
            "(an opponent token cannot be a trainable current sample)"
        )
    if bool((on_policy_b[acting] != mask_b[acting]).any()):
        raise ValueError(
            "on_policy_valid must equal policy_train_mask on acting tokens; "
            "frozen-opponent behaviour is never a current on-policy sample"
        )
    if bool((owner_i[chance] != NO_POLICY_OWNER_ID).any()) or bool(mask_b[chance].any()):
        raise ValueError("chance/environment steps must carry owner -1 and a False train mask")
    opponent = acting & (owner_i != CURRENT_POLICY_OWNER_ID)
    if bool((checkpoint_id.to(torch.long)[opponent] < 0).any()):
        raise ValueError("a frozen-opponent token must record its checkpoint id")
    if bool((policy_version.to(torch.long)[opponent] < 0).any()):
        raise ValueError("a frozen-opponent token must record its policy version")
    if bool((checkpoint_id.to(torch.long)[acting & (owner_i == CURRENT_POLICY_OWNER_ID)] != -1).any()):
        raise ValueError("a current-owned token must not carry an opponent checkpoint id")
    if bool((assignment_id.to(torch.long)[acting] < 0).any()):
        raise ValueError("every acting token must carry an assignment id")
    if not bool(torch.isfinite(behavior[acting]).all()) or bool((behavior[acting] <= 0).any()):
        raise ValueError("behavior_action_prob must be finite and positive on every acting token")
    seats = assigned_seats.to(torch.long)
    acting_player = player_ids.to(torch.long).clamp(min=0)
    owner_at_seat = seats.gather(2, acting_player.unsqueeze(-1)).squeeze(-1)
    if bool((owner_at_seat[acting] != owner_i[acting]).any()):
        raise ValueError("assigned_seats disagrees with the acting token's policy_owner")
    if bool((seats < NO_POLICY_OWNER_ID).any()):
        raise ValueError("assigned_seats contains an invalid owner id")

    if require_assignment_constant and B > 0:
        a = assignment_id.to(torch.long)
        for b in range(B):
            values = a[b][acting[b]]
            if values.numel() == 0:
                raise ValueError("a pooled trajectory must contain at least one acting token")
            if int(values.min().item()) != int(values.max().item()):
                raise ValueError(
                    "assignment_id may not change inside one episode; an opponent may not be "
                    "hot-switched without an explicit episode boundary"
                )
            # B indexes independent completed episodes. Different rows may
            # legitimately rotate teams or draw different frozen opponents;
            # only a change *within this episode* violates the routing contract.
            seat_rows = seats[b][acting[b]]
            first = seat_rows[0]
            if bool((seat_rows != first).any()):
                raise ValueError("assigned_seats may not change inside one episode")

    current_tokens = int((acting & mask_b).sum().item())
    opponent_tokens = int(opponent.sum().item())
    assignments = sorted({
        int(value) for value in assignment_id.to(torch.long)[acting].tolist()
    })
    owners = tuple(sorted({
        int(value) for value in owner_i[opponent].tolist()
    }))
    return PooledTrajectoryStats(
        current_token_count=current_tokens,
        opponent_token_count=opponent_tokens,
        current_assignment_count=len(assignments),
        opponent_owner_ids=owners,
        current_macro_count=0,
        opponent_macro_count=0,
    )


def macro_owner_counts(
    segmentation: MacroSegmentation,
    *,
    current_owner_id: int = CURRENT_POLICY_OWNER_ID,
) -> tuple[int, int]:
    """Count current-owned and opponent-owned complete macros."""
    trainable = segmentation.macro_train_mask
    valid = segmentation.macro_valid & (segmentation.macro_end_index >= 0)
    if trainable is None:
        return int(valid.sum().item()), 0
    current = int((valid & trainable.to(torch.bool)).sum().item())
    owner = getattr(segmentation, "macro_policy_owner", None)
    if owner is None:
        return current, int(valid.sum().item()) - current
    opponents = int(
        (valid & (~trainable.to(torch.bool)) & (owner.to(torch.long) >= 0)).sum().item()
    )
    return current, opponents


def validate_macro_owners(
    segmentation: MacroSegmentation,
    *,
    current_owner_id: int = CURRENT_POLICY_OWNER_ID,
) -> None:
    """Cross-check the segmentation's own owner arrays for mixed macros.

    ``segment_macro_actions`` already fails closed while building the mask;
    this second, independent check is used by tests and by the trainer before a
    loss so a hand-built ``MacroSegmentation`` cannot bypass the contract.
    """
    owner = getattr(segmentation, "macro_policy_owner", None)
    train = getattr(segmentation, "macro_train_mask", None)
    if owner is None or train is None:
        return
    if tuple(owner.shape) != tuple(segmentation.macro_valid.shape):
        raise ValueError("macro_policy_owner shape does not match macro slots")
    if tuple(train.shape) != tuple(segmentation.macro_valid.shape):
        raise ValueError("macro_train_mask shape does not match macro slots")
    valid = segmentation.macro_valid & (segmentation.macro_end_index >= 0)
    owner_i = owner.to(torch.long)
    if bool((owner_i[valid] < NO_POLICY_OWNER_ID).any()):
        raise ValueError("macro_policy_owner contains an invalid owner id")
    if bool((train.to(torch.bool)[valid] != (owner_i[valid] == int(current_owner_id))).any()):
        raise ValueError("macro_train_mask must be True exactly on current-owned macros")


__all__ = [
    "POOL_STEP_FIELDS", "POLICY_OWNER_SCHEMA", "PooledMacroSegmentation",
    "PooledTrajectoryStats", "macro_owner_counts", "pool_field_specs",
    "segment_pooled_macros", "validate_macro_owners", "validate_pooled_trajectory",
]
