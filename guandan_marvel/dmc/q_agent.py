# SPDX-License-Identifier: MIT
"""DMC candidate-level Q network (stage 3).

``DMCAgent`` wraps the existing MARVEL trunk (``guandan_marvel.model.agent.Agent``)
without modifying it.  The wrapper reuses the accepted candidate-scoring path:

.. code-block:: text

    history tokens -> state_tokens_embed + decoder -> hist [B, D]
    hist + state channels -> state_mlp                 -> s    [B, F]
    GD-DMC-ACTION-1 64-d -> action_encoder             -> a    [B, K, F]
    fused = s.unsqueeze(1) * a  (+ ActorFuse residual for v1.4)
    Q = logit_head(fused).squeeze(-1)                  -> Q    [B, K]

Only one additive module is introduced, ``action_encoder`` (a 64->F input
projection followed by the same ResGatedMLP stack the token action path uses),
because the pre-computed ``GD-DMC-ACTION-1`` features are not token rows.  No
file under ``model/`` is edited.

The wrapper is *freshly initialised*; it never loads weights in ``__init__``.
``load_state_dict`` accepts only a payload whose ``schema`` is
``dmc-v1``.  An IPPO/VRPO policy checkpoint is rejected with a pointer to the
explicit migration entry point (``guandan_marvel.training.policy_import``).
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn

from guandan_marvel.config import TrainConfig
from guandan_marvel.dmc import (
    DMC_ACTION_FEATURE_VERSION,
    DMC_CHECKPOINT_SCHEMA,
)
from guandan_marvel.model import Agent

#: Versioned dimension of ``GD-DMC-ACTION-1`` (stage 2 lock).
ACTION_FEATURE_DIM = 64

#: Sentinel written into invalid ``[B, K]`` candidate slots.  It is *not* a Q
#: value: the learner gathers the chosen (always valid) column and never reduces
#: over this axis.  A finite ``finfo.min`` is used instead of ``-inf`` so an
#: accidental mean/sum stays finite instead of producing NaN.
INVALID_Q_SENTINEL = float(torch.finfo(torch.float32).min)


class DMCCheckpointError(ValueError):
    """Raised when a checkpoint is not a ``dmc-v1`` policy payload."""


class DMCAgent(nn.Module):
    """Fresh candidate-level Q network for DMC.

    Parameters
    ----------
    cfg:
        The resolved training configuration.  ``cfg.game`` supplies the token
        vocabulary, channel width, player count and horizon; ``cfg.model``
        supplies the trunk dimensions.
    enable_ntp_head:
        Build the auxiliary next-token-prediction head.  Defaults to ``False``.
    ntp_weight:
        Auxiliary NTP loss weight.  The locked default is ``0.0``; a positive
        weight requires ``enable_ntp_head=True`` (validated here) and an NTP
        target supplied by the caller of the learner.
    action_feature_dim:
        Width of the versioned action feature vector.  Locked at 64.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        enable_ntp_head: bool = False,
        ntp_weight: float = 0.0,
        action_feature_dim: int = ACTION_FEATURE_DIM,
    ) -> None:
        super().__init__()
        if not isinstance(cfg, TrainConfig):
            raise TypeError("cfg must be a TrainConfig")
        action_feature_dim = int(action_feature_dim)
        if action_feature_dim != ACTION_FEATURE_DIM:
            raise ValueError(
                f"action_feature_dim must be {ACTION_FEATURE_DIM} for "
                f"{DMC_ACTION_FEATURE_VERSION}; got {action_feature_dim}"
            )
        weight = float(ntp_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("ntp_weight must be finite and non-negative")
        if weight > 0.0 and not enable_ntp_head:
            raise ValueError("ntp_weight > 0 requires enable_ntp_head=True")
        if not isinstance(enable_ntp_head, bool):
            raise TypeError("enable_ntp_head must be bool")

        self.cfg = cfg
        self.action_feature_dim = action_feature_dim
        self.ntp_weight = weight
        # The trunk keeps its normal fresh initialisation; no historical policy
        # is imported here.  The value head is not constructed: DMC has no
        # state-value critic.
        self.backbone = Agent(cfg, enable_value_head=False, enable_ntp_head=enable_ntp_head)

        feature_dim = int(self.backbone.feature_dim)
        # Additive action path for pre-computed candidate features.  It mirrors
        # ``BaseMarvelModule``'s ``action_mlp`` (input projection + ResGatedMLP).
        self.action_input_proj = Agent._build_input_proj(action_feature_dim, feature_dim)
        self.action_trunk = Agent._build_res_gated_mlp(
            feature_dim,
            n_layers=int(self.backbone.action_layers),
            expansion_ratio=int(self.backbone.resmlp_expansion_ratio),
        )

    # ------------------------------------------------------------------
    # Convenience properties (read-only views of the wrapped trunk)
    # ------------------------------------------------------------------
    @property
    def num_channels(self) -> int:
        return int(self.backbone.num_channels)

    @property
    def num_tokens(self) -> int:
        return int(self.backbone.num_tokens)

    @property
    def num_players(self) -> int:
        return int(self.backbone.num_players)

    @property
    def feature_dim(self) -> int:
        return int(self.backbone.feature_dim)

    @property
    def enable_ntp(self) -> bool:
        return self.backbone.ntp_head is not None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _action_embeddings(self, action_feats: torch.Tensor) -> torch.Tensor:
        weight = self.backbone.state_tokens_embed.weight
        return self.action_trunk(
            self.action_input_proj(action_feats.to(device = weight.device, dtype = weight.dtype))
        )

    def q_values(
        self,
        obs_tokens: torch.Tensor,
        obs_channels: torch.Tensor,
        action_feats: torch.Tensor,
        masks: torch.Tensor,
        *,
        token_lengths: torch.Tensor | None = None,
        player_ids: torch.Tensor | None = None,
        return_ntp: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Score every candidate: ``[B, K]`` scalar Q values in one forward.

        ``obs_tokens`` is a packed token sequence per decision (``int32`` or
        ``int64``), ``obs_channels`` the actor channel vector, ``action_feats``
        the ``[B, K, 64]`` ``GD-DMC-ACTION-1`` block and ``masks`` the
        ``[B, K]`` legal-candidate mask.  ``token_lengths`` gives the true
        length of each packed token row; padded positions must contain the
        trunk's pad token (their embedding is zero) and are excluded from
        attention.  ``player_ids`` is optional because the stage-3 signature
        does not carry the acting seat; when omitted the neutral (padding)
        player embedding is used, so all four seats share one network.

        Invalid ``(row, candidate)`` slots are filled with
        :data:`INVALID_Q_SENTINEL`; they never contribute to a loss.

        With ``return_ntp=True`` the auxiliary prediction for the next
        observation token row is returned alongside Q (``None`` when the head
        is disabled).
        """
        if obs_tokens.dim() != 2:
            raise ValueError(f"obs_tokens must be [B, T], got {tuple(obs_tokens.shape)}")
        B, T = obs_tokens.shape
        if T <= 0:
            raise ValueError("obs_tokens must contain at least one token")
        if not torch.is_floating_point(obs_channels):
            obs_channels = obs_channels.to(torch.float32)
        if obs_channels.dim() != 2 or obs_channels.shape[0] != B:
            raise ValueError(
                f"obs_channels must be [B={B}, C], got {tuple(obs_channels.shape)}"
            )
        if obs_channels.shape[1] != self.num_channels:
            raise ValueError(
                f"obs_channels width {obs_channels.shape[1]} does not match the trunk "
                f"channel width {self.num_channels}"
            )
        if action_feats.dim() != 3:
            raise ValueError(
                f"action_feats must be [B, K, {self.action_feature_dim}], "
                f"got {tuple(action_feats.shape)}"
            )
        if action_feats.shape[0] != B or action_feats.shape[2] != self.action_feature_dim:
            raise ValueError(
                f"action_feats must be [B={B}, K, {self.action_feature_dim}], "
                f"got {tuple(action_feats.shape)}"
            )
        K = int(action_feats.shape[1])
        if K <= 0:
            raise ValueError("action_feats must contain at least one candidate")
        if masks.shape != (B, K):
            raise ValueError(f"masks must be [B={B}, K={K}], got {tuple(masks.shape)}")
        if masks.dtype != torch.bool:
            raise TypeError("masks must be a bool tensor")
        masks = masks.to(device=obs_tokens.device)
        if token_lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=obs_tokens.device)
        else:
            if token_lengths.shape != (B,):
                raise ValueError(f"token_lengths must be [B={B}], got {tuple(token_lengths.shape)}")
            lengths = token_lengths.to(device=obs_tokens.device, dtype=torch.long)
            if bool(((lengths < 1) | (lengths > T)).any()):
                raise ValueError("token_lengths must lie in 1..T")
        if player_ids is None:
            seat_2d = torch.full(
                (B, T), -1, dtype=torch.long, device=obs_tokens.device,
            )
        else:
            if player_ids.shape != (B,):
                raise ValueError(f"player_ids must be [B={B}], got {tuple(player_ids.shape)}")
            if bool(((player_ids < 0) | (player_ids >= self.num_players)).any()):
                raise ValueError("player_ids must lie in 0..num_players-1")
            seat_2d = player_ids.to(device=obs_tokens.device, dtype=torch.long).unsqueeze(1).expand(B, T)

        if obs_tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError("obs_tokens must be int32 or int64")
        tokens = obs_tokens.to(torch.long)

        time_mask = torch.arange(T, device=obs_tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
        # ``_encode_history`` expects [B, T, L_event]; DMC packs one event token
        # per position, so L_event == 1.
        decoded, _ = self.backbone._encode_history(
            tokens.unsqueeze(-1), seat_2d, time_mask,
        )
        last_index = (lengths - 1).clamp(min = 0)
        flat_index = torch.arange(B, device=obs_tokens.device) * T + last_index
        hist = decoded.reshape(B * T, decoded.shape[-1]).index_select(0, flat_index)

        dtype = self.backbone.state_tokens_embed.weight.dtype
        state_input = torch.cat(
            [hist, obs_channels.to(device=hist.device, dtype=dtype)], dim = -1,
        )
        player_state = self.backbone.state_mlp(state_input)
        action_emb = self._action_embeddings(action_feats)
        fused = player_state.unsqueeze(1) * action_emb
        if self.backbone.actor_fuse is not None:
            fused = self._apply_actor_fuse(fused, player_state, action_emb, masks)
        q = self.backbone.logit_head(fused).squeeze(-1)
        # Use the *output* dtype's finite minimum: under CUDA fp16 autocast the
        # float32 sentinel would overflow to -inf.  The fp32 CPU path writes
        # exactly ``INVALID_Q_SENTINEL``.
        q = q.masked_fill(~masks, torch.finfo(q.dtype).min)

        if return_ntp:
            ntp = None
            if self.backbone.ntp_head is not None:
                ntp = self.backbone.ntp_head(player_state)
            return q, ntp
        return q

    def _apply_actor_fuse(
        self,
        fused: torch.Tensor,
        player_state: torch.Tensor,
        action_emb: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Residual ActorFuse branch evaluated only on legal candidates.

        Mirrors ``Agent._apply_actor_fuse``: illegal rows keep an exactly-zero
        residual, so masking never changes a candidate that cannot be chosen.
        """
        B, K, F = action_emb.shape
        legal = masks.nonzero(as_tuple = False)
        row_idx = legal[:, 0]
        act_idx = legal[:, 1]
        delta = fused.new_zeros((B, K, F))
        if row_idx.numel() > 0:
            gathered_state = player_state.index_select(0, row_idx)
            gathered_action = action_emb[row_idx, act_idx]
            fused_branch = self.backbone.actor_fuse(
                Agent.actor_fuse_features(gathered_state, gathered_action),
            )
            delta[row_idx, act_idx] = fused_branch.to(dtype = delta.dtype)
        return fused + delta

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def checkpoint_payload(self) -> dict[str, object]:
        """Return a ``dmc-v1`` payload that :meth:`load_state_dict` accepts."""
        return {
            "schema": DMC_CHECKPOINT_SCHEMA,
            "action_feature_version": DMC_ACTION_FEATURE_VERSION,
            "model_state": self.state_dict(),
        }

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load only a ``dmc-v1`` payload; never an IPPO/VRPO policy.

        A raw tensor mapping (the shape of an ``agent_state_dict`` saved by the
        PPO trainers) carries no DMC schema and is rejected: importing such a
        policy is an explicit migration, not a checkpoint reload.
        """
        if not isinstance(state_dict, Mapping):
            raise DMCCheckpointError(
                "DMC checkpoints must be a mapping with schema='dmc-v1'"
            )
        schema = state_dict.get("schema")
        if schema != DMC_CHECKPOINT_SCHEMA:
            algorithm = state_dict.get("algorithm")
            if algorithm is None and "agent_state_dict" in state_dict:
                algorithm = state_dict.get("algo_name", "ippo/vrpo")
            if algorithm is not None or "agent_state_dict" in state_dict:
                raise DMCCheckpointError(
                    "refusing to load a non-DMC policy checkpoint "
                    f"(algorithm={algorithm!r}, schema={schema!r}); DMC starts from a fresh "
                    "initialisation. Import an old policy explicitly through "
                    "guandan_marvel.training.policy_import, never by load_state_dict."
                )
            raise DMCCheckpointError(
                "DMC checkpoint schema mismatch: expected "
                f"schema={DMC_CHECKPOINT_SCHEMA!r}, got {schema!r}. Raw policy weights "
                "(for example an IPPO/VRPO agent_state_dict) must go through "
                "guandan_marvel.training.policy_import."
            )
        version = state_dict.get("action_feature_version")
        if version is not None and str(version) != DMC_ACTION_FEATURE_VERSION:
            raise DMCCheckpointError(
                f"checkpoint action_feature_version {version!r} != "
                f"{DMC_ACTION_FEATURE_VERSION!r}"
            )
        model_state = state_dict.get("model_state")
        if not isinstance(model_state, Mapping):
            raise DMCCheckpointError(
                "dmc-v1 checkpoint must carry a 'model_state' mapping"
            )
        return super().load_state_dict(model_state, strict = strict, assign = assign)


__all__ = [
    "ACTION_FEATURE_DIM",
    "DMCAgent",
    "DMCCheckpointError",
    "INVALID_Q_SENTINEL",
]
