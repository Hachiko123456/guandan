# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/model/critic.py.
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from guandan_marvel.model.tensordict import TensorDict

from guandan_marvel.config import TrainConfig
from .base import BaseMarvelModule
from .cache import KVCache
from .types import CriticOutput


class CentralCritic(BaseMarvelModule):
    checkpoint_state_key = "critic_state_dict"

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        output_mode: str = "qvalue",
    ):
        super().__init__(cfg)
        mode = output_mode.lower()
        if mode not in {"qvalue", "value"}:
            raise ValueError(f"Unknown CentralCritic output_mode '{output_mode}'.")
        self.output_mode = mode
        self.value_head = self._build_output_proj(self.feature_dim, std = 0.01)
        mixer_dim = self.num_players * self.feature_dim
        if self.model_architecture_version >= (1, 2):
            self.critic_mixer = self._build_res_gated_mlp(
                mixer_dim,
                n_layers = self.model_cfg.fuse_layers,
                expansion_ratio = self.resmlp_expansion_ratio,
            )
        elif self.model_architecture_version >= (1, 1):
            mixer_blocks = self._build_residual_blocks(
                mixer_dim,
                n_layers = self.model_cfg.fuse_layers,
                module_name = "model.critic_mixer (fuse_layers)",
                repeats = 4,
            )
            self.critic_mixer = nn.Sequential(*mixer_blocks)
        else:
            self.critic_mixer = self._build_legacy_mlp(
                input_dim = mixer_dim,
                hidden_dim = mixer_dim,
                n_layers = self.model_cfg.fuse_layers,
                output_dim = mixer_dim,
            )

    def _mix_critic_state(self, f_state: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        self._shape(
            f_state,
            (None, None, self.num_players, self.feature_dim),
            "f_state",
        )
        flat = f_state.flatten(start_dim = 2)
        mixed = self.critic_mixer(flat)
        if self.model_architecture_version <= (1, 0):
            mixed = mixed + flat
        mixed = mixed.unflatten(-1, (self.num_players, self.feature_dim))
        return self._apply_mask(mixed, time_mask)

    def _compute_qvalues(
        self,
        mixed_state: torch.Tensor,
        action_feats: torch.Tensor,
    ) -> torch.Tensor:
        B, T, P, D = self._shape(
            mixed_state,
            (None, None, self.num_players, self.feature_dim),
            "mixed_state",
        )
        self._shape(action_feats, (B, T, None, D), "action_feats")

        weight = self.value_head.weight
        self._ensure(
            weight.shape[0] == 1,
            "CentralCritic qvalue requires value_head output_dim=1.",
        )
        self._ensure(
            weight.shape[1] == D,
            f"value_head input_dim {weight.shape[1]} does not match feature_dim {D}.",
        )

        # Contract the feature dimension directly to avoid materializing [B, T, A, P, D].
        qvalues = torch.einsum(
            "btpd,btad,d->btap",
            mixed_state,
            action_feats,
            weight.squeeze(0),
        )
        bias = self.value_head.bias
        if bias is not None:
            qvalues = qvalues + bias.view(1, 1, 1, 1)
        return qvalues

    def forward(
        self,
        rollout: TensorDict,
        *,
        use_cache: bool = False,
        past_key_values: Optional[KVCache] = None,
    ) -> CriticOutput:
        self._validate_rollout(rollout, require_actions = (self.output_mode == "qvalue"))
        num_steps, player_ids = rollout["num_steps"], rollout["player_ids"]
        obs_tokens, obs_channels = rollout["obs_tokens"], rollout["obs_channels"]
        avail_tokens, avail_weights = rollout.get("avail_tokens"), rollout.get("avail_weights")

        if use_cache:
            raise ValueError("CentralCritic does not support cached decoding during rollout.")

        device = obs_tokens.device
        B, T, _, _ = self._shape(obs_tokens, (None, None, self.num_players, None), "obs_tokens")

        time_mask = self._build_time_mask(num_steps, T, device = device)
        flat_time_mask = time_mask.repeat_interleave(self.num_players, dim = 0)
        flat_player_ids = (
            torch.arange(self.num_players, device = device)
            .expand(B, self.num_players)
            .reshape(-1, 1)
            .expand(-1, T)
        )

        flat_obs_tokens = self._flatten_players(obs_tokens, name = "obs_tokens")
        flat_obs_channels = self._flatten_players(obs_channels, name = "obs_channels")

        decoded, _ = self._encode_history(
            flat_obs_tokens,
            flat_player_ids,
            flat_time_mask,
            use_cache = False,
            past_key_values = past_key_values,
        )
        fused_state_flat = self._encode_state_features(decoded, flat_obs_channels, flat_time_mask)
        fused_state = self._unflatten_players(
            fused_state_flat,
            batch_size = B,
            name = "fused_state",
        )

        if self.output_mode == "value":
            mixed_state = self._mix_critic_state(fused_state, time_mask)
            state_values = self.value_head(mixed_state).squeeze(-1)  # [B, T, P]
            state_values = self._apply_mask(state_values, time_mask.unsqueeze(-1))
            acting_players = player_ids.clamp(min = 0)
            acting_idx = acting_players.unsqueeze(-1)
            player_mask = (player_ids >= 0) & time_mask
            player_state_value = state_values.gather(dim = 2, index = acting_idx).squeeze(-1)
            player_state_value = player_state_value * player_mask.to(player_state_value.dtype)
            return CriticOutput(
                state_values = state_values,
                player_state_value = player_state_value,
                past_key_values = past_key_values,
            )

        elif self.output_mode == "qvalue":
            if avail_tokens is None or avail_weights is None:
                raise RuntimeError("qvalue critic mode requires avail_tokens and avail_weights tensors.")
            action_feats, avail_mask = self._encode_action_features(
                avail_tokens,
                avail_weights,
                time_mask,
            )
            mixed_state = self._mix_critic_state(fused_state, time_mask)  # [B, T, P, D]
            qvalues = self._compute_qvalues(mixed_state, action_feats)  # [B, T, A, P]
            qvalues = qvalues.masked_fill(~avail_mask.unsqueeze(-1), 0.0)
            qvalues = self._apply_mask(qvalues, time_mask.unsqueeze(-1).unsqueeze(-1))
            return CriticOutput(qvalues = qvalues, past_key_values = past_key_values)

        else:
            raise ValueError(f"Unknown output_mode '{self.output_mode}'.")
