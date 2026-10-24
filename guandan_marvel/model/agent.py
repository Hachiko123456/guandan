# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/model/agent.py.
from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
from guandan_marvel.model.tensordict import TensorDict

from guandan_marvel.config import TrainConfig
from .base import BaseMarvelModule
from .cache import DynamicCache, KVCache
from .types import ActiveAgentOutput


class Agent(BaseMarvelModule):
    checkpoint_state_key = "agent_state_dict"

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        enable_value_head: bool = False,
        enable_ntp_head: bool = False,
    ):
        super().__init__(cfg)
        self.value_head: nn.Module | None = (
            self._build_output_proj(self.feature_dim, std = 0.01) if enable_value_head else None
        )
        self.logit_head = self._build_output_proj(self.feature_dim)
        # v1.4 residual actor fuse.  The legacy multiply path
        # ``player_state * action_feats`` is kept verbatim and remains the ``z``
        # this branch adds its (initially exactly zero) residual to.
        self.actor_fuse: nn.Module | None = (
            self._build_actor_fuse(
                self.feature_dim,
                n_layers = int(self.model_cfg.fuse_layers),
                expansion_ratio = self.resmlp_expansion_ratio,
            )
            if self.actor_fuse_enabled
            else None
        )
        # Auxiliary next-observation token-prediction head (training-only).
        # Predicts which observation tokens become newly visible at the next
        # step given the current acting-player state representation.
        self.ntp_head: nn.Module | None = (
            self._build_output_proj(self.feature_dim, self.num_tokens, std = 0.02)
            if enable_ntp_head else None
        )

    @property
    def enable_ntp(self) -> bool:
        return self.ntp_head is not None

    @staticmethod
    def ntp_enabled_for_config(cfg: TrainConfig) -> bool:
        return float(getattr(cfg, "ntp_weight", 0.0) or 0.0) > 0.0

    def enable_value_head(self) -> None:
        if self.value_head is None:
            self.value_head = self._build_output_proj(self.feature_dim, std = 0.01)

    def forward(
        self,
        rollout: TensorDict,
        *,
        use_cache: bool = False,
        past_key_values: Optional[KVCache] = None,
        cache_only: bool = False,
    ) -> ActiveAgentOutput | None:
        """Run the agent policy on per-environment trajectories.

        Cache contract:
        - `rollout` stays environment-shaped (`B` batch, `P` players).
        - Internally the decoder path flattens players to `BP = B * P`.
        - When `use_cache=True`, `past_key_values.batch_size` must equal `BP`.

        When *cache_only* is True the decoder is run once over the full
        sequence to populate *past_key_values* (via a temporary
        :class:`DynamicCache`), skipping the state / action / policy
        heads entirely.  Returns ``None`` in this mode.
        """
        self._validate_rollout(rollout)
        num_steps, player_ids = rollout["num_steps"], rollout["player_ids"]
        obs_tokens, obs_channels = rollout["obs_tokens"], rollout["obs_channels"]
        avail_tokens, avail_weights = rollout["avail_tokens"], rollout["avail_weights"]

        device = obs_tokens.device
        B, T, P, _ = self._shape(obs_tokens, (None, None, self.num_players, None), "obs_tokens")
        B, T, A, L_act = self._shape(avail_tokens, (B, T, None, None), "avail_tokens")

        time_mask = self._build_time_mask(num_steps, T, device = device)
        flat_time_mask = time_mask.repeat_interleave(P, dim = 0)
        flat_num_steps = num_steps.repeat_interleave(P)
        flat_player_ids = torch.arange(P, device = device).expand(B, P).reshape(-1, 1).expand(-1, T)
        flat_obs_tokens = self._flatten_players(obs_tokens, name = "obs_tokens")

        if use_cache:
            if past_key_values is None:
                raise ValueError("past_key_values must be provided when use_cache=True.")
            if not cache_only:
                expected_batch = B * P
                if int(past_key_values.batch_size) != int(expected_batch):
                    raise ValueError(
                        "When use_cache=True, past_key_values.batch_size must match flattened player batch "
                        f"B*P={B}*{P}={expected_batch}; got {past_key_values.batch_size}."
                    )

        if cache_only:
            token_embeds = self._embed_tokens(flat_obs_tokens, flat_player_ids)
            position_ids = torch.arange(T, device = device).unsqueeze(0).expand(B * P, T)
            temp_cache = DynamicCache()
            self.decoder(
                inputs_embeds = token_embeds,
                attention_mask = flat_time_mask.long(),
                position_ids = position_ids,
                use_cache = True,
                past_key_values = temp_cache,
                return_dict = True,
            )
            past_key_values.populate(temp_cache, flat_num_steps)
            return None

        decoded, _ = self._encode_history(
            flat_obs_tokens,
            flat_player_ids,
            flat_time_mask,
            use_cache = use_cache,
            past_key_values = past_key_values,
        )

        # History/KV state remains per observer (B*P), but only the acting
        # player's state/action/policy heads affect the environment action.
        # Gather that observer before the expensive MLP heads instead of
        # encoding identical candidates four times and masking three copies.
        decoded_by_player = self._unflatten_players(
            decoded, batch_size = B, name = "decoded",
        )
        acting_players = player_ids.clamp(min = 0)
        mask = player_ids >= 0
        player_mask = mask & time_mask
        decoded_index = acting_players.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, decoded_by_player.shape[-1],
        )
        acting_decoded = decoded_by_player.gather(
            dim = 2, index = decoded_index,
        ).squeeze(2)
        channel_index = acting_players.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, obs_channels.shape[-1],
        )
        acting_channels = obs_channels.gather(
            dim = 2, index = channel_index,
        ).squeeze(2)

        player_state = self._encode_state_features(
            acting_decoded, acting_channels, player_mask,
        )
        ntp_logits = None
        if self.ntp_head is not None:
            ntp_logits = self.ntp_head(player_state)
        action_feats, avail_mask = self._encode_action_features(
            avail_tokens, avail_weights, player_mask,
        )
        fused_action_state = player_state.unsqueeze(-2) * action_feats
        if self.actor_fuse is not None:
            fused_action_state = self._apply_actor_fuse(
                fused_action_state, player_state, action_feats, avail_mask,
            )
        avail_logits = self.logit_head(fused_action_state).squeeze(-1)
        mask_value = torch.finfo(avail_logits.dtype).min
        avail_logits = avail_logits.masked_fill(~avail_mask, mask_value)
        avail_logits = avail_logits + avail_mask.to(dtype = avail_logits.dtype) * avail_weights.clamp(min = 1e-9).log()

        agent_probs = torch.softmax(avail_logits.float(), dim = -1)
        agent_probs = self._apply_mask(agent_probs, avail_mask).to(dtype = avail_logits.dtype)
        # Phase-2 macro-sequence PPO consumes an FP32 log_softmax of the masked
        # logits (never log(clamp(prob))).  This is a non-parametric extra field:
        # the legacy probs/argmax/parameter set/state_dict stay unchanged.
        avail_log_probs = torch.log_softmax(avail_logits.float(), dim = -1)
        sum_weights = avail_weights.sum(dim = -1, keepdim = True)
        env_uniform = torch.full_like(avail_weights, 1.0 / avail_weights.size(-1))
        env_probs = torch.where(sum_weights > 0, avail_weights / sum_weights.clamp(min = 1e-9), env_uniform)
        avail_probs = torch.where(mask.unsqueeze(-1), agent_probs, env_probs)

        state_values = None
        player_state_value = None
        if self.value_head is not None:
            value_logits = self.value_head(player_state).squeeze(-1)
            player_state_value = self._apply_mask(value_logits, player_mask)
            state_values = value_logits.new_zeros((B, T, P))
            state_values.scatter_(
                2,
                acting_players.unsqueeze(-1),
                player_state_value.unsqueeze(-1),
            )
            state_values = state_values * time_mask.unsqueeze(-1).to(state_values.dtype)

        return ActiveAgentOutput(
            avail_probs = avail_probs,
            mask = mask,
            past_key_values = past_key_values,
            state_values = state_values,
            player_state_value = player_state_value,
            ntp_logits = ntp_logits,
            avail_log_probs = avail_log_probs,
        )

    @staticmethod
    def actor_fuse_features(
        player_state: torch.Tensor,
        action_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Four-way fuse input ``u = [s, a, s * a, |s - a|]`` (last axis)."""
        return torch.cat(
            (
                player_state,
                action_feats,
                player_state * action_feats,
                (player_state - action_feats).abs(),
            ),
            dim = -1,
        )

    def _apply_actor_fuse(
        self,
        fused_action_state: torch.Tensor,
        player_state: torch.Tensor,
        action_feats: torch.Tensor,
        avail_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Add the residual fuse branch, computed only on legal candidates.

        ``avail_mask`` is ``[B, T, A]``; only the ``(row, action)`` pairs it marks
        true are gathered, run through the fuse MLP and scattered back, so the
        number of fused candidates equals the number of legal actions.  Illegal
        (and time-masked) candidates keep an exactly-zero residual and therefore
        reduce to the legacy multiply path.
        """
        B, T, A, D = action_feats.shape
        flat_mask = avail_mask.reshape(B * T, A)
        flat_state = player_state.reshape(B * T, D)
        flat_action = action_feats.reshape(B * T, A, D)

        legal_indices = flat_mask.nonzero(as_tuple = False)  # [N, 2] -> (row, action)
        row_idx = legal_indices[:, 0]
        act_idx = legal_indices[:, 1]

        delta = fused_action_state.new_zeros((B * T, A, D))
        if row_idx.numel() > 0:
            gathered_state = flat_state.index_select(0, row_idx)
            gathered_action = flat_action[row_idx, act_idx]
            fused = self.actor_fuse(
                self.actor_fuse_features(gathered_state, gathered_action),
            )
            # ``delta=delta.to(z.dtype)`` before the residual add: autocast may hand
            # back a reduced-precision branch output while ``z`` is FP32.
            delta[row_idx, act_idx] = fused.to(dtype = delta.dtype)
        return fused_action_state + delta.reshape(B, T, A, D)
