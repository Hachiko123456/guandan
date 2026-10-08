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

    def __init__(self, cfg: TrainConfig, *, enable_value_head: bool = False):
        super().__init__(cfg)
        self.value_head: nn.Module | None = (
            self._build_output_proj(self.feature_dim, std = 0.01) if enable_value_head else None
        )
        self.logit_head = self._build_output_proj(self.feature_dim)

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

        player_range = self.player_index.to(device = device)
        active_mask = (player_ids.unsqueeze(-1) == player_range) & (player_ids.unsqueeze(-1) >= 0)
        flat_active_mask = active_mask.permute(0, 2, 1).reshape(B * P, T).to(dtype = torch.bool)

        flat_obs_channels = self._flatten_players(obs_channels, name = "obs_channels")
        flat_avail_tokens = avail_tokens.repeat_interleave(P, dim = 0).reshape(B * P, T, A, L_act)
        flat_avail_weights = (
            avail_weights.repeat_interleave(P, dim = 0).reshape(B * P, T, A)
            * flat_time_mask.unsqueeze(-1).to(dtype = avail_weights.dtype)
        )

        decoded, _ = self._encode_history(
            flat_obs_tokens,
            flat_player_ids,
            flat_time_mask,
            use_cache = use_cache,
            past_key_values = past_key_values,
        )
        fused_state = self._encode_state_features(decoded, flat_obs_channels, flat_time_mask)

        player_state = self._apply_mask(fused_state, flat_active_mask)
        action_feats, avail_mask = self._encode_action_features(
            flat_avail_tokens,
            flat_avail_weights,
            flat_time_mask,
        )
        avail_mask = avail_mask & flat_active_mask.unsqueeze(-1)

        fused_action_state = player_state.unsqueeze(-2) * action_feats
        avail_logits = self.logit_head(fused_action_state).squeeze(-1)
        mask_value = torch.finfo(avail_logits.dtype).min
        avail_logits = avail_logits.masked_fill(~avail_mask, mask_value)
        avail_logits = avail_logits + avail_mask.to(dtype = avail_logits.dtype) * flat_avail_weights.clamp(min = 1e-9).log()

        flat_avail_probs = torch.softmax(avail_logits.float(), dim = -1)
        flat_avail_probs = self._apply_mask(flat_avail_probs, avail_mask).to(dtype = avail_logits.dtype)

        flat_state_values = None
        if self.value_head is not None:
            value_logits = self.value_head(player_state).squeeze(-1)
            flat_state_values = self._apply_mask(value_logits, flat_active_mask)

        BP, T, A = self._shape(flat_avail_probs, (None, None, None), "flat_avail_probs")
        self._ensure(BP % self.num_players == 0, "Agent output batch must be divisible by number of players.")
        self._shape(player_ids, (B, T), "player_ids")
        self._shape(avail_weights, (B, T, A), "avail_weights")
        self._shape(num_steps, (B,), "num_steps")

        avail_probs_full = self._unflatten_players(flat_avail_probs, batch_size = B, name = "flat_avail_probs")
        acting_players = player_ids.clamp(min = 0)
        mask = player_ids >= 0
        player_mask = mask & time_mask

        acting_idx = acting_players.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, A)
        agent_probs = avail_probs_full.gather(dim = 2, index = acting_idx).squeeze(2)

        sum_weights = avail_weights.sum(dim = -1, keepdim = True)
        env_uniform = torch.full_like(avail_weights, 1.0 / avail_weights.size(-1))
        env_probs = torch.where(sum_weights > 0, avail_weights / sum_weights.clamp(min = 1e-9), env_uniform)
        avail_probs = torch.where(mask.unsqueeze(-1), agent_probs, env_probs)

        state_values = None
        player_state_value = None
        if flat_state_values is not None:
            self._shape(flat_state_values, (BP, T), "flat_state_values")
            state_values = self._unflatten_players(
                flat_state_values,
                batch_size = B,
                name = "flat_state_values",
            )
            state_values = state_values * time_mask.unsqueeze(-1).to(state_values.dtype)
            acting_idx_values = acting_players.unsqueeze(-1)
            player_state_value = state_values.gather(dim = 2, index = acting_idx_values).squeeze(-1)
            player_state_value = player_state_value * player_mask.to(player_state_value.dtype)

        return ActiveAgentOutput(
            avail_probs = avail_probs,
            mask = mask,
            past_key_values = past_key_values,
            state_values = state_values,
            player_state_value = player_state_value,
        )
