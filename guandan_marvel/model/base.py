# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/model/base.py.
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from guandan_marvel.config import TrainConfig, deserialize_train_config
from guandan_marvel.utils import get_game_attr
from .cache import KVCache
from .checkpointable import CheckpointableModule
from .tensordict import TensorDict
from .trainable import TrainableModule
from .transformer import TransformerConfig, TransformerMLP, TransformerModel


def _dynamo_disable():
    disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    if callable(disable):
        return disable()

    def decorator(fn):
        return fn

    return decorator


class _FallbackRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim = -1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


if getattr(nn, "RMSNorm", None) is not None:
    RMSNorm = nn.RMSNorm
else:
    RMSNorm = _FallbackRMSNorm


# ---------------------------------------------------------------------------
# v1.2 ResGatedMLP — transformer MLP stack without attention
# ---------------------------------------------------------------------------

class ResGatedMLP(nn.Module):
    """Stack of pre-norm SwiGLU residual blocks with an output norm.

    Each layer applies ``x = x + MLP(RMSNorm(x))`` where ``MLP`` is
    :class:`~guandan_marvel.model.transformer.TransformerMLP` (SwiGLU).  A final
    ``RMSNorm`` is appended after the last block, matching the standard
    transformer output-norm convention.
    """

    def __init__(self, dim: int, *, n_layers: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        n_layers = int(n_layers)
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}.")
        intermediate_dim = int(dim * expansion_ratio)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm": RMSNorm(dim),
                "mlp": TransformerMLP(dim, intermediate_dim),
            }))
        self.output_norm = RMSNorm(dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = hidden_states
            hidden_states = layer["norm"](hidden_states)
            hidden_states = layer["mlp"](hidden_states)
            hidden_states = residual + hidden_states
        return self.output_norm(hidden_states)


class BaseMarvelModule(CheckpointableModule, TrainableModule):
    @staticmethod
    def _ensure(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    @staticmethod
    def _shape(tensor: torch.Tensor, dims: tuple[int | None, ...], name: str) -> tuple[int, ...]:
        BaseMarvelModule._ensure(
            tensor.dim() == len(dims),
            f"{name} must be {len(dims)}D, got {tensor.dim()}D.",
        )
        shape = tensor.shape
        for axis, target in enumerate(dims):
            if target is not None and shape[axis] != target:
                raise ValueError(f"{name} dim {axis} expected {target}, got {shape[axis]}.")
        return tuple(shape)

    # ------------------------------------------------------------------
    # v1.0/v1.1 legacy MLP builders (delegated to marvel.model.legacy)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_legacy_mlp(
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        output_dim: int,
    ) -> nn.Sequential:
        from .legacy import _build_legacy_mlp
        return _build_legacy_mlp(input_dim, hidden_dim, n_layers, output_dim)

    @staticmethod
    def _build_linear_rms_gelu_block(input_dim: int, output_dim: int) -> nn.Sequential:
        from .legacy import _build_linear_rms_gelu_block
        return _build_linear_rms_gelu_block(input_dim, output_dim)

    @staticmethod
    def _build_residual_mlp_block(dim: int, *, repeats: int = 4) -> nn.Module:
        from .legacy import _build_residual_mlp_block
        return _build_residual_mlp_block(dim, repeats = repeats)

    @staticmethod
    def _build_residual_blocks(
        dim: int,
        *,
        n_layers: int,
        module_name: str = "model.mlp_layers",
        repeats: int = 4,
    ) -> list[nn.Module]:
        from .legacy import _build_residual_blocks
        return _build_residual_blocks(dim, n_layers = n_layers, module_name = module_name, repeats = repeats)

    @staticmethod
    def _build_structured_mlp(
        *,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        output_dim: int,
    ) -> nn.Sequential:
        from .legacy import _build_structured_mlp
        return _build_structured_mlp(input_dim = input_dim, hidden_dim = hidden_dim, n_layers = n_layers, output_dim = output_dim)

    # ------------------------------------------------------------------
    # v1.2 GatedMLP builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_res_gated_mlp(
        dim: int,
        *,
        n_layers: int,
        expansion_ratio: int = 2,
    ) -> ResGatedMLP:
        return ResGatedMLP(dim, n_layers = n_layers, expansion_ratio = expansion_ratio)

    @staticmethod
    def _build_input_proj(input_dim: int, output_dim: int) -> nn.Linear:
        layer = BaseMarvelModule._init_module(nn.Linear(input_dim, output_dim))
        BaseMarvelModule._mark_no_weight_decay(layer)
        return layer

    @staticmethod
    def _build_output_proj(input_dim: int, output_dim: int = 1, std: float = 1.0) -> nn.Linear:
        layer = BaseMarvelModule._init_module(nn.Linear(input_dim, output_dim), std = std)
        BaseMarvelModule._mark_no_weight_decay(layer)
        return layer

    @staticmethod
    def _build_time_mask(num_steps: torch.Tensor, seq_len: int, *, device) -> torch.Tensor:
        return torch.arange(seq_len, device = device).unsqueeze(0) < num_steps.unsqueeze(1)

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = getattr(cfg, "model", cfg)
        architecture_raw = getattr(self.model_cfg, "architecture", "v1") or "v1"
        if isinstance(architecture_raw, (tuple, list)) and len(architecture_raw) == 2:
            self.model_architecture_version = (int(architecture_raw[0]), int(architecture_raw[1]))
        elif architecture_raw == "v1":
            self.model_architecture_version = (1, 0)
        elif architecture_raw == "v1.1":
            self.model_architecture_version = (1, 1)
        elif architecture_raw == "v1.2":
            self.model_architecture_version = (1, 2)
        elif architecture_raw == "v1.3":
            self.model_architecture_version = (1, 3)
        else:
            raise ValueError(
                "Unsupported model architecture "
                f"'{architecture_raw}'. Expected 'v1', 'v1.1', 'v1.2', 'v1.3', or a (major, minor) tuple."
            )
        game = cfg.game
        self.max_game_horizon = int(get_game_attr(game, "max_game_horizon"))

        self._wd_param_names: Optional[Tuple[str, ...]] = None

        self.pad_token_id = int(get_game_attr(game, "pad_token"))
        self.num_tokens = int(get_game_attr(game, "num_tokens"))
        self.num_channels = int(get_game_attr(game, "num_channels"))

        self.num_players = int(get_game_attr(game, "num_players"))
        self.decoder_n_layers = self.model_cfg.decoder_layers
        self.decoder_n_heads = self.model_cfg.decoder_heads
        self.decoder_ffn_ratio = getattr(self.model_cfg, "ffn_ratio", 2)

        self.history_dim = self.model_cfg.history_dim
        self.feature_dim = self.model_cfg.feature_dim
        self.state_layers = self.model_cfg.state_layers
        self.action_layers = self.model_cfg.action_layers

        self.state_tokens_embed = self._init_module(
            nn.Embedding(self.num_tokens, self.history_dim, padding_idx = self.pad_token_id),
            pad_token = self.pad_token_id,
        )
        self.state_player_embed = self._init_module(
            nn.Embedding(self.num_players + 1, self.history_dim, padding_idx = 0), pad_token = 0
        )
        self.action_tokens_embed = self._init_module(
            nn.Embedding(self.num_tokens, self.feature_dim, padding_idx = self.pad_token_id),
            pad_token = self.pad_token_id,
        )
        self.channel_encoder: nn.Module | None = None
        self.register_buffer(
            "player_index",
            torch.arange(self.num_players, dtype = torch.long).view(1, 1, self.num_players),
            persistent = False,
        )

        self.state_input_dim = self.history_dim + self.num_channels
        self.mlp_expansion_ratio = int(getattr(self.model_cfg, "ffn_ratio", 2))
        # v1.3: hard-coded FFN ratios — ResMLP=1, Transformer=2.
        self.resmlp_expansion_ratio = 1 if self.model_architecture_version >= (1, 3) else self.mlp_expansion_ratio
        if self.model_architecture_version >= (1, 2):
            self.state_mlp = nn.Sequential(
                self._build_input_proj(self.state_input_dim, self.feature_dim),
                self._build_res_gated_mlp(
                    self.feature_dim,
                    n_layers = self.state_layers,
                    expansion_ratio = self.resmlp_expansion_ratio,
                ),
            )
            self.action_mlp = nn.Sequential(
                self._build_input_proj(self.feature_dim, self.feature_dim),
                self._build_res_gated_mlp(
                    self.feature_dim,
                    n_layers = self.action_layers,
                    expansion_ratio = self.resmlp_expansion_ratio,
                ),
            )
        else:
            mlp_builder = (
                self._build_structured_mlp
                if self.model_architecture_version >= (1, 1)
                else self._build_legacy_mlp
            )
            self.state_mlp = mlp_builder(
                input_dim = self.state_input_dim,
                hidden_dim = self.feature_dim,
                n_layers = self.state_layers,
                output_dim = self.feature_dim,
            )
            self.action_mlp = mlp_builder(
                input_dim = self.feature_dim,
                hidden_dim = self.feature_dim,
                n_layers = self.action_layers,
                output_dim = self.feature_dim,
            )

        # v1.3: hard-coded Transformer FFN ratio of 2.
        transformer_ffn_ratio = 2 if self.model_architecture_version >= (1, 3) else self.decoder_ffn_ratio
        decoder_config = TransformerConfig(
            vocab_size = 0,
            hidden_size = self.history_dim,
            num_hidden_layers = self.decoder_n_layers,
            num_attention_heads = self.decoder_n_heads,
            intermediate_size = int(self.history_dim * transformer_ffn_ratio),
            max_position_embeddings = self.max_game_horizon,
            pad_token_id = self.pad_token_id,
            use_gated_attention = self.model_architecture_version >= (1, 1),
        )
        self.decoder = TransformerModel(decoder_config)

    @staticmethod
    def checkpoint_architecture(cfg: TrainConfig | None) -> str | None:
        if cfg is None:
            return None
        model_cfg = getattr(cfg, "model", None)
        raw = str(getattr(model_cfg, "architecture", "v1"))
        if raw not in {"v1", "v1.1", "v1.2", "v1.3"}:
            raise ValueError(
                "Unsupported model architecture "
                f"'{raw}'. Expected one of: v1, v1.1, v1.2, v1.3."
            )
        return raw

    @classmethod
    def validate_checkpoint_config_compatibility(
        cls,
        *,
        provided_cfg: TrainConfig,
        checkpoint_cfg: TrainConfig,
    ) -> None:
        if provided_cfg.game_name != checkpoint_cfg.game_name:
            raise ValueError(
                "Provided configuration game_name does not match checkpoint train_config "
                f"({provided_cfg.game_name} != {checkpoint_cfg.game_name})."
            )
        provided_arch = cls.checkpoint_architecture(provided_cfg)
        checkpoint_arch = cls.checkpoint_architecture(checkpoint_cfg)
        if provided_arch != checkpoint_arch:
            raise ValueError(
                "Provided configuration model.architecture does not match checkpoint train_config "
                f"({provided_arch} != {checkpoint_arch})."
            )

    @staticmethod
    def checkpoint_train_config(checkpoint: Mapping[str, Any]) -> TrainConfig | None:
        payload = checkpoint.get("train_config")
        if payload is None:
            return None
        return deserialize_train_config(payload)

    @classmethod
    def resolve_checkpoint_config(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        cfg: TrainConfig | None = None,
    ) -> TrainConfig:
        checkpoint_cfg = cls.checkpoint_train_config(checkpoint)
        if checkpoint_cfg is None:
            if cfg is None:
                raise KeyError(
                    "Checkpoint does not include 'train_config'. "
                    "Pass cfg=TrainConfig(...) explicitly when loading this checkpoint."
                )
            return cfg

        if cfg is not None:
            cls.validate_checkpoint_config_compatibility(
                provided_cfg = cfg,
                checkpoint_cfg = checkpoint_cfg,
            )
            return cfg
        return checkpoint_cfg

    _INTEGRAL_DTYPES = frozenset({
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.long,
    })

    def _validate_rollout(self, rollout: TensorDict, *, require_actions: bool = True) -> None:
        if not isinstance(rollout, TensorDict):
            raise TypeError(f"rollout must be TensorDict, got {type(rollout).__name__}.")

        required = ("num_steps", "player_ids", "obs_tokens", "obs_channels")
        if require_actions:
            required = required + ("avail_tokens", "avail_weights")

        missing = [k for k in required if k not in rollout.keys()]
        if missing:
            raise KeyError(f"rollout is missing required fields: {', '.join(missing)}.")

        non_tensors = [k for k in required if not isinstance(rollout[k], torch.Tensor)]
        if non_tensors:
            raise TypeError(f"rollout fields must be torch.Tensor: {', '.join(non_tensors)}.")

        device = rollout["num_steps"].device
        mismatched = [k for k in required if rollout[k].device != device]
        if mismatched:
            raise ValueError(f"rollout fields must share one device; mismatched: {', '.join(mismatched)}.")

        long_keys = ("num_steps", "player_ids", "obs_tokens") + (("avail_tokens",) if require_actions else ())
        for key in long_keys:
            if rollout[key].dtype != torch.long:
                raise TypeError(f"rollout['{key}'] must use torch.long dtype.")
        if rollout["obs_channels"].dtype not in self._INTEGRAL_DTYPES:
            raise TypeError("rollout['obs_channels'] must use an integral dtype.")
        float_keys = ("avail_weights",) if require_actions else ()
        for key in float_keys:
            if not torch.is_floating_point(rollout[key]):
                raise TypeError(f"rollout['{key}'] must use a floating dtype.")

        B = self._shape(rollout["obs_tokens"], (None, None, self.num_players, None), "obs_tokens")[0]
        T = rollout["player_ids"].shape[1]
        self._shape(rollout["num_steps"], (B,), "num_steps")
        self._shape(rollout["player_ids"], (B, T), "player_ids")
        self._shape(rollout["obs_channels"], (B, T, self.num_players, self.num_channels), "obs_channels")
        if require_actions:
            _, _, A, _ = self._shape(rollout["avail_tokens"], (B, T, None, None), "avail_tokens")
            self._shape(rollout["avail_weights"], (B, T, A), "avail_weights")

    @staticmethod
    def _apply_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expand = tensor.dim() - mask.dim()
        if expand > 0:
            mask = mask.reshape(*mask.shape, *([1] * expand))
        return tensor * mask.to(tensor.dtype)

    def _flatten_players(self, tensor: torch.Tensor, *, name: str) -> torch.Tensor:
        """Flatten the player axis into the batch axis.

        Expects `tensor` to be shaped `[B, T, P, ...]` and returns `[B*P, T, ...]`.
        """
        self._ensure(tensor.dim() >= 3, f"{name} must be at least 3D.")
        B, T, P = tensor.shape[:3]
        self._ensure(
            P == self.num_players,
            f"{name} player dim expected {self.num_players}, got {P}.",
        )
        return tensor.transpose(1, 2).reshape(B * P, T, *tensor.shape[3:])

    def _unflatten_players(
        self,
        tensor: torch.Tensor,
        *,
        batch_size: int | None = None,
        name: str,
    ) -> torch.Tensor:
        """Unflatten a player-flattened batch into `[B, T, P, ...]`."""
        self._ensure(tensor.dim() >= 2, f"{name} must be at least 2D.")
        BP, T = tensor.shape[:2]
        P = self.num_players
        if batch_size is None:
            self._ensure(
                BP % P == 0,
                f"{name} batch {BP} must be divisible by num_players {P}.",
            )
            B = BP // P
        else:
            B = int(batch_size)
            self._ensure(B * P == BP, f"{name} leading dim expected {B*P}, got {BP}.")

        reshaped = tensor.reshape(B, P, T, *tensor.shape[2:])
        perm = (0, 2, 1, *range(3, reshaped.dim()))
        return reshaped.permute(perm).contiguous()

    def _time_positions(
        self,
        time_mask: torch.Tensor,
        past_key_values: Optional[KVCache] = None,
    ) -> torch.Tensor:
        B, T = self._shape(time_mask, (None, None), "time_mask")
        if past_key_values is not None:
            return past_key_values.next_position_ids(time_mask.long()).view(B, T)
        return torch.arange(T, device = time_mask.device).unsqueeze(0).expand(B, T)

    def _embed_tokens(self, tokens: torch.Tensor, player_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.state_tokens_embed(tokens).sum(dim = -2)
        safe_ids = player_ids.clamp(min = -1)
        return embeds + self.state_player_embed(safe_ids.long() + 1)

    def _encode_channels(self, obs_channels: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = self._shape(obs_channels, (None, None, self.num_channels), "obs_channels")
        self._shape(time_mask, (B, T), "time_mask")
        leading = (B, T)

        if self.num_channels == 0:
            empty = torch.zeros(
                (*leading, 0),
                device = obs_channels.device,
                dtype = self.state_tokens_embed.weight.dtype,
            )
            return self._apply_mask(empty, time_mask)

        channel_input = obs_channels.to(
            dtype = self.state_tokens_embed.weight.dtype,
        )
        feats = (
            channel_input
            if self.channel_encoder is None
            else self.channel_encoder(channel_input)
        )
        return self._apply_mask(feats, time_mask)

    def _decode(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_cache: bool = False,
        past_key_values: Optional[KVCache] = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = inputs_embeds.shape[:2]
        if use_cache:
            self._ensure(T == 1, "When use_cache=True seq_len must equal 1 for fast rollout.")
            self._ensure(
                past_key_values is not None,
                "past_key_values must be provided when use_cache=True.",
            )
            self._ensure(
                past_key_values.batch_size == B,
                "KVCache batch dimension does not match inputs.",
            )
            new_mask = attention_mask.long()
            past_key_values.set_step_mask(new_mask)
            pos = (
                position_ids
                if position_ids is not None
                else past_key_values.next_position_ids(new_mask)
            )
            # Static KV caches report a fixed max length via get_seq_length() for shape stability,
            # so cached decoder calls must pass the real sequence position explicitly.
            cache_position = pos.max(dim = 0).values
            outputs = self.decoder(
                inputs_embeds = inputs_embeds,
                attention_mask = past_key_values.get_full_attention_mask(new_mask),
                position_ids = pos,
                cache_position = cache_position,
                use_cache = True,
                past_key_values = past_key_values,
                return_dict = True,
            )
            past_key_values.advance(new_mask)
            decoded = outputs.last_hidden_state
        else:
            pos = position_ids
            if pos is None:
                pos = torch.arange(T, device = inputs_embeds.device).unsqueeze(0).expand(B, T)
            decoded = self.decoder(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask.long(),
                position_ids = pos,
                use_cache = False,
                return_dict = True,
            ).last_hidden_state
        return self._apply_mask(decoded, attention_mask)

    def _encode_history(
        self,
        obs_tokens: torch.Tensor,
        player_ids: torch.Tensor,
        time_mask: torch.Tensor,
        *,
        use_cache: bool = False,
        past_key_values: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = self._shape(obs_tokens, (None, None, None), "obs_tokens")
        self._shape(time_mask, (B, T), "time_mask")
        self._shape(player_ids, (B, T), "player_ids")

        pos = self._time_positions(time_mask, past_key_values if use_cache else None)
        token_embeds = self._embed_tokens(obs_tokens, player_ids)
        decoded = self._decode(
            token_embeds,
            time_mask,
            use_cache = use_cache,
            past_key_values = past_key_values,
            position_ids = pos,
        )
        return self._apply_mask(decoded, time_mask), pos

    @_dynamo_disable()
    def _encode_sparse_states_eager(
        self,
        flat_fused: torch.Tensor,
        flat_mask: torch.Tensor,
    ) -> torch.Tensor:
        BT, _ = self._shape(flat_fused, (None, None), "flat_fused")
        self._shape(flat_mask, (BT,), "flat_mask")

        idx = flat_mask.nonzero(as_tuple = False).squeeze(-1)
        encoded = self.state_mlp(flat_fused.index_select(0, idx))
        out = encoded.new_zeros((flat_fused.shape[0], self.feature_dim))
        out.index_copy_(0, idx, encoded)
        return out

    def _encode_state_features(
        self,
        decoded_feats: torch.Tensor,
        obs_channels: torch.Tensor,
        time_mask: torch.Tensor,
        *,
        use_sparse: bool = True,
    ) -> torch.Tensor:
        channel_feats = self._encode_channels(obs_channels, time_mask)
        self._ensure(
            decoded_feats.shape[:-1] == channel_feats.shape[:-1],
            "decoded_feats and obs_channels must align before the feature dimension.",
        )
        fused = torch.cat([decoded_feats, channel_feats], dim = -1)

        B, T, _ = self._shape(fused, (None, None, None), "fused")
        self._shape(time_mask, (B, T), "time_mask")
        flat_fused = fused.reshape(B * T, fused.shape[-1])
        flat_mask = time_mask.reshape(B * T)

        if use_sparse:
            encoded_flat = self._encode_sparse_states_eager(flat_fused, flat_mask)
        else:
            masked_fused = flat_fused * flat_mask.unsqueeze(-1).to(dtype = flat_fused.dtype)
            encoded_flat = self.state_mlp(masked_fused)
            encoded_flat = encoded_flat * flat_mask.unsqueeze(-1).to(dtype = encoded_flat.dtype)

        encoded = encoded_flat.reshape(B, T, self.feature_dim)
        encoded = self._apply_mask(encoded, time_mask)
        return encoded

    @_dynamo_disable()
    def _encode_sparse_actions_eager(
        self,
        flat_tokens: torch.Tensor,
        flat_mask: torch.Tensor,
    ) -> torch.Tensor:
        BT, A, _ = self._shape(flat_tokens, (None, None, None), "flat_tokens")
        self._shape(flat_mask, (BT, A), "flat_mask")

        legal_indices = flat_mask.nonzero(as_tuple = False)  # [N, 2] -> (row, action)
        row_idx = legal_indices[:, 0]
        act_idx = legal_indices[:, 1]
        gathered_tokens = flat_tokens[row_idx, act_idx]  # [N, L]
        token_embeds = self.action_tokens_embed(gathered_tokens).sum(dim = -2)  # [N, D]
        encoded_sel = self.action_mlp(token_embeds)  # [N, D]

        # Under AMP autocast the MLP may return bf16/fp16 even though embedding weights are fp32.
        encoded_full = encoded_sel.new_zeros((BT, A, self.feature_dim))
        encoded_full[row_idx, act_idx] = encoded_sel
        return encoded_full

    def _encode_action_features(
        self,
        avail_tokens: torch.Tensor,
        avail_weights: torch.Tensor,
        time_mask: torch.Tensor,
        *,
        use_sparse: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self._shape(time_mask, (None, None), "time_mask")
        B, T, A, L = self._shape(avail_tokens, (B, T, None, None), "avail_tokens")
        self._shape(avail_weights, (B, T, A), "avail_weights")
        avail_mask = (avail_weights != 0) & time_mask.unsqueeze(-1)

        flat_tokens = avail_tokens.reshape(B * T, A, L)
        flat_mask = avail_mask.reshape(B * T, A)

        if use_sparse:
            # Sparse path: run the MLP only on legal actions, then scatter back.
            encoded_flat = self._encode_sparse_actions_eager(flat_tokens, flat_mask)
        else:
            # Dense path: run the action MLP on all actions and mask afterwards.
            token_embeds = self.action_tokens_embed(flat_tokens).sum(dim = -2)  # [B*T, A, D]
            token_embeds = token_embeds * flat_mask.unsqueeze(-1).to(dtype = token_embeds.dtype)
            encoded_flat = self.action_mlp(token_embeds.reshape(B * T * A, self.feature_dim)).reshape(
                B * T,
                A,
                self.feature_dim,
            )

        encoded = encoded_flat.reshape(B, T, A, self.feature_dim)
        encoded = self._apply_mask(encoded, avail_mask)
        return encoded, avail_mask


__all__ = ["BaseMarvelModule"]
