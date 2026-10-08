# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/model/transformer.py.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from .cache import Cache, DynamicCache


class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 0,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict[str, Any]] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        head_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = (
            int(num_attention_heads) if num_key_value_heads is None else int(num_key_value_heads)
        )
        self.hidden_act = str(hidden_act)
        self.max_position_embeddings = int(max_position_embeddings)
        self.initializer_range = float(initializer_range)
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_cache = bool(use_cache)
        self.pad_token_id = pad_token_id if pad_token_id is None else int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.pretraining_tp = int(pretraining_tp)
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.mlp_bias = bool(mlp_bias)
        self.head_dim = (
            int(self.hidden_size // self.num_attention_heads) if head_dim is None else int(head_dim)
        )
        self.use_gated_attention = bool(kwargs.pop("use_gated_attention", False))

        self.output_hidden_states = bool(kwargs.pop("output_hidden_states", False))
        self.return_dict = bool(kwargs.pop("return_dict", True))
        self.use_return_dict = bool(kwargs.pop("use_return_dict", self.return_dict))

        # Silently consume legacy keys that no longer apply.
        kwargs.pop("attn_implementation", None)
        kwargs.pop("output_attentions", None)

        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


class TransformerRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim = -1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _compute_default_rope_parameters(
    config: TransformerConfig,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    base = float(config.rope_theta)
    dim = int(getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads))
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype = torch.int64, device = device).float() / dim)
    )
    attention_factor = 1.0
    return inv_freq, attention_factor


class TransformerRotaryEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig, device = None) -> None:
        super().__init__()
        if getattr(config, "rope_scaling", None) is not None:
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            if rope_type not in (None, "default"):
                raise NotImplementedError(f"Unsupported rope_type={rope_type!r}.")
        self.rope_type = "default"
        self.max_seq_len_cached = int(config.max_position_embeddings)
        self.original_max_seq_len = int(config.max_position_embeddings)
        self.config = config

        inv_freq, self.attention_scaling = _compute_default_rope_parameters(config, device = device)
        self.register_buffer("inv_freq", inv_freq, persistent = False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type = device_type, enabled = False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim = -1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype = x.dtype), sin.to(dtype = x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TransformerMLP(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``.

    All internal Linears are initialized with ``normal_(std=INITIALIZER_STD)``
    and zero biases, matching the standard transformer initialization convention.
    """

    INITIALIZER_STD: float = 0.02

    @staticmethod
    def _init_linear(module: nn.Linear, *, std: float) -> nn.Linear:
        torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        return module

    def __init__(self, dim: int, intermediate_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = self._init_linear(nn.Linear(dim, intermediate_dim, bias = bias), std = self.INITIALIZER_STD)
        self.up_proj = self._init_linear(nn.Linear(dim, intermediate_dim, bias = bias), std = self.INITIALIZER_STD)
        self.down_proj = self._init_linear(nn.Linear(intermediate_dim, dim, bias = bias), std = self.INITIALIZER_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return expanded.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class TransformerAttention(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.num_key_value_groups = int(config.num_attention_heads // config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias = config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias = config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias = config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias = config.attention_bias,
        )
        self.gate_proj: nn.Linear | None = None
        if bool(getattr(config, "use_gated_attention", False)):
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias = True,
            )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        dropout = 0.0 if not self.training else self.attention_dropout

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask = causal_mask,
            dropout_p = dropout,
            scale = self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        if self.gate_proj is not None:
            gates = torch.sigmoid(self.gate_proj(hidden_states).view(*input_shape, -1, self.head_dim))
            attn_output = attn_output * gates.to(dtype = attn_output.dtype)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.self_attn = TransformerAttention(config = config, layer_idx = layer_idx)
        self.mlp = TransformerMLP(
            int(config.hidden_size),
            int(config.intermediate_size),
            bias = config.mlp_bias,
        )
        self.input_layernorm = TransformerRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = TransformerRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del position_ids, use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided by TransformerModel.")

        hidden_states = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            past_key_value = past_key_value,
            cache_position = cache_position,
            position_embeddings = position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = int(config.vocab_size)

        self.embed_tokens: nn.Embedding | None = None
        if self.vocab_size > 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = TransformerRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.rotary_emb = TransformerRotaryEmbedding(config = config)
        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, TransformerRMSNorm):
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self) -> nn.Embedding | None:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding | None) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[BaseModelOutputWithPast, Tuple[torch.Tensor, Optional[Cache]]]:
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else bool(output_hidden_states)
        )
        use_cache = self.config.use_cache if use_cache is None else bool(use_cache)
        if return_dict is None:
            return_dict = bool(self.config.return_dict)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("past_key_values must be a Cache instance or None.")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "input_ids requires a TransformerModel configured with input embeddings."
                )
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device = inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask = causal_mask,
                position_ids = position_ids,
                past_key_value = past_key_values,
                use_cache = use_cache,
                cache_position = cache_position,
                position_embeddings = position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_dict:
            return BaseModelOutputWithPast(
                last_hidden_state = hidden_states,
                past_key_values = past_key_values if use_cache else None,
                hidden_states = all_hidden_states,
            )

        return hidden_states, past_key_values if use_cache else None

    def _update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
    ) -> Optional[torch.Tensor]:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        if target_length is None:
            target_length = past_seen_tokens + sequence_length

        return self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length = sequence_length,
            target_length = int(target_length),
            dtype = dtype,
            cache_position = cache_position,
            batch_size = input_tensor.shape[0],
        )

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: Optional[torch.Tensor],
        *,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value = min_dtype,
            dtype = dtype,
            device = cache_position.device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal = 1)
        causal_mask = causal_mask * (
            torch.arange(target_length, device = cache_position.device) > cache_position.reshape(-1, 1)
        )
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                causal_mask.device
            )
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask,
                min_dtype,
            )

        return causal_mask


__all__ = [
    "TransformerConfig",
    "TransformerModel",
]
