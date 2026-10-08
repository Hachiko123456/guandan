# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/model/cache.py.
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


class Cache:
    """Minimal cache interface matching transformers' Cache protocol."""

    is_compileable: bool = False

    def __init__(self) -> None:
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Implement `update` in a Cache subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        del layer_idx
        raise NotImplementedError("Implement `get_seq_length` in a Cache subclass.")

    def get_max_cache_shape(self) -> Optional[int]:
        return None


class DynamicCache(Cache):
    """Dynamically-growing cache used by default autoregressive decoding paths."""

    def __init__(self, _distributed_cache_data: Optional[Iterable] = None) -> None:
        super().__init__()
        self._seen_tokens = 0
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        if _distributed_cache_data is not None:
            for key_states, value_states in _distributed_cache_data:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)

    def __len__(self) -> int:
        return len(self.key_cache)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        raise KeyError(f"Cache only has {len(self)} layers, attempted to access index {layer_idx}.")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.tensor([]))
                self.value_cache.append(torch.tensor([]))
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif not self.key_cache[layer_idx].numel():
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim = -2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        layer_idx = 0 if layer_idx is None else int(layer_idx)
        is_empty = (
            len(self.key_cache) == 0
            or len(self.key_cache) <= layer_idx
            or not self.key_cache[layer_idx].numel()
        )
        return self.key_cache[layer_idx].shape[-2] if not is_empty else 0

    def get_max_cache_shape(self) -> Optional[int]:
        return None

PastKVInput = Union[Sequence[Tuple[torch.Tensor, torch.Tensor]], Cache, None]


class KVCache(Cache):
    """Compile-friendly KV cache for MARVEL fast rollouts.

    This cache stores a fixed-size key/value window per transformer layer so that
    the decoder can be used under `torch.compile` without shape-changing caches.
    """
    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        *,
        max_ctx_len: int,
        num_heads: int,
        head_dim: int,
        value_heads: Optional[int] = None,
        value_head_dim: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device = None,
    ):
        # `transformers.cache_utils.Cache` has changed initialization semantics across
        # transformers releases. Some versions require arguments such as `layers` or
        # `layer_class_to_replicate`, while older versions accept no args. MARVEL's
        # cache implementation does not depend on the base initializer, so we only
        # ensure the `nn.Module` internals are initialized when applicable.
        if issubclass(Cache, nn.Module):
            nn.Module.__init__(self)
        if max_ctx_len < 1:
            raise ValueError("max_ctx_len must be at least 1")

        self.n_layers = int(n_layers)
        self.batch = int(batch_size)
        if self.batch <= 0:
            raise ValueError("batch_size must be positive")
        self.max_ctx_len = int(max_ctx_len)
        self.max_past_len = self.max_ctx_len

        self.num_heads = int(num_heads)
        self.num_value_heads = int(value_heads) if value_heads is not None else self.num_heads
        self.head_dim = int(head_dim)
        self.value_head_dim = int(value_head_dim) if value_head_dim is not None else self.head_dim

        if self.head_dim <= 0 or self.value_head_dim <= 0:
            raise ValueError("head dimensions must be positive")

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

        self._seq_lens = torch.zeros(self.batch, dtype = torch.long, device = self.device)
        self._position_ids = torch.zeros_like(self._seq_lens)
        self._step_mask = torch.zeros_like(self._seq_lens)

        key_shape = (
            self.n_layers,
            self.batch,
            self.num_heads,
            max(self.max_past_len, 1),
            self.head_dim,
        )
        value_shape = (
            self.n_layers,
            self.batch,
            self.num_value_heads,
            max(self.max_past_len, 1),
            self.value_head_dim,
        )

        self.keys = torch.zeros(key_shape, dtype = self.dtype, device = self.device)
        self.values = torch.zeros(value_shape, dtype = self.dtype, device = self.device)

        # Expose per-layer 4D views for transformers Cache utilities.
        self.key_cache = [self.keys[layer_idx] for layer_idx in range(self.n_layers)]
        self.value_cache = [self.values[layer_idx] for layer_idx in range(self.n_layers)]

        # Newer versions of transformers expect `Cache` objects to expose a `layers`
        # collection. We don't use it internally, but define it for compatibility.
        self.layers = [None] * self.n_layers

    def _ensure_column(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if tensor.dim() > 1 else tensor.unsqueeze(-1)

    @property
    def seq_lens(self) -> torch.Tensor:
        return self._seq_lens

    @property
    def batch_size(self) -> int:
        return self.batch

    @property
    def is_compileable(self) -> bool:
        return True

    def set_step_mask(self, new_mask: torch.Tensor) -> None:
        """Set the per-batch mask used by the next decoder forward.

        Entries with a 0 mask value keep their cache state intact and do not
        append a new token during the update calls from the transformer layers.
        """
        new_mask = self._ensure_column(new_mask)
        if new_mask.size(0) != self.batch:
            raise ValueError("new_mask batch dimension does not match the cache")
        if new_mask.size(-1) != 1:
            raise ValueError("new_mask must have a trailing singleton dimension")
        self._step_mask.copy_(new_mask.squeeze(-1).to(device = self.device, dtype = torch.long))

    def next_position_ids(self, new_mask: torch.Tensor) -> torch.Tensor:
        new_mask = self._ensure_column(new_mask)
        if new_mask.size(0) != self.batch:
            raise ValueError("new_mask batch dimension does not match the cache")
        return self._position_ids.unsqueeze(-1).to(device = new_mask.device)

    def get_full_attention_mask(self, new_mask: torch.Tensor) -> torch.Tensor:
        new_mask = self._ensure_column(new_mask)
        if new_mask.size(0) != self.batch:
            raise ValueError("new_mask batch dimension does not match the cache")

        device = self.device
        new_mask = new_mask.to(device = device, dtype = torch.long).squeeze(-1)
        write_len = (self._seq_lens + new_mask).clamp(max = self.max_past_len)

        positions = torch.arange(self.max_past_len, device = device)
        mask = positions.view(1, -1) < write_len.unsqueeze(-1)
        return mask.to(dtype = torch.long)

    def advance(self, new_mask: torch.Tensor) -> None:
        """Advance tracked positions and lengths after a cached decoder step."""
        new_mask = self._ensure_column(new_mask)
        if new_mask.size(0) != self.batch:
            raise ValueError("new_mask batch dimension does not match the cache")
        if new_mask.size(-1) != 1:
            raise ValueError("new_mask must have a trailing singleton dimension")

        incr = new_mask.squeeze(-1).to(device = self.device, dtype = self._seq_lens.dtype)
        self._seq_lens.add_(incr)
        self._seq_lens.clamp_(max = self.max_past_len)
        self._position_ids.add_(incr)
        self._position_ids.clamp_(max = self.max_ctx_len - 1)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # Return a constant to keep the transformers cache_position logic static; the actual
        # per-batch lengths live in `self._seq_lens` and are enforced via attention masks.
        del layer_idx
        return self.max_past_len

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_past_len

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor | None = None,
        layer_idx: int = 0,
    ) -> tuple[int, int]:
        """Return the key/value length and offset used by transformers mask builders.

        Some transformers versions rely on this helper when constructing causal
        masks. MARVEL uses a fixed-size cache window with no offset.
        """
        del cache_position, layer_idx
        return self.max_past_len, 0

    def update(  # type: ignore[override]
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, object]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the layer cache in-place and return the full fixed-size tensors."""
        del cache_kwargs  # Unused; MARVEL uses per-batch lengths instead of cache_position.

        if not (0 <= layer_idx < self.n_layers):
            raise ValueError(f"layer_idx out of range: {layer_idx}")

        key_view = self.key_cache[layer_idx]
        value_view = self.value_cache[layer_idx]

        if key_states.size(0) != self.batch or value_states.size(0) != self.batch:
            raise ValueError("key/value batch dimension does not match the cache")
        if key_states.size(1) != self.num_heads:
            raise ValueError("key head dimension mismatch")
        if value_states.size(1) != self.num_value_heads:
            raise ValueError("value head dimension mismatch")
        if key_states.size(-1) != self.head_dim or value_states.size(-1) != self.value_head_dim:
            raise ValueError("head dimension mismatch")

        if key_states.size(-2) != 1 or value_states.size(-2) != 1:
            raise ValueError("KVCache only supports seq_len=1 cached updates")

        device = key_view.device
        step_mask = self._step_mask.to(device = device, dtype = self._seq_lens.dtype)

        prev_len = self._seq_lens.to(device)
        window = self.max_past_len
        new_count = step_mask
        write_len = torch.clamp(prev_len + new_count, max = window)
        tail_len = torch.clamp(write_len - new_count, min = 0)
        tail_start = torch.clamp(prev_len - tail_len, min = 0)

        prev_keys = key_view.clone()
        prev_values = value_view.clone()
        key_view.zero_()
        value_view.zero_()

        arange_window = torch.arange(window, device = device)
        arange_window_expand = arange_window.unsqueeze(0).expand(self.batch, window)

        tail_mask = arange_window_expand < tail_len.unsqueeze(1)
        tail_source_idx = (tail_start.unsqueeze(1) + arange_window_expand).clamp_min(0)
        tail_source_idx = torch.minimum(
            tail_source_idx,
            torch.clamp(prev_len - 1, min = 0).unsqueeze(1),
        )

        gather_keys_idx = tail_source_idx.unsqueeze(1).unsqueeze(-1).expand(
            self.batch,
            self.num_heads,
            window,
            self.head_dim,
        )
        gather_values_idx = tail_source_idx.unsqueeze(1).unsqueeze(-1).expand(
            self.batch,
            self.num_value_heads,
            window,
            self.value_head_dim,
        )
        tail_keys = torch.gather(prev_keys, dim = -2, index = gather_keys_idx)
        tail_values = torch.gather(prev_values, dim = -2, index = gather_values_idx)
        tail_keys = tail_keys * tail_mask.unsqueeze(1).unsqueeze(-1)
        tail_values = tail_values * tail_mask.unsqueeze(1).unsqueeze(-1)

        key_view.copy_(tail_keys)
        value_view.copy_(tail_values)

        key_states = key_states.to(dtype = key_view.dtype, device = device)
        value_states = value_states.to(dtype = value_view.dtype, device = device)

        masked = step_mask.to(dtype = key_states.dtype).view(self.batch, 1, 1, 1)
        key_states = key_states * masked
        masked_values = step_mask.to(dtype = value_states.dtype).view(self.batch, 1, 1, 1)
        value_states = value_states * masked_values

        dest = tail_len.clamp_max(window - 1)
        dest_keys_idx = dest.view(self.batch, 1, 1, 1).expand(
            self.batch,
            self.num_heads,
            1,
            self.head_dim,
        )
        dest_values_idx = dest.view(self.batch, 1, 1, 1).expand(
            self.batch,
            self.num_value_heads,
            1,
            self.value_head_dim,
        )
        key_view.scatter_add_(-2, dest_keys_idx, key_states)
        value_view.scatter_add_(-2, dest_values_idx, value_states)

        return key_view, value_view

    def populate(self, source: "DynamicCache", seq_lens: torch.Tensor) -> None:
        """Bulk-populate from a :class:`DynamicCache` produced by a full-sequence forward.

        This replaces the step-by-step ``update()`` path used during KV cache
        rebuild, copying all layers in one shot.
        """
        seq_lens = seq_lens.to(device = self.device, dtype = torch.long)
        if seq_lens.size(0) != self.batch:
            raise ValueError("seq_lens batch dimension does not match the cache")
        T_src = source.get_seq_length(0)
        copy_len = min(T_src, self.max_past_len)
        for layer_idx in range(self.n_layers):
            k_src = source.key_cache[layer_idx][:, :, :copy_len, :]
            v_src = source.value_cache[layer_idx][:, :, :copy_len, :]
            self.keys[layer_idx, :, :, :copy_len, :] = k_src.to(dtype = self.dtype)
            self.values[layer_idx, :, :, :copy_len, :] = v_src.to(dtype = self.dtype)

        clamped = seq_lens.clamp(max = self.max_past_len)
        self._seq_lens.copy_(clamped)
        self._position_ids.copy_(seq_lens.clamp(max = self.max_ctx_len - 1))

    def reset(self, reset_mask: torch.Tensor):
        reset_mask = reset_mask.to(device = self.device)
        if reset_mask.dtype != torch.bool:
            reset_mask = reset_mask.to(torch.bool)
        if reset_mask.dim() > 1:
            reset_mask = reset_mask.squeeze(-1)
        if reset_mask.size(0) != self.batch:
            raise ValueError("reset_mask batch dimension does not match the cache")
        if not reset_mask.any():
            return

        self._seq_lens.masked_fill_(reset_mask, 0)
        self._position_ids.masked_fill_(reset_mask, 0)

        reset_kv = reset_mask.view(1, self.batch, 1, 1, 1)
        self.keys.masked_fill_(reset_kv, 0)
        self.values.masked_fill_(reset_kv, 0)


__all__ = ["Cache", "DynamicCache", "KVCache", "PastKVInput"]
