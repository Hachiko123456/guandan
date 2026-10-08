# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/model/tensordict.py.
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import Any

import torch


class TensorDict:
    """Small torch-only tensor mapping with explicit leading batch dimensions."""

    def __init__(self, data: Mapping[str, Any], *, batch_size: Iterable[int] | torch.Size, device: torch.device | str | None = None) -> None:
        object.__setattr__(self, "_batch_size", torch.Size(batch_size))
        resolved_device = torch.device(device) if device is not None else None
        converted: dict[str, torch.Tensor] = {}
        for key, value in data.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if resolved_device is not None:
                tensor = tensor.to(resolved_device)
            self._validate_tensor(str(key), tensor)
            converted[str(key)] = tensor
        object.__setattr__(self, "_data", converted)
        inferred_device = next(iter(converted.values())).device if converted else (resolved_device or torch.device("cpu"))
        object.__setattr__(self, "_device", inferred_device)

    def _validate_tensor(self, key: str, tensor: torch.Tensor) -> None:
        dims = len(self._batch_size)
        if tensor.dim() < dims or torch.Size(tensor.shape[:dims]) != self._batch_size:
            raise ValueError(f"TensorDict field '{key}' leading shape {tuple(tensor.shape[:dims])} does not match batch_size {tuple(self._batch_size)}.")

    @property
    def batch_size(self) -> torch.Size: return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def keys(self): return self._data.keys()
    def items(self): return self._data.items()
    def values(self): return self._data.values()
    def get(self, key: str, default: Any = None) -> Any: return self._data.get(key, default)

    def set(self, key: str, value: Any) -> "TensorDict":
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        self._validate_tensor(key, tensor)
        if self._data and tensor.device != self.device:
            raise ValueError(f"TensorDict field '{key}' must be on device {self.device}, got {tensor.device}.")
        self._data[key] = tensor
        return self

    def set_(self, key: str, value: Any) -> "TensorDict": return self.set(key, value)

    def select(self, *keys: str) -> "TensorDict":
        return TensorDict({key: self._data[key] for key in keys}, batch_size=self._batch_size, device=self.device)

    def to(self, *args: Any, **kwargs: Any) -> "TensorDict":
        converted = {key: value.to(*args, **kwargs) for key, value in self._data.items()}
        return TensorDict(converted, batch_size=self._batch_size)

    def clone(self) -> "TensorDict":
        return TensorDict({key: value.clone() for key, value in self._data.items()}, batch_size=self._batch_size)

    def detach(self) -> "TensorDict":
        return TensorDict({key: value.detach() for key, value in self._data.items()}, batch_size=self._batch_size)

    def _indexed_batch_size(self, index: Any) -> torch.Size:
        probe_device = index.device if isinstance(index, torch.Tensor) else torch.device("cpu")
        probe = torch.empty(self._batch_size, device=probe_device)[index]
        return torch.Size(probe.shape)

    def _slice_batch(self, index: Any) -> "TensorDict":
        sliced = {key: value[index] for key, value in self._data.items()}
        return TensorDict(sliced, batch_size=self._indexed_batch_size(index), device=self.device)

    def _assign_batch(self, index: Any, other: "TensorDict") -> None:
        if not isinstance(other, TensorDict):
            raise TypeError("Assigned value must be TensorDict.")
        if set(other.keys()) != set(self.keys()):
            raise KeyError("Assigned TensorDict fields must exactly match target fields.")
        for key, value in other.items():
            self._data[key][index] = value

    def __getitem__(self, item: Any) -> Any:
        return self._data[item] if isinstance(item, str) else self._slice_batch(item)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            self.set(key, value)
        elif isinstance(value, TensorDict):
            self._assign_batch(key, value)
        elif isinstance(value, MutableMapping):
            self._assign_batch(key, TensorDict(value, batch_size=self._indexed_batch_size(key), device=self.device))
        else:
            raise TypeError("Assigned value must be TensorDict or mapping.")

    def __contains__(self, key: object) -> bool: return key in self._data
    def __iter__(self) -> Iterator[str]: return iter(self._data)
    def __len__(self) -> int: return len(self._data)

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"TensorDict has no attribute '{name}'.")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self.set(name, value)

    def __repr__(self) -> str:
        fields = ", ".join(f"{key}={tuple(value.shape)}" for key, value in self._data.items())
        return f"TensorDict(batch_size={tuple(self.batch_size)}, {fields})"


__all__ = ["TensorDict"]
