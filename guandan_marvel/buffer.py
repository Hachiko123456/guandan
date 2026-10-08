# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/buffer.py.
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from guandan_marvel.config import GameSpec
from guandan_marvel.model.tensordict import TensorDict

FieldFactory = Callable[[int, torch.device], torch.Tensor]
FieldSpecLike = FieldFactory | "BufferFieldSpec" | torch.Tensor
BUFFER_STATE_SCHEMA = "guandan-marvel-buffer-v1"


def trajectory_field_specs(game: GameSpec | object) -> dict[str, FieldSpecLike]:
    spec = game if isinstance(game, GameSpec) else GameSpec.from_game(game)
    t, a, p = spec.max_game_horizon, spec.max_available_actions, spec.num_players
    l_obs, l_act, channels = spec.max_observation_tokens, spec.max_action_tokens, spec.num_channels
    return {
        "obs_tokens": lambda b, device: torch.zeros((b, t, p, l_obs), dtype=torch.long, device=device),
        "obs_channels": lambda b, device: torch.zeros((b, t, p, channels), dtype=torch.uint8, device=device),
        "player_ids": lambda b, device: torch.full((b, t), -1, dtype=torch.long, device=device),
        "avail_tokens": lambda b, device: torch.zeros((b, t, a, l_act), dtype=torch.long, device=device),
        "avail_weights": lambda b, device: torch.zeros((b, t, a), device=device),
        "avail_is_commit": lambda b, device: torch.zeros((b, t, a), dtype=torch.bool, device=device),
        "avail_probs": lambda b, device: torch.zeros((b, t, a), device=device),
        "action_taken": lambda b, device: torch.zeros((b, t), dtype=torch.long, device=device),
        "rewards": lambda b, device: torch.zeros((b, t, p), device=device),
        "is_terminal": lambda b, device: torch.zeros((b, t), dtype=torch.bool, device=device),
    }


@dataclass(frozen=True)
class BufferConfig:
    game: GameSpec | object
    device: torch.device | str
    field_specs: Mapping[str, FieldSpecLike] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "game", self.game if isinstance(self.game, GameSpec) else GameSpec.from_game(self.game))
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(self, "field_specs", dict(self.field_specs))


@dataclass(frozen=True)
class BufferFieldSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    fill: float | int | bool = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))


class TrajectoryBuffer:
    """TensorDict trajectory accumulator and bounded cyclic replay buffer."""

    def __init__(self, batch_size: int, config: BufferConfig):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        self.config = config
        self.device = torch.device(config.device)
        self.batch_size = int(batch_size)
        specs = trajectory_field_specs(config.game)
        specs.update(config.field_specs)
        self._field_factories: dict[str, FieldFactory] = {
            "num_steps": lambda b, device: torch.zeros((b,), dtype=torch.long, device=device),
            **{name: self._make_field_factory(value) for name, value in specs.items()},
        }
        self._storage = self._allocate(self.batch_size, self.device)
        self._ptr = 0
        self._size = 0

    @staticmethod
    def _make_field_factory(spec: FieldSpecLike) -> FieldFactory:
        if callable(spec):
            return spec
        if isinstance(spec, BufferFieldSpec):
            return lambda batch, device, s=spec: torch.full((batch, *s.shape), s.fill, dtype=s.dtype, device=device)
        if isinstance(spec, torch.Tensor):
            prototype = spec.detach().clone()
            return lambda batch, device, p=prototype: p.to(device).expand((batch, *p.shape)).clone()
        raise TypeError("Field specification must be callable, BufferFieldSpec, or tensor prototype.")

    def _allocate(self, batch_size: int, device: torch.device) -> TensorDict:
        fields = {name: factory(batch_size, device) for name, factory in self._field_factories.items()}
        return TensorDict(fields, batch_size=[batch_size], device=device)

    def __getitem__(self, item):
        return self._storage[item]

    def __setitem__(self, item, value) -> None:
        """Assign trajectory batches through the underlying TensorDict.

        The complete-trajectory sampler uses slice assignment when moving
        finished rollouts into its commit buffer.  Keeping the operation here
        preserves TensorDict shape/key validation instead of bypassing it.
        """
        self._storage[item] = value

    def __len__(self) -> int:
        return self._size

    @property
    def storage(self) -> TensorDict:
        return self._storage

    def add_step(self, batch_indices: torch.Tensor, transition: TensorDict) -> None:
        batch_indices = batch_indices.to(device=self.device, dtype=torch.long)
        if transition.batch_size != torch.Size([batch_indices.numel()]):
            raise ValueError("Transition batch size must match batch_indices.")
        time_indices = self._storage["num_steps"][batch_indices]
        horizon = self.config.game.max_game_horizon
        if bool((time_indices >= horizon).any()):
            raise OverflowError("TrajectoryBuffer horizon exceeded; refusing to truncate silently.")
        for key, value in transition.to(self.device).items():
            if key not in self._storage:
                raise KeyError(f"Unregistered trajectory field '{key}'.")
            self._storage[key][batch_indices, time_indices] = value
        self._storage["num_steps"][batch_indices] = time_indices + 1

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            self._storage = self._allocate(self.batch_size, self.device)
            self._ptr = 0
            self._size = 0
            return
        mask = mask.to(device=self.device, dtype=torch.bool)
        if mask.shape != (self.batch_size,):
            raise ValueError("Mask shape must equal [buffer batch_size].")
        count = int(mask.sum().item())
        if count:
            self._storage[mask] = self._allocate(count, self.device)

    def empty(self, *, batch_size: int | None = None, device: torch.device | str | None = None) -> TensorDict:
        return self._allocate(batch_size or self.batch_size, torch.device(device or self.device))

    def add(self, batch: TensorDict) -> None:
        count = int(batch.batch_size[0])
        if count > self.batch_size:
            raise ValueError("Cannot add more trajectories than buffer capacity.")
        indices = torch.arange(self._ptr, self._ptr + count, device=self.device) % self.batch_size
        self._storage[indices] = batch.to(self.device)
        self._ptr = (self._ptr + count) % self.batch_size
        self._size = min(self.batch_size, self._size + count)

    def sample(self, batch_size: int, *, keys: tuple[str, ...] | None = None) -> TensorDict:
        if self._size == 0:
            raise ValueError("Cannot sample an empty buffer.")
        if int(batch_size) > self._size:
            raise ValueError("Cannot sample more trajectories than are stored.")
        indices = torch.randperm(self._size, device=self.device)[: int(batch_size)]
        source = self._storage if keys is None else self._storage.select(*keys)
        return source[indices]

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": BUFFER_STATE_SCHEMA,
            "capacity": self.batch_size,
            "ptr": self._ptr,
            "size": self._size,
            "storage": {key: value.detach().cpu() for key, value in self._storage.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema") != BUFFER_STATE_SCHEMA:
            raise ValueError("Unsupported or missing trajectory buffer state schema.")
        if int(state["capacity"]) != self.batch_size:
            raise ValueError("Buffer capacity mismatch during restore.")
        raw_storage = state.get("storage")
        if not isinstance(raw_storage, Mapping):
            raise TypeError("Buffer state storage must be a mapping.")
        expected = set(self._storage.keys())
        if set(raw_storage) != expected:
            raise ValueError("Buffer state fields do not match the configured buffer.")
        for key, target in self._storage.items():
            value = torch.as_tensor(raw_storage[key], device=self.device)
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Buffer field '{key}' shape/dtype mismatch.")
            target.copy_(value)
        self._ptr = int(state["ptr"])
        self._size = int(state["size"])
        if not 0 <= self._ptr < self.batch_size or not 0 <= self._size <= self.batch_size:
            raise ValueError("Invalid replay pointer/size in buffer state.")


__all__ = ["BUFFER_STATE_SCHEMA", "BufferConfig", "BufferFieldSpec", "FieldSpecLike", "TrajectoryBuffer", "trajectory_field_specs"]
