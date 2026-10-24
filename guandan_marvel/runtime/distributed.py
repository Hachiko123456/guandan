# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/runtime/distributed.py.
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from guandan_marvel.config import DDPConfig


@dataclass(frozen=True)
class DistributedContext:
    is_initialized: bool
    is_primary: bool
    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    object_group: Any = field(default=None, repr=False, compare=False)

    @property
    def enabled(self) -> bool:
        return self.is_initialized and self.world_size > 1

    @classmethod
    def disabled(cls, device: torch.device | str = "cpu") -> "DistributedContext":
        return cls(False, True, 1, 0, 0, torch.device(device))

    def wrap_model(self, model: torch.nn.Module, *, find_unused_parameters: bool = False) -> torch.nn.Module:
        if not self.enabled:
            return model
        if self.device.type == "cuda":
            return DDP(
                model,
                device_ids=[self.device.index],
                output_device=self.device,
                find_unused_parameters=find_unused_parameters,
            )
        if dist.get_backend() != "gloo":
            raise RuntimeError("CPU DDP is allowed only with the explicit gloo test backend.")
        return DDP(model, find_unused_parameters=find_unused_parameters)

    def _reduce(self, value: float | int | torch.Tensor | None, *, average: bool):
        if value is None:
            return None
        tensor = value.detach().clone() if isinstance(value, torch.Tensor) else torch.tensor(value)
        target = self.device if self.enabled and dist.get_backend() == "nccl" else torch.device("cpu")
        tensor = tensor.to(target)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if average:
                tensor = tensor / self.world_size
        result = tensor.item()
        return float(result) if average or tensor.is_floating_point() else int(result)

    def reduce_mean(self, value):
        return self._reduce(value, average=True)

    def reduce_sum(self, value):
        return self._reduce(value, average=False)

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def with_cpu_object_collectives(self) -> "DistributedContext":
        """Use a Gloo side group for metadata only; DDP tensors stay on NCCL.

        Every rank must call in the same order, before any metadata exchange.
        The default group/model device are deliberately not changed.
        """
        if not self.enabled or self.object_group is not None:
            return self
        if not dist.is_gloo_available():
            raise RuntimeError("Gloo metadata backend unavailable; refusing implicit fallback")
        return replace(self, object_group=dist.new_group(backend="gloo"))

    @property
    def object_collective_backend(self) -> str:
        if not self.enabled:
            return "disabled"
        return str(dist.get_backend(self.object_group)) if self.object_group is not None else str(dist.get_backend())

    def all_gather_object(self, value: object) -> list[object]:
        if not self.enabled:
            return [value]
        gathered: list[object] = [None for _ in range(self.world_size)]
        if self.object_group is None:
            dist.all_gather_object(gathered, value)
        else:
            dist.all_gather_object(gathered, value, group=self.object_group)
        return gathered


class NullProgressBar:
    def set_postfix(self, **_kwargs: Any) -> None: pass
    def update(self, *_args: Any, **_kwargs: Any) -> None: pass
    def close(self) -> None: pass


class DistributedRuntime:
    def __init__(self, context: DistributedContext) -> None:
        self.context = context

    @property
    def enabled(self) -> bool:
        return self.context.enabled

    @property
    def is_primary(self) -> bool:
        return self.context.is_primary

    @property
    def world_size(self) -> int:
        return self.context.world_size

    @property
    def rank(self) -> int:
        return self.context.rank

    @property
    def local_rank(self) -> int:
        return self.context.local_rank

    @property
    def device(self) -> torch.device:
        return self.context.device

    @classmethod
    def from_config(cls, cfg: DDPConfig, *, device: str) -> "DistributedRuntime":
        env_world = int(os.environ.get("WORLD_SIZE", "1"))
        if not cfg.enabled:
            if env_world > 1:
                raise RuntimeError("WORLD_SIZE>1 but DDP is explicitly disabled; refusing partial-rank execution.")
            return cls(DistributedContext.disabled(device))
        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"DDP enabled but torchrun environment is incomplete: {', '.join(missing)}.")
        if cfg.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("DDP enabled with require_cuda=True, but CUDA is unavailable.")
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size <= 1:
            raise RuntimeError("DDP enabled but WORLD_SIZE is not greater than one.")
        rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        resolved = torch.device("cuda", local_rank) if cfg.require_cuda else torch.device(device)
        if resolved.type == "cuda":
            torch.cuda.set_device(resolved)
        backend = cfg.backend
        if backend == "nccl" and resolved.type != "cuda":
            raise RuntimeError("NCCL backend requires CUDA; no backend fallback is performed.")
        if resolved.type == "cpu" and backend != "gloo":
            raise RuntimeError("CPU DDP requires an explicit gloo backend; no backend fallback is performed.")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        context = DistributedContext(True, rank == 0, dist.get_world_size(), dist.get_rank(), local_rank, resolved)
        if context.world_size != world_size:
            raise RuntimeError("torchrun WORLD_SIZE and initialized process-group size differ")
        return cls(context)

    @classmethod
    def from_env(cls, cfg_device: str) -> "DistributedRuntime":
        """Compatibility path: only initializes when an explicit torchrun world exists."""
        enabled = int(os.environ.get("WORLD_SIZE", "1")) > 1
        require_cuda = torch.device(cfg_device).type == "cuda"
        backend = "nccl" if require_cuda else "gloo"
        return cls.from_config(DDPConfig(enabled=enabled, backend=backend, require_cuda=require_cuda), device=cfg_device)

    def close(self) -> None:
        if dist.is_initialized():
            if self.context.object_group is not None:
                dist.destroy_process_group(self.context.object_group)
            dist.destroy_process_group()


__all__ = ["DistributedContext", "DistributedRuntime", "NullProgressBar"]
