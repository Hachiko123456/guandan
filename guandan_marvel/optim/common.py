# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/optim/common.py.
from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from torch.optim import Optimizer

from guandan_marvel.utils import unwrap_distributed_module


class MultiOptimizer:
    def __init__(self, optimizers: Mapping[str, Optimizer]) -> None:
        if not optimizers:
            raise ValueError("MultiOptimizer requires at least one optimizer.")
        self.optimizers = dict(optimizers)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for optimizer in self.optimizers.values():
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(*args, **kwargs)

    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers.values():
            optimizer.step()

        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        optim_states = state_dict.get("optimizers", state_dict)
        for name, optimizer in self.optimizers.items():
            if name not in optim_states:
                raise KeyError(f"Missing optimizer state for '{name}'.")
            optimizer.load_state_dict(optim_states[name])


class ExponentialMovingAverageUpdater:
    """Keeps an exponential moving-average model across iterations."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        merged_model: torch.nn.Module,
        beta: float,
    ) -> None:
        self.model = model
        self.merged_model = merged_model
        self.beta = float(beta)
        if not (0.0 <= self.beta < 1.0):
            raise ValueError(f"beta must satisfy 0 <= beta < 1; got {beta}.")

        self.last_iteration = 0
        self.update_count = 0
        self.total_weight = 0.0
        self.sync()

    def _weight_sum_through(self, iteration: int) -> float:
        if iteration <= 0:
            return 0.0
        if self.beta == 0.0:
            return 1.0
        return float((1.0 - self.beta**iteration) / (1.0 - self.beta))

    def set_reference_iteration(self, iteration: int) -> None:
        value = int(iteration)
        if value < 0:
            raise ValueError(f"reference iteration must be >= 0; got {iteration}.")
        self.last_iteration = value
        self.update_count = value
        self.total_weight = self._weight_sum_through(value)

    def sync(self) -> None:
        source_model = unwrap_distributed_module(self.model)
        target_model = unwrap_distributed_module(self.merged_model)
        target_model.load_state_dict(source_model.state_dict(), strict = True)

    def step(self, *, iteration: int) -> None:
        value = int(iteration)
        if value <= 0:
            raise ValueError(f"iteration must be positive; got {iteration}.")
        if value <= self.last_iteration:
            raise ValueError(
                f"iteration must increase monotonically; got {value} after {self.last_iteration}."
            )

        source_model = unwrap_distributed_module(self.model)
        target_model = unwrap_distributed_module(self.merged_model)
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()

        source_keys = set(source_state.keys())
        target_keys = set(target_state.keys())
        if source_keys != target_keys:
            missing = sorted(source_keys - target_keys)
            extra = sorted(target_keys - source_keys)
            raise ValueError(
                "merged/model state_dict keys mismatch. "
                f"Missing merged keys: {missing}; Extra merged keys: {extra}."
            )

        previous_weight = float(self.total_weight)
        total_weight = self.beta * previous_weight + 1.0
        prev_scale = 0.0 if previous_weight <= 0.0 else (self.beta * previous_weight) / total_weight
        current_scale = 1.0 / total_weight

        with torch.no_grad():
            for key, source_value in source_state.items():
                current = source_value.detach()
                merged_value = target_state[key]
                if merged_value.dtype.is_floating_point:
                    merged_value.mul_(prev_scale).add_(current, alpha = current_scale)
                else:
                    merged_value.copy_(current)

        self.last_iteration = value
        self.update_count += 1
        self.total_weight = total_weight


class IterationWeightedAverageUpdater(ExponentialMovingAverageUpdater):
    """Backward-compatible alias for ExponentialMovingAverageUpdater."""


def apply_learning_rate(optimizer: Optimizer | MultiOptimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


__all__ = [
    "ExponentialMovingAverageUpdater",
    "IterationWeightedAverageUpdater",
    "MultiOptimizer",
    "apply_learning_rate",
]
