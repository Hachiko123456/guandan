# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/model/legacy.py.
"""V1.0/v1.1 MLP builders for backward compatibility.

These builders are used by v1.0 (legacy) and v1.1 (structured) model
architectures.  New models should prefer v1.2 ``ResGatedMLP`` from
``marvel.model.base``.
"""
from __future__ import annotations

import torch.nn as nn

from .trainable import TrainableModule


def _rms_norm(dim: int) -> nn.Module:
    from .base import RMSNorm
    return RMSNorm(dim)


class _ResidualMLPBlock(nn.Module):
    def __init__(self, layers: nn.Sequential) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, hidden_states):
        return hidden_states + self.layers(hidden_states)


def _build_linear_rms_gelu_block(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        TrainableModule._init_module(nn.Linear(input_dim, output_dim)),
        _rms_norm(output_dim),
        nn.GELU(),
    )


def _build_residual_mlp_block(dim: int, *, repeats: int = 4) -> nn.Module:
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}.")

    layers: list[nn.Module] = [
        _build_linear_rms_gelu_block(dim, dim)
        for _ in range(repeats)
    ]
    return _ResidualMLPBlock(nn.Sequential(*layers))


def _build_residual_blocks(
    dim: int,
    *,
    n_layers: int,
    module_name: str = "model.mlp_layers",
    repeats: int = 4,
) -> list[nn.Module]:
    n_layers = int(n_layers)
    if n_layers < 0 or n_layers % repeats != 0:
        raise ValueError(
            f"{module_name} must match {repeats}n for v1.1+ architectures; got {n_layers}."
        )
    block_count = n_layers // repeats
    return [
        _build_residual_mlp_block(dim, repeats = repeats)
        for _ in range(block_count)
    ]


def _build_structured_mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    n_layers: int,
    output_dim: int,
) -> nn.Sequential:
    n_layers = int(n_layers)
    residual_blocks = _build_residual_blocks(
        hidden_dim,
        n_layers = n_layers - 2,
        repeats = 4,
    )
    layers: list[nn.Module] = [
        _build_linear_rms_gelu_block(input_dim, hidden_dim),
        *residual_blocks,
        TrainableModule._init_module(nn.Linear(hidden_dim, output_dim)),
    ]
    return nn.Sequential(*layers)


def _build_legacy_mlp(
    input_dim: int,
    hidden_dim: int,
    n_layers: int,
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for _ in range(n_layers - 1):
        layers.append(TrainableModule._init_module(nn.Linear(prev_dim, hidden_dim)))
        layers.append(_rms_norm(hidden_dim))
        layers.append(nn.GELU())
        prev_dim = hidden_dim

    layers.append(TrainableModule._init_module(nn.Linear(prev_dim, output_dim)))
    layers.append(_rms_norm(output_dim))
    return nn.Sequential(*layers)


__all__ = [
    "_ResidualMLPBlock",
    "_build_legacy_mlp",
    "_build_linear_rms_gelu_block",
    "_build_residual_blocks",
    "_build_residual_mlp_block",
    "_build_structured_mlp",
]
