# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/model/trainable.py.
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
from guandan_marvel.optim.common import MultiOptimizer
from guandan_marvel.optim.muon import Muon

if TYPE_CHECKING:
    from guandan_marvel.config import OptimizerConfig


class TrainableModule(nn.Module):
    """Minimal train-time init/optimizer interface layered on top of ``nn.Module``."""

    _wd_param_names: Tuple[str, ...] | None

    @staticmethod
    def _init_module(
        module: nn.Module,
        *,
        pad_token: int | None = None,
        std: float = 1.0,
    ) -> nn.Module:
        weight = getattr(module, "weight", None)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(weight, mean = 0.0, std = std)
            if pad_token is not None:
                with torch.no_grad():
                    weight[pad_token].zero_()
        elif weight is not None:
            if weight.ndim >= 2:
                torch.nn.init.orthogonal_(weight, gain = std)
            else:
                torch.nn.init.normal_(weight, mean = 0.0, std = std)
        if getattr(module, "bias", None) is not None:
            torch.nn.init.zeros_(module.bias)
        return module

    @staticmethod
    def _mark_no_weight_decay(module: nn.Module) -> None:
        for param in module.parameters(recurse = False):
            param._no_weight_decay = True  # type: ignore[attr-defined]

    def _group_params_for_wd(self):
        wd_params, no_wd_params, wd_names = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            lower = name.lower()
            use_wd = (
                not getattr(param, "_no_weight_decay", False)
                and param.dim() >= 2
                and not name.endswith("bias")
                and "norm" not in lower
                and "embed" not in lower
            )
            (wd_params if use_wd else no_wd_params).append(param)
            if use_wd:
                wd_names.append(name)
        return wd_params, no_wd_params, wd_names

    def configure_optimizer(
        self,
        *,
        lr: float,
        optimizer: OptimizerConfig,
    ) -> torch.optim.Optimizer | MultiOptimizer:
        wd_params, no_wd_params, wd_names = self._group_params_for_wd()
        self._wd_param_names = tuple(wd_names)
        if optimizer.use_muon and wd_params:
            muon = Muon(
                wd_params,
                lr = lr,
                momentum = optimizer.muon_beta1,
                weight_decay = optimizer.weight_decay,
                adamw_beta1_for_rms_match = optimizer.adam_beta1,
            )
            if not no_wd_params:
                return muon
            adamw = torch.optim.AdamW(
                [{"params": no_wd_params, "weight_decay": 0.0}],
                lr = lr,
                betas = (optimizer.adam_beta1, optimizer.adam_beta2),
            )
            return MultiOptimizer({"muon": muon, "adamw": adamw})

        return torch.optim.AdamW(
            [
                {"params": wd_params, "weight_decay": optimizer.weight_decay},
                {"params": no_wd_params, "weight_decay": 0.0},
            ],
            lr = lr,
            betas = (optimizer.adam_beta1, optimizer.adam_beta2),
        )

    @property
    def weight_decay_param_names(self) -> Tuple[str, ...]:
        wd_names = getattr(self, "_wd_param_names", None)
        if wd_names is None:
            raise RuntimeError(
                "Optimizer has not been configured; call configure_optimizer() first."
            )
        return wd_names


__all__ = ["TrainableModule"]
