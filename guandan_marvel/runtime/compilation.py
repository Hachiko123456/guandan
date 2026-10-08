# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/runtime/compilation.py.
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable

import torch
import torch.nn as nn

from guandan_marvel.config import CompileConfig, TrainConfig

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class TorchCompileSettings:
    enabled: bool = False
    backend: str = "inductor"
    dynamic_shapes: bool = False

    @classmethod
    def from_cfg(cls, cfg: TrainConfig | CompileConfig) -> "TorchCompileSettings":
        raw = cfg.compile if isinstance(cfg, TrainConfig) else cfg
        return cls(bool(raw.enabled), str(raw.backend), bool(raw.dynamic_shapes))


class ModuleCompiler:
    """Explicit torch.compile gate; unavailable/invalid requests fail closed."""

    def __init__(self, settings: TorchCompileSettings, *, log: LogFn | None = None) -> None:
        self.settings = settings
        self._log = log
        self._logged = False

    @classmethod
    def from_cfg(cls, cfg: TrainConfig | CompileConfig, *, log: LogFn | None = None) -> "ModuleCompiler":
        return cls(TorchCompileSettings.from_cfg(cfg), log=log)

    def compile(self, module: nn.Module) -> nn.Module:
        if not self.settings.enabled:
            return module
        compiler = getattr(torch, "compile", None)
        if not callable(compiler):
            raise RuntimeError("torch.compile is unavailable; explicitly disable compile or install torch>=2.0.")
        if not self.settings.backend.strip():
            raise ValueError("compile backend must be non-empty when compile is enabled.")
        parameters = inspect.signature(compiler).parameters
        kwargs: dict[str, object] = {"backend": self.settings.backend}
        if "dynamic" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            kwargs["dynamic"] = self.settings.dynamic_shapes
        compiled = compiler(module, **kwargs)
        if self._log is not None and not self._logged:
            self._log(f"[Compile] enabled backend={self.settings.backend} dynamic={self.settings.dynamic_shapes}")
            self._logged = True
        return compiled


__all__ = ["ModuleCompiler", "TorchCompileSettings"]
