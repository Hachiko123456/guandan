# SPDX-License-Identifier: MIT
"""Lazy public API for the Guandan MARVEL runtime.

Importing :mod:`guandan_marvel` is intentionally extension-free.  The public
backend contract is:

- ``default_game`` or ``get_game_backend()``: configured default (C++).
- ``python_game`` or ``get_game_backend("python_reference")``: Python oracle.
- ``cpp_game`` or ``get_game_backend("cpp")``: explicit C++ runtime.
- ``game`` / ``import guandan_marvel.game``: historical Python-oracle module.

The historical ``game`` name therefore has one identity regardless of import
syntax.  No selector silently falls back between backends.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BufferConfig": ("guandan_marvel.buffer", "BufferConfig"),
    "BufferFieldSpec": ("guandan_marvel.buffer", "BufferFieldSpec"),
    "TrajectoryBuffer": ("guandan_marvel.buffer", "TrajectoryBuffer"),
    "trajectory_field_specs": ("guandan_marvel.buffer", "trajectory_field_specs"),
    "trajectory_memory_budget": ("guandan_marvel.buffer", "trajectory_memory_budget"),
    "require_trajectory_memory_budget": ("guandan_marvel.buffer", "require_trajectory_memory_budget"),
    "CompileConfig": ("guandan_marvel.config", "CompileConfig"),
    "DDPConfig": ("guandan_marvel.config", "DDPConfig"),
    "GameSpec": ("guandan_marvel.config", "GameSpec"),
    "ModelConfig": ("guandan_marvel.config", "ModelConfig"),
    "OptimizerConfig": ("guandan_marvel.config", "OptimizerConfig"),
    "TrainConfig": ("guandan_marvel.config", "TrainConfig"),
    "deserialize_train_config": ("guandan_marvel.config", "deserialize_train_config"),
    "Agent": ("guandan_marvel.model", "Agent"),
    "CentralCritic": ("guandan_marvel.model", "CentralCritic"),
    "TensorDict": ("guandan_marvel.model", "TensorDict"),
    "IPPOTrainer": ("guandan_marvel.training", "IPPOTrainer"),
    "VRPOTrainer": ("guandan_marvel.training", "VRPOTrainer"),
    "PythonReferenceSmokeGame": ("guandan_marvel.training", "PythonReferenceSmokeGame"),
    "DEFAULT_BACKEND": ("guandan_marvel.backends", "DEFAULT_BACKEND"),
    "SUPPORTED_BACKENDS": ("guandan_marvel.backends", "SUPPORTED_BACKENDS"),
    "BackendUnavailableError": ("guandan_marvel.backends", "BackendUnavailableError"),
    "get_game_backend": ("guandan_marvel.backends", "get_game_backend"),
}


def __getattr__(name: str) -> Any:
    if name == "game":
        value = import_module("guandan_marvel.game")
    elif name == "python_game":
        value = import_module("guandan_marvel.python_game")
    elif name == "cpp_game":
        value = import_module("guandan_marvel.cpp_game")
    elif name == "default_game":
        selector = import_module("guandan_marvel.backends").get_game_backend
        value = selector()
    elif name in _LAZY_EXPORTS:
        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attribute)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "Agent", "BackendUnavailableError", "BufferConfig", "BufferFieldSpec",
    "CentralCritic", "CompileConfig", "DDPConfig", "DEFAULT_BACKEND",
    "GameSpec", "IPPOTrainer", "ModelConfig", "OptimizerConfig",
    "PythonReferenceSmokeGame", "SUPPORTED_BACKENDS", "TensorDict",
    "TrainConfig", "TrajectoryBuffer", "VRPOTrainer", "cpp_game",
    "default_game", "deserialize_train_config", "game", "get_game_backend",
    "python_game", "require_trajectory_memory_budget", "trajectory_field_specs",
    "trajectory_memory_budget",
]
