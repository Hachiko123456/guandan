"""Explicit MARVEL game backend selection.

M05 acceptance makes the C++ runtime the training default while retaining the
Python adapter as an explicit, never-deleted differential oracle.  Selection
never falls back silently.
"""
from __future__ import annotations

import importlib
from types import ModuleType

DEFAULT_BACKEND = "cpp"
SUPPORTED_BACKENDS = ("cpp", "python_reference")


def get_game_backend(name: str = DEFAULT_BACKEND) -> ModuleType:
    normalized = str(name).strip().lower()
    if normalized == "python_reference":
        return importlib.import_module("guandan_marvel.game")
    if normalized == "cpp":
        try:
            return importlib.import_module("guandan_marvel.cpp_game")
        except ImportError as error:
            raise RuntimeError(
                "the requested cpp backend is unavailable; no Python fallback is permitted"
            ) from error
    raise ValueError(f"unsupported game backend {name!r}; expected one of {SUPPORTED_BACKENDS}")


__all__ = ["DEFAULT_BACKEND", "SUPPORTED_BACKENDS", "get_game_backend"]
