"""Explicit, lazy MARVEL game-backend selection."""
from __future__ import annotations

import importlib
from types import ModuleType

DEFAULT_BACKEND = "cpp"
SUPPORTED_BACKENDS = ("cpp", "python_reference")


class BackendUnavailableError(RuntimeError):
    """The explicitly requested backend cannot be imported."""


def get_game_backend(name: str = DEFAULT_BACKEND) -> ModuleType:
    normalized = str(name).strip().lower()
    if normalized == "python_reference":
        return importlib.import_module("guandan_marvel.python_game")
    if normalized == "cpp":
        try:
            backend = importlib.import_module("guandan_marvel.cpp_game")
        except ImportError as error:
            raise BackendUnavailableError(
                "the requested cpp backend is unavailable; no Python fallback is permitted"
            ) from error
        if not hasattr(backend, "EnvBatch"):
            raise BackendUnavailableError(
                "the requested cpp backend is only partially initialized; no Python fallback is permitted"
            )
        return backend
    raise ValueError(f"unsupported game backend {name!r}; expected one of {SUPPORTED_BACKENDS}")


__all__ = [
    "BackendUnavailableError", "DEFAULT_BACKEND", "SUPPORTED_BACKENDS",
    "get_game_backend",
]
