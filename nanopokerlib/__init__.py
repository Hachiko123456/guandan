"""Licensed C++ game runtimes used by the Guandan MARVEL migration."""
from __future__ import annotations

from importlib import import_module


def __getattr__(name: str):
    if name != "guandan":
        raise AttributeError(name)
    value = import_module("nanopokerlib.guandan")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"guandan"})


__all__ = ["guandan"]
