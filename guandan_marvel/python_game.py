"""Importable Python-reference backend alias.

The implementation remains in :mod:`guandan_marvel.game` for historical direct
imports. This module mirrors every public attribute of that implementation and
never loads the native extension.
"""
from __future__ import annotations

from . import game as _backend

for _name in dir(_backend):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_backend, _name)

__all__ = [name for name in globals() if not name.startswith("_")]
