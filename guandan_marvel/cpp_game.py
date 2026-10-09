"""Explicit MARVEL C++ game backend.

Importing this module requires the built pybind extension and never falls back
to the Python oracle.
"""
from __future__ import annotations

from nanopokerlib import guandan as _backend

if not hasattr(_backend, "EnvBatch"):
    raise ImportError("nanopokerlib.guandan is not fully initialized")

for _name in _backend.__all__:
    globals()[_name] = getattr(_backend, _name)

__all__ = list(_backend.__all__)
