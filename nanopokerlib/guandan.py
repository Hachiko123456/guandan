"""Guandan MARVEL V2 C++/pybind runtime.

The extension is explicit: importing this module fails if the C++ artifact was
not built. It never falls back to the Python oracle.
"""
from __future__ import annotations

import hashlib as _hashlib
import importlib as _importlib
from pathlib import Path as _Path
import platform as _platform
import sys as _sys

from guandan_marvel.config import GameSpec
from .build_identity import (
    NATIVE_BUILD_IDENTITY_SCHEMA as _NATIVE_BUILD_IDENTITY_SCHEMA,
    compute_native_source_manifest as _compute_native_source_manifest,
    find_repository_root as _find_repository_root,
)

_native = _importlib.import_module("nanopokerlib._guandan_cpp")
from ._guandan_cpp import *  # noqa: F401,F403,E402
from ._guandan_cpp import actions_from_np as _actions_from_np  # noqa: E402


def actions_from_np(rows):
    """Decode one or more structured rows into immutable C++ action records."""

    return tuple(_actions_from_np(rows))


def _file_sha256(path: _Path) -> str:
    digest = _hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_backend_build_identity() -> dict[str, object]:
    """Return the identity of the extension that is actually loaded."""

    extension_path = _Path(_native.__file__).resolve()
    repository_root = _find_repository_root(extension_path)
    current_manifest = None
    if repository_root is not None:
        current_manifest = _compute_native_source_manifest(repository_root)
    embedded_source = str(getattr(_native, "BUILD_SOURCE_SHA256", ""))
    return {
        "schema": str(getattr(_native, "BUILD_IDENTITY_SCHEMA", _NATIVE_BUILD_IDENTITY_SCHEMA)),
        "extension_module": _native.__name__,
        "extension_path": str(extension_path),
        "extension_filename": extension_path.name,
        "extension_sha256": _file_sha256(extension_path),
        "embedded_source_sha256": embedded_source,
        "compiler": str(getattr(_native, "BUILD_COMPILER", "unknown")),
        "cxx_standard": int(getattr(_native, "BUILD_CXX_STANDARD", 0)),
        "python_implementation": _platform.python_implementation(),
        "python_version": _platform.python_version(),
        "python_cache_tag": _sys.implementation.cache_tag,
        "current_source_manifest": current_manifest,
        "source_matches_loaded_binary": (
            None if current_manifest is None
            else current_manifest["hash"] == embedded_source
        ),
    }


BACKEND_BUILD_IDENTITY = get_backend_build_identity()

GAME_SPEC = GameSpec(
    identifier=GAME_IDENTIFIER,
    num_players=num_players,
    num_tokens=num_tokens,
    pad_token=pad_token,
    num_channels=num_channels,
    max_game_horizon=max_game_horizon,
    max_observation_tokens=max_observation_tokens,
    max_action_tokens=max_action_tokens,
    max_available_actions=max_available_actions,
    protocol_version=protocol_version,
)

__all__ = [name for name in globals() if not name.startswith("_")]
