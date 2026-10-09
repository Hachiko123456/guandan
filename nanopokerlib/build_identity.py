"""Deterministic source identity for the native Guandan extension build."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

NATIVE_BUILD_IDENTITY_SCHEMA = "guandan-native-build-identity-v1"
NATIVE_SOURCE_PATHS = (
    "nanopokerlib/cpp/bindings.cpp",
    "nanopokerlib/cpp/core/types.hpp",
    "nanopokerlib/cpp/guandan/guandan.cpp",
    "nanopokerlib/cpp/guandan/guandan.hpp",
)


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_native_source_manifest(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root)
    files = []
    for relative in NATIVE_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        files.append({
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    payload = {
        "schema": NATIVE_BUILD_IDENTITY_SCHEMA,
        "files": files,
    }
    return {**payload, "hash": canonical_json_hash(payload)}


def find_repository_root(start: str | Path) -> Path | None:
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if all((candidate / relative).is_file() for relative in NATIVE_SOURCE_PATHS):
            return candidate
    return None


__all__ = [
    "NATIVE_BUILD_IDENTITY_SCHEMA", "NATIVE_SOURCE_PATHS",
    "canonical_json_hash", "compute_native_source_manifest",
    "find_repository_root",
]
