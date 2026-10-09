# SPDX-License-Identifier: MIT
"""Shared deterministic JSON, hashing, and atomic-write helpers for M06."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def payload_hash(value: Mapping[str, object], *, hash_field: str = "hash") -> str:
    body = dict(value)
    body.pop(hash_field, None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, object],
    *,
    replace_fn: Callable[[str | bytes | os.PathLike[str] | os.PathLike[bytes], str | bytes | os.PathLike[str] | os.PathLike[bytes]], None] = os.replace,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    data = json.dumps(dict(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        replace_fn(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


__all__ = ["atomic_write_json", "canonical_json_bytes", "file_sha256", "load_json", "payload_hash"]
