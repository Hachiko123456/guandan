# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, object] | list[object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".tmp-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def append_jsonl(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def relative_file_manifest(root: str | Path, paths: list[Path]) -> list[dict[str, object]]:
    base = Path(root).resolve()
    rows = []
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        if not path.is_file() or not path.is_relative_to(base):
            raise ValueError(f"manifest path is outside root or not a file: {path}")
        rows.append({
            "path": path.relative_to(base).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


__all__ = ["append_jsonl", "atomic_json", "canonical_hash", "relative_file_manifest", "sha256_file"]
