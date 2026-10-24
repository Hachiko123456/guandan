# SPDX-License-Identifier: MIT
"""Atomic completion markers bound to plan/source/model/output hashes.

A directory merely *existing* is never evidence that a phase completed.  Every
marker declares the hashes of the files it produced; reading it re-hashes those
files, so a missing, half-written or tampered output invalidates the phase and
forces a re-run instead of silently pretending the work is done.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .common import MARKER_SCHEMA, PlanError, atomic_json, canonical, read_json, sha, verify_hash_record


class MarkerError(RuntimeError):
    """Raised when a completion marker is missing, malformed or inconsistent."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def marker_path(phase_dir: str | Path) -> Path:
    return Path(phase_dir) / "COMPLETE.json"


def write_marker(
    phase_dir: str | Path,
    *,
    phase: str,
    plan_hash: str,
    source_hash: str,
    input_hashes: Mapping[str, str],
    output_paths: Sequence[str | Path],
    root: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    directory = Path(phase_dir)
    root_path = Path(root).resolve()
    outputs: dict[str, str] = {}
    for value in output_paths:
        path = Path(value)
        resolved = path.resolve()
        if not resolved.is_file():
            raise MarkerError(f"declared output is missing: {path}")
        if not resolved.is_relative_to(root_path):
            raise MarkerError(f"declared output escapes the run root: {path}")
        outputs[resolved.relative_to(root_path).as_posix()] = sha(resolved)
    payload: dict[str, object] = {
        "schema": MARKER_SCHEMA,
        "phase": str(phase),
        "plan_hash": str(plan_hash),
        "source_hash": str(source_hash),
        "input_hashes": {str(k): str(v) for k, v in sorted(input_hashes.items())},
        "outputs": outputs,
        "created_at": _timestamp(),
        "metadata": dict(metadata or {}),
    }
    payload["hash"] = canonical(payload)
    atomic_json(marker_path(directory), payload)
    return payload


def read_marker(
    phase_dir: str | Path,
    *,
    phase: str,
    plan_hash: str,
    source_hash: str,
    root: str | Path,
    input_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    path = marker_path(phase_dir)
    if not path.is_file():
        raise MarkerError(f"phase {phase!r} has no completion marker")
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise MarkerError(f"phase {phase!r} marker is not an object")
    verify_hash_record(payload)
    if payload.get("schema") != MARKER_SCHEMA:
        raise MarkerError(f"phase {phase!r} marker schema mismatch")
    if payload.get("phase") != phase:
        raise MarkerError(f"marker phase {payload.get('phase')!r} != {phase!r}")
    if payload.get("plan_hash") != plan_hash:
        raise MarkerError(f"phase {phase!r} marker belongs to a different plan")
    if payload.get("source_hash") != source_hash:
        raise MarkerError(f"phase {phase!r} marker belongs to a different source")
    if input_hashes is not None:
        declared = dict(payload.get("input_hashes") or {})
        for key, expected in input_hashes.items():
            if declared.get(key) != expected:
                raise MarkerError(f"phase {phase!r} input hash mismatch for {key!r}")
    root_path = Path(root).resolve()
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        raise MarkerError(f"phase {phase!r} marker has no output list")
    for relative, digest in outputs.items():
        target = (root_path / str(relative)).resolve()
        if not target.is_file():
            raise MarkerError(f"phase {phase!r} output is missing: {relative}")
        if sha(target) != digest:
            raise MarkerError(f"phase {phase!r} output hash changed: {relative}")
    return dict(payload)


def phase_is_complete(
    phase_dir: str | Path,
    *,
    phase: str,
    plan_hash: str,
    source_hash: str,
    root: str | Path,
    input_hashes: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    try:
        read_marker(
            phase_dir, phase=phase, plan_hash=plan_hash, source_hash=source_hash,
            root=root, input_hashes=input_hashes,
        )
        return True, "complete"
    except (MarkerError, PlanError, ValueError, OSError) as error:
        return False, str(error)


__all__ = [
    "MarkerError", "marker_path", "phase_is_complete", "read_marker", "write_marker",
]
