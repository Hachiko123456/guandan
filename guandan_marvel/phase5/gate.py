# SPDX-License-Identifier: MIT
"""Fail-closed handling of the phase-5 B0 gate artifact.

A belief head may only be constructed from an artifact that is a well-formed,
self-consistent ``guandan-marvel-phase5-belief-gate-v1`` document whose verdict
is exactly ``PASS`` and whose design hash matches the currently running design.
Missing, malformed, REJECT and INCONCLUSIVE artifacts are all hard errors; a
strict loader never silently ignores a gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .design import (
    PHASE5_BELIEF_GATE_SCHEMA, PHASE5_DESIGN_SCHEMA, phase5_design_hash,
)

BELIEF_GATE_MISSING = "guandan-marvel-phase5-gate-missing"
_REQUIRED_KEYS = (
    "schema", "verdict", "design_hash", "source_hash", "config_hash",
    "dataset", "epsilon", "primary", "metrics", "sufficiency",
)


class BeliefGateError(RuntimeError):
    """A belief head was requested from an unusable B0 gate."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def gate_hash(gate: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in gate.items() if key != "gate_hash"}
    return hashlib.sha256(_canonical(payload)).hexdigest()


def seal_gate_artifact(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with its self-describing hash attached."""
    sealed = dict(gate)
    sealed["gate_hash"] = gate_hash(sealed)
    return sealed


def load_b0_gate(
    path: str | Path,
    *,
    expected_design_hash: str | None = None,
    require_pass: bool = True,
) -> dict[str, Any]:
    """Load and validate a B0 gate artifact; never accept an unknown verdict."""
    source = Path(path)
    if not source.is_file():
        raise BeliefGateError(f"{BELIEF_GATE_MISSING}: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BeliefGateError(f"belief gate is not readable JSON: {source}") from error
    if not isinstance(data, Mapping):
        raise BeliefGateError("belief gate must be a JSON object")
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise BeliefGateError(f"belief gate is missing required keys: {missing}")
    if data.get("schema") != PHASE5_BELIEF_GATE_SCHEMA:
        raise BeliefGateError(
            f"unexpected belief gate schema {data.get('schema')!r}; "
            f"expected {PHASE5_BELIEF_GATE_SCHEMA!r}"
        )
    verdict = data.get("verdict")
    if verdict not in ("PASS", "REJECT", "INCONCLUSIVE"):
        raise BeliefGateError(f"unknown belief gate verdict {verdict!r}")
    if require_pass and verdict != "PASS":
        raise BeliefGateError(
            f"belief head requires a PASS gate, got {verdict!r} "
            f"(reasons={data.get('reasons')!r})"
        )
    design_hash = data.get("design_hash")
    if not isinstance(design_hash, str) or len(design_hash) != 64:
        raise BeliefGateError("belief gate design_hash is missing or malformed")
    if expected_design_hash is not None and design_hash != str(expected_design_hash):
        raise BeliefGateError(
            "belief gate design hash does not match the running phase-5 design"
        )
    for key in ("source_hash", "config_hash"):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise BeliefGateError(f"belief gate {key} is missing")
    if not isinstance(data.get("dataset"), Mapping):
        raise BeliefGateError("belief gate dataset block is malformed")
    if not isinstance(data.get("primary"), Mapping) or not isinstance(
        data.get("metrics"), Mapping
    ):
        raise BeliefGateError("belief gate metric blocks are malformed")
    if not isinstance(data.get("sufficiency"), Mapping):
        raise BeliefGateError("belief gate sufficiency block is malformed")
    try:
        epsilon = float(data.get("epsilon"))
    except (TypeError, ValueError) as error:
        raise BeliefGateError("belief gate epsilon is not a number") from error
    if not 0.0 <= epsilon < float("inf"):
        raise BeliefGateError("belief gate epsilon is outside the finite non-negative range")
    recorded = data.get("gate_hash")
    if recorded is not None:
        actual = gate_hash(data)
        if str(recorded) != actual:
            raise BeliefGateError("belief gate hash is inconsistent with its contents")
    return dict(data)


def assert_belief_head_allowed(
    belief_config: object,
    *,
    expected_design_hash: str | None = None,
) -> dict[str, Any] | None:
    """Enforce the config contract for a training-time belief head.

    Returns ``None`` for the default (``build_head`` off) configuration and the
    validated artifact otherwise.  It never accepts a free-standing flag.
    """
    enabled = bool(getattr(belief_config, "build_head", False))
    if not enabled:
        return None
    weight = float(getattr(belief_config, "belief_weight", 0.0) or 0.0)
    if weight <= 0.0:
        raise BeliefGateError("belief head requires belief_weight > 0")
    artifact = getattr(belief_config, "gate_artifact", None)
    if not artifact:
        raise BeliefGateError(
            f"{BELIEF_GATE_MISSING}: belief.build_head requires belief.gate_artifact"
        )
    return load_b0_gate(
        artifact,
        expected_design_hash=(
            phase5_design_hash() if expected_design_hash is None else expected_design_hash
        ),
        require_pass=True,
    )


def phase5_design_identity() -> dict[str, Any]:
    return {"schema": PHASE5_DESIGN_SCHEMA, "design_hash": phase5_design_hash()}


__all__ = [
    "BELIEF_GATE_MISSING", "BeliefGateError", "assert_belief_head_allowed",
    "gate_hash", "load_b0_gate", "phase5_design_identity", "seal_gate_artifact",
]
