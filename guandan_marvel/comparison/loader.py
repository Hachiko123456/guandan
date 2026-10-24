# SPDX-License-Identifier: MIT
"""Shared artifact/control loading for comparison workers and validators."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from guandan_marvel.config import deserialize_train_config
from guandan_marvel.evaluation.checkpoints import V2PolicyArtifact, load_v2_policy
from guandan_marvel.evaluation.common import canonical_json_bytes, file_sha256, load_json, payload_hash
from guandan_marvel.evaluation.manifests import dataset_identity, deals_for, validate_deals_manifest
from guandan_marvel.model import Agent
from guandan_marvel.training.checkpoints import CheckpointCompatibilityError

from .plan import CONTROL_SCHEMA

RELAXED_ALLOWED_MESSAGES = ("build identity mismatch", "different native sources")
SOURCE_ARCHIVE_CONTROL_PATH = "guandan-marvel-source.zip"


def unpacked_source_present(base: Path) -> bool:
    """True when a validated source directory (any nesting) replaces the source zip."""
    return any(
        candidate.is_file() and candidate.parent.resolve() != base
        for candidate in base.rglob("source_manifest.json")
    )


def load_control(path: str | Path) -> tuple[dict[str, object], Path]:
    control_path = Path(path).resolve()
    control = load_json(control_path)
    if control.get("schema") != CONTROL_SCHEMA:
        raise ValueError("comparison control schema mismatch")
    if control.get("hash") != payload_hash(control):
        raise ValueError("comparison control hash mismatch")
    return control, control_path.parent


def safe_control_file(base: Path, spec: Mapping[str, object], label: str) -> Path:
    relative = Path(str(spec.get("path", "")))
    target = (base / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(base) or not target.is_file():
        raise ValueError(f"unsafe/missing comparison file: {label}")
    if target.stat().st_size != int(spec.get("bytes", -1)) or file_sha256(target) != str(spec.get("sha256")):
        raise ValueError(f"comparison file identity mismatch: {label}")
    return target


def validate_control_files(control: Mapping[str, object], base: Path) -> int:
    rows = control.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("comparison control files manifest is empty")
    skipped: list[str] = []
    for row in rows:
        relative = str(row.get("path"))
        target = (base / relative).resolve()
        if relative == SOURCE_ARCHIVE_CONTROL_PATH and not target.is_file() and unpacked_source_present(base):
            skipped.append(relative)
            continue
        safe_control_file(base, row, relative)
    if skipped:
        print(json.dumps({
            "event": "comparison_control_note",
            "skipped": skipped,
            "reason": "source zip absent because the upload platform unpacked it; the bootstrap validates the directory input",
        }, sort_keys=True), flush=True)
    return len(rows)


def load_v2_artifact(path: str | Path, *, device: str, expected_algorithm: str = "ippo") -> tuple[V2PolicyArtifact, str]:
    """Official strict loader first; relaxed (identity-check-skipping) fallback."""
    try:
        return load_v2_policy(path, device=device, expected_algorithm=expected_algorithm), "official"
    except (ValueError, CheckpointCompatibilityError) as error:
        message = str(error)
        if not any(allowed in message for allowed in RELAXED_ALLOWED_MESSAGES):
            raise
        return _load_v2_policy_relaxed(path, device=device, expected_algorithm=expected_algorithm), "relaxed"


def _load_v2_policy_relaxed(path: str | Path, *, device: str, expected_algorithm: str) -> V2PolicyArtifact:
    resolved_path = Path(path).resolve()
    payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    if payload.get("schema") != "guandan-marvel-evaluation-checkpoint-v1":
        raise ValueError("relaxed loader accepts only evaluation checkpoints")
    if payload.get("architecture") != "v1.3":
        raise ValueError("relaxed loader requires architecture v1.3")
    if payload.get("algorithm") != expected_algorithm:
        raise ValueError("relaxed loader algorithm mismatch")
    if payload.get("backend") != "cpp":
        raise ValueError("relaxed loader requires the cpp backend checkpoint")
    resolved_config = payload.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise ValueError("relaxed loader requires resolved_config")
    if payload.get("resolved_config_hash") != hashlib.sha256(canonical_json_bytes(resolved_config)).hexdigest():
        raise ValueError("relaxed loader resolved_config hash mismatch")
    cfg = deserialize_train_config(resolved_config)
    if cfg is None:
        raise ValueError("relaxed loader could not decode the train config")
    torch_device = torch.device(device)
    cfg.device = str(torch_device)
    state = payload.get("agent_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("relaxed loader requires agent_state_dict")
    algorithm = str(payload.get("algorithm"))
    enable_ntp = Agent.ntp_enabled_for_config(cfg) and algorithm == "ippo"
    model = Agent(cfg, enable_value_head=algorithm == "ippo", enable_ntp_head=enable_ntp).to(torch_device)
    model.load_state_dict(state, strict=True)
    model.eval()
    backend_identity = payload.get("backend_identity")
    return V2PolicyArtifact(
        path=resolved_path,
        sha256=file_sha256(resolved_path),
        schema=str(payload.get("schema")),
        algorithm=str(payload.get("algorithm")),
        architecture=str(payload.get("architecture")),
        protocol_version=str(payload.get("protocol_version")),
        backend=str(payload.get("backend")),
        backend_identity_hash="" if not isinstance(backend_identity, Mapping) else str(backend_identity.get("hash", "")),
        resolved_config_hash=str(payload.get("resolved_config_hash")),
        iteration=int(payload.get("iteration", 0)),
        model=model,
        training_data_provenance=None,
        metadata={"source_hash": payload.get("source_hash"), "relaxed_loader": True},
    )


def load_fabledan_model(weights_path: str | Path):
    from fabledan_vendor.model_np import NumpyModel
    return NumpyModel(str(weights_path))


def load_deals(base: Path, control: Mapping[str, object]) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    spec = control["deals_manifest"]
    path = safe_control_file(base, spec, "deals_manifest")
    deals = load_json(path)
    validate_deals_manifest(deals)
    if dataset_identity(deals, "final") != str(spec.get("dataset_hash")):
        raise ValueError("comparison deals dataset identity mismatch")
    rows = deals_for(deals, "final")
    if len(rows) != int(spec.get("deal_count", -1)):
        raise ValueError("comparison deals manifest count mismatch")
    return deals, rows


__all__ = [
    "load_control", "load_deals", "load_fabledan_model", "load_v2_artifact",
    "safe_control_file", "validate_control_files",
]
