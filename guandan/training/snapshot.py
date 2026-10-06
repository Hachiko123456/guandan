"""Inference-only policy snapshots extracted from current training checkpoints."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from .checkpoint import CheckpointError, TrainingCheckpoint, load_checkpoint

POLICY_SNAPSHOT_VERSION = 1


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    model_state: Mapping[str, torch.Tensor]
    model_config: dict[str, Any]
    metadata: dict[str, Any]
    training_deals: list[dict[str, Any]]


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _actor_state(checkpoint: TrainingCheckpoint) -> dict[str, torch.Tensor]:
    state = checkpoint.model_state
    actor = {key[len("actor."):]: value.detach().cpu().clone()
             for key, value in state.items() if key.startswith("actor.")}
    if not actor:
        raise CheckpointError("checkpoint has no actor state for policy snapshot")
    if any(not isinstance(value, torch.Tensor) for value in actor.values()):
        raise CheckpointError("policy actor state must contain tensors")
    return actor


def export_policy_snapshot(checkpoint: str | Path, path: str | Path) -> dict[str, Any]:
    """Export the actor from a current checkpoint without optimizer state.

    The source is loaded through the current safe checkpoint boundary.  This is
    an explicit export operation, not an implicit historical-checkpoint
    migration.  The resulting file contains actor weights and provenance only.
    """
    source = Path(checkpoint).resolve()
    destination = Path(path)
    loaded = load_checkpoint(source, map_location="cpu")
    actor = _actor_state(loaded)
    training_deals = deepcopy(loaded.training_state.get("engine", {}).get("collector", {}).get("deal_records", []))
    metadata = {
        "kind": "policy_snapshot",
        "source_checkpoint": str(source),
        "source_git_commit": loaded.metadata.git_commit,
        "source_fingerprint": loaded.metadata.source_fingerprint,
        "implementation_fingerprint": loaded.metadata.implementation_fingerprint,
        "semantic_contract": deepcopy(loaded.metadata.semantic_contract),
        "compatibility_schema_version": loaded.metadata.compatibility_schema_version,
        "profile": loaded.metadata.profile,
        "algorithm": loaded.metadata.algorithm,
        "update": loaded.metadata.update_count,
        "seed": loaded.metadata.seed_count,
        "model_version": loaded.metadata.model_version,
        "training_version": loaded.metadata.training_version,
        "protocol_versions": loaded.metadata.protocol_versions,
    }
    payload = {
        "policy_snapshot_version": POLICY_SNAPSHOT_VERSION,
        "model_state": actor,
        "model_config": deepcopy(loaded.model_config),
        "metadata": metadata,
        "training_deals": training_deals,
    }
    _atomic_save(payload, destination)
    return {
        "path": str(destination.resolve()),
        "policy_snapshot_version": POLICY_SNAPSHOT_VERSION,
        "algorithm": loaded.metadata.algorithm,
        "update": loaded.metadata.update_count,
        "source_checkpoint": str(source),
    }


def _validate_snapshot_payload(payload: Any, path: Path) -> PolicySnapshot:
    if type(payload) is not dict or payload.get("policy_snapshot_version") != POLICY_SNAPSHOT_VERSION:
        raise CheckpointError(f"unsupported or malformed policy snapshot: {path}")
    state = payload.get("model_state")
    if not isinstance(state, dict) or not state or any(type(k) is not str or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise CheckpointError("policy snapshot model_state must be a non-empty tensor mapping")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise CheckpointError("policy snapshot model_config must be a dictionary")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CheckpointError("policy snapshot metadata must be a dictionary")
    deals = payload.get("training_deals", [])
    if not isinstance(deals, list):
        raise CheckpointError("policy snapshot training_deals must be a list")
    return PolicySnapshot({k: v.detach().cpu() for k, v in state.items()}, deepcopy(config), deepcopy(metadata), deepcopy(deals))


def load_policy_snapshot(path: str | Path) -> PolicySnapshot:
    """Load either a current policy snapshot or a current training checkpoint.

    This accepts a training checkpoint only as a read-only inference source;
    it never loads or returns optimizer state.  Historical checkpoint formats
    are rejected by ``load_checkpoint`` and are not implicitly migrated.
    """
    resolved = Path(path).resolve()
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"could not safely load policy artifact {resolved}: {exc}") from exc
    if isinstance(payload, dict) and payload.get("policy_snapshot_version") == POLICY_SNAPSHOT_VERSION:
        return _validate_snapshot_payload(payload, resolved)
    # Evaluation may read a historical actor as a read-only baseline. This is
    # not a training resume or an implicit migration; export_policy_snapshot
    # remains current-schema-only.
    loaded = load_checkpoint(resolved, map_location="cpu", allow_legacy_readonly=True)
    return PolicySnapshot(
        {key[len("actor."):]: value.detach().cpu() for key, value in loaded.model_state.items() if key.startswith("actor.")},
        deepcopy(loaded.model_config),
        {
            "kind": "training_checkpoint",
            "source_checkpoint": str(resolved),
            "source_git_commit": loaded.metadata.git_commit,
            "source_fingerprint": loaded.metadata.source_fingerprint,
            "implementation_fingerprint": loaded.metadata.implementation_fingerprint,
            "semantic_contract": deepcopy(loaded.metadata.semantic_contract),
            "compatibility_schema_version": loaded.metadata.compatibility_schema_version,
            "profile": loaded.metadata.profile,
            "algorithm": loaded.metadata.algorithm,
            "update": loaded.metadata.update_count,
            "seed": loaded.metadata.seed_count,
            "model_version": loaded.metadata.model_version,
            "training_version": loaded.metadata.training_version,
            "protocol_versions": loaded.metadata.protocol_versions,
        },
        deepcopy(loaded.training_state.get("engine", {}).get("collector", {}).get("deal_records", [])),
    )


__all__ = ["POLICY_SNAPSHOT_VERSION", "PolicySnapshot", "export_policy_snapshot", "load_policy_snapshot"]
