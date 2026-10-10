# SPDX-License-Identifier: MIT
"""Strict evaluation-only checkpoint loading and merged-policy freezing."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

import torch

from guandan.training.snapshot import load_policy_snapshot
from guandan_marvel import cpp_game
from guandan_marvel.config import deserialize_train_config
from guandan_marvel.model import Agent
from guandan_marvel.training.checkpoints import (
    BACKEND_IDENTITY_SCHEMA,
    PRE_TRAINING_PROVENANCE_CHECKPOINT_SCHEMA, TRAINING_CHECKPOINT_SCHEMA,
    build_backend_identity,
    canonical_json_hash,
    validate_declared_hash,
    validate_source_fingerprint,
)

from .common import file_sha256

EVALUATION_CHECKPOINT_SCHEMA = "guandan-marvel-evaluation-checkpoint-v1"
LEGACY_V1_ACTION = "GD-ACTION-0.1"
LEGACY_V1_ENV = "GD-ENV-0.1"
SUPPORTED_ALGORITHMS = {"ippo", "vrpo"}


@dataclass(frozen=True)
class V2PolicyArtifact:
    path: Path
    sha256: str
    schema: str
    algorithm: str
    architecture: str
    protocol_version: str
    backend: str
    backend_identity_hash: str
    resolved_config_hash: str
    iteration: int
    model: Agent
    training_data_provenance: dict[str, object] | None
    metadata: dict[str, object]

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": "guandan-marvel-policy-identity-v1",
            "checkpoint_path": str(self.path),
            "checkpoint_sha256": self.sha256,
            "checkpoint_schema": self.schema,
            "algorithm": self.algorithm,
            "architecture": self.architecture,
            "protocol_version": self.protocol_version,
            "backend": self.backend,
            "backend_identity_hash": self.backend_identity_hash,
            "resolved_config_hash": self.resolved_config_hash,
            "source_commit": self.metadata.get("source_parent_commit"),
            "source_hash": self.metadata.get("source_hash"),
            "merged_policy": deepcopy(self.metadata.get("merged_policy")),
            "training_counters": {"iteration": self.iteration},
            "training_data_provenance_status": "complete" if self.training_data_provenance and self.training_data_provenance.get("complete") else "incomplete",
            "training_data_provenance_hash": None if self.training_data_provenance is None else self.training_data_provenance.get("hash"),
            "iteration": self.iteration,
        }


@dataclass(frozen=True)
class LegacyV1OpponentArtifact:
    path: Path
    sha256: str
    algorithm: str
    protocol_versions: dict[str, str]
    metadata: dict[str, object]
    training_deals: tuple[dict[str, object], ...]

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": "guandan-legacy-v1-opponent-identity-v1",
            "checkpoint_path": str(self.path),
            "checkpoint_sha256": self.sha256,
            "algorithm": self.algorithm,
            "protocol_versions": self.protocol_versions,
            "usage": "opponent_only",
            "training_resume_allowed": False,
        }


def _load_payload(path: Path) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    return payload


def _validate_v2_payload(payload: Mapping[str, object], *, expected_algorithm: str | None) -> None:
    schema = payload.get("schema")
    if schema not in {TRAINING_CHECKPOINT_SCHEMA, PRE_TRAINING_PROVENANCE_CHECKPOINT_SCHEMA, EVALUATION_CHECKPOINT_SCHEMA}:
        legacy = "|".join(str(payload.get(name, "")) for name in (
            "protocol_version", "action_version", "environment_version", "encoding_version",
        ))
        if LEGACY_V1_ACTION in legacy or LEGACY_V1_ENV in legacy or schema is None:
            raise ValueError("legacy V1 artifacts are opponent-only and cannot be loaded as MARVEL V2 policies")
        raise ValueError(f"unsupported evaluation checkpoint schema {schema!r}")
    algorithm = payload.get("algorithm")
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported checkpoint algorithm {algorithm!r}")
    if expected_algorithm is not None and algorithm != expected_algorithm:
        raise ValueError(f"checkpoint algorithm mismatch: expected {expected_algorithm!r}, got {algorithm!r}")
    if payload.get("architecture") != "v1.3":
        raise ValueError("evaluation requires MARVEL V2 architecture v1.3")
    if payload.get("protocol_version") != cpp_game.protocol_version:
        raise ValueError("checkpoint protocol mismatch")
    if payload.get("backend") != "cpp":
        raise ValueError("M06 evaluation requires a C++ backend checkpoint")

    backend_identity = payload.get("backend_identity")
    if not isinstance(backend_identity, Mapping) or backend_identity.get("schema") != BACKEND_IDENTITY_SCHEMA:
        raise ValueError("checkpoint backend identity is missing or unsupported")
    validate_declared_hash(backend_identity, name="checkpoint.backend_identity")
    current_identity = build_backend_identity(cpp_game)
    validate_declared_hash(current_identity, name="current.backend_identity")
    if backend_identity.get("hash") != current_identity.get("hash"):
        raise ValueError("checkpoint C++ backend/build identity mismatch")

    provenance = payload.get("training_data_provenance")
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise ValueError("checkpoint training_data_provenance is malformed")
        validate_declared_hash(provenance, name="checkpoint.training_data_provenance")
    if schema == TRAINING_CHECKPOINT_SCHEMA and provenance is None:
        raise ValueError("training checkpoint v5 requires training_data_provenance")

    resolved = payload.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise ValueError("checkpoint resolved_config is missing")
    if payload.get("resolved_config_hash") != canonical_json_hash(resolved):
        raise ValueError("checkpoint resolved_config hash mismatch")
    model_cfg = resolved.get("model")
    if not isinstance(model_cfg, Mapping) or model_cfg.get("architecture") != payload.get("architecture"):
        raise ValueError("checkpoint architecture metadata mismatch")
    game_cfg = resolved.get("game")
    if not isinstance(game_cfg, Mapping) or game_cfg.get("protocol_version") != cpp_game.protocol_version:
        raise ValueError("checkpoint game protocol metadata mismatch")
    state = payload.get("agent_state_dict")
    if not isinstance(state, Mapping) or not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("checkpoint agent_state_dict is missing or malformed")
    source_fingerprint = payload.get("source_fingerprint")
    if isinstance(source_fingerprint, Mapping):
        validate_source_fingerprint(source_fingerprint, name="checkpoint.source_fingerprint")


def load_v2_policy(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_algorithm: str | None = None,
) -> V2PolicyArtifact:
    resolved_path = Path(path).resolve()
    payload = _load_payload(resolved_path)
    _validate_v2_payload(payload, expected_algorithm=expected_algorithm)
    cfg = deserialize_train_config(payload["resolved_config"])
    if cfg is None:
        raise ValueError("checkpoint train config could not be decoded")
    cfg.device = str(torch.device(device))
    algorithm = str(payload["algorithm"])
    model = Agent(cfg, enable_value_head=algorithm == "ippo").to(torch.device(device))
    model.load_state_dict(payload["agent_state_dict"], strict=True)
    model.eval()
    backend_identity = payload["backend_identity"]
    return V2PolicyArtifact(
        path=resolved_path,
        sha256=file_sha256(resolved_path),
        schema=str(payload["schema"]),
        algorithm=algorithm,
        architecture=str(payload["architecture"]),
        protocol_version=str(payload["protocol_version"]),
        backend=str(payload["backend"]),
        backend_identity_hash=str(backend_identity["hash"]),
        resolved_config_hash=str(payload["resolved_config_hash"]),
        iteration=int(payload.get("iteration", 0)),
        model=model,
        training_data_provenance=None if payload.get("training_data_provenance") is None else deepcopy(payload["training_data_provenance"]),
        metadata={
            "merged_policy": deepcopy(payload.get("merged_policy")),
            "source_hash": payload.get("source_hash"),
            "source_parent_commit": payload.get("source_parent_commit"),
            "selection_identity": deepcopy(payload.get("selection_identity")),
            "champion_identity": deepcopy(payload.get("champion_identity")),
            "training_seed": payload["resolved_config"].get("seed"),
        },
    )



def verify_policy_identity(identity: Mapping[str, object], *, device: str | torch.device = "cpu") -> V2PolicyArtifact:
    required = {
        "schema", "checkpoint_path", "checkpoint_sha256", "checkpoint_schema",
        "algorithm", "architecture", "protocol_version", "backend",
        "backend_identity_hash", "resolved_config_hash", "iteration",
    }
    missing = sorted(required - set(identity))
    if missing:
        raise ValueError(f"policy identity is missing fields: {missing}")
    path = Path(str(identity["checkpoint_path"])).resolve()
    if not path.is_file() or file_sha256(path) != identity["checkpoint_sha256"]:
        raise ValueError("policy identity checkpoint file is missing or content hash changed")
    artifact = load_v2_policy(path, device=device, expected_algorithm=str(identity["algorithm"]))
    actual = artifact.identity
    for key in required:
        if actual.get(key) != identity.get(key):
            raise ValueError(f"policy identity field mismatch for {key}")
    for key in ("source_commit", "source_hash", "merged_policy", "training_counters",
                "training_data_provenance_status", "training_data_provenance_hash"):
        if key in identity and actual.get(key) != identity.get(key):
            raise ValueError(f"policy identity field mismatch for {key}")
    return artifact

def load_legacy_v1_opponent(path: str | Path) -> LegacyV1OpponentArtifact:
    resolved_path = Path(path).resolve()
    snapshot = load_policy_snapshot(resolved_path)
    metadata = dict(snapshot.metadata)
    protocols = metadata.get("protocol_versions")
    if not isinstance(protocols, Mapping):
        raise ValueError("legacy opponent protocol metadata is missing")
    action = str(protocols.get("action", protocols.get("action_version", "")))
    environment = str(protocols.get("environment", protocols.get("environment_version", "")))
    if action != LEGACY_V1_ACTION or environment != LEGACY_V1_ENV:
        raise ValueError("dedicated legacy adapter accepts only GD-ACTION/GD-ENV-0.1 artifacts")
    algorithm = str(metadata.get("algorithm", ""))
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError("legacy opponent algorithm metadata is invalid")
    return LegacyV1OpponentArtifact(
        path=resolved_path,
        sha256=file_sha256(resolved_path),
        algorithm=algorithm,
        protocol_versions={str(key): str(value) for key, value in protocols.items()},
        metadata=metadata,
        training_deals=tuple(dict(row) for row in snapshot.training_deals),
    )


def _atomic_torch_save(payload: Mapping[str, object], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def freeze_merged_policy(trainer: object, path: str | Path, *, iteration: int | None = None) -> Path:
    """Clone the merged actor into a versioned evaluation-only checkpoint.

    The function reads model/config identity only. Optimizer, sampler, RNG, and
    replay state are deliberately excluded so this artifact cannot be used to
    resume training.
    """
    model = getattr(trainer, "agent_merged", None)
    if model is None:
        raise TypeError("trainer must expose agent_merged")
    target_iteration = int(getattr(trainer, "iteration", 0) if iteration is None else iteration)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    payload: dict[str, object] = {
        "schema": EVALUATION_CHECKPOINT_SCHEMA,
        "algorithm": str(getattr(trainer, "algorithm")),
        "architecture": str(trainer.cfg.model.architecture),
        "protocol_version": str(trainer.cfg.game.protocol_version),
        "backend": str(trainer.backend_name),
        "backend_identity": deepcopy(trainer.backend_identity),
        "resolved_config": deepcopy(trainer.resolved_config),
        "resolved_config_hash": str(trainer.resolved_config_hash),
        "agent_state_dict": state,
        "iteration": target_iteration,
        "merged_policy": {
            "identity": str(trainer.merged_policy_identity),
            "beta": float(trainer.cfg.agent_merging_beta),
            "frozen_for_evaluation": True,
        },
        "source_fingerprint": deepcopy(trainer.source_fingerprint),
        "source_hash": str(trainer.source_hash),
        "source_parent_commit": str(trainer.source_parent_commit),
        "training_data_provenance": deepcopy(trainer.sampler.training_data_provenance()),
        "resume_classification": "evaluation_only",
        "training_resume_allowed": False,
    }
    return _atomic_torch_save(payload, Path(path))


__all__ = [
    "EVALUATION_CHECKPOINT_SCHEMA", "LegacyV1OpponentArtifact", "V2PolicyArtifact",
    "freeze_merged_policy", "load_legacy_v1_opponent", "load_v2_policy",
    "verify_policy_identity",
]
