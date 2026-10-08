# SPDX-License-Identifier: MIT
"""Versioned M03 training checkpoints and resume classification."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

TRAINING_CHECKPOINT_SCHEMA = "guandan-marvel-training-checkpoint-v2"
TRAINING_SEMANTIC_CONTRACT_SCHEMA = "guandan-marvel-training-contract-v1"
RESUME_KINDS = frozenset({
    "exact_resume", "compatible_resume", "warm_restart", "migration", "evaluation_only",
})


class CheckpointCompatibilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResumeDecision:
    kind: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in RESUME_KINDS:
            raise ValueError(f"unknown resume kind {self.kind!r}")


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serialize_config(config: object) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        value = config.to_dict()
    elif is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, Mapping):
        value = dict(config)
    else:
        raise TypeError("training config must be serializable as a mapping")
    return dict(value)


def compute_source_hash(repository_root: str | Path, paths: tuple[str, ...]) -> str:
    root = Path(repository_root)
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError("checkpoint contains CUDA RNG but CUDA is unavailable")
        torch.cuda.set_rng_state_all(list(cuda_state))


def build_semantic_contract(
    *,
    algorithm: str,
    architecture: str,
    game_identifier: str,
    protocol_version: str,
    sampler_contract: str,
    sampler_state_schema: str,
    buffer_schema: str,
    q_target_contract: str | None,
) -> dict[str, object]:
    contract = {
        "schema": TRAINING_SEMANTIC_CONTRACT_SCHEMA,
        "algorithm": str(algorithm),
        "architecture": str(architecture),
        "game_identifier": str(game_identifier),
        "protocol_version": str(protocol_version),
        "sampler_contract": str(sampler_contract),
        "sampler_state_schema": str(sampler_state_schema),
        "buffer_schema": str(buffer_schema),
        "on_policy": {
            "purpose": "on_policy",
            "sampling_temperature": 1.0,
            "forced_action_indices": None,
            "old_probability_field": "behavior_action_prob",
            "reference_policy_field": "policy_probs",
            "eligibility_field": "on_policy_valid",
        },
        "q_target_contract": q_target_contract,
    }
    return {**contract, "hash": canonical_json_hash(contract)}


def classify_resume(
    checkpoint: Mapping[str, object],
    *,
    expected_contract: Mapping[str, object],
    expected_source_hash: str,
    expected_config_hash: str | None = None,
    require_replay: bool,
    for_evaluation: bool = False,
) -> ResumeDecision:
    schema = checkpoint.get("schema")
    if schema != TRAINING_CHECKPOINT_SCHEMA:
        legacy_protocols = "|".join(str(checkpoint.get(key, "")) for key in (
            "protocol_version", "action_version", "environment_version", "encoding_version",
        ))
        if "GD-ACTION-0.1" in legacy_protocols or "GD-ENV-0.1" in legacy_protocols or schema in {None, "guandan-training-checkpoint-v1"}:
            if for_evaluation:
                return ResumeDecision("evaluation_only", ("legacy V1 checkpoint",))
            raise CheckpointCompatibilityError("legacy V1 checkpoint cannot resume MARVEL V2 training")
        raise CheckpointCompatibilityError(f"unsupported checkpoint schema {schema!r}")

    if for_evaluation:
        return ResumeDecision("evaluation_only", ("explicit evaluation load",))

    reasons: list[str] = []
    checkpoint_contract = checkpoint.get("semantic_contract")
    if not isinstance(checkpoint_contract, Mapping):
        return ResumeDecision("migration", ("missing semantic contract",))
    if checkpoint_contract.get("hash") != expected_contract.get("hash"):
        return ResumeDecision("migration", ("semantic contract hash mismatch",))

    required = {
        "agent_resume_state_dict", "agent_state_dict", "agent_optimizer_state_dict",
        "iteration", "durable_updates", "counters", "rng_state", "sampler_state",
        "schedule_state", "merged_policy", "resolved_config", "resolved_config_hash", "source_hash",
    }
    if checkpoint_contract.get("algorithm") == "vrpo":
        required.update({"critic_state_dict", "critic_optimizer_state_dict"})
    missing = sorted(required - set(checkpoint))
    if require_replay and "replay_buffer_state" not in checkpoint:
        missing.append("replay_buffer_state")
    if missing:
        return ResumeDecision("warm_restart", tuple(f"missing {name}" for name in missing))

    if expected_config_hash is not None and str(checkpoint.get("resolved_config_hash")) != str(expected_config_hash):
        reasons.append("resolved config hash differs while semantic contract matches")
    if str(checkpoint.get("source_hash")) != str(expected_source_hash):
        reasons.append("source hash differs while semantic contract matches")
    if reasons:
        return ResumeDecision("compatible_resume", tuple(reasons))
    return ResumeDecision("exact_resume", ())


class TrainingCheckpointStore:
    @staticmethod
    def save(payload: Mapping[str, object], path: str | Path) -> Path:
        if payload.get("schema") != TRAINING_CHECKPOINT_SCHEMA:
            raise ValueError("refusing to save an unversioned M03 training checkpoint")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
        try:
            torch.save(dict(payload), temp)
            temp.replace(destination)
        finally:
            if temp.exists():
                temp.unlink()
        return destination

    @staticmethod
    def load(path: str | Path, *, device: torch.device | str = "cpu") -> Mapping[str, object]:
        payload = torch.load(Path(path), map_location=torch.device(device), weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError("training checkpoint payload must be a mapping")
        return payload


__all__ = [
    "CheckpointCompatibilityError", "RESUME_KINDS", "ResumeDecision",
    "TRAINING_CHECKPOINT_SCHEMA", "TRAINING_SEMANTIC_CONTRACT_SCHEMA",
    "TrainingCheckpointStore", "build_semantic_contract", "canonical_json_hash",
    "capture_rng_state", "classify_resume", "compute_source_hash",
    "restore_rng_state", "serialize_config",
]
