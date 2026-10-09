# SPDX-License-Identifier: MIT
"""Versioned M03 training checkpoints and conservative resume classification."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

TRAINING_CHECKPOINT_SCHEMA = "guandan-marvel-training-checkpoint-v4"
PRE_BACKEND_IDENTITY_CHECKPOINT_SCHEMA = "guandan-marvel-training-checkpoint-v3"
PRE_REPAIR_TRAINING_CHECKPOINT_SCHEMA = "guandan-marvel-training-checkpoint-v2"
TRAINING_SEMANTIC_CONTRACT_SCHEMA = "guandan-marvel-training-contract-v3"
SEMANTIC_CONFIG_SCHEMA = "guandan-marvel-semantic-config-v1"
SOURCE_FINGERPRINT_SCHEMA = "guandan-marvel-source-fingerprint-v1"
EXECUTION_FINGERPRINT_SCHEMA = "guandan-marvel-execution-fingerprint-v1"
BACKEND_IDENTITY_SCHEMA = "guandan-marvel-backend-identity-v1"
RESUME_KINDS = frozenset({
    "exact_resume", "compatible_resume", "warm_restart", "migration", "evaluation_only",
})
NON_SEMANTIC_CONFIG_PATHS = frozenset({
    "project_name", "run_name", "logging_dir", "debug", "resume_from",
    "total_iterations", "checkpoint_interval", "evaluation_interval",
    "param_stats_interval",
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


def _hash_payload(mapping: Mapping[str, object]) -> str:
    return canonical_json_hash({key: value for key, value in mapping.items() if key != "hash"})


def validate_declared_hash(mapping: Mapping[str, object], *, name: str) -> None:
    declared = mapping.get("hash")
    if not isinstance(declared, str) or not declared:
        raise CheckpointCompatibilityError(f"{name}.hash is missing")
    actual = _hash_payload(mapping)
    if declared != actual:
        raise CheckpointCompatibilityError(
            f"{name}.hash is inconsistent: declared={declared}, recomputed={actual}"
        )


def _mapping_differences(left: object, right: object, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        keys = sorted(set(left) | set(right), key=str)
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                differences.append(f"{path}: checkpoint=<missing> expected={right[key]!r}")
            elif key not in right:
                differences.append(f"{path}: checkpoint={left[key]!r} expected=<missing>")
            else:
                differences.extend(_mapping_differences(left[key], right[key], path))
        return differences
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [f"{prefix}: checkpoint length={len(left)} expected length={len(right)}"]
        differences: list[str] = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            differences.extend(_mapping_differences(a, b, f"{prefix}[{index}]"))
        return differences
    if left != right:
        return [f"{prefix}: checkpoint={left!r} expected={right!r}"]
    return []


def config_differences(
    checkpoint_config: Mapping[str, object],
    expected_config: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    all_differences = _mapping_differences(checkpoint_config, expected_config)
    allowed: list[str] = []
    forbidden: list[str] = []
    for difference in all_differences:
        path = difference.split(":", 1)[0]
        root = path.split(".", 1)[0]
        (allowed if root in NON_SEMANTIC_CONFIG_PATHS else forbidden).append(difference)
    return allowed, forbidden


def build_semantic_configuration(config: object, *, algorithm: str) -> dict[str, object]:
    resolved = serialize_config(config)
    model = dict(resolved["model"])
    game = dict(resolved["game"])
    ppo = dict(resolved["ppo"])
    vrpo = dict(resolved["vrpo"])
    optimizer = dict(resolved["optimizer"])
    schedule = dict(resolved["schedule"])
    semantic = {
        "schema": SEMANTIC_CONFIG_SCHEMA,
        "algorithm": str(algorithm),
        "model": model,
        "game": game,
        "discount_gamma": resolved["discount_gamma"],
        "policy_update": {
            "base_lr": ppo["base_lr"],
            "agent_epoch": ppo["agent_epoch"],
            "gae_lambda": ppo["gae_lambda"],
            "base_clip_coef": ppo["base_clip_coef"],
            "base_reg_coef": ppo["base_reg_coef"],
            "value_loss_coef": resolved["value_loss_coef"],
            "mini_batches": resolved["mini_batches"],
        },
        "critic_update": None if algorithm == "ippo" else {
            "base_lr": vrpo["base_lr"],
            "gae_lambda": vrpo["gae_lambda"],
            "aux_epoch": vrpo["aux_epoch"],
        },
        "optimizer": optimizer,
        "schedule": {
            **schedule,
            "lr_stable_iterations": resolved["lr_stable_iterations"],
            "reg_stable_iterations": resolved["reg_stable_iterations"],
        },
        "merged_policy": {
            "beta": resolved["agent_merging_beta"],
            "definition": "current_policy_copy" if resolved["agent_merging_beta"] == 0.0 else "exponential_moving_average",
        },
        "sampling_and_replay": {
            "seed": resolved["seed"],
            "rollout_batch_size": resolved["rollout_batch_size"],
            "collect_batch_size": resolved["collect_batch_size"],
            "replay_buffer_scale": resolved["replay_buffer_scale"],
            "replay_buffer_device": vrpo["replay_buffer_device"],
            "checkpoint_replay_buffer": resolved["checkpoint_replay_buffer"],
        },
        "execution_semantics": {
            "deterministic": resolved["deterministic"],
            "deterministic_disable_tf32": resolved["deterministic_disable_tf32"],
            "amp_enabled": resolved["amp_enabled"],
            "compile": dict(resolved["compile"]),
            "ddp": dict(resolved["ddp"]),
        },
    }
    return {**semantic, "hash": canonical_json_hash(semantic)}


def build_semantic_contract(
    *,
    semantic_config: Mapping[str, object],
    sampler_contract: str,
    sampler_state_schema: str,
    buffer_schema: str,
    q_target_contract: str | None,
) -> dict[str, object]:
    validate_declared_hash(semantic_config, name="semantic_config")
    contract = {
        "schema": TRAINING_SEMANTIC_CONTRACT_SCHEMA,
        "semantic_config": dict(semantic_config),
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


def compute_source_fingerprint(
    repository_root: str | Path,
    components: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    root = Path(repository_root)
    component_payload: dict[str, object] = {}
    for component, paths in sorted(components.items()):
        files = []
        for relative in sorted(set(paths)):
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append({
                "path": relative.replace("\\", "/"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            })
        component_payload[str(component)] = {
            "files": files,
            "hash": canonical_json_hash(files),
        }
    fingerprint = {
        "schema": SOURCE_FINGERPRINT_SCHEMA,
        "components": component_payload,
    }
    return {**fingerprint, "hash": canonical_json_hash(fingerprint)}


def validate_source_fingerprint(fingerprint: Mapping[str, object], *, name: str) -> None:
    if fingerprint.get("schema") != SOURCE_FINGERPRINT_SCHEMA:
        raise CheckpointCompatibilityError(f"{name}.schema is unsupported")
    components = fingerprint.get("components")
    if not isinstance(components, Mapping):
        raise CheckpointCompatibilityError(f"{name}.components is missing")
    for component, payload in components.items():
        if not isinstance(payload, Mapping):
            raise CheckpointCompatibilityError(f"{name}.components.{component} is invalid")
        files = payload.get("files")
        if not isinstance(files, list):
            raise CheckpointCompatibilityError(f"{name}.components.{component}.files is invalid")
        actual = canonical_json_hash(files)
        if payload.get("hash") != actual:
            raise CheckpointCompatibilityError(
                f"{name}.components.{component}.hash is inconsistent: "
                f"declared={payload.get('hash')}, recomputed={actual}"
            )
    validate_declared_hash(fingerprint, name=name)


def build_execution_fingerprint(device: torch.device | str) -> dict[str, object]:
    resolved = torch.device(device)
    payload = {
        "schema": EXECUTION_FINGERPRINT_SCHEMA,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device_type": resolved.type,
        "cuda_available": torch.cuda.is_available(),
    }
    return {**payload, "hash": canonical_json_hash(payload)}


def build_backend_identity(game: object) -> dict[str, object]:
    backend = str(getattr(game, "BACKEND", "")).strip()
    if backend not in {"python_reference", "cpp"}:
        raise CheckpointCompatibilityError(
            f"runtime game must declare BACKEND as python_reference or cpp, got {backend!r}"
        )
    identifier = str(getattr(game, "identifier", getattr(game, "GAME_IDENTIFIER", "")))
    protocol = str(getattr(game, "protocol_version", ""))
    module_name = str(getattr(game, "__name__", f"{type(game).__module__}.{type(game).__qualname__}"))
    implementation: dict[str, object]
    if backend == "cpp":
        builder = getattr(game, "get_backend_build_identity", None)
        if not callable(builder):
            raise CheckpointCompatibilityError("cpp backend does not expose build identity")
        implementation = dict(builder())
        if implementation.get("source_matches_loaded_binary") is False:
            raise CheckpointCompatibilityError(
                "loaded cpp extension was built from different native sources"
            )
        required = {
            "schema", "extension_module", "extension_path", "extension_sha256",
            "embedded_source_sha256", "compiler", "cxx_standard",
        }
        missing = sorted(required - set(implementation))
        if missing:
            raise CheckpointCompatibilityError(
                "cpp build identity is incomplete: " + ", ".join(missing)
            )
    else:
        implementation = {
            "kind": "python_source",
            "module": module_name,
        }
    payload = {
        "schema": BACKEND_IDENTITY_SCHEMA,
        "backend": backend,
        "module": module_name,
        "game_identifier": identifier,
        "protocol_version": protocol,
        "implementation": implementation,
    }
    return {**payload, "hash": canonical_json_hash(payload)}


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


def classify_resume(
    checkpoint: Mapping[str, object],
    *,
    expected_contract: Mapping[str, object],
    expected_source_fingerprint: Mapping[str, object],
    expected_execution_fingerprint: Mapping[str, object],
    expected_backend_identity: Mapping[str, object],
    expected_config: Mapping[str, object],
    require_replay: bool,
    for_evaluation: bool = False,
) -> ResumeDecision:
    schema = checkpoint.get("schema")
    legacy_protocols = "|".join(str(checkpoint.get(key, "")) for key in (
        "protocol_version", "action_version", "environment_version", "encoding_version",
    ))
    if "GD-ACTION-0.1" in legacy_protocols or "GD-ENV-0.1" in legacy_protocols or schema in {None, "guandan-training-checkpoint-v1"}:
        if for_evaluation:
            return ResumeDecision("evaluation_only", ("legacy V1 checkpoint",))
        raise CheckpointCompatibilityError("legacy V1 checkpoint cannot resume MARVEL V2 training")
    if schema == PRE_REPAIR_TRAINING_CHECKPOINT_SCHEMA:
        if for_evaluation:
            return ResumeDecision("evaluation_only", ("pre-repair M03 checkpoint",))
        return ResumeDecision(
            "migration",
            ("pre-repair checkpoint lacks semantic_config v1 and source_fingerprint v1",),
        )
    if schema == PRE_BACKEND_IDENTITY_CHECKPOINT_SCHEMA:
        if for_evaluation:
            return ResumeDecision("evaluation_only", ("pre-backend-identity checkpoint",))
        return ResumeDecision(
            "migration",
            ("checkpoint lacks explicit runtime backend and native build identity",),
        )
    if schema != TRAINING_CHECKPOINT_SCHEMA:
        raise CheckpointCompatibilityError(f"unsupported checkpoint schema {schema!r}")
    if for_evaluation:
        return ResumeDecision("evaluation_only", ("explicit evaluation load",))

    checkpoint_backend = checkpoint.get("backend_identity")
    if not isinstance(checkpoint_backend, Mapping):
        return ResumeDecision("migration", ("backend_identity is missing",))
    validate_declared_hash(checkpoint_backend, name="backend_identity")
    validate_declared_hash(expected_backend_identity, name="expected_backend_identity")
    backend_diffs = _mapping_differences(
        checkpoint_backend, expected_backend_identity, "backend_identity"
    )
    if backend_diffs:
        return ResumeDecision("migration", tuple(backend_diffs))
    if checkpoint.get("backend") != expected_backend_identity.get("backend"):
        return ResumeDecision("migration", (
            f"backend: checkpoint={checkpoint.get('backend')!r} "
            f"expected={expected_backend_identity.get('backend')!r}",
        ))

    checkpoint_contract = checkpoint.get("semantic_contract")
    if not isinstance(checkpoint_contract, Mapping):
        return ResumeDecision("migration", ("semantic_contract is missing",))
    validate_declared_hash(checkpoint_contract, name="semantic_contract")
    validate_declared_hash(expected_contract, name="expected_semantic_contract")
    checkpoint_semantic = checkpoint_contract.get("semantic_config")
    expected_semantic = expected_contract.get("semantic_config")
    if not isinstance(checkpoint_semantic, Mapping) or not isinstance(expected_semantic, Mapping):
        return ResumeDecision("migration", ("semantic_config is missing",))
    validate_declared_hash(checkpoint_semantic, name="semantic_contract.semantic_config")
    validate_declared_hash(expected_semantic, name="expected_semantic_contract.semantic_config")
    contract_diffs = _mapping_differences(checkpoint_contract, expected_contract, "semantic_contract")
    if contract_diffs:
        return ResumeDecision("migration", tuple(contract_diffs))

    resolved_config = checkpoint.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        return ResumeDecision("migration", ("resolved_config is missing",))
    declared_config_hash = checkpoint.get("resolved_config_hash")
    actual_config_hash = canonical_json_hash(resolved_config)
    if declared_config_hash != actual_config_hash:
        raise CheckpointCompatibilityError(
            "resolved_config_hash is inconsistent: "
            f"declared={declared_config_hash}, recomputed={actual_config_hash}"
        )

    checkpoint_source = checkpoint.get("source_fingerprint")
    if not isinstance(checkpoint_source, Mapping):
        return ResumeDecision("migration", ("source_fingerprint is missing",))
    validate_source_fingerprint(checkpoint_source, name="source_fingerprint")
    validate_source_fingerprint(expected_source_fingerprint, name="expected_source_fingerprint")
    if checkpoint.get("source_hash") != checkpoint_source.get("hash"):
        raise CheckpointCompatibilityError(
            "source_hash is inconsistent with source_fingerprint.hash"
        )
    source_diffs = _mapping_differences(checkpoint_source, expected_source_fingerprint, "source_fingerprint")
    if source_diffs:
        return ResumeDecision("migration", tuple(source_diffs))

    checkpoint_execution = checkpoint.get("execution_fingerprint")
    if not isinstance(checkpoint_execution, Mapping):
        return ResumeDecision("migration", ("execution_fingerprint is missing",))
    validate_declared_hash(checkpoint_execution, name="execution_fingerprint")
    validate_declared_hash(expected_execution_fingerprint, name="expected_execution_fingerprint")
    execution_diffs = _mapping_differences(checkpoint_execution, expected_execution_fingerprint, "execution_fingerprint")
    if execution_diffs:
        return ResumeDecision("migration", tuple(execution_diffs))

    required = {
        "agent_resume_state_dict", "agent_state_dict", "agent_optimizer_state_dict",
        "iteration", "durable_updates", "counters", "rng_state", "sampler_state",
        "schedule_state", "merged_policy", "resolved_config", "resolved_config_hash",
        "source_fingerprint", "source_hash", "execution_fingerprint",
        "backend", "backend_identity",
    }
    if checkpoint_semantic.get("algorithm") == "vrpo":
        required.update({"critic_state_dict", "critic_optimizer_state_dict"})
    missing = sorted(required - set(checkpoint))
    if require_replay and "replay_buffer_state" not in checkpoint:
        missing.append("replay_buffer_state")
    if missing:
        return ResumeDecision("warm_restart", tuple(f"missing {name}" for name in missing))

    allowed_config, forbidden_config = config_differences(resolved_config, expected_config)
    if forbidden_config:
        return ResumeDecision("migration", tuple(forbidden_config))
    if allowed_config:
        return ResumeDecision("compatible_resume", tuple(allowed_config))
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
    "BACKEND_IDENTITY_SCHEMA", "CheckpointCompatibilityError",
    "EXECUTION_FINGERPRINT_SCHEMA", "NON_SEMANTIC_CONFIG_PATHS",
    "PRE_BACKEND_IDENTITY_CHECKPOINT_SCHEMA", "PRE_REPAIR_TRAINING_CHECKPOINT_SCHEMA",
    "RESUME_KINDS", "ResumeDecision", "SEMANTIC_CONFIG_SCHEMA",
    "SOURCE_FINGERPRINT_SCHEMA", "TRAINING_CHECKPOINT_SCHEMA",
    "TRAINING_SEMANTIC_CONTRACT_SCHEMA", "TrainingCheckpointStore",
    "build_backend_identity", "build_execution_fingerprint",
    "build_semantic_configuration",
    "build_semantic_contract", "canonical_json_hash", "capture_rng_state",
    "classify_resume", "compute_source_fingerprint", "config_differences",
    "restore_rng_state", "serialize_config", "validate_declared_hash",
    "validate_source_fingerprint",
]
