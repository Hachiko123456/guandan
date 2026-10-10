# SPDX-License-Identifier: MIT
"""Kaggle dataset/profile validation built on the shared stdlib bootstrap resolver."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .bootstrap import (
    BOOTSTRAP_NAME, DATASET_MANIFEST_NAME, DATASET_MANIFEST_SCHEMA, DatasetInput,
    RECOVERY_MANIFEST_SCHEMA, RecoveryInput, SOURCE_MANIFEST_SCHEMA, SUPPORTED_MODES,
    SourceInput, canonical_hash, discover_dataset, discover_recovery, materialize_recovery_input, resolve_dataset_input,
    validate_dataset_structure, validate_recovery_directory,
)


def validate_dataset_manifest(payload: Mapping[str, object], *, manifest_path: Path | None = None) -> None:
    validate_dataset_structure(payload)
    mode = str(payload["mode"])
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or mode not in profiles:
        raise ValueError(f"dataset manifest must include profiles.{mode}")
    profile = profiles[mode]
    if not isinstance(profile, Mapping) or profile.get("schema") != "guandan-marvel-kaggle-profile-v1":
        raise ValueError(f"profiles.{mode} is missing a versioned profile")
    if profile.get("mode") != mode:
        raise ValueError(f"profiles.{mode}.mode mismatch")
    training_modes = {"pilot", "resume", "full", "single_gpu_control", "dual_gpu_control"}
    fresh_training_modes = {"pilot", "full", "single_gpu_control", "dual_gpu_control"}
    if mode in training_modes:
        ddp = profile.get("ddp")
        if not isinstance(ddp, Mapping):
            raise ValueError("training profiles require a DDP object")
        world_size = int(ddp.get("world_size", 0))
        if world_size not in {1, 2} or ddp.get("launcher") != "torchrun":
            raise ValueError("training profiles require torchrun world_size 1 or 2")
        if mode in {"pilot", "resume", "full", "dual_gpu_control"} and world_size != 2:
            raise ValueError(f"{mode} requires torchrun world_size=2")
        if mode == "single_gpu_control" and world_size != 1:
            raise ValueError("single_gpu_control requires torchrun world_size=1")
    if mode in fresh_training_modes:
        training = profile.get("training")
        if not isinstance(training, Mapping):
            raise ValueError(f"{mode} profile requires a training object")
        world_size = int(profile["ddp"]["world_size"])
        for key in ("rollout_batch_size", "collect_batch_size"):
            if int(training.get(key, 0)) <= 0 or int(training[key]) % world_size:
                raise ValueError(f"training.{key} must be positive and divisible by world_size={world_size}")
    if mode in {"single_gpu_control", "dual_gpu_control"}:
        benchmark = profile.get("benchmark")
        if not isinstance(benchmark, Mapping) or benchmark.get("schema") != "guandan-marvel-m08-benchmark-contract-v1":
            raise ValueError(f"{mode} requires a versioned benchmark contract")
        expected_control = "single_gpu" if mode == "single_gpu_control" else "dual_gpu_ddp"
        if benchmark.get("control") != expected_control:
            raise ValueError(f"{mode} benchmark control mismatch")
        workload = {"algorithm": profile.get("algorithm"), "training": dict(profile["training"])}
        if benchmark.get("workload_hash") != canonical_hash(workload):
            raise ValueError(f"{mode} workload hash mismatch")
        warmup = int(benchmark.get("warmup_iterations", -1))
        measured = int(benchmark.get("measured_iterations", -1))
        target = int(profile["training"].get("target_iterations", 0))
        if warmup < 0 or measured <= 0 or warmup + measured != target:
            raise ValueError(f"{mode} benchmark iteration partition is invalid")
        if int(profile["training"].get("evaluation_interval", -1)) != 0:
            raise ValueError(f"{mode} must disable periodic evaluation")
    if mode == "resume":
        controls = profile.get("resume")
        if not isinstance(controls, Mapping) or int(controls.get("additional_iterations", 0)) <= 0:
            raise ValueError("resume profile requires positive resume.additional_iterations")
        forbidden = sorted(set(controls) - {
            "additional_iterations", "checkpoint_interval", "evaluation_interval", "max_wall_seconds",
        })
        if forbidden:
            raise ValueError(f"resume profile contains forbidden semantic overrides: {forbidden}")
        if "training" in profile or "algorithm" in profile:
            raise ValueError("resume profile must inherit algorithm/training semantics from recovery")
    recovery = payload.get("recovery")
    if not isinstance(recovery, Mapping):
        raise ValueError("dataset manifest recovery object is required")
    if mode in {"resume", "eval_only"} and not bool(recovery.get("required", False)):
        raise ValueError(f"{mode} dataset manifest must require recovery")
    if manifest_path is not None:
        resolved = resolve_dataset_input(manifest_path)
        if resolved.manifest != dict(payload):
            raise ValueError("dataset manifest changed during validation")


__all__ = [
    "BOOTSTRAP_NAME", "DATASET_MANIFEST_NAME", "DATASET_MANIFEST_SCHEMA", "DatasetInput",
    "RECOVERY_MANIFEST_SCHEMA", "RecoveryInput", "SOURCE_MANIFEST_SCHEMA", "SUPPORTED_MODES",
    "SourceInput", "discover_dataset", "discover_recovery", "materialize_recovery_input", "resolve_dataset_input",
    "validate_dataset_manifest", "validate_recovery_directory",
]
