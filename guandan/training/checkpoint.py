"""Versioned model checkpoint helpers for the A05 training support layer.

This module deliberately does not contain a trainer.  It provides the small,
strict persistence boundary a trainer can use: protocol versions are checked
before any model or optimizer state is applied, and resume counters/provenance
are kept next to the serialized state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import pickle
import subprocess
import tempfile
from collections.abc import Mapping
from typing import Any

import torch

from guandan.environment_types import (
    ACTION_VERSION,
    ENCODING_VERSION,
    ENV_VERSION,
    RULES_VERSION,
)

from .config import AcceptanceProfile, load_profile


CHECKPOINT_VERSION = 1
PROTOCOL_VERSION_KEYS = ("rules_version", "action_version", "environment_version", "encoding_version")
CURRENT_PROTOCOL_VERSIONS = {
    "rules_version": RULES_VERSION,
    "action_version": ACTION_VERSION,
    "environment_version": ENV_VERSION,
    "encoding_version": ENCODING_VERSION,
}
_AUTO_GIT_COMMIT = object()


class CheckpointError(ValueError):
    """Raised when a checkpoint is malformed or cannot be resumed safely."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised when a checkpoint targets a different protocol or run context."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Metadata required to identify and resume one training checkpoint."""

    checkpoint_version: int
    rules_version: str
    action_version: str
    environment_version: str
    encoding_version: str
    profile: str
    algorithm: str
    git_commit: str | None
    update_count: int
    seed_count: int

    @property
    def update(self) -> int:
        """Short alias used by training loops for the update counter."""

        return self.update_count

    @property
    def seed(self) -> int:
        """Short alias used by training loops for the seed counter."""

        return self.seed_count

    @property
    def protocol_versions(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in PROTOCOL_VERSION_KEYS}

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["update"] = self.update_count
        result["seed"] = self.seed_count
        result["protocol_versions"] = self.protocol_versions
        return result

    def __getitem__(self, key: str) -> Any:
        if key == "update":
            return self.update_count
        if key == "seed":
            return self.seed_count
        if key == "protocol_versions":
            return self.protocol_versions
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc


@dataclass(slots=True)
class TrainingCheckpoint:
    """A loaded checkpoint and its typed resume metadata."""

    metadata: CheckpointMetadata
    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        payload = self.metadata.as_dict()
        payload.update(
            {
                "metadata": self.metadata,
                "model_state": self.model_state,
                "optimizer_state": self.optimizer_state,
            }
        )
        return payload

    def __getitem__(self, key: str) -> Any:
        if key == "metadata":
            return self.metadata
        if key == "model_state":
            return self.model_state
        if key == "optimizer_state":
            return self.optimizer_state
        return self.metadata[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


# Public alias for callers that prefer the shorter name.
LoadedCheckpoint = TrainingCheckpoint


def _strict_counter(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointError(f"{name} must be a non-negative integer")
    return value


def _validate_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointError(f"{name} must be a non-empty string")
    return value


def _git_commit() -> str | None:
    """Return the current commit when this source tree is inside Git."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _profile_name(profile: str | AcceptanceProfile) -> str:
    if isinstance(profile, AcceptanceProfile):
        return profile.name
    if not isinstance(profile, str) or not profile:
        raise CheckpointError("profile must be an acceptance profile name")
    # Validate the name against the same strict source used by the config API.
    load_profile(profile)
    return profile


def _state_dict(value: Any, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "state_dict") and callable(value.state_dict):
        value = value.state_dict()
    if not isinstance(value, Mapping):
        raise CheckpointError(f"{name} must provide a state_dict mapping")
    return dict(value)


def _protocol_versions(value: Mapping[str, Any] | None) -> dict[str, str]:
    versions = dict(CURRENT_PROTOCOL_VERSIONS if value is None else value)
    if set(versions) != set(PROTOCOL_VERSION_KEYS):
        raise CheckpointError(
            "protocol_versions must contain exactly " + ", ".join(PROTOCOL_VERSION_KEYS)
        )
    for key in PROTOCOL_VERSION_KEYS:
        _validate_nonempty_string(versions[key], key)
    return {key: versions[key] for key in PROTOCOL_VERSION_KEYS}


def _metadata_from_payload(payload: Mapping[str, Any]) -> CheckpointMetadata:
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint payload must be a mapping")

    checkpoint_version = payload.get("checkpoint_version")
    if checkpoint_version != CHECKPOINT_VERSION:
        raise IncompatibleCheckpointError(
            f"unsupported checkpoint version: {checkpoint_version!r}; expected {CHECKPOINT_VERSION}"
        )

    versions: dict[str, Any] = {}
    nested_versions = payload.get("protocol_versions")
    if nested_versions is not None:
        if not isinstance(nested_versions, Mapping):
            raise CheckpointError("protocol_versions must be a mapping")
        versions.update(nested_versions)
    for key in PROTOCOL_VERSION_KEYS:
        if key in payload:
            if key in versions and versions[key] != payload[key]:
                raise CheckpointError(f"conflicting {key} declarations in checkpoint")
            versions[key] = payload[key]
    if set(versions) != set(PROTOCOL_VERSION_KEYS):
        raise CheckpointError("checkpoint is missing one or more protocol version fields")
    for key in PROTOCOL_VERSION_KEYS:
        if versions[key] != CURRENT_PROTOCOL_VERSIONS[key]:
            raise IncompatibleCheckpointError(
                f"incompatible {key}: {versions[key]!r}; expected {CURRENT_PROTOCOL_VERSIONS[key]!r}"
            )

    profile = _validate_nonempty_string(payload.get("profile"), "profile")
    try:
        load_profile(profile)
    except Exception as exc:
        raise CheckpointError(f"checkpoint references invalid profile: {profile}") from exc
    algorithm = _validate_nonempty_string(payload.get("algorithm"), "algorithm")
    git_commit = payload.get("git_commit")
    if git_commit is not None:
        _validate_nonempty_string(git_commit, "git_commit")

    update_count = payload.get("update_count", payload.get("update"))
    seed_count = payload.get("seed_count", payload.get("seed"))
    if "update_count" in payload and "update" in payload and payload["update_count"] != payload["update"]:
        raise CheckpointError("conflicting update counter declarations in checkpoint")
    if "seed_count" in payload and "seed" in payload and payload["seed_count"] != payload["seed"]:
        raise CheckpointError("conflicting seed counter declarations in checkpoint")

    return CheckpointMetadata(
        checkpoint_version=CHECKPOINT_VERSION,
        rules_version=versions["rules_version"],
        action_version=versions["action_version"],
        environment_version=versions["environment_version"],
        encoding_version=versions["encoding_version"],
        profile=profile,
        algorithm=algorithm,
        git_commit=git_commit,
        update_count=_strict_counter(update_count, "update_count"),
        seed_count=_strict_counter(seed_count, "seed_count"),
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def save_checkpoint(
    path: str | Path,
    model: Any = None,
    optimizer: Any = None,
    *,
    profile: str | AcceptanceProfile = "local_fast",
    algorithm: str = "ippo",
    update: int = 0,
    seed: int = 0,
    update_count: int | None = None,
    seed_count: int | None = None,
    git_commit: str | None | object = _AUTO_GIT_COMMIT,
    protocol_versions: Mapping[str, Any] | None = None,
    model_state: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
) -> CheckpointMetadata:
    """Save model/optimizer state and versioned resume metadata atomically.

    ``model`` and ``optimizer`` may be live PyTorch objects or callers may pass
    their state dictionaries through ``model_state``/``optimizer_state``.  The
    latter is useful for lightweight checkpoint tests and non-module wrappers.
    """

    if model_state is not None and model is not None:
        raise CheckpointError("pass either model or model_state, not both")
    if optimizer_state is not None and optimizer is not None:
        raise CheckpointError("pass either optimizer or optimizer_state, not both")
    model_state_value = _state_dict(model_state if model_state is not None else model, "model")
    if model_state_value is None:
        raise CheckpointError("model state is required")
    optimizer_state_value = _state_dict(
        optimizer_state if optimizer_state is not None else optimizer, "optimizer"
    )
    if update_count is not None:
        if update != 0 and update != update_count:
            raise CheckpointError("update and update_count disagree")
        update = update_count
    if seed_count is not None:
        if seed != 0 and seed != seed_count:
            raise CheckpointError("seed and seed_count disagree")
        seed = seed_count
    update = _strict_counter(update, "update_count")
    seed = _strict_counter(seed, "seed_count")
    profile_name = _profile_name(profile)
    algorithm = _validate_nonempty_string(algorithm, "algorithm")
    versions = _protocol_versions(protocol_versions)
    commit = _git_commit() if git_commit is _AUTO_GIT_COMMIT else git_commit
    if commit is not None:
        _validate_nonempty_string(commit, "git_commit")

    metadata = CheckpointMetadata(
        checkpoint_version=CHECKPOINT_VERSION,
        rules_version=versions["rules_version"],
        action_version=versions["action_version"],
        environment_version=versions["environment_version"],
        encoding_version=versions["encoding_version"],
        profile=profile_name,
        algorithm=algorithm,
        git_commit=commit,
        update_count=update,
        seed_count=seed,
    )
    payload = metadata.as_dict()
    payload.update(
        {
            "model_state": model_state_value,
            "optimizer_state": optimizer_state_value,
        }
    )
    _atomic_torch_save(payload, Path(path))
    return metadata


def _torch_load(path: Path, map_location: Any) -> Any:
    # ``weights_only=False`` is explicit because the checkpoint envelope is a
    # metadata dictionary as well as tensor state.  The fallback keeps this
    # helper usable with older supported torch releases.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: str | Path,
    model: Any = None,
    optimizer: Any = None,
    *,
    expected_profile: str | AcceptanceProfile | None = None,
    expected_algorithm: str | None = None,
    expected_protocol_versions: Mapping[str, Any] | None = None,
    map_location: Any = "cpu",
) -> TrainingCheckpoint:
    """Load a checkpoint, validate protocol compatibility, and optionally resume.

    Model and optimizer state are applied only after all version and metadata
    checks pass.  This prevents a failed compatibility check from partially
    mutating a caller's live model.
    """

    checkpoint_path = Path(path)
    try:
        payload = _torch_load(checkpoint_path, map_location)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:  # type: ignore[name-defined]
        raise CheckpointError(f"could not load checkpoint {checkpoint_path}: {exc}") from exc
    except Exception as exc:
        raise CheckpointError(f"could not load checkpoint {checkpoint_path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint payload must be a mapping")
    metadata = _metadata_from_payload(payload)

    if expected_profile is not None:
        expected_name = _profile_name(expected_profile)
        if metadata.profile != expected_name:
            raise IncompatibleCheckpointError(
                f"checkpoint profile {metadata.profile!r} does not match expected {expected_name!r}"
            )
    if expected_algorithm is not None:
        expected_algorithm = _validate_nonempty_string(expected_algorithm, "expected_algorithm")
        if metadata.algorithm != expected_algorithm:
            raise IncompatibleCheckpointError(
                f"checkpoint algorithm {metadata.algorithm!r} does not match expected {expected_algorithm!r}"
            )
    if expected_protocol_versions is not None:
        expected = _protocol_versions(expected_protocol_versions)
        if metadata.protocol_versions != expected:
            raise IncompatibleCheckpointError(
                f"checkpoint protocol versions {metadata.protocol_versions!r} do not match expected {expected!r}"
            )

    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise CheckpointError("checkpoint model_state must be a mapping")
    optimizer_state = payload.get("optimizer_state")
    if optimizer_state is not None and not isinstance(optimizer_state, Mapping):
        raise CheckpointError("checkpoint optimizer_state must be a mapping or None")

    # All validation above happens before either load_state_dict call.
    if model is not None:
        if not hasattr(model, "load_state_dict"):
            raise CheckpointError("model must provide load_state_dict")
        model.load_state_dict(model_state)
    if optimizer is not None:
        if optimizer_state is None:
            raise CheckpointError("checkpoint does not contain optimizer state")
        if not hasattr(optimizer, "load_state_dict"):
            raise CheckpointError("optimizer must provide load_state_dict")
        optimizer.load_state_dict(optimizer_state)

    return TrainingCheckpoint(metadata=metadata, model_state=model_state, optimizer_state=optimizer_state)


def resume_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    **kwargs: Any,
) -> TrainingCheckpoint:
    """Load and apply a checkpoint, returning counters/provenance for a loop."""

    return load_checkpoint(path, model=model, optimizer=optimizer, **kwargs)


# Explicit alias for code that names the operation as a load-and-resume.
load_and_resume_checkpoint = resume_checkpoint


__all__ = [
    "CHECKPOINT_VERSION",
    "PROTOCOL_VERSION_KEYS",
    "CURRENT_PROTOCOL_VERSIONS",
    "CheckpointError",
    "IncompatibleCheckpointError",
    "CheckpointMetadata",
    "TrainingCheckpoint",
    "LoadedCheckpoint",
    "save_checkpoint",
    "load_checkpoint",
    "resume_checkpoint",
    "load_and_resume_checkpoint",
]



