"""Safe v2 checkpoint boundary; the trainer owns runtime/collector restoration.

``training_state`` is an opaque safe-value dictionary, not a collector API. It
can hold environment bytes, CPU/CUDA RNG tensors, primitive NumPy generator
states, counters and provenance. Loading never unpickles environment bytes or
applies RNG state. Callers must require and restore their own resume fields.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import torch

from guandan.environment_types import ACTION_VERSION, ENCODING_VERSION, ENV_VERSION, RULES_VERSION
from guandan.model import PolicyValueNet

from .config import AcceptanceProfile, ProfileConfigError, load_profile

CHECKPOINT_VERSION = 2
MODEL_VERSION = "GD-MODEL-MLP-0.2"
TRAINING_VERSION = "GD-TRAIN-0.2"
PROTOCOL_VERSION_KEYS = ("rules_version", "action_version", "environment_version", "encoding_version")
CURRENT_PROTOCOL_VERSIONS = {
    "rules_version": RULES_VERSION,
    "action_version": ACTION_VERSION,
    "environment_version": ENV_VERSION,
    "encoding_version": ENCODING_VERSION,
}
_MODEL_ATTRIBUTES = (
    "token_vocab_size", "num_state_channels", "hidden_dim", "pad_token_id",
    "max_observation_tokens", "max_legal_next_tokens",
)
_AUTO_GIT_COMMIT = object()


class CheckpointError(ValueError):
    """A checkpoint is malformed or cannot be resumed safely."""


class IncompatibleCheckpointError(CheckpointError):
    """A checkpoint targets a different version, architecture or run context."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
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
    model_version: str = MODEL_VERSION
    training_version: str = TRAINING_VERSION

    @property
    def update(self) -> int:
        return self.update_count

    @property
    def seed(self) -> int:
        return self.seed_count

    @property
    def protocol_versions(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in PROTOCOL_VERSION_KEYS}

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "update": self.update, "seed": self.seed,
                "protocol_versions": self.protocol_versions}

    def __getitem__(self, key: str) -> Any:
        if key not in self.as_dict():
            raise KeyError(key)
        return getattr(self, key)


@dataclass(slots=True)
class TrainingCheckpoint:
    metadata: CheckpointMetadata
    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any] | None
    model_config: dict[str, Any] = field(default_factory=dict)
    training_state: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {**self.metadata.as_dict(), "metadata": self.metadata,
                "model_state": self.model_state, "optimizer_state": self.optimizer_state,
                "model_config": self.model_config, "training_state": self.training_state}

    def __getitem__(self, key: str) -> Any:
        if key in {"metadata", "model_state", "optimizer_state", "model_config", "training_state"}:
            return getattr(self, key)
        return self.metadata[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


LoadedCheckpoint = TrainingCheckpoint


def _strict_counter(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CheckpointError(f"{name} must be a non-negative integer")
    return value


def _string(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise CheckpointError(f"{name} must be a non-empty string")
    return value


def _safe_tree(value: Any, name: str, ancestors: frozenset[int] = frozenset()) -> None:
    """Reject unsafe objects, cycles and nonfinite data even on the save path."""
    if len(ancestors) > 64:
        raise CheckpointError(f"{name}: nesting exceeds 64 levels")
    if value is None or type(value) in (str, bytes, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointError(f"{name}: nonfinite number")
        return
    if type(value) is torch.Tensor:
        if vars(value):
            raise CheckpointError(f"{name}: tensors must not carry custom Python attributes")
        if value.device.type == "meta" or value.layout != torch.strided or value.is_quantized:
            raise CheckpointError(f"{name}: expected a materialized dense tensor")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise CheckpointError(f"{name}: nonfinite tensor")
        return
    if type(value) not in (dict, OrderedDict, list, tuple):
        raise CheckpointError(f"{name}: unsupported object type {type(value).__name__}")
    if id(value) in ancestors:
        raise CheckpointError(f"{name}: cyclic container")
    ancestors = ancestors | {id(value)}
    if isinstance(value, dict):
        for key, child in value.items():
            if type(key) not in (str, int):
                raise CheckpointError(f"{name}: mapping keys must be strings or integers")
            _safe_tree(child, f"{name}[{key!r}]", ancestors)
        # PyTorch state_dict uses this attribute for module-local versions.
        if isinstance(value, OrderedDict):
            if set(vars(value)) - {"_metadata"}:
                raise CheckpointError(f"{name}: unsupported mapping attributes")
            if hasattr(value, "_metadata"):
                _safe_tree(value._metadata, f"{name}._metadata", ancestors)
    else:
        for index, child in enumerate(value):
            _safe_tree(child, f"{name}[{index}]", ancestors)


def _safe_dict(value: Any, name: str) -> dict:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise CheckpointError(f"{name} must be a dictionary with string keys")
    _safe_tree(value, name)
    return value


def _same(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is torch.Tensor:
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            check=True, capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _profile_name(profile: str | AcceptanceProfile) -> str:
    name = profile.name if isinstance(profile, AcceptanceProfile) else profile
    _string(name, "profile")
    try:
        load_profile(name)
    except ProfileConfigError as exc:
        raise CheckpointError(f"invalid profile: {name!r}") from exc
    return name


def _algorithm(value: Any, profile: str) -> str:
    value = _string(value, "algorithm")
    if value not in load_profile(profile).algorithms:
        raise CheckpointError(f"unsupported algorithm: {value!r}")
    return value


def _state_dict(value: Any, name: str) -> Mapping | None:
    if value is None:
        return None
    if callable(getattr(value, "state_dict", None)):
        value = value.state_dict()
    if not isinstance(value, Mapping):
        raise CheckpointError(f"{name} must provide a state_dict mapping")
    _safe_tree(value, name)
    return value


def _model_state(value: Any, target: Any = None) -> None:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise CheckpointError("model_state must be a string-keyed tensor mapping")
    _safe_tree(value, "model_state")
    if any(type(tensor) is not torch.Tensor for tensor in value.values()):
        raise CheckpointError("model_state entries must be tensors")
    if target is None:
        return
    current = _state_dict(target, "target model")
    if current is None or set(current) != set(value):
        raise IncompatibleCheckpointError("model_state keys do not match target model")
    for key, tensor in value.items():
        other = current[key]
        if tensor.shape != other.shape or tensor.dtype != other.dtype:
            raise IncompatibleCheckpointError(f"model_state[{key!r}] shape/dtype does not match target model")


def _inferred_config(model: Any, state: Mapping) -> dict[str, Any]:
    if isinstance(model, PolicyValueNet):
        return {key: getattr(model, key) for key in _MODEL_ATTRIBUTES}
    if isinstance(model, torch.nn.ModuleDict):
        return {name: _inferred_config(child, child.state_dict()) for name, child in model.items()}
    result = {"state_shapes": {key: list(value.shape) for key, value in state.items()}}
    if isinstance(model, torch.nn.Module):
        result["class"] = f"{type(model).__module__}.{type(model).__qualname__}"
    return result


def _protocol_versions(value: Mapping[str, Any] | None) -> dict[str, str]:
    if value is not None and not isinstance(value, Mapping):
        raise CheckpointError("protocol_versions must be a mapping")
    versions = dict(CURRENT_PROTOCOL_VERSIONS if value is None else value)
    if set(versions) != set(PROTOCOL_VERSION_KEYS):
        raise CheckpointError("protocol_versions must contain exactly " + ", ".join(PROTOCOL_VERSION_KEYS))
    return {key: _string(versions[key], key) for key in PROTOCOL_VERSION_KEYS}


def _metadata_from_payload(payload: Mapping) -> CheckpointMetadata:
    version = payload.get("checkpoint_version")
    if type(version) is not int or version != CHECKPOINT_VERSION:
        raise IncompatibleCheckpointError(f"unsupported checkpoint_version: {version!r}; expected {CHECKPOINT_VERSION}")
    for key, expected in (("model_version", MODEL_VERSION), ("training_version", TRAINING_VERSION)):
        if payload.get(key) != expected:
            raise IncompatibleCheckpointError(f"incompatible {key}: {payload.get(key)!r}; expected {expected!r}")
    nested = payload.get("protocol_versions", {})
    if not isinstance(nested, Mapping):
        raise CheckpointError("protocol_versions must be a mapping")
    versions = dict(nested)
    for key in PROTOCOL_VERSION_KEYS:
        if key in payload:
            if key in versions and versions[key] != payload[key]:
                raise CheckpointError(f"conflicting {key} declarations in checkpoint")
            versions[key] = payload[key]
    versions = _protocol_versions(versions)
    for key, expected in CURRENT_PROTOCOL_VERSIONS.items():
        if versions[key] != expected:
            raise IncompatibleCheckpointError(f"incompatible {key}: {versions[key]!r}; expected {expected!r}")
    profile = _profile_name(payload.get("profile"))
    algorithm = _algorithm(payload.get("algorithm"), profile)
    if "git_commit" not in payload:
        raise CheckpointError("checkpoint is missing git_commit")
    commit = payload["git_commit"]
    if commit is not None:
        _string(commit, "git_commit")
    counters = {}
    for long, short in (("update_count", "update"), ("seed_count", "seed")):
        counter = _strict_counter(payload.get(long, payload.get(short)), long)
        if short in payload and _strict_counter(payload[short], short) != counter:
            raise CheckpointError(f"conflicting {short} counter declarations in checkpoint")
        counters[long] = counter
    return CheckpointMetadata(
        checkpoint_version=version, **versions, profile=profile, algorithm=algorithm,
        git_commit=commit, **counters, model_version=MODEL_VERSION, training_version=TRAINING_VERSION,
    )


def _parameter_names(optimizer: Any, model: Any) -> list[list[str]] | None:
    if not isinstance(optimizer, torch.optim.Optimizer) or not isinstance(model, torch.nn.Module):
        return None
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        return [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]
    except KeyError as exc:
        raise CheckpointError("optimizer contains a parameter outside the supplied model") from exc


def _optimizer_state(value: Any, target: Any = None, model: Any = None) -> None:
    if value is None:
        if target is not None:
            raise CheckpointError("checkpoint does not contain optimizer state")
        return
    if not isinstance(value, Mapping) or set(value) != {"state", "param_groups"}:
        raise CheckpointError("optimizer_state must contain state and param_groups")
    _safe_tree(value, "optimizer_state")
    states, groups = value["state"], value["param_groups"]
    if not isinstance(states, dict) or not isinstance(groups, list):
        raise CheckpointError("optimizer_state has invalid state/param_groups")
    ids = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise CheckpointError("optimizer_state param_groups require params lists")
        ids.extend(_strict_counter(pid, "optimizer parameter id") for pid in group["params"])
        if "param_names" in group:
            names = group["param_names"]
            if (not isinstance(names, list) or len(names) != len(group["params"])
                    or any(type(name) is not str for name in names)):
                raise CheckpointError("optimizer_state has invalid param_names")
    if len(ids) != len(set(ids)) or any(type(pid) is not int or pid not in ids for pid in states):
        raise CheckpointError("optimizer_state has duplicate or unknown parameter ids")
    # Torch serializes distinct parameters in group order as consecutive IDs.
    # Reject ID reordering rather than silently attaching moments to another
    # same-shaped parameter; tensor shape checks alone cannot detect that.
    if ids != list(range(len(ids))):
        raise CheckpointError("optimizer_state parameter ids/order are not canonical")
    for state in states.values():
        _safe_dict(state, "optimizer parameter state")
    if target is None:
        return
    if not isinstance(target, torch.optim.Optimizer):
        raise CheckpointError("optimizer must be a torch.optim.Optimizer")
    if len(groups) != len(target.param_groups):
        raise IncompatibleCheckpointError("optimizer_state parameter group count mismatch")
    names = _parameter_names(target, model)
    for index, (saved, current) in enumerate(zip(groups, target.param_groups)):
        if len(saved["params"]) != len(current["params"]):
            raise IncompatibleCheckpointError("optimizer_state parameter group size mismatch")
        required = set(current) - {"param_names", "initial_lr"}
        if not required.issubset(saved):
            raise CheckpointError("optimizer_state is missing parameter group fields")
        expected_names = names[index] if names is not None else current.get("param_names")
        if "param_names" in saved and expected_names is not None and saved["param_names"] != expected_names:
            raise IncompatibleCheckpointError("optimizer_state parameter names/order mismatch")
        for pid, parameter in zip(saved["params"], current["params"]):
            state = states.get(pid, {})
            if not state:  # Adam legitimately has no state before a parameter's first step.
                continue
            if isinstance(target, (torch.optim.Adam, torch.optim.AdamW)):
                required_state = {"step", "exp_avg", "exp_avg_sq"}
                if saved.get("amsgrad"):
                    required_state.add("max_exp_avg_sq")
                if not required_state.issubset(state):
                    raise CheckpointError("optimizer_state is missing Adam state fields")
                for key in required_state - {"step"}:
                    if type(state[key]) is not torch.Tensor:
                        raise CheckpointError(f"optimizer_state {key} must be a tensor")
            for key, tensor in state.items():
                if key == "step":
                    if type(tensor) is torch.Tensor:
                        if tensor.ndim != 0 or tensor.is_complex():
                            raise CheckpointError("optimizer_state step must be a scalar counter")
                        tensor = tensor.item()
                    if type(tensor) not in (int, float) or tensor < 0 or int(tensor) != tensor:
                        raise CheckpointError("optimizer_state step must be a non-negative counter")
                elif type(tensor) is torch.Tensor and (tensor.shape != parameter.shape or tensor.dtype != parameter.dtype):
                    raise IncompatibleCheckpointError(f"optimizer_state {key} shape/dtype mismatch")


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
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
    model_config: dict | None = None,
    training_state: dict | None = None,
) -> CheckpointMetadata:
    """Atomically save a v2 envelope, retaining the existing caller interface.

    Config defaults to PolicyValueNet's six constructor attributes, or nested
    child configs for ModuleDict. Other modules use class/state shapes; a raw
    state mapping only permits shape inference. Supply explicit model_config
    (and expected_model_config on load) for richer/custom architectures.
    ``None`` training_state becomes {}; this is NOT a complete-resume claim.
    """
    if model_state is not None and model is not None:
        raise CheckpointError("pass either model or model_state, not both")
    if optimizer_state is not None and optimizer is not None:
        raise CheckpointError("pass either optimizer or optimizer_state, not both")
    weights = _state_dict(model if model_state is None else model_state, "model_state")
    _model_state(weights)
    opt_state = _state_dict(optimizer if optimizer_state is None else optimizer_state, "optimizer_state")
    names = _parameter_names(optimizer, model)
    if names is not None and opt_state is not None:
        opt_state = dict(opt_state)
        opt_state["param_groups"] = [dict(group, param_names=group_names)
                                     for group, group_names in zip(opt_state["param_groups"], names)]
    _optimizer_state(opt_state, optimizer if isinstance(optimizer, torch.optim.Optimizer) else None, model)
    config = _safe_dict(_inferred_config(model, weights) if model_config is None else model_config, "model_config")
    training = _safe_dict({} if training_state is None else training_state, "training_state")
    for value, alias, label in ((update, update_count, "update"), (seed, seed_count, "seed")):
        _strict_counter(value, label)
        if alias is not None:
            _strict_counter(alias, label + "_count")
            if value != 0 and value != alias:
                raise CheckpointError(f"{label} and {label}_count disagree")
    profile_name = _profile_name(profile)
    commit = _git_commit() if git_commit is _AUTO_GIT_COMMIT else git_commit
    if commit is not None:
        _string(commit, "git_commit")
    metadata = CheckpointMetadata(
        checkpoint_version=CHECKPOINT_VERSION, **_protocol_versions(protocol_versions),
        profile=profile_name, algorithm=_algorithm(algorithm, profile_name), git_commit=commit,
        update_count=update if update_count is None else update_count,
        seed_count=seed if seed_count is None else seed_count,
    )
    payload = {**metadata.as_dict(), "model_state": weights, "optimizer_state": opt_state,
               "model_config": config, "training_state": training}
    _safe_tree(payload, "checkpoint")
    _atomic_torch_save(payload, Path(path))
    return metadata


def _apply_atomically(model: Any, optimizer: Any, weights: Mapping, opt_state: Mapping | None) -> None:
    if model is None and optimizer is None:
        return
    if model is not None and not isinstance(model, torch.nn.Module):
        raise CheckpointError("model must be a torch.nn.Module")
    # Stage together so the cloned optimizer remains attached to cloned model
    # parameters. Neither validation nor staged load may touch the live objects.
    try:
        staged_model, staged_optimizer = deepcopy((model, optimizer))
        if staged_model is not None:
            staged_model.load_state_dict(weights, strict=True)
        if staged_optimizer is not None:
            staged_optimizer.load_state_dict(deepcopy(opt_state))
    except Exception as exc:
        raise CheckpointError(f"checkpoint state preflight failed: {exc}") from exc
    before_model = deepcopy(model.state_dict()) if model is not None else None
    before_optimizer = None
    if optimizer is not None:
        memo = {id(p): p for group in optimizer.param_groups for p in group["params"]}
        before_optimizer = deepcopy((optimizer.state, optimizer.param_groups), memo)
    try:
        if model is not None:
            model.load_state_dict(weights, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(deepcopy(opt_state))
    except Exception as exc:
        # Do not call possibly failing/user-hooked load_state_dict for rollback.
        if model is not None:
            with torch.no_grad():
                for key, tensor in model.state_dict().items():
                    tensor.copy_(before_model[key])
        if optimizer is not None:
            optimizer.state, optimizer.param_groups = before_optimizer
        raise CheckpointError(f"checkpoint state application failed (rolled back): {exc}") from exc


def load_checkpoint(
    path: str | Path,
    model: Any = None,
    optimizer: Any = None,
    *,
    expected_profile: str | AcceptanceProfile | None = None,
    expected_algorithm: str | None = None,
    expected_protocol_versions: Mapping[str, Any] | None = None,
    map_location: Any = "cpu",
    expected_model_config: dict | None = None,
    expected_model_version: str = MODEL_VERSION,
    expected_training_version: str = TRAINING_VERSION,
) -> TrainingCheckpoint:
    """Validate all metadata/state before transactional model+optimizer loading.

    Only weights_only=True is used; unsupported PyTorch releases fail closed.
    Training state is returned unchanged, not applied or interpreted. v1 files
    lack compatibility/resume guarantees and require an explicit offline migration.
    """
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"could not safely load checkpoint {path}: {exc}") from exc
    if type(payload) is not dict:
        raise CheckpointError("checkpoint payload must be a dictionary")
    _safe_tree(payload, "checkpoint")
    metadata = _metadata_from_payload(payload)
    for key, expected in (("model_version", expected_model_version), ("training_version", expected_training_version)):
        if getattr(metadata, key) != _string(expected, "expected_" + key):
            raise IncompatibleCheckpointError(f"checkpoint {key} does not match expected {expected!r}")
    if expected_profile is not None and metadata.profile != _profile_name(expected_profile):
        raise IncompatibleCheckpointError("checkpoint profile does not match expected profile")
    if expected_algorithm is not None and metadata.algorithm != _string(expected_algorithm, "expected_algorithm"):
        raise IncompatibleCheckpointError("checkpoint algorithm does not match expected algorithm")
    if expected_protocol_versions is not None and metadata.protocol_versions != _protocol_versions(expected_protocol_versions):
        raise IncompatibleCheckpointError("checkpoint protocol versions do not match expected versions")
    for key in ("model_state", "optimizer_state", "model_config", "training_state"):
        if key not in payload:
            raise CheckpointError(f"checkpoint is missing {key}")
    config = _safe_dict(payload["model_config"], "model_config")
    training = _safe_dict(payload["training_state"], "training_state")
    weights, opt_state = payload["model_state"], payload["optimizer_state"]
    _model_state(weights, model)
    _optimizer_state(opt_state, optimizer, model)
    if expected_model_config is None and model is not None:
        # Raw-state callers cannot infer non-tensor architectural attributes.
        expected_model_config = (_inferred_config(None, model.state_dict())
                                 if set(config) == {"state_shapes"}
                                 else _inferred_config(model, model.state_dict()))
    if expected_model_config is not None and not _same(config, _safe_dict(expected_model_config, "expected_model_config")):
        raise IncompatibleCheckpointError("checkpoint model_config does not match expected model_config")
    _apply_atomically(model, optimizer, weights, opt_state)
    return TrainingCheckpoint(metadata, weights, opt_state, config, training)


def resume_checkpoint(path: str | Path, model: Any, optimizer: Any | None = None, **kwargs: Any) -> TrainingCheckpoint:
    return load_checkpoint(path, model=model, optimizer=optimizer, **kwargs)


load_and_resume_checkpoint = resume_checkpoint

__all__ = [
    "CHECKPOINT_VERSION", "MODEL_VERSION", "TRAINING_VERSION", "PROTOCOL_VERSION_KEYS",
    "CURRENT_PROTOCOL_VERSIONS", "CheckpointError", "IncompatibleCheckpointError",
    "CheckpointMetadata", "TrainingCheckpoint", "LoadedCheckpoint", "save_checkpoint",
    "load_checkpoint", "resume_checkpoint", "load_and_resume_checkpoint",
]
