# SPDX-License-Identifier: MIT
# Reworked from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/runtime/checkpoints.py.
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from guandan_marvel.model.checkpointable import load_checkpoint_payload_file, save_checkpoint_payload_file
from guandan_marvel.utils import unwrap_distributed_module

CHECKPOINT_SCHEMA = "guandan-marvel-checkpoint-v1"


@dataclass
class CheckpointState:
    """Flat bindings from versioned checkpoint keys to stateful objects or scalars."""

    entries: dict[str, Any] = field(default_factory=dict)
    iteration_key: str = "iteration"

    def __getitem__(self, key: str) -> Any: return self.entries[key]
    def __setitem__(self, key: str, value: Any) -> None: self.entries[key] = value
    def __contains__(self, key: str) -> bool: return key in self.entries
    def get(self, key: str, default: Any = None) -> Any: return self.entries.get(key, default)

    @property
    def iteration(self) -> int:
        return int(self.entries.get(self.iteration_key, 0) or 0)


def _is_stateful(obj: Any) -> bool:
    return callable(getattr(obj, "state_dict", None)) and callable(getattr(obj, "load_state_dict", None))


def build_checkpoint_payload_from_state(state: CheckpointState) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema": CHECKPOINT_SCHEMA}
    for key, obj in state.entries.items():
        if obj is None:
            continue
        base = unwrap_distributed_module(obj) if isinstance(obj, torch.nn.Module) else obj
        build = getattr(base, "build_checkpoint_payload", None)
        if isinstance(obj, torch.nn.Module) and callable(build):
            fragment = build(checkpoint_state_key=key)
            if key not in fragment:
                raise KeyError(f"Model checkpoint fragment must include '{key}'.")
            payload[key] = fragment[key]
        elif _is_stateful(base):
            payload[key] = base.state_dict()
        else:
            payload[key] = obj
    return payload


def load_checkpoint_into_state(checkpoint: Mapping[str, Any], state: CheckpointState) -> None:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported or missing checkpoint schema; expected {CHECKPOINT_SCHEMA}.")
    for key, obj in list(state.entries.items()):
        if obj is None or key not in checkpoint:
            continue
        base = unwrap_distributed_module(obj) if isinstance(obj, torch.nn.Module) else obj
        load_payload = getattr(base, "load_checkpoint_payload", None)
        if isinstance(obj, torch.nn.Module) and callable(load_payload):
            load_payload(checkpoint, checkpoint_state_key=key)
        elif _is_stateful(base):
            result = base.load_state_dict(checkpoint[key])
            missing = list(getattr(result, "missing_keys", ()))
            if missing:
                raise RuntimeError(f"Checkpoint field '{key}' is missing state keys: {missing}.")
        else:
            state.entries[key] = checkpoint[key]


class CheckpointStore:
    """Atomic, local checkpoint I/O. Remote artifact transport belongs to later stages."""

    @staticmethod
    def save_checkpoint_payload_file(payload: Mapping[str, Any], checkpoint_path: str | Path) -> Path:
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Refusing to save an unversioned checkpoint payload.")
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        try:
            save_checkpoint_payload_file(payload, temp)
            temp.replace(path)
        finally:
            if temp.exists():
                temp.unlink()
        return path

    @staticmethod
    def load_checkpoint_payload_file(checkpoint_path: str | Path, *, device: torch.device | str) -> Mapping[str, Any]:
        payload = load_checkpoint_payload_file(checkpoint_path, device=torch.device(device))
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"Unsupported or missing checkpoint schema; expected {CHECKPOINT_SCHEMA}.")
        return payload


class CheckpointManager:
    def __init__(self, *, checkpoint_dir: Path, config_payload: Any = None, config_payload_key: str = "train_config", store: CheckpointStore | None = None, **_ignored: Any) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_payload = config_payload
        self.config_payload_key = str(config_payload_key)
        self.store = store or CheckpointStore()

    def _serialized_config_payload(self) -> Any:
        if hasattr(self.config_payload, "to_dict"):
            return self.config_payload.to_dict()
        if is_dataclass(self.config_payload):
            return asdict(self.config_payload)
        return self.config_payload

    def save_from_state(self, state: CheckpointState, *, checkpoint_path: Path | None = None, **_ignored: Any) -> Path:
        payload = build_checkpoint_payload_from_state(state)
        config = self._serialized_config_payload()
        if config is not None:
            payload[self.config_payload_key] = config
        path = Path(checkpoint_path) if checkpoint_path is not None else self.checkpoint_dir / f"ckpt_{state.iteration:08d}.pt"
        if state.iteration <= 0 and checkpoint_path is None:
            raise ValueError("A positive iteration or explicit checkpoint_path is required.")
        return self.store.save_checkpoint_payload_file(payload, path)

    def load_into_state(self, checkpoint_path: Path, state: CheckpointState, *, device: torch.device | str, restore_optimizers: bool = True) -> None:
        payload = self.store.load_checkpoint_payload_file(checkpoint_path, device=device)
        if restore_optimizers:
            load_checkpoint_into_state(payload, state)
            return
        optimizer_entries: dict[str, Any] = {}
        for key, obj in list(state.entries.items()):
            if not isinstance(obj, torch.nn.Module) and _is_stateful(obj):
                optimizer_entries[key] = state.entries.pop(key)
        try:
            load_checkpoint_into_state(payload, state)
        finally:
            state.entries.update(optimizer_entries)

    @staticmethod
    def _default_checkpoint_state_key(model_cls: type[torch.nn.Module], explicit: str | None) -> str:
        key = explicit or getattr(model_cls, "checkpoint_state_key", None)
        if not key:
            raise ValueError("checkpoint_state_key must be supplied for this model class.")
        return str(key)

    @classmethod
    def load_model(cls, checkpoint_path: str | Path, *, model_cls: type[torch.nn.Module], device: torch.device | str, cfg: Any | None = None, checkpoint_state_key: str | None = None, model_kwargs: Mapping[str, Any] | None = None, eval_mode: bool = True, freeze_parameters: bool = True) -> torch.nn.Module:
        resolved_device = torch.device(device)
        payload = CheckpointStore.load_checkpoint_payload_file(checkpoint_path, device=resolved_device)
        resolve_cfg = getattr(model_cls, "resolve_checkpoint_config", None)
        resolved_cfg = resolve_cfg(payload, cfg=cfg) if callable(resolve_cfg) else cfg
        kwargs = dict(model_kwargs or {})
        model = model_cls(resolved_cfg, **kwargs) if resolved_cfg is not None else model_cls(**kwargs)
        model = model.to(resolved_device)
        key = cls._default_checkpoint_state_key(model_cls, checkpoint_state_key)
        loader = getattr(model, "load_checkpoint_payload", None)
        if not callable(loader):
            raise TypeError(f"{model_cls.__name__} must implement load_checkpoint_payload().")
        loader(payload, checkpoint_state_key=key)
        if eval_mode:
            model.eval()
        if freeze_parameters:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        setattr(model, "train_config", resolved_cfg)
        return model


__all__ = ["CHECKPOINT_SCHEMA", "CheckpointManager", "CheckpointState", "CheckpointStore", "build_checkpoint_payload_from_state", "load_checkpoint_into_state"]
