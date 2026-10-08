# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/model/checkpointable.py.
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


def load_checkpoint_payload_file(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location = device)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint payload to be a mapping, got {type(checkpoint)}.")
    return checkpoint


def save_checkpoint_payload_file(
    checkpoint: Mapping[str, Any],
    checkpoint_path: str | Path,
) -> Path:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(dict(checkpoint), str(path))
    return path


class CheckpointableModule(nn.Module):
    """Checkpoint state interface layered on top of ``nn.Module``."""

    checkpoint_state_key: str | None = None

    def checkpoint_state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.state_dict()

    def load_checkpoint_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        # Unified checkpoint restore policy:
        # - Allow extra checkpoint keys (for cross-pipeline model variants).
        # - Require every model parameter/buffer key to be present.
        incompatible_keys = self.load_state_dict(state_dict, strict = False)
        missing_keys = list(incompatible_keys.missing_keys)
        if missing_keys:
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                f"Missing key(s) in state_dict: {missing_keys}."
            )

    def build_checkpoint_payload(
        self,
        *,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        checkpoint_state_key: str,
    ) -> dict[str, Mapping[str, torch.Tensor]]:
        key = str(checkpoint_state_key)
        if not key.strip():
            raise ValueError("checkpoint_state_key must be a non-empty string.")
        payload_state = self.checkpoint_state_dict() if state_dict is None else state_dict
        return {key: payload_state}

    def load_checkpoint_payload(
        self,
        checkpoint: Mapping[str, Any],
        *,
        checkpoint_state_key: str,
        **_kwargs: Any,
    ) -> None:
        key = str(checkpoint_state_key)
        if not key.strip():
            raise ValueError("checkpoint_state_key must be a non-empty string.")
        state_dict = checkpoint.get(key)
        if state_dict is None:
            raise KeyError(f"Checkpoint is missing required field '{key}'.")
        self.load_checkpoint_state_dict(state_dict)


__all__ = [
    "CheckpointableModule",
    "load_checkpoint_payload_file",
    "save_checkpoint_payload_file",
]
