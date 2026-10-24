# SPDX-License-Identifier: MIT
"""Lossless valid-prefix layout for COMPLETE trajectories; never reduces game capacity."""
from __future__ import annotations
import torch
from guandan_marvel.model.tensordict import TensorDict

LAYOUT_SCHEMA = "guandan-marvel-completed-valid-prefix-v1"

def trim_completed_trajectory_padding(rollout: TensorDict) -> TensorDict:
    lengths = rollout["num_steps"]
    terminal = rollout["is_terminal"]
    if lengths.dim() != 1 or lengths.numel() == 0:
        raise ValueError("valid-prefix layout requires a nonempty trajectory batch")
    if lengths.dtype != torch.long or terminal.dtype != torch.bool or terminal.dim() != 2 or terminal.shape[0] != lengths.numel():
        raise ValueError("invalid complete-trajectory dtype/shape")
    horizon = int(terminal.shape[1])
    if bool(((lengths < 1) | (lengths > horizon)).any()):
        raise ValueError("invalid complete-trajectory lengths")
    if not bool(terminal.gather(1, (lengths.long() - 1).unsqueeze(1)).all()):
        raise ValueError("refusing to crop incomplete/truncated trajectories")
    end = int(lengths.max().item())
    fields = {name: (value[:, :end] if value.dim() >= 2 and value.shape[1] == horizon else value)
              for name, value in rollout.items()}
    return TensorDict(fields, batch_size=rollout.batch_size, device=rollout.device)
