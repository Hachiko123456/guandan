# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/model/types.py.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .cache import KVCache


@dataclass
class AgentOutput:
    avail_probs: torch.Tensor
    state_values: torch.Tensor | None = None
    past_key_values: Optional[KVCache] = None


@dataclass
class ActiveAgentOutput:
    """Agent outputs gathered for the active player."""

    avail_probs: torch.Tensor  # [B, T, A]
    mask: torch.Tensor  # [B, T] boolean mask of active player slots
    past_key_values: Optional[KVCache] = None
    state_values: torch.Tensor | None = None  # [B, T, P] aligned to acting player
    player_state_value: torch.Tensor | None = None  # [B, T] gathered acting-player values


@dataclass
class CriticOutput:
    qvalues: torch.Tensor | None = None  # [B, T, A, P]
    state_values: torch.Tensor | None = None
    player_state_value: torch.Tensor | None = None  # [B, T] gathered acting-player values
    past_key_values: Optional[KVCache] = None
