# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/env_batch.py.
"""Conversion from the NumPy V2 game wire to MARVEL TensorDict records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from guandan_marvel.model.tensordict import TensorDict


@dataclass(frozen=True)
class EnvOutcome:
    data: TensorDict

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.data[key]


class EnvStepBatch:
    """Parsed environment response used by samplers and trajectory buffers."""

    def __init__(self, observation: TensorDict, outcome: EnvOutcome):
        self._observation = observation
        self._outcome = outcome

    @classmethod
    def from_env_step(cls, envs_step, cfg) -> "EnvStepBatch":
        batch = int(envs_step.rewards.shape[0])
        device = torch.device(cfg.device)
        rewards = torch.as_tensor(envs_step.rewards, dtype=torch.float32, device=device)
        terminal = torch.as_tensor(envs_step.is_terminal, dtype=torch.bool, device=device)
        outcome = EnvOutcome(TensorDict(
            {"rewards": rewards, "is_terminal": terminal},
            batch_size=[batch], device=device,
        ))
        observation = TensorDict(
            {
                "obs_tokens": torch.as_tensor(envs_step.observations.tokens, dtype=torch.long, device=device),
                "obs_channels": torch.as_tensor(envs_step.observations.channels, dtype=torch.uint8, device=device),
                "player_ids": torch.as_tensor(envs_step.player_ids, dtype=torch.long, device=device),
                "avail_tokens": torch.as_tensor(envs_step.available_actions.tokens, dtype=torch.long, device=device),
                "avail_weights": torch.as_tensor(envs_step.available_actions.weights, dtype=torch.float32, device=device),
                "avail_is_commit": torch.as_tensor(envs_step.available_actions.is_commit, dtype=torch.bool, device=device),
            },
            batch_size=[batch], device=device,
        )
        return cls(observation, outcome)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._observation[key]

    @property
    def device(self) -> torch.device:
        return self._observation.device

    @property
    def batch_size(self) -> torch.Size:
        return self._observation.batch_size

    @property
    def outcome(self) -> EnvOutcome:
        return self._outcome

    def record_agent_outputs(
        self, *, avail_probs: torch.Tensor | None = None,
        action_taken: torch.Tensor | None = None,
        step_fields: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        if avail_probs is not None:
            self._observation.set("avail_probs", avail_probs)
        if action_taken is not None:
            self._observation.set("action_taken", action_taken)
        if step_fields:
            for name, tensor in step_fields.items():
                self._observation.set(name, tensor)

    def build_transition(self, outcome: EnvOutcome | None = None) -> TensorDict:
        final_outcome = self._outcome.data if outcome is None else outcome.data
        merged = {key: value for key, value in self._observation.items()}
        merged.update({key: value for key, value in final_outcome.items()})
        return TensorDict(merged, batch_size=self._observation.batch_size, device=self._observation.device)


__all__ = ["EnvOutcome", "EnvStepBatch"]
