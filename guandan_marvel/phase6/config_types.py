# SPDX-License-Identifier: MIT
"""Lightweight phase-6 configuration types (no torch/model imports).

Keeping these dataclasses here lets ``guandan_marvel.config`` import them
without creating a cycle through the model or evaluation packages.  The
runtime pool module re-exports :class:`OpponentPoolConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .design import POOL_SAMPLING_SCHEMA


@dataclass
class OpponentPoolConfig:
    """Trainer-facing configuration.  ``enabled=False`` is bitwise inert."""

    enabled: bool = False
    manifest_path: str | None = None
    entries: tuple[Mapping[str, object], ...] = ()
    sampling_seed: int = 0
    sampling_scheme: str = POOL_SAMPLING_SCHEMA
    assignment_scheme: str = "team_rotation_v1"
    max_resident_policies: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("opponent_pool.enabled must be bool")
        if not self.enabled:
            # A disabled feature must be completely inert.  Silently ignoring a
            # configured entry list would hide a user error, so reject it.
            if self.manifest_path is not None or self.entries:
                raise ValueError(
                    "opponent_pool is disabled but a manifest/entries were provided; "
                    "set enabled=True or remove them"
                )
            return
        if self.manifest_path is not None and self.entries:
            raise ValueError(
                "opponent_pool accepts either manifest_path or inline entries, not both"
            )
        if self.manifest_path is None and not self.entries:
            raise ValueError("opponent_pool.enabled requires a manifest or inline entries")
        if self.manifest_path is not None and not str(self.manifest_path).strip():
            raise ValueError("opponent_pool.manifest_path must be non-empty")
        if not str(self.sampling_scheme).strip():
            raise ValueError("opponent_pool.sampling_scheme must be non-empty")
        if self.assignment_scheme != "team_rotation_v1":
            raise ValueError(
                f"unsupported assignment_scheme {self.assignment_scheme!r}; "
                "phase 6 pre-registers team_rotation_v1"
            )
        if isinstance(self.sampling_seed, bool) or int(self.sampling_seed) < 0:
            raise ValueError("opponent_pool.sampling_seed must be a non-negative integer")
        if isinstance(self.max_resident_policies, bool) or int(self.max_resident_policies) <= 0:
            raise ValueError("opponent_pool.max_resident_policies must be positive")
        object.__setattr__(self, "sampling_seed", int(self.sampling_seed))
        object.__setattr__(self, "max_resident_policies", int(self.max_resident_policies))
        object.__setattr__(self, "entries", tuple(dict(entry) for entry in self.entries))


__all__ = ["OpponentPoolConfig"]
