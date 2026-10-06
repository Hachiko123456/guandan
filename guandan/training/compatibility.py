"""Checkpoint compatibility contracts.

This module deliberately keeps resume policy separate from checkpoint storage and
from the source-control commit.  A semantic contract describes the training
meaning; fingerprints describe the implementation that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from guandan.environment_types import ACTION_VERSION, ENCODING_VERSION, ENV_VERSION, RULES_VERSION

COMPATIBILITY_SCHEMA_VERSION = 1


class ResumeMode(str, Enum):
    EXACT = "exact_resume"
    COMPATIBLE = "compatible_resume"
    MIGRATION = "migration"


class CompatibilityError(ValueError):
    """Raised when a compatibility contract is malformed."""


def _json_value(value: Any, path: str = "contract") -> Any:
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise CompatibilityError(f"{path} keys must be strings")
        return {key: _json_value(value[key], f"{path}.{key}") for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise CompatibilityError(f"{path} contains unsupported value type {type(value).__name__}")


def canonical_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CompatibilityError("semantic_contract must be a mapping")
    return _json_value(value)


def contract_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(canonical_contract(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticContract:
    """Stable, JSON-safe description of training semantics."""

    values: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", canonical_contract(self.values))
        if not self.values:
            raise CompatibilityError("semantic_contract must not be empty")

    @property
    def digest(self) -> str:
        return contract_digest(self.values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)

    @classmethod
    def from_value(cls, value: "SemanticContract | Mapping[str, Any]") -> "SemanticContract":
        return value if isinstance(value, cls) else cls(dict(value))


@dataclass(frozen=True, slots=True)
class ResumeDecision:
    mode: ResumeMode
    reasons: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.mode is not ResumeMode.MIGRATION




def build_training_semantic_contract(
    *,
    profile: str,
    algorithm: str,
    model_config: Mapping[str, Any],
    rollout_envs: int,
    token_steps_per_update: int,
    seed: int,
    epochs: int,
    gamma: float,
    gae_lambda: float,
    optimizer: Mapping[str, Any],
    level_rank: int,
    execution_mode: str = "reference",
) -> dict[str, Any]:
    """Build the canonical contract for a resumable training run.

    Execution-only settings such as device, checkpoint interval, session budget,
    logging cadence and output paths are intentionally excluded.
    """
    return SemanticContract({
        "algorithm": algorithm,
        "profile": profile,
        "model_config": dict(model_config),
        "protocol_versions": {
            "rules_version": RULES_VERSION,
            "action_version": ACTION_VERSION,
            "environment_version": ENV_VERSION,
            "encoding_version": ENCODING_VERSION,
        },
        "rollout": {
            "envs": int(rollout_envs),
            "token_steps_per_update": int(token_steps_per_update),
            "seed": int(seed),
            "level_rank": int(level_rank),
        },
        "objective": {
            "epochs": int(epochs),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
        },
        "optimizer": dict(optimizer),
        "execution_mode": execution_mode,
    }).as_dict()


def classify_resume(
    checkpoint_contract: SemanticContract | Mapping[str, Any] | None,
    expected_contract: SemanticContract | Mapping[str, Any] | None,
    *,
    checkpoint_source_fingerprint: str | None = None,
    expected_source_fingerprint: str | None = None,
    checkpoint_implementation_fingerprint: str | None = None,
    expected_implementation_fingerprint: str | None = None,
) -> ResumeDecision:
    """Classify a resume without changing or migrating checkpoint data.

    Missing contracts are intentionally classified as migration.  Callers may
    inspect that result, but the normal load path must reject it rather than
    silently converting an old checkpoint.
    """
    if checkpoint_contract is None or expected_contract is None:
        return ResumeDecision(ResumeMode.MIGRATION, ("semantic contract is missing",))
    checkpoint = SemanticContract.from_value(checkpoint_contract)
    expected = SemanticContract.from_value(expected_contract)
    if checkpoint.digest != expected.digest:
        return ResumeDecision(ResumeMode.MIGRATION, ("semantic contract differs",))
    source_same = checkpoint_source_fingerprint == expected_source_fingerprint
    implementation_same = checkpoint_implementation_fingerprint == expected_implementation_fingerprint
    if source_same and implementation_same:
        return ResumeDecision(ResumeMode.EXACT)
    reasons = []
    if not source_same:
        reasons.append("source fingerprint differs")
    if not implementation_same:
        reasons.append("implementation fingerprint differs")
    return ResumeDecision(ResumeMode.COMPATIBLE, tuple(reasons))


def require_resume_mode(decision: ResumeDecision, requested: ResumeMode | str | None = None) -> ResumeDecision:
    """Validate an optional requested mode and reject migration explicitly."""
    requested_mode = None if requested is None else ResumeMode(requested)
    if decision.mode is ResumeMode.MIGRATION:
        raise CompatibilityError("checkpoint requires explicit migration: " + "; ".join(decision.reasons))
    if requested_mode is not None and requested_mode is not decision.mode:
        raise CompatibilityError(f"requested {requested_mode.value}, actual {decision.mode.value}")
    return decision


__all__ = [
    "COMPATIBILITY_SCHEMA_VERSION", "CompatibilityError", "ResumeMode",
    "SemanticContract", "ResumeDecision", "canonical_contract", "contract_digest",
    "classify_resume", "require_resume_mode", "build_training_semantic_contract",
]
