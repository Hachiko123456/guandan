"""Validated local/remote execution profiles for A05-A08."""
from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any

PROFILE_CONFIG_VERSION = "profiles-0.2"
PROFILE_NAMES = ("local_fast", "remote_full")

class ProfileConfigError(ValueError): pass
class UnknownProfileError(ProfileConfigError): pass

@dataclass(frozen=True, slots=True)
class TrainingSettings:
    updates_per_algorithm: int
    rollout_envs: int
    token_steps_per_update: int
    checkpoint_interval: int
    resume_updates: int
    max_hours: float

@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    deal_groups_per_pairing: int
    seat_rotations: tuple[int, ...]
    opponents: tuple[str, ...]
    max_hours: float

@dataclass(frozen=True, slots=True)
class AcceptanceProfile:
    name: str
    mode: str
    device: str
    algorithms: tuple[str, ...]
    training: TrainingSettings
    evaluation: EvaluationSettings
    version: str = PROFILE_CONFIG_VERSION

    @property
    def updates_per_algorithm(self): return self.training.updates_per_algorithm
    @property
    def rollout_envs(self): return self.training.rollout_envs
    @property
    def rollout_steps(self): return self.training.token_steps_per_update
    @property
    def token_steps_per_update(self): return self.training.token_steps_per_update
    @property
    def checkpoint_interval(self): return self.training.checkpoint_interval
    @property
    def resume_updates(self): return self.training.resume_updates
    @property
    def max_hours(self): return self.training.max_hours
    @property
    def games_per_pairing(self): return self.evaluation.deal_groups_per_pairing * len(self.evaluation.seat_rotations)


def _positive(value: Any, name: str, *, integer: bool = True):
    if integer:
        if type(value) is not int or value <= 0: raise ProfileConfigError(f"{name} must be positive integer")
    elif type(value) not in (int, float) or not float(value) > 0: raise ProfileConfigError(f"{name} must be positive number")
    return value


def load_profiles(path: str | Path | None = None) -> dict[str, AcceptanceProfile]:
    source = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "configs" / "acceptance_profiles.json"
    try: payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc: raise ProfileConfigError(f"cannot read profiles: {exc}") from exc
    if payload.get("version") != PROFILE_CONFIG_VERSION or set(payload.get("profiles", {})) != set(PROFILE_NAMES):
        raise ProfileConfigError("profile catalog must be profiles-0.2 with local_fast and remote_full")
    result = {}
    for name, raw in payload["profiles"].items():
        if raw.get("mode") not in {"local", "remote"}: raise ProfileConfigError(f"{name}: invalid mode")
        if raw.get("device") not in {"auto", "cuda"}: raise ProfileConfigError(f"{name}: invalid device")
        train = raw.get("training", {})
        eval_cfg = raw.get("evaluation", {})
        envs = train.get("rollout_envs")
        if type(envs) is not int or envs <= 0: raise ProfileConfigError(f"{name}: rollout_envs must be positive integer")
        for key in ("updates_per_algorithm", "token_steps_per_update", "checkpoint_interval", "resume_updates"):
            _positive(train.get(key), f"{name}.training.{key}")
        _positive(train.get("max_hours"), f"{name}.training.max_hours", integer=False)
        rotations = tuple(eval_cfg.get("seat_rotations", ()))
        if rotations != (0, 1, 2, 3): raise ProfileConfigError(f"{name}: seat_rotations must be [0,1,2,3]")
        opponents = tuple(eval_cfg.get("opponents", ()))
        if opponents != ("random", "rule", "snapshot"): raise ProfileConfigError(f"{name}: invalid opponents")
        _positive(eval_cfg.get("deal_groups_per_pairing"), f"{name}.evaluation.deal_groups_per_pairing")
        _positive(eval_cfg.get("max_hours"), f"{name}.evaluation.max_hours", integer=False)
        result[name] = AcceptanceProfile(name, raw["mode"], raw["device"], tuple(raw["algorithms"]), TrainingSettings(train["updates_per_algorithm"], envs, train["token_steps_per_update"], train["checkpoint_interval"], train["resume_updates"], float(train["max_hours"])), EvaluationSettings(eval_cfg["deal_groups_per_pairing"], rotations, opponents, float(eval_cfg["max_hours"])))
    return result


def load_profile(name: str, path: str | Path | None = None) -> AcceptanceProfile:
    if name not in PROFILE_NAMES: raise UnknownProfileError(name)
    return load_profiles(path)[name]

__all__ = ["AcceptanceProfile", "TrainingSettings", "EvaluationSettings", "ProfileConfigError", "UnknownProfileError", "load_profiles", "load_profile"]
