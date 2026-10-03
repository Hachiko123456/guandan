"""Typed views over the canonical, strictly validated execution profiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from guandan import execution_profiles

PROFILE_CONFIG_VERSION = execution_profiles.PROFILE_VERSION
PROFILE_NAMES = execution_profiles.PROFILE_NAMES


class ProfileConfigError(ValueError):
    """The canonical profile catalog is missing or invalid."""


class UnknownProfileError(ProfileConfigError):
    """A profile outside the canonical catalog was requested."""


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
    def updates_per_algorithm(self) -> int:
        return self.training.updates_per_algorithm

    @property
    def rollout_envs(self) -> int:
        return self.training.rollout_envs

    @property
    def rollout_steps(self) -> int:
        return self.training.token_steps_per_update

    @property
    def token_steps_per_update(self) -> int:
        return self.training.token_steps_per_update

    @property
    def checkpoint_interval(self) -> int:
        return self.training.checkpoint_interval

    @property
    def resume_updates(self) -> int:
        return self.training.resume_updates

    @property
    def max_hours(self) -> float:
        return self.training.max_hours

    @property
    def games_per_pairing(self) -> int:
        return self.evaluation.deal_groups_per_pairing * len(self.evaluation.seat_rotations)


def _profile_root(path: str | Path | None) -> Path:
    if path is None:
        return Path(__file__).resolve().parents[2]
    source = Path(path)
    if source.is_dir():
        return source
    # Retain the file-path API for canonical catalogs without copying files or
    # introducing a second JSON parser/validator. Alternate filenames fail closed.
    if source.name != "acceptance_profiles.json" or source.parent.name != "configs":
        raise ProfileConfigError("path must be a project root or configs/acceptance_profiles.json")
    return source.parent.parent


def load_profile(name: str, path: str | Path | None = None) -> AcceptanceProfile:
    """Read a canonical catalog, without executing either local or remote work."""
    if name not in PROFILE_NAMES:
        raise UnknownProfileError(f"unknown execution profile: {name!r}")
    try:
        resolved = execution_profiles.load_execution_profile(_profile_root(path), name)
    except (OSError, ValueError) as exc:
        raise ProfileConfigError(str(exc)) from exc
    raw = resolved["config"]
    train, evaluation = raw["training"], raw["evaluation"]
    return AcceptanceProfile(
        name=resolved["name"],
        version=resolved["version"],
        mode=raw["mode"],
        device=raw["device"],
        algorithms=tuple(raw["algorithms"]),
        training=TrainingSettings(**{**train, "max_hours": float(train["max_hours"])}),
        evaluation=EvaluationSettings(
            deal_groups_per_pairing=evaluation["deal_groups_per_pairing"],
            seat_rotations=tuple(evaluation["seat_rotations"]),
            opponents=tuple(evaluation["opponents"]),
            max_hours=float(evaluation["max_hours"]),
        ),
    )


def load_profiles(path: str | Path | None = None) -> dict[str, AcceptanceProfile]:
    return {name: load_profile(name, path) for name in PROFILE_NAMES}


__all__ = [
    "PROFILE_CONFIG_VERSION", "PROFILE_NAMES", "AcceptanceProfile", "TrainingSettings",
    "EvaluationSettings", "ProfileConfigError", "UnknownProfileError", "load_profiles", "load_profile",
]
