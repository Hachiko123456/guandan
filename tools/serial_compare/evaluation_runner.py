# SPDX-License-Identifier: MIT
"""Game execution shared by the evaluation worker and the preflight smoke."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

from .common import canonical, sha
from .evaluation import ARM_SPECS, validate_record


class ModelResolutionError(RuntimeError):
    pass


def model_paths_for_run(plan: Mapping[str, object], dataset: str | Path, finals: Mapping[str, object] | None = None) -> dict[str, Path]:
    dataset = Path(dataset)
    paths = {
        "p375": dataset / "policy_375.pt",
        "p250": dataset / "anchor_250.pt",
        "p500": dataset / "policy_500.pt",
    }
    if finals:
        for name in ("control_final", "candidate_final"):
            value = finals.get(name)
            if not value:
                raise ModelResolutionError(f"final checkpoint {name!r} is missing from the freeze record")
            paths[name] = Path(str(value))
    for name, path in paths.items():
        if not Path(path).is_file():
            raise ModelResolutionError(f"model file for {name!r} is missing: {path}")
    return paths


class ArtifactCache:
    def __init__(self, paths: Mapping[str, Path], *, device: str = "cpu") -> None:
        self.paths = {str(k): Path(v) for k, v in paths.items()}
        self.device = str(device)
        self._artifacts: dict[str, object] = {}

    def get(self, name: str):
        if name not in self._artifacts:
            from guandan_marvel.evaluation.checkpoints import load_v2_policy

            path = self.paths[name]
            artifact = load_v2_policy(str(path), device=self.device, expected_algorithm="ippo")
            self._artifacts[name] = artifact
        return self._artifacts[name]

    def hash_of(self, name: str) -> str:
        return sha(self.paths[name])


def run_tasks(
    tasks: Sequence[Mapping[str, object]],
    *,
    cache: ArtifactCache,
    models: Mapping[str, Mapping[str, object]],
    device: str = "cpu",
    max_steps: int = 1024,
    on_record: Callable[[Mapping[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    from guandan_marvel.evaluation.host import play_game

    records: list[dict[str, object]] = []
    for task in tasks:
        arm = str(task["arm"])
        candidate_key, opponent_key = ARM_SPECS[arm]
        candidate = cache.get(candidate_key)
        opponent = None if opponent_key is None else cache.get(opponent_key)
        deal = dict(task["deal"])
        assignment = int(task["assignment"])
        raw = play_game(
            deal=deal,
            rotation=assignment,
            opponent_id=arm if opponent_key is not None else "rule",
            opponent_kind="v2" if opponent is not None else "rule",
            candidate=candidate,
            opponent_v2=opponent,
            temperature=0.0,
            device=device,
            max_steps=int(max_steps),
        )
        record = {
            "task_id": str(task["task_id"]),
            "task_hash": canonical(dict(task)),
            "arm": arm,
            "deal_id": str(task["deal_id"]),
            "deal_seed": int(task["deal_seed"]),
            "deal_index": int(task.get("deal_index", -1)),
            "seat_rotation": assignment,
            "candidate_team": assignment % 2,
            "temperature": 0.0,
            "evaluation_horizon": int(max_steps),
            "evaluation_protocol": "two_unique_team_assignments_v1",
            "candidate_model": candidate_key,
            "opponent_model": opponent_key,
            "candidate_checkpoint_sha256": str(models[candidate_key]["sha256"]),
            "opponent_checkpoint_sha256": None if opponent_key is None else str(models[opponent_key]["sha256"]),
            **{key: raw[key] for key in (
                "status", "base_deal_sha256", "action_trace_sha256", "candidate_win",
                "candidate_team_reward", "token_steps", "committed_steps", "illegal_count",
                "truncated_count", "outcome_class", "ranking", "winner_team", "runtime_seconds",
            ) if key in raw},
        }
        record["record_hash"] = canonical({k: v for k, v in record.items() if k != "record_hash"})
        validate_record(record, task, models=models)
        if on_record is not None:
            # Persist each finished game *before* playing the next one, so a
            # timeout/kill keeps games 1..N-1 instead of losing the whole shard.
            on_record(record)
        records.append(record)
    return records


__all__ = ["ArtifactCache", "ModelResolutionError", "model_paths_for_run", "run_tasks"]
