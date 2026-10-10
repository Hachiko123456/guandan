# SPDX-License-Identifier: MIT
"""Bounded post-checkpoint evaluation used by M07 pilot/full delivery modes."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import time

from guandan_marvel.evaluation.checkpoints import load_v2_policy
from guandan_marvel.evaluation.host import play_game
from guandan_marvel.evaluation.manifests import build_deals_manifest, deals_for

from .common import append_jsonl, atomic_json

EVALUATION_SUMMARY_SCHEMA = "guandan-marvel-kaggle-evaluation-summary-v1"
CHAMPION_MANIFEST_SCHEMA = "guandan-marvel-kaggle-champion-manifest-v1"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def write_empty_evaluation(output_root: str | Path, *, reason: str) -> Path:
    return atomic_json(Path(output_root) / "evaluation_summary.json", {
        "schema": EVALUATION_SUMMARY_SCHEMA,
        "status": "not_run",
        "reason": reason,
        "formal_selection": False,
        "strength_claim": False,
        "remote_verified": False,
        "generated_at": _timestamp(),
    })


def write_empty_champion(output_root: str | Path, *, reason: str) -> Path:
    return atomic_json(Path(output_root) / "champion_manifest.json", {
        "schema": CHAMPION_MANIFEST_SCHEMA,
        "status": "not_established",
        "reason": reason,
        "promotion_performed": False,
        "strength_claim": False,
        "generated_at": _timestamp(),
    })


def run_smoke_evaluation(
    checkpoint: str | Path,
    *,
    output_root: str | Path,
    base_deals: int,
    seed_base: int,
    device: str,
) -> dict[str, object]:
    root = Path(output_root)
    artifact = load_v2_policy(checkpoint, device=device)
    manifest = build_deals_manifest(
        counts={"training": 0, "selection": int(base_deals), "final": 0},
        bases={"training": seed_base - 10_000, "selection": seed_base, "final": seed_base + 10_000},
        level_ranks=(2,), leader_seats=(0,), previous_results=(None,),
    )
    records_path = root / "evaluation_games.jsonl"
    records_path.unlink(missing_ok=True)
    records = []
    started = time.perf_counter()
    for opponent in ("random", "rule"):
        for deal in deals_for(manifest, "selection"):
            for rotation in range(4):
                row = play_game(
                    deal=deal,
                    rotation=rotation,
                    opponent_id=opponent,
                    opponent_kind=opponent,
                    candidate=artifact,
                    temperature=0.0,
                    device=device,
                )
                append_jsonl(records_path, row)
                records.append(row)
    complete = [row for row in records if row.get("status") == "complete"]
    successes = sum(bool(row.get("candidate_win")) for row in complete)
    low, high = _wilson(successes, len(complete))
    summary = {
        "schema": EVALUATION_SUMMARY_SCHEMA,
        "status": "complete" if len(complete) == len(records) else "incomplete",
        "scope": "bounded M07 pipeline evaluation; not formal selection or final holdout",
        "candidate": artifact.identity,
        "base_deals": int(base_deals),
        "seat_rotations": [0, 1, 2, 3],
        "opponents": ["random", "rule"],
        "games": len(records),
        "complete_games": len(complete),
        "candidate_wins": successes,
        "win_rate": successes / len(complete) if complete else None,
        "descriptive_wilson95_low": low,
        "descriptive_wilson95_high": high,
        "illegal_count": sum(int(row.get("illegal_count", 0)) for row in records),
        "truncated_count": sum(int(row.get("truncated_count", 0)) for row in records),
        "elapsed_seconds": time.perf_counter() - started,
        "formal_selection": False,
        "final_holdout": False,
        "promotion_eligible": False,
        "strength_claim": False,
        "remote_verified": False,
        "generated_at": _timestamp(),
    }
    atomic_json(root / "evaluation_summary.json", summary)
    write_empty_champion(root, reason="M07 bounded evaluation never promotes a champion")
    return summary


__all__ = [
    "CHAMPION_MANIFEST_SCHEMA", "EVALUATION_SUMMARY_SCHEMA", "run_smoke_evaluation",
    "write_empty_champion", "write_empty_evaluation",
]
