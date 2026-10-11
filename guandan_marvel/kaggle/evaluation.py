# SPDX-License-Identifier: MIT
"""Bounded post-checkpoint evaluation used by M07 pilot/full delivery modes."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import multiprocessing
from pathlib import Path
import time
import traceback
from typing import Mapping, Sequence

from guandan.cards import Rank
from guandan.state import PreviousHandResult
from guandan_marvel.evaluation.checkpoints import load_v2_policy
from guandan_marvel.evaluation.host import play_game
from guandan_marvel.evaluation.manifests import build_deals_manifest, dataset_identity, deals_for

from .common import append_jsonl, atomic_json

EVALUATION_SUMMARY_SCHEMA = "guandan-marvel-kaggle-evaluation-summary-v2"
CHAMPION_MANIFEST_SCHEMA = "guandan-marvel-kaggle-champion-manifest-v1"
EVALUATION_PROCESS_RESULT_SCHEMA = "guandan-marvel-kaggle-evaluation-process-result-v1"


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


def write_failed_evaluation(
    output_root: str | Path,
    *,
    iteration: int,
    error: BaseException | str,
    checkpoint: str | Path | None,
) -> Path:
    message = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return atomic_json(Path(output_root) / "evaluation_summary.json", {
        "schema": EVALUATION_SUMMARY_SCHEMA,
        "status": "failed",
        "iteration": int(iteration),
        "checkpoint": None if checkpoint is None else str(Path(checkpoint)),
        "error": message,
        "formal_selection": False,
        "final_holdout": False,
        "promotion_eligible": False,
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




def _previous_result_from_payload(value: Mapping[str, object] | None) -> PreviousHandResult | None:
    if value is None:
        return None
    return PreviousHandResult(
        tuple(int(item) for item in value["ranking"]),
        int(value["winner_team"]),
        str(value["outcome_class"]),
        bool(value.get("tribute_cancelled", False)),
    )


def build_bounded_selection_manifest(
    *,
    base_deals: int,
    seed_base: int,
    level_ranks: Sequence[int] | None,
    leader_seats: Sequence[int] | None,
    previous_results: Sequence[Mapping[str, object] | None] | None,
) -> dict[str, object]:
    resolved_levels = tuple(int(value) for value in (
        (int(Rank.TWO),) if level_ranks is None else level_ranks
    ))
    resolved_leaders = tuple(int(value) for value in ((0,) if leader_seats is None else leader_seats))
    resolved_previous = tuple(
        _previous_result_from_payload(value)
        for value in ((None,) if previous_results is None else previous_results)
    )
    return build_deals_manifest(
        counts={"training": 0, "selection": int(base_deals), "final": 0},
        bases={"training": max(0, int(seed_base) - 10_000), "selection": int(seed_base), "final": int(seed_base) + 500_000_000},
        level_ranks=resolved_levels, leader_seats=resolved_leaders, previous_results=resolved_previous,
    )


def run_smoke_evaluation(
    checkpoint: str | Path,
    *,
    output_root: str | Path,
    base_deals: int,
    seed_base: int,
    device: str,
    opponents: Sequence[str] = ("random", "rule"),
    seat_rotations: Sequence[int] = (0, 1, 2, 3),
    level_ranks: Sequence[int] | None = None,
    leader_seats: Sequence[int] | None = None,
    previous_results: Sequence[Mapping[str, object] | None] | None = None,
) -> dict[str, object]:
    """Run bounded C++ evaluation in the current process.

    Training callers must use :func:`run_smoke_evaluation_isolated`; this direct
    entrypoint is retained for eval_only and focused evaluation tests where no
    live trainer RNG/state exists in the calling process.
    """
    root = Path(output_root)
    artifact = load_v2_policy(checkpoint, device=device)
    manifest = build_bounded_selection_manifest(
        base_deals=base_deals, seed_base=seed_base, level_ranks=level_ranks,
        leader_seats=leader_seats, previous_results=previous_results,
    )
    resolved_opponents = tuple(str(value) for value in opponents)
    resolved_rotations = tuple(int(value) for value in seat_rotations)
    if not resolved_opponents or any(value not in {"random", "rule"} for value in resolved_opponents):
        raise ValueError("bounded evaluation opponents must be a non-empty subset of random/rule")
    if not resolved_rotations or any(value not in {0, 1, 2, 3} for value in resolved_rotations):
        raise ValueError("bounded evaluation seat rotations must be non-empty values in 0..3")
    records_path = root / "evaluation_games.jsonl"
    records_path.unlink(missing_ok=True)
    records = []
    started = time.perf_counter()
    for opponent in resolved_opponents:
        for deal in deals_for(manifest, "selection"):
            for rotation in resolved_rotations:
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
        "scope": "bounded Kaggle pipeline evaluation; not formal selection or final holdout",
        "candidate": artifact.identity,
        "base_deals": int(base_deals),
        "selection_dataset_hash": dataset_identity(manifest, "selection"),
        "selection_generator_version": manifest["generator_version"],
        "selection_coverage": manifest["coverage"],
        "level_ranks": list(manifest["coverage"]["level_ranks"]),
        "seat_rotations": list(resolved_rotations),
        "opponents": list(resolved_opponents),
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
    write_empty_champion(root, reason="bounded Kaggle evaluation never promotes a champion")
    return summary


def _evaluation_process_main(request_path: str, result_path: str) -> None:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    try:
        summary = run_smoke_evaluation(
            request["checkpoint"],
            output_root=request["output_root"],
            base_deals=int(request["base_deals"]),
            seed_base=int(request["seed_base"]),
            device=str(request["device"]),
            opponents=tuple(request["opponents"]),
            seat_rotations=tuple(int(value) for value in request["seat_rotations"]),
            level_ranks=tuple(int(value) for value in request["level_ranks"]),
            leader_seats=tuple(int(value) for value in request["leader_seats"]),
            previous_results=tuple(request["previous_results"]),
        )
        atomic_json(result_path, {
            "schema": EVALUATION_PROCESS_RESULT_SCHEMA,
            "status": "complete",
            "summary": summary,
            "finished_at": _timestamp(),
        })
    except BaseException as error:
        atomic_json(result_path, {
            "schema": EVALUATION_PROCESS_RESULT_SCHEMA,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "finished_at": _timestamp(),
        })
        raise


def run_smoke_evaluation_isolated(
    checkpoint: str | Path,
    *,
    output_root: str | Path,
    base_deals: int,
    seed_base: int,
    device: str,
    opponents: Sequence[str] = ("random", "rule"),
    seat_rotations: Sequence[int] = (0, 1, 2, 3),
    level_ranks: Sequence[int] | None = None,
    leader_seats: Sequence[int] | None = None,
    previous_results: Sequence[Mapping[str, object] | None] | None = None,
    timeout_seconds: float = 900.0,
) -> dict[str, object]:
    """Run real model loading/evaluation in a spawned process.

    The child receives only immutable paths and JSON-friendly configuration, so
    Python/NumPy/Torch CPU/CUDA RNG and live trainer state remain isolated on
    both success and failure paths.
    """
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "evaluation_process_request.json"
    result_path = root / "evaluation_process_result.json"
    result_path.unlink(missing_ok=True)
    atomic_json(request_path, {
        "schema": "guandan-marvel-kaggle-evaluation-process-request-v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "output_root": str(root),
        "base_deals": int(base_deals),
        "seed_base": int(seed_base),
        "device": str(device),
        "opponents": [str(value) for value in opponents],
        "seat_rotations": [int(value) for value in seat_rotations],
        "level_ranks": [int(value) for value in ((int(Rank.TWO),) if level_ranks is None else level_ranks)],
        "leader_seats": [int(value) for value in ((0,) if leader_seats is None else leader_seats)],
        "previous_results": [None if value is None else dict(value) for value in ((None,) if previous_results is None else previous_results)],
    })
    process = multiprocessing.get_context("spawn").Process(
        target=_evaluation_process_main,
        args=(str(request_path), str(result_path)),
        name="guandan-marvel-m07-evaluation",
    )
    process.start()
    process.join(timeout=max(float(timeout_seconds), 1.0))
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        raise TimeoutError(f"bounded evaluation exceeded {timeout_seconds} seconds")
    if not result_path.is_file():
        raise RuntimeError(f"evaluation process exited {process.exitcode} without a result")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or process.exitcode != 0:
        raise RuntimeError(
            "isolated evaluation failed: "
            f"exit={process.exitcode}; {result.get('error_type')}: {result.get('error')}"
        )
    summary = result.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("isolated evaluation returned a malformed summary")
    return summary


__all__ = [
    "CHAMPION_MANIFEST_SCHEMA", "EVALUATION_PROCESS_RESULT_SCHEMA", "EVALUATION_SUMMARY_SCHEMA",
    "build_bounded_selection_manifest", "run_smoke_evaluation", "run_smoke_evaluation_isolated", "write_empty_champion",
    "write_empty_evaluation", "write_failed_evaluation",
]
