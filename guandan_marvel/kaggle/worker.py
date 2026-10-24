# SPDX-License-Identifier: MIT
"""Per-rank torchrun worker for one-model two-rank MARVEL DDP training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Mapping, Sequence

import torch

from guandan.cards import Rank
from guandan_marvel.kaggle.bootstrap import validate_recovery_directory
from guandan_marvel.kaggle.budget import (
    observe_completed_game_budget, validate_completed_game_budget_contract,
)
from guandan_marvel.kaggle.checkpoints import (
    build_recovery_bundle,
    freeze_distributed_policy,
    load_ddp_manifest,
    load_rank_checkpoint_for_resume,
    prune_run_artifacts,
    save_distributed_checkpoint,
    update_checkpoint_index,
)
from guandan_marvel.kaggle.common import append_jsonl, atomic_json, canonical_hash, sha256_file
from guandan_marvel.kaggle.evaluation import (
    run_smoke_evaluation_isolated, write_empty_champion, write_empty_evaluation,
    write_failed_evaluation,
)
from guandan_marvel.kaggle.profiles import (
    build_resume_train_config, build_train_config, require_strict_resume_decision,
)
from guandan_marvel.kaggle.progress import LiveProgressHeartbeat
from guandan_marvel.training.local import IPPOTrainer, VRPOTrainer
from guandan_marvel.utils import configure_torch
from guandan_marvel.kaggle.pilot_observability import PilotObserver


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


EVALUATION_RENDEZVOUS_SCHEMA = "guandan-marvel-ddp-evaluation-rendezvous-v1"
PLATEAU_GATE_SCHEMA = "guandan-marvel-plateau-gate-v1"
ANCHOR_HISTORY_SCHEMA = "guandan-marvel-anchor-history-v1"


def _plateau_gate_config(profile: Mapping[str, object]) -> Mapping[str, object] | None:
    value = profile.get("plateau_gate")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("plateau_gate must be an object")
    if value.get("schema") != PLATEAU_GATE_SCHEMA:
        raise ValueError("plateau_gate schema mismatch")
    if not bool(value.get("enabled", True)):
        return None
    if int(value.get("min_evaluations", 3)) < 2:
        raise ValueError("plateau_gate.min_evaluations must be >= 2")
    if int(value.get("patience", 2)) < 1:
        raise ValueError("plateau_gate.patience must be >= 1")
    if float(value.get("min_delta", 0.01)) < 0.0:
        raise ValueError("plateau_gate.min_delta must be non-negative")
    metric = str(value.get("metric", "rule_win_rate"))
    if not metric.endswith("_win_rate"):
        raise ValueError("plateau_gate.metric must end with '_win_rate'")
    if "stop_authority" in value:
        raise ValueError("plateau gate is record-only; stop_authority is not supported")
    return value


def _anchor_win_rates(run_root: Path) -> dict[str, dict[str, float]]:
    """Aggregate the latest bounded evaluation records per opponent."""
    path = run_root / "evaluation_games.jsonl"
    if not path.is_file():
        return {}
    totals: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, Mapping) or row.get("status") != "complete":
            continue
        opponent = str(row.get("opponent_id", row.get("opponent_kind", "")) or "")
        if not opponent:
            continue
        bucket = totals.setdefault(opponent, {"wins": 0.0, "games": 0.0})
        bucket["games"] += 1.0
        bucket["wins"] += float(bool(row.get("candidate_win")))
    result: dict[str, dict[str, float]] = {}
    for opponent, bucket in totals.items():
        games = int(bucket["games"])
        if games <= 0:
            continue
        result[opponent] = {
            "wins": int(bucket["wins"]),
            "games": games,
            "win_rate": bucket["wins"] / games,
        }
    return result


def _evaluate_plateau_gate(
    run_root: Path, gate: Mapping[str, object], *, iteration: int,
) -> tuple[bool, dict[str, object]]:
    """Record the latest anchor win rates and report a stall as ``plateau_alert``.

    The gate is permanently record-only: the returned boolean is the alert
    (the legacy stall condition), never a stop order, and the worker never
    terminates a session on it.  Sessions end only as ``complete`` or
    ``budget_stopped``.
    """
    metric = str(gate.get("metric", "rule_win_rate"))
    opponent = metric[: -len("_win_rate")]
    rates = _anchor_win_rates(run_root)
    history_path = run_root / "anchor_history.json"
    payload: dict[str, object] = {}
    if history_path.is_file():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping) and loaded.get("schema") == ANCHOR_HISTORY_SCHEMA:
                payload = dict(loaded)
        except (OSError, json.JSONDecodeError):
            payload = {}
    entries = [dict(item) for item in payload.get("entries", []) if isinstance(item, Mapping)]
    stale_count = int(payload.get("stale_count", 0))
    alert = False
    row = rates.get(opponent)
    if row is not None and int(row["games"]) > 0:
        value = float(row["win_rate"])
        previous = [float(item["metric_value"]) for item in entries if item.get("metric") == metric]
        best_before = max(previous) if previous else None
        improved = best_before is None or value > best_before + float(gate.get("min_delta", 0.01))
        stale_count = 0 if improved else stale_count + 1
        entries.append({
            "iteration": int(iteration),
            "metric": metric,
            "metric_value": value,
            "win_rates": {name: item["win_rate"] for name, item in sorted(rates.items())},
            "games": {name: int(item["games"]) for name, item in sorted(rates.items())},
        })
        alert = len(entries) >= int(gate.get("min_evaluations", 3)) and stale_count >= int(gate.get("patience", 2))
    payload = {
        "schema": ANCHOR_HISTORY_SCHEMA,
        "metric": metric,
        "min_delta": float(gate.get("min_delta", 0.01)),
        "min_evaluations": int(gate.get("min_evaluations", 3)),
        "patience": int(gate.get("patience", 2)),
        "stale_count": stale_count,
        "entries": entries,
        "plateau_alert": bool(alert),
        "updated_at": _timestamp(),
    }
    atomic_json(history_path, payload)
    return bool(alert), payload


def _publish_evaluation_rendezvous(
    path: Path, *, iteration: int, event: Mapping[str, object],
) -> Path:
    return atomic_json(path, {
        "schema": EVALUATION_RENDEZVOUS_SCHEMA,
        "iteration": int(iteration),
        "event": dict(event),
        "published_at": _timestamp(),
    })


def _wait_evaluation_rendezvous(
    path: Path, *, iteration: int, timeout_seconds: float, poll_seconds: float = 0.25,
) -> dict[str, object]:
    """Wait on the shared run filesystem, not a pending NCCL collective."""
    deadline = time.monotonic() + max(float(timeout_seconds), 1.0)
    last_error: str | None = None
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                payload = _load_json(path)
                if payload.get("schema") != EVALUATION_RENDEZVOUS_SCHEMA:
                    raise ValueError("evaluation rendezvous schema mismatch")
                if int(payload.get("iteration", -1)) != int(iteration):
                    raise ValueError("evaluation rendezvous iteration mismatch")
                event = payload.get("event")
                if not isinstance(event, Mapping):
                    raise ValueError("evaluation rendezvous event is malformed")
                return dict(event)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                last_error = f"{type(error).__name__}: {error}"
        time.sleep(max(float(poll_seconds), 0.01))
    detail = "" if last_error is None else f"; last_error={last_error}"
    raise TimeoutError(
        f"timed out waiting for rank-0 evaluation rendezvous at iteration {iteration}{detail}"
    )


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


_MACRO_COUNT_SUM_SUFFIXES = ("/macro_count_local", "/token_count_local")
_MACRO_TOKEN_WEIGHTED_SUFFIXES = ("/legacy_token_kl", "/token_value_loss", "/raw_entropy")


def _macro_metric_weight_kind(key: str, row: Mapping[str, object]) -> str:
    """Weighting unit of a macro metric key.

    ``sum``
        ``*/macro_count_local`` and ``*/token_count_local`` are counts: they add up
        across ranks (the summed macro count equals ``macro_count_global``).
    ``token``
        token-unit diagnostics (legacy token KL, Profile B token value loss, raw
        entropy) are weighted by ``token_count_local``; when that key is absent the
        macro count is used as the documented fallback.
    ``macro``
        everything else (per-macro statistics and per-sample macro means) is
        weighted by ``macro_count_local``.  ``value_loss`` is a token mean for the
        sequence ablation and a macro-start mean for ``macro_decision_v1``, so the
        row's ``macro/value_sample_unit`` discriminates those two cases.
    """
    if key.endswith(_MACRO_COUNT_SUM_SUFFIXES):
        return "sum"
    if key.endswith(_MACRO_TOKEN_WEIGHTED_SUFFIXES):
        return "token"
    if key.endswith("/value_loss") and str(row.get("macro/value_sample_unit", "")) == "token":
        return "token"
    return "macro"


def _macro_metric_count_key(key: str, kind: str, row: Mapping[str, object]) -> str:
    prefix = "macro_initial/" if key.startswith("macro_initial/") else "macro/"
    if kind == "token":
        token_key = f"{prefix}token_count_local"
        raw = row.get(token_key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return token_key
    return f"{prefix}macro_count_local"


def _is_rank_consistent_macro_metric(key: str) -> bool:
    """``*/adv_*_global`` metrics come from the all-reduce and are rank-identical."""
    return (
        key.startswith(("macro/", "macro_initial/"))
        and "/adv_" in key
        and key.endswith("_global")
    )


def _validate_macro_weight(key: str, count_key: str, raw: object) -> float:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(
            f"DDP macro metric {key!r} has no usable {count_key!r} weight on a rank; "
            "refusing to silently drop that rank"
        )
    weight = float(raw)
    if math.isnan(weight) or weight < 0.0:
        raise ValueError(f"DDP macro metric {key!r} has an invalid weight {weight!r}")
    return weight


def _aggregate_macro_metric(key: str, rows: Sequence[Mapping[str, object]], values: list[float]) -> float:
    """Sample-count-aware macro metric aggregate.

    - ``*/adv_*_global`` keys are already global for every rank; their cross-rank
      values are asserted identical (1e-9) and taken as-is.
    - count keys (``*/macro_count_local``, ``*/token_count_local``) are summed.
    - every other macro key is a per-sample statistic or per-sample mean; weighting
      by the corresponding local count makes the aggregate equal the single-process
      concatenated-batch reference.  Missing weights fail fast (a rank is never
      silently dropped); an all-zero weight set falls back to the equal mean.
    """
    if _is_rank_consistent_macro_metric(key):
        first = values[0]
        for value in values[1:]:
            if abs(value - first) > 1e-9:
                raise ValueError(
                    f"DDP macro global metric {key!r} differs across ranks: {values}"
                )
        return first
    present = [
        row for row in rows
        if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
    ]
    kind = _macro_metric_weight_kind(key, present[0])
    if kind == "sum":
        total = 0.0
        for row in present:
            total += _validate_macro_weight(key, key, row.get(key))
        return total
    total_weight = 0.0
    total_value = 0.0
    for row in present:
        count_key = _macro_metric_count_key(key, kind, row)
        weight = _validate_macro_weight(key, count_key, row.get(count_key))
        total_weight += weight
        total_value += weight * float(row[key])
    if total_weight <= 0.0:
        return sum(float(row[key]) for row in present) / len(present)
    return total_value / total_weight


def _aggregate_metrics(context, local: Mapping[str, object]) -> dict[str, object]:
    rows = context.all_gather_object(dict(local))
    numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    sums = {
        "statistics/env_steps_total", "statistics/game_steps_total", "statistics/completed_games_total",
        "health/illegal_actions", "health/truncated_games", "health/nonfinite_losses",
        "health/nonfinite_gradients", "health/action_capacity_overflows", "health/horizon_overflows",
        "perf/env_steps_per_second", "perf/game_steps_per_second", "perf/completed_games_per_second",
    }
    maxima = {
        "pilot/peak_allocated_bytes", "pilot/peak_reserved_bytes", "pilot/iteration_seconds",
        "statistics/iterations", "statistics/durable_updates", "statistics/learner_updates",
        "perf/rollout_seconds", "perf/learner_seconds", "perf/evaluation_seconds",
    }
    aggregated: dict[str, object] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
        if not values:
            continue
        if key in sums or key.endswith("_total"):
            value = sum(values)
        elif key in maxima:
            value = max(values)
        elif key.startswith("macro/") or key.startswith("macro_initial/"):
            value = _aggregate_macro_metric(key, rows, values)
        else:
            value = sum(values) / len(values)
        aggregated[key] = int(value) if all(float(item).is_integer() for item in values) and key.startswith("statistics/") else value
    primary = rows[0]
    for key, value in primary.items():
        if key not in aggregated and not isinstance(value, (int, float)):
            aggregated[key] = value
    if any("pilot/logit_weight_changed" in row for row in rows):
        aggregated["pilot/ranks_with_parameter_change"] = sum(bool(row.get("pilot/logit_weight_changed")) for row in rows)
    aggregated.update({
        "runtime/ddp_aggregated": True,
        "runtime/world_size": context.world_size,
        "runtime/rank_metric_records": len(rows),
        "timestamp": _timestamp(),
    })
    return aggregated


def _write_benchmark_summary(
    run_root: Path,
    *,
    profile: Mapping[str, object],
    world_size: int,
    session_start_iteration: int,
    target_iterations: int,
    session_elapsed_seconds: float,
) -> Path | None:
    benchmark = profile.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return None
    metrics_path = run_root / "metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    warmup = int(benchmark["warmup_iterations"])
    expected_measured = int(benchmark["measured_iterations"])
    warmup_boundary = int(session_start_iteration) + warmup
    measured = [row for row in rows if int(row.get("statistics/iterations", -1)) > warmup_boundary]
    if len(measured) != expected_measured:
        raise RuntimeError(
            "benchmark measured-iteration coverage mismatch: "
            f"expected={expected_measured}, actual={len(measured)}"
        )
    baseline_rows = [row for row in rows if int(row.get("statistics/iterations", -1)) <= warmup_boundary]
    baseline = baseline_rows[-1] if baseline_rows else {}
    final = measured[-1]

    def delta(key: str) -> int:
        return int(final.get(key, 0)) - int(baseline.get(key, 0))

    rollout_seconds = [float(row["perf/rollout_seconds"]) for row in measured]
    learner_seconds = [float(row["perf/learner_seconds"]) for row in measured]
    core_seconds = sum(r + l for r, l in zip(rollout_seconds, learner_seconds, strict=True))
    measured_env_steps = delta("statistics/env_steps_total")
    measured_game_steps = delta("statistics/game_steps_total")
    measured_games = delta("statistics/completed_games_total")
    health_keys = (
        "health/illegal_actions", "health/truncated_games", "health/nonfinite_losses",
        "health/nonfinite_gradients", "health/action_capacity_overflows", "health/horizon_overflows",
    )
    payload = {
        "schema": "guandan-marvel-m08-benchmark-summary-v1",
        "comparison_id": benchmark["comparison_id"],
        "control": benchmark["control"],
        "workload_hash": benchmark["workload_hash"],
        "world_size": int(world_size),
        "session_start_iteration": int(session_start_iteration),
        "target_iterations": int(target_iterations),
        "warmup_iterations": warmup,
        "measured_iterations": [int(row["statistics/iterations"]) for row in measured],
        "session_elapsed_seconds": float(session_elapsed_seconds),
        "training_core_seconds": core_seconds,
        "rollout_seconds": rollout_seconds,
        "learner_seconds": learner_seconds,
        "measured_completed_games": measured_games,
        "measured_env_steps": measured_env_steps,
        "measured_game_steps": measured_game_steps,
        "core_completed_games_per_second": measured_games / max(core_seconds, 1e-9),
        "core_env_steps_per_second": measured_env_steps / max(core_seconds, 1e-9),
        "core_game_steps_per_second": measured_game_steps / max(core_seconds, 1e-9),
        "health_totals": {
            key: sum(float(row.get(key, 0.0)) for row in measured)
            for key in health_keys
        },
        "benchmark_contract": dict(benchmark),
        "generated_at": _timestamp(),
        "remote_verified": False,
    }
    return atomic_json(run_root / "benchmark_summary.json", payload)


def _native_extension(game: object) -> Path | None:
    builder = getattr(game, "get_backend_build_identity", None)
    if not callable(builder):
        return None
    payload = builder()
    path = Path(str(payload.get("extension_path", "")))
    return path if path.is_file() else None


def _resolve_recovery_checkpoint(recovery_root: Path) -> tuple[Path, dict[str, object]]:
    payload = validate_recovery_directory(recovery_root)
    relative = Path(str(payload["checkpoint"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("recovery checkpoint path is unsafe")
    manifest = (recovery_root / relative).resolve()
    if not manifest.is_relative_to(recovery_root.resolve()) or not manifest.is_file():
        raise FileNotFoundError("recovery checkpoint manifest is missing")
    if payload.get("checkpoint_manifest_hash") != load_ddp_manifest(manifest, verify_shards=True).get("manifest_hash"):
        raise ValueError("recovery/DDP checkpoint manifest identity mismatch")
    return manifest.parent, payload


def _completed_game_budget(
    profile: Mapping[str, object],
    *,
    algorithm: str,
    global_collect_batch_size: int,
    target_iterations: int,
) -> dict[str, object] | None:
    value = profile.get("completed_game_budget")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("completed_game_budget must be an object")
    return validate_completed_game_budget_contract(
        value, expected_algorithm=algorithm,
        global_collect_batch_size=global_collect_batch_size,
        target_iterations=target_iterations,
    )


def _write_budget_observation(
    run_root: Path,
    observation: Mapping[str, object],
    *,
    status: str,
    wall_time_stopped: bool = False,
    error: str | None = None,
) -> Path:
    payload = {
        **dict(observation),
        "status": str(status),
        "wall_time_stopped": bool(wall_time_stopped),
        "error": error,
        "updated_at": _timestamp(),
        "remote_verified": False,
        "strength_claim": False,
    }
    return atomic_json(run_root / "completed_game_budget.json", payload)


def _restore_anchor_history(
    recovery_root: Path, run_root: Path, *, recovery_payload: Mapping[str, object],
) -> Path | None:
    """Carry the plateau-gate history from a recovery bundle into a new run root."""
    declared = {
        str(row.get("path"))
        for row in recovery_payload.get("files", [])
        if isinstance(row, Mapping)
    }
    source = recovery_root / "artifacts" / "anchor_history.json"
    target = run_root / "anchor_history.json"
    if "artifacts/anchor_history.json" in declared and source.is_file() and not target.is_file():
        shutil.copy2(source, target)
    return target if target.is_file() else None


def _primary_outcome(context, primary_event: Mapping[str, object] | None, *, stage: str) -> dict[str, object]:
    rows = context.all_gather_object(dict(primary_event) if context.is_primary and primary_event is not None else None)
    event = rows[0]
    if not isinstance(event, Mapping):
        raise RuntimeError(f"rank 0 did not publish a {stage} outcome")
    if event.get("status") != "complete":
        raise RuntimeError(f"{stage} failed on rank 0: {event.get('error')}")
    return dict(event)


def shared_budget_decision(
    context, *, rank: int, local_elapsed: float, max_wall_seconds: float,
    iteration: int, target_iterations: int,
) -> tuple[bool, list[dict[str, object]]]:
    local_vote = bool(
        max_wall_seconds > 0 and local_elapsed >= max_wall_seconds and iteration < target_iterations
    )
    votes = [dict(row) for row in context.all_gather_object({
        "rank": int(rank), "elapsed_seconds": float(local_elapsed), "budget_reached": local_vote,
    })]
    return any(bool(row["budget_reached"]) for row in votes), votes


def _controls(profile: Mapping[str, object], mode: str) -> Mapping[str, object]:
    value = profile.get("resume" if mode == "resume" else "training")
    if not isinstance(value, Mapping):
        raise ValueError(f"{mode} profile controls are missing")
    return value


def evaluation_parameters(
    evaluation_cfg: Mapping[str, object], *, role: str | None = None,
) -> dict[str, object]:
    """Resolve the evaluation tier actually passed to the isolated evaluator.

    Phase-4 profiles declare the tiers under ``selection``/``fast_anchor``; older
    profiles keep the flat keys.  A nested ``selection`` block always wins so a
    delivery can never silently fall back to the 8-deal anchor for its selection.
    """
    section = evaluation_cfg
    resolved_role = role
    if role is not None:
        candidate = evaluation_cfg.get(role)
        if isinstance(candidate, Mapping):
            section = candidate
        elif role == "selection":
            raise ValueError("evaluation.selection must be declared for selection evaluation")
    return {
        "base_deals": int(section.get("base_deals", 1)),
        "seed_base": int(section.get("seed_base", 870700)),
        "opponents": tuple(section.get("opponents", ("random", "rule"))),
        "seat_rotations": tuple(int(value) for value in section.get("seat_rotations", (0, 1, 2, 3))),
        "level_ranks": tuple(int(value) for value in section.get("level_ranks", (int(Rank.TWO),))),
        "leader_seats": tuple(int(value) for value in section.get("leader_seats", (0,))),
        "previous_results": tuple(section.get("previous_results", (None,))),
        "timeout_seconds": float(section.get("timeout_seconds", evaluation_cfg.get("timeout_seconds", 900))),
        "cluster_by": str(section.get("cluster_by", "deal_id")),
        "role": resolved_role or "flat",
    }


def run_due_evaluation(
    evaluation_checkpoint, evaluation_cfg: Mapping[str, object], *, output_root, device: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run the due evaluation tier and return ``(summary, parameters)``.

    A phase-4 profile with a nested ``selection`` block always runs the
    selection tier (52 deals, 13 levels, 4 leaders, 4 rotations, deal-clustered).
    Legacy flat profiles keep their existing behaviour.
    """
    evaluation_role = "selection" if isinstance(evaluation_cfg.get("selection"), Mapping) else None
    parameters = evaluation_parameters(evaluation_cfg, role=evaluation_role)
    summary = run_smoke_evaluation_isolated(
        evaluation_checkpoint,
        output_root=output_root,
        base_deals=int(parameters["base_deals"]),
        seed_base=int(parameters["seed_base"]),
        device=device,
        opponents=tuple(parameters["opponents"]),
        seat_rotations=tuple(parameters["seat_rotations"]),
        level_ranks=tuple(parameters["level_ranks"]),
        leader_seats=tuple(parameters["leader_seats"]),
        previous_results=tuple(parameters["previous_results"]),
        timeout_seconds=float(parameters["timeout_seconds"]),
    )
    return summary, parameters


def _fault_mapping(profile: Mapping[str, object]) -> Mapping[str, object]:
    value = profile.get("fault_injection", {})
    return value if isinstance(value, Mapping) else {}


def _fault_iterations(faults: Mapping[str, object], key: str) -> set[int]:
    values = faults.get(key, [])
    return {int(value) for value in values} if isinstance(values, (list, tuple)) else set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-content-manifest-hash", required=True)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--local-cpu-test", action="store_true")
    args = parser.parse_args(argv)

    profile = _load_json(args.profile.resolve())
    mode = str(profile.get("mode", ""))
    if mode not in {"pilot", "resume", "full", "single_gpu_control", "dual_gpu_control"}:
        raise ValueError(f"worker cannot run mode {mode!r}")
    if args.local_cpu_test:
        backend = str(profile.get("local_test_backend", "cpp"))
        horizon = int(profile.get("smoke_horizon", 512))
        if backend == "cpp":
            from guandan_marvel.training.smoke import CppSmokeGame
            game = CppSmokeGame(horizon)
        elif backend == "python_reference":
            from guandan_marvel.training.smoke import PythonReferenceSmokeGame
            game = PythonReferenceSmokeGame(horizon)
        else:
            raise ValueError(f"unsupported local test backend {backend!r}")
    else:
        from guandan_marvel import cpp_game as game

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    expected_world_size = int((profile.get("ddp") or {}).get("world_size", 0))
    if expected_world_size not in {1, 2}:
        raise RuntimeError(f"profile declares unsupported world_size={expected_world_size}")
    if world_size != expected_world_size:
        raise RuntimeError(
            f"torchrun/profile world size mismatch: runtime={world_size}, profile={expected_world_size}"
        )
    if rank not in set(range(world_size)) or local_rank not in set(range(world_size)):
        raise RuntimeError(
            f"rank/local_rank are outside world_size={world_size}: rank={rank}, local_rank={local_rank}"
        )

    run_root = args.run_root.resolve()
    rank_root = run_root / "ranks" / f"rank{rank:03d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    diagnostic_path = rank_root / "diagnostic.jsonl"
    append_jsonl(diagnostic_path, {
        "schema": "guandan-marvel-rank-diagnostic-v1",
        "event": "starting", "rank": rank, "local_rank": local_rank,
        "world_size": world_size, "pid": os.getpid(), "timestamp": _timestamp(),
    })

    recovery_payload: dict[str, object] | None = None
    restored_manifest: dict[str, object] | None = None
    checkpoint_payload: dict[str, object] | None = None
    checkpoint_root: Path | None = None
    if mode == "resume":
        if args.recovery_root is None:
            raise ValueError("resume mode requires --recovery-root")
        recovery_source = args.recovery_root.resolve()
        checkpoint_root, recovery_payload = _resolve_recovery_checkpoint(recovery_source)
        if rank == 0:
            restored_history = _restore_anchor_history(
                recovery_source, run_root, recovery_payload=recovery_payload,
            )
            if restored_history is not None:
                append_jsonl(diagnostic_path, {
                    "schema": "guandan-marvel-rank-diagnostic-v1", "event": "anchor_history_restored",
                    "rank": rank, "path": str(restored_history), "timestamp": _timestamp(),
                })
        if (
            recovery_payload.get("source_commit") != args.source_commit
            or recovery_payload.get("source_archive_sha256") != args.source_archive_sha256
            or recovery_payload.get("source_content_manifest_hash") != args.source_content_manifest_hash
        ):
            raise ValueError(
                "recovery belongs to a different source package; use the exact Dataset/source ZIP that created it"
            )
        checkpoint_payload, restored_manifest = load_rank_checkpoint_for_resume(
            checkpoint_root, rank=rank, expected_world_size=world_size,
            expected_source_commit=args.source_commit,
        )
        cfg = build_resume_train_config(
            profile, checkpoint_payload=checkpoint_payload, game=game, run_root=rank_root,
        )
    else:
        cfg = build_train_config(profile, game=game, run_root=rank_root, local_cpu_test=args.local_cpu_test)

    if (profile.get("pilot_observability") or {}).get("enabled", False):
        torch.set_num_threads(int((profile.get("pilot_observability") or {}).get("cpu_threads", 1)))
        torch.set_num_interop_threads(1)
    configure_torch(cfg)
    trainer_cls = IPPOTrainer if cfg.algo_name == "ippo" else VRPOTrainer
    trainer = trainer_cls(cfg, game=game, run_dir=rank_root / "trainer")
    context = trainer.distributed
    object_backend = profile.get("object_collectives_backend")
    if object_backend is not None:
        if object_backend != "gloo":
            raise ValueError("only an explicit Gloo metadata side group is supported")
        context = context.with_cpu_object_collectives()
        trainer.distributed = context
        trainer.distributed_runtime.context = context
    io_options = profile.get("checkpoint_io") or {}
    if set(io_options) - {"compression_level"}:
        raise ValueError("unknown checkpoint_io controls")
    compression_level = io_options.get("compression_level", 6)
    if isinstance(compression_level, bool) or not isinstance(compression_level, int) or not 1 <= compression_level <= 9:
        raise ValueError("invalid checkpoint compression level")
    bundle_options = {"compresslevel": compression_level} if io_options else {}
    warm_start_load = None
    if args.warm_start_checkpoint is not None:
        if mode == "resume":
            raise ValueError("resume mode cannot also use a warm-start checkpoint")
        warm_path = args.warm_start_checkpoint.resolve()
        if not warm_path.is_file():
            raise FileNotFoundError(warm_path)
        warm_payload = torch.load(warm_path, map_location=trainer.device, weights_only=False)
        if not isinstance(warm_payload, Mapping):
            raise TypeError("warm-start checkpoint payload must be a mapping")
        warm_start_load = trainer.load_warm_start_policy(warm_payload)
        append_jsonl(diagnostic_path, {
            "schema": "guandan-marvel-rank-diagnostic-v1", "event": "warm_start_loaded",
            "rank": rank, "checkpoint": str(warm_path), "load": warm_start_load,
            "timestamp": _timestamp(),
        })
    controls = _controls(profile, mode)
    max_wall_seconds = float(controls.get("max_wall_seconds", 0) or 0)
    checkpoint_interval = int(controls["checkpoint_interval"])
    evaluation_interval = int(controls["evaluation_interval"])
    checkpoint_keep_last = int(controls.get("checkpoint_keep_last", 2))
    evaluation_keep_last = int(controls.get("evaluation_keep_last", 2))
    latest_checkpoint: Path | None = checkpoint_root
    latest_evaluation: Path | None = None
    latest_metrics: dict[str, object] | None = None
    budget_stopped = False
    failure_text: str | None = None
    faults = _fault_mapping(profile)
    plateau_gate = _plateau_gate_config(profile)
    if faults and not args.local_cpu_test:
        raise ValueError("fault_injection is permitted only for explicit local CPU validation")

    if checkpoint_payload is not None:
        decision = require_strict_resume_decision(trainer.load_checkpoint_payload(checkpoint_payload))
        session_start_iteration = trainer.iteration
        target_iterations = cfg.total_iterations
        append_jsonl(diagnostic_path, {
            "schema": "guandan-marvel-rank-diagnostic-v1", "event": "resumed",
            "rank": rank, "iteration": trainer.iteration, "decision": decision.kind,
            "decision_reasons": list(decision.reasons),
            "checkpoint_manifest_hash": restored_manifest["manifest_hash"], "timestamp": _timestamp(),
        })
    else:
        session_start_iteration = trainer.iteration
        target_iterations = int(controls.get("target_iterations", cfg.total_iterations))
    if target_iterations <= trainer.iteration:
        raise ValueError("target iteration must be greater than restored/current iteration")
    completed_game_budget = _completed_game_budget(
        profile, algorithm=cfg.algo_name,
        global_collect_batch_size=cfg.collect_batch_size,
        target_iterations=target_iterations,
    )
    initial_global_completed = (
        0 if restored_manifest is None
        else int((restored_manifest.get("global_counters") or {}).get("completed_games_total", -1))
    )
    budget_observation = None if completed_game_budget is None else observe_completed_game_budget(
        completed_game_budget, iteration=trainer.iteration, completed_games=initial_global_completed,
    )

    heartbeat = LiveProgressHeartbeat(
        run_root=run_root, rank_root=rank_root, mode=mode, rank=rank,
        world_size=world_size, target_iterations=target_iterations,
        session_start_iteration=session_start_iteration,
        interval_seconds=float(profile.get("heartbeat_interval_seconds", profile.get("telemetry_interval_seconds", 5))),
        is_primary=context.is_primary,
    )
    trainer.set_phase_callback(heartbeat.set_phase)
    if context.is_primary:
        atomic_json(run_root / "resolved_config.json", {
            "schema": "guandan-marvel-kaggle-resolved-config-v2",
            "mode": mode,
            "profile": profile,
            "train_config": trainer.resolved_config,
            "train_config_hash": trainer.resolved_config_hash,
            "resume_inherited": mode == "resume",
            "recovery_manifest_hash": None if recovery_payload is None else recovery_payload.get("manifest_hash"),
            "source_commit": args.source_commit,
            "source_archive_sha256": args.source_archive_sha256,
            "source_content_manifest_hash": args.source_content_manifest_hash,
            "world_size": world_size,
            "warm_start": warm_start_load,
            "remote_verified": False,
        })
        write_empty_champion(run_root, reason="no evaluation has completed")
        write_empty_evaluation(run_root, reason="no evaluation has completed")
        if budget_observation is not None:
            _write_budget_observation(run_root, budget_observation, status="running")
    observer = PilotObserver(profile, trainer, run_root, rank_root)
    observer.export_initial(mode=mode)
    heartbeat.start()
    try:
        while trainer.iteration < target_iterations:
            heartbeat.set_phase("rollout")
            observer.begin()
            local_metrics = trainer.train_iteration()
            local_metrics.update(observer.finish(local_metrics))
            append_jsonl(diagnostic_path, {
                "schema": "guandan-marvel-rank-diagnostic-v1", "event": "iteration_completed",
                "rank": rank, "iteration": trainer.iteration, "metrics": local_metrics,
                "timestamp": _timestamp(),
            })
            aggregated = _aggregate_metrics(context, local_metrics)
            if completed_game_budget is not None:
                budget_observation = observe_completed_game_budget(
                    completed_game_budget,
                    iteration=trainer.iteration,
                    completed_games=int(aggregated["statistics/completed_games_total"]),
                )
            if context.is_primary:
                latest_metrics = dict(aggregated)
                append_jsonl(run_root / "metrics.jsonl", latest_metrics)
                if budget_observation is not None:
                    _write_budget_observation(run_root, budget_observation, status="running")
            heartbeat.update(iteration=trainer.iteration, metrics=aggregated)
            observer.console(aggregated, target=target_iterations)

            rank_delays = faults.get("rank_boundary_delay_seconds", {})
            if isinstance(rank_delays, Mapping):
                delay = float(rank_delays.get(str(rank), rank_delays.get(rank, 0.0)) or 0.0)
                if delay > 0:
                    time.sleep(delay)
            offsets = faults.get("budget_elapsed_offset_seconds", {})
            offset = float(offsets.get(str(rank), offsets.get(rank, 0.0)) or 0.0) if isinstance(offsets, Mapping) else 0.0
            local_elapsed = time.perf_counter() - heartbeat.started + offset
            budget_reached, budget_votes = shared_budget_decision(
                context, rank=rank, local_elapsed=local_elapsed,
                max_wall_seconds=max_wall_seconds, iteration=trainer.iteration,
                target_iterations=target_iterations,
            )
            append_jsonl(diagnostic_path, {
                "schema": "guandan-marvel-rank-diagnostic-v1", "event": "budget_decision",
                "rank": rank, "iteration": trainer.iteration, "votes": budget_votes,
                "decision": budget_reached, "timestamp": _timestamp(),
            })
            benchmark_contract = profile.get("benchmark")
            persist_checkpoint = not isinstance(benchmark_contract, Mapping) or bool(
                benchmark_contract.get("persist_checkpoint", True)
            )
            checkpoint_due = persist_checkpoint and (
                trainer.iteration % checkpoint_interval == 0
                or trainer.iteration == target_iterations
                or budget_reached
            )
            if checkpoint_due:
                heartbeat.set_phase("checkpoint")
                checkpoint_dir = run_root / "checkpoints" / f"iter{trainer.iteration:06d}"
                checkpoint_started = time.perf_counter()
                saved = save_distributed_checkpoint(trainer, checkpoint_dir, context=context)
                io_timing = {"save_shards_seconds": time.perf_counter() - checkpoint_started}
                primary_event = None
                if context.is_primary:
                    try:
                        latest_checkpoint = saved
                        verify_started = time.perf_counter()
                        ddp_manifest = load_ddp_manifest(checkpoint_dir, verify_shards=True)
                        if observer.enabled:
                            pilot_manifest = ddp_manifest
                            freeze_distributed_policy(
                                trainer, run_root / f"pilot_candidate_iter{trainer.iteration:06d}.pt",
                                ddp_manifest=pilot_manifest,
                            )
                        io_timing["verify_and_policy_export_seconds"] = time.perf_counter() - verify_started
                        index_started = time.perf_counter()
                        evaluation_due = evaluation_interval > 0 and (
                            trainer.iteration % evaluation_interval == 0 or trainer.iteration == target_iterations
                        )
                        update_checkpoint_index(
                            run_root, checkpoint_dir=checkpoint_dir, ddp_manifest=ddp_manifest,
                            evaluation_status="pending" if evaluation_due else "not_due",
                        )
                        retention = prune_run_artifacts(
                            run_root,
                            checkpoint_keep_last=checkpoint_keep_last,
                            evaluation_keep_last=evaluation_keep_last,
                        )
                        if trainer.iteration in _fault_iterations(faults, "bundle_fail_iterations"):
                            raise RuntimeError(f"injected recovery bundle failure at iteration {trainer.iteration}")
                        io_timing["index_and_retention_seconds"] = time.perf_counter() - index_started
                        bundle_started = time.perf_counter()
                        bundle = build_recovery_bundle(
                            run_root=run_root,
                            destination=run_root / "guandan-marvel-recovery.zip",
                            source_commit=args.source_commit,
                            source_archive_sha256=args.source_archive_sha256,
                            source_content_manifest_hash=args.source_content_manifest_hash,
                            package_id=args.package_id,
                            status="durable_pending_evaluation" if evaluation_due else (
                                "partial" if trainer.iteration < target_iterations else "complete"
                            ),
                            native_extension=_native_extension(game),
                            **bundle_options,
                        )
                        io_timing["recovery_bundle_seconds"] = time.perf_counter() - bundle_started
                        primary_event = {
                            "status": "complete", "evaluation_due": evaluation_due,
                            "checkpoint": str(checkpoint_dir), "bundle": str(bundle),
                            "ddp_manifest_hash": ddp_manifest["manifest_hash"],
                            "retention": retention,
                        }
                    except BaseException as error:
                        primary_event = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                wait_started = time.perf_counter()
                boundary = _primary_outcome(context, primary_event, stage="durable checkpoint publication")
                io_timing.update({"publication_wait_seconds": time.perf_counter() - wait_started,
                                  "checkpoint_total_seconds": time.perf_counter() - checkpoint_started})
                append_jsonl(rank_root / "checkpoint_io.jsonl", {
                    "schema": "guandan-marvel-checkpoint-io-v1", "rank": rank,
                    "iteration": trainer.iteration, "timestamp": _timestamp(),
                    "compression_level": compression_level,
                    "object_collective_backend": context.object_collective_backend,
                    **io_timing,
                })
                latest_checkpoint = checkpoint_dir
                heartbeat.update(latest_checkpoint=checkpoint_dir.relative_to(run_root).as_posix())

                if bool(boundary["evaluation_due"]):
                    heartbeat.set_phase("evaluation")
                    evaluation_checkpoint = run_root / "evaluation_checkpoints" / f"iter{trainer.iteration:06d}.pt"
                    eval_event = None
                    if context.is_primary:
                        try:
                            ddp_manifest = load_ddp_manifest(checkpoint_dir, verify_shards=True)
                            freeze_distributed_policy(trainer, evaluation_checkpoint, ddp_manifest=ddp_manifest)
                            if trainer.iteration in _fault_iterations(faults, "evaluation_fail_iterations"):
                                raise RuntimeError(f"injected evaluation failure at iteration {trainer.iteration}")
                            evaluation_cfg = profile.get("evaluation") or {}
                            summary, parameters = run_due_evaluation(
                                evaluation_checkpoint, evaluation_cfg,
                                output_root=run_root, device=str(trainer.device),
                            )
                            evaluation_timeout_seconds = float(parameters["timeout_seconds"])
                            expected_selection_hash = evaluation_cfg.get("expected_selection_dataset_hash")
                            if (
                                expected_selection_hash is not None
                                and summary.get("selection_dataset_hash") != expected_selection_hash
                            ):
                                raise ValueError("evaluation selection dataset identity mismatch")
                            latest_evaluation = evaluation_checkpoint
                            update_checkpoint_index(
                                run_root, checkpoint_dir=checkpoint_dir, ddp_manifest=ddp_manifest,
                                evaluation_checkpoint=evaluation_checkpoint,
                                evaluation_status=str(summary["status"]),
                            )
                            prune_run_artifacts(
                                run_root,
                                checkpoint_keep_last=checkpoint_keep_last,
                                evaluation_keep_last=evaluation_keep_last,
                            )
                            build_recovery_bundle(
                                run_root=run_root,
                                destination=run_root / "guandan-marvel-recovery.zip",
                                source_commit=args.source_commit,
                                source_archive_sha256=args.source_archive_sha256,
                                source_content_manifest_hash=args.source_content_manifest_hash,
                                package_id=args.package_id,
                                status="partial" if trainer.iteration < target_iterations else "complete",
                                native_extension=_native_extension(game),
                                **bundle_options,
                            )
                            eval_event = {
                                "status": "complete", "evaluation_status": summary["status"],
                                "evaluation_checkpoint": str(evaluation_checkpoint),
                                "evaluation_parameters": {
                                    key: (list(value) if isinstance(value, tuple) else value)
                                    for key, value in parameters.items()
                                },
                            }
                            if plateau_gate is not None:
                                gate_alert, gate_payload = _evaluate_plateau_gate(
                                    run_root, plateau_gate, iteration=trainer.iteration,
                                )
                                # Record-only: the worker never stops on the gate and the
                                # event carries the alert only (no stop semantics).
                                eval_event["plateau_alert"] = bool(gate_alert)
                                eval_event["plateau_gate"] = {
                                    "stale_count": int(gate_payload.get("stale_count", 0)),
                                    "evaluations": len(gate_payload.get("entries", [])),
                                    "metric": gate_payload.get("metric"),
                                }
                        except BaseException as error:
                            message = f"{type(error).__name__}: {error}"
                            write_failed_evaluation(
                                run_root, iteration=trainer.iteration, error=message,
                                checkpoint=evaluation_checkpoint if evaluation_checkpoint.is_file() else checkpoint_dir,
                            )
                            ddp_manifest = load_ddp_manifest(checkpoint_dir, verify_shards=True)
                            update_checkpoint_index(
                                run_root, checkpoint_dir=checkpoint_dir, ddp_manifest=ddp_manifest,
                                evaluation_checkpoint=evaluation_checkpoint if evaluation_checkpoint.is_file() else None,
                                evaluation_status="failed", evaluation_error=message,
                            )
                            prune_run_artifacts(
                                run_root,
                                checkpoint_keep_last=checkpoint_keep_last,
                                evaluation_keep_last=evaluation_keep_last,
                            )
                            try:
                                build_recovery_bundle(
                                    run_root=run_root,
                                    destination=run_root / "guandan-marvel-recovery.zip",
                                    source_commit=args.source_commit,
                                    source_archive_sha256=args.source_archive_sha256,
                                    source_content_manifest_hash=args.source_content_manifest_hash,
                                    package_id=args.package_id,
                                    status="evaluation_failed",
                                    native_extension=_native_extension(game),
                                    **bundle_options,
                                )
                            except Exception:
                                # The pre-evaluation durable bundle remains atomically intact.
                                pass
                            eval_event = {"status": "failed", "error": message, "evaluation_status": "failed"}
                    evaluation_cfg = profile.get("evaluation") or {}
                    evaluation_timeout_seconds = float(evaluation_cfg.get("timeout_seconds", 900))
                    rendezvous_path = run_root / f"evaluation_rendezvous_iter{trainer.iteration:06d}.json"
                    if context.is_primary:
                        if not isinstance(eval_event, Mapping):
                            eval_event = {"status": "failed", "error": "rank 0 produced no evaluation event"}
                        _publish_evaluation_rendezvous(
                            rendezvous_path, iteration=trainer.iteration, event=eval_event,
                        )
                        shared_eval = dict(eval_event)
                    else:
                        shared_eval = _wait_evaluation_rendezvous(
                            rendezvous_path, iteration=trainer.iteration,
                            timeout_seconds=evaluation_timeout_seconds + 120.0,
                        )
                    heartbeat.update(
                        latest_evaluation=None if latest_evaluation is None else latest_evaluation.relative_to(run_root).as_posix(),
                        latest_evaluation_status=shared_eval.get("evaluation_status", "unknown"),
                    )
                    if shared_eval.get("status") != "complete":
                        raise RuntimeError(f"evaluation failed on rank 0: {shared_eval.get('error')}")
                    # Plateau gate is record-only for this delivery: a stall alert is
                    # logged in anchor_history.json / evaluation events but never
                    # terminates the session.
            if budget_reached:
                budget_stopped = True
                break
        final_phase = "budget_stopped" if budget_stopped else "complete"
        if completed_game_budget is not None:
            if budget_observation is None:
                raise RuntimeError("completed-game budget observation is missing")
            if not budget_stopped and budget_observation.get("complete") is not True:
                raise RuntimeError("training ended without completing the exact completed-game budget")
            if context.is_primary:
                _write_budget_observation(
                    run_root, budget_observation,
                    status="incomplete" if budget_stopped else "complete",
                    wall_time_stopped=budget_stopped,
                )
        session_elapsed_seconds = time.perf_counter() - heartbeat.started
        if context.is_primary and final_phase == "complete":
            _write_benchmark_summary(
                run_root,
                profile=profile,
                world_size=world_size,
                session_start_iteration=session_start_iteration,
                target_iterations=target_iterations,
                session_elapsed_seconds=session_elapsed_seconds,
            )
        context.barrier()
        heartbeat.update(iteration=trainer.iteration, metrics=latest_metrics, phase=final_phase, complete=not budget_stopped)
        append_jsonl(diagnostic_path, {
            "schema": "guandan-marvel-rank-diagnostic-v1", "event": final_phase,
            "rank": rank, "iteration": trainer.iteration, "timestamp": _timestamp(),
        })
        heartbeat.close(phase=final_phase, complete=not budget_stopped)
        return 0
    except BaseException as error:
        failure_text = f"{type(error).__name__}: {error}"
        if context.is_primary and budget_observation is not None:
            _write_budget_observation(
                run_root, budget_observation, status="failed", error=failure_text,
            )
        append_jsonl(diagnostic_path, {
            "schema": "guandan-marvel-rank-diagnostic-v1", "event": "failed", "rank": rank,
            "error": failure_text, "timestamp": _timestamp(),
        })
        heartbeat.update(iteration=trainer.iteration, metrics=latest_metrics)
        heartbeat.close(phase="failed", complete=False, error=failure_text)
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
