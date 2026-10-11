# SPDX-License-Identifier: MIT
"""Per-rank torchrun worker for one-model two-rank MARVEL DDP training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Mapping

from guandan_marvel.kaggle.bootstrap import validate_recovery_directory
from guandan_marvel.kaggle.checkpoints import (
    build_recovery_bundle,
    freeze_distributed_policy,
    load_ddp_manifest,
    load_rank_checkpoint_for_resume,
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


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _aggregate_metrics(context, local: Mapping[str, object]) -> dict[str, object] | None:
    rows = context.all_gather_object(dict(local))
    if not context.is_primary:
        return None
    numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    sums = {
        "statistics/env_steps_total", "statistics/game_steps_total", "statistics/completed_games_total",
        "health/illegal_actions", "health/truncated_games", "health/nonfinite_losses",
        "health/nonfinite_gradients", "health/action_capacity_overflows", "health/horizon_overflows",
        "perf/env_steps_per_second", "perf/game_steps_per_second", "perf/completed_games_per_second",
    }
    maxima = {
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
        else:
            value = sum(values) / len(values)
        aggregated[key] = int(value) if all(float(item).is_integer() for item in values) and key.startswith("statistics/") else value
    primary = rows[0]
    for key, value in primary.items():
        if key not in aggregated and not isinstance(value, (int, float)):
            aggregated[key] = value
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
        checkpoint_root, recovery_payload = _resolve_recovery_checkpoint(args.recovery_root.resolve())
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

    configure_torch(cfg)
    trainer_cls = IPPOTrainer if cfg.algo_name == "ippo" else VRPOTrainer
    trainer = trainer_cls(cfg, game=game, run_dir=rank_root / "trainer")
    context = trainer.distributed
    controls = _controls(profile, mode)
    max_wall_seconds = float(controls.get("max_wall_seconds", 0) or 0)
    checkpoint_interval = int(controls["checkpoint_interval"])
    evaluation_interval = int(controls["evaluation_interval"])
    latest_checkpoint: Path | None = checkpoint_root
    latest_evaluation: Path | None = None
    latest_metrics: dict[str, object] | None = None
    budget_stopped = False
    failure_text: str | None = None
    faults = _fault_mapping(profile)
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
            "remote_verified": False,
        })
        write_empty_champion(run_root, reason="no evaluation has completed")
        write_empty_evaluation(run_root, reason="no evaluation has completed")
    heartbeat.start()
    try:
        while trainer.iteration < target_iterations:
            heartbeat.set_phase("rollout")
            local_metrics = trainer.train_iteration()
            append_jsonl(diagnostic_path, {
                "schema": "guandan-marvel-rank-diagnostic-v1", "event": "iteration_completed",
                "rank": rank, "iteration": trainer.iteration, "metrics": local_metrics,
                "timestamp": _timestamp(),
            })
            aggregated = _aggregate_metrics(context, local_metrics)
            if context.is_primary:
                latest_metrics = dict(aggregated or {})
                append_jsonl(run_root / "metrics.jsonl", latest_metrics)
            heartbeat.update(iteration=trainer.iteration, metrics=latest_metrics if context.is_primary else local_metrics)

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
                saved = save_distributed_checkpoint(trainer, checkpoint_dir, context=context)
                primary_event = None
                if context.is_primary:
                    try:
                        latest_checkpoint = saved
                        ddp_manifest = load_ddp_manifest(checkpoint_dir, verify_shards=True)
                        evaluation_due = evaluation_interval > 0 and (
                            trainer.iteration % evaluation_interval == 0 or trainer.iteration == target_iterations
                        )
                        update_checkpoint_index(
                            run_root, checkpoint_dir=checkpoint_dir, ddp_manifest=ddp_manifest,
                            evaluation_status="pending" if evaluation_due else "not_due",
                        )
                        if trainer.iteration in _fault_iterations(faults, "bundle_fail_iterations"):
                            raise RuntimeError(f"injected recovery bundle failure at iteration {trainer.iteration}")
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
                        )
                        primary_event = {
                            "status": "complete", "evaluation_due": evaluation_due,
                            "checkpoint": str(checkpoint_dir), "bundle": str(bundle),
                            "ddp_manifest_hash": ddp_manifest["manifest_hash"],
                        }
                    except BaseException as error:
                        primary_event = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                boundary = _primary_outcome(context, primary_event, stage="durable checkpoint publication")
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
                            summary = run_smoke_evaluation_isolated(
                                evaluation_checkpoint,
                                output_root=run_root,
                                base_deals=int(evaluation_cfg.get("base_deals", 1)),
                                seed_base=int(evaluation_cfg.get("seed_base", 870700)),
                                device=str(trainer.device),
                                opponents=tuple(evaluation_cfg.get("opponents", ("random", "rule"))),
                                seat_rotations=tuple(evaluation_cfg.get("seat_rotations", (0, 1, 2, 3))),
                                timeout_seconds=float(evaluation_cfg.get("timeout_seconds", 900)),
                            )
                            latest_evaluation = evaluation_checkpoint
                            update_checkpoint_index(
                                run_root, checkpoint_dir=checkpoint_dir, ddp_manifest=ddp_manifest,
                                evaluation_checkpoint=evaluation_checkpoint,
                                evaluation_status=str(summary["status"]),
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
                            )
                            eval_event = {
                                "status": "complete", "evaluation_status": summary["status"],
                                "evaluation_checkpoint": str(evaluation_checkpoint),
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
                                )
                            except Exception:
                                # The pre-evaluation durable bundle remains atomically intact.
                                pass
                            eval_event = {"status": "failed", "error": message, "evaluation_status": "failed"}
                    rows = context.all_gather_object(eval_event if context.is_primary else None)
                    shared_eval = rows[0]
                    heartbeat.update(
                        latest_evaluation=None if latest_evaluation is None else latest_evaluation.relative_to(run_root).as_posix(),
                        latest_evaluation_status="failed" if not isinstance(shared_eval, Mapping) else shared_eval.get("evaluation_status", "unknown"),
                    )
                    if not isinstance(shared_eval, Mapping) or shared_eval.get("status") != "complete":
                        raise RuntimeError(f"evaluation failed on rank 0: {None if shared_eval is None else shared_eval.get('error')}")
            if budget_reached:
                budget_stopped = True
                break
        final_phase = "budget_stopped" if budget_stopped else "complete"
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
