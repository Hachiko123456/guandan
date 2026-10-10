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
import sys
import time
from typing import Mapping

import torch

from guandan_marvel.kaggle.checkpoints import (
    build_recovery_bundle,
    freeze_distributed_policy,
    load_ddp_manifest,
    load_distributed_checkpoint,
    save_distributed_checkpoint,
    update_checkpoint_index,
)
from guandan_marvel.kaggle.common import append_jsonl, atomic_json, canonical_hash, sha256_file
from guandan_marvel.kaggle.evaluation import run_smoke_evaluation, write_empty_champion, write_empty_evaluation
from guandan_marvel.kaggle.profiles import build_train_config
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


def _progress(
    *,
    phase: str,
    iteration: int,
    target_iterations: int,
    metrics: Mapping[str, object] | None,
    started: float,
    checkpoint: str | None,
    evaluation: str | None,
    complete: bool,
    error: str | None = None,
) -> dict[str, object]:
    elapsed = time.perf_counter() - started
    remaining = max(target_iterations - iteration, 0)
    eta = (elapsed / iteration * remaining) if iteration > 0 else None
    return {
        "schema": "guandan-marvel-progress-v1",
        "phase": phase,
        "iteration": iteration,
        "target_iterations": target_iterations,
        "completed_games": None if metrics is None else metrics.get("statistics/completed_games_total"),
        "env_steps": None if metrics is None else metrics.get("statistics/env_steps_total"),
        "game_steps": None if metrics is None else metrics.get("statistics/game_steps_total"),
        "env_steps_per_second": None if metrics is None else metrics.get("perf/env_steps_per_second"),
        "game_steps_per_second": None if metrics is None else metrics.get("perf/game_steps_per_second"),
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "latest_checkpoint": checkpoint,
        "latest_evaluation": evaluation,
        "heartbeat_at": _timestamp(),
        "complete": complete,
        "error": error,
        "remote_verified": False,
    }


def _native_extension(game: object) -> Path | None:
    builder = getattr(game, "get_backend_build_identity", None)
    if not callable(builder):
        return None
    payload = builder()
    path = Path(str(payload.get("extension_path", "")))
    return path if path.is_file() else None


def _resolve_recovery_checkpoint(recovery_root: Path) -> Path:
    manifest_path = recovery_root / "recovery_manifest.json"
    payload = _load_json(manifest_path)
    if payload.get("schema") != "guandan-marvel-recovery-manifest-v1":
        raise ValueError("unsupported recovery manifest schema")
    declared = payload.get("manifest_hash")
    actual = canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    if declared != actual:
        raise ValueError("recovery manifest hash mismatch")
    for row in payload.get("files", []):
        relative_file = Path(str(row.get("path", "")))
        if relative_file.is_absolute() or ".." in relative_file.parts:
            raise ValueError(f"unsafe recovery file path: {relative_file}")
        file_path = (recovery_root / relative_file).resolve()
        if not file_path.is_relative_to(recovery_root.resolve()) or not file_path.is_file():
            raise FileNotFoundError(f"recovery file is missing: {relative_file}")
        if file_path.stat().st_size != int(row.get("bytes", -1)) or sha256_file(file_path) != row.get("sha256"):
            raise ValueError(f"recovery file hash mismatch: {relative_file}")
    relative = Path(str(payload["checkpoint"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("recovery checkpoint path is unsafe")
    manifest = (recovery_root / relative).resolve()
    if not manifest.is_relative_to(recovery_root.resolve()) or not manifest.is_file():
        raise FileNotFoundError("recovery checkpoint manifest is missing")
    return manifest.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--local-cpu-test", action="store_true")
    args = parser.parse_args(argv)

    profile = _load_json(args.profile.resolve())
    mode = str(profile.get("mode", ""))
    if mode not in {"pilot", "resume", "full"}:
        raise ValueError(f"worker cannot run mode {mode!r}")
    if args.local_cpu_test:
        from guandan_marvel.training.smoke import PythonReferenceSmokeGame
        game = PythonReferenceSmokeGame(int(profile.get("smoke_horizon", 512)))
    else:
        from guandan_marvel import cpp_game as game

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size != 2 or rank not in {0, 1} or local_rank not in {0, 1}:
        raise RuntimeError("M07 worker requires torchrun with exactly two ranks")

    run_root = args.run_root.resolve()
    rank_root = run_root / "ranks" / f"rank{rank:03d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    diagnostic_path = rank_root / "diagnostic.jsonl"
    append_jsonl(diagnostic_path, {
        "schema": "guandan-marvel-rank-diagnostic-v1",
        "event": "starting", "rank": rank, "local_rank": local_rank,
        "world_size": world_size, "pid": os.getpid(), "timestamp": _timestamp(),
    })

    cfg = build_train_config(profile, game=game, run_root=rank_root, local_cpu_test=args.local_cpu_test)
    configure_torch(cfg)
    trainer_cls = IPPOTrainer if cfg.algo_name == "ippo" else VRPOTrainer
    trainer = trainer_cls(cfg, game=game, run_dir=rank_root / "trainer")
    context = trainer.distributed
    started = time.perf_counter()
    latest_checkpoint: Path | None = None
    latest_evaluation: Path | None = None
    latest_metrics: dict[str, object] | None = None
    target_iterations = int(profile["training"].get("target_iterations", 0))
    max_wall_seconds = float(profile["training"].get("max_wall_seconds", 0) or 0)
    budget_stopped = False
    if context.is_primary:
        atomic_json(run_root / "resolved_config.json", {
            "schema": "guandan-marvel-kaggle-resolved-config-v1",
            "mode": mode,
            "profile": profile,
            "train_config": trainer.resolved_config,
            "train_config_hash": trainer.resolved_config_hash,
            "source_commit": args.source_commit,
            "source_archive_sha256": args.source_archive_sha256,
            "world_size": world_size,
            "remote_verified": False,
        })
        write_empty_champion(run_root, reason="no evaluation has completed")
        write_empty_evaluation(run_root, reason="no evaluation has completed")
        atomic_json(run_root / "progress.json", _progress(
            phase="starting", iteration=0, target_iterations=target_iterations, metrics=None,
            started=started, checkpoint=None, evaluation=None, complete=False,
        ))
    try:
        if mode == "resume":
            if args.recovery_root is None:
                raise ValueError("resume mode requires --recovery-root")
            checkpoint_root = _resolve_recovery_checkpoint(args.recovery_root.resolve())
            decision, restored_manifest = load_distributed_checkpoint(trainer, checkpoint_root, context=context)
            if decision.kind not in {"exact_resume", "compatible_resume"}:
                raise RuntimeError(f"resume classification is not trainable: {decision.kind}")
            additional = int(profile["training"].get("additional_iterations", 0))
            if additional <= 0:
                raise ValueError("resume profile additional_iterations must be positive")
            target_iterations = trainer.iteration + additional
            latest_checkpoint = checkpoint_root
            append_jsonl(diagnostic_path, {
                "schema": "guandan-marvel-rank-diagnostic-v1", "event": "resumed",
                "rank": rank, "iteration": trainer.iteration, "decision": decision.kind,
                "checkpoint_manifest_hash": restored_manifest["manifest_hash"], "timestamp": _timestamp(),
            })
        checkpoint_interval = int(profile["training"]["checkpoint_interval"])
        evaluation_interval = int(profile["training"]["evaluation_interval"])
        if target_iterations <= trainer.iteration:
            raise ValueError("target iteration must be greater than restored/current iteration")
        while trainer.iteration < target_iterations:
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
                atomic_json(run_root / "progress.json", _progress(
                    phase="training", iteration=trainer.iteration, target_iterations=target_iterations,
                    metrics=latest_metrics, started=started,
                    checkpoint=None if latest_checkpoint is None else latest_checkpoint.relative_to(run_root).as_posix() if latest_checkpoint.is_relative_to(run_root) else str(latest_checkpoint),
                    evaluation=None if latest_evaluation is None else latest_evaluation.relative_to(run_root).as_posix(),
                    complete=False,
                ))
            budget_reached = max_wall_seconds > 0 and (time.perf_counter() - started) >= max_wall_seconds and trainer.iteration < target_iterations
            checkpoint_due = trainer.iteration % checkpoint_interval == 0 or trainer.iteration == target_iterations or budget_reached
            if checkpoint_due:
                checkpoint_dir = run_root / "checkpoints" / f"iter{trainer.iteration:06d}"
                saved = save_distributed_checkpoint(trainer, checkpoint_dir, context=context)
                context.barrier()
                if context.is_primary:
                    latest_checkpoint = saved
                    ddp_manifest = load_ddp_manifest(checkpoint_dir, verify_shards=True)
                    evaluation_checkpoint = None
                    evaluation_due = evaluation_interval > 0 and (
                        trainer.iteration % evaluation_interval == 0 or trainer.iteration == target_iterations
                    )
                    if evaluation_due and not args.local_cpu_test:
                        evaluation_checkpoint = run_root / "evaluation_checkpoints" / f"iter{trainer.iteration:06d}.pt"
                        freeze_distributed_policy(trainer, evaluation_checkpoint, ddp_manifest=ddp_manifest)
                        evaluation_cfg = profile.get("evaluation") or {}
                        run_smoke_evaluation(
                            evaluation_checkpoint,
                            output_root=run_root,
                            base_deals=int(evaluation_cfg.get("base_deals", 1)),
                            seed_base=int(evaluation_cfg.get("seed_base", 870700)),
                            device=str(trainer.device),
                        )
                        latest_evaluation = evaluation_checkpoint
                    update_checkpoint_index(
                        run_root, checkpoint_dir=checkpoint_dir, ddp_manifest=ddp_manifest,
                        evaluation_checkpoint=evaluation_checkpoint,
                    )
                    build_recovery_bundle(
                        run_root=run_root,
                        destination=run_root / "guandan-marvel-recovery.zip",
                        source_commit=args.source_commit,
                        source_archive_sha256=args.source_archive_sha256,
                        package_id=args.package_id,
                        status="partial" if trainer.iteration < target_iterations else "complete",
                        native_extension=_native_extension(game),
                    )
                context.barrier()
            if budget_reached:
                budget_stopped = True
                break
        if context.is_primary:
            atomic_json(run_root / "progress.json", _progress(
                phase="budget_stopped" if budget_stopped else "complete", iteration=trainer.iteration, target_iterations=target_iterations,
                metrics=latest_metrics, started=started,
                checkpoint=None if latest_checkpoint is None else latest_checkpoint.relative_to(run_root).as_posix(),
                evaluation=None if latest_evaluation is None else latest_evaluation.relative_to(run_root).as_posix(),
                complete=not budget_stopped,
            ))
        append_jsonl(diagnostic_path, {
            "schema": "guandan-marvel-rank-diagnostic-v1",
            "event": "budget_stopped" if budget_stopped else "complete",
            "rank": rank, "iteration": trainer.iteration, "timestamp": _timestamp(),
        })
        return 0
    except BaseException as error:
        append_jsonl(diagnostic_path, {
            "schema": "guandan-marvel-rank-diagnostic-v1", "event": "failed", "rank": rank,
            "error": f"{type(error).__name__}: {error}", "timestamp": _timestamp(),
        })
        if context.is_primary:
            atomic_json(run_root / "progress.json", _progress(
                phase="failed", iteration=trainer.iteration, target_iterations=target_iterations,
                metrics=latest_metrics, started=started,
                checkpoint=None if latest_checkpoint is None else str(latest_checkpoint),
                evaluation=None if latest_evaluation is None else str(latest_evaluation),
                complete=False, error=f"{type(error).__name__}: {error}",
            ))
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
