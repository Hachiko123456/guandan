# SPDX-License-Identifier: MIT
"""Gate L notebook runner: DMC training with embedded evaluation, one session.

The training loop reuses :class:`guandan_marvel.dmc.train.DMCTrainer`, so the
locked stage-4 contract is untouched: ε-greedy collect -> replay -> MSE(Q, z)
gradient steps at the cycle boundary (AdamW 1e-4, wd=0, clip 1.0, CUDA fp16
autocast + GradScaler).  Around it this runner adds only the pilot-style
bookkeeping ``docs/dmc_prompts/05_gate_l_local.md`` pre-declares:

* budget clocks come from the **hashed plan**, never from the training config:
  training stops at a cycle boundary once ``budgets.train_seconds`` elapsed;
  the rest of the plan budget covers the pre-declared evaluations;
* interval checkpoints every ``checkpoint_interval_cycles`` cycles plus a
  final checkpoint (``dmc-v1`` full payloads, same as stage 4);
* one mini evaluation vs rule at ≥ ``schedule.mini_eval.fraction`` of the
  training budget, then the final evaluations (vs rule, vs random) — all ε=0
  greedy via :mod:`dmc_gate_l.evaluate`, no replay pushes, no ε-clock drift;
* ``artifacts/RESULT.json`` (the pre-declared gate table evaluated against
  measured values), a hashed ``RESULTS_MANIFEST.json``, a recovery index and
  the results zip next to the artifacts.

Boundaries (from the plan, enforced by never calling the relevant code):
no FableDan, no final holdout, no belief module, no opponent pool, no formal
production promotion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from guandan_marvel.dmc.train import (
    DMC_HEARTBEAT_SCHEMA,
    DMC_METRICS_SCHEMA,
    DMC_PROGRESS_SCHEMA,
    DMC_RUN_SUMMARY_SCHEMA,
    DMCTrainer,
    DMCConfigError,
    load_run_config,
)
from .evaluate import play_eval_games

PLAN_SCHEMA = "guandan-dmc-gate-l-plan-v1"
RESULT_SCHEMA = "guandan-dmc-gate-l-result-v1"
RESULTS_MANIFEST_SCHEMA = "guandan-dmc-gate-l-results-manifest-v1"
RECOVERY_SCHEMA = "guandan-dmc-gate-l-recovery-v1"
RESULTS_ZIP_NAME = "guandan_dmc_gate_l_results.zip"

_ZIP_DATE = (2026, 10, 25, 0, 0, 0)
#: Interval checkpoints retained inside the results zip (the newest ones; the
#: full set stays in the kernel output working directory).
_ZIP_CHECKPOINT_KEEP = 2


def canonical(value: Any) -> str:
    text = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def _json_safe(value: Any) -> Any:
    """Map non-finite floats to null so strict JSON rows stay writable."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluate_gates(
    gates: Mapping[str, Any],
    *,
    health: Mapping[str, Any],
    eval_health: Mapping[str, Any],
    metrics_rows: list[Mapping[str, Any]],
    mini: Mapping[str, Any] | None,
    final_rule: Mapping[str, Any],
    final_random: Mapping[str, Any],
    games_completed: int,
    wall_seconds: float,
) -> dict[str, Any]:
    """Score the pre-declared gate table (thresholds come from the plan)."""
    health_gate = gates["health"]
    health_ok = (
        int(health.get("illegal", -1)) == int(health_gate["illegal"])
        and int(health.get("nonfinite", -1)) == int(health_gate["nonfinite"])
        and int(health.get("truncated", -1)) == int(health_gate["truncated"])
        and int(eval_health.get("illegal", -1)) == 0
        and int(eval_health.get("nonfinite", -1)) == 0
        and int(eval_health.get("truncated", -1)) == 0
    )

    qz_values = [
        float(row["qz_r2_last"])
        for row in metrics_rows
        if row.get("qz_r2_last") is not None
    ]
    final_qz_r2 = qz_values[-1] if qz_values else None
    rose = None
    first_quarter = last_quarter = None
    if len(qz_values) >= 4:
        cut = max(1, len(qz_values) // 4)
        first_quarter = sum(qz_values[:cut]) / cut
        last_quarter = sum(qz_values[-cut:]) / cut
        rose = bool(last_quarter > first_quarter)
    qz_min = float(gates["q_calibration"]["final_qz_r2_min"])
    qz_pass = final_qz_r2 is not None and final_qz_r2 >= qz_min and rose is True

    rule_gate = gates["vs_rule"]
    rule_final = final_rule["summary"]["win_rate"]
    rule_mini = None if mini is None else mini["summary"]["win_rate"]
    rule_pass = (
        rule_final is not None
        and rule_final >= float(rule_gate["final_min"])
        and rule_mini is not None
        and rule_final > rule_mini
    )
    if rule_final is None:
        rule_disposition = "eval_incomplete"
    elif rule_final < float(rule_gate["mini_min_for_continue"]):
        rule_disposition = "ABORT_investigate_implementation"
    elif not rule_pass:
        rule_disposition = "one_resume_allowed_same_recovery_chain"
    else:
        rule_disposition = "none"

    random_pass = (
        final_random["summary"]["win_rate"] is not None
        and final_random["summary"]["win_rate"] >= float(gates["vs_random"]["final_min"])
    )

    throughput_gate = gates["throughput"]
    throughput_pass = (
        int(games_completed) >= int(throughput_gate["min_games_in_budget"])
        and float(wall_seconds) <= float(throughput_gate["budget_seconds"])
    )

    return {
        "health": {
            "thresholds": dict(health_gate),
            "measured_training": {key: health.get(key) for key in ("illegal", "nonfinite", "truncated")},
            "measured_evaluation": dict(eval_health),
            "pass": health_ok,
            "disposition": None if health_ok else "stop_return_to_stage_2_4",
        },
        "q_calibration": {
            "final_qz_r2_min": qz_min,
            "measured_final_qz_r2": final_qz_r2,
            "first_quarter_mean": first_quarter,
            "last_quarter_mean": last_quarter,
            "rose": rose,
            "step_cycles": len(qz_values),
            "pass": qz_pass,
            "disposition": None if qz_pass else "investigate_z_path_seating_features",
        },
        "vs_rule": {
            "final_min": float(rule_gate["final_min"]),
            "mini_min_for_continue": float(rule_gate["mini_min_for_continue"]),
            "measured_mini": rule_mini,
            "measured_final": rule_final,
            "rose_from_mini": (None if rule_mini is None or rule_final is None else bool(rule_final > rule_mini)),
            "pass": rule_pass,
            "disposition": rule_disposition,
        },
        "vs_random": {
            "final_min": float(gates["vs_random"]["final_min"]),
            "measured_final": final_random["summary"]["win_rate"],
            "pass": random_pass,
            "disposition": None if random_pass else "one_resume_allowed_same_recovery_chain",
        },
        "throughput": {
            "min_games_in_budget": int(throughput_gate["min_games_in_budget"]),
            "stretch_games": int(throughput_gate["stretch_games"]),
            "budget_seconds": float(throughput_gate["budget_seconds"]),
            "measured_games": int(games_completed),
            "measured_wall_seconds": float(wall_seconds),
            "measured_games_per_second": (
                float(games_completed) / float(wall_seconds) if wall_seconds > 0 else None
            ),
            "pass": throughput_pass,
            "disposition": (
                None if throughput_pass else "throughput_optimization_first_not_stage_6"
            ),
        },
        "overall_verdict": (
            "PASS" if health_ok and qz_pass and rule_pass and random_pass and throughput_pass
            else "FAIL"
        ),
    }


def _write_heartbeat(run_dir: Path, trainer: DMCTrainer, *, phase: str, elapsed: float) -> None:
    _atomic_write_json(run_dir / "heartbeat.json", {
        "schema": DMC_HEARTBEAT_SCHEMA,
        "phase": phase,
        "cycle": trainer.cycles,
        "games_completed": trainer.sampler.games_completed,
        "samples_pushed": trainer.sampler.samples_pushed,
        "replay_size": len(trainer.replay),
        "epsilon": trainer.sampler.epsilon(),
        "elapsed_seconds": elapsed,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    })


def _zip_files(rows: list[tuple[str, Path]], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, path in rows:
            info = zipfile.ZipInfo(name, _ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            with Path(path).open("rb") as source:
                bundle.writestr(info, source.read(), compress_type=zipfile.ZIP_DEFLATED,
                                compresslevel=1)


def run_notebook(
    *,
    dataset: Path,
    run_root: Path,
    clock: Callable[[], float] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run Gate L: budgeted DMC training + pre-declared evaluations + results zip.

    ``dataset`` is the unpacked payload directory (``plan.json`` and
    ``gate_l_config.json``); ``run_root`` is the Kaggle working directory for
    this run.  Returns the RESULT mapping (also written to
    ``<run_root>/artifacts/RESULT.json``).
    """
    dataset = Path(dataset)
    run_root = Path(run_root)
    started = time.perf_counter()
    elapsed = (lambda: time.perf_counter() - started) if clock is None else clock

    plan = _read_json(dataset / "plan.json")
    if plan.get("schema") != PLAN_SCHEMA:
        raise DMCConfigError(
            f"unsupported Gate L plan schema {plan.get('schema')!r}; expected {PLAN_SCHEMA!r}"
        )
    budgets = plan["budgets"]
    schedule = plan["schedule"]
    gates = plan["gates"]
    train_budget = float(budgets["train_seconds"])
    mini_schedule = schedule["mini_eval"]
    final_schedule = schedule["final_eval"]
    seed_base = schedule["eval_seed_base"]

    config = load_run_config(_read_json(dataset / "gate_l_config.json"))
    config = dataclasses.replace(config, output_dir=str(run_root / "run"))
    run_dir = run_root / "run"
    eval_dir = run_root / "eval"
    artifacts_dir = run_root / "artifacts"
    for directory in (run_dir, eval_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    trainer = DMCTrainer(config)
    device = trainer.device
    eval_device = str(device)
    log(f"[DMC-GATE-L] device={device} envs={config.num_envs} batch={config.batch_size} "
        f"games_target={config.total_games} train_budget={train_budget:.0f}s")

    _write_heartbeat(run_dir, trainer, phase="start", elapsed=elapsed())
    _atomic_write_json(run_dir / "progress.json", {
        "schema": DMC_PROGRESS_SCHEMA,
        "status": "running",
        "run_name": config.run_name,
        "cycle": trainer.cycles,
        "games_completed": trainer.sampler.games_completed,
        "total_games": config.total_games,
        "latest_checkpoint": None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    })

    metrics_path = run_dir / "metrics.jsonl"
    metrics_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    interval = int(config.checkpoint_interval_cycles)
    mini_threshold = train_budget * float(mini_schedule["fraction_of_train_budget"])
    mini_result: dict[str, Any] | None = None
    stop_reason = "complete"
    last_heartbeat = 0.0

    def progress_callback(progress: Mapping[str, Any]) -> None:
        nonlocal last_heartbeat
        now = elapsed()
        if now - last_heartbeat < float(config.heartbeat_seconds):
            return
        last_heartbeat = now
        _write_heartbeat(run_dir, trainer, phase="collect", elapsed=now)
        log(f"[DMC-GATE-L] collect: games={progress['games_completed']}/{config.total_games} "
            f"samples={progress['samples_pushed']} epsilon={trainer.sampler.epsilon():.4f}")

    while trainer.sampler.games_completed < config.total_games:
        if elapsed() >= train_budget:
            stop_reason = "budget_stopped"
            log("[DMC-GATE-L] training budget reached at a cycle boundary; stopping cleanly")
            break
        metrics = trainer.train_cycle(progress_callback=progress_callback)
        metrics_rows.append(metrics)
        _append_jsonl(metrics_path, {"schema": DMC_METRICS_SCHEMA, **metrics})
        _write_heartbeat(run_dir, trainer, phase="train", elapsed=elapsed())
        gps = metrics["games_per_second"]
        log(f"[DMC-GATE-L] cycle {metrics['cycle']}: games={metrics['games']} "
            f"samples={metrics['samples_pushed']} steps={metrics['train_steps']} "
            f"gps={'n/a' if gps is None else f'{gps:.3f}'}")
        if interval > 0 and trainer.cycles % interval == 0:
            path = run_dir / f"checkpoint_cycle_{trainer.cycles:06d}.pt"
            trainer.save_checkpoint(path)
            checkpoints.append({
                "path": str(path), "cycles": trainer.cycles,
                "games_completed": trainer.sampler.games_completed,
            })
            _atomic_write_json(run_dir / "progress.json", {
                "schema": DMC_PROGRESS_SCHEMA,
                "status": "running",
                "run_name": config.run_name,
                "cycle": trainer.cycles,
                "games_completed": trainer.sampler.games_completed,
                "total_games": config.total_games,
                "latest_checkpoint": str(path),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            })
        if mini_result is None and elapsed() >= mini_threshold:
            log(f"[DMC-GATE-L] mini evaluation vs rule ({mini_schedule['vs_rule_games']} games, ε=0)")
            mini_result = play_eval_games(
                trainer.q_agent,
                opponent_kind="rule",
                games=int(mini_schedule["vs_rule_games"]),
                base_seed=int(seed_base["mini_vs_rule"]),
                device=eval_device,
                max_obs_tokens=int(config.max_obs_tokens),
                obs_head_tokens=int(config.obs_head_tokens),
                log=log,
            )
            _append_jsonl(eval_dir / "mini_vs_rule.rows.jsonl", mini_result["summary"])
            for row in mini_result["rows"]:
                _append_jsonl(eval_dir / "mini_vs_rule.rows.jsonl", row)
            _atomic_write_json(eval_dir / "mini_vs_rule.summary.json", mini_result["summary"])
            _write_heartbeat(run_dir, trainer, phase="mini_eval", elapsed=elapsed())

    final_checkpoint = run_dir / "checkpoint_final.pt"
    trainer.save_checkpoint(final_checkpoint)
    checkpoints.append({
        "path": str(final_checkpoint), "cycles": trainer.cycles,
        "games_completed": trainer.sampler.games_completed,
    })

    log(f"[DMC-GATE-L] final evaluation vs rule ({final_schedule['vs_rule_games']} games, ε=0)")
    final_rule = play_eval_games(
        trainer.q_agent,
        opponent_kind="rule",
        games=int(final_schedule["vs_rule_games"]),
        base_seed=int(seed_base["final_vs_rule"]),
        device=eval_device,
        max_obs_tokens=int(config.max_obs_tokens),
        obs_head_tokens=int(config.obs_head_tokens),
        log=log,
    )
    log(f"[DMC-GATE-L] final evaluation vs random ({final_schedule['vs_random_games']} games, ε=0)")
    final_random = play_eval_games(
        trainer.q_agent,
        opponent_kind="random",
        games=int(final_schedule["vs_random_games"]),
        base_seed=int(seed_base["final_vs_random"]),
        device=eval_device,
        max_obs_tokens=int(config.max_obs_tokens),
        obs_head_tokens=int(config.obs_head_tokens),
        log=log,
    )
    for name, result in (("final_vs_rule", final_rule), ("final_vs_random", final_random)):
        _append_jsonl(eval_dir / f"{name}.rows.jsonl", result["summary"])
        for row in result["rows"]:
            _append_jsonl(eval_dir / f"{name}.rows.jsonl", row)
        _atomic_write_json(eval_dir / f"{name}.summary.json", result["summary"])

    wall_seconds = elapsed()
    health = trainer.sampler.health()
    eval_health = {
        "illegal": (
            mini_result["summary"]["illegal"] if mini_result else 0
        ) + final_rule["summary"]["illegal"] + final_random["summary"]["illegal"],
        "truncated": (
            mini_result["summary"]["truncated"] if mini_result else 0
        ) + final_rule["summary"]["truncated"] + final_random["summary"]["truncated"],
        "nonfinite": (
            mini_result["summary"]["nonfinite"] if mini_result else 0
        ) + final_rule["summary"]["nonfinite"] + final_random["summary"]["nonfinite"],
    }
    games_completed = int(trainer.sampler.games_completed)
    summary = {
        "schema": DMC_RUN_SUMMARY_SCHEMA,
        "run_name": config.run_name,
        "stop_reason": stop_reason,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp_fp16": device.type == "cuda",
        "scaler_enabled": bool(trainer.learner.scaler.is_enabled()),
        "torch_version": torch.__version__,
        "channel_dim": trainer.replay.channel_dim,
        "max_obs_tokens": trainer.replay.max_obs_tokens,
        "replay_reset": trainer.replay_reset,
        "totals": {
            "cycles": trainer.cycles,
            "games_completed": games_completed,
            "samples_pushed": trainer.sampler.samples_pushed,
            "train_steps": trainer.total_train_steps,
            "low_buffer_skips": trainer.low_buffer_skips,
            "step_skipped_total": trainer.step_skipped_total,
            "nonfinite_count": trainer.nonfinite_count,
            "wall_seconds": wall_seconds,
        },
        "throughput": {
            "games_per_second": (games_completed / wall_seconds) if wall_seconds > 0 else None,
            "samples_per_second": (
                trainer.sampler.samples_pushed / wall_seconds
            ) if wall_seconds > 0 else None,
        },
        "health": health,
        "resolved_train_config": trainer.train_config.to_dict(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - wall_seconds)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    }
    _atomic_write_json(run_dir / "summary.json", _json_safe(summary))
    _atomic_write_json(run_dir / "progress.json", {
        "schema": DMC_PROGRESS_SCHEMA,
        "status": stop_reason,
        "run_name": config.run_name,
        "cycle": trainer.cycles,
        "games_completed": games_completed,
        "total_games": config.total_games,
        "latest_checkpoint": str(final_checkpoint),
        "summary": "summary.json",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    })

    recovery = {
        "schema": RECOVERY_SCHEMA,
        "checkpoints": [
            {**row, "sha256": _sha(row["path"]), "bytes": Path(row["path"]).stat().st_size}
            for row in checkpoints
        ],
        "replay_reset_expected": True,
        "resume_hint": (
            "python -m guandan_marvel.dmc.train --config gate_l_config.json "
            "--resume <checkpoint_final.pt> (the replay restarts empty and the reset is recorded)"
        ),
        "note": "same-session evaluation only; a resume continues training from the checkpoint",
    }
    _atomic_write_json(artifacts_dir / "recovery_index.json", recovery)

    result = {
        "schema": RESULT_SCHEMA,
        "plan_hash": plan["plan_hash"],
        "status": stop_reason,
        "training": {
            "games_completed": games_completed,
            "samples_pushed": int(trainer.sampler.samples_pushed),
            "cycles": trainer.cycles,
            "train_steps": trainer.total_train_steps,
            "health": health,
            "throughput": summary["throughput"],
            "wall_seconds": wall_seconds,
        },
        "evaluations": {
            "mini_vs_rule": None if mini_result is None else mini_result["summary"],
            "final_vs_rule": final_rule["summary"],
            "final_vs_random": final_random["summary"],
        },
        "gates": _evaluate_gates(
            gates,
            health=health,
            eval_health=eval_health,
            metrics_rows=metrics_rows,
            mini=mini_result,
            final_rule=final_rule,
            final_random=final_random,
            games_completed=games_completed,
            wall_seconds=wall_seconds,
        ),
        "boundaries": plan["boundaries"],
        "deviations": plan["deviations"],
        "note": (
            "pipeline fetch marks DMC runs FETCHED_INCOMPLETE because its result verifier is "
            "serial-compare bound; this RESULT.json + RESULTS_MANIFEST.json are the "
            "authoritative verification for Gate L"
        ),
    }
    result["hash"] = canonical({key: value for key, value in result.items() if key != "hash"})
    _atomic_write_json(artifacts_dir / "RESULT.json", result)

    members: list[tuple[str, Path]] = [
        ("artifacts/RESULT.json", artifacts_dir / "RESULT.json"),
        ("artifacts/recovery_index.json", artifacts_dir / "recovery_index.json"),
        ("artifacts/training/metrics.jsonl", metrics_path),
        ("artifacts/training/summary.json", run_dir / "summary.json"),
        ("artifacts/training/progress.json", run_dir / "progress.json"),
    ]
    for name in ("mini_vs_rule", "final_vs_rule", "final_vs_random"):
        if (eval_dir / f"{name}.rows.jsonl").is_file():
            members.append((f"artifacts/eval/{name}.rows.jsonl", eval_dir / f"{name}.rows.jsonl"))
        if (eval_dir / f"{name}.summary.json").is_file():
            members.append((f"artifacts/eval/{name}.summary.json", eval_dir / f"{name}.summary.json"))
    for row in checkpoints[-_ZIP_CHECKPOINT_KEEP:]:
        members.append(("artifacts/checkpoints/" + Path(row["path"]).name, Path(row["path"])))
    file_rows = [
        {"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)}
        for name, path in members
    ]
    manifest = {
        "schema": RESULTS_MANIFEST_SCHEMA,
        "plan_hash": plan["plan_hash"],
        "mode": "full",
        "evidence_complete": bool(
            mini_result is not None
            and final_rule["summary"]["games"] == int(final_schedule["vs_rule_games"])
            and final_random["summary"]["games"] == int(final_schedule["vs_random_games"])
            and final_checkpoint.is_file()
        ),
        "omissions": [],
        "files": file_rows,
    }
    manifest["hash"] = canonical({key: value for key, value in manifest.items() if key != "hash"})
    manifest_path = run_root / "RESULTS_MANIFEST.json"
    _atomic_write_json(manifest_path, manifest)
    _zip_files([(manifest_path.name, manifest_path), *members], run_root / RESULTS_ZIP_NAME)
    log(f"[DMC-GATE-L] done: {stop_reason} games={games_completed} "
        f"verdict={result['gates']['overall_verdict']}")
    return result


__all__ = ["RESULT_SCHEMA", "RESULTS_ZIP_NAME", "canonical", "run_notebook"]
