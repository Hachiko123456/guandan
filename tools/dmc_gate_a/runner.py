# SPDX-License-Identifier: MIT
"""Gate A notebook runner: parallel DMC training + the four-lane evaluation.

``docs/dmc_prompts/06_gate_a_kaggle.md`` (2026-10-26 revision) fixes this shape:

* **stage-07A parallel topology** — collector ``cuda:1`` / learner ``cuda:0``
  through :func:`guandan_marvel.dmc.parallel.train_from_config_parallel`, so
  the locked learning semantics are exactly the ones stages 3/4/07A/07B
  already validated; this runner adds no arithmetic of its own;
* **one 9 h session** — ``budgets.train_seconds`` comes from the hashed plan
  and is enforced as the trainer's wall budget (checked at cycle boundaries);
  the rest of the plan budget covers the four evaluation lanes;
* **a resume chain** — a ``dmc-v1`` full checkpoint found at the pre-declared
  input filename resumes training exactly (the trainer re-validates the saved
  ``run_config``); every session emits ``dmc_gate_a_resume.pt`` for the next
  one, and 100,000 games are reached across 2–3 sessions;
* **the four-lane matrix** — rule / random / legacy / m09-iter2000 h2h, all
  ε=0 greedy, each on its declared driver and pairing (see
  :mod:`dmc_gate_a.lanes`);
* **pre-declared gates** — thresholds live in the hashed plan; this module
  only measures and compares, never re-tunes.

The m09 lane is proven runnable *before* training starts
(:func:`dmc_gate_a.m09.precheck`): an unreproducible native runtime must cost
seconds, not the whole session.

Boundaries (from the plan, enforced by never calling the relevant code): no
FableDan, no final holdout, no belief module, no opponent pool, no formal
production promotion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from guandan_marvel.dmc.parallel import (
    PARALLEL_SUMMARY_KEY,
    # The coordinator-side evaluation must rebuild the agent with exactly the
    # config the learner trained with.  The mapping lives in the parallel
    # trainer; importing it (rather than copying it) keeps one source of truth
    # and leaves ``parallel.py`` byte-identical to the stage-07A artifact that
    # Gate L was delivered and verified against.
    _build_train_config,
    train_from_config_parallel,
)
from guandan_marvel.dmc.q_agent import DMCAgent
from guandan_marvel.dmc.train import (
    DMC_CHECKPOINT_SCHEMA,
    DMC_TRAIN_PAYLOAD_KIND,
    DMCConfigError,
    DMCRunConfig,
    load_run_config,
)

from .lanes import LANE_DRIVERS, LANE_ROTATIONS, play_lane_games
from .m09 import (
    M09LaneUnavailable,
    load_legacy_artifact,
    load_m09_artifact,
    precheck as m09_precheck,
)

PLAN_SCHEMA = "guandan-dmc-gate-a-plan-v1"
RESULT_SCHEMA = "guandan-dmc-gate-a-result-v1"
RESULTS_MANIFEST_SCHEMA = "guandan-dmc-gate-a-results-manifest-v1"
RECOVERY_SCHEMA = "guandan-dmc-gate-a-recovery-v1"
RESULTS_ZIP_NAME = "guandan_dmc_gate_a_results.zip"
RESUME_FILENAME = "dmc_gate_a_resume.pt"

_ZIP_DATE = (2026, 10, 26, 0, 0, 0)
#: Interval checkpoints retained inside the results zip (the newest only; the
#: full set stays in the kernel output working directory).
_ZIP_CHECKPOINT_KEEP = 1


def canonical(value: Any) -> str:
    text = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(_json_safe(payload), sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8",
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


def _zip_files(rows: list[tuple[str, Path]], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, path in rows:
            info = zipfile.ZipInfo(name, _ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            with Path(path).open("rb") as source:
                bundle.writestr(info, source.read(), compress_type=zipfile.ZIP_DEFLATED,
                                compresslevel=1)


# ---------------------------------------------------------------------------
# Resume chain
# ---------------------------------------------------------------------------
def discover_resume(
    *,
    input_root: Path | None,
    filename: str,
    run_name: str,
) -> tuple[Path | None, dict[str, Any]]:
    """Find the previous session's checkpoint, or record a fresh start.

    The file is accepted only when it is a full ``dmc-v1`` training payload of
    this run; the trainer then re-validates the saved ``run_config`` field by
    field (a resume must replay the exact training configuration).
    """
    if input_root is None or not Path(input_root).is_dir():
        return None, {"status": "FRESH", "reason": "no input root", "filename": filename}
    matches = [
        path for path in sorted(Path(input_root).rglob(filename))
        if path.is_file() and not path.is_symlink()
    ]
    if not matches:
        return None, {"status": "FRESH", "reason": "no resume checkpoint in the input",
                      "filename": filename}
    if len(matches) > 1:
        raise DMCConfigError(
            "expected at most one resume checkpoint; found: "
            + ", ".join(str(path) for path in matches)
        )
    path = matches[0]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != DMC_CHECKPOINT_SCHEMA:
        raise DMCConfigError(f"resume checkpoint schema {payload.get('schema')!r} is not dmc-v1")
    if payload.get("payload_kind") != DMC_TRAIN_PAYLOAD_KIND:
        raise DMCConfigError("resume checkpoint is not a full DMC training checkpoint")
    saved = payload.get("run_config")
    if not isinstance(saved, Mapping) or saved.get("run_name") != run_name:
        raise DMCConfigError(
            f"resume checkpoint belongs to run {None if saved is None else saved.get('run_name')!r}, "
            f"not {run_name!r}"
        )
    counters = payload.get("counters") or {}
    record = {
        "status": "RESUMED",
        "filename": filename,
        "path": str(path),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        "cycles": int(counters.get("cycles", 0)),
        "games_completed": int(counters.get("games_completed", 0)),
        "samples_pushed": int(counters.get("samples_pushed", 0)),
        "replay_reset_expected": True,
    }
    return path, record


# ---------------------------------------------------------------------------
# Evaluation + gates
# ---------------------------------------------------------------------------
def _load_agent(run_config: DMCRunConfig, checkpoint: Path, *, device: str) -> DMCAgent:
    """Rebuild the trained Q agent from its final ``dmc-v1`` payload."""
    agent = DMCAgent(_build_train_config(run_config, device))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent.load_state_dict(payload)
    agent.eval()
    return agent


def _lane_disposition(gate: Mapping[str, Any], measured: float | None) -> tuple[bool, bool, str]:
    """Return ``(passed, aborted, disposition)`` for one rate gate."""
    if measured is None:
        return False, False, "lane_missing"
    if measured < float(gate["abort_below"]):
        return False, True, "ABORT"
    if measured >= float(gate["pass_min"]):
        return True, False, "PASS"
    return False, False, "BELOW_PASS"


def evaluate_gates(
    gates: Mapping[str, Any],
    *,
    lanes: Mapping[str, Mapping[str, Any]],
    train_health: Mapping[str, Any],
    eval_health: Mapping[str, Any],
    skipped: list[str],
) -> dict[str, Any]:
    """Score the pre-declared Gate A table (thresholds come from the plan)."""
    health_gate = gates["health"]
    health_ok = all(
        int(train_health.get(key, -1)) == int(health_gate[key])
        and int(eval_health.get(key, -1)) == int(health_gate[key])
        for key in ("illegal", "nonfinite", "truncated")
    )

    def rate(name: str) -> float | None:
        summary = lanes.get(name)
        return None if summary is None else summary.get("win_rate")

    rule_gate = gates["vs_rule"]
    rule_rate = rate("rule")
    rule_pass, rule_abort, rule_disp = _lane_disposition(rule_gate, rule_rate)
    reference = float(rule_gate["reference_gate_l_win_rate"])
    above_reference = None if rule_rate is None else bool(rule_rate > reference)
    rule_pass = bool(rule_pass and above_reference)

    m09_gate = gates["h2h_m09"]
    m09_rate = rate("m09")
    m09_pass, m09_abort, m09_disp = _lane_disposition(m09_gate, m09_rate)

    random_gate = gates["vs_random"]
    random_rate = rate("random")
    random_pass, random_abort, random_disp = _lane_disposition(random_gate, random_rate)

    aborted = rule_abort or m09_abort or random_abort
    if aborted:
        verdict = "ABORT"
    elif health_ok and rule_pass and m09_pass and random_pass and not skipped:
        verdict = "PASS"
    elif skipped:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"

    return {
        "health": {
            "thresholds": dict(health_gate),
            "measured_training": {
                key: train_health.get(key) for key in ("illegal", "nonfinite", "truncated")
            },
            "measured_evaluation": dict(eval_health),
            "pass": health_ok,
            "disposition": None if health_ok else "stop_return_to_stage_2_4",
        },
        "vs_rule": {
            "pass_min": float(rule_gate["pass_min"]),
            "abort_below": float(rule_gate["abort_below"]),
            "reference_gate_l_win_rate": reference,
            "measured": rule_rate,
            "above_gate_l_reference": above_reference,
            "pass": rule_pass,
            "abort": rule_abort,
            "disposition": rule_disp,
        },
        "h2h_m09": {
            "pass_min": float(m09_gate["pass_min"]),
            "abort_below": float(m09_gate["abort_below"]),
            "measured": m09_rate,
            "pass": m09_pass,
            "abort": m09_abort,
            "disposition": m09_disp,
        },
        "vs_random": {
            "pass_min": float(random_gate["pass_min"]),
            "abort_below": float(random_gate["abort_below"]),
            "measured": random_rate,
            "pass": random_pass,
            "abort": random_abort,
            "disposition": random_disp,
        },
        "lanes_skipped": list(skipped),
        "overall_verdict": verdict,
        "abort_disposition": (
            "do not change the algorithm or add compute; differential-test the four "
            "suspects (features / enumeration / z label / replay ratio) item by item"
            if aborted else None
        ),
    }


def stage7_projection(
    *,
    games_completed: int,
    samples_pushed: int,
    train_seconds: float,
    target_samples: int = 250_000_000,
) -> dict[str, Any]:
    """Record-only calendar projection for stage 7 (no gate reads it)."""
    if games_completed <= 0 or train_seconds <= 0:
        return {"status": "UNAVAILABLE"}
    gps = games_completed / train_seconds
    sps = samples_pushed / train_seconds
    samples_per_game = samples_pushed / games_completed
    return {
        "status": "RECORD_ONLY",
        "games_per_second": gps,
        "samples_per_second": sps,
        "samples_per_game": samples_per_game,
        "target_samples": target_samples,
        "games_to_target": target_samples / samples_per_game if samples_per_game else None,
        "gpu_hours_to_target": (target_samples / sps / 3600.0) if sps else None,
        "days_to_target_at_2_gpu": (target_samples / sps / 86400.0) if sps else None,
        "games_per_9h_session": gps * 32_400.0,
        "note": (
            "projection only: stage 7 is authorised by the gate table, never by this estimate"
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_notebook(
    *,
    dataset: Path,
    run_root: Path,
    clock: Callable[[], float] | None = None,
    log: Callable[[str], None] = print,
    m09_loader: Callable[..., Any] | None = None,
    legacy_loader: Callable[..., Any] | None = None,
    resume_input_root: Path | None = None,
) -> dict[str, Any]:
    """Run Gate A: parallel training + four-lane evaluation + results zip.

    ``dataset`` is the unpacked payload directory (``plan.json``,
    ``gate_a_config.json`` and the ``m09/`` bundle); ``run_root`` is the Kaggle
    working directory for this run.  ``m09_loader`` / ``legacy_loader`` /
    ``resume_input_root`` are explicit test seams — production passes none of
    them, and the RESULT records which loader actually ran.
    """
    dataset = Path(dataset)
    run_root = Path(run_root)
    started = time.perf_counter()
    elapsed = (lambda: time.perf_counter() - started) if clock is None else clock

    plan = _read_json(dataset / "plan.json")
    if plan.get("schema") != PLAN_SCHEMA:
        raise DMCConfigError(
            f"unsupported Gate A plan schema {plan.get('schema')!r}; expected {PLAN_SCHEMA!r}"
        )
    budgets = plan["budgets"]
    schedule = plan["schedule"]
    gates = plan["gates"]
    topology = plan["topology"]
    train_budget = float(budgets["train_seconds"])
    eval_reserve = float(budgets["eval_reserve_seconds"])
    total_budget = float(budgets["total_seconds"])

    config = load_run_config(_read_json(dataset / "gate_a_config.json"))
    if config.max_wall_seconds is None or float(config.max_wall_seconds) != train_budget:
        raise DMCConfigError(
            "the training config's max_wall_seconds must equal the plan's train budget "
            f"({config.max_wall_seconds!r} != {train_budget!r})"
        )
    cycle_games = int(config.games_per_cycle) * int(config.checkpoint_interval_cycles)
    if cycle_games != int(schedule["checkpoint_interval_games"]):
        raise DMCConfigError(
            f"checkpoint cadence {cycle_games} games != pre-declared "
            f"{schedule['checkpoint_interval_games']}"
        )

    run_dir = run_root / "run"
    eval_dir = run_root / "eval"
    artifacts_dir = run_root / "artifacts"
    for directory in (run_dir, eval_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # The parallel topology is a CUDA contract.  A CPU config is only reachable
    # in tests (the verifier pins ``device: "cuda"`` in every delivered plan),
    # so the degraded branch is recorded rather than silently taken.
    cuda_run = str(config.device).startswith("cuda")
    names: list[str] = []
    if cuda_run:
        if int(torch.cuda.device_count()) != int(topology["required_gpu_count"]):
            raise DMCConfigError(
                f"Gate A requires exactly {topology['required_gpu_count']} GPUs, "
                f"found {torch.cuda.device_count()}"
            )
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if any(str(topology["gpu_name_contains"]) not in name for name in names):
            raise DMCConfigError(
                f"Gate A requires {topology['gpu_name_contains']} GPUs, found {names}"
            )
        learner_device = str(topology["learner_device"])
        collector_device = str(topology["collector_device"])
    else:
        learner_device = collector_device = "cpu"
    log(f"[DMC-GATE-A] topology learner={learner_device} collector={collector_device} "
        f"gpus={names} envs={config.num_envs} cycle={config.games_per_cycle} "
        f"batch={config.batch_size}x{config.accumulation_steps} "
        f"train_budget={train_budget:.0f}s target={config.total_games} games")

    _atomic_write_json(run_dir / "heartbeat.json", {
        "schema": "guandan-marvel-dmc-heartbeat-v1",
        "phase": "precheck",
        "elapsed_seconds": elapsed(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    # ---- fail-fast: the m09 lane must be runnable on this runtime ----------
    log("[DMC-GATE-A] m09 lane precheck (official loader + native identity)")
    precheck = m09_precheck(
        dataset, plan, device=learner_device, loader=m09_loader,
        load_legacy=legacy_loader is None, warmup=m09_loader is None,
    )
    log(f"[DMC-GATE-A] m09 precheck READY sha={precheck['checkpoint_sha256'][:12]} "
        f"iter={precheck['checkpoint_iteration']} identity={precheck['identity_hash'][:12]}")
    m09_artifact = (m09_loader or load_m09_artifact)(dataset, plan, device=learner_device)
    legacy_artifact = (legacy_loader or load_legacy_artifact)(dataset, plan)

    # ---- resume chain -----------------------------------------------------
    resume_block = plan["resume"]
    resume_path, resume_record = discover_resume(
        input_root=(
            Path(resume_input_root) if resume_input_root is not None
            else Path(resume_block["input_root"])
        ),
        filename=str(resume_block["filename"]),
        run_name=str(config.run_name),
    )
    if resume_path is None and resume_input_root is None:
        # A declared-required resume that is missing is fatal; a fresh start is
        # legal only when the plan allows it.
        if str(resume_block.get("requirement", "optional")) == "required":
            raise DMCConfigError(
                "the plan requires a resume checkpoint but none was found at "
                f"{resume_block['input_root']}/**/{resume_block['filename']}"
            )
    log(f"[DMC-GATE-A] resume: {resume_record['status']}"
        + (f" from {resume_record.get('path')} (games={resume_record.get('games_completed')})"
           if resume_path is not None else ""))

    # ---- training ---------------------------------------------------------
    run_config = dataclasses.replace(config, output_dir=str(run_dir))
    train_started = elapsed()
    summary = train_from_config_parallel(
        run_config,
        device_override=learner_device,
        collector_device_override=collector_device,
        resume_path=resume_path,
        output_dir_override=run_dir,
        log=log,
    )
    train_seconds = elapsed() - train_started
    final_checkpoint = run_dir / "checkpoint_final.pt"
    if not final_checkpoint.is_file():
        raise DMCConfigError("the parallel trainer did not write checkpoint_final.pt")
    games_completed = int(summary["totals"]["games_completed"])
    samples_pushed = int(summary["totals"]["samples_pushed"])
    # The trainer's counters are cumulative (a resumed sampler starts at the
    # checkpoint's count), so the session delta is separated from the total.
    resumed_games = int(resume_record.get("games_completed", 0) or 0)
    resumed_samples = int(resume_record.get("samples_pushed", 0) or 0)
    session_games = max(0, games_completed - resumed_games)
    session_samples = max(0, samples_pushed - resumed_samples)
    log(f"[DMC-GATE-A] training done: {summary['stop_reason']} games={games_completed} "
        f"(+{session_games} this session) samples={samples_pushed} wall={train_seconds:.0f}s")

    # ---- evaluation -------------------------------------------------------
    eval_started = elapsed()
    agent = _load_agent(run_config, final_checkpoint, device=learner_device)
    lane_summaries: dict[str, dict[str, Any]] = {}
    lane_records: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for spec in schedule["lanes"]:
        kind = str(spec["kind"])
        remaining = total_budget - elapsed()
        expected = float(spec["expected_seconds"])
        if remaining < expected * 0.8:
            log(f"[DMC-GATE-A] skipping lane {kind}: remaining {remaining:.0f}s < "
                f"expected {expected:.0f}s")
            skipped.append(kind)
            lane_records[kind] = {"status": "SKIPPED_BUDGET", "remaining_seconds": remaining,
                                  "expected_seconds": expected}
            continue
        _atomic_write_json(run_dir / "heartbeat.json", {
            "schema": "guandan-marvel-dmc-heartbeat-v1",
            "phase": f"eval_{kind}",
            "elapsed_seconds": elapsed(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        log(f"[DMC-GATE-A] lane {kind}: {spec['games']} games, driver {LANE_DRIVERS[kind]}, "
            f"{len(LANE_ROTATIONS[kind])} rotations, ε=0")
        lane_started = elapsed()
        result = play_lane_games(
            agent,
            kind=kind,
            games=int(spec["games"]),
            base_seed=int(spec["seed_base"]),
            device=learner_device,
            max_obs_tokens=int(config.max_obs_tokens),
            obs_head_tokens=int(config.obs_head_tokens),
            legacy_artifact=legacy_artifact,
            m09_artifact=m09_artifact,
            log=log,
        )
        lane_seconds = elapsed() - lane_started
        summary_row = dict(result["summary"])
        summary_row["expected_seconds"] = expected
        summary_row["measured_seconds"] = lane_seconds
        lane_summaries[kind] = summary_row
        lane_records[kind] = {"status": "RUN", "measured_seconds": lane_seconds,
                              "expected_seconds": expected}
        _append_jsonl(eval_dir / f"{kind}.rows.jsonl", summary_row)
        for row in result["rows"]:
            _append_jsonl(eval_dir / f"{kind}.rows.jsonl", row)
        _atomic_write_json(eval_dir / f"{kind}.summary.json", summary_row)
        log(f"[DMC-GATE-A] lane {kind}: win_rate={summary_row['win_rate']} "
            f"games={summary_row['games_completed']}/{summary_row['games']} "
            f"{lane_seconds:.0f}s ({summary_row['games_per_second']:.3f} games/s)")

    wall_seconds = elapsed()
    eval_seconds = wall_seconds - eval_started
    if eval_seconds > eval_reserve:
        log(f"[DMC-GATE-A] evaluation overran the {eval_reserve:.0f}s reserve by "
            f"{eval_seconds - eval_reserve:.0f}s")
    eval_health = {
        key: sum(int(row.get(key, 0)) for row in lane_summaries.values())
        for key in ("illegal", "truncated", "nonfinite")
    }
    train_health = dict(summary.get("health") or {})
    gate_result = evaluate_gates(
        gates,
        lanes=lane_summaries,
        train_health=train_health,
        eval_health=eval_health,
        skipped=skipped,
    )

    # ---- result + evidence ------------------------------------------------
    result = {
        "schema": RESULT_SCHEMA,
        "plan_hash": plan["plan_hash"],
        "status": summary["stop_reason"],
        "env_driver": plan.get("env_driver"),
        "topology": {
            "learner_device": summary.get(PARALLEL_SUMMARY_KEY, {}).get("learner_device"),
            "collector_device": summary.get(PARALLEL_SUMMARY_KEY, {}).get("collector_device"),
            "single_device_degraded": summary.get(PARALLEL_SUMMARY_KEY, {}).get(
                "single_device_degraded"),
            "behavioral_weight_lag_cycles": summary.get(PARALLEL_SUMMARY_KEY, {}).get(
                "behavioral_weight_lag_cycles"),
            "gpus": names,
        },
        "resume": resume_record,
        "session": {
            "games_completed_this_session": session_games,
            "samples_pushed_this_session": session_samples,
            "games_completed_cumulative": games_completed,
            "samples_pushed_cumulative": samples_pushed,
            "train_seconds_this_session": train_seconds,
            "eval_seconds_this_session": eval_seconds,
            "eval_reserve_seconds": eval_reserve,
            "eval_overrun_seconds": max(0.0, eval_seconds - eval_reserve),
            "wall_seconds_this_session": wall_seconds,
            "total_games_target": int(config.total_games),
            "cycles": int(summary["totals"]["cycles"]),
            "train_steps": int(summary["totals"]["train_steps"]),
            "low_buffer_skips": int(summary["totals"]["low_buffer_skips"]),
            "step_skipped_total": int(summary["totals"]["step_skipped_total"]),
            "nonfinite_count": int(summary["totals"]["nonfinite_count"]),
            "stop_reason": summary["stop_reason"],
        },
        "throughput": {
            **dict(summary["throughput"]),
            "games_per_second_this_session": (
                session_games / train_seconds if train_seconds > 0 else None
            ),
            "samples_per_second_this_session": (
                session_samples / train_seconds if train_seconds > 0 else None
            ),
        },
        "m09_lane": precheck,
        "lanes": lane_summaries,
        "lane_records": lane_records,
        "gates": gate_result,
        "stage7_projection": stage7_projection(
            games_completed=session_games,
            samples_pushed=session_samples,
            train_seconds=train_seconds,
        ),
        "boundaries": plan["boundaries"],
        "deviations": plan["deviations"],
        "note": (
            "pipeline fetch marks DMC runs FETCHED_INCOMPLETE because its result verifier is "
            "serial-compare bound; this RESULT.json + RESULTS_MANIFEST.json are the "
            "authoritative Gate A verification"
        ),
    }
    result["hash"] = canonical({key: value for key, value in result.items() if key != "hash"})
    _atomic_write_json(artifacts_dir / "RESULT.json", result)

    resume_copy = run_root / RESUME_FILENAME
    shutil.copyfile(final_checkpoint, resume_copy)
    interval = sorted(run_dir.glob("checkpoint_cycle_*.pt"))
    members: list[tuple[str, Path]] = [
        ("artifacts/RESULT.json", artifacts_dir / "RESULT.json"),
        ("artifacts/training/metrics.jsonl", run_dir / "metrics.jsonl"),
        ("artifacts/training/summary.json", run_dir / "summary.json"),
        ("artifacts/training/progress.json", run_dir / "progress.json"),
    ]
    for kind in lane_summaries:
        members.append((f"artifacts/eval/{kind}.rows.jsonl", eval_dir / f"{kind}.rows.jsonl"))
        members.append((f"artifacts/eval/{kind}.summary.json",
                        eval_dir / f"{kind}.summary.json"))
    for path in interval[-_ZIP_CHECKPOINT_KEEP:]:
        members.append((f"artifacts/checkpoints/{path.name}", path))
    members.append((RESUME_FILENAME, resume_copy))

    recovery = {
        "schema": RECOVERY_SCHEMA,
        "resume_checkpoint": RESUME_FILENAME,
        "resume_sha256": _sha(resume_copy),
        "resume_bytes": resume_copy.stat().st_size,
        "interval_checkpoints": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}
            for path in interval
        ],
        "resume_hint": (
            "upload this zip's dmc_gate_a_resume.pt as a Kaggle dataset file named "
            f"{RESUME_FILENAME}; the next session picks it up under /kaggle/input and "
            "resumes at its cycle count with replay_reset=true"
        ),
        "replay_reset_expected": True,
    }
    _atomic_write_json(artifacts_dir / "recovery_index.json", recovery)
    members.insert(1, ("artifacts/recovery_index.json", artifacts_dir / "recovery_index.json"))

    file_rows = [
        {"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)}
        for name, path in members
    ]
    manifest = {
        "schema": RESULTS_MANIFEST_SCHEMA,
        "plan_hash": plan["plan_hash"],
        "mode": "full",
        "evidence_complete": bool(
            not skipped
            and all(
                lane_summaries.get(str(spec["kind"]), {}).get("games_completed")
                == int(spec["games"])
                for spec in schedule["lanes"]
            )
            and final_checkpoint.is_file()
        ),
        "omissions": [{"lane": kind, "reason": "budget"} for kind in skipped],
        "files": file_rows,
    }
    manifest["hash"] = canonical({key: value for key, value in manifest.items() if key != "hash"})
    manifest_path = run_root / "RESULTS_MANIFEST.json"
    _atomic_write_json(manifest_path, manifest)
    _zip_files([(manifest_path.name, manifest_path), *members], run_root / RESULTS_ZIP_NAME)
    log(f"[DMC-GATE-A] done: {summary['stop_reason']} games={games_completed} "
        f"verdict={gate_result['overall_verdict']} "
        f"rule={gate_result['vs_rule']['measured']} m09={gate_result['h2h_m09']['measured']}")
    return result


__all__ = [
    "PLAN_SCHEMA",
    "RESULT_SCHEMA",
    "RESULTS_MANIFEST_SCHEMA",
    "RESULTS_ZIP_NAME",
    "RESUME_FILENAME",
    "M09LaneUnavailable",
    "canonical",
    "discover_resume",
    "evaluate_gates",
    "run_notebook",
    "stage7_projection",
]
