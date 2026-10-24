# SPDX-License-Identifier: MIT
"""Comparison worker: sharded, resumable, heartbeat-monitored task loop."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from guandan_marvel.evaluation.common import atomic_write_json

from .driver import play_comparison_game
from .loader import load_control, load_deals, load_fabledan_model, load_v2_artifact, safe_control_file, validate_control_files
from .plan import BOOTCAMP_FINAL_CONTROL_KEY, BOOTCAMP_FINAL_KIND, enumerate_tasks

WORKER_SCHEMA = "guandan-marvel-comparison-worker-v1"


def gpu_snapshot() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        rows = []
        for line in result.stdout.splitlines():
            values = [x.strip() for x in line.split(",")]
            if len(values) == 6:
                rows.append({
                    "index": int(values[0]), "name": values[1],
                    "utilization_percent": float(values[2]),
                    "memory_used_mib": float(values[3]), "memory_total_mib": float(values[4]),
                    "power_watts": float(values[5]),
                })
        return rows
    except Exception as error:
        return [{"status": "unavailable", "error": f"{type(error).__name__}: {error}"}]


def _load_shard_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def run_worker(args: argparse.Namespace) -> int:
    control, base = load_control(args.control)
    validate_control_files(control, base)
    deals, rows = load_deals(base, control)
    all_tasks = enumerate_tasks(control, deals)
    tasks = all_tasks[args.worker_index :: args.workers]
    if args.max_tasks is not None:
        tasks = tasks[: int(args.max_tasks)]

    output_dir = Path(args.working_root).resolve() / "comparison_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"games.worker{args.worker_index}.jsonl"
    previous = _load_shard_records(shard_path)
    completed = {str(row["task_id"]) for row in previous}
    resumed_errors = sum(1 for row in previous if row.get("status") == "error")

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(json.dumps({"event": "comparison_device_fallback", "requested": device, "using": "cpu"}), flush=True)
        device = "cpu"
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    arm_map = {str(arm["id"]): arm for arm in control["arms"]}
    kinds = {str(arm_map[str(task["arm_id"])]["candidate_kind"]) for task in tasks}
    opponent_kinds = {str(arm_map[str(task["arm_id"])]["opponent_kind"]) for task in tasks}
    loaders: dict[str, str] = {}
    m09_artifacts: dict[str, object] = {}
    for control_key, kind in (
        ("candidate", "m09"),
        ("champion", "m09_champion"),
        (BOOTCAMP_FINAL_CONTROL_KEY, BOOTCAMP_FINAL_KIND),
    ):
        if kind not in kinds and kind not in opponent_kinds:
            continue
        artifact, mode = load_v2_artifact(
            safe_control_file(base, control[control_key], control_key),
            device=device, expected_algorithm="ippo",
        )
        m09_artifacts[kind] = artifact
        loaders[kind] = mode
    fabledan_model = None
    if "fabledan" in kinds or "fabledan" in opponent_kinds:
        fabledan_model = load_fabledan_model(safe_control_file(base, control["fabledan_weights"], "fabledan_weights"))
        loaders["fabledan"] = "numpy-vendor"

    started = time.monotonic()
    last_beat = started
    completed_count = 0
    errors = 0
    consecutive_errors = 0
    skipped: dict[str, int] = {}
    total = len(tasks)
    plan_by_arm: dict[str, int] = {}
    for task in tasks:
        arm_id = str(task["arm_id"])
        plan_by_arm[arm_id] = plan_by_arm.get(arm_id, 0) + 1
    done_by_arm: dict[str, int] = {}
    print(json.dumps({
        "event": "comparison_worker_start",
        "worker_index": args.worker_index,
        "workers": args.workers,
        "device": str(device),
        "tasks": total,
        "resumed_from_shard": len(completed),
        "artifact_loaders": loaders,
        "arm_plan": plan_by_arm,
    }, sort_keys=True), flush=True)

    with shard_path.open("a", encoding="utf-8", newline="\n") as stream:
        for task in tasks:
            task_id = str(task["task_id"])
            if task_id in completed:
                completed_count += 1
                continue
            elapsed = time.monotonic() - started
            if int(task["priority"]) > 0 and float(args.soft_time_limit) > 0 and elapsed > float(args.soft_time_limit):
                arm_id = str(task["arm_id"])
                skipped[arm_id] = skipped.get(arm_id, 0) + 1
                continue
            arm = arm_map[str(task["arm_id"])]
            deal = rows[int(task["deal_index"])]
            record: dict[str, object]
            try:
                record = play_comparison_game(
                    arm_id=str(task["arm_id"]),
                    task_id=task_id,
                    candidate_kind=str(arm["candidate_kind"]),
                    opponent_kind=str(arm["opponent_kind"]),
                    candidate_label=str(arm["candidate_label"]),
                    opponent_label=str(arm["opponent_label"]),
                    deal=deal,
                    rotation=int(task["rotation"]),
                    m09_artifacts=m09_artifacts,
                    fabledan_model=fabledan_model,
                    temperature=float(control.get("temperature", 0.0)),
                    device=device,
                    deep_verify=bool(control.get("worker_deep_verify", False)),
                )
                record["artifact_loaders"] = dict(loaders)
                consecutive_errors = 0
            except Exception as error:  # noqa: BLE001 - recorded per task
                errors += 1
                consecutive_errors += 1
                record = {
                    "schema": "guandan-marvel-comparison-game-v1",
                    "status": "error",
                    "task_id": task_id,
                    "arm_id": str(task["arm_id"]),
                    "deal_id": str(task["deal_id"]),
                    "seat_rotation": int(task["rotation"]),
                    "error": f"{type(error).__name__}: {error}",
                }
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                if consecutive_errors >= 5:
                    raise RuntimeError("aborting worker: five consecutive comparison errors")
                completed_count += 1
                continue
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            completed_count += 1
            arm_key = str(task["arm_id"])
            done_by_arm[arm_key] = done_by_arm.get(arm_key, 0) + 1
            now = time.monotonic()
            if now - last_beat >= float(args.heartbeat_seconds):
                elapsed_now = now - started
                print(json.dumps({
                    "event": "comparison_worker_heartbeat",
                    "worker_index": args.worker_index,
                    "completed": completed_count,
                    "total": total,
                    "current_arm": arm_key,
                    "done_by_arm": dict(done_by_arm),
                    "games_per_second": completed_count / elapsed_now if elapsed_now > 0 else None,
                    "errors": errors + resumed_errors,
                    "skipped": skipped,
                    "elapsed_seconds": elapsed_now,
                    "eta_seconds": None if completed_count == 0 else elapsed_now * (total - completed_count) / completed_count,
                    "gpu": gpu_snapshot(),
                }, sort_keys=True), flush=True)
                last_beat = now

    status = {
        "schema": WORKER_SCHEMA,
        "worker_index": int(args.worker_index),
        "workers": int(args.workers),
        "tasks": total,
        "completed": completed_count,
        "errors_including_resume": errors + resumed_errors,
        "skipped_due_to_time": skipped,
        "artifact_loaders": loaders,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(output_dir / f"worker{args.worker_index}_status.json", status)
    print(json.dumps({"event": "comparison_worker_done", **status}, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--working-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--soft-time-limit", type=float, default=0.0, help="seconds; lower-priority arms stop afterwards")
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0 or not 0 <= args.worker_index < args.workers:
        raise SystemExit("invalid worker shard arguments")
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
