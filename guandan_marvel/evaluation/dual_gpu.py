# SPDX-License-Identifier: MIT
"""Deterministic independent-worker orchestration for formal evaluation."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from .common import atomic_write_json, payload_hash
from .resumable import IncompleteEvaluationError, ResumableEvaluationStore, task_key, validate_resumable_plan

DUAL_GPU_RUN_SCHEMA = "guandan-marvel-deterministic-worker-group-v1"
CONSISTENCY_SCHEMA = "guandan-marvel-single-dual-consistency-v1"


class FormalWorkerFailure(RuntimeError):
    pass


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def deterministic_task_shards(plan: Mapping[str, object], world_size: int) -> list[list[int]]:
    validate_resumable_plan(plan)
    if world_size <= 0:
        raise ValueError("formal worker world size must be positive")
    shards = [[] for _ in range(world_size)]
    for task in plan["tasks"]:
        index = int(task["task_index"])
        shards[index % world_size].append(index)
    return shards


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    worker_index: int
    device_index: int
    command: tuple[str, ...]


class DeterministicWorkerGroup:
    """Launch fixed GPU shards; never schedules work by completion speed."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        plan: Mapping[str, object],
        commands: Sequence[WorkerCommand],
        base_env: Mapping[str, str] | None = None,
        heartbeat_seconds: float = 5.0,
    ) -> None:
        validate_resumable_plan(plan)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plan = dict(plan)
        self.commands = tuple(commands)
        self.base_env = dict(os.environ if base_env is None else base_env)
        self.heartbeat_seconds = float(heartbeat_seconds)
        if not self.commands:
            raise ValueError("formal worker group requires at least one command")
        world_size = len(self.commands)
        indices = sorted(command.worker_index for command in self.commands)
        devices = sorted(command.device_index for command in self.commands)
        if indices != list(range(world_size)) or len(set(devices)) != world_size:
            raise ValueError("formal worker commands must cover unique contiguous workers/devices")
        self.shards = deterministic_task_shards(self.plan, world_size)

    def run(self) -> dict[str, object]:
        started = time.monotonic()
        processes: dict[int, subprocess.Popen[str]] = {}
        threads: list[threading.Thread] = []
        logs: dict[int, Path] = {}
        lock = threading.Lock()
        statuses: dict[int, dict[str, object]] = {}

        def forward(worker: int, process: subprocess.Popen[str], log_path: Path) -> None:
            assert process.stdout is not None
            with log_path.open("a", encoding="utf-8", newline="\n") as stream:
                for line in process.stdout:
                    text = line.rstrip("\n")
                    with lock:
                        print(f"[FORMAL worker={worker}] {text}", flush=True)
                        stream.write(text + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())

        for command in self.commands:
            env = dict(self.base_env)
            env.update({
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": str(command.device_index),
                "GUANDAN_FORMAL_WORKER_INDEX": str(command.worker_index),
                "GUANDAN_FORMAL_WORLD_SIZE": str(len(self.commands)),
                "GUANDAN_FORMAL_PLAN_HASH": str(self.plan["hash"]),
            })
            log_path = self.output_dir / f"worker{command.worker_index}.log"
            logs[command.worker_index] = log_path
            process = subprocess.Popen(
                list(command.command),
                cwd=self.output_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes[command.worker_index] = process
            thread = threading.Thread(
                target=forward,
                args=(command.worker_index, process, log_path),
                name=f"formal-worker-{command.worker_index}-forwarder",
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        while any(process.poll() is None for process in processes.values()):
            time.sleep(min(self.heartbeat_seconds, 0.2))
            snapshot = {
                "schema": "guandan-marvel-formal-worker-heartbeat-v1",
                "plan_hash": self.plan["hash"],
                "elapsed_seconds": time.monotonic() - started,
                "workers": [
                    {
                        "worker_index": worker,
                        "pid": process.pid,
                        "status": "running" if process.poll() is None else "exited",
                        "exit_code": process.poll(),
                        "assigned_tasks": len(self.shards[worker]),
                    }
                    for worker, process in sorted(processes.items())
                ],
                "heartbeat_at": _timestamp(),
            }
            atomic_write_json(self.output_dir / "worker_group_heartbeat.json", snapshot)

        for thread in threads:
            thread.join(timeout=5.0)
        failed: list[dict[str, object]] = []
        for worker, process in sorted(processes.items()):
            exit_code = int(process.returncode or 0)
            status = {
                "worker_index": worker,
                "device_index": next(command.device_index for command in self.commands if command.worker_index == worker),
                "pid": process.pid,
                "exit_code": exit_code,
                "assigned_tasks": len(self.shards[worker]),
                "log": str(logs[worker]),
            }
            statuses[worker] = status
            if exit_code != 0:
                failed.append(status)
        body: dict[str, object] = {
            "schema": DUAL_GPU_RUN_SCHEMA,
            "plan_hash": self.plan["hash"],
            "world_size": len(self.commands),
            "sharding": "task_index_mod_world_size",
            "shards": self.shards,
            "workers": [statuses[index] for index in sorted(statuses)],
            "status": "failed" if failed else "complete",
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": _timestamp(),
        }
        result = {**body, "hash": payload_hash(body)}
        atomic_write_json(self.output_dir / "worker_group_result.json", result)
        if failed:
            raise FormalWorkerFailure(f"formal evaluation worker failure blocks merge/promotion: {failed}")
        return result


def compare_formal_outputs(single_dir: str | Path, dual_dir: str | Path) -> dict[str, object]:
    single_root = Path(single_dir).resolve()
    dual_root = Path(dual_dir).resolve()
    single_plan = json.loads((single_root / "evaluation_plan.json").read_text(encoding="utf-8"))
    dual_plan = json.loads((dual_root / "evaluation_plan.json").read_text(encoding="utf-8"))
    if single_plan != dual_plan:
        raise ValueError("single/dual evaluation plans differ")
    single_store = ResumableEvaluationStore(single_root, single_plan)
    dual_store = ResumableEvaluationStore(dual_root, dual_plan)
    single_records, _ = single_store.load_records([single_root / "games.jsonl"])
    dual_records, _ = dual_store.load_records([dual_root / "games.jsonl"])
    ordered_single = [single_records[task_key(task)] for task in single_plan["tasks"]]
    ordered_dual = [dual_records[task_key(task)] for task in dual_plan["tasks"]]
    fields = (
        "opponent_id", "deal_id", "seat_rotation", "candidate_win", "winner_team",
        "ranking", "outcome_class", "illegal_count", "truncated_count", "token_steps",
        "committed_steps", "action_trace_sha256", "formal_record_hash",
    )
    mismatches: list[dict[str, object]] = []
    for index, (left, right) in enumerate(zip(ordered_single, ordered_dual)):
        different = [field for field in fields if left.get(field) != right.get(field)]
        if different:
            mismatches.append({"task_index": index, "fields": different})
    single_summary = json.loads((single_root / "evaluation_summary.json").read_text(encoding="utf-8"))
    dual_summary = json.loads((dual_root / "evaluation_summary.json").read_text(encoding="utf-8"))
    result_body = {
        "schema": CONSISTENCY_SCHEMA,
        "plan_hash": single_plan["hash"],
        "task_count": len(ordered_single),
        "records_identical": ordered_single == ordered_dual,
        "field_mismatches": mismatches,
        "ordering_identical": [task_key(row) for row in ordered_single] == [task_key(row) for row in ordered_dual],
        "summary_hash_identical": single_summary.get("hash") == dual_summary.get("hash"),
        "records_hash_identical": single_summary.get("records_hash") == dual_summary.get("records_hash"),
        "promotion_input_identical": {
            "plan_hash": single_summary.get("plan_hash"),
            "summary_hash": single_summary.get("hash"),
        } == {
            "plan_hash": dual_summary.get("plan_hash"),
            "summary_hash": dual_summary.get("hash"),
        },
        "complete": not mismatches and ordered_single == ordered_dual and single_summary.get("hash") == dual_summary.get("hash"),
    }
    result = {**result_body, "hash": payload_hash(result_body)}
    atomic_write_json(dual_root / "single_dual_consistency.json", result)
    if result["complete"] is not True:
        raise ValueError(f"single/dual formal evaluation differs: {result}")
    return result


__all__ = [
    "CONSISTENCY_SCHEMA", "DUAL_GPU_RUN_SCHEMA", "DeterministicWorkerGroup",
    "FormalWorkerFailure", "WorkerCommand", "compare_formal_outputs", "deterministic_task_shards",
]
