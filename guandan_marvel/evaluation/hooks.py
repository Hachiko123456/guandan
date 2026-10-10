# SPDX-License-Identifier: MIT
"""Process-isolated periodic training evaluator hook with durable result delivery."""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import multiprocessing
import os
from pathlib import Path
from threading import Lock
from typing import Mapping

from .checkpoints import freeze_merged_policy
from .common import atomic_write_json, file_sha256, payload_hash

HOOK_SCHEMA = "guandan-marvel-training-evaluator-hook-v2"
HOOK_EVENT_SCHEMA = "guandan-marvel-training-evaluator-event-v1"
HOOK_WORK_SCHEMA = "guandan-marvel-training-evaluator-work-v1"
_TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_entrypoint(value: str):
    module_name, separator, attribute = str(value).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("evaluation worker entrypoint must be 'module:function'")
    target = importlib.import_module(module_name)
    function = getattr(target, attribute, None)
    if not callable(function):
        raise TypeError(f"evaluation worker entrypoint is not callable: {value}")
    return function


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _worker_main(work: Mapping[str, object]) -> dict[str, object]:
    """Top-level spawn target. It never receives a live trainer object."""
    status_path = Path(str(work["worker_status_path"]))
    running = {
        "schema": HOOK_EVENT_SCHEMA,
        "task_id": work["task_id"],
        "iteration": int(work["iteration"]),
        "status": "running",
        "snapshot": work["snapshot"],
        "snapshot_sha256": work["snapshot_sha256"],
        "started_at": _timestamp(),
    }
    running["hash"] = payload_hash(running)
    atomic_write_json(status_path, running)
    try:
        function = _resolve_entrypoint(str(work["entrypoint"]))
        result = function(
            snapshot=Path(str(work["snapshot"])),
            iteration=int(work["iteration"]),
            config=dict(work.get("config", {})),
        )
        event: dict[str, object] = {
            "schema": HOOK_EVENT_SCHEMA,
            "task_id": work["task_id"],
            "iteration": int(work["iteration"]),
            "status": "completed",
            "snapshot": work["snapshot"],
            "snapshot_sha256": work["snapshot_sha256"],
            "result": result,
            "finished_at": _timestamp(),
        }
    except BaseException as error:
        event = {
            "schema": HOOK_EVENT_SCHEMA,
            "task_id": work["task_id"],
            "iteration": int(work["iteration"]),
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "snapshot": work["snapshot"],
            "snapshot_sha256": work["snapshot_sha256"],
            "error_type": type(error).__name__,
            "error": str(error),
            "finished_at": _timestamp(),
        }
    event["hash"] = payload_hash(event)
    atomic_write_json(status_path, event)
    return event


@dataclass
class _Task:
    task_id: str
    iteration: int
    snapshot: Path
    snapshot_sha256: str
    task_dir: Path
    future: Future
    terminal_event: dict[str, object] | None = None
    delivered: bool = False


class TrainingEvaluatorHook:
    """Freeze policy at durable boundaries and evaluate in spawned processes.

    Process workers receive only the immutable snapshot path, iteration, a
    versioned entrypoint string, JSON-compatible config, and output paths.
    They never receive the live trainer, optimizer, sampler, replay, or RNG.
    """

    def __init__(
        self,
        *,
        interval: int,
        snapshot_dir: str | Path,
        evaluator_entrypoint: str,
        evaluator_config: Mapping[str, object] | None = None,
        max_pending: int = 1,
    ):
        if isinstance(interval, bool) or int(interval) <= 0:
            raise ValueError("evaluation interval must be positive")
        if isinstance(max_pending, bool) or int(max_pending) <= 0:
            raise ValueError("max_pending must be positive")
        _resolve_entrypoint(evaluator_entrypoint)
        self.interval = int(interval)
        self.snapshot_dir = Path(snapshot_dir).resolve()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator_entrypoint = str(evaluator_entrypoint)
        self.evaluator_config = dict(evaluator_config or {})
        # Validate that config is spawn/JSON friendly before training starts.
        json.dumps(self.evaluator_config, sort_keys=True, allow_nan=False)
        self.max_pending = int(max_pending)
        self.journal_path = self.snapshot_dir / "evaluation_hook_events.jsonl"
        self.executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self._lock = Lock()
        self._tasks: dict[int, _Task] = {}
        self._ready: list[dict[str, object]] = []
        self._seen_iterations: set[int] = set()
        self._closed = False

    def _record(self, event: Mapping[str, object]) -> dict[str, object]:
        value = dict(event)
        value.setdefault("schema", HOOK_EVENT_SCHEMA)
        value.setdefault("recorded_at", _timestamp())
        value["hash"] = payload_hash(value)
        _append_jsonl(self.journal_path, value)
        return value

    def _harvest_done(self) -> None:
        for iteration, task in sorted(self._tasks.items()):
            if task.terminal_event is not None or not task.future.done():
                continue
            if task.future.cancelled():
                event = {
                    "schema": HOOK_EVENT_SCHEMA,
                    "task_id": task.task_id,
                    "iteration": iteration,
                    "status": "cancelled",
                    "snapshot": str(task.snapshot),
                    "snapshot_sha256": task.snapshot_sha256,
                    "finished_at": _timestamp(),
                }
            else:
                try:
                    raw = task.future.result()
                    if not isinstance(raw, Mapping) or raw.get("status") not in _TERMINAL:
                        raise RuntimeError("evaluation worker returned a malformed terminal event")
                    event = dict(raw)
                except BaseException as error:
                    event = {
                        "schema": HOOK_EVENT_SCHEMA,
                        "task_id": task.task_id,
                        "iteration": iteration,
                        "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        "snapshot": str(task.snapshot),
                        "snapshot_sha256": task.snapshot_sha256,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "finished_at": _timestamp(),
                    }
            event["hash"] = payload_hash(event)
            atomic_write_json(task.task_dir / "result.json", event)
            task.terminal_event = self._record(event)
            self._ready.append(dict(task.terminal_event))

    def collect_ready(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            self._harvest_done()
            ready = tuple(dict(event) for event in self._ready)
            self._ready.clear()
            for event in ready:
                task = self._tasks.get(int(event["iteration"]))
                if task is not None:
                    task.delivered = True
                self._record({
                    "schema": HOOK_EVENT_SCHEMA,
                    "task_id": event.get("task_id"),
                    "iteration": int(event["iteration"]),
                    "status": "delivered",
                    "terminal_status": event["status"],
                    "delivered_at": _timestamp(),
                })
            return ready

    def poll(self) -> dict[int, object]:
        """Compatibility view. Failures are returned as events, never raised/lost."""
        return {int(event["iteration"]): event for event in self.collect_ready()}

    def on_iteration_completed(self, trainer: object, *, iteration: int) -> dict[str, object]:
        value = int(iteration)
        if self._closed:
            return self._record({"iteration": value, "status": "closed"})
        if value % self.interval != 0:
            return {"schema": HOOK_SCHEMA, "status": "not_due", "iteration": value}
        with self._lock:
            self._harvest_done()
            if value in self._seen_iterations:
                return self._record({"iteration": value, "status": "duplicate_rejected"})
            self._seen_iterations.add(value)
            pending = sum(task.terminal_event is None and not task.future.done() for task in self._tasks.values())
            if pending >= self.max_pending:
                return self._record({"iteration": value, "status": "skipped_pending"})

            staging = self.snapshot_dir / f".iter{value:06d}.staging-{os.getpid()}.pt"
            freeze_merged_policy(trainer, staging, iteration=value)
            digest = file_sha256(staging)
            snapshot = self.snapshot_dir / f"guandan-marvel-v2-{trainer.algorithm}-eval-iter{value:06d}-{digest[:12]}.pt"
            os.replace(staging, snapshot)
            task_id = f"iter{value:06d}-{digest[:12]}"
            task_dir = self.snapshot_dir / "tasks" / task_id
            task_dir.mkdir(parents=True, exist_ok=False)
            work: dict[str, object] = {
                "schema": HOOK_WORK_SCHEMA,
                "task_id": task_id,
                "iteration": value,
                "snapshot": str(snapshot),
                "snapshot_sha256": digest,
                "entrypoint": self.evaluator_entrypoint,
                "config": dict(self.evaluator_config),
                "worker_status_path": str(task_dir / "worker_status.json"),
            }
            atomic_write_json(task_dir / "work.json", work)
            future = self.executor.submit(_worker_main, work)
            task = _Task(task_id, value, snapshot, digest, task_dir, future)
            self._tasks[value] = task
            return self._record({
                "schema": HOOK_EVENT_SCHEMA,
                "task_id": task_id,
                "iteration": value,
                "status": "scheduled",
                "snapshot": str(snapshot),
                "snapshot_sha256": digest,
                "work_path": str((task_dir / "work.json").resolve()),
            })

    def status_records(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            self._harvest_done()
            records = []
            for iteration, task in sorted(self._tasks.items()):
                if task.terminal_event is not None:
                    records.append(dict(task.terminal_event))
                else:
                    records.append({
                        "schema": HOOK_EVENT_SCHEMA,
                        "task_id": task.task_id,
                        "iteration": iteration,
                        "status": "pending",
                        "snapshot": str(task.snapshot),
                        "snapshot_sha256": task.snapshot_sha256,
                    })
            return tuple(records)

    def close(self, *, wait: bool = True) -> tuple[dict[str, object], ...]:
        with self._lock:
            if self._closed:
                return tuple()
            self._closed = True
            if not wait:
                for task in self._tasks.values():
                    if task.terminal_event is None:
                        if task.future.cancel():
                            event = {
                                "schema": HOOK_EVENT_SCHEMA,
                                "task_id": task.task_id,
                                "iteration": task.iteration,
                                "status": "cancelled",
                                "snapshot": str(task.snapshot),
                                "snapshot_sha256": task.snapshot_sha256,
                                "finished_at": _timestamp(),
                            }
                            event["hash"] = payload_hash(event)
                            atomic_write_json(task.task_dir / "result.json", event)
                            task.terminal_event = self._record(event)
                            self._ready.append(dict(task.terminal_event))
                        else:
                            self._record({
                                "task_id": task.task_id,
                                "iteration": task.iteration,
                                "status": "shutdown_pending",
                                "snapshot": str(task.snapshot),
                                "snapshot_sha256": task.snapshot_sha256,
                            })
        self.executor.shutdown(wait=wait, cancel_futures=not wait)
        if wait:
            with self._lock:
                self._harvest_done()
        return self.collect_ready()


__all__ = [
    "HOOK_EVENT_SCHEMA", "HOOK_SCHEMA", "HOOK_WORK_SCHEMA", "TrainingEvaluatorHook",
]
