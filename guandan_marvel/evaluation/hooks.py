# SPDX-License-Identifier: MIT
"""Non-blocking periodic training evaluator hook."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Callable

from .checkpoints import freeze_merged_policy

HOOK_SCHEMA = "guandan-marvel-training-evaluator-hook-v1"


class TrainingEvaluatorHook:
    """Freeze merged policy every N durable iterations and evaluate in a worker.

    The caller thread only clones/writes an immutable evaluation-only snapshot
    and submits work. Evaluation never receives the live trainer, optimizer,
    sampler, RNG, or replay objects.
    """

    def __init__(
        self,
        *,
        interval: int,
        snapshot_dir: str | Path,
        evaluator: Callable[[Path, int], object],
        max_pending: int = 1,
    ):
        if isinstance(interval, bool) or int(interval) <= 0:
            raise ValueError("evaluation interval must be positive")
        if isinstance(max_pending, bool) or int(max_pending) <= 0:
            raise ValueError("max_pending must be positive")
        self.interval = int(interval)
        self.snapshot_dir = Path(snapshot_dir)
        self.evaluator = evaluator
        self.max_pending = int(max_pending)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guandan-m06-eval")
        self._lock = Lock()
        self._futures: dict[int, Future] = {}
        self._closed = False

    def on_iteration_completed(self, trainer: object, *, iteration: int) -> dict[str, object]:
        if self._closed:
            return {"schema": HOOK_SCHEMA, "status": "closed", "iteration": int(iteration)}
        if int(iteration) % self.interval != 0:
            return {"schema": HOOK_SCHEMA, "status": "not_due", "iteration": int(iteration)}
        with self._lock:
            self._futures = {key: future for key, future in self._futures.items() if not future.done()}
            if len(self._futures) >= self.max_pending:
                return {"schema": HOOK_SCHEMA, "status": "skipped_pending", "iteration": int(iteration)}
            snapshot = self.snapshot_dir / f"guandan-marvel-v2-{trainer.algorithm}-eval-iter{int(iteration):06d}.pt"
            freeze_merged_policy(trainer, snapshot, iteration=int(iteration))
            future = self.executor.submit(self.evaluator, snapshot, int(iteration))
            self._futures[int(iteration)] = future
        return {
            "schema": HOOK_SCHEMA,
            "status": "scheduled",
            "iteration": int(iteration),
            "snapshot": str(snapshot.resolve()),
        }

    def poll(self) -> dict[int, object]:
        completed: dict[int, object] = {}
        with self._lock:
            for iteration, future in list(self._futures.items()):
                if not future.done():
                    continue
                completed[iteration] = future.result()
                del self._futures[iteration]
        return completed

    def close(self, *, wait: bool = True) -> None:
        self._closed = True
        self.executor.shutdown(wait=wait, cancel_futures=not wait)


__all__ = ["HOOK_SCHEMA", "TrainingEvaluatorHook"]
