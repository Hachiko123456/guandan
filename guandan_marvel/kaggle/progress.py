# SPDX-License-Identifier: MIT
"""Notebook-visible progress heartbeat without touching training RNG."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Mapping

from .common import atomic_json
from .telemetry import gpu_snapshot

PROGRESS_SCHEMA = "guandan-marvel-progress-v2"
RANK_HEARTBEAT_SCHEMA = "guandan-marvel-rank-heartbeat-v1"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unknown(value: object) -> str:
    return "unknown" if value is None else str(value)


def format_console_summary(progress: Mapping[str, object]) -> str:
    gpus = progress.get("gpus") or []
    gpu_parts = []
    for row in gpus:
        index = row.get("index")
        if row.get("status") == "ok":
            gpu_parts.append(
                f"gpu{index}=util:{_unknown(row.get('utilization_percent'))}% "
                f"mem:{_unknown(row.get('memory_used_mib'))}/{_unknown(row.get('memory_total_mib'))}MiB"
            )
        else:
            gpu_parts.append(f"gpu{index}=unavailable")
    ranks = progress.get("ranks") or []
    rank_parts = [
        f"r{row.get('rank')}:{row.get('phase', 'unknown')} age={_unknown(row.get('heartbeat_age_seconds'))}s"
        for row in ranks
    ]
    return (
        "[M07 heartbeat] "
        f"mode={progress.get('mode')} phase={progress.get('phase')} "
        f"iteration={progress.get('iteration')}/{progress.get('target_iterations')} "
        f"games={_unknown(progress.get('completed_games'))} "
        f"env_steps={_unknown(progress.get('env_steps'))} "
        f"game_steps={_unknown(progress.get('game_steps'))} "
        f"env_sps={_unknown(progress.get('env_steps_per_second'))} "
        f"game_sps={_unknown(progress.get('game_steps_per_second'))} "
        f"elapsed={progress.get('session_elapsed_seconds', 0):.1f}s "
        f"eta={_unknown(progress.get('eta_seconds'))} "
        f"checkpoint={_unknown(progress.get('latest_checkpoint'))} "
        f"evaluation={_unknown(progress.get('latest_evaluation_status'))} "
        + " ".join(gpu_parts + rank_parts)
    )


class LiveProgressHeartbeat:
    def __init__(
        self,
        *,
        run_root: str | Path,
        rank_root: str | Path,
        mode: str,
        rank: int,
        world_size: int,
        target_iterations: int,
        session_start_iteration: int,
        interval_seconds: float = 5.0,
        is_primary: bool,
    ) -> None:
        self.run_root = Path(run_root)
        self.rank_root = Path(rank_root)
        self.mode = str(mode)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.target_iterations = int(target_iterations)
        self.session_start_iteration = int(session_start_iteration)
        self.interval_seconds = max(float(interval_seconds), 1.0)
        self.is_primary = bool(is_primary)
        self.started = time.perf_counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict[str, object] = {
            "phase": "initializing",
            "iteration": int(session_start_iteration),
            "metrics": None,
            "latest_checkpoint": None,
            "latest_evaluation": None,
            "latest_evaluation_status": "not_run",
            "error": None,
            "complete": False,
        }

    def update(self, **values: object) -> None:
        with self._lock:
            self._state.update(values)

    def set_phase(self, phase: str) -> None:
        self.update(phase=str(phase))

    def _rank_payload(self) -> dict[str, object]:
        with self._lock:
            state = dict(self._state)
        return {
            "schema": RANK_HEARTBEAT_SCHEMA,
            "rank": self.rank,
            "pid": __import__("os").getpid(),
            "phase": state["phase"],
            "iteration": int(state["iteration"]),
            "timestamp": _timestamp(),
            "monotonic_seconds": time.perf_counter(),
        }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            state = dict(self._state)
        metrics = state.get("metrics") if isinstance(state.get("metrics"), Mapping) else None
        elapsed = time.perf_counter() - self.started
        iteration = int(state["iteration"])
        session_completed = max(iteration - self.session_start_iteration, 0)
        remaining = max(self.target_iterations - iteration, 0)
        eta = elapsed / session_completed * remaining if session_completed > 0 and remaining > 0 else None
        ranks = []
        now = time.perf_counter()
        for rank in range(self.world_size):
            path = self.run_root / "ranks" / f"rank{rank:03d}" / "heartbeat.json"
            if not path.is_file():
                ranks.append({"rank": rank, "status": "starting", "phase": "unknown", "heartbeat_age_seconds": None})
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                ranks.append({
                    "rank": rank,
                    "status": "alive",
                    "phase": row.get("phase", "unknown"),
                    "iteration": row.get("iteration"),
                    "pid": row.get("pid"),
                    "heartbeat_at": row.get("timestamp"),
                    "heartbeat_age_seconds": max(0.0, now - float(row["monotonic_seconds"])),
                })
            except Exception as error:
                ranks.append({"rank": rank, "status": "unavailable", "phase": "unknown", "error": str(error), "heartbeat_age_seconds": None})
        return {
            "schema": PROGRESS_SCHEMA,
            "mode": self.mode,
            "phase": state["phase"],
            "iteration": iteration,
            "session_start_iteration": self.session_start_iteration,
            "session_completed_iterations": session_completed,
            "target_iterations": self.target_iterations,
            "completed_games": None if metrics is None else metrics.get("statistics/completed_games_total"),
            "env_steps": None if metrics is None else metrics.get("statistics/env_steps_total"),
            "game_steps": None if metrics is None else metrics.get("statistics/game_steps_total"),
            "env_steps_per_second": None if metrics is None else metrics.get("perf/env_steps_per_second"),
            "game_steps_per_second": None if metrics is None else metrics.get("perf/game_steps_per_second"),
            "session_elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "latest_checkpoint": state.get("latest_checkpoint"),
            "latest_evaluation": state.get("latest_evaluation"),
            "latest_evaluation_status": state.get("latest_evaluation_status"),
            "heartbeat_at": _timestamp(),
            "ranks": ranks,
            "gpus": gpu_snapshot() if self.is_primary else [],
            "complete": bool(state.get("complete", False)),
            "error": state.get("error"),
            "remote_verified": False,
        }

    def emit(self) -> None:
        atomic_json(self.rank_root / "heartbeat.json", self._rank_payload())
        if self.is_primary:
            progress = self.snapshot()
            atomic_json(self.run_root / "progress.json", progress)
            print(format_console_summary(progress), flush=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.emit()
            except Exception as error:
                print(f"[M07 heartbeat] unavailable: {type(error).__name__}: {error}", flush=True)
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            return
        self.emit()
        self._thread = threading.Thread(target=self._run, name=f"m07-heartbeat-rank{self.rank}", daemon=True)
        self._thread.start()

    def close(self, *, phase: str, complete: bool, error: str | None = None) -> None:
        self.update(phase=phase, complete=complete, error=error)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 2)
            self._thread = None
        self.emit()


__all__ = ["LiveProgressHeartbeat", "PROGRESS_SCHEMA", "format_console_summary"]
