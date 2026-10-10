# SPDX-License-Identifier: MIT
"""Best-effort local/Kaggle telemetry with explicit unavailable states."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

import psutil
import torch

from .common import append_jsonl


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_snapshot() -> list[dict[str, object]]:
    command = [
        "nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
    except Exception as error:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count:
            return [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "status": "partial",
                    "error": f"{type(error).__name__}: {error}",
                    "memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "memory_reserved_bytes": int(torch.cuda.memory_reserved(index)),
                }
                for index in range(count)
            ]
        return [{"index": None, "status": "unavailable", "error": f"{type(error).__name__}: {error}"}]
    rows = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            continue
        def number(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None
        rows.append({
            "index": int(fields[0]), "name": fields[1], "status": "ok",
            "utilization_percent": number(fields[2]),
            "memory_used_mib": number(fields[3]), "memory_total_mib": number(fields[4]),
            "power_watts": number(fields[5]),
        })
    return rows or [{"index": None, "status": "unavailable", "error": "nvidia-smi returned no GPU rows"}]


def collect_telemetry(*, phase: str, rank_processes: list[dict[str, object]] | None = None) -> dict[str, object]:
    virtual = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.cwd()))
    return {
        "schema": "guandan-marvel-telemetry-v1",
        "timestamp": utc_timestamp(),
        "phase": phase,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "ram_used_bytes": int(virtual.used),
        "ram_total_bytes": int(virtual.total),
        "disk_used_bytes": int(disk.used),
        "disk_total_bytes": int(disk.total),
        "gpus": gpu_snapshot(),
        "rank_processes": rank_processes or [],
    }


class TelemetryRecorder:
    def __init__(self, path: str | Path, *, interval_seconds: float = 5.0) -> None:
        self.path = Path(path)
        self.interval_seconds = max(float(interval_seconds), 1.0)
        self._phase = "starting"
        self._rank_processes: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_phase(self, phase: str) -> None:
        self._phase = str(phase)

    def set_rank_processes(self, rows: list[dict[str, object]]) -> None:
        self._rank_processes = [dict(row) for row in rows]

    def snapshot(self) -> dict[str, object]:
        return collect_telemetry(phase=self._phase, rank_processes=self._rank_processes)

    def _run(self) -> None:
        while not self._stop.is_set():
            append_jsonl(self.path, self.snapshot())
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="guandan-marvel-telemetry", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 2)
        self._thread = None


def runtime_environment() -> dict[str, object]:
    return {
        "schema": "guandan-marvel-runtime-environment-v1",
        "timestamp": utc_timestamp(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "gpus": gpu_snapshot(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_bytes": int(psutil.virtual_memory().total),
    }


__all__ = ["TelemetryRecorder", "collect_telemetry", "gpu_snapshot", "runtime_environment", "utc_timestamp"]
