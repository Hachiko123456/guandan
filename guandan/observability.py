"""Low-overhead structured observability for training and deployment runs.

The hot path only enqueues small dictionaries. A daemon writer performs JSONL
writes and periodic system sampling so telemetry cannot synchronously stall
rollout or optimization.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid
from typing import Any, Iterator, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, AttributeError):
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def collect_system_telemetry(root: str | Path | None = None) -> dict[str, Any]:
    """Return best-effort CPU/RAM/disk/GPU facts with explicit unavailable values."""
    result: dict[str, Any] = {"timestamp": utc_now(), "cpu": {}, "memory": {}, "disk": {}, "gpus": []}
    try:
        import psutil  # type: ignore
        result["cpu"] = {"status": "ok", "percent": psutil.cpu_percent(interval=None), "count": psutil.cpu_count()}
        memory = psutil.virtual_memory()
        result["memory"] = {"status": "ok", "percent": memory.percent, "used_bytes": memory.used,
                            "available_bytes": memory.available, "total_bytes": memory.total}
    except Exception as exc:
        result["cpu"] = _unavailable(f"psutil: {type(exc).__name__}: {exc}")
        result["memory"] = _unavailable(f"psutil: {type(exc).__name__}: {exc}")
    try:
        usage = shutil.disk_usage(Path(root) if root is not None else Path.cwd())
        result["disk"] = {"status": "ok", "used_bytes": usage.used, "free_bytes": usage.free,
                           "total_bytes": usage.total, "percent": usage.used / usage.total * 100}
    except Exception as exc:
        result["disk"] = _unavailable(f"disk: {type(exc).__name__}: {exc}")
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                result["gpus"].append({"index": index, "status": "ok",
                    "name": pynvml.nvmlDeviceGetName(handle).decode(errors="replace"),
                    "utilization_percent": util.gpu, "memory_utilization_percent": util.memory,
                    "memory_used_bytes": memory.used, "memory_free_bytes": memory.free,
                    "memory_total_bytes": memory.total})
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        result["gpus"] = [_unavailable(f"NVML: {type(exc).__name__}: {exc}")]
    return result


class RunObservability:
    """Asynchronous JSONL event/metric writer and live progress publisher."""
    def __init__(self, root: str | Path, metadata: Mapping[str, Any], *, telemetry_interval: float = 10.0):
        self.root = Path(root).resolve()
        self.live = self.root / "live"
        self.live.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._closed = False
        self._dropped = 0
        self._started = time.monotonic()
        self._telemetry_interval = max(0.1, float(telemetry_interval))
        base = dict(metadata)
        base.setdefault("run_id", uuid.uuid4().hex)
        base.setdefault("started_at", utc_now())
        base.setdefault("observability_schema", "guandan-observability-v1")
        self.metadata = base
        atomic_json(self.root / "run.json", base)
        atomic_json(self.live / "progress.json", {**base, "status": "starting", "updated_at": utc_now()})
        self._writer = threading.Thread(target=self._writer_loop, name="guandan-observability", daemon=True)
        self._sampler = threading.Thread(target=self._telemetry_loop, name="guandan-telemetry", daemon=True)
        self._writer.start(); self._sampler.start()
        # Capture a baseline even when a short smoke/plan run finishes before
        # the first periodic telemetry interval.
        self.system_metric()

    @property
    def dropped_records(self) -> int:
        return self._dropped

    def _enqueue(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._closed:
            return
        record = {**self.metadata, **dict(payload), "timestamp": utc_now(),
                  "elapsed_seconds": time.monotonic() - self._started}
        try:
            self._queue.put_nowait((kind, record))
        except queue.Full:
            self._dropped += 1

    def event(self, name: str, **fields: Any) -> None:
        self._enqueue("events.jsonl", {"event": name, **fields})

    def metric(self, **fields: Any) -> None:
        self._enqueue("metrics.jsonl", fields)

    def system_metric(self, payload: Mapping[str, Any] | None = None) -> None:
        self._enqueue("system_metrics.jsonl", dict(payload or collect_system_telemetry(self.root)))

    def progress(self, **fields: Any) -> None:
        # Progress is also queued: no atomic file write occurs on the caller thread.
        self._enqueue("__progress__.json", {**fields, "updated_at": utc_now()})

    @contextmanager
    def phase(self, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
        started = time.monotonic()
        self.event("phase_started", phase=name, **fields)
        data: dict[str, Any] = {"phase": name, "started_at": started}
        try:
            yield data
        except BaseException as exc:
            elapsed = time.monotonic() - started
            self.event("phase_failed", phase=name, duration_seconds=elapsed,
                       error_type=type(exc).__name__, error=str(exc))
            raise
        else:
            elapsed = time.monotonic() - started
            self.event("phase_completed", phase=name, duration_seconds=elapsed, **fields)
            self.metric(phase=name, duration_seconds=elapsed, **fields)

    def flush(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

    def close(self) -> None:
        if self._closed:
            return
        self.event("observability_closed", dropped_records=self._dropped)
        self._stop.set()
        self._queue.put(None)
        self._writer.join(timeout=10)
        self._sampler.join(timeout=2)
        self.flush(1)
        self._closed = True

    def _writer_loop(self) -> None:
        streams: dict[str, Any] = {}
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                filename, record = item
                if filename == "__progress__.json":
                    atomic_json(self.live / "progress.json", record)
                    continue
                stream = streams.get(filename)
                if stream is None:
                    path = self.root / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    stream = path.open("a", encoding="utf-8", buffering=1)
                    streams[filename] = stream
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            for stream in streams.values():
                stream.close()

    def _telemetry_loop(self) -> None:
        while not self._stop.wait(self._telemetry_interval):
            self.system_metric()

    def __enter__(self) -> "RunObservability":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
