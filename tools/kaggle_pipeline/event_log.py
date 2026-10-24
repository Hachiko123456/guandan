"""Durable, append-only and credential-safe logging for Kaggle pipeline runs."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
from typing import Any, Callable, Iterator, Mapping


EVENT_SCHEMA = "guandan-kaggle-event-v1"
FAILURE_SCHEMA = "guandan-kaggle-failure-v1"
_REDACTED = "[REDACTED]"
_NONFINITE = "[NONFINITE]"
_SENSITIVE_KEY = re.compile(
    r"token|key|authorization|cookie|secret|password|traceback|exc_info|stack|context",
    re.IGNORECASE,
)
_KGAT = re.compile(r"KGAT_\S+", re.IGNORECASE)
_BEARER = re.compile(r"\bBearer[ \t]+\S+", re.IGNORECASE)

_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.resolve()))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _file_lock(stream) -> Iterator[None]:
    """Lock the first byte without adding a lock artifact to the run directory."""
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _safe_text(value: str) -> str:
    return _BEARER.sub(_REDACTED, _KGAT.sub(_REDACTED, value))


def _safe_key(value: Any) -> tuple[str, bool]:
    if isinstance(value, str):
        raw = value
    elif isinstance(value, Path):
        raw = str(value)
    elif isinstance(value, float) and not math.isfinite(value):
        raw = _NONFINITE
    elif value is None or isinstance(value, (bool, int, float)):
        raw = str(value)
    else:
        raw = type(value).__name__
    return _safe_text(raw), bool(_SENSITIVE_KEY.search(raw))


def sanitize(value: Any) -> Any:
    """Return JSON-safe data without invoking arbitrary object representations."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _NONFINITE
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, Path):
        return _safe_text(str(value))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            safe_key, sensitive = _safe_key(key)
            result[safe_key] = _REDACTED if sensitive else sanitize(item)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize(item) for item in value]
    return _safe_text(type(value).__name__)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


class EventLog:
    """Per-run durable logs with strict sequence recovery and safe serialization."""

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        run_id: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(run_id, str) or not run_id or _safe_text(run_id) != run_id:
            raise ValueError("run_id must be a non-empty, non-sensitive string")
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.events_path = self.run_dir / "events.jsonl"
        self.console_path = self.run_dir / "console.log"
        self.failure_path = self.run_dir / "failure.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = _thread_lock(self.events_path)

        # Opening here both establishes the two stable log paths and rejects a
        # damaged journal before any new output can be produced.
        with self._locked_events() as stream:
            self._validate_events(stream)
        self.console_path.open("ab").close()

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return datetime")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _timestamp(self) -> str:
        return self._now().isoformat(timespec="milliseconds")

    @contextmanager
    def _locked_events(self):
        with self._lock:
            with self.events_path.open("a+b") as stream:
                with _file_lock(stream):
                    yield stream

    def _validate_events(self, stream) -> int:
        stream.seek(0)
        raw = stream.read()
        if not raw:
            return 0
        expected = 1
        for line_number, raw_line in enumerate(raw.splitlines(), start=1):
            if not raw_line.strip():
                raise ValueError(f"malformed events.jsonl line {line_number}")
            try:
                record = json.loads(raw_line.decode("utf-8"), parse_constant=_reject_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"malformed events.jsonl line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"invalid event object at line {line_number}")
            if record.get("schema") != EVENT_SCHEMA:
                raise ValueError(f"invalid event schema at line {line_number}")
            if record.get("run_id") != self.run_id:
                raise ValueError(f"event run_id mismatch at line {line_number}")
            if type(record.get("sequence")) is not int or record["sequence"] != expected:
                raise ValueError(f"non-sequential event at line {line_number}")
            if not isinstance(record.get("stage"), str) or not record["stage"]:
                raise ValueError(f"invalid event stage at line {line_number}")
            if not isinstance(record.get("event"), str) or not record["event"]:
                raise ValueError(f"invalid event name at line {line_number}")
            timestamp = record.get("time")
            try:
                parsed = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else None
            except ValueError as exc:
                raise ValueError(f"invalid event time at line {line_number}") from exc
            if parsed is None or parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
                raise ValueError(f"invalid event time at line {line_number}")
            expected += 1
        return expected - 1

    def _append_event_locked(
        self,
        stream,
        *,
        stage: Any,
        event: Any,
        status: Any = None,
        attempt: Any = None,
        elapsed_seconds: Any = None,
        details: Any = None,
    ) -> dict[str, Any]:
        last_sequence = self._validate_events(stream)
        record: dict[str, Any] = {
            "schema": EVENT_SCHEMA,
            "sequence": last_sequence + 1,
            "time": self._timestamp(),
            "run_id": self.run_id,
            "stage": sanitize(stage),
            "event": sanitize(event),
        }
        for key, value in (("status", status), ("attempt", attempt),
                           ("elapsed_seconds", elapsed_seconds), ("details", details)):
            if value is not None:
                record[key] = sanitize(value)
        if not isinstance(record["stage"], str) or not record["stage"]:
            raise ValueError("stage must be a non-empty string")
        if not isinstance(record["event"], str) or not record["event"]:
            raise ValueError("event must be a non-empty string")

        stream.seek(0, os.SEEK_END)
        if stream.tell() and not self._ends_with_newline(stream):
            stream.write(b"\n")
        stream.write(_json_bytes(record))
        stream.flush()
        os.fsync(stream.fileno())
        return record

    @staticmethod
    def _ends_with_newline(stream) -> bool:
        position = stream.tell()
        stream.seek(-1, os.SEEK_END)
        result = stream.read(1) == b"\n"
        stream.seek(position)
        return result

    def event(
        self,
        stage: str,
        event: str,
        status: Any = None,
        attempt: Any = None,
        elapsed_seconds: Any = None,
        details: Any = None,
    ) -> dict[str, Any]:
        with self._locked_events() as stream:
            return self._append_event_locked(
                stream,
                stage=stage,
                event=event,
                status=status,
                attempt=attempt,
                elapsed_seconds=elapsed_seconds,
                details=details,
            )

    @staticmethod
    def _exception_fields(error: BaseException) -> tuple[str, int | None]:
        exception_type = _safe_text(type(error).__name__)
        status_code = None
        try:
            from .backend import BackendError

            if isinstance(error, BackendError):
                status_code = error.status_code
        except ImportError:
            pass
        return exception_type, status_code

    def failure(
        self,
        stage: str,
        operation: str,
        error: BaseException,
        outcome: str,
        recoverable: bool,
        next_command: Any = None,
    ) -> dict[str, Any]:
        if outcome not in {"definite_failure", "outcome_unknown", "reconciled"}:
            raise ValueError("invalid failure outcome")
        if type(recoverable) is not bool:
            raise TypeError("recoverable must be bool")
        safe_operation = sanitize(operation)
        if not isinstance(safe_operation, str) or not safe_operation:
            raise ValueError("operation must be a non-empty string")
        exception_type, status_code = self._exception_fields(error)
        safe_message = f"{safe_operation} failed"
        details = {
            "operation": safe_operation,
            "exception_type": exception_type,
            "status_code": status_code,
            "outcome": outcome,
            "safe_message": safe_message,
            "recoverable": recoverable,
            "next_command": next_command,
        }

        with self._locked_events() as stream:
            event_record = self._append_event_locked(
                stream, stage=stage, event="failure", status=outcome, details=details
            )
            failure = {
                "schema": FAILURE_SCHEMA,
                "time": event_record["time"],
                "run_id": self.run_id,
                "stage": event_record["stage"],
                "operation": safe_operation,
                "exception_type": exception_type,
                "status_code": status_code,
                "outcome": outcome,
                "safe_message": safe_message,
                "recoverable": recoverable,
                "next_command": sanitize(next_command),
                "event_sequence": event_record["sequence"],
            }
            self._atomic_json(self.failure_path, failure)
        return failure

    def _atomic_json(self, path: Path, value: Mapping[str, Any]) -> None:
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.run_dir, prefix=".failure-", suffix=".tmp", delete=False
            ) as stream:
                temporary_name = stream.name
                stream.write(_json_bytes(value))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass

    def console(self, status: Any, message: Any = None, details: Any = None) -> str:
        timestamp = self._now()
        safe_status = sanitize(status)
        if not isinstance(safe_status, str) or not safe_status:
            raise ValueError("status must be a non-empty string")
        safe_status = safe_status.replace("\r", "\\r").replace("\n", "\\n")
        line = f"[{timestamp:%H:%M:%S}] {safe_status}"
        if message is not None:
            safe_message = sanitize(message)
            if isinstance(safe_message, str):
                rendered = safe_message.replace("\r", "\\r").replace("\n", "\\n")
            else:
                rendered = json.dumps(safe_message, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False)
            line += f" {rendered}"
        if details is not None:
            line += " " + json.dumps(sanitize(details), ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False)

        encoded = (line + "\n").encode("utf-8")
        with self._locked_events():
            with self.console_path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        return line


__all__ = ["EVENT_SCHEMA", "FAILURE_SCHEMA", "EventLog", "sanitize"]
