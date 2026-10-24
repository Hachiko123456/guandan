"""Offline durability, recovery, concurrency and redaction tests for EventLog."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from tools.kaggle_pipeline.backend import BackendError
from tools.kaggle_pipeline.event_log import EventLog


TOKEN = "KGAT_synthetic-never-write"
BEARER_SECRET = "bearer-value-never-write"
START = datetime(2026, 10, 24, 4, 5, 6, 789000, tzinfo=timezone.utc)


class FixedClock:
    def __init__(self, value=START):
        self.value = value

    def __call__(self):
        result = self.value
        self.value += timedelta(milliseconds=1)
        return result


def records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exact_event_and_console_files_are_durable_and_utc(tmp_path, capsys):
    log = EventLog(tmp_path, "run-1", clock=FixedClock())
    event = log.event("dataset", "status_changed", "ready", 2, 1.25,
                      {"dataset_ref": "owner/name", "version": 1})
    line = log.console("DATASET_READY", "version=1", {"elapsed": "1s"})

    assert event == {
        "schema": "guandan-kaggle-event-v1",
        "sequence": 1,
        "time": "2026-10-24T04:05:06.789+00:00",
        "run_id": "run-1",
        "stage": "dataset",
        "event": "status_changed",
        "status": "ready",
        "attempt": 2,
        "elapsed_seconds": 1.25,
        "details": {"dataset_ref": "owner/name", "version": 1},
    }
    assert records(tmp_path / "events.jsonl") == [event]
    assert line == '[04:05:06] DATASET_READY version=1 {"elapsed":"1s"}'
    assert (tmp_path / "console.log").read_text(encoding="utf-8") == line + "\n"
    assert capsys.readouterr().out == line + "\n"
    assert set(path.name for path in tmp_path.iterdir()) == {"events.jsonl", "console.log"}


def test_reopen_continues_sequence_and_validates_every_existing_line(tmp_path):
    EventLog(tmp_path, "same-run", clock=FixedClock()).event("prepare", "started")
    reopened = EventLog(tmp_path, "same-run", clock=FixedClock(START + timedelta(seconds=1)))
    assert reopened.event("prepare", "finished")["sequence"] == 2

    path = tmp_path / "events.jsonl"
    content = path.read_text(encoding="utf-8")
    path.write_text(content.replace('"sequence":2', '"sequence":7'), encoding="utf-8")
    with pytest.raises(ValueError, match="non-sequential event at line 2"):
        EventLog(tmp_path, "same-run")


@pytest.mark.parametrize("bad_line", [
    "not-json\n",
    '{"schema":"wrong","sequence":1,"run_id":"run","stage":"s","event":"e","time":"2026-01-01T00:00:00+00:00"}\n',
    '{"schema":"guandan-kaggle-event-v1","sequence":1,"run_id":"other","stage":"s","event":"e","time":"2026-01-01T00:00:00+00:00"}\n',
    '{"schema":"guandan-kaggle-event-v1","sequence":1,"run_id":"run","stage":"s","event":"e","time":"2026-01-01T00:00:00"}\n',
    "\n",
])
def test_reopen_rejects_malformed_schema_identity_or_time(tmp_path, bad_line):
    (tmp_path / "events.jsonl").write_text(bad_line, encoding="utf-8")
    with pytest.raises(ValueError):
        EventLog(tmp_path, "run")


def test_two_instances_serialize_concurrent_appends(tmp_path):
    first = EventLog(tmp_path, "parallel")
    second = EventLog(tmp_path, "parallel")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda pair: pair[1].event("watch", "heartbeat", details={"n": pair[0]}),
                                    [(index, first if index % 2 else second) for index in range(40)]))

    persisted = records(tmp_path / "events.jsonl")
    assert sorted(record["sequence"] for record in results) == list(range(1, 41))
    assert [record["sequence"] for record in persisted] == list(range(1, 41))
    assert sorted(record["details"]["n"] for record in persisted) == list(range(40))


def test_file_lock_serializes_two_processes(tmp_path):
    script = """
from pathlib import Path
import sys
from tools.kaggle_pipeline.event_log import EventLog
log = EventLog(Path(sys.argv[1]), "process-run")
for index in range(20):
    log.event("watch", "heartbeat", details={"worker": sys.argv[2], "n": index})
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), worker],
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for worker in ("a", "b")
    ]
    outputs = [process.communicate(timeout=30) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], outputs
    persisted = records(tmp_path / "events.jsonl")
    assert [record["sequence"] for record in persisted] == list(range(1, 41))
    assert {record["details"]["worker"] for record in persisted} == {"a", "b"}


class HostileError(RuntimeError):
    def __str__(self):
        raise AssertionError("EventLog must not stringify arbitrary exceptions")


class HostileObject:
    def __repr__(self):
        raise AssertionError("EventLog must not repr arbitrary objects")


def test_recursive_sanitization_prevents_secrets_and_untrusted_objects(tmp_path, capsys):
    log = EventLog(tmp_path, "safe", clock=FixedClock())
    injected = {
        "api_TOKEN": TOKEN,
        f"KGAT_password": "plain-sensitive-value",
        "nested": [f"prefix {TOKEN} suffix", {"Authorization": f"Bearer {BEARER_SECRET}"}],
        "message": f"Bearer {BEARER_SECRET}",
        "path": Path(f"folder/{TOKEN}/file"),
        "object": HostileObject(),
        "traceback": f"raw context {TOKEN}",
    }
    event = log.event("upload", "returned", details=injected)
    log.console("FAILED", injected, injected)
    failure = log.failure("upload", f"push-{TOKEN}", HostileError(TOKEN), "outcome_unknown",
                          True, next_command=f"resume --header 'Bearer {BEARER_SECRET}'")

    assert event["details"]["api_TOKEN"] == "[REDACTED]"
    assert event["details"]["object"] == "HostileObject"
    assert failure["exception_type"] == "HostileError"
    combined = "".join(path.read_text(encoding="utf-8") for path in tmp_path.iterdir())
    combined += capsys.readouterr().out
    assert TOKEN not in combined
    assert BEARER_SECRET not in combined
    assert "plain-sensitive-value" not in combined
    assert "[REDACTED]" in combined


def test_failure_has_safe_fields_backend_status_and_corresponding_event(tmp_path):
    log = EventLog(tmp_path, "failed-run", clock=FixedClock())
    failure = log.failure("dataset_upload", "create_dataset", BackendError("create_dataset", 503),
                          "definite_failure", False, "rerun --watch")

    assert failure == {
        "schema": "guandan-kaggle-failure-v1",
        "time": "2026-10-24T04:05:06.789+00:00",
        "run_id": "failed-run",
        "stage": "dataset_upload",
        "operation": "create_dataset",
        "exception_type": "BackendError",
        "status_code": 503,
        "outcome": "definite_failure",
        "safe_message": "create_dataset failed",
        "recoverable": False,
        "next_command": "rerun --watch",
        "event_sequence": 1,
    }
    assert json.loads((tmp_path / "failure.json").read_text(encoding="utf-8")) == failure
    event = records(tmp_path / "events.jsonl")[0]
    assert event["sequence"] == failure["event_sequence"]
    assert event["event"] == "failure"
    assert event["status"] == "definite_failure"
    assert event["details"]["status_code"] == 503


def test_flush_and_fsync_are_used_and_nonfinite_values_never_reach_json(tmp_path, monkeypatch):
    from tools.kaggle_pipeline import event_log as module

    real_fsync = module.os.fsync
    synced = []

    def recording_fsync(fd):
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", recording_fsync)
    log = EventLog(tmp_path, "durable", clock=FixedClock())
    event = log.event("watch", "heartbeat", elapsed_seconds=math.inf,
                      details={"nan": math.nan, "negative": -math.inf})
    log.console("RUNNING", details={"value": math.nan})
    log.failure("watch", "poll", RuntimeError(TOKEN), "definite_failure", True)

    assert event["elapsed_seconds"] == "[NONFINITE]"
    assert event["details"] == {"nan": "[NONFINITE]", "negative": "[NONFINITE]"}
    assert len(synced) == 4  # event, console, failure event, atomic failure JSON
    for path in (tmp_path / "events.jsonl", tmp_path / "console.log", tmp_path / "failure.json"):
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text and TOKEN not in text


@pytest.mark.parametrize("outcome", ["maybe", "", None])
def test_failure_rejects_unknown_outcomes_without_writing_event(tmp_path, outcome):
    log = EventLog(tmp_path, "run")
    with pytest.raises(ValueError, match="invalid failure outcome"):
        log.failure("stage", "operation", RuntimeError("ignored"), outcome, True)
    assert (tmp_path / "events.jsonl").read_bytes() == b""
