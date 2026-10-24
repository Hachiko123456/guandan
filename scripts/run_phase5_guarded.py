# SPDX-License-Identifier: MIT
"""Run one bounded phase-5 command; latch any unexpected failure, never retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "runs" / "phase5" / "execution_guard"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def preflight(root: Path = ROOT) -> dict:
    sources = sorted((root / "guandan_marvel" / "phase5").glob("*.py"))
    if not sources:
        raise RuntimeError("no phase-5 sources found")
    for path in sources:
        compile(path.read_bytes(), str(path), "exec")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    package = importlib.import_module("guandan_marvel.phase5")
    missing = [name for name in package.__all__ if not hasattr(package, name)]
    if missing:
        raise RuntimeError(f"invalid phase5 public exports: {missing}")
    importlib.import_module("guandan_marvel.phase5.belief_collect")
    return {"compiled_files": len(sources), "imports_and_exports": "PASS"}


def stop_child_tree(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if child.poll() is None:
        child.kill()
    child.wait()


def run_once(arguments: list[str], *, state_dir: Path = DEFAULT_STATE,
             timeout_seconds: float = 120, label: str = "phase5",
             preflight_check=preflight,
             python_executable: str = sys.executable) -> tuple[int, dict]:
    state_dir = Path(state_dir)
    latch = state_dir / "failure.json"
    if latch.exists():
        prior = json.loads(latch.read_text(encoding="utf-8"))
        return 3, {"status": "BLOCKED_AWAITING_USER", "failure_id": prior["failure_id"],
                   "previous_failure": str(latch), "command_executed": False}
    if not arguments or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("a command and finite positive timeout are required")
    identifier = uuid.uuid4().hex
    started = time.perf_counter()
    record = {"run_id": identifier, "label": label, "command": [python_executable, *arguments],
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": timeout_seconds, "expected_exit": 0,
              "attempts": 1, "automatic_retries": 0}
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = state_dir / "active.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return 3, {"status": "BLOCKED_ACTIVE_RUN", "command_executed": False,
                   "lock": str(lock)}
    os.close(descriptor)
    # Recheck under the exclusive lock before starting the child.
    if latch.exists():
        prior = json.loads(latch.read_text(encoding="utf-8"))
        lock.unlink()
        return 3, {"status": "BLOCKED_AWAITING_USER", "failure_id": prior["failure_id"],
                   "command_executed": False}
    child = None
    try:
        record["preflight"] = preflight_check()
        logfile = state_dir / f"{identifier}.log"
        record["log_path"] = str(logfile.resolve())
        with logfile.open("w", encoding="utf-8") as stream:
            stream.write("COMMAND: " + subprocess.list2cmdline(record["command"]) + "\n")
            stream.flush()
            child = subprocess.Popen(
                record["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            try:
                actual_exit = child.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                stop_child_tree(child)
                record.update(status="FAILED_AWAITING_USER", failure_kind="timeout",
                              actual_exit_code=child.returncode)
            else:
                record["actual_exit_code"] = actual_exit
                record["status"] = "PASS" if actual_exit == 0 else "FAILED_AWAITING_USER"
                if actual_exit != 0:
                    record["failure_kind"] = "unexpected_exit"
    except (Exception, KeyboardInterrupt) as error:
        if child is not None:
            stop_child_tree(child)
        record.update(status="FAILED_AWAITING_USER", failure_kind="preflight_or_runner_error",
                      error=f"{type(error).__name__}: {error}")
    try:
        record["wall_seconds"] = time.perf_counter() - started
        if record["status"] != "PASS":
            record["failure_id"] = identifier
            write_json(latch, record)
        write_json(state_dir / f"{identifier}.json", record)
        return (0 if record["status"] == "PASS" else 1), record
    finally:
        lock.unlink(missing_ok=True)


def acknowledge_failure(failure_id: str, *, state_dir: Path = DEFAULT_STATE) -> dict:
    """Coordinator-only after a direct user confirmation; not an auto-retry API."""
    latch = Path(state_dir) / "failure.json"
    record = json.loads(latch.read_text(encoding="utf-8"))
    if record["failure_id"] != failure_id:
        raise ValueError("failure id does not match; failure remains latched")
    if (Path(state_dir) / "active.lock").exists():
        raise RuntimeError("cannot acknowledge while a command is active")
    record["acknowledged_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(Path(state_dir) / f"{failure_id}.acknowledged.json", record)
    latch.unlink()
    return {"status": "ACKNOWLEDGED", "failure_id": failure_id,
            "command_executed": False, "automatic_retry": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    run = commands.add_parser("run")
    run.add_argument("--label", required=True)
    run.add_argument("--timeout-seconds", type=float, default=120)
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    commands.add_parser("status")
    ack = commands.add_parser("acknowledge")
    ack.add_argument("--failure-id", required=True)
    args = parser.parse_args(argv)
    if args.operation == "status":
        latch = DEFAULT_STATE / "failure.json"
        result = json.loads(latch.read_text(encoding="utf-8")) if latch.exists() else {"status": "READY"}
        code = 3 if latch.exists() else 0
    elif args.operation == "acknowledge":
        result = acknowledge_failure(args.failure_id)
        code = 0
    else:
        required_python = (ROOT / ".venv-gpu" / "Scripts" / "python.exe").resolve()
        if Path(sys.executable).resolve() != required_python:
            parser.error(f"use the GPU environment: {required_python}")
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        code, result = run_once(arguments, timeout_seconds=args.timeout_seconds,
                                label=args.label)
    print(json.dumps(result, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
