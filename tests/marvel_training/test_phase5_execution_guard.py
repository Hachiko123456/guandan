"""A failing managed command is recorded once, then blocks every retry."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("phase5_guard_under_test", ROOT / "scripts/run_phase5_guarded.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def no_preflight():
    return {"fixture": "no neural work"}


def test_failure_is_latched_and_different_command_cannot_retry(tmp_path):
    rc, first = guard.run_once(["-c", "raise SystemExit(7)"], state_dir=tmp_path,
                               preflight_check=no_preflight)
    assert rc == 1 and first["actual_exit_code"] == 7
    assert first["automatic_retries"] == 0 and first["attempts"] == 1
    marker = tmp_path / "must_not_exist"
    rc, second = guard.run_once(["-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                                state_dir=tmp_path, preflight_check=no_preflight)
    assert rc == 3 and second["status"] == "BLOCKED_AWAITING_USER"
    assert not marker.exists()
    assert second["failure_id"] == first["failure_id"]


def test_success_preserves_full_output_and_real_exit_status(tmp_path):
    rc, result = guard.run_once(["-c", "print('complete stdout'); import sys; print('stderr', file=sys.stderr)"],
                               state_dir=tmp_path, preflight_check=no_preflight)
    assert rc == 0 and result["actual_exit_code"] == 0
    text = Path(result["log_path"]).read_text(encoding="utf-8")
    assert "complete stdout" in text and "stderr" in text
    assert not (tmp_path / "failure.json").exists()
    assert not (tmp_path / "active.lock").exists()


def test_preflight_failure_never_executes_child(tmp_path):
    marker = tmp_path / "not_executed"
    def invalid():
        raise SyntaxError("fixture compilation failed")
    rc, result = guard.run_once(["-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                               state_dir=tmp_path, preflight_check=invalid)
    assert rc == 1 and not marker.exists()
    assert "SyntaxError" in result["error"]
    assert (tmp_path / "failure.json").is_file()


def test_timeout_stops_process_and_latches_failure(tmp_path):
    started = time.perf_counter()
    rc, result = guard.run_once(["-c", "import time; time.sleep(20)"], state_dir=tmp_path,
                               timeout_seconds=0.2, preflight_check=no_preflight)
    assert time.perf_counter() - started < 10
    assert rc == 1 and result["failure_kind"] == "timeout"
    assert (tmp_path / "failure.json").exists()
    assert not (tmp_path / "active.lock").exists()


def test_concurrent_run_cannot_launch_second_child(tmp_path):
    (tmp_path / "active.lock").write_text("active", encoding="utf-8")
    marker = tmp_path / "not_executed"
    rc, result = guard.run_once(["-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                               state_dir=tmp_path, preflight_check=no_preflight)
    assert rc == 3 and result["status"] == "BLOCKED_ACTIVE_RUN" and not marker.exists()


def test_acknowledge_requires_matching_failure_and_never_auto_runs(tmp_path):
    _, result = guard.run_once(["-c", "raise SystemExit(2)"], state_dir=tmp_path,
                               preflight_check=no_preflight)
    with pytest.raises(ValueError):
        guard.acknowledge_failure("wrong", state_dir=tmp_path)
    assert (tmp_path / "failure.json").exists()
    receipt = guard.acknowledge_failure(result["failure_id"], state_dir=tmp_path)
    assert not (tmp_path / "failure.json").exists()
    assert receipt["command_executed"] is False and receipt["automatic_retry"] is False
    assert list(tmp_path.glob("*.acknowledged.json"))


@pytest.mark.parametrize("tool", ["Bash", "Edit", "Write", "NotebookEdit"])
def test_implementer_hook_blocks_mutation_tools_while_latched(tmp_path, tool):
    failure = tmp_path / "runs/phase5/execution_guard/failure.json"
    failure.parent.mkdir(parents=True)
    failure.write_text(json.dumps({"failure_id": "fixture-failure"}), encoding="utf-8")
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(tmp_path))
    result = subprocess.run([sys.executable, str(ROOT / ".claude/hooks/validate_phase5_failure_latch.py")],
                            input=json.dumps({"tool_name": tool}), capture_output=True, text=True, env=env)
    assert result.returncode == 2
    assert "fixture-failure" in result.stderr and "STOP" in result.stderr


def test_hook_allows_report_reading_and_ready_workspace(tmp_path):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(tmp_path))
    command = [sys.executable, str(ROOT / ".claude/hooks/validate_phase5_failure_latch.py")]
    for tool in ("Read", "Bash"):
        result = subprocess.run(command, input=json.dumps({"tool_name": tool}),
                                capture_output=True, text=True, env=env)
        assert result.returncode == 0


def test_real_compile_import_and_exports_preflight_is_clean():
    result = guard.preflight()
    assert result["compiled_files"] >= 7
    assert result["imports_and_exports"] == "PASS"


def test_implementer_frontmatter_registers_real_pretooluse_command():
    text = (ROOT / ".claude/agents/phase-implementer.md").read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith("hooks: "))
    hooks = json.loads(line[len("hooks: "):])
    item = hooks["PreToolUse"][0]
    assert item["matcher"] == "Bash|Edit|Write|NotebookEdit"
    hook = item["hooks"][0]
    assert hook["type"] == "command"
    assert "validate_phase5_failure_latch.py" in hook["command"]
    assert "args" not in hook  # command hooks accept a command string, not args


@pytest.mark.parametrize("bad", [
    ["--deals", "0"], ["--deals", "17"], ["--batch-slots", "9"],
    ["--start-index", "-1"], ["--max-wall-seconds", "nan"],
    ["--max-wall-seconds", "181"], ["--device", "cpu"],
    ["--output", str(ROOT / "runs/marvel_v14_phase4/delivery/wrong.json")],
])
def test_pilot_cli_rejects_invalid_or_unsafe_budget_without_starting_collection(bad):
    spec = importlib.util.spec_from_file_location("phase5_pilot_cli", ROOT / "scripts/run_phase5_collection_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    with pytest.raises(SystemExit) as stopped:
        pilot.main(bad)
    assert stopped.value.code == 2
