"""Bounded parent diagnostics without importing a model or starting training."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from serial_compare import delivery, pipeline  # noqa: E402


def _actions(tmp_path, runner=None):
    return pipeline.ProductionActions(
        plan={}, root=tmp_path, dataset=tmp_path, source=tmp_path, runner=runner,
    )


def _failing_runner(text):
    def run(command, **kwargs):
        kwargs["stdout"].write(text)
        return subprocess.CompletedProcess(command, 1)

    return run


def test_first_rank_error_survives_long_elastic_tail_and_full_log_is_preserved(tmp_path):
    original = (
        "startup banner\n" + "setup progress\n" * 100
        + "[rank1]: Traceback (most recent call last):\n"
        + '[rank1]:   File "worker.py", line 8, in main\n'
        + "[rank1]: RuntimeError: original NCCL initialization failed\n"
        + "elastic launcher noise\n" * 300
        + "torch.distributed.elastic.multiprocessing.errors.ChildFailedError\n"
    )
    log = tmp_path / "worker.log"
    log.write_text("previous session retained\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as caught:
        _actions(tmp_path, _failing_runner(original))._run(["torchrun"], log=log, timeout=5)
    message = str(caught.value)
    assert str(log.resolve()) in message
    assert "exit 1" in message
    assert "original NCCL initialization failed" in message
    assert "first traceback/exception:" in message
    assert "ChildFailedError" in message
    assert "log head:" in message and "log tail:" in message
    assert log.read_text(encoding="utf-8") == "previous session retained\n[cmd] torchrun\n" + original


def test_early_import_failure_in_real_python_child_is_visible(tmp_path):
    log = tmp_path / "import.log"
    with pytest.raises(RuntimeError) as caught:
        _actions(tmp_path)._run(
            [sys.executable, "-c", "import _serial_compare_missing_startup_module_"],
            log=log, timeout=10,
        )
    assert "ModuleNotFoundError" in str(caught.value)
    assert "_serial_compare_missing_startup_module_" in str(caught.value)
    assert str(log.resolve()) in str(caught.value)


def test_exception_without_traceback_after_startup_noise_is_retained(tmp_path):
    output = "startup\n" * 100 + "ImportError: backend unavailable\n" + "elastic noise\n" * 100
    with pytest.raises(RuntimeError, match="ImportError: backend unavailable"):
        _actions(tmp_path, _failing_runner(output))._run(
            ["worker"], log=tmp_path / "worker.log", timeout=5,
        )


@pytest.mark.parametrize("flag", [["--run-dir", "arm"], ["--run-dir=arm"]])
def test_structured_rank_error_precedes_log_and_omits_unknown_fields(tmp_path, flag):
    cwd = tmp_path / "source"
    errors = cwd / "arm" / "worker_errors"
    errors.mkdir(parents=True)
    rank_file = errors / "rank000.json"
    payload = {
        "rank": 0, "local_rank": 1, "world_size": 2, "stage": "initialize",
        "type": "RuntimeError", "message": "rank CUDA device mismatch",
        "traceback": "Traceback (most recent call last):\nRuntimeError: rank CUDA device mismatch",
        "environment": {"TOKEN": "must-not-appear"},
    }
    # An unrelated run must never be discovered by a workspace-wide scan.
    other = tmp_path / "unrelated" / "worker_errors"
    other.mkdir(parents=True)
    (other / "rank001.json").write_text('{"message": "unrelated-private-error"}', encoding="utf-8")
    log = tmp_path / "worker.log"
    def current_failure(command, **kwargs):
        rank_file.write_text(json.dumps(payload), encoding="utf-8")
        kwargs["stdout"].write("ChildFailedError\n")
        return subprocess.CompletedProcess(command, 1)
    with pytest.raises(RuntimeError) as caught:
        _actions(tmp_path, current_failure)._run(
            ["worker", *flag], log=log, timeout=5, cwd=cwd,
        )
    message = str(caught.value)
    assert str(rank_file.resolve()) in message
    for key in ("rank", "local_rank", "world_size", "stage", "type", "message", "traceback"):
        assert f'"{key}"' in message
    assert message.index("rank CUDA device mismatch") < message.index("log head:")
    assert "must-not-appear" not in message
    assert "environment" not in message
    assert "unrelated-private-error" not in message


def test_retry_only_shows_current_attempt_error_and_ignores_old_sidecar(tmp_path):
    log = tmp_path / "retry.log"
    log.write_text("Traceback (most recent call last):\nRuntimeError: previous attempt failure\n", encoding="utf8")
    errors = tmp_path / "run/worker_errors"
    errors.mkdir(parents=True)
    (errors / "rank000.json").write_text(json.dumps({"message": "obsolete sidecar failure"}), encoding="utf8")
    current = "Traceback (most recent call last):\nImportError: current attempt failure\n" + "elastic noise\n" * 300
    with pytest.raises(RuntimeError) as caught:
        _actions(tmp_path, _failing_runner(current))._run(
            ["worker", "--run-dir", str(errors.parent)], log=log, timeout=5,
        )
    message = str(caught.value)
    assert "current attempt failure" in message
    assert "previous attempt failure" not in message
    assert "obsolete sidecar failure" not in message
    assert "previous attempt failure" in log.read_text(encoding="utf8")


@pytest.mark.parametrize("contents", ["{partial", "[]", "x" * 65537], ids=["partial", "non-object", "oversized"])
def test_bad_rank_sidecar_falls_back_to_original_log(tmp_path, contents):
    errors = tmp_path / "run" / "worker_errors"
    errors.mkdir(parents=True)
    (errors / "rank000.json").write_text(contents, encoding="utf-8")
    with pytest.raises(RuntimeError, match="ImportError: original import failure"):
        _actions(tmp_path, _failing_runner("ImportError: original import failure\n"))._run(
            ["worker", "--run-dir", str(errors.parent)], log=tmp_path / "worker.log", timeout=5,
        )


def test_empty_log_failure_remains_actionable(tmp_path):
    def run(command, **kwargs):
        kwargs["stdout"].seek(0)
        kwargs["stdout"].truncate()
        return subprocess.CompletedProcess(command, 7)

    log = tmp_path / "empty.log"
    with pytest.raises(RuntimeError) as caught:
        _actions(tmp_path, run)._run(["worker"], log=log, timeout=5)
    assert "exit 7" in str(caught.value)
    assert str(log.resolve()) in str(caught.value)
    assert "[empty]" in str(caught.value)
    assert log.read_bytes() == b""


def test_runner_timeout_preserves_original_exception_and_displays_log(tmp_path, capsys):
    error = subprocess.TimeoutExpired(["worker"], 5, output=b"original-output", stderr=b"original-stderr")

    def run(command, **kwargs):
        kwargs["stdout"].write("RuntimeError: original error before timeout\n")
        raise error

    log = tmp_path / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        _actions(tmp_path, run)._run(["worker"], log=log, timeout=5)
    assert caught.value is error
    assert error.cmd == ["worker"] and error.timeout == 5
    assert error.output == b"original-output" and error.stderr == b"original-stderr"
    formatted = "".join(traceback.format_exception(error))
    stderr = capsys.readouterr().err
    for diagnostic in (str(log.resolve()), "original error before timeout", "timed out after 5 seconds"):
        assert diagnostic in formatted
        assert diagnostic in stderr


@pytest.mark.parametrize("platform", ["posix", "nt"])
def test_timeout_still_kills_and_reaps_children(tmp_path, monkeypatch, capsys, platform):
    events = []
    error = subprocess.TimeoutExpired(["worker"], 1)

    class Process:
        pid = 4242
        returncode = -9

        def communicate(self, timeout=None):
            events.append(("communicate", timeout))
            if timeout is not None:
                raise error

        def kill(self):
            events.append("kill")

    def popen(command, **kwargs):
        assert kwargs["start_new_session"] is (platform == "posix")
        kwargs["stdout"].write("ImportError: failure before timeout\n")
        return Process()

    # Keep pathlib's real platform untouched while checking both cleanup paths.
    monkeypatch.setattr(pipeline, "os", SimpleNamespace(
        environ=os.environ, name=platform, getpgid=lambda pid: 77,
        killpg=lambda pg, sig: events.append(("killpg", pg)),
    ))
    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    log = tmp_path / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        _actions(tmp_path)._run(["worker"], log=log, timeout=1)
    assert caught.value is error
    assert events[-2:] == ["kill", ("communicate", None)]
    assert (("killpg", 77) in events) is (platform == "posix")
    assert "failure before timeout" in capsys.readouterr().err


def test_launch_oserror_propagates_and_missing_log_diagnostics_are_safe(tmp_path, monkeypatch):
    error = FileNotFoundError("original executable missing")
    log = tmp_path / "missing.log"

    def popen(*args, **kwargs):
        raise error

    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    with pytest.raises(FileNotFoundError) as caught:
        _actions(tmp_path)._run(["missing-worker"], log=log, timeout=5)
    assert caught.value is error
    assert "original executable missing" in str(error)
    assert str(log.resolve()) in "\n".join(error.__notes__)
    missing = tmp_path / "never-created.log"
    diagnostics = pipeline._failure_diagnostics(missing, ["worker"], tmp_path)
    assert str(missing.resolve()) in diagnostics
    assert "log unavailable" in diagnostics


def test_diagnostics_are_bounded_for_huge_lines_and_rank_traceback(tmp_path):
    errors = tmp_path / "run" / "worker_errors"
    errors.mkdir(parents=True)
    rank_error = json.dumps({
        "rank": 0, "type": "RuntimeError", "message": "structured original error",
        "traceback": "x" * 40000,
    })
    output = "RuntimeError: original error\n" + "x" * 100000 + "\nfinal elastic summary\n"
    log = tmp_path / "large.log"
    def current_failure(command, **kwargs):
        (errors / "rank000.json").write_text(rank_error, encoding="utf-8")
        kwargs["stdout"].write(output)
        return subprocess.CompletedProcess(command, 1)
    with pytest.raises(RuntimeError) as caught:
        _actions(tmp_path, current_failure)._run(
            ["worker", "--run-dir", str(errors.parent)], log=log, timeout=5,
        )
    assert "structured original error" in str(caught.value)
    assert "RuntimeError: original error" in str(caught.value)
    assert "final elastic summary" in str(caught.value)
    assert len(str(caught.value)) < 32000
    assert log.read_text(encoding="utf-8") == "[cmd] worker --run-dir " + str(errors.parent) + "\n" + output


def test_injected_runner_success_keeps_existing_contract(tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("worker completed\n")
        return subprocess.CompletedProcess(command, 0)

    cwd = tmp_path / "source"
    log = tmp_path / "success.log"
    assert _actions(tmp_path, run)._run(["worker", "arg"], log=log, timeout=0.5, cwd=cwd) is None
    command, kwargs = calls[0]
    assert command == ["worker", "arg"]
    assert kwargs["timeout"] == 1.0 and kwargs["cwd"] == str(cwd)
    assert kwargs["stderr"] == subprocess.STDOUT
    assert "PYTHONUNBUFFERED" in kwargs["env"]
    assert log.read_text(encoding="utf-8").endswith("worker completed\n")


def test_bootstrap_failure_serializes_unknown_updates_as_null(tmp_path):
    # Execute the actual generated exception handler only, without bootstrapping
    # a Kaggle runtime or building any delivery bundle.
    module = ast.parse(delivery.BOOTSTRAP)
    handlers = [node for node in ast.walk(module) if isinstance(node, ast.ExceptHandler)]
    handler = next(node for node in handlers if any(
        isinstance(child, ast.Constant) and child.value == "bootstrap_error.json"
        for child in ast.walk(node)
    ))
    error = RuntimeError("failure after progress may have occurred")
    guarded = ast.Try(body=[ast.Raise(exc=ast.Name(id="error", ctx=ast.Load()))], handlers=[handler], orelse=[], finalbody=[])
    code = compile(ast.fix_missing_locations(ast.Module(body=[guarded], type_ignores=[])), "bootstrap-handler", "exec")
    run_root = tmp_path / "run"
    with pytest.raises(RuntimeError) as caught:
        exec(code, {"error": error, "RUN_ROOT": run_root, "Path": Path, "json": json, "traceback": traceback})
    assert caught.value is error
    payload = json.loads((tmp_path / "run_bootstrap" / "bootstrap_error.json").read_text(encoding="utf-8"))
    assert payload["training_updates_this_session"] is None
    assert payload["status"] == "failed"
    assert "failure after progress" in payload["traceback"]
