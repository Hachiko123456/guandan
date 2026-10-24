"""Offline public entry-point checks; never bind a historical delivery."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.kaggle_pipeline import core


@pytest.fixture
def cli_config(tmp_path):
    config = json.loads(core.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["account_file"] = str(tmp_path / "account.json")
    config["output_root"] = str(tmp_path / "runs")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, config


def invoke(config_path: Path, *args: str):
    # Forbid even importing kaggle (its package has authentication side effects).
    # Other core imports and actual CLI argument parsing still run normally.
    script = """
import builtins, socket, sys
sys.path.insert(0, sys.argv.pop(1))
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'kaggle' or name.startswith('kaggle.'):
        raise AssertionError('offline command imported Kaggle')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
def forbidden(*args, **kwargs):
    raise AssertionError('offline command attempted network')
socket.create_connection = forbidden
socket.socket.connect = forbidden
from scripts.kaggle_pipeline import main
raise SystemExit(main(sys.argv[1:]))
"""
    return subprocess.run([sys.executable, "-c", script, str(core.ROOT), "--config", str(config_path), *args],
                          capture_output=True, text=True, encoding="utf-8", timeout=30)


def test_doctor_missing_account_is_actionable_and_offline(cli_config):
    path, values = cli_config
    result = invoke(path, "doctor")
    assert result.returncode == 2, result.stderr
    data = json.loads(result.stdout)
    assert data["status"] == "ACCOUNT_REQUIRED"
    assert data["account_file"] == values["account_file"]
    assert data["default_delivery"] is None and data["remote_calls"] is False
    assert not Path(values["output_root"]).exists()


def test_dry_run_without_account_or_delivery_cannot_start_job(cli_config):
    path, values = cli_config
    result = invoke(path, "run", "--dry-run")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["status"] == "DRY_RUN" and data["delivery"] is None
    assert data["remote_calls"] is False
    assert not Path(values["output_root"]).exists()


def test_run_does_not_select_history_or_create_state(cli_config):
    path, values = cli_config
    result = invoke(path, "run", "--no-wait")
    assert result.returncode == 1
    assert "No prepared new delivery" in result.stdout
    assert not Path(values["output_root"]).exists()


def test_doctor_filled_account_never_prints_key(cli_config):
    path, values = cli_config
    secret = "synthetic-credential-must-never-be-printed"
    Path(values["account_file"]).write_text(json.dumps({"username": "testowner", "key": secret}), encoding="utf-8")
    result = invoke(path, "doctor")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "READY"
    assert secret not in result.stdout + result.stderr


def test_malformed_account_not_echoed(cli_config):
    path, values = cli_config
    secret = "synthetic-malformed-secret"
    Path(values["account_file"]).write_text('{"key":"' + secret, encoding="utf-8")
    result = invoke(path, "doctor")
    assert result.returncode == 2
    assert secret not in result.stdout + result.stderr


def test_build_dry_run_never_invokes_builder(cli_config):
    path, values = cli_config
    # A nonexistent builder input must still be safe in dry-run.
    values["build"] = {"output": "nonexistent", "model_dir": "nonexistent", "base_inputs": "nonexistent"}
    path.write_text(json.dumps(values), encoding="utf-8")
    result = invoke(path, "prepare", "--build", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "DRY_RUN"
    assert not Path(values["output_root"]).exists()


def test_build_uses_training_python_not_upload_environment(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    python = repository / ".venv-gpu" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(core, "ROOT", repository)
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(core.subprocess, "run", fake_run)
    config = {"output_root": str(tmp_path / "runs"), "build": {
        "output": "new-delivery", "model_dir": "models", "base_inputs": "base",
    }}
    assert core.build_delivery(config) == repository / "new-delivery"
    command = calls[0][0]
    assert command[0] == str(python)
    assert command[1].endswith("build_t4x2_serial_compare.py")
    assert "--model-dir" in command and "--base-inputs" in command


def test_generic_upload_wrapper_has_no_delivery_version_and_safe_modes():
    text = (core.ROOT / "scripts/upload_kaggle.ps1").read_text(encoding="utf-8")
    assert "delivery_v12" not in text and "delivery_v13" not in text
    for switch in ("$DryRun", "$NoWait", "$Watch", "$Status", "$Prune", "$ExecutePrune"):
        assert switch in text
    assert "run -Current" in text
