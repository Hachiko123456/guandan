"""Offline pipeline contracts; all deliveries and remote results are synthetic."""
from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import zipfile

import pytest

from tools.kaggle_pipeline import core, results
from tools.serial_compare.delivery import NOTEBOOK, PAYLOAD


SECRET = "synthetic-test-key-never-snapshot"
OWNER = "offline-tester"
ARCHIVE = "guandan_serial_compare_results.zip"
PHASES = (
    "bootstrap/input_check", "preflight/control", "preflight/candidate",
    "freeze_execution_budget", "train/control", "train/candidate",
    "freeze_final_checkpoints", "evaluate/confirmation_primary",
    "evaluate/development_diagnostics", "summarize_and_package",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def signed(value):
    value = {k: v for k, v in value.items() if k != "hash"}
    return {**value, "hash": digest(encoded(value))}


def write_json(path, value):
    path.write_bytes(encoded(value))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def forbid_network_and_sdk(monkeypatch):
    """Fail before any accidental connection or Kaggle auto-auth import."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline test attempted a real network/SDK operation")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    original_import = builtins.__import__

    def offline_import(name, *args, **kwargs):
        if name.split(".")[0] in {"kaggle", "kagglesdk"}:
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", offline_import)


@pytest.fixture
def sample(tmp_path):
    account_file = tmp_path / "account.json"
    write_json(account_file, {"username": OWNER, "key": SECRET})
    config_file = tmp_path / "config.json"
    write_json(config_file, {
        "schema": "guandan-kaggle-pipeline-config-v1",
        "account_file": str(account_file), "output_root": str(tmp_path / "runs"),
        "experiment": "unit", "accelerator": "NvidiaTeslaT4",
        "timeout_seconds": 600, "poll_seconds": 1, "dataset_wait_seconds": 2,
        "enable_internet": False, "dataset_license": "other", "build": None,
    })
    config = core.load_config(config_file)
    delivery = tmp_path / "new-synthetic-delivery"
    delivery.mkdir()
    plan = {"schema": "synthetic-plan", "budgets": {"total_seconds": 1}}
    plan["plan_hash"] = digest(encoded(plan))
    write_json(delivery / "plan.json", plan)
    (delivery / PAYLOAD).write_bytes(b"tiny synthetic payload, not a historical model")
    write_json(delivery / NOTEBOOK, {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": ["print('synthetic next delivery')\n"]}],
    })
    return config, delivery


def prepare(sample):
    config, delivery = sample
    return core.prepare(config, delivery, verifier=lambda d: {"status": "PASS"})


class LostResponse(RuntimeError):
    pass


class FakeBackend:
    """Persisted flags must precede fake side effects, just as for remote calls."""
    def __init__(self, config, state):
        self.folder = Path(config["output_root"]) / state["run_id"]
        self.dataset = None
        self.kernel = None
        self.calls = []
        self.dataset_state = "ready"
        self.lose_dataset_response = False
        self.lose_push_response = False
        self.remote_status = "complete"
        self.bundle = None
        self.state = state

    def count(self, operation):
        return sum(call[0] == operation for call in self.calls)

    def dataset_info(self, ref):
        self.calls.append(("dataset_info", ref))
        return None if self.dataset is None else dict(self.dataset)

    def create_dataset(self, folder):
        self.calls.append(("create_dataset", folder))
        persisted = read_json(self.folder / "run.json")
        assert persisted["dataset_attempted"] is True
        assert persisted["kernel_attempted"] is False
        assert persisted["status"] == "DATASET_UPLOADING"
        assert {p.name for p in folder.iterdir()} == {PAYLOAD, "dataset-metadata.json"}
        self.dataset = {
            "ready": self.dataset_state == "ready", "status": self.dataset_state,
            "is_private": True, "version": 1,
            "files": [{"name": PAYLOAD, "bytes": (folder / PAYLOAD).stat().st_size}],
        }
        if self.lose_dataset_response:
            raise LostResponse("synthetic lost Dataset response")
        return {"is_private": True}

    def kernel_info(self, ref):
        self.calls.append(("kernel_info", ref))
        return None if self.kernel is None else dict(self.kernel)

    def push_kernel(self, folder, accelerator, timeout_seconds):
        self.calls.append(("push_kernel", folder, accelerator, timeout_seconds))
        persisted = read_json(self.folder / "run.json")
        assert persisted["kernel_attempted"] is True
        assert persisted["status"] == "SUBMITTING"
        metadata = read_json(folder / "kernel-metadata.json")
        assert metadata["is_private"] is True
        assert metadata["enable_internet"] is False
        assert metadata["dataset_sources"] == [persisted["dataset_ref"]]
        assert accelerator == "NvidiaTeslaT4"
        self.kernel = {
            "version": 1, "code_sha256": persisted["notebook_digest"],
            "is_private": True, "dataset_sources": [persisted["dataset_ref"]],
        }
        if self.lose_push_response:
            raise LostResponse("synthetic lost Notebook response")
        return {"version": 1}

    def kernel_status(self, ref, version):
        self.calls.append(("kernel_status", ref, version))
        return self.remote_status

    def download_outputs(self, ref, version, folder):
        self.calls.append(("download_outputs", ref, version))
        if self.bundle is not None:
            shutil.copyfile(self.bundle, folder / ARCHIVE)


def submit(sample, backend, state):
    return core.submit(sample[0], state["run_id"], backend=backend, username=OWNER)


def synthetic_bundle(folder, plan_hash, *, incomplete=False, missing_recovery=False,
                     wrong_plan=False, tamper=False):
    """Sign independently constructed business/recovery records and every ZIP member."""
    folder.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "guandan-serial-compare-result-v1", "plan_hash": "wrong" if wrong_plan else plan_hash,
        "status": "complete", "technical_failure": None,
        "phases": {phase: {"status": "complete"} for phase in PHASES},
        "confirmation": {"complete": True}, "development": {"complete": not incomplete},
        "verdict": {"verdict": "INCONCLUSIVE"},
    }
    members = {"artifacts/RESULT.json": encoded(signed(result))}
    arms = {}
    for arm in ("control", "candidate"):
        relative = f"phases/train_{arm}/run/checkpoints/latest.pt"
        data = ("synthetic recovery " + arm).encode()
        members["artifacts/" + relative] = data
        arms[arm] = {
            "run_dir": f"artifacts/phases/train_{arm}/run",
            "latest": {"checkpoint": "checkpoints/latest.pt", "iteration": 1},
            "files": [{"path": relative, "bytes": len(data), "sha256": digest(data)}],
        }
    if not missing_recovery:
        members["artifacts/recovery_index.json"] = encoded(signed({
            "schema": "guandan-serial-compare-recovery-index-v1", "arms": arms,
        }))
    manifest = signed({
        "schema": "guandan-serial-compare-results-manifest-v1", "mode": "full",
        "evidence_complete": True, "omissions": [],
        "files": [{"path": name, "bytes": len(data), "sha256": digest(data)}
                  for name, data in members.items()],
    })
    members["RESULTS_MANIFEST.json"] = encoded(manifest)
    if tamper:
        # Same length: size checks alone cannot catch this corruption.
        name = "artifacts/phases/train_control/run/checkpoints/latest.pt"
        members[name] = b"X" * len(members[name])
    archive = folder / ARCHIVE
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return archive


def test_prepare_is_offline_without_history_or_secret_snapshots(sample, monkeypatch):
    config, delivery = sample
    def unexpected_backend(*args, **kwargs):
        pytest.fail("prepare must never construct a remote backend")
    monkeypatch.setattr(core, "backend_for", unexpected_backend)
    with pytest.raises(core.PipelineError):
        core.locate(config)
    state = prepare(sample)
    folder = core.locate(config)
    assert folder.name == state["run_id"]
    assert state["status"] == "PREPARED"
    assert state["owner"] is None
    assert state["hashes"] == {
        name: digest((delivery / name).read_bytes()) for name in ("plan.json", PAYLOAD, NOTEBOOK)
    }
    assert {p.name for p in (folder / "dataset").iterdir()} == {PAYLOAD}
    assert {p.name for p in (folder / "kernel").iterdir()} == {NOTEBOOK}
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["sequence"], row["stage"], row["event"]) for row in events] == [
        (1, "prepare", "validated"),
    ]
    assert (folder / "console.log").is_file()
    assert read_json(folder / "delivery.json")["hashes"] == state["hashes"]
    for path in folder.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), str(path)


@pytest.mark.parametrize("explicit_delivery", [False, True])
def test_dry_run_needs_no_account_and_creates_no_staging(sample, explicit_delivery):
    config, delivery = sample
    Path(config["account_file"]).unlink()
    result = core.prepare(config, delivery if explicit_delivery else None,
                          dry_run=True, verifier=lambda d: {"status": "PASS"})
    assert result["status"] == "DRY_RUN"
    assert result["remote_calls"] is False
    assert not Path(config["output_root"]).exists()
    assert SECRET not in json.dumps(result)


def test_repeat_prepare_and_submit_preserve_snapshot_and_submit_once(sample):
    state = prepare(sample)
    backend = FakeBackend(sample[0], state)
    submitted = submit(sample, backend, state)
    before = {p.relative_to(backend.folder): p.read_bytes()
              for p in backend.folder.rglob("*") if p.is_file()}
    repeated = prepare(sample)
    assert repeated["run_id"] == state["run_id"]
    assert repeated["kernel_attempted"] is True
    assert repeated["dataset_attempted"] is True
    assert repeated["kernel_version"] == submitted["kernel_version"] == 1
    assert before == {p.relative_to(backend.folder): p.read_bytes()
                      for p in backend.folder.rglob("*") if p.is_file()}
    assert {p.name for p in (backend.folder / "dataset").iterdir()} == {
        PAYLOAD, "dataset-metadata.json",
    }
    assert {p.name for p in (backend.folder / "kernel").iterdir()} == {
        NOTEBOOK, "kernel-metadata.json",
    }
    assert (backend.folder / "records/dataset/upload.json").is_file()
    assert (backend.folder / "records/dataset/processing.jsonl").is_file()
    assert (backend.folder / "records/kernel/submission.json").is_file()
    event_names = [
        json.loads(line)["event"]
        for line in (backend.folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "intent_persisted" in event_names and "remote_call_returned" in event_names
    submit(sample, backend, repeated)
    assert backend.count("create_dataset") == backend.count("push_kernel") == 1
    assert SECRET not in json.dumps(read_json(backend.folder / "run.json"))
    for path in backend.folder.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes()


@pytest.mark.parametrize("dataset_state", ["pending", "error"])
def test_dataset_not_ready_never_pushes_kernel(sample, monkeypatch, dataset_state):
    state = prepare(sample)
    backend = FakeBackend(sample[0], state)
    backend.dataset_state = dataset_state
    clock = [0]
    monkeypatch.setattr(core.time, "monotonic", lambda: clock[0])
    def advance(seconds):
        clock[0] += seconds
    with pytest.raises(core.PipelineError, match="Dataset"):
        core.submit(sample[0], state["run_id"], backend=backend, username=OWNER, sleep=advance)
    assert backend.count("create_dataset") == 1
    assert backend.count("push_kernel") == 0
    persisted = read_json(backend.folder / "run.json")
    assert persisted["dataset_attempted"] is True
    assert persisted["kernel_attempted"] is False


@pytest.mark.parametrize("operation", ["dataset", "kernel"])
def test_lost_response_reconciles_without_duplicate_mutation(sample, operation):
    state = prepare(sample)
    backend = FakeBackend(sample[0], state)
    setattr(backend, "lose_dataset_response" if operation == "dataset" else "lose_push_response", True)
    with pytest.raises(LostResponse):
        submit(sample, backend, state)
    persisted = read_json(backend.folder / "run.json")
    assert persisted["dataset_attempted"] is True
    if operation == "kernel":
        assert persisted["kernel_attempted"] is True
        assert backend.kernel["code_sha256"] == persisted["notebook_digest"]
    recovered = submit(sample, backend, state)
    assert recovered["status"] == "SUBMITTED"
    assert recovered["kernel_version"] == 1
    assert backend.count("create_dataset") == backend.count("push_kernel") == 1


@pytest.mark.parametrize("operation", ["dataset", "kernel"])
def test_unknown_404_after_lost_response_stays_blocked(sample, operation):
    state = prepare(sample)
    backend = FakeBackend(sample[0], state)
    setattr(backend, "lose_dataset_response" if operation == "dataset" else "lose_push_response", True)
    with pytest.raises(LostResponse):
        submit(sample, backend, state)
    setattr(backend, operation, None)  # Adapter's 404 contract: None, not permission to repeat.
    counts = (backend.count("create_dataset"), backend.count("push_kernel"))
    for _ in range(2):
        with pytest.raises(core.PipelineError, match="unknown"):
            submit(sample, backend, state)
    assert counts == (backend.count("create_dataset"), backend.count("push_kernel"))


@pytest.mark.parametrize("field,bad_value", [
    ("is_private", False), ("version", 2), ("code_sha256", "0" * 64),
    ("dataset_sources", ["different-owner/unrelated-input"]),
])
def test_kernel_reconciliation_refuses_wrong_identity(sample, field, bad_value):
    state = prepare(sample)
    backend = FakeBackend(sample[0], state)
    backend.lose_push_response = True
    with pytest.raises(LostResponse):
        submit(sample, backend, state)
    backend.kernel[field] = bad_value
    with pytest.raises(core.PipelineError):
        submit(sample, backend, state)
    assert backend.count("push_kernel") == 1
    assert read_json(backend.folder / "run.json")["kernel_version"] is None


@pytest.mark.parametrize("field,bad_value", [("is_private", False), ("version", 2)])
def test_dataset_reconciliation_refuses_private_or_version_mismatch(sample, field, bad_value):
    state = prepare(sample)
    backend = FakeBackend(sample[0], state)
    backend.lose_dataset_response = True
    with pytest.raises(LostResponse):
        submit(sample, backend, state)
    backend.dataset[field] = bad_value
    with pytest.raises(core.PipelineError):
        submit(sample, backend, state)
    assert backend.count("create_dataset") == 1
    assert backend.count("push_kernel") == 0


@pytest.mark.parametrize("relative", ["plan.json", "dataset/" + PAYLOAD, "kernel/" + NOTEBOOK,
                                     "dataset/credentials.json"])
def test_staging_tamper_rejected_before_any_remote_operation(sample, relative):
    state = prepare(sample)
    backend = FakeBackend(sample[0], state)
    (backend.folder / relative).write_bytes(b"changed input")
    with pytest.raises(core.PipelineError):
        submit(sample, backend, state)
    assert backend.calls == []


def test_account_mismatch_never_uses_backend(sample):
    state = prepare(sample)
    backend = FakeBackend(sample[0], state)
    submit(sample, backend, state)
    backend.calls.clear()
    with pytest.raises(core.PipelineError, match="account"):
        core.submit(sample[0], state["run_id"], backend=backend, username="another-owner")
    assert backend.calls == []


def test_invalid_account_error_does_not_reveal_key(sample, capsys):
    config, _ = sample
    write_json(Path(config["account_file"]), {"username": "invalid/user", "key": SECRET})
    with pytest.raises(core.PipelineError) as error:
        core.account(config)
    assert str(Path(config["account_file"])) in str(error.value)
    assert SECRET not in str(error.value)
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err


def test_status_and_fetch_pin_saved_version_even_if_remote_latest_changes(sample, tmp_path):
    state = prepare(sample)
    backend = FakeBackend(sample[0], state)
    submitted = submit(sample, backend, state)
    backend.kernel["version"] = 99
    backend.bundle = synthetic_bundle(tmp_path / "remote-output", state["plan_hash"])
    assert core.status(sample[0], state["run_id"], backend=backend)["remote_status"] == "complete"
    fetched = core.fetch(sample[0], state["run_id"], backend=backend)
    assert fetched["status"] == "VERIFIED"
    assert fetched["verification"]["integrity"] == "PASS"
    assert fetched["kernel_version"] == 1
    expected = (submitted["kernel_ref"], 1)
    assert all(call[1:] == expected for call in backend.calls if call[0] == "kernel_status")
    downloads = [call for call in backend.calls if call[0] == "download_outputs"]
    assert len(downloads) == 1
    assert downloads[0][1:3] == expected
    assert read_json(Path(fetched["output_dir"]) / "PIPELINE_VERIFICATION.json")["status"] == "VERIFIED"
    assert (backend.folder / "records/kernel/status.jsonl").is_file()
    event_names = [
        json.loads(line)["event"]
        for line in (backend.folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "started" in event_names and "completed" in event_names


@pytest.mark.parametrize("change,expected", [
    ({}, "VERIFIED"), ({"incomplete": True}, "FETCHED_INCOMPLETE"),
    ({"missing_recovery": True}, "FETCHED_INCOMPLETE"),
])
def test_result_integrity_is_distinct_from_business_completion(tmp_path, change, expected):
    plan_hash = "a" * 64
    synthetic_bundle(tmp_path, plan_hash, **change)
    verified = results.verify_results(tmp_path, plan_hash)
    assert verified["integrity"] == "PASS"
    assert verified["status"] == expected
    assert verified["plan_hash"] == plan_hash


@pytest.mark.parametrize("change", [{"wrong_plan": True}, {"tamper": True}])
def test_result_wrong_plan_or_same_size_tampering_fails(tmp_path, change):
    synthetic_bundle(tmp_path, "b" * 64, **change)
    with pytest.raises(results.ResultError):
        results.verify_results(tmp_path, "b" * 64)


def test_duplicate_identical_result_archives_are_valid(tmp_path):
    archive = synthetic_bundle(tmp_path, "c" * 64)
    nested = tmp_path / "artifacts"
    nested.mkdir()
    shutil.copyfile(archive, nested / ARCHIVE)
    assert results.verify_results(tmp_path, "c" * 64)["status"] == "VERIFIED"


def test_conflicting_result_archive_copies_are_refused(tmp_path):
    synthetic_bundle(tmp_path, "d" * 64)
    synthetic_bundle(tmp_path / "artifacts", "d" * 64, incomplete=True)
    with pytest.raises(results.ResultError):
        results.verify_results(tmp_path, "d" * 64)


def test_run_lock_excludes_other_process_and_releases(sample):
    state = prepare(sample)
    folder = core.locate(sample[0], state["run_id"])
    program = (
        "import sys\nfrom pathlib import Path\n"
        "from tools.kaggle_pipeline.core import run_lock, PipelineError\n"
        "try:\n with run_lock(Path(sys.argv[1])):\n  print('ACQUIRED')\n"
        "except PipelineError:\n print('BLOCKED')\n"
    )
    command = [sys.executable, "-B", "-c", program, str(folder)]
    with core.run_lock(folder):
        blocked = subprocess.run(command, cwd=core.ROOT, capture_output=True, text=True, timeout=20)
    assert blocked.returncode == 0, blocked.stderr
    assert blocked.stdout.strip() == "BLOCKED"
    released = subprocess.run(command, cwd=core.ROOT, capture_output=True, text=True, timeout=20)
    assert released.returncode == 0, released.stderr
    assert released.stdout.strip() == "ACQUIRED"
