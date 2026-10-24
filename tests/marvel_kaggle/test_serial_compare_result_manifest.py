"""Check real result ZIPs, including an injected first-write space failure."""
from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from serial_compare import reporting  # noqa: E402
from serial_compare.common import canonical  # noqa: E402


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _fixture(tmp_path):
    artifacts = tmp_path / "artifacts"
    inputs = tmp_path / "dataset"
    _write(inputs / "plan.json", b'{"plan_hash":"fixture"}')
    _write(inputs / "models/source.pt", b"static source policy bytes")
    _write(inputs / "deals_confirmation.json", b"[]")
    recovery = {}
    # Cover a normal directory, a singular checkpoint directory, and a
    # single-process recovery file outside any checkpoint directory.
    for arm, checkpoint in (("control", "checkpoints/iter4"), ("candidate", "latest.pt")):
        run = artifacts / "phases" / f"train_{arm}" / "run"
        files = ([run / checkpoint / "rank000.pt", run / checkpoint / "manifest.json"]
                 if arm == "control" else [run / checkpoint])
        for path in files:
            _write(path, f"latest {arm}: {path.name}".encode())
        index = _write(run / "checkpoint_manifest.json", json.dumps({
            "latest": {"checkpoint": checkpoint, "iteration": 4},
        }).encode())
        for path in [*files, index]:
            recovery["artifacts/" + path.relative_to(artifacts).as_posix()] = path.read_bytes()
        _write(run / "checkpoints/iter2/rank000.pt", b"older recovery")
    evidence = {}
    for relative, data in [
        ("phases/evaluate_confirmation_primary/games.jsonl", b'{"game_id":1}\n'),
        ("phases/evaluate_confirmation_primary/summary.json", b'{"completed_games":1}'),
        ("phases/evaluate_development_diagnostics/games.jsonl", b'{"game_id":2}\n'),
        ("diagnostics/extra.log", b"static diagnostics"),
    ]:
        _write(artifacts / relative, data)
        evidence["artifacts/" + relative] = data
    return artifacts, inputs, recovery, evidence


def _check_zip(archive):
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert len(names) == len(set(names)), "duplicate ZIP entries"
        assert names.count("RESULTS_MANIFEST.json") == 1
        assert not any(name.endswith("/RESULTS_MANIFEST.json") for name in names)
        manifest = json.loads(bundle.read("RESULTS_MANIFEST.json"))
        assert manifest["schema"] == reporting.RESULTS_MANIFEST_SCHEMA
        assert manifest["hash"] == canonical({key: value for key, value in manifest.items() if key != "hash"})
        declared = manifest["files"]
        assert len(declared) == len({record["path"] for record in declared})
        assert {record["path"] for record in declared} == set(names) - {"RESULTS_MANIFEST.json"}
        assert "RESULTS_MANIFEST.json" not in {record["path"] for record in declared}
        contents = {name: bundle.read(name) for name in names}
        for record in declared:
            data = contents[record["path"]]
            assert record["bytes"] == len(data)
            assert record["sha256"] == hashlib.sha256(data).hexdigest()
    return manifest, contents


@pytest.mark.parametrize("fallback", [False, True])
def test_result_manifest_matches_real_archive_and_discloses_omissions(tmp_path, monkeypatch, fallback):
    artifacts, inputs, recovery, evidence = _fixture(tmp_path)
    original_write_zip = reporting._write_zip
    attempted = []

    def write_with_first_failure(members, destination):
        attempted.append(members)
        if fallback and len(attempted) == 1:
            raise OSError(errno.ENOSPC, "injected first-write space pressure")
        # The successful write always uses the real implementation.
        original_write_zip(members, destination)

    monkeypatch.setattr(reporting, "_write_zip", write_with_first_failure)
    written = reporting.write_reports(
        artifacts, {"status": "complete", "confirmation": {"complete": True}}, resume_inputs=inputs,
    )
    assert len(attempted) == (2 if fallback else 1)
    archive = Path(written["archive"])
    assert written["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest, contents = _check_zip(archive)
    full_manifest = json.loads(dict(attempted[0])["RESULTS_MANIFEST.json"])
    assert full_manifest["mode"] == "full"
    assert full_manifest["evidence_complete"] is True
    assert full_manifest["omissions"] == []
    for name, data in recovery.items():
        assert contents[name] == data
        # Neither the original recovery nor its index may be deleted.
        assert (artifacts / name.removeprefix("artifacts/")).read_bytes() == data
    for path in inputs.rglob("*"):
        if path.is_file():
            assert contents["input/" + path.relative_to(inputs).as_posix()] == path.read_bytes()
    recovery_index = json.loads(contents["artifacts/recovery_index.json"])
    for arm in recovery_index["arms"].values():
        for record in arm["files"]:
            data = contents["artifacts/" + record["path"]]
            assert record["bytes"] == len(data)
            assert record["sha256"] == hashlib.sha256(data).hexdigest()
    assert not any("iter2/" in name for name in contents)
    if fallback:
        assert manifest["mode"] == "space_pressure_fallback"
        assert manifest["evidence_complete"] is False
        assert manifest["hash"] != full_manifest["hash"]
        omissions = {record["path"]: record for record in manifest["omissions"]}
        assert len(omissions) == len(manifest["omissions"])
        assert set(omissions) == set(evidence)
        assert not (set(omissions) & set(contents))
        assert {record["path"] for record in full_manifest["files"]} == (
            {record["path"] for record in manifest["files"]} | set(omissions)
        )
        for name, data in evidence.items():
            assert omissions[name]["bytes"] == len(data)
            assert omissions[name]["sha256"] == hashlib.sha256(data).hexdigest()
            assert (artifacts / name.removeprefix("artifacts/")).read_bytes() == data
    else:
        assert manifest == full_manifest
        for name, data in evidence.items():
            assert contents[name] == data


def test_fallback_keeps_singular_checkpoint_tree_and_index(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    run = artifacts / "phases/train_control/run"
    _write(run / "checkpoint/latest/rank000.pt", b"only recovery")
    _write(run / "checkpoint/latest/metadata.json", b"{}")
    _write(run / "checkpoint_manifest.json", b'{"latest":{"checkpoint":"checkpoint/latest"}}')
    original_write_zip = reporting._write_zip
    calls = 0

    def fail_once(members, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.ENOSPC, "injected")
        original_write_zip(members, destination)

    monkeypatch.setattr(reporting, "_write_zip", fail_once)
    written = reporting.write_reports(artifacts, {"status": "in_progress"})
    manifest, contents = _check_zip(written["archive"])
    assert calls == 2
    for path in run.rglob("*"):
        if path.is_file():
            assert contents["artifacts/" + path.relative_to(artifacts).as_posix()] == path.read_bytes()
    assert manifest["mode"] == "space_pressure_fallback"
    assert manifest["omissions"] == []
    assert manifest["evidence_complete"] is True  # No evidence dropped in this fixture.
