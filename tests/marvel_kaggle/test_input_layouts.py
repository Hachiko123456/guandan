from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from guandan_marvel.kaggle.bootstrap import (
    RECOVERY_MANIFEST_SCHEMA, canonical_hash, discover_dataset, discover_recovery,
)
from guandan_marvel.kaggle.package import build_delivery


@pytest.fixture(scope="module")
def layout_delivery(tmp_path_factory):
    output = tmp_path_factory.mktemp("layout-delivery") / "delivery"
    build_delivery(Path(__file__).resolve().parents[2], output, allow_dirty=True)
    return output


def _copy_mode(delivery: Path, tmp_path: Path, mode: str = "pilot") -> Path:
    target = tmp_path / "input" / "wrapper" / f"dataset-{mode}"
    shutil.copytree(delivery / "datasets" / mode, target)
    return target


def _unpack_source(dataset_root: Path, *, wrapper: str = "auto/unpacked") -> Path:
    archive = dataset_root / "guandan-marvel-source.zip"
    destination = dataset_root / wrapper
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink()
    return destination / "guandan-marvel-source"


def _recovery_payload(*, source: dict[str, object], iteration: int, content: bytes = b"checkpoint"):
    file_row = {"path": "checkpoint/rank000.pt", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    body = {
        "schema": RECOVERY_MANIFEST_SCHEMA,
        "package_id": "guandan-marvel-t4x2-20261010",
        "created_at": "2026-10-10T00:00:00+00:00",
        "status": "partial",
        "source_commit": source["git_commit"],
        "source_archive_sha256": source["archive_sha256"],
        "source_content_manifest_hash": source["content_manifest_hash"],
        "checkpoint": "checkpoint/manifest.json",
        "checkpoint_manifest_hash": "a" * 64,
        "iteration": iteration,
        "runtime_binary": None,
        "files": [file_row],
        "remote_verified": False,
    }
    return {**body, "manifest_hash": canonical_hash(body)}, content


def _write_recovery_zip(path: Path, *, source: dict[str, object], iteration: int) -> dict[str, object]:
    payload, content = _recovery_payload(source=source, iteration=iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("recovery_manifest.json", json.dumps(payload))
        archive.writestr("checkpoint/rank000.pt", content)
    return payload


def _write_recovery_directory(path: Path, *, source: dict[str, object], iteration: int) -> dict[str, object]:
    payload, content = _recovery_payload(source=source, iteration=iteration)
    (path / "checkpoint").mkdir(parents=True)
    (path / "checkpoint/rank000.pt").write_bytes(content)
    (path / "recovery_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_source_zip_retained_still_resolves(layout_delivery, tmp_path):
    dataset_root = _copy_mode(layout_delivery, tmp_path)
    found = discover_dataset(tmp_path / "input")
    assert found.source.kind == "zip"
    assert found.source.archive_sha256_verified is True
    assert found.source.path == dataset_root / "guandan-marvel-source.zip"


def test_source_auto_unpacked_without_zip_resolves_with_wrappers(layout_delivery, tmp_path):
    dataset_root = _copy_mode(layout_delivery, tmp_path)
    source_root = _unpack_source(dataset_root, wrapper="outer/inner")
    before = {p.relative_to(source_root).as_posix(): p.read_bytes() for p in source_root.rglob("*") if p.is_file()}
    found = discover_dataset(tmp_path / "input")
    after = {p.relative_to(source_root).as_posix(): p.read_bytes() for p in source_root.rglob("*") if p.is_file()}
    assert found.source.kind == "directory"
    assert found.source.path == source_root.resolve()
    assert found.source.archive_sha256_verified is False
    assert before == after


def test_identical_source_zip_and_directory_are_deduplicated(layout_delivery, tmp_path):
    dataset_root = _copy_mode(layout_delivery, tmp_path)
    archive = dataset_root / "guandan-marvel-source.zip"
    destination = dataset_root / "auto"
    destination.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    found = discover_dataset(tmp_path / "input")
    assert found.source.kind == "directory"
    assert found.source.archive_sha256_verified is True
    assert found.source.archive_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_unpacked_source_tamper_missing_and_conflict_fail_closed(layout_delivery, tmp_path):
    dataset_root = _copy_mode(layout_delivery, tmp_path)
    source_root = _unpack_source(dataset_root)
    target = next(path for path in source_root.rglob("*.py") if path.name != "bootstrap.py")
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        discover_dataset(tmp_path / "input")

    shutil.rmtree(tmp_path / "input")
    dataset_root = _copy_mode(layout_delivery, tmp_path)
    source_root = _unpack_source(dataset_root)
    missing = next(path for path in source_root.rglob("*.py"))
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="neither the declared source ZIP"):
        discover_dataset(tmp_path / "input")

    shutil.rmtree(tmp_path / "input")
    dataset_root = _copy_mode(layout_delivery, tmp_path)
    source_root = _unpack_source(dataset_root)
    conflict = dataset_root / "other/source_manifest.json"
    conflict.parent.mkdir(parents=True)
    payload = json.loads((source_root / "source_manifest.json").read_text(encoding="utf-8"))
    payload["git_commit"] = "0" * 40
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = canonical_hash(body)
    conflict.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="conflicting unpacked source"):
        discover_dataset(tmp_path / "input")


def test_source_zip_path_traversal_and_symlink_fail_closed(layout_delivery, tmp_path):
    dataset_root = _copy_mode(layout_delivery, tmp_path)
    archive = dataset_root / "guandan-marvel-source.zip"
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("../escape", b"x")
    # Update declared archive bytes hash so traversal validation, not outer hash, is exercised.
    manifest_path = dataset_root / "guandan_marvel_dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        discover_dataset(tmp_path / "input")


def test_recovery_zip_and_unpacked_directory_with_wrappers(layout_delivery, tmp_path):
    dataset = json.loads((layout_delivery / "datasets/resume/guandan_marvel_dataset.json").read_text(encoding="utf-8"))
    source = dataset["source"]
    input_root = tmp_path / "input"
    zip_path = input_root / "zip-wrapper/guandan-marvel-recovery.zip"
    payload = _write_recovery_zip(zip_path, source=source, iteration=3)
    found = discover_recovery(input_root, source_commit=source["git_commit"], source_archive_sha256=source["archive_sha256"], source_content_manifest_hash=source["content_manifest_hash"], required=True)
    assert found.kind == "zip" and found.manifest["iteration"] == 3

    directory = input_root / "directory-wrapper/outer/recovery"
    _write_recovery_directory(directory, source=source, iteration=3)
    found = discover_recovery(input_root, source_commit=source["git_commit"], source_archive_sha256=source["archive_sha256"], source_content_manifest_hash=source["content_manifest_hash"], required=True)
    assert found.kind == "directory"
    assert found.content_manifest_hash == payload["manifest_hash"]


def test_recovery_tamper_missing_identity_and_conflict_fail_closed(layout_delivery, tmp_path):
    source = json.loads((layout_delivery / "datasets/resume/guandan_marvel_dataset.json").read_text(encoding="utf-8"))["source"]
    root = tmp_path / "input"
    directory = root / "recovery"
    _write_recovery_directory(directory, source=source, iteration=1)
    (directory / "checkpoint/rank000.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        discover_recovery(root, source_commit=source["git_commit"], source_archive_sha256=source["archive_sha256"], source_content_manifest_hash=source["content_manifest_hash"], required=True)

    shutil.rmtree(root)
    directory = root / "recovery"
    _write_recovery_directory(directory, source=source, iteration=1)
    (directory / "checkpoint/rank000.pt").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        discover_recovery(root, source_commit=source["git_commit"], source_archive_sha256=source["archive_sha256"], source_content_manifest_hash=source["content_manifest_hash"], required=True)

    shutil.rmtree(root)
    _write_recovery_directory(root / "one", source=source, iteration=1)
    _write_recovery_directory(root / "two", source=source, iteration=2)
    with pytest.raises(RuntimeError, match="multiple different"):
        discover_recovery(root, source_commit=source["git_commit"], source_archive_sha256=source["archive_sha256"], source_content_manifest_hash=source["content_manifest_hash"], required=True)

    shutil.rmtree(root)
    wrong = dict(source); wrong["git_commit"] = "0" * 40
    _write_recovery_directory(root / "wrong", source=wrong, iteration=1)
    with pytest.raises(RuntimeError, match="different source package"):
        discover_recovery(root, source_commit=source["git_commit"], source_archive_sha256=source["archive_sha256"], source_content_manifest_hash=source["content_manifest_hash"], required=True)


@pytest.mark.parametrize("unpacked", [False, True])
def test_generated_notebook_actual_bootstrap_cell_handles_layout(layout_delivery, tmp_path, monkeypatch, unpacked):
    dataset_root = _copy_mode(layout_delivery, tmp_path)
    if unpacked:
        _unpack_source(dataset_root, wrapper="kaggle/auto")
    notebook = json.loads((layout_delivery / "guandan_marvel_t4x2.ipynb").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    code = code.replace("Path('/kaggle/input')", f"Path({str(tmp_path / 'input')!r})")
    code = code.replace("Path('/kaggle/working')", f"Path({str(tmp_path / 'working')!r})")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or subprocess.CompletedProcess(args[0], 0))
    namespace: dict[str, object] = {}
    exec(compile(code, "generated-notebook-bootstrap", "exec"), namespace)
    prepared = namespace["PREPARED"]
    assert prepared["source_input_kind"] == ("directory" if unpacked else "zip")
    assert Path(prepared["source_root"]).is_dir()
    assert calls
    build_call = next(row for row in calls if "build_ext" in row[0][0])
    assert build_call[1]["timeout"] == 660.0
    assert build_call[1]["env"]["PYTHONUNBUFFERED"] == "1"
    progress = json.loads((tmp_path / "working/guandan-marvel-bootstrap/bootstrap_progress.json").read_text(encoding="utf-8"))
    assert progress["stage"] == "ready_for_runner"
    assert progress["status"] == "complete"


def test_generated_notebook_build_timeout_writes_downloadable_diagnostics(layout_delivery, tmp_path, monkeypatch):
    _copy_mode(layout_delivery, tmp_path)
    notebook = json.loads((layout_delivery / "guandan_marvel_t4x2.ipynb").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    code = code.replace("Path('/kaggle/input')", f"Path({str(tmp_path / 'input')!r})")
    code = code.replace("Path('/kaggle/working')", f"Path({str(tmp_path / 'working')!r})")

    def timeout_build(command, *args, **kwargs):
        if "build_ext" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", timeout_build)
    with pytest.raises(RuntimeError, match="native_build timed out"):
        exec(compile(code, "generated-notebook-bootstrap-timeout", "exec"), {})
    diagnostic_root = tmp_path / "working/guandan-marvel-bootstrap"
    failure = json.loads((diagnostic_root / "bootstrap_failure.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "RuntimeError"
    assert failure["stage"] == "native_build"
    assert (diagnostic_root / "bootstrap.log").is_file()
    bundle = tmp_path / "working/guandan-marvel-bootstrap-diagnostics.zip"
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        assert {"bootstrap.log", "bootstrap_progress.json", "bootstrap_failure.json"} <= set(archive.namelist())


def test_generated_notebook_actual_bootstrap_cell_handles_unpacked_recovery(layout_delivery, tmp_path, monkeypatch):
    dataset_root = _copy_mode(layout_delivery, tmp_path, mode="resume")
    _unpack_source(dataset_root, wrapper="auto/source")
    dataset = json.loads((dataset_root / "guandan_marvel_dataset.json").read_text(encoding="utf-8"))
    recovery_root = tmp_path / "input/other/wrapper/recovery"
    _write_recovery_directory(recovery_root, source=dataset["source"], iteration=1)
    notebook = json.loads((layout_delivery / "guandan_marvel_t4x2.ipynb").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][1]["source"])
    code = code.replace("Path('/kaggle/input')", f"Path({str(tmp_path / 'input')!r})")
    code = code.replace("Path('/kaggle/working')", f"Path({str(tmp_path / 'working')!r})")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))
    namespace: dict[str, object] = {}
    exec(compile(code, "generated-notebook-bootstrap-recovery", "exec"), namespace)
    prepared = namespace["PREPARED"]
    assert prepared["source_input_kind"] == "directory"
    assert prepared["recovery_input_kind"] == "directory"
    assert Path(prepared["recovery_root"]).is_dir()

def test_source_zip_symlink_member_fails_closed(layout_delivery, tmp_path):
    dataset_root = _copy_mode(layout_delivery, tmp_path)
    archive = dataset_root / "guandan-marvel-source.zip"
    info = zipfile.ZipInfo("guandan-marvel-source/link")
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr(info, "target")
    manifest_path = dataset_root / "guandan_marvel_dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ZIP symlink"):
        discover_dataset(tmp_path / "input")

def test_source_auto_unpack_flattened_into_dataset_directory(layout_delivery, tmp_path):
    dataset_root = _copy_mode(layout_delivery, tmp_path)
    archive = dataset_root / "guandan-marvel-source.zip"
    temp = dataset_root / "temp-extract"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(temp)
    extracted = temp / "guandan-marvel-source"
    for child in list(extracted.iterdir()):
        destination = dataset_root / child.name
        if destination.exists() and destination.is_file():
            destination.unlink()
        shutil.move(str(child), destination)
    shutil.rmtree(temp)
    archive.unlink()
    found = discover_dataset(tmp_path / "input")
    assert found.source.kind == "directory"
    assert found.source.path == dataset_root.resolve()
