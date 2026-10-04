"""Portable committed source export and provenance tests. No Kaggle/network use."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

import pytest

from guandan.deployment.package import build_package
from guandan.deployment.provenance import source_provenance


REMOTE_PROFILE = {
    "version": "profiles-0.2",
    "profiles": {
        "remote_full": {
            "mode": "remote",
            "device": "cuda",
            "algorithms": ["ippo", "vrpo"],
            "training": {
                "updates_per_algorithm": 100,
                "rollout_envs": 8,
                "token_steps_per_update": 1024,
                "checkpoint_interval": 10,
                "resume_updates": 2,
                "max_hours": 12,
            },
            "evaluation": {
                "deal_groups_per_pairing": 250,
                "seat_rotations": [0, 1, 2, 3],
                "opponents": ["random", "rule", "snapshot"],
                "max_hours": 12,
            },
        }
    },
}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode("utf-8").strip()


def fixture_repo(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    files = {
        "configs/acceptance_profiles.json": json.dumps(REMOTE_PROFILE, sort_keys=True),
        "pyproject.toml": "[project]\nname = 'fixture'\nversion = '0'\n",
        "guandan/__init__.py": "VALUE = 17\n",
        "guandan/deployment/__init__.py": "\n",
        "guandan/deployment/provenance.py": "SOURCE = 'fixture'\n",
        "guandan/deployment/package.py": "SOURCE = 'fixture'\n",
        "guandan/deployment/control.py": "SOURCE = 'fixture'\n",
        "guandan/deployment/session.py": "SOURCE = 'fixture'\n",
        "scripts/kaggle_entry.py": "SOURCE = 'fixture'\n",
        "scripts/kaggle_environment_check.py": "SOURCE = 'fixture'\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture source")
    return root


def unpacked_export(tmp_path, name="export"):
    work = tmp_path / name
    work.mkdir()
    root = fixture_repo(work)
    result = build_package(root, work / "source.zip")
    target = work / "unpacked"
    with zipfile.ZipFile(result["archive"]) as archive:
        archive.extractall(target)
    return target / "guandan"


def test_source_archive_is_reproducible_and_gitless_verifiable(tmp_path):
    root = fixture_repo(tmp_path)
    first = build_package(root, tmp_path / "first.zip")
    second = build_package(root, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]
    assert first["uploaded"] is False and first["kaggle_verified"] is False
    target = tmp_path / "unpacked"
    with zipfile.ZipFile(first["archive"]) as archive:
        assert all("/.git/" not in name for name in archive.namelist())
        archive.extractall(target)
    exported = target / "guandan"
    metadata = source_provenance(exported)
    assert metadata["git_commit"] == git(root, "rev-parse", "HEAD")
    assert metadata["source_kind"] == "verified_export"
    assert not (exported / ".git").exists()
    assert (exported / "configs/acceptance_profiles.json").is_file()
    with pytest.raises(FileExistsError):
        build_package(root, tmp_path / "first.zip")


def test_export_rejects_uncommitted_inputs_and_tamper(tmp_path):
    root = fixture_repo(tmp_path)
    (root / "extra.py").write_text("unreviewed = True")
    with pytest.raises(ValueError, match="commit"):
        build_package(root, tmp_path / "invalid.zip")
    git(root, "add", "extra.py")
    git(root, "commit", "-qm", "review extra")
    result = build_package(root, tmp_path / "valid.zip")
    with zipfile.ZipFile(result["archive"]) as archive:
        archive.extractall(tmp_path / "unpacked")
    exported = tmp_path / "unpacked/guandan"
    (exported / "extra.py").write_text("tampered = True")
    with pytest.raises(ValueError, match="mismatch"):
        source_provenance(exported)


@pytest.mark.parametrize("filename", ["secret.pem", "kaggle.json", ".env", "model.pt"])
def test_secret_and_checkpoint_files_are_never_exported(tmp_path, filename):
    root = fixture_repo(tmp_path)
    (root / filename).write_text("synthetic sentinel not a real secret")
    git(root, "add", ".")
    git(root, "commit", "-qm", "synthetic unsafe fixture")
    with pytest.raises(ValueError, match="sensitive"):
        build_package(root, tmp_path / "unsafe.zip")


def test_manifest_path_escape_is_rejected(tmp_path):
    root = tmp_path / "exported"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1")
    manifest = {
        "format": "guandan-source-v1",
        "git_commit": "a" * 40,
        "files": {"../outside.py": hashlib.sha256(outside.read_bytes()).hexdigest()},
        "profile": "remote_full",
        "uploaded": False,
        "accepted": False,
        "kaggle_verified": False,
    }
    (root / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        source_provenance(root)


@pytest.mark.parametrize("filename", ["/absolute.py", "C:/host.py", "sub\\host.py", "./host.py"])
def test_manifest_rejects_noncanonical_paths(tmp_path, filename):
    root = tmp_path / "exported"
    root.mkdir()
    manifest = {
        "format": "guandan-source-v1",
        "git_commit": "a" * 40,
        "files": {filename: "0" * 64},
        "profile": "remote_full",
        "uploaded": False,
        "accepted": False,
        "kaggle_verified": False,
    }
    (root / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        source_provenance(root)


def test_manifest_requires_nonempty_exact_file_set(tmp_path):
    exported = unpacked_export(tmp_path, "empty")
    manifest_path = exported / "SOURCE_MANIFEST.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["files"] = {}
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        source_provenance(exported)

    exported = unpacked_export(tmp_path, "extra")
    (exported / "unlisted.py").write_text("not in manifest", encoding="utf-8")
    with pytest.raises(ValueError, match="file-set mismatch"):
        source_provenance(exported)

    exported = unpacked_export(tmp_path, "missing")
    data = json.loads((exported / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    missing = next(iter(data["files"]))
    (exported / missing).unlink()
    with pytest.raises(ValueError, match="file-set mismatch"):
        source_provenance(exported)


@pytest.mark.parametrize("field,value", [
    ("profile", "local_fast"),
    ("uploaded", True),
    ("accepted", True),
    ("kaggle_verified", True),
])
def test_manifest_rejects_forged_deployment_metadata(tmp_path, field, value):
    exported = unpacked_export(tmp_path, field)
    manifest_path = exported / "SOURCE_MANIFEST.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data[field] = value
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        source_provenance(exported)


@pytest.mark.parametrize("missing", ["profile", "uploaded", "accepted", "kaggle_verified"])
def test_manifest_rejects_missing_deployment_metadata(tmp_path, missing):
    exported = unpacked_export(tmp_path, f"missing-{missing}")
    manifest_path = exported / "SOURCE_MANIFEST.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.pop(missing)
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema/keys"):
        source_provenance(exported)


def test_manifest_rejects_linked_source(tmp_path):
    exported = unpacked_export(tmp_path, "linked")
    target = exported / "guandan/link.py"
    try:
        os.symlink(exported / "guandan/__init__.py", target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="linked source"):
        source_provenance(exported)


def test_manifest_rejects_link_seam_without_os_link_permission(tmp_path, monkeypatch):
    exported = unpacked_export(tmp_path, "link-seam")
    linked = exported / "guandan/link.py"
    linked.write_text("link seam", encoding="utf-8")
    manifest_path = exported / "SOURCE_MANIFEST.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["files"]["guandan/link.py"] = hashlib.sha256(linked.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    original_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == linked or original_symlink(path))
    with pytest.raises(ValueError, match="linked source"):
        source_provenance(exported)

    if hasattr(Path, "is_junction"):
        original_junction = Path.is_junction
        monkeypatch.setattr(Path, "is_junction", lambda path: path == linked or original_junction(path))
        monkeypatch.setattr(Path, "is_symlink", original_symlink)
        with pytest.raises(ValueError, match="linked source"):
            source_provenance(exported)


def test_git_checkout_provenance_is_actual_commit_and_dirty_flag(tmp_path):
    root = fixture_repo(tmp_path)
    clean = source_provenance(root)
    assert clean["source_kind"] == "git" and clean["git_worktree_clean"] is True
    (root / "dirty").write_text("x")
    assert source_provenance(root)["git_worktree_clean"] is False
