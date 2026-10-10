from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile

import pytest

from guandan_marvel.kaggle.common import canonical_hash
from guandan_marvel.kaggle.discovery import discover_dataset, discover_recovery
from guandan_marvel.kaggle.package import (
    NOTEBOOK_NAME, build_delivery, validate_delivery, validate_notebook, validate_source_archive,
)


@pytest.fixture(scope="module")
def delivery(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path_factory.mktemp("m07-delivery") / "guandan-marvel-t4x2-20261010"
    build_delivery(root, output, allow_dirty=True)
    return output


def test_delivery_builds_all_modes_and_validates(delivery):
    result = validate_delivery(delivery)
    assert result["valid"] is True
    assert result["remote_verified"] is False
    assert {row["mode"] for row in result["datasets"]} == {"preflight", "pilot", "resume", "full", "eval_only"}
    assert (delivery / "README_中文.md").is_file()
    assert (delivery / NOTEBOOK_NAME).is_file()


def test_notebook_has_no_edit_placeholders_and_parses(delivery):
    result = validate_notebook(delivery / NOTEBOOK_NAME)
    assert result["syntax_valid"] is True
    assert result["placeholders_absent"] is True
    text = (delivery / NOTEBOOK_NAME).read_text(encoding="utf-8")
    assert "torchrun --standalone --nproc_per_node=2" in text
    assert "/kaggle/input" in text
    assert "/kaggle/working" in text


def test_dataset_discovery_is_unique_and_conflicts_fail_closed(delivery, tmp_path):
    input_root = tmp_path / "input"
    shutil.copytree(delivery / "datasets" / "preflight", input_root / "one")
    found = discover_dataset(input_root)
    assert found.manifest["mode"] == "preflight"
    shutil.copytree(delivery / "datasets" / "pilot", input_root / "two")
    with pytest.raises(RuntimeError, match="exactly one"):
        discover_dataset(input_root)


def test_source_zip_rejects_unsafe_extra_member(delivery, tmp_path):
    source = delivery / "datasets" / "pilot" / "guandan-marvel-source.zip"
    assert validate_source_archive(source)["valid"] is True
    tampered = tmp_path / "tampered.zip"
    shutil.copy2(source, tampered)
    with zipfile.ZipFile(tampered, "a") as archive:
        archive.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="unsafe"):
        validate_source_archive(tampered)


def _recovery(path: Path, *, commit: str, archive_sha: str, iteration: int) -> None:
    body = {
        "schema": "guandan-marvel-recovery-manifest-v2",
        "package_id": "guandan-marvel-t4x2-20261010",
        "created_at": "2026-10-10T00:00:00+00:00",
        "status": "partial",
        "source_commit": commit,
        "source_archive_sha256": archive_sha,
        "checkpoint": "checkpoint/manifest.json",
        "checkpoint_manifest_hash": "a" * 64,
        "iteration": iteration,
        "runtime_binary": None,
        "files": [],
        "remote_verified": False,
    }
    payload = {**body, "manifest_hash": canonical_hash(body)}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("recovery_manifest.json", json.dumps(payload))


def test_recovery_discovery_matches_source_and_rejects_ambiguity(delivery, tmp_path):
    dataset = json.loads((delivery / "datasets" / "resume" / "guandan_marvel_dataset.json").read_text(encoding="utf-8"))
    source = dataset["source"]
    root = tmp_path / "input"
    (root / "a").mkdir(parents=True)
    first = root / "a" / "guandan-marvel-recovery.zip"
    _recovery(first, commit=source["git_commit"], archive_sha=source["archive_sha256"], iteration=1)
    found = discover_recovery(root, source_commit=source["git_commit"], source_archive_sha256=source["archive_sha256"], required=True)
    assert found is not None and found.manifest["iteration"] == 1
    (root / "b").mkdir()
    second = root / "b" / "guandan-marvel-partial.zip"
    _recovery(second, commit=source["git_commit"], archive_sha=source["archive_sha256"], iteration=2)
    with pytest.raises(RuntimeError, match="multiple matching"):
        discover_recovery(root, source_commit=source["git_commit"], source_archive_sha256=source["archive_sha256"], required=True)


def test_full_is_not_the_recommended_default(delivery):
    manifest = json.loads((delivery / "delivery_manifest.json").read_text(encoding="utf-8"))
    assert manifest["default_recommended_mode"] == "preflight"
    full = json.loads((delivery / "datasets" / "full" / "guandan_marvel_dataset.json").read_text(encoding="utf-8"))
    assert full["mode"] == "full"
    assert full["profiles"]["full"]["user_selected_full_manifest_required"] is True
