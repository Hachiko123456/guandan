from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]


def generator():
    spec = importlib.util.spec_from_file_location(
        "prepare_performance_discovery",
        ROOT / "scripts" / "prepare_kaggle_performance.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discovery_function():
    namespace = {}
    exec(generator().DISCOVERY, namespace)
    discover_inputs = namespace.get("discover_inputs")
    assert callable(discover_inputs)
    return discover_inputs


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source_directory(path: Path, manifest: bytes) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SOURCE_MANIFEST.json").write_bytes(manifest)
    (path / "guandan").mkdir()
    (path / "guandan" / "performance_gate.py").write_text("# fake checkpoint-free source\n", encoding="utf-8")
    return hashlib.sha256(manifest).hexdigest()


def write_source_zip(path: Path, manifest: bytes) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("guandan/SOURCE_MANIFEST.json", manifest)
        archive.writestr("guandan/guandan/performance_gate.py", "# fake source\n")
    return digest(path), hashlib.sha256(manifest).hexdigest()


def write_progress_directory(path: Path) -> None:
    for algorithm in ("ippo", "vrpo"):
        checkpoint = path / "workers" / algorithm / "checkpoint.txt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(f"fake-{algorithm}-checkpoint\n", encoding="utf-8")


def write_progress_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("workers/ippo/checkpoint.txt", "fake-ippo-checkpoint\n")
        archive.writestr("workers/vrpo/checkpoint.txt", "fake-vrpo-checkpoint\n")


def no_budget_check():
    return None


def test_auto_finds_renamed_source_zip_in_nested_kaggle_dataset_layout(tmp_path):
    discover_inputs = discovery_function()
    input_root = tmp_path / "kaggle" / "input"
    renamed_zip = (
        input_root
        / "datasets"
        / "owner"
        / "name"
        / "versions"
        / "17"
        / "uploaded-source-without-standard-name"
    )
    manifest = json.dumps({"format": "guandan-source-v1", "files": {}}, sort_keys=True).encode()
    source_sha256, manifest_sha256 = write_source_zip(renamed_zip, manifest)

    result = discover_inputs(
        "",
        None,
        input_root,
        no_budget_check,
        source_sha256,
        manifest_sha256,
    )

    assert result == {
        "source": str(renamed_zip.resolve()),
        "progress": None,
        "source_mode": "auto",
        "progress_mode": "fresh_seeded",
    }


def test_auto_source_excludes_old_expanded_manifest_even_when_nested_like_kaggle_input(tmp_path):
    discover_inputs = discovery_function()
    input_root = tmp_path / "kaggle" / "input"
    old_root = input_root / "datasets" / "owner" / "name" / "versions" / "3" / "source"
    current_root = input_root / "datasets" / "owner" / "name" / "versions" / "9" / "source"
    old_manifest = json.dumps({"git_commit": "old"}, sort_keys=True).encode()
    current_manifest = json.dumps({"git_commit": "current"}, sort_keys=True).encode()
    write_source_directory(old_root, old_manifest)
    write_source_directory(current_root, current_manifest)

    result = discover_inputs(
        "",
        None,
        input_root,
        no_budget_check,
        "e" * 64,
        hashlib.sha256(current_manifest).hexdigest(),
    )

    assert result["source"] == str(current_root.resolve())
    assert result["source_mode"] == "auto"
    assert result["progress"] is None
    assert result["progress_mode"] == "fresh_seeded"


@pytest.mark.parametrize("kind", ["directory", "zip"])
def test_auto_finds_progress_with_both_algorithm_roots(tmp_path, kind):
    discover_inputs = discovery_function()
    input_root = tmp_path / "kaggle" / "input"
    source = input_root / "source-upload"
    manifest = b'{"format":"guandan-source-v1","files":{}}\n'
    source_sha256, manifest_sha256 = write_source_zip(source, manifest)
    if kind == "directory":
        progress = input_root / "datasets" / "owner" / "name" / "versions" / "4" / "renamed-progress"
        write_progress_directory(progress)
    else:
        progress = input_root / "datasets" / "owner" / "name" / "versions" / "4" / "renamed-progress-upload.zip"
        write_progress_zip(progress)

    result = discover_inputs(
        str(source),
        "",
        input_root,
        no_budget_check,
        source_sha256,
        manifest_sha256,
    )

    assert result["source"] == str(source.resolve())
    assert result["source_mode"] == "explicit"
    assert result["progress"] == str(progress.resolve())
    assert result["progress_mode"] == "auto"


def test_explicit_progress_reports_explicit_mode(tmp_path):
    discover_inputs = discovery_function()
    input_root = tmp_path / "input"
    source = input_root / "source.zip"
    progress = input_root / "progress"
    manifest = b'{"format":"guandan-source-v1","files":{}}\n'
    source_sha256, manifest_sha256 = write_source_zip(source, manifest)
    write_progress_directory(progress)

    result = discover_inputs(
        str(source),
        str(progress),
        input_root,
        no_budget_check,
        source_sha256,
        manifest_sha256,
    )

    assert result["progress"] == str(progress.resolve())
    assert result["progress_mode"] == "explicit"


def test_auto_progress_rejects_multiple_candidates_instead_of_picking_newest(tmp_path):
    discover_inputs = discovery_function()
    input_root = tmp_path / "input"
    source = input_root / "source.zip"
    manifest = b'{"format":"guandan-source-v1","files":{}}\n'
    source_sha256, manifest_sha256 = write_source_zip(source, manifest)
    first = input_root / "datasets" / "owner" / "name" / "versions" / "1" / "checkpoint-a"
    second = input_root / "datasets" / "owner" / "name" / "versions" / "99" / "checkpoint-b-renamed.zip"
    write_progress_directory(first)
    write_progress_zip(second)

    with pytest.raises(ValueError):
        discover_inputs(
            str(source),
            "",
            input_root,
            no_budget_check,
            source_sha256,
            manifest_sha256,
        )


def test_auto_progress_without_a_candidate_rejects_instead_of_silently_seeding_randomly(tmp_path):
    discover_inputs = discovery_function()
    input_root = tmp_path / "input"
    source = input_root / "source.zip"
    manifest = b'{"format":"guandan-source-v1","files":{}}\n'
    source_sha256, manifest_sha256 = write_source_zip(source, manifest)

    with pytest.raises(ValueError):
        discover_inputs(
            str(source),
            "",
            input_root,
            no_budget_check,
            source_sha256,
            manifest_sha256,
        )


def test_none_progress_is_explicit_fresh_seeded_mode(tmp_path):
    discover_inputs = discovery_function()
    input_root = tmp_path / "input"
    source = input_root / "source.zip"
    manifest = b'{"format":"guandan-source-v1","files":{}}\n'
    source_sha256, manifest_sha256 = write_source_zip(source, manifest)

    result = discover_inputs(
        str(source),
        None,
        input_root,
        no_budget_check,
        source_sha256,
        manifest_sha256,
    )

    assert result["progress"] is None
    assert result["progress_mode"] == "fresh_seeded"


def test_explicit_source_and_progress_must_be_inside_input_root(tmp_path):
    discover_inputs = discovery_function()
    input_root = tmp_path / "input"
    inside_source = input_root / "source.zip"
    outside_source = tmp_path / "outside-source.zip"
    outside_progress = tmp_path / "outside-progress"
    manifest = b'{"format":"guandan-source-v1","files":{}}\n'
    source_sha256, manifest_sha256 = write_source_zip(inside_source, manifest)
    write_source_zip(outside_source, manifest)
    write_progress_directory(outside_progress)

    with pytest.raises(ValueError):
        discover_inputs(
            str(outside_source),
            None,
            input_root,
            no_budget_check,
            digest(outside_source),
            hashlib.sha256(manifest).hexdigest(),
        )
    with pytest.raises(ValueError):
        discover_inputs(
            str(inside_source),
            str(outside_progress),
            input_root,
            no_budget_check,
            source_sha256,
            manifest_sha256,
        )


def test_auto_scan_does_not_follow_symlink_outside_input_root(tmp_path):
    discover_inputs = discovery_function()
    input_root = tmp_path / "input"
    outside = tmp_path / "outside"
    manifest = b'{"format":"guandan-source-v1","files":{}}\n'
    write_source_directory(outside, manifest)
    input_root.mkdir(parents=True)
    link = input_root / "datasets" / "owner" / "name" / "versions" / "latest"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this Windows host")

    with pytest.raises(ValueError):
        discover_inputs(
            "",
            None,
            input_root,
            no_budget_check,
            "0" * 64,
            hashlib.sha256(manifest).hexdigest(),
        )


def test_auto_scan_invokes_check_budget_and_aborts_when_budget_expires(tmp_path):
    discover_inputs = discovery_function()
    input_root = tmp_path / "input"
    input_root.mkdir(parents=True)
    (input_root / "one-entry").write_text("scan me\n", encoding="utf-8")
    calls = []

    def check_budget():
        calls.append(True)
        raise TimeoutError("discovery budget exhausted")

    with pytest.raises(TimeoutError, match="discovery budget exhausted"):
        discover_inputs(
            "",
            None,
            input_root,
            check_budget,
            "0" * 64,
            "0" * 64,
        )
    assert calls


def test_auto_scan_enforces_max_entries(tmp_path):
    discover_inputs = discovery_function()
    input_root = tmp_path / "input"
    noise = input_root / "datasets" / "owner" / "name" / "versions" / "1" / "noise"
    noise.mkdir(parents=True)
    for index in range(8):
        (noise / f"entry-{index}.txt").write_text("not an input\n", encoding="utf-8")

    with pytest.raises(ValueError):
        discover_inputs(
            "",
            None,
            input_root,
            no_budget_check,
            "0" * 64,
            "0" * 64,
            max_entries=3,
        )


