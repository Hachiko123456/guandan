"""Phase-1 baseline manifest hardening: no self-referential hashes, deterministic rebuild."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CREATED_AT = "2026-10-18T00:00:00+00:00"
OFFICIAL_MANIFEST = ROOT / "docs" / "marvel_v14" / "phase1_baseline_manifest.json"

# The exact, minimal set of frozen phase-1 snapshot inputs that phase 4 changes:
#   * the native reward engine and the context-assignment binding (C++),
#   * one untracked adapter module (candidate_win sign derivation),
#   * this test module itself,
#   * the rebuilt native extension binary.
PHASE4_DRIFT_WHITELIST = frozenset({
    "nanopokerlib/cpp/guandan/guandan.cpp",
    "nanopokerlib/cpp/bindings.cpp",
    "guandan_marvel/comparison/driver.py",
    "tests/marvel_training/test_marvel_v14_phase1_baseline.py",
    "nanopokerlib/_guandan_cpp.cp312-win_amd64.pyd",
})


def test_phase4_drift_whitelist_is_minimal_and_exact():
    assert len(PHASE4_DRIFT_WHITELIST) == 5
    assert "nanopokerlib/cpp/guandan/guandan.cpp" in PHASE4_DRIFT_WHITELIST
    assert "nanopokerlib/cpp/bindings.cpp" in PHASE4_DRIFT_WHITELIST


def _load_builder():
    path = ROOT / "scripts" / "build_marvel_v14_phase1_baseline.py"
    spec = importlib.util.spec_from_file_location("marvel_v14_phase1_baseline_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build(builder, destination: Path) -> None:
    rc = builder.main([
        "--manifest", str(destination / "manifest.json"),
        "--report", str(destination / "report.md"),
        "--probe-dir", str(destination / "probe"),
        "--created-at", CREATED_AT,
    ])
    assert rc == 0


def test_generated_output_paths_are_excluded_from_untracked_snapshot():
    builder = _load_builder()
    assert builder._is_excluded(
        "docs/marvel_v14/phase1_baseline_manifest.json",
        ("docs/marvel_v14/phase1_baseline_manifest.json",),
    )
    assert builder._is_excluded(
        "docs/marvel_v14/phase1_baseline_report.md",
        ("docs/marvel_v14/phase1_baseline_report.md",),
    )
    assert builder._is_excluded(
        "runs/marvel_v14_phase1/probe_inputs.pt",
        ("runs/marvel_v14_phase1",),
    )
    assert not builder._is_excluded(
        "docs/marvel_v14_plan_verification.md",
        ("runs/marvel_v14_phase1", "docs/marvel_v14/phase1_baseline_manifest.json"),
    )


def test_manifest_verifier_detects_a_tampered_declared_input(tmp_path):
    builder = _load_builder()
    target = tmp_path / "frozen.txt"
    target.write_text("frozen", encoding="utf-8")
    manifest = {
        "snapshot_inputs": {
            "files": [{
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": builder._sha256_file(target),
            }],
        },
    }
    assert builder.verify_manifest_hashes(manifest, root=ROOT)["ok"]
    target.write_text("tampered", encoding="utf-8")
    verification = builder.verify_manifest_hashes(manifest, root=ROOT)
    assert not verification["ok"]
    assert any("sha256 mismatch" in failure for failure in verification["failures"])


def test_fixed_created_at_rebuilds_identical_artifacts(tmp_path):
    builder = _load_builder()
    _build(builder, tmp_path / "one")
    _build(builder, tmp_path / "two")
    one = tmp_path / "one"
    two = tmp_path / "two"
    assert (one / "manifest.json").read_bytes() == (two / "manifest.json").read_bytes()
    assert (one / "report.md").read_bytes() == (two / "report.md").read_bytes()
    assert (one / "probe" / "probe_inputs.pt").read_bytes() == (two / "probe" / "probe_inputs.pt").read_bytes()
    assert (one / "probe" / "probe_outputs.pt").read_bytes() == (two / "probe" / "probe_outputs.pt").read_bytes()
    assert (one / "probe" / "macro_probe.json").read_bytes() == (two / "probe" / "macro_probe.json").read_bytes()

    first = json.loads((one / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((two / "manifest.json").read_text(encoding="utf-8"))
    assert first["created_at"] == second["created_at"] == CREATED_AT
    assert first["snapshot_inputs"] == second["snapshot_inputs"]
    assert first["snapshot_inputs_hash"] == second["snapshot_inputs_hash"]
    assert first["hash"] == second["hash"]
    assert (
        first["probe"]["model_output_content_sha256"]
        == second["probe"]["model_output_content_sha256"]
    )
    assert first["probe"]["inputs_sha256"] == second["probe"]["inputs_sha256"]

    result = json.loads((one / "probe" / "baseline_build_result.json").read_text(encoding="utf-8"))
    assert result["manifest"]["sha256"] == builder._sha256_file(one / "manifest.json")
    assert result["verification"]["ok"], result["verification"]["failures"]
    verification = builder.verify_manifest_hashes(first, root=ROOT, manifest_dir=one)
    assert verification["ok"], verification["failures"]
    assert verification["checked_count"] >= 50


def test_official_manifest_has_no_self_references():
    if not OFFICIAL_MANIFEST.is_file():
        pytest.skip("official phase-1 manifest has not been built in this workspace")
    manifest = json.loads(OFFICIAL_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == "guandan-marvel-v14-phase1-baseline-manifest-v2"
    untracked = {
        entry["path"]
        for entry in manifest["snapshot_inputs"]["git"]["untracked_source_files"]
    }
    assert "docs/marvel_v14/phase1_baseline_manifest.json" not in untracked
    assert "docs/marvel_v14/phase1_baseline_report.md" not in untracked
    assert not any(path.startswith("runs/marvel_v14_phase1/") for path in untracked)
    outputs = manifest["generated_outputs"]
    assert outputs["manifest"]["path"].endswith("phase1_baseline_manifest.json")
    assert "sha256" not in outputs["manifest"]
    builder = _load_builder()
    verification = builder.verify_manifest_hashes(
        manifest, root=ROOT, manifest_dir=OFFICIAL_MANIFEST.parent,
    )
    # The phase-1 manifest is frozen historical evidence and is never rewritten.
    # Phase 4 legitimately changes the rank-point reward engine and the
    # context-assignment binding, which touch exactly these source/binary inputs.
    # Every *other* snapshot input must still match byte-for-byte.
    drifted: set[str] = set()
    unexpected: list[str] = []
    root_prefix = str(ROOT).replace("\\", "/") + "/"
    for failure in verification["failures"]:
        if "sha256 mismatch for " in failure:
            path = failure.split("sha256 mismatch for ", 1)[1].split(": declared=", 1)[0].replace("\\", "/")
            if path.startswith(root_prefix):
                path = path[len(root_prefix):]
            drifted.add(path)
        else:
            unexpected.append(failure)
    assert not unexpected, unexpected
    assert drifted == PHASE4_DRIFT_WHITELIST, (drifted, PHASE4_DRIFT_WHITELIST)
    assert verification["checked_count"] >= 60
    from nanopokerlib.guandan import BACKEND_BUILD_IDENTITY
    assert BACKEND_BUILD_IDENTITY["source_matches_loaded_binary"] is True
