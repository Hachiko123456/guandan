"""Gate A delivery builder/verifier tests (stage 6).

The decisive checks:

* the pre-declared constants are the ones ``docs/dmc_prompts/06_gate_a_kaggle.md``
  fixes (9 h budget, four 512-game lanes, 75/60/90 gates, the frozen m09 hash);
* a synthetic delivery builds and verifies end to end, including the m09
  identity chain (checkpoint-recorded identity -> shipped binary -> shipped
  native sources -> plan);
* the verifier is a real second opinion: a drifted gate table, a drifted
  budget, a drifted config or a tampered top-level plan are all rejected;
* the builder refuses a runtime root that does not match the frozen
  checkpoint's recorded ``extension_path``.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import torch

from scripts import build_dmc_gate_a_delivery as builder
from scripts import verify_dmc_gate_a_delivery as verifier

REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOT = "/kaggle/working/guandan-marvel-source"
EXTENSION = "_guandan_cpp.cpython-313-x86_64-linux-gnu.so"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _native_sources() -> dict[str, bytes]:
    return {
        "nanopokerlib/cpp/bindings.cpp": b"// bindings\n",
        "nanopokerlib/cpp/core/types.hpp": b"// types\n",
        "nanopokerlib/cpp/guandan/guandan.cpp": b"// guandan\n",
        "nanopokerlib/cpp/guandan/guandan.hpp": b"// guandan header\n",
    }


def _manifest_hash(sources: dict[str, bytes]) -> str:
    files = [
        {"path": name, "bytes": len(sources[name]), "sha256": hashlib.sha256(sources[name]).hexdigest()}
        for name in sorted(sources)
    ]
    payload = {"schema": "guandan-native-build-identity-v1", "files": files}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_m09_source(tmp_path: Path) -> Path:
    """A frozen-looking m09 dataset: checkpoint + anchor + runtime + source zip."""
    source = tmp_path / "m09-source"
    (source / "runtime").mkdir(parents=True)
    binary = source / "runtime" / EXTENSION
    binary.write_bytes(b"\x7fELF fake native runtime\n")
    sources = _native_sources()
    manifest = _manifest_hash(sources)
    identity = {
        "schema": "guandan-marvel-backend-identity-v1",
        "backend": "cpp",
        "module": "guandan_marvel.cpp_game",
        "game_identifier": "GuandanMARVELV2Cpp",
        "protocol_version": "GD-ACTION-0.2-MARVEL|GD-ENV-0.2-MARVEL|GD-ENCODING-0.2-MARVEL",
        "implementation": {
            "schema": "guandan-native-build-identity-v1",
            "extension_module": "nanopokerlib._guandan_cpp",
            "extension_path": f"{SOURCE_ROOT}/nanopokerlib/{EXTENSION}",
            "extension_filename": EXTENSION,
            "extension_sha256": _sha(binary),
            "embedded_source_sha256": manifest,
            "compiler": "13.3.0",
            "cxx_standard": 202002,
            "python_implementation": "CPython",
            "python_version": "3.13.15",
            "python_cache_tag": "cpython-313",
            "current_source_manifest": {"schema": "guandan-native-build-identity-v1", "hash": manifest},
            "source_matches_loaded_binary": True,
        },
    }
    identity["hash"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in identity.items() if k != "hash"},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    torch.save(
        {
            "schema": "guandan-marvel-training-checkpoint-v5",
            "algorithm": "ippo",
            "architecture": "v1.3",
            "iteration": 2000,
            "backend_identity": identity,
            "agent_state_dict": {"w": torch.zeros(1)},
        },
        source / "ippo_iter002000_final_candidate.pt",
    )
    torch.save({"algorithm": "ippo", "metadata": {"protocol_versions": {}}},
               source / "legacy_v1_anchor.pt")
    with zipfile.ZipFile(source / "guandan-marvel-source.zip", "w") as bundle:
        for name, data in sources.items():
            bundle.writestr(f"guandan-marvel-source/{name}", data)
    return source


def _pin_synthetic(monkeypatch, source: Path) -> None:
    """Point the verifier's frozen-hash check at the synthetic candidate."""
    candidate_sha = _sha(source / "ippo_iter002000_final_candidate.pt")
    monkeypatch.setattr(verifier, "FROZEN_M09_SHA256", candidate_sha)
    monkeypatch.setattr(verifier, "FROZEN_M09_ITERATION", 2000)


# ---------------------------------------------------------------------------
# Pre-declared constants
# ---------------------------------------------------------------------------
def test_frozen_constants_match_the_prompt() -> None:
    assert verifier.FROZEN_M09_SHA256 == (
        "af0df1eaeb67c88e467ee78bf6b0494c49d000f5733d655ee81b40351d4ca9c8"
    )
    assert verifier.FROZEN_M09_ITERATION == 2000
    assert verifier.EXPECTED_BUDGETS["platform_timeout_seconds"] == 32_400  # 9 h
    assert verifier.EXPECTED_BUDGETS["train_seconds"] == 28_700
    assert (
        verifier.EXPECTED_BUDGETS["total_seconds"]
        + verifier.BUDGET_MARGIN_SECONDS
        <= verifier.EXPECTED_BUDGETS["platform_timeout_seconds"]
    )
    assert [row["kind"] for row in verifier.EXPECTED_LANES] == [
        "rule", "m09", "random", "legacy",
    ]
    assert [row["games"] for row in verifier.EXPECTED_LANES] == [512, 512, 512, 512]
    assert [row["rotations"] for row in verifier.EXPECTED_LANES] == [2, 4, 2, 2]
    assert verifier.EXPECTED_GATES["vs_rule"]["pass_min"] == 0.75
    assert verifier.EXPECTED_GATES["vs_rule"]["reference_gate_l_win_rate"] == 0.652
    assert verifier.EXPECTED_GATES["h2h_m09"]["pass_min"] == 0.60
    assert verifier.EXPECTED_GATES["vs_random"]["pass_min"] == 0.90
    assert verifier.EXPECTED_GATES["health"] == {"illegal": 0, "nonfinite": 0, "truncated": 0}
    assert verifier.EXPECTED_GATES["throughput"] == {"mode": "record_only"}
    assert verifier.EXPECTED_TOPOLOGY["collector_device"] == "cuda:1"
    assert verifier.EXPECTED_TOPOLOGY["learner_device"] == "cuda:0"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_synthetic_delivery_builds_and_verifies(tmp_path, monkeypatch) -> None:
    source = _make_m09_source(tmp_path)
    _pin_synthetic(monkeypatch, source)
    builder.build_delivery(
        REPO, tmp_path / "delivery", git_commit="TESTCOMMIT", m09_source=source,
    )
    record = verifier.verify(tmp_path / "delivery")
    assert record["status"] == "PASS"
    assert record["budget_total_seconds"] == verifier.EXPECTED_BUDGETS["total_seconds"]
    assert record["lanes"] == ["rule", "m09", "random", "legacy"]
    assert record["m09"]["python_version"] == "3.13.15"
    assert record["m09"]["extension_filename"] == EXTENSION
    assert record["overlay_files"] >= len(verifier.REQUIRED_OVERLAY)
    assert record["driver_files"] == len(verifier.REQUIRED_DRIVER)
    assert record["hash"] == verifier.canonical(
        {k: v for k, v in record.items() if k != "hash"}
    )


# ---------------------------------------------------------------------------
# Negative cases: the verifier is a second opinion
# ---------------------------------------------------------------------------
def test_verifier_rejects_drifted_gate_table(tmp_path, monkeypatch) -> None:
    source = _make_m09_source(tmp_path)
    _pin_synthetic(monkeypatch, source)
    drifted = json.loads(json.dumps(builder.GATES))
    drifted["vs_rule"]["pass_min"] = 0.70
    monkeypatch.setattr(builder, "GATES", drifted)
    builder.build_delivery(REPO, tmp_path / "d", git_commit="TESTCOMMIT", m09_source=source)
    with pytest.raises(verifier.QAFailure, match="gates drifted"):
        verifier.verify(tmp_path / "d")


def test_verifier_rejects_drifted_budget(tmp_path, monkeypatch) -> None:
    source = _make_m09_source(tmp_path)
    _pin_synthetic(monkeypatch, source)
    drifted = dict(builder.BUDGETS)
    drifted["train_seconds"] = 30_000
    monkeypatch.setattr(builder, "BUDGETS", drifted)
    builder.build_delivery(REPO, tmp_path / "d", git_commit="TESTCOMMIT", m09_source=source)
    with pytest.raises(verifier.QAFailure, match="budgets.train_seconds drifted"):
        verifier.verify(tmp_path / "d")


def test_verifier_rejects_drifted_m09_identity(tmp_path, monkeypatch) -> None:
    source = _make_m09_source(tmp_path)
    _pin_synthetic(monkeypatch, source)
    builder.build_delivery(REPO, tmp_path / "d", git_commit="TESTCOMMIT", m09_source=source)
    # A different "frozen" candidate hash must be refused outright.
    monkeypatch.setattr(verifier, "FROZEN_M09_SHA256", "0" * 64)
    with pytest.raises(verifier.QAFailure, match="frozen m09 candidate hash drifted"):
        verifier.verify(tmp_path / "d")


def test_verifier_rejects_tampered_top_level_plan(tmp_path, monkeypatch) -> None:
    source = _make_m09_source(tmp_path)
    _pin_synthetic(monkeypatch, source)
    builder.build_delivery(REPO, tmp_path / "d", git_commit="TESTCOMMIT", m09_source=source)
    plan_path = tmp_path / "d" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["deviations"].append("hand edit")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(verifier.QAFailure, match="top-level plan.json differs"):
        verifier.verify(tmp_path / "d")


def test_verifier_rejects_tampered_notebook(tmp_path, monkeypatch) -> None:
    source = _make_m09_source(tmp_path)
    _pin_synthetic(monkeypatch, source)
    builder.build_delivery(REPO, tmp_path / "d", git_commit="TESTCOMMIT", m09_source=source)
    notebook_path = tmp_path / "d" / verifier.NOTEBOOK
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"][1]["source"] = ["print('tampered')\n"]
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    with pytest.raises(verifier.QAFailure):
        verifier.verify(tmp_path / "d")


def test_builder_rejects_a_runtime_root_the_checkpoint_did_not_record(tmp_path, monkeypatch) -> None:
    source = _make_m09_source(tmp_path)
    monkeypatch.setattr(builder, "SOURCE_ROOT", "/kaggle/working/somewhere-else")
    with pytest.raises(ValueError, match="different runtime root"):
        builder.build_delivery(REPO, tmp_path / "d", git_commit="TESTCOMMIT", m09_source=source)


def test_builder_rejects_a_binary_that_does_not_match_the_identity(tmp_path) -> None:
    source = _make_m09_source(tmp_path)
    (source / "runtime" / EXTENSION).write_bytes(b"a different binary\n")
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        builder.build_delivery(REPO, tmp_path / "d", git_commit="TESTCOMMIT", m09_source=source)
