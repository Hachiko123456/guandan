"""Phase-3 actor-fuse A/B harness: isolation guarantees and cheap runnability.

The formal Control/Treatment run from the M09 v1.3 warm start is NOT executed in
this phase.  These tests validate the harness itself (configuration isolation,
refusals, report integrity) and run it end-to-end in ``--smoke`` mode on a
synthetic v1.3 policy plus the reduced C++ smoke game.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from guandan_marvel.config import MACRO_FUSE_MODEL_ARCHITECTURE
from guandan_marvel.training.checkpoints import TrainingCheckpointStore, canonical_json_hash
from guandan_marvel.training.smoke import CppSmokeGame
from scripts.run_marvel_v14_phase3_fuse_ab import (
    AB_SCHEMA, CONTROL_ARCHITECTURE, DEFAULT_DIMS, TREATMENT_ARCHITECTURE,
    _arm_difference, _synthetic_warm_start, run_fuse_ab,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_marvel_v14_phase3_fuse_ab.py"


@pytest.fixture()
def synthetic_warm_start(tmp_path):
    game = CppSmokeGame(max_game_horizon=512)
    path = _synthetic_warm_start(tmp_path / "synthetic-v13.pt", game)
    payload = TrainingCheckpointStore.load(path, device="cpu")
    return payload, game


def test_ab_harness_isolates_architecture_only(tmp_path, synthetic_warm_start):
    payload, game = synthetic_warm_start
    assert payload["schema"] == "guandan-marvel-evaluation-checkpoint-v1"
    assert payload["architecture"] == CONTROL_ARCHITECTURE

    report = run_fuse_ab(
        warm_start_payload=payload, game=game, output_dir=tmp_path / "ab",
        iterations=2, seed=4242, amp_enabled=False, dims=DEFAULT_DIMS, run_tag="smoke",
    )
    assert report["schema"] == AB_SCHEMA and report["executed"] is True
    # The only configured difference is the architecture under test.
    assert report["arm_configuration_differences"] == ["model.architecture"]
    assert report["control"]["architecture"] == CONTROL_ARCHITECTURE
    assert report["treatment"]["architecture"] == TREATMENT_ARCHITECTURE
    assert report["control"]["data_manifest_hash"] == report["treatment"]["data_manifest_hash"]
    assert report["control"]["seed"] == report["treatment"]["seed"] == 4242
    assert report["amp_enabled"] is False
    assert report["isolation"] == {
        "same_warm_start": True,
        "same_seed": True,
        "same_data_manifest": True,
        "amp_off": True,
        "resume_from_control_supported": False,
    }
    # Treatment carries exactly the fuse parameters.
    assert report["parameter_delta"] == (
        report["treatment"]["parameter_count"] - report["control"]["parameter_count"]
    )
    assert report["parameter_delta"] > 0
    dim = DEFAULT_DIMS["feature_dim"]
    layers = DEFAULT_DIMS["fuse_layers"]
    expected = 4 * dim * dim + dim + layers * (dim + 3 * dim * dim) + dim + dim * dim
    assert report["parameter_delta"] == expected
    # Both arms migrated from the same v1.3 policy; only the treatment gained fuse.
    assert report["control"]["warm_start_migration"]["fuse_migrated"] is False
    assert report["treatment"]["warm_start_migration"]["fuse_migrated"] is True
    assert report["treatment"]["warm_start_migration"]["source_architecture"] == "v1.3"
    # Metrics are finite for every iteration of both arms.
    for arm in (report["control"], report["treatment"]):
        assert arm["iterations"] == 2
        assert arm["fuse_layers"] == DEFAULT_DIMS["fuse_layers"]
        for snapshot in arm["metrics"]:
            assert snapshot["macro/pg_loss"] is not None
    # The emitted report self-hashes.
    assert report["hash"] == canonical_json_hash({k: v for k, v in report.items() if k != "hash"})
    saved = json.loads((tmp_path / "ab" / "ab_report.json").read_text(encoding="utf-8"))
    assert saved["hash"] == report["hash"]
    # Checkpoint identity recorded by the harness matches the files on disk.
    import hashlib

    for arm in (report["control"], report["treatment"]):
        path = Path(arm["checkpoint_path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == arm["checkpoint_sha256"]


def test_ab_harness_refuses_a_non_v13_warm_start(tmp_path, synthetic_warm_start):
    payload, game = synthetic_warm_start
    forged = dict(payload)
    forged["architecture"] = MACRO_FUSE_MODEL_ARCHITECTURE
    with pytest.raises(ValueError, match="requires the shared M09 v1.3"):
        run_fuse_ab(
            warm_start_payload=forged, game=game, output_dir=tmp_path / "bad",
            iterations=1, seed=1, amp_enabled=False, dims=DEFAULT_DIMS, run_tag="bad",
        )


def test_ab_arm_difference_detector_flags_any_other_change():
    control = {"model": {"architecture": "v1.3", "feature_dim": 24}, "seed": 1, "run_name": "a"}
    treatment = {"model": {"architecture": "v1.4-macro-fuse", "feature_dim": 24}, "seed": 1, "run_name": "b"}
    assert _arm_difference(control, treatment) == ["model.architecture"]
    treatment["seed"] = 2
    assert set(_arm_difference(control, treatment)) == {"model.architecture", "seed"}


@pytest.mark.slow
def test_ab_harness_cli_smoke_is_runnable(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--smoke", "--output-dir", str(tmp_path / "cli"),
            "--iterations", "2", "--seed", "4242",
        ],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["smoke"] is True
    assert summary["arm_configuration_differences"] == ["model.architecture"]
    assert summary["parameter_delta"] > 0
    assert Path(summary["report"]).is_file()
