from __future__ import annotations

import json
from pathlib import Path

from guandan_marvel.kaggle.algorithm_compare import _decision, _paired_rule_difference
from guandan_marvel.kaggle.m09_selection_delivery import (
    SELECTION_PACKAGE_ID, build_selection_delivery, build_selection_manifest,
    validate_selection_delivery,
)

ROOT = Path(__file__).resolve().parents[2]
IPPO = ROOT / "runs/M09_full_training/remote/r2_ippo_pilot_20261013T051332Z/extracted-output"
VRPO = ROOT / "runs/M09_full_training/remote/r2_vrpo_pilot_20261013T062752Z/extracted-output"


def test_decision_gate_follows_locked_direct_rule_and_ippo_fallback():
    assert _decision({"low": 0.51, "high": 0.70}, {"ci95_low": -0.1, "ci95_high": 0.1})["selected_algorithm"] == "vrpo"
    assert _decision({"low": 0.30, "high": 0.49}, {"ci95_low": -0.1, "ci95_high": 0.1})["selected_algorithm"] == "ippo"
    assert _decision({"low": 0.45, "high": 0.55}, {"ci95_low": 0.01, "ci95_high": 0.2})["status"] == "select_vrpo_paired_rule"
    assert _decision({"low": 0.45, "high": 0.55}, {"ci95_low": -0.2, "ci95_high": -0.01})["status"] == "select_ippo_paired_rule"
    fallback = _decision({"low": 0.45, "high": 0.55}, {"ci95_low": -0.1, "ci95_high": 0.1})
    assert fallback["status"] == "retain_ippo_inconclusive"
    assert fallback["selected_algorithm"] == "ippo"
    assert fallback["final_holdout_used"] is False


def test_real_paired_rule_anchors_have_exact_65x4_coverage():
    result = _paired_rule_difference(
        ippo_path=IPPO / "evaluation_games.jsonl",
        vrpo_path=VRPO / "evaluation_games.jsonl",
        ippo_sha256="b00a2c178e8b3e99d80418ae283a785285639b8172a0d96232beef7233b9d41c",
        vrpo_sha256="db8aa71f22ec88091acb189e6c57df2ac9ac0c1d18e3a931f4d7a1ec2cca9853",
    )
    assert result["clusters"] == 65
    assert result["paired_rows"] == 260
    assert -1.0 <= result["mean_difference_vrpo_minus_ippo"] <= 1.0


def test_selection_manifest_binds_real_checkpoints_and_locked_plan(tmp_path):
    plan = json.loads((ROOT / "configs/marvel_m09_full_plan.json").read_text(encoding="utf-8"))
    manifest = build_selection_manifest(
        ippo_checkpoint=IPPO / "evaluation_checkpoints/iter000100.pt",
        vrpo_checkpoint=VRPO / "evaluation_checkpoints/iter000100.pt",
        ippo_anchor=IPPO / "evaluation_games.jsonl",
        vrpo_anchor=VRPO / "evaluation_games.jsonl",
        plan=plan,
    )
    assert manifest["package_id"] == SELECTION_PACKAGE_ID
    assert manifest["algorithms"]["ippo"]["sha256"] == "b00a2c178e8b3e99d80418ae283a785285639b8172a0d96232beef7233b9d41c"
    assert manifest["algorithms"]["vrpo"]["sha256"] == "db8aa71f22ec88091acb189e6c57df2ac9ac0c1d18e3a931f4d7a1ec2cca9853"
    assert manifest["selection_dataset"]["base_deals"] == 65
    assert manifest["gate"]["final_holdout_forbidden"] is True


def test_selection_delivery_builds_and_validates(tmp_path):
    output = tmp_path / SELECTION_PACKAGE_ID
    build_selection_delivery(
        ROOT, output, ippo_output_root=IPPO, vrpo_output_root=VRPO,
        allow_dirty=True,
    )
    result = validate_selection_delivery(output)
    assert result["valid"] is True
    assert result["package_id"] == SELECTION_PACKAGE_ID
    assert result["ippo_checkpoint_sha256"] == "b00a2c178e8b3e99d80418ae283a785285639b8172a0d96232beef7233b9d41c"
    assert result["vrpo_checkpoint_sha256"] == "db8aa71f22ec88091acb189e6c57df2ac9ac0c1d18e3a931f4d7a1ec2cca9853"
    assert (output / "dataset/m09_algorithm_selection.json").is_file()
    assert (output / "guandan_marvel_m09_algorithm_selection.ipynb").is_file()
