from __future__ import annotations

import json
from pathlib import Path

import pytest

from guandan_marvel.kaggle.budget import (
    build_completed_game_budget_contract, observe_completed_game_budget,
    validate_completed_game_budget_contract,
)
from guandan_marvel.kaggle.m09_delivery import (
    M09_ROLES, build_m09_delivery, build_m09_profiles, run_m09_safety_check,
    validate_m09_delivery, validate_m09_plan,
)


def test_completed_game_budget_is_exact_and_rejects_skew():
    contract = build_completed_game_budget_contract(
        comparison_id="m09-test", algorithm="ippo", target_completed_games=400,
        global_collect_batch_size=4, expected_iterations=100,
        selection_dataset_hash="a" * 64,
    )
    validate_completed_game_budget_contract(
        contract, expected_algorithm="ippo", global_collect_batch_size=4,
        target_iterations=100,
    )
    assert observe_completed_game_budget(contract, iteration=99, completed_games=396)["complete"] is False
    assert observe_completed_game_budget(contract, iteration=100, completed_games=400)["complete"] is True
    with pytest.raises(RuntimeError, match="not exact"):
        observe_completed_game_budget(contract, iteration=100, completed_games=396)
    with pytest.raises(ValueError, match="algorithm/profile mismatch"):
        validate_completed_game_budget_contract(contract, expected_algorithm="vrpo")


def test_m09_plan_locks_equal_ippo_vrpo_workloads_and_selection_identity():
    root = Path(__file__).resolve().parents[2]
    validated = validate_m09_plan(root)
    plan = validated["plan"]
    profiles = build_m09_profiles(plan, validated["locked_config"])
    assert tuple(profiles) == M09_ROLES
    ippo = profiles["ippo_pilot"]
    vrpo = profiles["vrpo_pilot"]
    assert ippo["training"] == vrpo["training"]
    assert ippo["ddp"] == vrpo["ddp"]
    assert ippo["evaluation"] == vrpo["evaluation"]
    assert ippo["algorithm"] == "ippo"
    assert vrpo["algorithm"] == "vrpo"
    assert ippo["completed_game_budget"]["target_completed_games"] == 400
    assert vrpo["completed_game_budget"]["target_completed_games"] == 400
    assert ippo["training"]["target_iterations"] == 100
    assert ippo["training"]["agent_merging_beta"] == 0.0
    assert ippo["training"]["checkpoint_keep_last"] == 2
    assert ippo["training"]["evaluation_keep_last"] == 2
    assert ippo["evaluation"]["expected_selection_dataset_hash"] == validated["selection_dataset_hash"]
    assert profiles["ippo_resume"]["expected_algorithm"] == "ippo"
    assert profiles["vrpo_resume"]["expected_algorithm"] == "vrpo"
    assert plan["delivery"]["main_full_generated"] is False
    assert plan["claims"]["final_holdout_run"] is False


def test_m09_safety_check_rejects_old_full_dataset_reuse_boundary():
    root = Path(__file__).resolve().parents[2]
    result = run_m09_safety_check(root, require_clean=False)
    assert result["valid"] is True
    assert result["m08_commit_is_ancestor"] is True
    assert result["forbidden_package_ids_absent"] is True
    assert result["package_id"] == "guandan-marvel-m09-algorithm-pilot-20261011-r2"
    assert result["main_full_generated"] is False
    assert result["final_holdout_generated"] is False
    assert result["remote_verified"] is False


@pytest.fixture(scope="module")
def m09_delivery(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path_factory.mktemp("m09-delivery") / "guandan-marvel-m09-algorithm-pilot-20261011-r2"
    build_m09_delivery(root, output, allow_dirty=True)
    return output


def test_m09_delivery_builds_fresh_role_datasets(m09_delivery):
    result = validate_m09_delivery(m09_delivery)
    assert result["valid"] is True
    assert result["old_full_dataset_reused"] is False
    assert result["remote_verified"] is False
    assert result["m09_accepted"] is False
    assert [row["role"] for row in result["datasets"]] == list(M09_ROLES)
    for role in M09_ROLES:
        root = m09_delivery / "datasets" / role
        assert (root / "guandan_marvel_dataset.json").is_file()
        assert (root / "guandan_marvel_bootstrap.py").is_file()
        assert (root / "guandan-marvel-source.zip").is_file()
        assert (root / "source_manifest.json").is_file()
        assert (root / "marvel_m09_full_plan.json").is_file()
        assert (root / "marvel_m08_locked_config.json").is_file()


def test_m09_dataset_claims_and_resume_binding_are_fail_closed(m09_delivery):
    ippo = json.loads((m09_delivery / "datasets/ippo_pilot/guandan_marvel_dataset.json").read_text(encoding="utf-8"))
    vrpo = json.loads((m09_delivery / "datasets/vrpo_pilot/guandan_marvel_dataset.json").read_text(encoding="utf-8"))
    ippo_resume = json.loads((m09_delivery / "datasets/ippo_resume/guandan_marvel_dataset.json").read_text(encoding="utf-8"))
    assert ippo["package_id"] == vrpo["package_id"]
    assert ippo["package_id"] not in {
        "guandan-marvel-t4x2-20261010", "guandan-marvel-t4x2-20261010-r2",
        "guandan-marvel-m09-algorithm-pilot-20261011",
    }
    assert ippo["claims"]["remote_verified"] is False
    assert ippo["claims"]["formal_training_complete"] is False
    assert ippo["claims"]["model_strength"] is False
    assert ippo["profiles"]["full"]["m09"]["shared_workload_hash"] == vrpo["profiles"]["full"]["m09"]["shared_workload_hash"]
    assert ippo_resume["recovery"]["required"] is True
    assert ippo_resume["profiles"]["resume"]["expected_algorithm"] == "ippo"
    assert ippo_resume["profiles"]["resume"]["resume"]["target_iterations"] == 100
