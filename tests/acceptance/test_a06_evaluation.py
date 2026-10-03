"""A06 evaluation acceptance: real fixed groups, rotations and baselines."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from guandan.evaluation import EVALUATION_SEED_BASE, EvaluationIncomplete, evaluate_profile
from guandan.execution_profiles import load_execution_profile

@pytest.mark.acceptance
def test_a06_local_fast_completes_48_real_terminal_games(tmp_path: Path):
    result = evaluate_profile(profile="local_fast", evidence_path=tmp_path / "evaluation.json")
    assert result.status == "complete"
    assert result.total_games == 48
    assert result.completed_games_per_opponent == {"random": 16, "rule": 16, "snapshot": 16}
    assert len(result.records) == 48
    assert all(record.done and record.terminal for record in result.records)
    assert {record.seat_rotation for record in result.records} == {0,1,2,3}
    assert all(sum(record.rewards) == 0 for record in result.records)
    assert all(set(record.ranking) == {1,2,3,4} for record in result.records)
    assert (tmp_path / "evaluation.json").is_file()

@pytest.mark.acceptance
def test_a06_repeatability_and_training_eval_seed_separation():
    a=evaluate_profile(profile="local_fast", seed=EVALUATION_SEED_BASE)
    b=evaluate_profile(profile="local_fast", seed=EVALUATION_SEED_BASE)
    first, second = a.as_dict(), b.as_dict()
    first.pop("runtime_seconds", None); second.pop("runtime_seconds", None)
    assert first == second
    training_seeds=set(range(10_000,10_000+48))
    assert not any(record.seed in training_seeds for record in a.records)

@pytest.mark.acceptance
def test_a06_profile_targets_are_loaded_not_hardcoded():
    profile=load_execution_profile(Path(__file__).resolve().parents[2],"local_fast")
    assert profile["derived"]["total_games_all_opponents"] == 48
    assert profile["config"]["evaluation"]["seat_rotations"] == [0,1,2,3]

@pytest.mark.acceptance
def test_a06_timeout_is_explicit():
    with pytest.raises(EvaluationIncomplete) as error:
        evaluate_profile(profile="local_fast", max_hours=1e-12)
    assert error.value.report["status"] == "incomplete"
