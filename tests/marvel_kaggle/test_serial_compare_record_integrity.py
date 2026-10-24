"""Configuration-bound records and true preflight completion contracts."""
import copy
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from serial_compare.common import build_deal_manifest, canonical
from serial_compare.evaluation import tasks_for, merge_records, validate_preflight_records, EvaluationError


MODELS = {"control_final": {"sha256": "a" * 64}, "p375": {"sha256": "b" * 64}}


def fixture():
    tasks = tasks_for(build_deal_manifest("preflight", count=1)["deals"], ("control_vs_375",))
    records = []
    for task in tasks:
        row = {"task_id": task["task_id"], "task_hash": canonical(task), "arm": task["arm"],
               "deal_id": task["deal_id"], "deal_seed": task["deal_seed"],
               "seat_rotation": task["assignment"], "candidate_team": task["assignment"] % 2,
               "temperature": 0.0, "evaluation_horizon": 1024, "base_deal_sha256": "c" * 64,
               "action_trace_sha256": "d" * 64, "candidate_checkpoint_sha256": "a" * 64,
               "opponent_checkpoint_sha256": "b" * 64, "status": "complete",
               "candidate_win": task["assignment"] == 0, "candidate_team_reward": 1.0 if task["assignment"] == 0 else -1.0,
               "ranking": [1, 2, 4, 3], "winner_team": 0, "outcome_class": "HEAD_LAST",
               "token_steps": 20, "committed_steps": 8, "illegal_count": 0, "truncated_count": 0}
        row["record_hash"] = canonical(row)
        records.append(row)
    return tasks, records


def resign(row):
    row["record_hash"] = canonical({k: v for k, v in row.items() if k != "record_hash"})


def test_valid_records_and_preflight_pass():
    tasks, records = fixture()
    assert len(validate_preflight_records(records, tasks, models=MODELS)) == 2


def test_same_ids_but_changed_rule_config_is_not_the_same_task():
    tasks, records = fixture()
    changed = copy.deepcopy(tasks)
    for task in changed:
        task["deal"]["level_rank"] = 15
        task["deal"]["previous_result"] = {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN", "tribute_cancelled": False}
    with pytest.raises(EvaluationError, match="task hash"):
        merge_records(records, [], tasks=changed, models=MODELS)


def test_changed_result_cannot_keep_old_record_hash():
    tasks, records = fixture()
    records[1].update(candidate_win=True, candidate_team_reward=1.0)
    with pytest.raises(EvaluationError, match="record hash"):
        merge_records(records, [], tasks=tasks, models=MODELS)


@pytest.mark.parametrize("changes", [
    {"winner_team": 1}, {"outcome_class": "DOUBLE_DOWN"}, {"ranking": [1, 1, 3, 4]},
    {"candidate_win": True, "candidate_team_reward": 3.0},
])
def test_even_resigned_terminal_result_must_match_ranking(changes):
    tasks, records = fixture()
    records[0].update(changes)
    resign(records[0])
    with pytest.raises(EvaluationError):
        merge_records(records, [], tasks=tasks, models=MODELS)


def test_truncated_games_persist_as_evidence_but_cannot_pass_preflight():
    tasks, records = fixture()
    for row in records:
        row.update(status="incomplete", truncated_count=1, token_steps=1024, committed_steps=0)
        resign(row)
    assert len(merge_records(records, [], tasks=tasks, models=MODELS)[0]) == 2
    with pytest.raises(EvaluationError, match="truncated"):
        validate_preflight_records(records, tasks, models=MODELS)


def test_duplicate_or_partial_records_do_not_pass_preflight():
    tasks, records = fixture()
    with pytest.raises(EvaluationError, match="duplicate"):
        validate_preflight_records([records[0], records[0]], tasks, models=MODELS)
    with pytest.raises(EvaluationError, match="incomplete"):
        validate_preflight_records(records, tasks, models=MODELS, ignored_partial_lines=[1])
