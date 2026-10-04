from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import pytest

from guandan.evaluation_stats import summarize_games, wilson_interval


def game(opponent="random", group=0, rotation=0, win=True):
    candidate = rotation % 2
    winner = candidate if win else 1 - candidate
    return {
        "opponent": opponent, "deal_group": group, "seat_rotation": rotation,
        "policy_seat": rotation, "candidate_team": candidate,
        "done": True, "truncated": False,
        "ranking": [1, 3, 2, 4] if winner == 0 else [3, 1, 4, 2],
        "winner_team": winner,
        "rewards": [1 if seat % 2 == winner else -1 for seat in range(4)],
        "team_rewards": [1 if team == winner else -1 for team in range(2)],
        "token_steps": 20 + rotation, "committed_steps": 10,
        "base_deal_sha256": hashlib.sha256(str(group).encode()).hexdigest(),
        "seed": 100 + group,
    }


@pytest.mark.parametrize("wins,expected", [
    (0, (0.0, 0.2775327998628892)),
    (10, (0.7224672001371107, 1.0)),
    (5, (0.236593090512564, 0.763406909487436)),
])
def test_wilson_handcomputed_zero_all_half_wins(wins, expected):
    assert wilson_interval(wins, 10) == pytest.approx(expected, abs=1e-14)


def test_wilson_known_nist_example_and_custom_confidence():
    # 4/20, 90%: Wilson (not Clopper-Pearson's wider exact interval).
    assert wilson_interval(4, 20, 0.9) == pytest.approx((0.0931180165812109, 0.3783766746860239))
    narrow = wilson_interval(4, 20, 0.8)
    wide = wilson_interval(4, 20, 0.99)
    assert wide[0] < narrow[0] < narrow[1] < wide[1]


@pytest.mark.parametrize("wins,total", [(-1, 10), (11, 10), (0, 0), (0, -1), (True, 10), (1, True), (1.0, 10), (1, 10.0), ("1", 10)])
def test_wilson_rejects_invalid_counts(wins, total):
    with pytest.raises(ValueError):
        wilson_interval(wins, total)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1, float("nan"), float("inf"), True, "0.95"])
def test_wilson_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        wilson_interval(1, 2, confidence)


def test_wilson_near_one_confidence_is_finite_and_large_count_is_stable():
    low, high = wilson_interval(1, 2, math.nextafter(1.0, 0.0))
    assert 0 <= low < 0.5 < high <= 1
    assert all(math.isfinite(x) for x in (low, high))
    assert wilson_interval(500_000_000, 1_000_000_000) == pytest.approx((0.499969010248444, 0.500030989751556))


def test_48_game_three_opponent_four_group_report_without_cluster_inflation():
    records = [game(opponent, group, rotation, rotation < 2)
               for opponent in ("random", "rule", "snapshot")
               for group in range(4) for rotation in range(4)]
    before = deepcopy(records)
    result = summarize_games(records)
    assert records == before
    assert result["overall"]["completed"] == 48
    assert result["overall"]["wins"] == result["overall"]["losses"] == 24
    assert result["overall"]["confidence"]["independent_deal_groups"] == 4
    assert result["overall"]["opponent_deal_groups"] == 12
    assert result["overall"]["token_steps"] == sum(record["token_steps"] for record in records)
    assert result["overall"]["committed_steps"] == 480
    for summary in result["per_opponent"].values():
        assert summary["completed"] == 16
        assert summary["wins"] == summary["losses"] == 8
        assert summary["win_rate"] == 0.5 and summary["mean_team_reward"] == 0
        confidence = summary["confidence"]
        assert confidence["method"] == "Wilson95"
        assert confidence["interval"] == pytest.approx(wilson_interval(8, 16))
        assert confidence["independent_deal_groups"] == 4
        assert confidence["cluster_adjusted"] is False
        assert "not independent" in confidence["warning"]
        assert "do not infer policy strength" in confidence["warning"]
    json.dumps(result, allow_nan=False)


def test_rotation_and_group_breakdowns_are_distinct():
    records = [game(group=10, rotation=r, win=r == 0) for r in range(4)]
    records += [game(group=20, rotation=r, win=r != 0) for r in range(4)]
    summary = summarize_games(records)["per_opponent"]["random"]
    assert set(summary["by_rotation"]) == {"0", "1", "2", "3"}
    assert set(summary["by_deal_group"]) == {"10", "20"}
    assert all(row["completed"] == 2 and row["wins"] == 1 for row in summary["by_rotation"].values())
    assert summary["by_deal_group"]["10"]["wins"] == 1
    assert summary["by_deal_group"]["20"]["wins"] == 3
    assert summary["by_deal_group"]["10"]["rotations"] == [0, 1, 2, 3]
    assert summary["confidence"]["independent_deal_groups"] == 2


def test_fewer_groups_and_partial_rotations_are_not_fabricated():
    result = summarize_games([game(group=9, rotation=1, win=True), game(group=9, rotation=3, win=False)])
    summary = result["per_opponent"]["random"]
    assert summary["completed"] == 2 and summary["deal_groups"] == 1
    assert summary["confidence"]["independent_deal_groups"] == 1
    assert summary["by_rotation"]["0"]["completed"] == 0
    assert summary["by_rotation"]["0"]["win_rate"] is None
    assert summary["by_deal_group"]["9"]["rotations"] == [1, 3]
    assert result["overall"]["completed"] == 2
    assert result["per_opponent"]["rule"]["completed"] == 0


def test_repeated_base_deal_under_new_group_id_is_one_independent_cluster():
    a, b = game(group=0), game(group=1)
    b["base_deal_sha256"] = a["base_deal_sha256"].upper()
    summary = summarize_games([a, b])["per_opponent"]["random"]
    assert summary["deal_groups"] == 2
    assert summary["confidence"]["independent_deal_groups"] == 1


def test_win_reward_is_candidate_team_not_always_team_zero():
    record = game(rotation=3, win=True)
    summary = summarize_games([record])["per_opponent"]["random"]
    assert record["winner_team"] == 1
    assert summary["wins"] == 1 and summary["mean_team_reward"] == 1
    summary = summarize_games([game(rotation=2, win=False)])["per_opponent"]["random"]
    assert summary["losses"] == 1 and summary["mean_team_reward"] == -1


def test_empty_input_returns_zero_counts_and_no_interval_or_estimate():
    summary = summarize_games([])
    assert summary["overall"]["completed"] == 0
    assert summary["overall"]["win_rate"] is None
    assert summary["overall"]["mean_team_reward"] is None
    assert summary["overall"]["confidence"]["interval"] is None
    assert summary["overall"]["confidence"]["independent_deal_groups"] == 0
    assert set(summary["per_opponent"]) == {"random", "rule", "snapshot"}
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("done", False), ("done", 1), ("truncated", True), ("truncated", 0),
    ("ranking", [1, 1, 3, 4]), ("ranking", [0, 1, 2, 3]),
    ("ranking", [1, 2, 3]), ("ranking", [True, 3, 2, 4]),
    ("ranking", [3, 1, 4, 2]), ("ranking", [1.0, 3, 2, 4]),
    ("winner_team", 1), ("winner_team", 2), ("winner_team", True),
    ("candidate_team", 1), ("policy_seat", 1), ("seat_rotation", 4),
    ("rewards", [1, 1, 1, -1]), ("rewards", [0, -1, 1, -1]),
    ("rewards", [float("nan"), -1, 1, -1]), ("rewards", [True, -1, 1, -1]),
    ("rewards", [1, -1, 1]), ("team_rewards", [-1, 1]),
    ("team_rewards", [1, -1, 1]), ("team_rewards", [1, float("inf")]),
    ("opponent", "unknown"), ("opponent", []), ("deal_group", "0"),
    ("seed", True), ("token_steps", -1), ("token_steps", 1.5),
    ("committed_steps", 21), ("committed_steps", -1),
    ("base_deal_sha256", "bad-hash"), ("base_deal_sha256", "g" * 64),
])
def test_malformed_record_is_rejected(field, value):
    record = game()
    record[field] = value
    with pytest.raises(ValueError, match="record 0"):
        summarize_games([record])


@pytest.mark.parametrize("records", [{}, (), [None], ["game"], [{}]])
def test_invalid_record_container_and_missing_fields(records):
    with pytest.raises(ValueError):
        summarize_games(records)


def test_duplicate_key_rejected_even_with_different_result_or_seed():
    first = game()
    duplicate = game(win=False)
    duplicate["seed"] += 1
    with pytest.raises(ValueError, match="duplicate"):
        summarize_games([first, duplicate])


def test_different_opponents_are_not_duplicate_games():
    result = summarize_games([game("random"), game("rule"), game("snapshot")])
    assert result["overall"]["completed"] == 3
    assert result["overall"]["confidence"]["independent_deal_groups"] == 1


def test_same_group_rotations_require_matching_base_deal_hash():
    a, b = game(rotation=0), game(rotation=1)
    b["base_deal_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="different base-deal hashes"):
        summarize_games([a, b])


def test_record_order_does_not_change_summary():
    records = [game(group=g, rotation=r, win=r % 2 == 0) for g in (7, 2) for r in range(4)]
    assert summarize_games(records) == summarize_games(list(reversed(records)))
