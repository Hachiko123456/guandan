# SPDX-License-Identifier: MIT
"""Cluster-aware GuanDan evaluation statistics."""
from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence

CLUSTER_STATS_SCHEMA = "guandan-marvel-clustered-summary-v1"
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _critical95(df: int) -> float:
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if df in _T975:
        return _T975[df]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960


def clustered_mean_ci95(cluster_values: Sequence[float]) -> dict[str, object]:
    values = [float(value) for value in cluster_values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("cluster values must be finite")
    if not values:
        return {"method": "deal-cluster-t95", "clusters": 0, "mean": None, "low": None, "high": None, "conclusive": False}
    estimate = mean(values)
    if len(values) < 2:
        return {"method": "deal-cluster-t95", "clusters": len(values), "mean": estimate, "low": None, "high": None, "conclusive": False}
    standard_error = stdev(values) / math.sqrt(len(values))
    margin = _critical95(len(values) - 1) * standard_error
    return {
        "method": "deal-cluster-t95",
        "clusters": len(values),
        "mean": estimate,
        "standard_error": standard_error,
        "low": max(0.0, estimate - margin),
        "high": min(1.0, estimate + margin),
        "conclusive": True,
    }


def _rate(values: Sequence[bool]) -> float | None:
    return sum(bool(value) for value in values) / len(values) if values else None


def _validate_rotation_clusters(records: Sequence[Mapping[str, object]]) -> tuple[dict[tuple[str, str], list[Mapping[str, object]]], list[str]]:
    clusters: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    errors: list[str] = []
    seen: set[tuple[str, str, int]] = set()
    for record in records:
        opponent = str(record.get("opponent_id", ""))
        deal_id = str(record.get("deal_id", ""))
        rotation = record.get("seat_rotation")
        if isinstance(rotation, bool) or rotation not in range(4):
            errors.append(f"invalid rotation for {opponent}/{deal_id}: {rotation!r}")
            continue
        key = (opponent, deal_id, int(rotation))
        if key in seen:
            errors.append(f"duplicate game record {key}")
            continue
        seen.add(key)
        clusters.setdefault((opponent, deal_id), []).append(record)
    for key, group in clusters.items():
        rotations = sorted(int(record["seat_rotation"]) for record in group)
        hashes = {str(record.get("base_deal_sha256")) for record in group}
        if rotations != [0, 1, 2, 3]:
            errors.append(f"cluster {key} rotations are {rotations}, expected [0, 1, 2, 3]")
        if len(hashes) != 1:
            errors.append(f"cluster {key} does not share one base deal")
    return clusters, errors


def _summarize_subset(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    complete = [record for record in records if record.get("status") == "complete"]
    wins = [bool(record["candidate_win"]) for record in complete]
    team02 = [bool(record["candidate_win"]) for record in complete if int(record["candidate_team"]) == 0]
    team13 = [bool(record["candidate_win"]) for record in complete if int(record["candidate_team"]) == 1]
    by_rotation = {
        str(rotation): _rate([bool(record["candidate_win"]) for record in complete if int(record["seat_rotation"]) == rotation])
        for rotation in range(4)
    }
    observed_rotation_rates = [value for value in by_rotation.values() if value is not None]
    runtime = sum(float(record.get("runtime_seconds", 0.0)) for record in records)
    clusters: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for record in complete:
        clusters.setdefault((str(record["opponent_id"]), str(record["deal_id"])), []).append(record)
    cluster_means = [
        mean(float(bool(record["candidate_win"])) for record in group)
        for group in clusters.values()
        if sorted(int(record["seat_rotation"]) for record in group) == [0, 1, 2, 3]
    ]
    ci = clustered_mean_ci95(cluster_means)
    outcomes = [str(record["outcome_class"]) for record in complete]
    return {
        "games": len(records),
        "completed_games": len(complete),
        "complete": len(complete) == len(records),
        "eval/win_rate": _rate(wins),
        "eval/clustered_ci95_low": ci["low"],
        "eval/clustered_ci95_high": ci["high"],
        "eval/cluster_count": ci["clusters"],
        "eval/team02_win_rate": _rate(team02),
        "eval/team13_win_rate": _rate(team13),
        "eval/seat_rotation_gap": max(observed_rotation_rates) - min(observed_rotation_rates) if observed_rotation_rates else None,
        "eval/mean_team_reward": mean(float(record["candidate_team_reward"]) for record in complete) if complete else None,
        "eval/double_down_rate": outcomes.count("DOUBLE_DOWN") / len(outcomes) if outcomes else None,
        "eval/head_third_rate": outcomes.count("HEAD_THIRD") / len(outcomes) if outcomes else None,
        "eval/head_last_rate": outcomes.count("HEAD_LAST") / len(outcomes) if outcomes else None,
        "eval/illegal_count": sum(int(record.get("illegal_count", 0)) for record in records),
        "eval/truncated_count": sum(int(record.get("truncated_count", 0)) for record in records),
        "eval/games_per_second": len(complete) / runtime if runtime > 0 else None,
        "by_rotation": by_rotation,
        "clustered_ci95": ci,
    }


def summarize_records(records: Sequence[Mapping[str, object]], *, expected_opponents: Iterable[str] | None = None) -> dict[str, object]:
    rows = [dict(record) for record in records]
    clusters, coverage_errors = _validate_rotation_clusters(rows)
    opponent_ids = sorted(set(str(record.get("opponent_id")) for record in rows))
    if expected_opponents is not None:
        expected = sorted(str(value) for value in expected_opponents)
        missing = sorted(set(expected) - set(opponent_ids))
        extra = sorted(set(opponent_ids) - set(expected))
        if missing:
            coverage_errors.append(f"missing opponents: {missing}")
        if extra:
            coverage_errors.append(f"unexpected opponents: {extra}")
        opponent_ids = expected
    per_opponent = {
        opponent: _summarize_subset([record for record in rows if record.get("opponent_id") == opponent])
        for opponent in opponent_ids
    }
    overall = _summarize_subset(rows)
    overall["complete"] = bool(rows) and overall["complete"] and not coverage_errors
    return {
        "schema": CLUSTER_STATS_SCHEMA,
        "overall": overall,
        "per_opponent": per_opponent,
        "seat_rotation_coverage": {
            "expected_rotations": [0, 1, 2, 3],
            "cluster_count": len(clusters),
            "complete": not coverage_errors,
            "errors": coverage_errors,
        },
    }


__all__ = ["CLUSTER_STATS_SCHEMA", "clustered_mean_ci95", "summarize_records"]
