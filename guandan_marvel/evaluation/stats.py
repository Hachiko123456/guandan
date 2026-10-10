# SPDX-License-Identifier: MIT
"""Conservative deal-cluster statistics and exact evaluation-plan coverage."""
from __future__ import annotations

import math
from statistics import mean, variance
from typing import Iterable, Mapping, Sequence

CLUSTER_STATS_SCHEMA = "guandan-marvel-clustered-summary-v2"
CLUSTER_CI_POLICY_SCHEMA = "guandan-marvel-bounded-cluster-ci-v1"


def clustered_mean_ci95(cluster_values: Sequence[float]) -> dict[str, object]:
    """95% empirical-Bernstein interval for independent bounded deal means.

    Four seat rotations are averaged before this function is called. The
    finite-sample range term prevents all-win/all-loss samples from producing
    a degenerate [1,1]/[0,0] interval.
    """
    values = [float(value) for value in cluster_values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("cluster values must be finite and in [0,1]")
    if not values:
        return {"schema": CLUSTER_CI_POLICY_SCHEMA, "method": "bounded-empirical-Bernstein-95", "clusters": 0, "mean": None, "low": None, "high": None, "conclusive": False}
    estimate = mean(values)
    if len(values) < 2:
        return {"schema": CLUSTER_CI_POLICY_SCHEMA, "method": "bounded-empirical-Bernstein-95", "clusters": 1, "mean": estimate, "low": None, "high": None, "conclusive": False}
    sample_variance = variance(values)
    log_term = math.log(3.0 / 0.05)
    margin = math.sqrt(2.0 * sample_variance * log_term / len(values)) + 3.0 * log_term / len(values)
    return {
        "schema": CLUSTER_CI_POLICY_SCHEMA,
        "method": "bounded-empirical-Bernstein-95",
        "confidence": 0.95,
        "clusters": len(values),
        "mean": estimate,
        "sample_variance": sample_variance,
        "low": max(0.0, estimate - margin),
        "high": min(1.0, estimate + margin),
        "conclusive": True,
        "finite_sample_range_correction": 3.0 * log_term / len(values),
    }


def _rate(values: Sequence[bool]) -> float | None:
    return sum(bool(value) for value in values) / len(values) if values else None


def _finite(value: object, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum or maximum is not None and result > maximum:
        raise ValueError(f"{name} is outside [{minimum},{maximum}]")
    return result


def _validate_complete_record(record: Mapping[str, object]) -> None:
    if type(record.get("candidate_win")) is not bool:
        raise ValueError("candidate_win must be bool")
    for name in ("candidate_team", "winner_team", "seat_rotation"):
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be int")
    if int(record["candidate_team"]) not in (0, 1) or int(record["winner_team"]) not in (0, 1):
        raise ValueError("team IDs must be 0 or 1")
    _finite(record.get("candidate_team_reward"), "candidate_team_reward", minimum=-1.0, maximum=1.0)
    _finite(record.get("runtime_seconds"), "runtime_seconds", minimum=0.0)
    for name in ("token_steps", "committed_steps", "illegal_count", "truncated_count"):
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative int")
    if int(record["committed_steps"]) > int(record["token_steps"]):
        raise ValueError("committed_steps cannot exceed token_steps")


def _plan_expectations(plan: Mapping[str, object]) -> tuple[set[tuple[str, str, int]], dict[str, str | None], str]:
    if not isinstance(plan, Mapping) or not isinstance(plan.get("matrix"), list):
        raise ValueError("evaluation plan is malformed")
    opponents: dict[str, str | None] = {}
    for row in plan["matrix"]:
        if not isinstance(row, Mapping):
            raise ValueError("evaluation plan matrix row is malformed")
        opponent = str(row.get("id", ""))
        if not opponent or opponent in opponents:
            raise ValueError("evaluation plan opponent IDs must be unique")
        opponents[opponent] = None if row.get("checkpoint_sha256") is None else str(row["checkpoint_sha256"])
    deal_ids = plan.get("deal_ids")
    rotations = plan.get("seat_rotations")
    if not isinstance(deal_ids, list) or not isinstance(rotations, list):
        raise ValueError("evaluation plan deal/rotation lists are malformed")
    if len(set(map(str, deal_ids))) != len(deal_ids) or sorted(rotations) != [0, 1, 2, 3]:
        raise ValueError("evaluation plan deal IDs or rotations are invalid")
    expected = {(opponent, str(deal), int(rotation)) for opponent in opponents for deal in deal_ids for rotation in rotations}
    candidate = plan.get("candidate")
    if not isinstance(candidate, Mapping) or not candidate.get("checkpoint_sha256"):
        raise ValueError("evaluation plan candidate identity is missing")
    return expected, opponents, str(candidate["checkpoint_sha256"])


def _validate_records(
    records: Sequence[Mapping[str, object]],
    *,
    plan: Mapping[str, object] | None,
    expected_opponents: Iterable[str] | None,
) -> tuple[dict[tuple[str, str], list[Mapping[str, object]]], list[str], list[str]]:
    errors: list[str] = []
    clusters: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    seen: set[tuple[str, str, int]] = set()
    expected = None
    opponent_checkpoint: dict[str, str | None] = {}
    candidate_sha = None
    if plan is not None:
        expected, opponent_checkpoint, candidate_sha = _plan_expectations(plan)
        opponent_ids = list(opponent_checkpoint)
    else:
        opponent_ids = sorted(set(str(record.get("opponent_id")) for record in records))
        if expected_opponents is not None:
            required = sorted(str(value) for value in expected_opponents)
            missing_opponents = sorted(set(required) - set(opponent_ids))
            extra_opponents = sorted(set(opponent_ids) - set(required))
            if missing_opponents:
                errors.append(f"missing opponents: {missing_opponents}")
            if extra_opponents:
                errors.append(f"unexpected opponents: {extra_opponents}")
            opponent_ids = required
    deal_hashes: dict[str, str] = {}
    for index, record in enumerate(records):
        try:
            opponent = str(record.get("opponent_id", ""))
            deal_id = str(record.get("deal_id", ""))
            rotation = record.get("seat_rotation")
            if not opponent or not deal_id or isinstance(rotation, bool) or rotation not in range(4):
                raise ValueError("opponent/deal/rotation identity is malformed")
            key = (opponent, deal_id, int(rotation))
            if key in seen:
                raise ValueError(f"duplicate game record {key}")
            seen.add(key)
            if expected is not None and key not in expected:
                raise ValueError(f"extra record not present in evaluation plan: {key}")
            if candidate_sha is not None and record.get("candidate_checkpoint_sha256") != candidate_sha:
                raise ValueError("record candidate checkpoint identity differs from plan")
            expected_opponent_sha = opponent_checkpoint.get(opponent)
            if expected_opponent_sha is not None and record.get("opponent_checkpoint_sha256") != expected_opponent_sha:
                raise ValueError("record opponent checkpoint identity differs from plan")
            status = record.get("status")
            if status not in {"complete", "incomplete"}:
                raise ValueError("record status must be complete or incomplete")
            if status == "complete":
                _validate_complete_record(record)
            digest = str(record.get("base_deal_sha256", ""))
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
                raise ValueError("base_deal_sha256 is malformed")
            if deal_id in deal_hashes and deal_hashes[deal_id] != digest:
                raise ValueError("the same planned deal has different base-deal hashes across opponents/rotations")
            deal_hashes[deal_id] = digest
            clusters.setdefault((opponent, deal_id), []).append(record)
        except ValueError as error:
            errors.append(f"record {index}: {error}")
    if expected is not None:
        missing = sorted(expected - seen)
        if missing:
            errors.append(f"missing planned records: {missing}")
    for key, group in clusters.items():
        rotations = sorted(int(record["seat_rotation"]) for record in group)
        if rotations != [0, 1, 2, 3]:
            errors.append(f"cluster {key} rotations are {rotations}, expected [0, 1, 2, 3]")
    return clusters, errors, opponent_ids


def _summarize_subset(records: Sequence[Mapping[str, object]], *, compute_ci: bool) -> dict[str, object]:
    complete = [record for record in records if record.get("status") == "complete"]
    wins = [bool(record["candidate_win"]) for record in complete]
    team02 = [bool(record["candidate_win"]) for record in complete if int(record["candidate_team"]) == 0]
    team13 = [bool(record["candidate_win"]) for record in complete if int(record["candidate_team"]) == 1]
    by_rotation = {
        str(rotation): _rate([bool(record["candidate_win"]) for record in complete if int(record["seat_rotation"]) == rotation])
        for rotation in range(4)
    }
    observed = [value for value in by_rotation.values() if value is not None]
    runtime = sum(float(record.get("runtime_seconds", 0.0)) for record in records if isinstance(record.get("runtime_seconds", 0.0), (int, float)))
    clusters: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for record in complete:
        clusters.setdefault((str(record["opponent_id"]), str(record["deal_id"])), []).append(record)
    cluster_means = [
        mean(float(bool(record["candidate_win"])) for record in group)
        for group in clusters.values()
        if sorted(int(record["seat_rotation"]) for record in group) == [0, 1, 2, 3]
    ]
    ci = clustered_mean_ci95(cluster_means) if compute_ci else {
        "schema": CLUSTER_CI_POLICY_SCHEMA,
        "method": "not-reported-across-correlated-opponents",
        "clusters": len({str(record["deal_id"]) for record in complete}),
        "mean": None,
        "low": None,
        "high": None,
        "conclusive": False,
    }
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
        "eval/seat_rotation_gap": max(observed) - min(observed) if observed else None,
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


def summarize_records(
    records: Sequence[Mapping[str, object]],
    *,
    plan: Mapping[str, object] | None = None,
    expected_opponents: Iterable[str] | None = None,
) -> dict[str, object]:
    rows = [dict(record) for record in records]
    clusters, coverage_errors, opponent_ids = _validate_records(rows, plan=plan, expected_opponents=expected_opponents)
    per_opponent = {
        opponent: _summarize_subset([record for record in rows if record.get("opponent_id") == opponent], compute_ci=True)
        for opponent in opponent_ids
    }
    overall = _summarize_subset(rows, compute_ci=False)
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
            "plan_bound": plan is not None,
        },
    }


__all__ = [
    "CLUSTER_CI_POLICY_SCHEMA", "CLUSTER_STATS_SCHEMA", "clustered_mean_ci95", "summarize_records",
]
