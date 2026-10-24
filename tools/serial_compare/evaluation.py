# SPDX-License-Identifier: MIT
"""Evaluation task identity, persistence, sharding and deal-clustered statistics.

The confirmation set is the only formal verdict source.  Development arms are
auxiliary diagnostics and are never merged into the confirmation sample.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .common import (
    CONFIRMATION_ARMS, DEVELOPMENT_ARMS, UNIQUE_ASSIGNMENTS, PlanError,
    bounded_cluster_ci, canonical, formal_verdict, read_json, validate_deal_manifest,
)

EVALUATION_PROTOCOL = "two_unique_team_assignments_v1"
CONFIRMATION_SCHEMA = "guandan-serial-compare-confirmation-summary-v1"
DEVELOPMENT_SCHEMA = "guandan-serial-compare-development-summary-v1"

# (candidate model key, opponent model key or None for the fixed rule agent)
ARM_SPECS: dict[str, tuple[str, str | None]] = {
    "candidate_vs_control": ("candidate_final", "control_final"),
    "control_vs_375": ("control_final", "p375"),
    "candidate_vs_375": ("candidate_final", "p375"),
    "control_vs_rule": ("control_final", None),
    "candidate_vs_rule": ("candidate_final", None),
    "control_vs_500": ("control_final", "p500"),
    "candidate_vs_500": ("candidate_final", "p500"),
}


class EvaluationError(RuntimeError):
    """Raised when evaluation records violate the frozen protocol."""


class TruncatedGameError(EvaluationError):
    """A game hit the horizon without finishing and carries no usable identity.

    Raised *before* a truncated record would be written to disk, so a shard is
    never lost later with a confusing "lacks an action-trace identity" error.
    The evaluator host normally supplies ``action_trace_sha256`` and
    ``committed_steps`` for an incomplete game; this error only fires when a
    caller hands the worker a record that cannot be audited.
    """


def _is_hex_digest(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _unique(items: Iterable[object], name: str) -> None:
    values = list(items)
    if len(set(values)) != len(values):
        raise EvaluationError(f"duplicate {name}")


def tasks_for(deals: Sequence[Mapping[str, object]], arms: Sequence[str]) -> list[dict[str, object]]:
    if not arms:
        raise EvaluationError("at least one evaluation arm is required")
    unknown = sorted(set(arms) - set(ARM_SPECS))
    if unknown:
        raise EvaluationError(f"unknown evaluation arms: {unknown}")
    _unique(arms, "arm")
    tasks: list[dict[str, object]] = []
    for deal_index, deal in enumerate(deals):
        for assignment in UNIQUE_ASSIGNMENTS:
            for arm in arms:
                tasks.append({
                    "task_id": f"{arm}/{deal['deal_id']}/a{assignment}",
                    "arm": arm,
                    "deal_index": deal_index,
                    "deal_id": str(deal["deal_id"]),
                    "deal_seed": int(deal["seed"]),
                    "assignment": int(assignment),
                    "deal": dict(deal),
                })
    _unique((task["task_id"] for task in tasks), "task id")
    return tasks


def confirmation_tasks(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    deals = validate_deal_manifest(manifest, expected_domain="confirmation")
    return tasks_for(deals, CONFIRMATION_ARMS)


def development_tasks(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    deals = validate_deal_manifest(manifest, expected_domain="development")
    return tasks_for(deals, DEVELOPMENT_ARMS)


def expected_games(tasks: Sequence[Mapping[str, object]]) -> int:
    return len(tasks)


def schedule(tasks: Sequence[Mapping[str, object]], *, ranks: int = 2) -> list[list[dict[str, object]]]:
    """Balance *every* arm across ranks; pending work is rebalanced on resume.

    A two-model arm costs twice a rule arm, so a round-robin by task count alone
    would put all heavy games on one worker.
    """
    if int(ranks) <= 0:
        raise EvaluationError("ranks must be positive")
    _unique((task["task_id"] for task in tasks), "pending task")
    assigned: list[list[dict[str, object]]] = [[] for _ in range(int(ranks))]
    cost = [0] * int(ranks)
    for arm in ARM_SPECS:
        subset = sorted(
            (task for task in tasks if task["arm"] == arm),
            key=lambda task: (task["deal_index"], task["assignment"]),
        )
        weight = 1 if ARM_SPECS[arm][1] is None else 2
        for task in subset:
            rank = min(range(int(ranks)), key=lambda index: (cost[index], len(assigned[index]), index))
            assigned[rank].append(dict(task))
            cost[rank] += weight
    return assigned


def _record_hash(record: Mapping[str, object]) -> str:
    return canonical(dict(record))


def validate_record(record: Mapping[str, object], task: Mapping[str, object], *, models: Mapping[str, Mapping[str, object]]) -> None:
    if record.get("task_hash") != canonical(dict(task)):
        raise EvaluationError("record task hash/configuration mismatch")
    if record.get("record_hash") != canonical({k: v for k, v in record.items() if k != "record_hash"}):
        raise EvaluationError("record hash mismatch")
    if str(record.get("task_id")) != str(task["task_id"]):
        raise EvaluationError("record task id mismatch")
    if str(record.get("arm")) != str(task["arm"]):
        raise EvaluationError("record arm mismatch")
    if str(record.get("deal_id")) != str(task["deal_id"]):
        raise EvaluationError("record deal id mismatch")
    if int(record.get("deal_seed", -1)) != int(task["deal_seed"]):
        raise EvaluationError("record deal seed mismatch")
    if int(record.get("seat_rotation", -1)) != int(task["assignment"]):
        raise EvaluationError("record assignment mismatch")
    if int(record.get("candidate_team", -1)) != int(task["assignment"]) % 2:
        raise EvaluationError("record candidate team disagrees with the assignment")
    if record.get("temperature") != 0.0:
        raise EvaluationError("evaluation must use the deterministic temperature")
    if int(record.get("evaluation_horizon", -1)) != 1024:
        raise EvaluationError("evaluation horizon changed")
    digest = str(record.get("base_deal_sha256", ""))
    if not _is_hex_digest(digest):
        raise EvaluationError("record lacks a physical deal identity")
    candidate_key, opponent_key = ARM_SPECS[str(task["arm"])]
    expected_candidate = str(models[candidate_key]["sha256"])
    if str(record.get("candidate_checkpoint_sha256")) != expected_candidate:
        raise EvaluationError("record candidate checkpoint identity mismatch")
    expected_opponent = None if opponent_key is None else str(models[opponent_key]["sha256"])
    if record.get("opponent_checkpoint_sha256") != expected_opponent:
        raise EvaluationError("record opponent checkpoint identity mismatch")
    status = record.get("status")
    if status not in {"complete", "incomplete"}:
        raise EvaluationError("record status must be complete or incomplete")
    if not _is_hex_digest(record.get("action_trace_sha256")):
        if status == "incomplete":
            raise TruncatedGameError(
                "truncated game has no action-trace identity; refusing to persist an "
                "unauditable record (the host must supply action_trace_sha256/committed_steps)"
            )
        raise EvaluationError("record lacks an action-trace identity")
    if status == "complete":
        if not isinstance(record.get("candidate_win"), bool):
            raise EvaluationError("complete record needs a boolean candidate_win")
        if int(record.get("illegal_count", 0)) != 0 or int(record.get("truncated_count", 0)) != 0:
            raise EvaluationError("complete record has illegal/truncated games")
        reward = record.get("candidate_team_reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise EvaluationError("complete record has no finite team reward")
        if bool(record["candidate_win"]) != (float(reward) > 0.0):
            raise EvaluationError("candidate_win disagrees with the terminal reward sign")
        ranking = record.get("ranking")
        if not isinstance(ranking, (list, tuple)) or len(ranking) != 4 or any(type(v) is not int for v in ranking) or set(ranking) != {1, 2, 3, 4}:
            raise EvaluationError("complete record requires valid finishing ranks per seat")
        head = ranking.index(1)
        winner = head % 2
        outcome = {2: "DOUBLE_DOWN", 3: "HEAD_THIRD", 4: "HEAD_LAST"}[ranking[(head + 2) % 4]]
        points = {"DOUBLE_DOWN": 3.0, "HEAD_THIRD": 2.0, "HEAD_LAST": 1.0}[outcome]
        expected_reward = points if winner == int(record["candidate_team"]) else -points
        if type(record.get("winner_team")) is not int or record["winner_team"] != winner:
            raise EvaluationError("winner_team disagrees with finishing ranks")
        if record.get("outcome_class") != outcome or float(reward) != expected_reward:
            raise EvaluationError("terminal outcome/reward disagrees with finishing ranks")
        for name in ("token_steps", "committed_steps"):
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise EvaluationError(f"invalid {name}")
        if int(record["token_steps"]) > 1024 or int(record["committed_steps"]) > int(record["token_steps"]):
            raise EvaluationError("record exceeds the declared horizon")
    else:
        # A truncated game is retained as evidence but never counted.  It may
        # legitimately have zero committed steps, but its step counters must
        # still be integers so the record cannot hide an absurd horizon.
        if int(record.get("truncated_count", 0)) < 1:
            raise EvaluationError("incomplete record must declare truncated_count >= 1")
        for name in ("token_steps", "committed_steps", "illegal_count"):
            value = record.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EvaluationError(f"invalid truncated {name}")
        if int(record.get("token_steps", 0)) > 1024:
            raise EvaluationError("truncated record exceeds the declared horizon")


def merge_records(
    existing: Sequence[Mapping[str, object]],
    incoming: Sequence[Mapping[str, object]],
    *,
    tasks: Sequence[Mapping[str, object]],
    models: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Idempotent, conflict-rejecting merge of persisted game records."""
    expected = {str(task["task_id"]): task for task in tasks}
    merged: dict[str, dict[str, object]] = {}
    duplicates = 0
    for record in list(existing) + list(incoming):
        key = str(record.get("task_id"))
        if key not in expected:
            raise EvaluationError(f"record for an unexpected task: {key!r}")
        validate_record(record, expected[key], models=models)
        row = dict(record)
        if key in merged:
            if _record_hash(merged[key]) != _record_hash(row):
                raise EvaluationError(
                    f"conflicting duplicate result for {key!r}; refusing to double-count wins"
                )
            duplicates += 1
            continue
        merged[key] = row
    deal_hashes: dict[str, str] = {}
    for key, record in merged.items():
        deal_id = str(record["deal_id"])
        digest = str(record["base_deal_sha256"])
        if deal_hashes.setdefault(deal_id, digest) != digest:
            raise EvaluationError(f"deal {deal_id!r} changed its physical identity between records")
    return ([merged[key] for key in sorted(merged)], {
        "records": len(merged),
        "duplicate_identical_records": duplicates,
        "conflicts_rejected": 0,
    })


def read_record_file(path: str | Path, *, allow_partial_tail: bool = True) -> tuple[list[dict[str, object]], list[int]]:
    data = Path(path).read_text(encoding="utf-8")
    lines = data.splitlines()
    records: list[dict[str, object]] = []
    ignored: list[int] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(dict(json.loads(line)))
        except ValueError:
            if allow_partial_tail and index == len(lines) - 1 and not data.endswith("\n"):
                ignored.append(index + 1)
                continue
            raise EvaluationError(f"corrupt persisted game record in {path} line {index + 1}")
    return records, ignored


def collect_records(directory: str | Path) -> tuple[list[dict[str, object]], list[int]]:
    root = Path(directory)
    records: list[dict[str, object]] = []
    ignored: list[int] = []
    for path in sorted(root.glob("games.rank*.jsonl")):
        chunk, skipped = read_record_file(path)
        records.extend(chunk)
        ignored.extend(skipped)
    return records, ignored


def pending_tasks(tasks: Sequence[Mapping[str, object]], completed: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    done = {str(record.get("task_id")) for record in completed}
    return [dict(task) for task in tasks if str(task["task_id"]) not in done]


def validate_preflight_records(records, tasks, *, models, ignored_partial_lines=()):
    """A preflight may not promote truncated, duplicate or missing games."""
    if ignored_partial_lines or len(records) != len(tasks):
        raise EvaluationError("preflight evaluation is incomplete")
    merged, stats = merge_records([], records, tasks=tasks, models=models)
    if stats["duplicate_identical_records"] or len(merged) != len(tasks):
        raise EvaluationError("preflight contains duplicate or missing tasks")
    if any(row.get("status") != "complete" for row in merged):
        raise EvaluationError("preflight has truncated/incomplete games; formal training denied")
    return merged


def _cluster_means(
    records: Sequence[Mapping[str, object]], *, field: str = "candidate_win",
) -> tuple[dict[str, float], list[str]]:
    """Per-deal means for deals whose *both* unique assignments completed.

    A deal with only one completed assignment (the other truncated, still
    running, or missing) is excluded from the interval and reported as
    incomplete; it is **not** an exception and its single assignment is never
    treated as an independent cluster.  Duplicate assignments and identity
    violations still fail closed.
    """
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["deal_id"]), []).append(record)
    means: dict[str, float] = {}
    excluded: list[str] = []
    for deal_id in sorted(grouped):
        rows = grouped[deal_id]
        by_assignment: dict[int, list[Mapping[str, object]]] = {}
        for row in rows:
            if row.get("status") != "complete":
                continue
            by_assignment.setdefault(int(row["seat_rotation"]), []).append(row)
        if not by_assignment:
            excluded.append(deal_id)
            continue
        unexpected = sorted(set(by_assignment) - set(UNIQUE_ASSIGNMENTS))
        if unexpected:
            raise EvaluationError(
                f"deal {deal_id!r} has an unexpected assignment set {unexpected}"
            )
        resolved: dict[int, Mapping[str, object]] = {}
        for assignment, candidates in by_assignment.items():
            distinct = {_record_hash(candidate) for candidate in candidates}
            if len(distinct) != 1:
                raise EvaluationError(
                    f"deal {deal_id!r} needs exactly the two unique assignments; assignment "
                    f"{assignment} is duplicated with conflicting records"
                )
            resolved[assignment] = candidates[0]
        if sorted(resolved) != list(UNIQUE_ASSIGNMENTS):
            # Exactly one of the two unique assignments finished: incomplete.
            excluded.append(deal_id)
            continue
        means[deal_id] = statistics.mean(float(resolved[assignment][field]) for assignment in UNIQUE_ASSIGNMENTS)
    return means, excluded


def summarize_arm(
    records: Sequence[Mapping[str, object]], arm: str, *, compute_ci: bool = True,
) -> dict[str, object]:
    subset = [record for record in records if str(record.get("arm")) == arm]
    complete = [record for record in subset if record.get("status") == "complete"]
    wins, excluded = _cluster_means(subset)
    rewards, _ = _cluster_means(subset, field="candidate_team_reward")
    ordered = [wins[deal] for deal in sorted(wins)]
    ci = bounded_cluster_ci(ordered) if compute_ci else {
        "method": "not-reported-across-correlated-opponents",
        "clusters": len(wins), "mean": None, "low": None, "high": None,
        "interval_available": False,
    }
    outcomes = [str(record.get("outcome_class")) for record in complete]
    return {
        "arm": arm,
        "candidate_model": ARM_SPECS[arm][0],
        "opponent": ARM_SPECS[arm][1] or "rule",
        "records": len(subset),
        "completed_games": len(complete),
        "incomplete_records": len(subset) - len(complete),
        "independent_deals": len(wins),
        "excluded_incomplete_deals": sorted(excluded),
        "excluded_incomplete_deal_count": len(excluded),
        "win_rate": (sum(ordered) / len(ordered)) if ordered else None,
        "raw_team_reward_mean": (sum(rewards.values()) / len(rewards)) if rewards else None,
        "clustered_ci95": ci,
        "double_down_rate": outcomes.count("DOUBLE_DOWN") / len(outcomes) if outcomes else None,
        "head_third_rate": outcomes.count("HEAD_THIRD") / len(outcomes) if outcomes else None,
        "head_last_rate": outcomes.count("HEAD_LAST") / len(outcomes) if outcomes else None,
    }


def summarize_confirmation(
    records: Sequence[Mapping[str, object]], tasks: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected_ids = {str(task["task_id"]) for task in tasks}
    present = {str(record.get("task_id")) for record in records if record.get("status") == "complete"}
    complete = expected_ids == present
    summary = summarize_arm(records, "candidate_vs_control")
    summary.update({
        "schema": CONFIRMATION_SCHEMA,
        "domain": "confirmation",
        "expected_games": len(expected_ids),
        "complete": bool(complete),
        "missing_tasks": sorted(expected_ids - present),
        "statistics": "deal-clustered bounded empirical-Bernstein 95%",
        "verdict": formal_verdict(summary["clustered_ci95"], complete=bool(complete)),
        "fabledan": "NOT_RUN",
        "final_holdout": "NOT_TOUCHED",
    })
    return summary


def summarize_development(
    records: Sequence[Mapping[str, object]], tasks: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected_ids = {str(task["task_id"]) for task in tasks}
    present = {str(record.get("task_id")) for record in records if record.get("status") == "complete"}
    arms = {
        arm: summarize_arm(records, arm)
        for arm in DEVELOPMENT_ARMS
        if any(str(record.get("arm")) == arm for record in records)
    }
    return {
        "schema": DEVELOPMENT_SCHEMA,
        "domain": "development",
        "expected_games": len(expected_ids),
        "completed_games": len(present),
        "complete": expected_ids == present,
        "missing_tasks": sorted(expected_ids - present),
        "arms": arms,
        "candidate_vs_control_is_diagnostic": True,
        "not_merged_into_confirmation": True,
        "note": (
            "375 is inside the training pool (targeted benefit only); 500 is an unfiltered old "
            "opponent; rule is a fixed heuristic, not a FableDan substitute"
        ),
    }


def tasks_manifest(
    *, confirmation: Mapping[str, object], development: Mapping[str, object],
    models: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    confirmation_rows = confirmation_tasks(confirmation)
    development_rows = development_tasks(development)
    payload = {
        "schema": "guandan-serial-compare-task-manifest-v1",
        "protocol": EVALUATION_PROTOCOL,
        "assignments": list(UNIQUE_ASSIGNMENTS),
        "confirmation": {
            "manifest_hash": confirmation.get("hash"),
            "deal_count": len(confirmation["deals"]),
            "games": len(confirmation_rows),
            "arms": list(CONFIRMATION_ARMS),
            "task_ids": [task["task_id"] for task in confirmation_rows],
        },
        "development": {
            "manifest_hash": development.get("hash"),
            "deal_count": len(development["deals"]),
            "games": len(development_rows),
            "arms": list(DEVELOPMENT_ARMS),
            "task_ids": [task["task_id"] for task in development_rows],
        },
        "total_formal_games": len(confirmation_rows) + len(development_rows),
        "model_hashes": {name: str(value["sha256"]) for name, value in models.items()},
    }
    payload["hash"] = canonical(payload)
    return payload


__all__ = [
    "ARM_SPECS", "CONFIRMATION_SCHEMA", "DEVELOPMENT_SCHEMA", "EVALUATION_PROTOCOL",
    "EvaluationError", "TruncatedGameError", "collect_records", "confirmation_tasks", "development_tasks",
    "expected_games", "merge_records", "pending_tasks", "read_record_file", "schedule",
    "summarize_arm", "summarize_confirmation", "summarize_development", "tasks_for",
    "tasks_manifest", "validate_record",
]
