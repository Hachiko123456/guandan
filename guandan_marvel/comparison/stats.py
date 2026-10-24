# SPDX-License-Identifier: MIT
"""Comparison statistics: cluster means, bootstrap CI, paired anchor deltas."""
from __future__ import annotations

import math
from statistics import mean
from typing import Mapping, Sequence

import numpy as np

from guandan_marvel.evaluation.manifests import deals_for
from guandan_marvel.evaluation.stats import clustered_mean_ci95

from .plan import (
    BOOTCAMP_FINAL_CANDIDATE_ARM, BOOTCAMP_FINAL_MIRROR_ARM, BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED, ROTATIONS, SUMMARY_SCHEMA, TRAINING_LEADER_SEAT, TRAINING_LEVEL_RANK,
)


def cluster_values(records: Sequence[Mapping[str, object]], *, require_complete: bool = True) -> tuple[list[float], int]:
    by_deal: dict[str, list[int]] = {}
    for record in records:
        if record.get("status") != "complete":
            continue
        deal_id = str(record["deal_id"])
        wins = by_deal.setdefault(deal_id, [])
        wins.append(1 if record["candidate_win"] else 0)
    values: list[float] = []
    incomplete = 0
    for deal_id, wins in sorted(by_deal.items()):
        if len(wins) != len(ROTATIONS):
            incomplete += 1
            if require_complete:
                continue
        values.append(sum(wins) / len(wins))
    return values, incomplete


def bootstrap_mean_ci(values: Sequence[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, object]:
    data = np.asarray([float(v) for v in values], dtype=np.float64)
    if data.size == 0:
        return {"clusters": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(data, size=(int(samples), data.size), replace=True)
    means = draws.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "clusters": int(data.size),
        "mean": float(data.mean()),
        "low": float(low),
        "high": float(high),
        "method": "cluster-bootstrap-percentile-95",
        "samples": int(samples),
    }


def _rate(values: Sequence[bool]) -> float | None:
    return sum(bool(v) for v in values) / len(values) if values else None


def deal_contexts(deals: Mapping[str, object]) -> dict[str, dict[str, int]]:
    """Map deal_id -> {level_rank, leader_seat} from a deals manifest."""
    datasets = deals.get("datasets")
    rows = datasets.get("final") if isinstance(datasets, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("deals manifest has no final deal rows")
    contexts: dict[str, dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("deal row must be a mapping")
        deal_id = str(row["deal_id"])
        if deal_id in contexts:
            raise ValueError(f"duplicate deal id in deals manifest: {deal_id}")
        contexts[deal_id] = {
            "level_rank": int(row["level_rank"]),
            "leader_seat": int(row["leader_seat"]),
        }
    return contexts


def context_slices(
    records: Sequence[Mapping[str, object]],
    contexts: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    """Gate-B readings split by deal context (full / level 15 / in-distribution / OOD)."""
    complete = [
        record for record in records
        if record.get("status") == "complete" and str(record.get("deal_id")) in contexts
    ]

    def _in_distribution(record: Mapping[str, object]) -> bool:
        context = contexts[str(record["deal_id"])]
        return (
            int(context["level_rank"]) == TRAINING_LEVEL_RANK
            and int(context["leader_seat"]) == TRAINING_LEADER_SEAT
        )

    groups = {
        "full": complete,
        "level_rank_15": [r for r in complete if int(contexts[str(r["deal_id"])]["level_rank"]) == TRAINING_LEVEL_RANK],
        "level_rank_15_leader_0": [r for r in complete if _in_distribution(r)],
        "out_of_distribution": [r for r in complete if not _in_distribution(r)],
    }
    slices: dict[str, object] = {}
    for name, subset in groups.items():
        values, _ = cluster_values(subset)
        slices[name] = {
            "games": len(subset),
            "clusters": len(values),
            "win_rate": _rate([bool(record["candidate_win"]) for record in subset]),
            "cluster_mean": mean(values) if values else None,
            "bootstrap": bootstrap_mean_ci(values),
        }
    return slices


def arm_summary(
    records: Sequence[Mapping[str, object]],
    *,
    skipped: int = 0,
    contexts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, object]:
    complete = [r for r in records if r.get("status") == "complete"]
    errors = [r for r in records if r.get("status") == "error"]
    incomplete = [r for r in records if r.get("status") == "incomplete"]
    values, cluster_incomplete = cluster_values(complete)
    by_rotation: dict[str, object] = {}
    for rotation in ROTATIONS:
        subset = [r for r in complete if int(r["seat_rotation"]) == rotation]
        by_rotation[str(rotation)] = {
            "games": len(subset),
            "win_rate": _rate([bool(r["candidate_win"]) for r in subset]),
        }
    runtime = sum(float(r.get("runtime_seconds", 0.0)) for r in complete)
    rewards = [float(r["candidate_team_reward"]) for r in complete]
    double_downs = [bool(r["candidate_win"]) and str(r.get("outcome_class")) == "DOUBLE_DOWN" for r in complete]
    diagnostics_totals: dict[str, int] = {}
    for record in complete:
        for key, value in (record.get("fd_diagnostics") or {}).items():
            diagnostics_totals[key] = diagnostics_totals.get(key, 0) + int(value)
    return {
        "games": len(records),
        "complete": len(complete),
        "errors": len(errors),
        "incomplete": len(incomplete),
        "skipped_due_to_time": int(skipped),
        "wins": sum(1 for r in complete if r["candidate_win"]),
        "losses": sum(1 for r in complete if not r["candidate_win"]),
        "game_level_win_rate": _rate([bool(r["candidate_win"]) for r in complete]),
        "clusters": {
            "count": len(values),
            "incomplete_clusters_excluded": cluster_incomplete,
            "bootstrap": bootstrap_mean_ci(values),
            "bernstein": clustered_mean_ci95(values),
        },
        "by_rotation": by_rotation,
        "mean_candidate_team_reward": mean(rewards) if rewards else None,
        "double_down_win_rate": _rate(double_downs),
        "runtime_seconds": runtime,
        "games_per_second": (len(complete) / runtime) if runtime > 0 else None,
        "illegal_count": sum(int(r.get("illegal_count", 0)) for r in complete),
        "truncated_count": sum(int(r.get("truncated_count", 0)) for r in complete),
        "fd_diagnostics_totals": diagnostics_totals or None,
        "by_context": None if contexts is None else context_slices(records, contexts),
    }


REQUIRED_ROTATIONS = frozenset(ROTATIONS)


def _arm_deal_index(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    """Per-deal rotation index with fail-closed completeness accounting."""
    deals: dict[str, dict[str, object]] = {}
    counts = {
        "error_records": 0, "incomplete_records": 0, "duplicate_records": 0,
        "illegal_records": 0, "truncated_records": 0, "bad_rotation_records": 0,
    }
    for row in records:
        deal_id = str(row.get("deal_id", ""))
        if not deal_id:
            continue
        entry = deals.setdefault(
            deal_id, {"rotations": set(), "wins": {}, "invalid": False, "problems": set()},
        )
        status = row.get("status")
        if status != "complete":
            key = "error_records" if status == "error" else "incomplete_records"
            counts[key] += 1
            entry["invalid"] = True
            entry["problems"].add("error" if status == "error" else "incomplete")
            continue
        rotation = row.get("seat_rotation")
        if isinstance(rotation, bool) or rotation not in ROTATIONS:
            counts["bad_rotation_records"] += 1
            entry["invalid"] = True
            entry["problems"].add("bad_rotation")
            continue
        rotation = int(rotation)
        if rotation in entry["rotations"]:
            counts["duplicate_records"] += 1
            entry["invalid"] = True
            entry["problems"].add("duplicate_rotation")
            continue
        entry["rotations"].add(rotation)
        entry["wins"][rotation] = 1.0 if row.get("candidate_win") else 0.0
        if int(row.get("illegal_count", 0) or 0) != 0:
            counts["illegal_records"] += 1
            entry["invalid"] = True
            entry["problems"].add("illegal")
        if int(row.get("truncated_count", 0) or 0) != 0:
            counts["truncated_records"] += 1
            entry["invalid"] = True
            entry["problems"].add("truncated")
    return deals, counts


def paired_delta(
    arm_a_records: Sequence[Mapping[str, object]],
    arm_b_records: Sequence[Mapping[str, object]],
    *,
    label: str,
    expected_deals: Sequence[str] | None = None,
) -> dict[str, object]:
    """Paired deal delta with strict completeness gating.

    A deal enters the bootstrap only when both arms hold exactly rotations
    {0,1,2,3} with one complete record per rotation, zero illegal/truncated
    records and no error/incomplete rows.  Error, incomplete, duplicate and
    mismatched deals are counted and excluded; the endpoint is complete only
    when every planned deal is a valid pair and nothing unexpected appears.
    """
    deals_a, counts_a = _arm_deal_index(arm_a_records)
    deals_b, counts_b = _arm_deal_index(arm_b_records)
    expected = None if expected_deals is None else [str(value) for value in expected_deals]
    expected_set = set() if expected is None else set(expected)
    observed = set(deals_a) | set(deals_b)
    if expected is None:
        evaluated_ids = sorted(observed)
        planned_deals = None
        unexpected_deals = 0
    else:
        evaluated_ids = sorted(expected_set)
        planned_deals = len(expected_set)
        unexpected_deals = len(observed - expected_set)
    problems = {
        "incomplete_deals": 0,
        "duplicate_rotation_deals": 0,
        "mismatched_deals": 0,
        "invalid_deals": 0,
        "missing_deals": 0,
        "unexpected_deals": unexpected_deals,
        "bad_rotation_deals": 0,
    }
    record_counts = {
        key: int(counts_a.get(key, 0)) + int(counts_b.get(key, 0))
        for key in set(counts_a) | set(counts_b)
    }
    deltas: list[float] = []
    complete_ids: list[str] = []
    for deal_id in evaluated_ids:
        entry_a, entry_b = deals_a.get(deal_id), deals_b.get(deal_id)
        if entry_a is None or entry_b is None:
            problems["missing_deals"] += 1
            continue
        rotations_a, rotations_b = entry_a["rotations"], entry_b["rotations"]
        if rotations_a != rotations_b:
            problems["mismatched_deals"] += 1
            continue
        if rotations_a != REQUIRED_ROTATIONS:
            problems["incomplete_deals"] += 1
            continue
        if entry_a["invalid"] or entry_b["invalid"]:
            if "duplicate_rotation" in (entry_a["problems"] | entry_b["problems"]):
                problems["duplicate_rotation_deals"] += 1
            if "bad_rotation" in (entry_a["problems"] | entry_b["problems"]):
                problems["bad_rotation_deals"] += 1
            problems["invalid_deals"] += 1
            continue
        wins_a = entry_a["wins"]
        wins_b = entry_b["wins"]
        deltas.append(
            sum(float(wins_a[rotation]) for rotation in ROTATIONS) / len(ROTATIONS)
            - sum(float(wins_b[rotation]) for rotation in ROTATIONS) / len(ROTATIONS)
        )
        complete_ids.append(deal_id)
    complete = (
        bool(complete_ids)
        and not any(int(value) > 0 for value in problems.values())
        and not any(int(value) > 0 for value in record_counts.values())
        and (planned_deals is None or len(complete_ids) == planned_deals)
    )
    return {
        "label": label,
        "planned_deals": planned_deals,
        "paired_deals": len(complete_ids),
        "complete": complete,
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "bootstrap": bootstrap_mean_ci(deltas),
        "problems": problems,
        "record_counts": record_counts,
    }


def primary_decision(arm_stats: Mapping[str, object]) -> dict[str, object]:
    interval = ((arm_stats.get("clusters") or {}).get("bootstrap") or {})
    low, high = interval.get("low"), interval.get("high")
    if low is None or high is None:
        return {"verdict": "no_data", "rule": "cluster-bootstrap-95 of the head-to-head arm"}
    if low > 0.5:
        verdict = "m09_stronger"
    elif high < 0.5:
        verdict = "fabledan_stronger"
    else:
        verdict = "inconclusive_at_this_sample_size"
    return {
        "verdict": verdict,
        "head_to_head_win_rate": interval.get("mean"),
        "ci95_low": low,
        "ci95_high": high,
        "rule": "m09_stronger if ci95_low>0.5; fabledan_stronger if ci95_high<0.5; else inconclusive",
    }


def paired_primary_decision(paired_stats: Mapping[str, object]) -> dict[str, object]:
    """Primary verdict; strength verdicts require complete planned evidence."""
    problems = {
        str(key): int(value)
        for key, value in (paired_stats.get("problems") or {}).items()
        if int(value) > 0
    }
    record_counts = {
        str(key): int(value)
        for key, value in (paired_stats.get("record_counts") or {}).items()
        if int(value) > 0
    }
    if not bool(paired_stats.get("complete")):
        return {
            "verdict": "incomplete_evidence",
            "mean_delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "paired_deals": paired_stats.get("paired_deals"),
            "planned_deals": paired_stats.get("planned_deals"),
            "problems": problems,
            "record_counts": record_counts,
            "rule": (
                "primary endpoint requires every planned paired deal to hold exactly rotations "
                "{0,1,2,3} on both arms with zero illegal/truncated/error records; otherwise no strength verdict"
            ),
        }
    interval = paired_stats.get("bootstrap") or {}
    low, high = interval.get("low"), interval.get("high")
    if low is None or high is None:
        return {
            "verdict": "no_data",
            "mean_delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "paired_deals": paired_stats.get("paired_deals"),
            "planned_deals": paired_stats.get("planned_deals"),
            "problems": problems,
            "record_counts": record_counts,
            "rule": "cluster-bootstrap-95 of the paired deal deltas",
        }
    if low > 0.0:
        verdict = "bootcamp_final_stronger"
    elif high < 0.0:
        verdict = "m09_iter2000_stronger"
    else:
        verdict = "inconclusive_at_this_sample_size"
    return {
        "verdict": verdict,
        "mean_delta": paired_stats.get("mean_delta"),
        "ci95_low": low,
        "ci95_high": high,
        "paired_deals": paired_stats.get("paired_deals"),
        "planned_deals": paired_stats.get("planned_deals"),
        "problems": problems,
        "record_counts": record_counts,
        "rule": "bootcamp_final_stronger if paired ci95_low>0; m09_iter2000_stronger if ci95_high<0; else inconclusive",
    }


def _planned_deal_ids(deals: Mapping[str, object], arm: Mapping[str, object]) -> list[str]:
    rows = deals_for(deals, "final")
    count = int(arm.get("deals", 0))
    if count <= 0 or count > len(rows):
        raise ValueError(f"arm {arm.get('id')!r} deal plan is outside the manifest: {count}/{len(rows)}")
    return [str(row["deal_id"]) for row in rows[:count]]


def build_summary(
    control: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    skipped_by_arm: Mapping[str, int],
    deals: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contexts = None if deals is None else deal_contexts(deals)
    arms: dict[str, object] = {}
    by_arm: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        by_arm.setdefault(str(record.get("arm_id")), []).append(record)
    arm_plans = {str(arm["id"]): arm for arm in control["arms"]}
    for arm in control["arms"]:
        arm_id = str(arm["id"])
        arms[arm_id] = arm_summary(
            by_arm.get(arm_id, []), skipped=skipped_by_arm.get(arm_id, 0), contexts=contexts,
        )

    def _expected_for(arm_id: str) -> list[str] | None:
        if deals is None or arm_id not in arm_plans:
            return None
        return _planned_deal_ids(deals, arm_plans[arm_id])

    paired = {}
    if "fabledan_vs_rule" in arms and "m09_vs_rule" in arms:
        paired["fabledan_vs_rule_minus_m09_vs_rule"] = paired_delta(
            by_arm.get("fabledan_vs_rule", []), by_arm.get("m09_vs_rule", []),
            label="fabledan_vs_rule_minus_m09_vs_rule",
            expected_deals=_expected_for("fabledan_vs_rule"),
        )
    if "fabledan_vs_random" in arms and "m09_vs_random" in arms:
        paired["fabledan_vs_random_minus_m09_vs_random"] = paired_delta(
            by_arm.get("fabledan_vs_random", []), by_arm.get("m09_vs_random", []),
            label="fabledan_vs_random_minus_m09_vs_random",
            expected_deals=_expected_for("fabledan_vs_random"),
        )
    paired_endpoint_key = f"{BOOTCAMP_FINAL_CANDIDATE_ARM}_minus_{BOOTCAMP_FINAL_MIRROR_ARM}"
    if BOOTCAMP_FINAL_CANDIDATE_ARM in arm_plans and BOOTCAMP_FINAL_MIRROR_ARM in arm_plans:
        planned_a = _expected_for(BOOTCAMP_FINAL_CANDIDATE_ARM)
        planned_b = _expected_for(BOOTCAMP_FINAL_MIRROR_ARM)
        if planned_a is not None and planned_b is not None and planned_a != planned_b:
            raise ValueError("bootcamp paired arms must plan the same deals in the same order")
        paired[paired_endpoint_key] = paired_delta(
            by_arm.get(BOOTCAMP_FINAL_CANDIDATE_ARM, []), by_arm.get(BOOTCAMP_FINAL_MIRROR_ARM, []),
            label=paired_endpoint_key,
            expected_deals=planned_a,
        )
    primary = primary_decision(arms.get("h2h", {}))
    primary_endpoint = "h2h"
    endpoint = paired.get(paired_endpoint_key)
    if isinstance(endpoint, Mapping):
        primary = paired_primary_decision(endpoint)
        primary_endpoint = BOOTCAMP_FINAL_CANDIDATE_ARM
    return {
        "schema": SUMMARY_SCHEMA,
        "control_hash": control.get("hash"),
        "comparison_id": control.get("comparison_id"),
        "scope": control.get("scope"),
        "arms": arms,
        "paired": paired,
        "primary_endpoint": primary_endpoint,
        "primary": primary,
    }


def render_markdown(summary: Mapping[str, object]) -> str:
    lines = ["# M09 vs FableDan comparison report", ""]
    scope = summary.get("scope") or {}
    lines.append(f"- comparison_id: `{summary.get('comparison_id')}`")
    lines.append(f"- scope: `{scope}`")
    lines.append(f"- control hash: `{summary.get('control_hash')}`")
    lines.append("")
    lines.append("| arm | complete | win rate | cluster mean | bootstrap 95% | bernstein 95% |")
    lines.append("|---|---|---|---|---|---|")
    for arm_id, stats in (summary.get("arms") or {}).items():
        clusters = stats.get("clusters") or {}
        boot = clusters.get("bootstrap") or {}
        bern = clusters.get("bernstein") or {}
        def fmt(value):
            return "n/a" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {arm_id} | {stats.get('complete')}/{stats.get('games')} | {fmt(stats.get('game_level_win_rate'))} "
            f"| {fmt(boot.get('mean'))} | [{fmt(boot.get('low'))}, {fmt(boot.get('high'))}] "
            f"| [{fmt(bern.get('low'))}, {fmt(bern.get('high'))}] |"
        )
    lines.append("")
    for arm_id, stats in (summary.get("arms") or {}).items():
        slices = stats.get("by_context")
        if not slices:
            continue
        def fmt(value):
            return "n/a" if value is None else f"{float(value):.4f}"
        parts = []
        for name in ("full", "level_rank_15", "level_rank_15_leader_0", "out_of_distribution"):
            row = slices.get(name) or {}
            parts.append(f"{name} {fmt(row.get('win_rate'))} (n={row.get('games')})")
        lines.append(f"- {arm_id} context: " + "; ".join(parts))
    lines.append("")
    primary = summary.get("primary") or {}
    if primary.get("verdict") == "incomplete_evidence":
        lines.append(f"**Primary verdict [`{summary.get('primary_endpoint')}`]: `incomplete_evidence`** "
                     f"(paired {primary.get('paired_deals')}/{primary.get('planned_deals')} planned deals; "
                     f"problems {primary.get('problems')}; record counts {primary.get('record_counts')})")
    elif "mean_delta" in primary:
        lines.append(f"**Primary verdict [`{summary.get('primary_endpoint')}`]: `{primary.get('verdict')}`** "
                     f"(paired mean delta {primary.get('mean_delta')} over {primary.get('paired_deals')} deals, "
                     f"CI95 [{primary.get('ci95_low')}, {primary.get('ci95_high')}])")
    else:
        lines.append(f"**Primary verdict [`{summary.get('primary_endpoint')}`]: `{primary.get('verdict')}`** "
                     f"(h2h win rate {primary.get('head_to_head_win_rate')}, "
                     f"CI95 [{primary.get('ci95_low')}, {primary.get('ci95_high')}])")
    lines.append("")
    for label, paired in (summary.get("paired") or {}).items():
        lines.append(
            f"- {label}: mean delta {paired.get('mean_delta')} over {paired.get('paired_deals')} paired deals "
            f"(complete={paired.get('complete')}, problems={paired.get('problems')})"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "arm_summary", "bootstrap_mean_ci", "build_summary", "cluster_values",
    "context_slices", "deal_contexts", "paired_delta", "paired_primary_decision",
    "primary_decision", "render_markdown",
]
