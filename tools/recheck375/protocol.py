"""Two unique team assignments, cache equivalence and deal-cluster statistics."""
from __future__ import annotations
import collections
import json
import math
import statistics
from pathlib import Path
from .common import PROTOCOL, canonical, verify_hash

ARMS = {
    "p250_vs_rule": ("p250", None),
    "p500_vs_rule": ("p500", None),
    "p500_vs_p250": ("p500", "p250"),
    "p375_vs_rule": ("p375", None),
    "p375_vs_p250": ("p375", "p250"),
    "p375_vs_p500": ("p375", "p500"),
}
CACHED_ARMS = tuple(ARMS)[:3]
NEW_ARMS = tuple(ARMS)[3:]
OLD_TO_NEW = dict(zip(("control_vs_rule", "candidate_vs_rule", "candidate_vs_control"), CACHED_ARMS))
TRACE_FIELDS = ("policies_by_seat", "action_trace_sha256", "ranking", "winner_team", "candidate_win", "candidate_team_reward", "token_steps", "committed_steps")


def validate_manifest(manifest, plan):
    verify_hash(manifest)
    deals = manifest["datasets"]["selection"]
    if manifest["hash"] != plan["development_hash"] or len(deals) != plan["deal_count"]:
        raise ValueError("different endpoint development set")
    if manifest["datasets"]["training"] or manifest["datasets"]["final"]:
        raise ValueError("this evaluation may not consume training/final-holdout datasets")
    if len({d["deal_id"] for d in deals}) != len(deals) or len({d["seed"] for d in deals}) != len(deals):
        raise ValueError("duplicate development deals")
    return deals


def tasks_for(manifest, arms, plan, *, first_deals=None):
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ValueError("unrecognized evaluation arms")
    deals = validate_manifest(manifest, plan)
    limit = len(deals) if first_deals is None else min(first_deals, len(deals))
    tasks = []
    for index, deal in enumerate(deals[:limit]):
        for rotation in (0, 1):
            for arm in arms:
                tasks.append({"task_id": f"{arm}/{deal['deal_id']}/{rotation}", "arm": arm,
                              "deal_index": index, "deal": dict(deal), "rotation": rotation})
    return tasks


def schedule(tasks):
    """Balance EVERY arm, not alternating a mixed arm list between GPUs.

    Pending work is rebalanced on resume. Reversing assignment order on odd
    deals also avoids pinning one physical team to one worker for the full run.
    """
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate pending tasks")
    assigned = [[], []]
    cost = [0, 0]
    for arm in ARMS:
        subset = sorted((t for t in tasks if t["arm"] == arm),
                        key=lambda t: (t["deal_index"], t["rotation"] if t["deal_index"] % 2 == 0 else 1 - t["rotation"]))
        for task in subset:
            rank = min(range(2), key=lambda r: (cost[r], len(assigned[r]), r))
            assigned[rank].append(task)
            cost[rank] += 1 if ARMS[arm][1] is None else 2
    return assigned


def assert_outcome(row):
    if row.get("status") != "complete" or row.get("illegal_count") != 0 or row.get("truncated_count") != 0:
        raise ValueError("incomplete/illegal/truncated evaluation game")
    ranking = row["ranking"]
    if sorted(ranking) != [1, 2, 3, 4]:
        raise ValueError("invalid terminal ranking")
    first = ranking.index(1)
    winner = first % 2
    mate_rank = ranking[(first + 2) % 4]
    points = {2: 3, 3: 2, 4: 1}[mate_rank]
    outcome = {2: "DOUBLE_DOWN", 3: "HEAD_THIRD", 4: "HEAD_LAST"}[mate_rank]
    expected_reward = points if row["candidate_team"] == winner else -points
    if row["winner_team"] != winner or row["outcome_class"] != outcome:
        raise ValueError("ranking/outcome mismatch")
    if row["candidate_team_reward"] != expected_reward or type(row["candidate_win"]) is not bool or row["candidate_win"] != (expected_reward > 0):
        raise ValueError("ranking/reward/win mismatch")
    if row["rewards"] != [float(points if seat % 2 == winner else -points) for seat in range(4)]:
        raise ValueError("terminal reward vector mismatch")
    for name in ("token_steps", "committed_steps"):
        if type(row.get(name)) is not int or row[name] <= 0:
            raise ValueError("invalid actual step counter")
    if row["token_steps"] > 1024 or row["committed_steps"] > row["token_steps"]:
        raise ValueError("invalid horizon/commit counter")
    if row["engine_finish_token_step"] != row["token_steps"] or row["engine_finish_committed_step"] != row["committed_steps"]:
        raise ValueError("engine/host terminal-step mismatch")
    if not isinstance(row.get("runtime_seconds"), (int, float)) or not math.isfinite(row["runtime_seconds"]) or row["runtime_seconds"] < 0:
        raise ValueError("invalid game timing")
    for name in ("base_deal_sha256", "action_trace_sha256"):
        value = row.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("missing physical/action-trace identity")


def validate_row(row, task, plan, *, old=False):
    assert_outcome(row)
    arm = task["arm"]
    candidate, opponent = ARMS[arm]
    deal = task["deal"]
    expected = {
        "deal_id": deal["deal_id"], "deal_seed": deal["seed"], "level_rank": deal["level_rank"],
        "leader_seat": deal["leader_seat"], "seat_rotation": task["rotation"],
        "candidate_team": task["rotation"] % 2, "temperature": 0.0,
        "candidate_checkpoint_sha256": plan["models"][candidate]["sha256"],
        "opponent_checkpoint_sha256": None if opponent is None else plan["models"][opponent]["sha256"],
        "opponent_kind": "rule" if opponent is None else "v2",
        "opponent_id": row["arm"] if old else arm,
        "logical_to_physical": [(seat + task["rotation"]) % 4 for seat in range(4)],
        "evaluation_manifest_hash": plan["development_hash"], "evaluation_horizon": 1024,
    }
    if not old:
        expected.update({"arm": arm, "task_id": task["task_id"], "task_hash": canonical(task),
                         "evaluation_protocol": PROTOCOL, "evaluation_plan_hash": plan["plan_hash"], "evaluation_horizon": 1024})
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"task identity mismatch {key}: {task['task_id']}")
    roles = ["candidate" if seat % 2 == row["candidate_team"] else row["opponent_id"] for seat in range(4)]
    if row["policies_by_seat"] != roles:
        raise ValueError("wrong team/seat assignment")


def validate_records(records, tasks, plan, *, allow_subset=False):
    expected = {t["task_id"]: t for t in tasks}
    found = {}
    physical = {}
    for row in records:
        key = row.get("task_id")
        if key not in expected or key in found:
            raise ValueError("duplicate or unexpected task record")
        validate_row(row, expected[key], plan)
        deal_id = row["deal_id"]
        if physical.setdefault(deal_id, row["base_deal_sha256"]) != row["base_deal_sha256"]:
            raise ValueError("same deal changed across arms/assignments")
        found[key] = row
    if not allow_subset and set(found) != set(expected):
        raise ValueError(f"missing {len(set(expected) - set(found))} evaluation tasks")
    return found


def compare_replay(actual, expected):
    fields = TRACE_FIELDS + ("base_deal_sha256", "candidate_checkpoint_sha256", "opponent_checkpoint_sha256", "candidate_team")
    for name in fields:
        if actual[name] != expected[name]:
            raise ValueError("cached replay mismatch: " + name)


def audit_cache(raw_records, manifest, plan, prior_summary):
    """Prove 4->2 equivalence over ALL prior endpoint games before using cache."""
    deals = validate_manifest(manifest, plan)
    expected = {}
    for index, deal in enumerate(deals):
        for old_arm, new_arm in OLD_TO_NEW.items():
            for rotation in range(4):
                task = {"task_id": f"{new_arm}/{deal['deal_id']}/{rotation}", "arm": new_arm,
                        "deal_index": index, "deal": dict(deal), "rotation": rotation}
                expected[(old_arm, deal["deal_id"], rotation)] = task
    grouped = collections.defaultdict(dict)
    seen = set()
    for row in raw_records:
        key = (row.get("arm"), row.get("deal_id"), row.get("seat_rotation"))
        if key not in expected or key in seen:
            raise ValueError("duplicate/unplanned original cache record")
        seen.add(key)
        validate_row(row, expected[key], plan, old=True)
        if row["continuation_plan_hash"] != plan["prior_training_plan_hash"]:
            raise ValueError("cache belongs to another prior training run")
        if row["task_id"] != f"{key[0]}/{key[1]}/{key[2]}":
            raise ValueError("wrong original cache task id")
        grouped[(key[0], key[1])][key[2]] = row
    if seen != set(expected):
        raise ValueError("incomplete original cache")
    unique = []
    for (old_arm, deal_id), group in sorted(grouped.items()):
        for left, right in ((0, 2), (1, 3)):
            compare_replay(group[left], group[right])
        for rotation in (0, 1):
            original = group[rotation]
            row = dict(original)
            task = expected[(old_arm, deal_id, rotation)]
            row.update({"arm": task["arm"], "opponent_id": task["arm"],
                        "policies_by_seat": ["candidate" if p == "candidate" else task["arm"] for p in original["policies_by_seat"]],
                        "task_id": task["task_id"], "task_hash": canonical(task),
                        "record_origin": "cached_prior_run", "cached_original_task_id": original["task_id"],
                        "cached_original_record_hash": canonical(original), "evaluation_protocol": PROTOCOL,
                        "evaluation_plan_hash": plan["plan_hash"]})
            unique.append(row)
    tasks = tasks_for(manifest, CACHED_ARMS, plan)
    validate_records(unique, tasks, plan)
    summaries = summarize_arms(unique)
    for old_arm, arm in OLD_TO_NEW.items():
        old = prior_summary["arms"][old_arm]
        new = summaries[arm]
        if old["win_rate"] != new["win_rate"] or old["raw_team_reward_mean"] != new["raw_team_reward_mean"]:
            raise ValueError("2-assignment cache changed old win/score mean")
        for field in ("low", "high", "sample_variance"):
            if abs(old["clustered_ci95"][field] - new["clustered_ci95"][field]) > 1e-12:
                raise ValueError("2-assignment cache changed old cluster CI")
    return unique, {"status": "PASS", "prior_actual_records": len(raw_records), "unique_records_reused": len(unique),
                    "identical_rotation_pairs_verified": len(raw_records) // 2,
                    "win_score_and_ci_unchanged": True, "independent_deals": len(deals)}


def bounded_ci(values, *, lower=0.0, upper=1.0):
    if upper <= lower or not values:
        raise ValueError("missing bounded samples")
    scaled = [(float(v) - lower) / (upper - lower) for v in values]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scaled):
        raise ValueError("sample outside declared bounds")
    n = len(values)
    mean = statistics.mean(scaled)
    if n < 2:
        return {"clusters": n, "mean": statistics.mean(values), "low": None, "high": None, "interval_available": False}
    variance = statistics.variance(scaled)
    log_term = math.log(3 / 0.05)
    margin = math.sqrt(2 * variance * log_term / n) + 3 * log_term / n
    return {"method": "bounded-empirical-Bernstein-95", "confidence": .95, "clusters": n,
            "mean": statistics.mean(values), "low": lower + max(0, mean - margin) * (upper - lower),
            "high": lower + min(1, mean + margin) * (upper - lower), "sample_variance": variance,
            "interval_available": True, "note": "interval_available is NOT a claim of improvement"}


def cluster_means(records, arm, field):
    grouped = collections.defaultdict(list)
    for row in records:
        if row["arm"] == arm:
            grouped[row["deal_id"]].append(row)
    means = {}
    for deal_id, rows in grouped.items():
        if sorted(r["seat_rotation"] for r in rows) != [0, 1]:
            raise ValueError("each deal needs exactly two UNIQUE team assignments")
        means[deal_id] = statistics.mean(float(r[field]) for r in rows)
    return means


def summarize_arms(records):
    result = {}
    for arm in ARMS:
        subset = [r for r in records if r["arm"] == arm]
        if not subset:
            continue
        wins = cluster_means(subset, arm, "candidate_win")
        rewards = cluster_means(subset, arm, "candidate_team_reward")
        result[arm] = {"records": len(subset), "independent_deals": len(wins), "win_rate": statistics.mean(wins.values()),
                       "clustered_ci95": bounded_ci(list(wins.values())), "raw_team_reward_mean": statistics.mean(rewards.values())}
    return result


def paired_difference(records, candidate_arm, control_arm):
    a = cluster_means(records, candidate_arm, "candidate_win")
    b = cluster_means(records, control_arm, "candidate_win")
    if set(a) != set(b):
        raise ValueError("paired arms use different deals")
    differences = [a[d] - b[d] for d in sorted(a)]
    return bounded_ci(differences, lower=-1, upper=1)


def decision_summary(records):
    arms = summarize_arms(records)
    pairs = {"p375_rule_minus_p250": paired_difference(records, "p375_vs_rule", "p250_vs_rule"),
             "p500_rule_minus_p250": paired_difference(records, "p500_vs_rule", "p250_vs_rule"),
             "p375_rule_minus_p500": paired_difference(records, "p375_vs_rule", "p500_vs_rule")}
    ci = arms["p375_vs_p250"]["clustered_ci95"]
    supported = ci["low"] > .5 and pairs["p375_rule_minus_p250"]["low"] > 0
    if arms["p375_vs_p250"]["win_rate"] > .5 and arms["p375_vs_p500"]["win_rate"] > .5 and pairs["p375_rule_minus_p500"]["mean"] >= 0:
        direction = "possible_late_drift_hypothesis_only_consider_conservative_update_test"
    elif arms["p375_vs_p250"]["win_rate"] <= .5 and arms["p500_vs_p250"]["win_rate"] <= .5:
        direction = "no_broad_gain_test_training_opponent_distribution_before_scaleup"
    else:
        direction = "mixed_matrix_no_single_root_cause_established"
    return {"schema": "guandan-same-set-checkpoint-diagnostic-v1", "arms": arms, "paired_rule_differences": pairs,
            "diagnostic_direction_hypothesis_not_causation": direction,
            "verdict": "PROMISING_375_NEEDS_FRESH_CONFIRMATION" if supported else "INCONCLUSIVE_KEEP_250_REFERENCE",
            "formal_promotion": False, "posthoc_reused_development_set": True, "new_training_games": 0,
            "next_step": "fresh development confirmation before any scale-up" if supported else "do not blindly scale; use the aligned matrix to select a single-variable experiment",
            "fabledan": "NOT_RUN", "final_holdout": "NOT_TOUCHED"}


def read_records(path, *, allow_partial_tail=False):
    data = Path(path).read_text(encoding="utf8")
    lines = data.splitlines()
    records, ignored = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            if allow_partial_tail and index == len(lines) - 1 and not data.endswith("\n"):
                ignored.append(index + 1)
            else:
                raise ValueError("corrupt persisted game record")
    return records, ignored
