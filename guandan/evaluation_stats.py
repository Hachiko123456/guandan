"""Validated descriptive summaries of completed, rotated evaluation games.

Wilson score interval formula: NIST/SEMATECH e-Handbook, section 7.2.4.1,
https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
The per-game interval is NOT a cluster-adjusted confidence interval and is
not evidence of policy strength when rotations share underlying deals.
"""
from __future__ import annotations

import math
import re
from numbers import Real
from statistics import NormalDist


OPPONENTS = ("random", "rule", "snapshot")
CORRELATION_WARNING = (
    "Correlated rotations of the same base deal are not independent trials. "
    "The Wilson95 interval uses game counts and is descriptive only, not "
    "cluster-adjusted; do not infer policy strength or significance from it. "
    "independent_deal_groups counts distinct base-deal hashes, not rotations "
    "or opponent pairings; distinct hashes alone do not establish independence."
)
_REQUIRED = (
    "opponent", "deal_group", "seat_rotation", "policy_seat", "candidate_team",
    "done", "truncated", "ranking", "winner_team", "rewards", "team_rewards",
    "token_steps", "committed_steps", "base_deal_sha256", "seed",
)


def wilson_interval(wins: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score bounds; total must be positive, 0 <= wins <= total.

    Uses the lower-tail NormalDist quantile to avoid rounding (1+c)/2 to
    exactly one for confidence values very close to one. Zero observations
    have no estimated interval and are rejected (summaries instead use None).
    """
    if type(wins) is not int or type(total) is not int:
        raise ValueError("wins and total must be integers, not booleans")
    if total <= 0 or not 0 <= wins <= total:
        raise ValueError("require total > 0 and 0 <= wins <= total")
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError("confidence must be a finite real number in (0, 1)")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be a finite real number in (0, 1)")
    z = -NormalDist().inv_cdf((1.0 - confidence) / 2.0)
    p = wins / total
    inv_n = 1 / total
    z2_n = z * z * inv_n
    denominator = 1 + z2_n
    center = (p + z2_n / 2) / denominator
    radius = z * math.sqrt((p * (1 - p) + z2_n / 4) * inv_n) / denominator
    lower = 0.0 if wins == 0 else max(0.0, center - radius)
    upper = 1.0 if wins == total else min(1.0, center + radius)
    return lower, upper


def _integer(record: dict, name: str, allowed=None) -> int:
    value = record[name]
    if type(value) is not int or (allowed is not None and value not in allowed):
        raise ValueError(f"{name} must be an integer in the permitted range")
    return value


def _rewards(record: dict, name: str, expected: list[int]) -> None:
    values = record[name]
    if not isinstance(values, list) or len(values) != len(expected):
        raise ValueError(f"{name} must be a list of length {len(expected)}")
    for value, wanted in zip(values, expected):
        if isinstance(value, bool) or not isinstance(value, Real) or value != wanted:
            raise ValueError(f"{name} must be finite +/-1 and match winner_team")


def _validate_record(record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError("each record must be a dictionary")
    missing = [name for name in _REQUIRED if name not in record]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    if record["opponent"] not in OPPONENTS:
        raise ValueError("opponent must be random, rule, or snapshot")
    _integer(record, "deal_group")
    _integer(record, "seed")
    rotation = _integer(record, "seat_rotation", range(4))
    policy_seat = _integer(record, "policy_seat", range(4))
    team = _integer(record, "candidate_team", range(2))
    winner = _integer(record, "winner_team", range(2))
    if policy_seat != rotation or team != rotation % 2:
        raise ValueError("policy_seat/candidate_team must match seat_rotation")
    if record["done"] is not True:
        raise ValueError("nonterminal record: done must be True")
    if record["truncated"] is not False:
        raise ValueError("truncated must be False for a completed game")
    ranking = record["ranking"]
    if (not isinstance(ranking, list) or len(ranking) != 4
            or any(type(rank) is not int for rank in ranking)
            or set(ranking) != {1, 2, 3, 4}):
        raise ValueError("ranking must contain each rank 1..4 once, indexed by seat")
    if ranking.index(1) % 2 != winner:
        raise ValueError("winner_team must match the rank-1 seat's team")
    _rewards(record, "rewards", [1 if seat % 2 == winner else -1 for seat in range(4)])
    _rewards(record, "team_rewards", [1 if team_id == winner else -1 for team_id in range(2)])
    token_steps = _integer(record, "token_steps")
    committed_steps = _integer(record, "committed_steps")
    if not 0 <= committed_steps <= token_steps:
        raise ValueError("require 0 <= committed_steps <= token_steps")
    digest = record["base_deal_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError("base_deal_sha256 must be a 64-character hexadecimal digest")


def _counts(records: list[dict]) -> dict:
    completed = len(records)
    wins = sum(record["winner_team"] == record["candidate_team"] for record in records)
    return {
        "completed": completed,
        "wins": wins,
        "losses": completed - wins,
        "win_rate": wins / completed if completed else None,
        "mean_team_reward": (
            sum(record["team_rewards"][record["candidate_team"]] for record in records) / completed
            if completed else None
        ),
        "token_steps": sum(record["token_steps"] for record in records),
        "committed_steps": sum(record["committed_steps"] for record in records),
    }


def _confidence(records: list[dict], counts: dict) -> dict:
    return {
        "method": "Wilson95",
        "confidence": 0.95,
        "interval": list(wilson_interval(counts["wins"], counts["completed"])) if records else None,
        "warning": CORRELATION_WARNING,
        "independent_deal_groups": len({record["base_deal_sha256"].lower() for record in records}),
        "game_count": len(records),
        "cluster_adjusted": False,
    }


def summarize_games(records: list[dict]) -> dict:
    """Validate first, then return JSON-ready overall and per-opponent results.

    All three opponent keys and four string rotation keys are always present;
    by_deal_group contains only observed integer group IDs converted to strings.
    Empty subsets have counts zero and rates/interval None (not fabricated 0%).
    Partial sets of rotations/groups are accepted and reported without padding
    game counts. Repeated base-deal hashes never inflate the independent group
    count, including across opponents in the overall summary. Inputs are not
    modified. Duplicate (opponent, deal_group, rotation) records are rejected.
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list of dictionaries")
    seen = set()
    group_hashes = {}
    for index, record in enumerate(records):
        try:
            _validate_record(record)
            key = (record["opponent"], record["deal_group"], record["seat_rotation"])
            if key in seen:
                raise ValueError(f"duplicate (opponent, deal_group, rotation): {key}")
            seen.add(key)
            group_key = key[:2]
            digest = record["base_deal_sha256"].lower()
            if group_key in group_hashes and group_hashes[group_key] != digest:
                raise ValueError("rotations in the same opponent/deal_group have different base-deal hashes")
            group_hashes[group_key] = digest
        except ValueError as exc:
            raise ValueError(f"record {index}: {exc}") from exc

    per_opponent = {}
    for opponent in OPPONENTS:
        games = [record for record in records if record["opponent"] == opponent]
        summary = _counts(games)
        summary["by_rotation"] = {
            str(rotation): _counts([record for record in games if record["seat_rotation"] == rotation])
            for rotation in range(4)
        }
        by_group = {}
        for group in sorted({record["deal_group"] for record in games}):
            grouped = [record for record in games if record["deal_group"] == group]
            by_group[str(group)] = {
                **_counts(grouped),
                "base_deal_sha256": grouped[0]["base_deal_sha256"].lower(),
                "rotations": sorted(record["seat_rotation"] for record in grouped),
            }
        summary["by_deal_group"] = by_group
        summary["deal_groups"] = len(by_group)
        summary["confidence"] = _confidence(games, summary)
        per_opponent[opponent] = summary

    overall = _counts(records)
    overall["deal_groups"] = len({record["deal_group"] for record in records})
    overall["opponent_deal_groups"] = len(group_hashes)
    overall["confidence"] = _confidence(records, overall)
    return {"overall": overall, "per_opponent": per_opponent}


__all__ = ["summarize_games", "wilson_interval"]
