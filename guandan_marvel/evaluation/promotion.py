# SPDX-License-Identifier: MIT
"""Promotion decisions and atomic champion manifest updates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Callable, Mapping

from .common import atomic_write_json, load_json, payload_hash

PROMOTION_DECISION_SCHEMA = "guandan-marvel-promotion-decision-v1"
CHAMPION_MANIFEST_SCHEMA = "guandan-marvel-champion-manifest-v1"
REQUIRED_MATRIX = ("champion", "rule", "random", "legacy", "opponent_pool")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PromotionPolicy:
    ci95_low_threshold: float = 0.52
    min_clusters: int = 2
    rule_min_win_rate: float = 0.40
    random_min_win_rate: float = 0.70

    def __post_init__(self) -> None:
        for name in ("ci95_low_threshold", "rule_min_win_rate", "random_min_win_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if isinstance(self.min_clusters, bool) or int(self.min_clusters) < 2:
            raise ValueError("min_clusters must be at least 2")


def decide_promotion(
    summary: Mapping[str, object],
    *,
    candidate_identity: Mapping[str, object],
    champion_identity: Mapping[str, object],
    evaluation_hash: str,
    selection_deals_hash: str,
    policy: PromotionPolicy | None = None,
) -> dict[str, object]:
    policy = policy or PromotionPolicy()
    reasons: list[str] = []
    status = "promote"
    per_opponent = summary.get("per_opponent")
    coverage = summary.get("seat_rotation_coverage")
    if not isinstance(per_opponent, Mapping) or not isinstance(coverage, Mapping):
        raise ValueError("malformed evaluation summary")
    missing = [name for name in REQUIRED_MATRIX if name not in per_opponent]
    if missing:
        status = "inconclusive"
        reasons.append(f"fixed evaluation matrix incomplete: missing {missing}")
    if not coverage.get("complete", False):
        status = "inconclusive"
        reasons.append("seat rotation/deal cluster coverage is incomplete")
    for name in REQUIRED_MATRIX:
        result = per_opponent.get(name)
        if isinstance(result, Mapping) and not result.get("complete", False):
            status = "inconclusive"
            reasons.append(f"{name} evaluation is incomplete")
    total_illegal = sum(int(result.get("eval/illegal_count", 0)) for result in per_opponent.values() if isinstance(result, Mapping))
    total_truncated = sum(int(result.get("eval/truncated_count", 0)) for result in per_opponent.values() if isinstance(result, Mapping))
    if total_illegal or total_truncated:
        status = "reject"
        reasons.append(f"illegal/truncated counts are nonzero: {total_illegal}/{total_truncated}")
    if status == "promote" and candidate_identity.get("checkpoint_sha256") == champion_identity.get("checkpoint_sha256"):
        status = "reject"
        reasons.append("candidate is byte-identical to current champion")

    champion = per_opponent.get("champion", {})
    champion_rate = champion.get("eval/win_rate") if isinstance(champion, Mapping) else None
    champion_low = champion.get("eval/clustered_ci95_low") if isinstance(champion, Mapping) else None
    champion_clusters = int(champion.get("eval/cluster_count", 0)) if isinstance(champion, Mapping) else 0
    if status == "promote" and (champion_rate is None or champion_clusters < policy.min_clusters or champion_low is None):
        status = "inconclusive"
        reasons.append("candidate-vs-champion clustered sample is insufficient")
    if status == "promote" and float(champion_rate) <= 0.5:
        status = "reject"
        reasons.append("candidate-vs-champion point estimate is not above 50%")
    if status == "promote" and float(champion_low) < policy.ci95_low_threshold:
        status = "inconclusive"
        reasons.append("candidate-vs-champion CI lower bound does not reach the promotion threshold")

    rule_rate = per_opponent.get("rule", {}).get("eval/win_rate") if isinstance(per_opponent.get("rule"), Mapping) else None
    random_rate = per_opponent.get("random", {}).get("eval/win_rate") if isinstance(per_opponent.get("random"), Mapping) else None
    if status == "promote" and (rule_rate is None or float(rule_rate) < policy.rule_min_win_rate):
        status = "reject"
        reasons.append("candidate regresses below the predefined rule anchor tolerance")
    if status == "promote" and (random_rate is None or float(random_rate) < policy.random_min_win_rate):
        status = "reject"
        reasons.append("candidate regresses below the predefined random anchor tolerance")
    if status == "promote":
        reasons.append("all predefined promotion gates passed")
    decision: dict[str, object] = {
        "schema": PROMOTION_DECISION_SCHEMA,
        "status": status,
        "promote": status == "promote",
        "reasons": reasons,
        "policy": {
            "ci95_low_threshold": policy.ci95_low_threshold,
            "min_clusters": policy.min_clusters,
            "rule_min_win_rate": policy.rule_min_win_rate,
            "random_min_win_rate": policy.random_min_win_rate,
        },
        "candidate": dict(candidate_identity),
        "current_champion": dict(champion_identity),
        "selection_deals_hash": selection_deals_hash,
        "evaluation_hash": evaluation_hash,
        "decided_at": _timestamp(),
    }
    decision["hash"] = payload_hash(decision)
    return decision


class ChampionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, object]:
        payload = load_json(self.path)
        if payload.get("schema") != CHAMPION_MANIFEST_SCHEMA:
            raise ValueError("unsupported champion manifest schema")
        if payload.get("hash") != payload_hash(payload):
            raise ValueError("champion manifest hash mismatch")
        if not isinstance(payload.get("champion"), Mapping):
            raise ValueError("champion manifest is missing champion identity")
        return payload

    def bootstrap(self, identity: Mapping[str, object], *, reason: str = "initial_champion_bootstrap") -> dict[str, object]:
        if self.path.exists():
            return self.load()
        payload: dict[str, object] = {
            "schema": CHAMPION_MANIFEST_SCHEMA,
            "generation": 1,
            "champion": dict(identity),
            "promotion": {
                "kind": "bootstrap",
                "reason": reason,
                "selection_deals_hash": None,
                "promotion_evaluation_hash": None,
                "promoted_at": _timestamp(),
            },
            "previous_champion": None,
        }
        payload["hash"] = payload_hash(payload)
        atomic_write_json(self.path, payload)
        return payload

    def promote(
        self,
        identity: Mapping[str, object],
        *,
        decision: Mapping[str, object],
        replace_fn: Callable = os.replace,
    ) -> dict[str, object]:
        if decision.get("schema") != PROMOTION_DECISION_SCHEMA or not decision.get("promote", False):
            raise ValueError("champion update requires a successful versioned promotion decision")
        current = self.load()
        if decision.get("current_champion", {}).get("checkpoint_sha256") != current["champion"].get("checkpoint_sha256"):
            raise ValueError("promotion decision was made against a stale champion")
        payload: dict[str, object] = {
            "schema": CHAMPION_MANIFEST_SCHEMA,
            "generation": int(current["generation"]) + 1,
            "champion": dict(identity),
            "promotion": {
                "kind": "selection",
                "decision_hash": decision["hash"],
                "selection_deals_hash": decision["selection_deals_hash"],
                "promotion_evaluation_hash": decision["evaluation_hash"],
                "promoted_at": _timestamp(),
            },
            "previous_champion": current["champion"],
        }
        payload["hash"] = payload_hash(payload)
        atomic_write_json(self.path, payload, replace_fn=replace_fn)
        return payload


__all__ = [
    "CHAMPION_MANIFEST_SCHEMA", "PROMOTION_DECISION_SCHEMA", "ChampionStore",
    "PromotionPolicy", "REQUIRED_MATRIX", "decide_promotion",
]
