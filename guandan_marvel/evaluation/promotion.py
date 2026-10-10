# SPDX-License-Identifier: MIT
"""Promotion policy v2 and identity-bound atomic champion manifests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Callable, Mapping

from .checkpoints import verify_policy_identity
from .common import atomic_write_json, canonical_json_bytes, load_json, payload_hash

PROMOTION_DECISION_SCHEMA = "guandan-marvel-promotion-decision-v2"
PROMOTION_POLICY_SCHEMA = "guandan-marvel-promotion-policy-v2"
CHAMPION_MANIFEST_SCHEMA = "guandan-marvel-champion-manifest-v2"
PRE_TRANSACTION_CHAMPION_MANIFEST_SCHEMA = "guandan-marvel-champion-manifest-v1"
REQUIRED_MATRIX = ("champion", "rule", "random", "legacy", "opponent_pool")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return canonical_json_bytes(dict(left)) == canonical_json_bytes(dict(right))


@dataclass(frozen=True)
class PromotionPolicy:
    mode: str = "formal"
    ci95_low_threshold: float = 0.52
    min_clusters: int = 32
    rule_min_win_rate: float = 0.40
    random_min_win_rate: float = 0.70
    schema: str = PROMOTION_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROMOTION_POLICY_SCHEMA:
            raise ValueError("unsupported promotion policy schema")
        if self.mode not in {"formal", "smoke"}:
            raise ValueError("promotion policy mode must be formal or smoke")
        for name in ("ci95_low_threshold", "rule_min_win_rate", "random_min_win_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
        minimum = 32 if self.mode == "formal" else 1
        if isinstance(self.min_clusters, bool) or int(self.min_clusters) < minimum:
            raise ValueError(f"{self.mode} min_clusters must be at least {minimum}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None):
        if value is None:
            return cls()
        allowed = {"schema", "mode", "ci95_low_threshold", "min_clusters", "rule_min_win_rate", "random_min_win_rate"}
        return cls(**{key: item for key, item in dict(value).items() if key in allowed})

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "ci95_low_threshold": self.ci95_low_threshold,
            "min_clusters": self.min_clusters,
            "rule_min_win_rate": self.rule_min_win_rate,
            "random_min_win_rate": self.random_min_win_rate,
            "zero_variance_policy": "bounded empirical-Bernstein interval remains non-degenerate at finite n",
        }


def _finite_rate(value: object, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return result


def _validate_summary(summary: Mapping[str, object]) -> None:
    per_opponent = summary.get("per_opponent")
    coverage = summary.get("seat_rotation_coverage")
    if not isinstance(per_opponent, Mapping) or not isinstance(coverage, Mapping):
        raise ValueError("malformed evaluation summary")
    if set(per_opponent) != set(REQUIRED_MATRIX):
        raise ValueError("evaluation summary does not contain the exact fixed matrix")
    for name, result in per_opponent.items():
        if not isinstance(result, Mapping):
            raise ValueError(f"malformed {name} summary")
        _finite_rate(result.get("eval/win_rate"), f"{name}.win_rate", allow_none=True)
        _finite_rate(result.get("eval/clustered_ci95_low"), f"{name}.ci_low", allow_none=True)
        _finite_rate(result.get("eval/clustered_ci95_high"), f"{name}.ci_high", allow_none=True)
        clusters = result.get("eval/cluster_count")
        if isinstance(clusters, bool) or not isinstance(clusters, int) or clusters < 0:
            raise ValueError(f"{name}.cluster_count is invalid")
        for field in ("eval/illegal_count", "eval/truncated_count"):
            count = result.get(field)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{name}.{field} is invalid")


def decide_promotion(
    summary: Mapping[str, object],
    *,
    candidate_identity: Mapping[str, object],
    champion_manifest: Mapping[str, object],
    evaluation_hash: str,
    selection_deals_hash: str,
    plan_hash: str,
    summary_hash: str,
    selection_eligible: bool,
    policy: PromotionPolicy | None = None,
) -> dict[str, object]:
    policy = policy or PromotionPolicy()
    _validate_summary(summary)
    if champion_manifest.get("schema") not in {CHAMPION_MANIFEST_SCHEMA, PRE_TRANSACTION_CHAMPION_MANIFEST_SCHEMA}:
        raise ValueError("unsupported champion baseline schema")
    if champion_manifest.get("hash") != payload_hash(champion_manifest):
        raise ValueError("champion baseline hash mismatch")
    champion_identity = champion_manifest.get("champion")
    if not isinstance(champion_identity, Mapping):
        raise ValueError("champion baseline identity is missing")
    reasons: list[str] = []
    status = "promote"
    per_opponent = summary["per_opponent"]
    coverage = summary["seat_rotation_coverage"]

    if not selection_eligible:
        status = "inconclusive"
        reasons.append("candidate/opponent training provenance is incomplete; smoke-only evaluation cannot promote")
    if policy.mode == "smoke":
        status = "inconclusive"
        reasons.append("smoke promotion policy never performs automatic promotion")
    if not coverage.get("complete", False):
        status = "inconclusive"
        reasons.append("evaluation plan/seat/deal coverage is incomplete")
    for name in REQUIRED_MATRIX:
        if not per_opponent[name].get("complete", False):
            status = "inconclusive"
            reasons.append(f"{name} evaluation is incomplete")
    total_illegal = sum(int(result["eval/illegal_count"]) for result in per_opponent.values())
    total_truncated = sum(int(result["eval/truncated_count"]) for result in per_opponent.values())
    if total_illegal or total_truncated:
        status = "reject"
        reasons.append(f"illegal/truncated counts are nonzero: {total_illegal}/{total_truncated}")
    if status == "promote" and candidate_identity.get("checkpoint_sha256") == champion_identity.get("checkpoint_sha256"):
        status = "reject"
        reasons.append("candidate is byte-identical to current champion")

    champion = per_opponent["champion"]
    champion_rate = champion.get("eval/win_rate")
    champion_low = champion.get("eval/clustered_ci95_low")
    champion_clusters = int(champion.get("eval/cluster_count", 0))
    if status == "promote" and (champion_rate is None or champion_low is None or champion_clusters < policy.min_clusters):
        status = "inconclusive"
        reasons.append(
            f"candidate-vs-champion requires at least {policy.min_clusters} independent deal clusters with a finite bounded CI"
        )
    if status == "promote" and float(champion_rate) <= 0.5:
        status = "reject"
        reasons.append("candidate-vs-champion point estimate is not above 50%")
    if status == "promote" and float(champion_low) < policy.ci95_low_threshold:
        status = "inconclusive"
        reasons.append("candidate-vs-champion CI lower bound does not reach the promotion threshold")

    rule_rate = per_opponent["rule"].get("eval/win_rate")
    random_rate = per_opponent["random"].get("eval/win_rate")
    if status == "promote" and (rule_rate is None or float(rule_rate) < policy.rule_min_win_rate):
        status = "reject"
        reasons.append("candidate regresses below the predefined rule anchor tolerance")
    if status == "promote" and (random_rate is None or float(random_rate) < policy.random_min_win_rate):
        status = "reject"
        reasons.append("candidate regresses below the predefined random anchor tolerance")
    if status == "promote":
        reasons.append("all versioned formal promotion gates passed")

    decision: dict[str, object] = {
        "schema": PROMOTION_DECISION_SCHEMA,
        "status": status,
        "promote": status == "promote",
        "reasons": reasons,
        "policy": policy.to_dict(),
        "candidate": dict(candidate_identity),
        "current_champion": dict(champion_identity),
        "current_champion_manifest_hash": champion_manifest["hash"],
        "current_champion_generation": int(champion_manifest["generation"]),
        "evaluation_mode": "selection",
        "selection_eligible": bool(selection_eligible),
        "selection_deals_hash": str(selection_deals_hash),
        "plan_hash": str(plan_hash),
        "summary_hash": str(summary_hash),
        "evaluation_hash": str(evaluation_hash),
        "decided_at": _timestamp(),
    }
    decision["hash"] = payload_hash(decision)
    return decision


class ChampionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, object]:
        payload = load_json(self.path)
        if payload.get("schema") not in {CHAMPION_MANIFEST_SCHEMA, PRE_TRANSACTION_CHAMPION_MANIFEST_SCHEMA}:
            raise ValueError("unsupported champion manifest schema")
        if payload.get("hash") != payload_hash(payload):
            raise ValueError("champion manifest hash mismatch")
        if not isinstance(payload.get("champion"), Mapping):
            raise ValueError("champion manifest is missing champion identity")
        return payload

    def bootstrap(self, identity: Mapping[str, object], *, reason: str = "initial_champion_bootstrap") -> dict[str, object]:
        if self.path.exists():
            return self.load()
        verify_policy_identity(identity)
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
            "pool_sync": {"status": "not_required", "transaction_id": None},
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
        expected_plan_hash: str,
        expected_summary_hash: str,
        transaction_id: str,
        replace_fn: Callable = os.replace,
    ) -> dict[str, object]:
        if decision.get("schema") != PROMOTION_DECISION_SCHEMA or decision.get("hash") != payload_hash(decision):
            raise ValueError("promotion decision schema/hash is invalid")
        if decision.get("status") != "promote" or decision.get("promote") is not True:
            raise ValueError("champion update requires a successful promotion decision")
        if decision.get("evaluation_mode") != "selection" or not decision.get("selection_eligible", False):
            raise ValueError("only a provenance-complete selection decision may promote")
        if decision.get("plan_hash") != expected_plan_hash or decision.get("summary_hash") != expected_summary_hash:
            raise ValueError("promotion decision plan/summary identity mismatch")
        if not _identity_equal(identity, decision.get("candidate", {})):
            raise ValueError("promotion candidate identity differs from the approved decision")
        verified = verify_policy_identity(identity)
        if not _identity_equal(verified.identity, identity):
            raise ValueError("promotion candidate checkpoint content/identity mismatch")

        current = self.load()
        if decision.get("current_champion_manifest_hash") != current["hash"]:
            raise ValueError("promotion decision was made against a stale champion manifest")
        if int(decision.get("current_champion_generation", -1)) != int(current["generation"]):
            raise ValueError("promotion decision champion generation is stale")
        if not _identity_equal(decision.get("current_champion", {}), current["champion"]):
            raise ValueError("promotion decision champion identity is stale")
        payload: dict[str, object] = {
            "schema": CHAMPION_MANIFEST_SCHEMA,
            "generation": int(current["generation"]) + 1,
            "champion": dict(identity),
            "promotion": {
                "kind": "selection",
                "decision_hash": decision["hash"],
                "selection_deals_hash": decision["selection_deals_hash"],
                "promotion_evaluation_hash": decision["evaluation_hash"],
                "plan_hash": decision["plan_hash"],
                "summary_hash": decision["summary_hash"],
                "promoted_at": _timestamp(),
            },
            "pool_sync": {"status": "pending", "transaction_id": str(transaction_id), "pool_manifest_hash": None},
            "previous_champion": current["champion"],
        }
        payload["hash"] = payload_hash(payload)
        atomic_write_json(self.path, payload, replace_fn=replace_fn)
        return payload

    def mark_pool_synced(self, *, transaction_id: str, pool_manifest_hash: str) -> dict[str, object]:
        current = self.load()
        sync = current.get("pool_sync") or {}
        if sync.get("status") != "pending" or sync.get("transaction_id") != transaction_id:
            raise ValueError("champion has no matching pending pool transaction")
        current["pool_sync"] = {
            "status": "synced",
            "transaction_id": transaction_id,
            "pool_manifest_hash": pool_manifest_hash,
            "synced_at": _timestamp(),
        }
        current["hash"] = payload_hash(current)
        atomic_write_json(self.path, current)
        return current


__all__ = [
    "CHAMPION_MANIFEST_SCHEMA", "PRE_TRANSACTION_CHAMPION_MANIFEST_SCHEMA",
    "PROMOTION_DECISION_SCHEMA", "PROMOTION_POLICY_SCHEMA", "ChampionStore",
    "PromotionPolicy", "REQUIRED_MATRIX", "decide_promotion",
]
