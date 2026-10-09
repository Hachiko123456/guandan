# SPDX-License-Identifier: MIT
"""Versioned training/selection/final deal manifests and final-use audit."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from guandan.cards import PRINTED_RANKS
from guandan.environment_types import GameConfig
from guandan.state import PreviousHandResult

from .common import atomic_write_json, load_json, payload_hash

DEALS_MANIFEST_SCHEMA = "guandan-marvel-deals-manifest-v1"
FINAL_HOLDOUT_AUDIT_SCHEMA = "guandan-marvel-final-holdout-audit-v1"
DEAL_GENERATOR_VERSION = "guandan-marvel-deal-seed-generator-v1"
DOMAINS = ("training", "selection", "final")
DEFAULT_BASES = {"training": 100_000, "selection": 1_000_000_000, "final": 1_500_000_000}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _previous_payload(previous: PreviousHandResult | None) -> dict[str, object] | None:
    if previous is None:
        return None
    return {
        "ranking": list(previous.ranking),
        "winner_team": previous.winner_team,
        "outcome_class": previous.outcome_class,
        "tribute_cancelled": previous.tribute_cancelled,
    }


def _previous_from_payload(payload: Mapping[str, object] | None) -> PreviousHandResult | None:
    if payload is None:
        return None
    return PreviousHandResult(
        tuple(int(value) for value in payload["ranking"]),
        int(payload["winner_team"]),
        str(payload["outcome_class"]),
        bool(payload.get("tribute_cancelled", False)),
    )


def build_deals_manifest(
    *,
    counts: Mapping[str, int],
    bases: Mapping[str, int] | None = None,
    level_ranks: Sequence[int] = tuple(int(rank) for rank in PRINTED_RANKS),
    leader_seats: Sequence[int] = (0, 1, 2, 3),
    previous_results: Sequence[PreviousHandResult | None] = (None,),
) -> dict[str, object]:
    bases = dict(DEFAULT_BASES if bases is None else bases)
    if set(counts) != set(DOMAINS) or set(bases) != set(DOMAINS):
        raise ValueError(f"counts and bases must define exactly {DOMAINS}")
    if not level_ranks or not leader_seats or not previous_results:
        raise ValueError("level_ranks, leader_seats, and previous_results must be non-empty")
    datasets: dict[str, list[dict[str, object]]] = {}
    for domain in DOMAINS:
        count = counts[domain]
        base = bases[domain]
        if isinstance(count, bool) or int(count) < 0:
            raise ValueError(f"{domain} count must be a nonnegative integer")
        if isinstance(base, bool) or int(base) < 0:
            raise ValueError(f"{domain} seed base must be nonnegative")
        rows = []
        for index in range(int(count)):
            previous = previous_results[index % len(previous_results)]
            rows.append({
                "deal_id": f"{domain}-{index:06d}",
                "domain": domain,
                "seed": int(base) + index,
                "level_rank": int(level_ranks[index % len(level_ranks)]),
                "leader_seat": int(leader_seats[index % len(leader_seats)]),
                "previous_result": _previous_payload(previous),
            })
        datasets[domain] = rows
    payload: dict[str, object] = {
        "schema": DEALS_MANIFEST_SCHEMA,
        "generator_version": DEAL_GENERATOR_VERSION,
        "created_at": _timestamp(),
        "datasets": datasets,
        "coverage": {
            "level_ranks": sorted({int(value) for value in level_ranks}),
            "leader_seats": sorted({int(value) for value in leader_seats}),
            "tribute_modes": sorted({
                "none" if value is None else (
                    "cancelled" if value.tribute_cancelled else value.outcome_class.lower()
                ) for value in previous_results
            }),
        },
    }
    payload["hash"] = payload_hash(payload)
    validate_deals_manifest(payload)
    return payload


def validate_deals_manifest(payload: Mapping[str, object]) -> None:
    if payload.get("schema") != DEALS_MANIFEST_SCHEMA:
        raise ValueError("unsupported deals manifest schema")
    if payload.get("generator_version") != DEAL_GENERATOR_VERSION:
        raise ValueError("unsupported deal generator version")
    if payload.get("hash") != payload_hash(payload):
        raise ValueError("deals manifest hash mismatch")
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(DOMAINS):
        raise ValueError("deals manifest must contain training/selection/final datasets")
    all_seeds: dict[int, str] = {}
    all_ids: set[str] = set()
    for domain in DOMAINS:
        rows = datasets[domain]
        if not isinstance(rows, list):
            raise ValueError(f"{domain} dataset must be a list")
        for row in rows:
            if not isinstance(row, Mapping) or row.get("domain") != domain:
                raise ValueError(f"malformed {domain} deal row")
            deal_id = str(row.get("deal_id", ""))
            seed = row.get("seed")
            if not deal_id or deal_id in all_ids:
                raise ValueError("deal IDs must be non-empty and globally unique")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("deal seed must be a nonnegative integer")
            if seed in all_seeds:
                raise ValueError(f"deal seed overlap: {seed} appears in {all_seeds[seed]} and {domain}")
            all_ids.add(deal_id)
            all_seeds[seed] = domain
            GameConfig(
                level_rank=int(row["level_rank"]),
                seed=seed,
                leader_seat=int(row["leader_seat"]),
                previous_result=_previous_from_payload(row.get("previous_result")),
            )


def deals_for(payload: Mapping[str, object], domain: str) -> tuple[dict[str, object], ...]:
    validate_deals_manifest(payload)
    if domain not in DOMAINS:
        raise ValueError(f"unknown deal domain {domain!r}")
    return tuple(dict(row) for row in payload["datasets"][domain])


def deal_game_config(row: Mapping[str, object]) -> GameConfig:
    return GameConfig(
        level_rank=int(row["level_rank"]),
        seed=int(row["seed"]),
        leader_seat=int(row["leader_seat"]),
        previous_result=_previous_from_payload(row.get("previous_result")),
    )


class FinalHoldoutAudit:
    """One-shot, candidate-locked audit boundary for final holdout use."""

    def __init__(self, path: str | Path, *, deals_manifest_hash: str):
        self.path = Path(path)
        self.deals_manifest_hash = str(deals_manifest_hash)

    def initialize(self) -> dict[str, object]:
        if self.path.exists():
            return self.load()
        payload: dict[str, object] = {
            "schema": FINAL_HOLDOUT_AUDIT_SCHEMA,
            "deals_manifest_hash": self.deals_manifest_hash,
            "locked_candidate_sha256": None,
            "uses": [],
        }
        payload["hash"] = payload_hash(payload)
        atomic_write_json(self.path, payload)
        return payload

    def load(self) -> dict[str, object]:
        payload = load_json(self.path)
        if payload.get("schema") != FINAL_HOLDOUT_AUDIT_SCHEMA:
            raise ValueError("unsupported final holdout audit schema")
        if payload.get("deals_manifest_hash") != self.deals_manifest_hash:
            raise ValueError("final holdout audit references a different deals manifest")
        if payload.get("hash") != payload_hash(payload):
            raise ValueError("final holdout audit hash mismatch")
        if not isinstance(payload.get("uses"), list):
            raise ValueError("final holdout uses must be a list")
        return payload

    def lock_candidate(self, candidate_sha256: str) -> dict[str, object]:
        payload = self.initialize()
        current = payload.get("locked_candidate_sha256")
        if current not in (None, candidate_sha256):
            raise ValueError("final holdout is already locked to a different candidate")
        if payload["uses"]:
            raise ValueError("cannot relock final holdout after use")
        payload["locked_candidate_sha256"] = candidate_sha256
        payload["hash"] = payload_hash(payload)
        atomic_write_json(self.path, payload)
        return payload

    def record_use(self, *, candidate_sha256: str, evaluation_hash: str, purpose: str) -> dict[str, object]:
        if purpose != "final_audit":
            raise ValueError("final holdout may only be used for final_audit")
        payload = self.load()
        if payload.get("locked_candidate_sha256") != candidate_sha256:
            raise ValueError("final holdout candidate identity mismatch")
        if payload["uses"]:
            raise ValueError("final holdout has already been consumed; generate a new version")
        payload["uses"].append({
            "used_at": _timestamp(),
            "candidate_sha256": candidate_sha256,
            "evaluation_hash": evaluation_hash,
            "purpose": purpose,
        })
        payload["hash"] = payload_hash(payload)
        atomic_write_json(self.path, payload)
        return payload


__all__ = [
    "DEALS_MANIFEST_SCHEMA", "DEAL_GENERATOR_VERSION", "DOMAINS", "FinalHoldoutAudit",
    "build_deals_manifest", "deal_game_config", "deals_for", "validate_deals_manifest",
]
