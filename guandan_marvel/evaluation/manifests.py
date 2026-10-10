# SPDX-License-Identifier: MIT
"""Versioned training/selection/final deal manifests and final-use state machine."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

from guandan.cards import PRINTED_RANKS
from guandan.environment_types import GameConfig
from guandan.state import PreviousHandResult

from .common import atomic_write_json, load_json, payload_hash

DEALS_MANIFEST_SCHEMA = "guandan-marvel-deals-manifest-v2"
PRE_DATASET_IDENTITY_DEALS_MANIFEST_SCHEMA = "guandan-marvel-deals-manifest-v1"
FINAL_HOLDOUT_AUDIT_SCHEMA = "guandan-marvel-final-holdout-audit-v2"
DEAL_GENERATOR_VERSION = "guandan-marvel-deal-seed-generator-v1"
DOMAINS = ("training", "selection", "final")
DEFAULT_BASES = {"training": 100_000, "selection": 1_000_000_000, "final": 1_500_000_000}
DOMAIN_SEMANTICS = {
    "training": "seed is the initial EnvBatch stream seed anchor; subsequent auto-reset deals come from each slot's continuous PRNG stream and are identified by checkpoint deal hashes",
    "selection": "each row seed independently initializes one fixed evaluation deal",
    "final": "each row seed independently initializes one fixed holdout deal",
}
_FINAL_STATES = frozenset({"unused", "reserved", "running", "consumed", "failed", "interrupted"})


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


def _dataset_hash(generator_version: str, domain: str, rows: Sequence[Mapping[str, object]], semantics: str) -> str:
    return payload_hash({
        "schema": "guandan-marvel-deal-dataset-identity-v1",
        "generator_version": generator_version,
        "domain": domain,
        "seed_semantics": semantics,
        "deals": [dict(row) for row in rows],
    })


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
        "domain_seed_semantics": dict(DOMAIN_SEMANTICS),
        "dataset_hashes": {
            domain: _dataset_hash(DEAL_GENERATOR_VERSION, domain, datasets[domain], DOMAIN_SEMANTICS[domain])
            for domain in DOMAINS
        },
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
    schema = payload.get("schema")
    if schema not in {DEALS_MANIFEST_SCHEMA, PRE_DATASET_IDENTITY_DEALS_MANIFEST_SCHEMA}:
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
    if schema == DEALS_MANIFEST_SCHEMA:
        declared = payload.get("dataset_hashes")
        if not isinstance(declared, Mapping) or set(declared) != set(DOMAINS):
            raise ValueError("deals manifest v2 requires per-domain dataset hashes")
        for domain in DOMAINS:
            semantics = payload.get("domain_seed_semantics")
            if not isinstance(semantics, Mapping) or semantics.get(domain) != DOMAIN_SEMANTICS[domain]:
                raise ValueError(f"{domain} seed semantics are missing or unsupported")
            expected = _dataset_hash(str(payload["generator_version"]), domain, datasets[domain], str(semantics[domain]))
            if declared.get(domain) != expected:
                raise ValueError(f"{domain} dataset identity hash mismatch")


def dataset_identity(payload: Mapping[str, object], domain: str) -> str:
    validate_deals_manifest(payload)
    if domain not in DOMAINS:
        raise ValueError(f"unknown deal domain {domain!r}")
    if payload.get("schema") != DEALS_MANIFEST_SCHEMA:
        raise ValueError("legacy deals manifest lacks stable dataset identity")
    return str(payload["dataset_hashes"][domain])


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
    """Atomic one-shot state machine for a stable final dataset identity."""

    def __init__(self, path: str | Path, *, deals_manifest: Mapping[str, object]):
        validate_deals_manifest(deals_manifest)
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.manifest_hash = str(deals_manifest["hash"])
        self.dataset_hash = dataset_identity(deals_manifest, "final")
        self.generator_version = str(deals_manifest["generator_version"])
        self.deal_count = len(deals_for(deals_manifest, "final"))

    @contextmanager
    def _exclusive(self, *, timeout_seconds: float = 10.0):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout_seconds
        descriptor = None
        while descriptor is None:
            try:
                descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for final holdout audit lock")
                time.sleep(0.01)
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.close(descriptor)
            descriptor = None
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)

    def _validate(self, payload: Mapping[str, object]) -> dict[str, object]:
        if payload.get("schema") != FINAL_HOLDOUT_AUDIT_SCHEMA:
            raise ValueError("unsupported final holdout audit schema")
        identity = payload.get("holdout_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("final holdout identity is missing")
        if identity.get("dataset_hash") != self.dataset_hash or identity.get("generator_version") != self.generator_version:
            raise ValueError("final holdout audit references different holdout data")
        if payload.get("hash") != payload_hash(payload):
            raise ValueError("final holdout audit hash mismatch")
        if payload.get("state") not in _FINAL_STATES:
            raise ValueError("final holdout audit state is invalid")
        if not isinstance(payload.get("history"), list):
            raise ValueError("final holdout history must be a list")
        return dict(payload)

    def initialize(self) -> dict[str, object]:
        with self._exclusive():
            if self.path.exists():
                return self._validate(load_json(self.path))
            payload: dict[str, object] = {
                "schema": FINAL_HOLDOUT_AUDIT_SCHEMA,
                "state": "unused",
                "holdout_identity": {
                    "manifest_hash_at_registration": self.manifest_hash,
                    "dataset_hash": self.dataset_hash,
                    "generator_version": self.generator_version,
                    "deal_count": self.deal_count,
                },
                "locked_candidate_sha256": None,
                "active_evaluation": None,
                "history": [],
            }
            payload["hash"] = payload_hash(payload)
            atomic_write_json(self.path, payload)
            return payload

    def load(self) -> dict[str, object]:
        return self._validate(load_json(self.path))

    def _mutate(self, function) -> dict[str, object]:
        with self._exclusive():
            payload = self._validate(load_json(self.path))
            function(payload)
            payload["hash"] = payload_hash(payload)
            atomic_write_json(self.path, payload)
            return payload

    def lock_candidate(self, candidate_sha256: str) -> dict[str, object]:
        if not candidate_sha256:
            raise ValueError("candidate SHA-256 must be non-empty")
        def apply(payload):
            if payload["state"] != "unused":
                raise ValueError("cannot lock final candidate after reservation/use")
            current = payload.get("locked_candidate_sha256")
            if current not in (None, candidate_sha256):
                raise ValueError("final holdout is already locked to a different candidate")
            payload["locked_candidate_sha256"] = candidate_sha256
        return self._mutate(apply)

    def reserve(
        self,
        *,
        evaluation_id: str,
        candidate_sha256: str,
        plan_identity_hash: str,
        config_hash: str,
    ) -> dict[str, object]:
        if not all(str(value) for value in (evaluation_id, candidate_sha256, plan_identity_hash, config_hash)):
            raise ValueError("final reservation identity fields must be non-empty")
        def apply(payload):
            if payload.get("locked_candidate_sha256") != candidate_sha256:
                raise ValueError("final holdout candidate identity mismatch")
            if payload["state"] != "unused":
                raise ValueError(f"final holdout is not unused: state={payload['state']}")
            if any(item.get("evaluation_id") == evaluation_id for item in payload["history"]):
                raise ValueError("final evaluation_id has already been used")
            active = {
                "evaluation_id": evaluation_id,
                "candidate_sha256": candidate_sha256,
                "plan_identity_hash": plan_identity_hash,
                "config_hash": config_hash,
                "reserved_at": _timestamp(),
            }
            payload["state"] = "reserved"
            payload["active_evaluation"] = active
            payload["history"].append({**active, "event": "reserved"})
        return self._mutate(apply)

    def mark_running(self, *, evaluation_id: str) -> dict[str, object]:
        def apply(payload):
            active = payload.get("active_evaluation") or {}
            if payload["state"] != "reserved" or active.get("evaluation_id") != evaluation_id:
                raise ValueError("final evaluation is not the matching reserved run")
            payload["state"] = "running"
            payload["history"].append({"evaluation_id": evaluation_id, "event": "running", "at": _timestamp()})
        return self._mutate(apply)

    def finish(
        self,
        *,
        evaluation_id: str,
        status: str,
        evaluation_hash: str | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        if status not in {"consumed", "failed", "interrupted"}:
            raise ValueError("final terminal status must be consumed/failed/interrupted")
        def apply(payload):
            active = payload.get("active_evaluation") or {}
            if payload["state"] not in {"reserved", "running"} or active.get("evaluation_id") != evaluation_id:
                raise ValueError("final evaluation terminal update does not match active reservation")
            payload["state"] = status
            payload["history"].append({
                "evaluation_id": evaluation_id,
                "event": status,
                "evaluation_hash": evaluation_hash,
                "error": error,
                "at": _timestamp(),
            })
            payload["active_evaluation"] = {**active, "terminal_status": status, "evaluation_hash": evaluation_hash, "error": error}
        return self._mutate(apply)


__all__ = [
    "DEALS_MANIFEST_SCHEMA", "DEAL_GENERATOR_VERSION", "DOMAIN_SEMANTICS", "DOMAINS", "FinalHoldoutAudit",
    "PRE_DATASET_IDENTITY_DEALS_MANIFEST_SCHEMA", "build_deals_manifest", "dataset_identity",
    "deal_game_config", "deals_for", "validate_deals_manifest",
]
