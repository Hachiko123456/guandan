# SPDX-License-Identifier: MIT
"""Actual training-data provenance and pre-inference isolation checks."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

from guandan_marvel import cpp_game

from .checkpoints import V2PolicyArtifact
from .manifests import deal_game_config, deals_for, dataset_identity, validate_deals_manifest

DATA_ISOLATION_SCHEMA = "guandan-marvel-data-isolation-verification-v1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def materialize_deal_identity(row: Mapping[str, object]) -> dict[str, object]:
    config = deal_game_config(row)
    env = cpp_game.EnvBatch(1, configs=[config], auto_reset=False)
    env.reset(int(row["seed"]))
    state = env.debug_state(0)["state"]
    identity = {
        "hands": [sorted(int(card) for card in hand) for hand in state["hands"]],
        "level_rank": int(state["level_rank"]),
        "leader_seat": int(state["leader_seat"]),
        "previous_result": state.get("previous_result"),
    }
    return {
        "deal_id": str(row["deal_id"]),
        "seed": int(row["seed"]),
        "deal_sha256": _canonical_hash(identity),
    }


def verify_data_isolation(
    *,
    artifacts: Iterable[V2PolicyArtifact],
    deals_manifest: Mapping[str, object],
    domain: str,
    policy: str,
) -> dict[str, object]:
    """Reject overlap before policy inference; old checkpoints may be smoke-only."""
    if policy not in {"formal", "smoke_only"}:
        raise ValueError("provenance policy must be formal or smoke_only")
    validate_deals_manifest(deals_manifest)
    dataset_hash = dataset_identity(deals_manifest, domain)
    materialized = [materialize_deal_identity(row) for row in deals_for(deals_manifest, domain)]
    seeds = {row["seed"] for row in materialized}
    hashes = {row["deal_sha256"] for row in materialized}
    checks = []
    reasons: list[str] = []
    complete = True
    for artifact in artifacts:
        provenance = artifact.training_data_provenance
        if not provenance or not provenance.get("complete", False):
            complete = False
            known_seed = artifact.metadata.get("training_seed")
            seed_overlap = sorted(seeds & ({int(known_seed)} if known_seed is not None else set()))
            if seed_overlap:
                raise ValueError(
                    f"{domain} seed overlaps known checkpoint training seed for {artifact.sha256}: {seed_overlap}"
                )
            reasons.append(f"checkpoint {artifact.sha256} lacks complete actual training provenance")
            checks.append({
                "checkpoint_sha256": artifact.sha256,
                "provenance_status": "incomplete",
                "known_training_seed": known_seed,
                "seed_overlap": seed_overlap,
                "deal_hash_overlap": [],
            })
            continue
        if provenance.get("domain") != "training" or not provenance.get("manifest_hash"):
            raise ValueError("checkpoint training provenance domain/manifest identity is invalid")
        training_seed = provenance.get("initial_env_seed")
        training_seeds = provenance.get("initial_env_seeds")
        if training_seeds is None:
            known_training_seeds = {int(training_seed)} if training_seed is not None else set()
        elif isinstance(training_seeds, (list, tuple)):
            known_training_seeds = {int(value) for value in training_seeds}
        else:
            raise ValueError("checkpoint training provenance initial_env_seeds is malformed")
        seed_overlap = sorted(seeds & known_training_seeds)
        training_hashes = {str(value) for value in provenance.get("deal_hashes", [])}
        deal_overlap = sorted(hashes & training_hashes)
        checks.append({
            "checkpoint_sha256": artifact.sha256,
            "provenance_status": "complete",
            "training_manifest_hash": provenance.get("manifest_hash"),
            "training_generator_version": provenance.get("generator_version"),
            "seed_semantics": provenance.get("seed_semantics"),
            "training_initial_seeds": sorted(known_training_seeds),
            "seed_overlap": seed_overlap,
            "deal_hash_overlap": deal_overlap,
        })
        if seed_overlap or deal_overlap:
            raise ValueError(
                f"{domain} data overlaps actual training source for checkpoint {artifact.sha256}: "
                f"seeds={seed_overlap}, deals={deal_overlap}"
            )
    if not complete and policy == "formal":
        raise ValueError(f"formal {domain} evaluation requires complete training provenance: {reasons}")
    if domain == "final" and not complete:
        raise ValueError("final holdout evaluation never permits incomplete training provenance")
    return {
        "schema": DATA_ISOLATION_SCHEMA,
        "domain": domain,
        "dataset_hash": dataset_hash,
        "policy": policy,
        "formal_selection_eligible": complete,
        "checks": checks,
        "reasons": reasons,
        "materialized_deals": materialized,
    }


__all__ = ["DATA_ISOLATION_SCHEMA", "materialize_deal_identity", "verify_data_isolation"]
