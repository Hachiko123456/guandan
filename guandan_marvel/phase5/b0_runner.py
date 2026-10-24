# SPDX-License-Identifier: MIT
"""Assemble the B0 dataset, run the A/B/C/D comparison and apply the gate.

This module contains no policy training.  It consumes the frozen-policy
collection from :mod:`guandan_marvel.phase5.belief_collect`, builds the locked
group split, runs the four pre-registered baselines on CUDA, and returns a gate
artifact.  All artifact hashes are computed from the actual data and source
files, never from a static declaration.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .belief_data import (
    PublicLedger, dataset_hash, group_split, leakage_probe_identity,
    public_ledger_features, public_ledger_snapshot, split_hash,
)
from .belief_probe import (
    b0_gate, evaluate_bucket_results, run_b0_probes, split_indices,
    verify_group_disjoint,
)
from .design import (
    PHASE5_B0_PLAN, PHASE5_PUBLIC_LEDGER_SCHEMA, PUBLIC_FEATURE_WIDTH,
    TOTAL_PHYSICAL_CARDS, phase5_design_hash,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMPONENTS = (
    "guandan_marvel/phase5",
    "guandan_marvel/config.py",
    "guandan_marvel/model/agent.py",
    "guandan_marvel/training/checkpoints.py",
    "scripts/run_phase5_b0.py",
    "scripts/run_phase5_guarded.py",
    "scripts/run_phase5_collection_pilot.py",
    "scripts/run_phase5_symmetry_diagnostic.py",
)


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8",
    )
    return hashlib.sha256(payload).hexdigest()


def source_hash(root: str | Path = ROOT) -> str:
    """Hash every source file that can change B0 data, probes or the gate."""
    base = Path(root)
    files: list[Path] = []
    for component in SOURCE_COMPONENTS:
        path = base / component
        if path.is_dir():
            files.extend(sorted(path.glob("*.py")))
        elif path.is_file():
            files.append(path)
    entries = []
    for path in sorted(set(files)):
        relative = path.relative_to(base).as_posix()
        entries.append({
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        })
    return canonical_json_hash(entries)


def config_hash(config: object) -> str:
    if config is None:
        return canonical_json_hash({"config": None})
    if isinstance(config, Mapping):
        value = dict(config)
    elif is_dataclass(config):
        value = asdict(config)
    elif hasattr(config, "to_dict"):
        value = config.to_dict()
    else:
        value = {"repr": repr(config)}
    return canonical_json_hash(value)


def build_dataset(
    samples: Sequence[Any],
    *,
    split_seed: int | None = None,
    fractions: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Stack the model-visible inputs/labels and assign the locked group split."""
    if not samples:
        raise ValueError("B0 dataset requires at least one belief sample")
    from .belief_collect import sample_matrix

    matrix = sample_matrix(samples)
    for key in ("phase", "level_rank", "leader_seat", "tribute_mode"):
        matrix[key] = np.asarray([str(getattr(sample, key)) for sample in samples])
    plan = PHASE5_B0_PLAN
    assignment = group_split(
        matrix["groups"],
        seed=int(plan["split_seed"] if split_seed is None else split_seed),
        fractions=(
            dict(plan["split"]) if fractions is None else dict(fractions)
        ),
    )
    records = [sample.to_record() for sample in samples]
    return {
        "matrix": matrix,
        "assignment": assignment,
        "split_hash": split_hash(assignment),
        "records_hash": dataset_hash(records),
        "sample_count": len(samples),
        "group_count": int(len(set(matrix["groups"].tolist()))),
    }


def sufficiency(
    matrix: Mapping[str, np.ndarray],
    assignment: Mapping[str, str],
    *,
    completed_deals: int,
    collection_truncated: bool,
) -> dict[str, Any]:
    indices = split_indices(matrix["groups"], assignment)
    test = indices["test"]
    metadata = {
        key: sorted(set(np.asarray(matrix[key])[test].tolist()))
        for key in ("level_rank", "leader_seat", "tribute_mode")
        if key in matrix
    }
    return {
        "completed_deals": int(completed_deals),
        "collection_truncated": bool(collection_truncated),
        "train_samples": int(indices["train"].size),
        "validation_samples": int(indices["validation"].size),
        "test_samples": int(test.size),
        "test_deals": int(len(set(np.asarray(matrix["groups"])[test].tolist()))),
        "test_level_ranks": int(len(metadata.get("level_rank", []))),
        "test_leader_seats": int(len(metadata.get("leader_seat", []))),
        "test_tribute_modes": int(len(metadata.get("tribute_mode", []))),
        "test_level_values": metadata.get("level_rank", []),
        "test_leader_values": metadata.get("leader_seat", []),
        "test_tribute_values": metadata.get("tribute_mode", []),
    }


def leakage_audit(samples: Sequence[Any]) -> dict[str, Any]:
    """Runtime no-leakage audit over the model-visible inputs.

    * the v2 public-ledger features must replay exactly from the stored visible
      causal event trajectory plus the actor's channel row (no hidden hand);
    * the leakage identity must recompute from the visible bytes;
    * physical conservation and finished-player zeros must hold;
    * every ledger bucket must be non-negative and conserve the physical cards;
    * the stored ``public_cards`` bits must equal the replay's public set.
    """
    failures: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        public = np.asarray(sample.public)
        if public.shape != (PUBLIC_FEATURE_WIDTH,):
            failures.append({"index": index, "check": "public_feature_width"})
            continue
        if sample.audit.get("public_ledger_schema") != PHASE5_PUBLIC_LEDGER_SCHEMA:
            failures.append({"index": index, "check": "public_ledger_schema"})
        try:
            ledger = PublicLedger.from_records(sample.public_events)
            snapshot = public_ledger_snapshot(
                sample.channels, actor=sample.actor, ledger=ledger,
            )
            recomputed = snapshot["features"]
        except (ValueError, TypeError) as error:
            failures.append({
                "index": index, "check": "public_ledger_replay",
                "error": f"{type(error).__name__}: {error}",
            })
            recomputed = None
            snapshot = None
        if recomputed is not None and not np.array_equal(recomputed, public):
            failures.append({"index": index, "check": "public_features"})
        if snapshot is not None:
            if not bool(snapshot["ledger_conserved"]):
                failures.append({"index": index, "check": "ledger_conservation"})
            replay_public = np.zeros(TOTAL_PHYSICAL_CARDS, dtype=np.uint8)
            for card in ledger.public_cards():
                replay_public[int(card)] = 1
            if not np.array_equal(replay_public, np.asarray(sample.public_cards, dtype=np.uint8)):
                failures.append({"index": index, "check": "public_card_bits"})
            if int(sample.audit.get("public_event_count", -1)) != len(ledger.events):
                failures.append({"index": index, "check": "public_event_count"})
        identity = leakage_probe_identity(sample.tokens, sample.channels)
        if sample.audit.get("leakage_identity") != identity:
            failures.append({"index": index, "check": "leakage_identity"})
        if not bool(sample.audit.get("conserved")):
            failures.append({"index": index, "check": "conservation"})
        if not bool(sample.audit.get("finished_players_zero")):
            failures.append({"index": index, "check": "finished_zero"})
        if not bool(sample.audit.get("known_owner_lower_bound")):
            failures.append({"index": index, "check": "known_owner_lower_bound"})
    return {
        "pass": not failures,
        "sample_count": len(samples),
        "failures": failures,
        "checks": [
            "public_ledger_replays_from_visible_event_trajectory",
            "public_features_reproduce_from_visible_bytes",
            "public_card_bits_match_ledger",
            "ledger_conservation",
            "known_owner_lower_bound",
            "leakage_identity_reproduces",
            "physical_conservation",
            "finished_players_zero",
        ],
    }


def evaluate_b0(
    samples: Sequence[Any],
    *,
    completed_deals: int,
    collection_truncated: bool,
    device: str,
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    max_probe_seconds: float,
    epsilon: float | None = None,
    bootstrap: Mapping[str, Any] | None = None,
    source_hash_value: str | None = None,
    config_hash_value: str | None = None,
) -> dict[str, Any]:
    """Build the locked dataset, run probes and return the sealed gate payload."""
    dataset = build_dataset(samples)
    matrix = dataset["matrix"]
    assignment = dataset["assignment"]
    audit = leakage_audit(samples)
    sufficient = sufficiency(
        matrix, assignment, completed_deals=completed_deals,
        collection_truncated=collection_truncated,
    )
    group_disjoint = verify_group_disjoint(matrix["groups"], assignment)
    probe = run_b0_probes(
        matrix, assignment, device=device, seeds=seeds, epochs=epochs,
        batch_size=batch_size, learning_rate=learning_rate,
        weight_decay=weight_decay, patience=patience,
        max_probe_seconds=max_probe_seconds,
    )
    gate_bootstrap = dict(
        PHASE5_B0_PLAN["gate"]["bootstrap"] if bootstrap is None else bootstrap
    )
    buckets = evaluate_bucket_results(probe, bootstrap=gate_bootstrap)
    gate = b0_gate(
        probe=probe,
        bootstrap=gate_bootstrap,
        epsilon=float(
            PHASE5_B0_PLAN["gate"]["epsilon"] if epsilon is None else epsilon
        ),
        sufficiency=sufficient,
        leakage_pass=bool(audit["pass"]),
        group_disjoint=bool(group_disjoint),
        bucket_results=buckets,
        design_hash=phase5_design_hash(),
        source_hash=source_hash_value or source_hash(),
        config_hash=config_hash_value or canonical_json_hash({"config": "unset"}),
        dataset={
            "records_hash": dataset["records_hash"],
            "split_hash": dataset["split_hash"],
            "sample_count": dataset["sample_count"],
            "group_count": dataset["group_count"],
        },
    )
    return {
        "gate": gate,
        "probe": probe,
        "dataset": {
            "split_assignment": dict(sorted(assignment.items())),
            "split_hash": dataset["split_hash"],
            "records_hash": dataset["records_hash"],
            "sample_count": dataset["sample_count"],
            "group_count": dataset["group_count"],
        },
        "leakage_audit": audit,
        "group_disjoint": bool(group_disjoint),
        "sufficiency": sufficient,
    }


__all__ = [
    "SOURCE_COMPONENTS", "build_dataset", "canonical_json_hash", "config_hash",
    "evaluate_b0", "leakage_audit", "source_hash", "sufficiency",
]
