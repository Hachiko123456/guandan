# SPDX-License-Identifier: MIT
"""Offline audit of the 4096 delivered B0 samples under the 05E global caps.

This is evidence normalisation, not a new experiment: no deal is collected, no
probe is fitted, no split/seed/epsilon/threshold changes.  It proves

* every delivered public ledger satisfies the new global physical-entity caps
  (``sum_s K[s,r] <= C_r`` and ``sum_s K[s,r] + u[r] <= C_r``);
* Baseline B's prediction is bitwise identical to the pre-05E formula;
* the canonical records hash reproduces from the delivered JSONL, while the old
  int-key serialisation reproduces the historical ``dataset_records_hash``.

The original ``b0_r4.*`` files are read-only inputs and are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B0 = ROOT / "runs" / "phase5" / "b0"
OUTPUT = ROOT / "runs" / "phase5" / "closeout_05e" / "derived_records_hash_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_baseline_b(public: np.ndarray) -> np.ndarray:
    """Pre-05E Baseline B formula, implemented independently of production."""
    remaining = public[:, 0:3]
    known = public[:, 3:48].reshape(-1, 3, 15)
    unknown = public[:, 48:63]
    escrow = public[:, 63]
    max_copies = np.asarray([8] * 13 + [2, 2], dtype=np.float64)
    hand_capacity = remaining - known.sum(axis=2)
    unknown_total = unknown.sum(axis=1)
    capacity_total = hand_capacity.sum(axis=1) + escrow
    if np.any(np.abs(unknown_total - capacity_total) > 1e-9):
        raise ValueError("reference ledger does not conserve")
    allocation = np.zeros_like(known)
    positive = unknown_total > 0
    allocation[positive] = (
        hand_capacity[positive, :, None]
        * unknown[positive, None, :]
        / unknown_total[positive, None, None]
    )
    prediction = known + allocation
    return np.minimum(np.maximum(prediction, 0.0), max_copies[None, None, :])


def old_dataset_hash(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def restore_int_keys(record: dict) -> dict:
    """Undo the JSON key stringification for the nested int-key mappings."""
    restored = json.loads(json.dumps(record))
    ledger = restored.get("ledger")
    if isinstance(ledger, dict) and isinstance(ledger.get("known_owner"), dict):
        ledger["known_owner"] = {
            int(key): value for key, value in ledger["known_owner"].items()
        }
    return restored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(B0 / "b0_r4.samples.jsonl"))
    parser.add_argument("--result", default=str(B0 / "b0_r4.json"))
    parser.add_argument("--gate", default=str(B0 / "b0_r4.gate.json"))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from guandan_marvel.phase5.belief_data import (
        RECORDS_CANONICALIZATION, dataset_hash,
    )
    from guandan_marvel.phase5.belief_probe import BaselineB

    started = time.perf_counter()
    samples = Path(args.samples).resolve()
    result_path = Path(args.result).resolve()
    gate_path = Path(args.gate).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(records) != 4096:
        raise SystemExit(f"expected 4096 delivered samples, got {len(records)}")
    public = np.asarray([record["public"] for record in records], dtype=np.float64)
    if public.shape != (4096, 64):
        raise SystemExit(f"unexpected public feature shape {public.shape}")

    # 1. Every delivered ledger must satisfy the new global caps.
    BaselineB().predict(public)  # raises on any malformed ledger
    # 2. Bitwise comparison against an independent pre-05E formula.
    reference = reference_baseline_b(public)
    produced = BaselineB().predict(public)
    if not np.array_equal(reference, produced):
        difference = np.abs(reference - produced)
        raise SystemExit(
            "Baseline B prediction changed under 05E: max abs difference "
            f"{float(difference.max())} at {np.unravel_index(int(difference.argmax()), difference.shape)}"
        )
    # 3. Canonical hash reproduces from the delivered JSONL; the old int-key
    #    form reproduces the historical records hash (root-cause evidence).
    canonical_hash = dataset_hash(records)
    roundtrip_hash = dataset_hash([json.loads(json.dumps(record)) for record in records])
    if canonical_hash != roundtrip_hash:
        raise SystemExit("canonical records hash is not stable under JSON roundtrip")
    in_memory = [restore_int_keys(record) for record in records]
    historical = old_dataset_hash(in_memory)
    declared = str(result.get("dataset_records_hash"))
    if historical != declared:
        raise SystemExit(
            f"old int-key serialisation does not reproduce the historical hash: "
            f"{historical} != {declared}"
        )
    if canonical_hash == historical:
        raise SystemExit("canonicalisation did not change the affected hash")
    primary = gate.get("primary", {})
    audit = {
        "schema": "guandan-marvel-phase5-derived-provenance-v1",
        "kind": "records_hash_canonicalisation_and_global_entity_caps",
        "status": "PASS",
        "origin": {
            "samples_path": str(samples),
            "samples_sha256": sha256(samples),
            "result_path": str(result_path),
            "result_sha256": sha256(result_path),
            "gate_path": str(gate_path),
            "gate_file_sha256": sha256(gate_path),
            "gate_hash": gate.get("hash", result.get("b0_gate_hash")),
            "sample_count": len(records),
            "b0_gate": result.get("b0_gate"),
            "design_hash": result.get("design_hash"),
            "old_source_hash": result.get("source_hash"),
            "old_config_hash": result.get("config_hash"),
            "old_dataset_split_hash": result.get("dataset_split_hash"),
            "old_dataset_records_hash": declared,
        },
        "records_canonicalization": {
            "version": RECORDS_CANONICALIZATION,
            "new_canonical_hash": canonical_hash,
            "json_roundtrip_hash": roundtrip_hash,
            "historical_int_key_hash": historical,
            "root_cause": (
                "json.dumps(sort_keys=True) sorted in-memory integer map keys "
                "numerically and reloaded string keys lexicographically"
            ),
            "conversion_source_sha256": {
                "guandan_marvel/phase5/belief_data.py": sha256(
                    ROOT / "guandan_marvel/phase5/belief_data.py"
                ),
                "guandan_marvel/phase5/belief_probe.py": sha256(
                    ROOT / "guandan_marvel/phase5/belief_probe.py"
                ),
            },
        },
        "global_entity_caps": {
            "rule": "sum_s K[s,r] <= C_r and sum_s K[s,r] + u[r] <= C_r",
            "C_r": [8] * 13 + [2, 2],
            "samples_checked": int(public.shape[0]),
            "all_samples_legal": True,
            "predictions_bitwise_identical": True,
            "prediction_element_count": int(produced.size),
            "max_abs_difference_vs_reference": 0.0,
        },
        "gate_science": {
            "verdict": gate.get("verdict"),
            "epsilon": gate.get("epsilon"),
            "primary_mean_improvement": primary.get("mean_improvement"),
            "primary_ci": primary.get("ci"),
            "seed_improvements": primary.get("seed_improvements"),
            "unchanged": True,
            "note": (
                "The delivered gate's scientific numbers are unchanged: this audit "
                "only normalises record hashing and tightens malformed-input "
                "rejection. The new global source hash differs because training "
                "code changed; it is not claimed that the new source ran 512 deals."
            ),
        },
        "lineage": {
            "original_files_rewritten": False,
            "recollection": False,
            "probe_retrained": False,
            "confirmation_set_reopened": False,
            "epsilon_or_seed_changed": False,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "canonical_hash": canonical_hash,
        "historical_hash": historical,
        "samples_sha256": audit["origin"]["samples_sha256"],
        "gate_file_sha256": audit["origin"]["gate_file_sha256"],
        "predictions_bitwise_identical": True,
        "wall_seconds": audit["wall_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
