# SPDX-License-Identifier: MIT
"""M09 paired IPPO-vs-VRPO selection on locked deal clusters."""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from guandan_marvel.evaluation.checkpoints import load_v2_policy
from guandan_marvel.evaluation.host import play_game
from guandan_marvel.evaluation.manifests import dataset_identity, deals_for
from guandan_marvel.evaluation.stats import clustered_mean_ci95

from .common import append_jsonl, atomic_json, canonical_hash, sha256_file
from .evaluation import build_bounded_selection_manifest
from .telemetry import runtime_environment

ALGORITHM_SELECTION_MANIFEST_SCHEMA = "guandan-marvel-m09-algorithm-selection-manifest-v1"
ALGORITHM_SELECTION_SUMMARY_SCHEMA = "guandan-marvel-m09-algorithm-selection-summary-v1"
ALGORITHM_SELECTION_DECISION_SCHEMA = "guandan-marvel-m09-algorithm-selection-decision-v1"
PAIRED_DIFFERENCE_SCHEMA = "guandan-marvel-paired-cluster-difference-v1"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _safe_file(base: Path, spec: Mapping[str, object], *, label: str) -> Path:
    relative = Path(str(spec.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path is unsafe")
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise FileNotFoundError(f"{label} file is missing")
    expected = str(spec.get("sha256", ""))
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def validate_selection_manifest(payload: Mapping[str, object], *, path: Path) -> dict[str, object]:
    if payload.get("schema") != ALGORITHM_SELECTION_MANIFEST_SCHEMA:
        raise ValueError("unsupported algorithm-selection manifest schema")
    declared = str(payload.get("manifest_hash", ""))
    actual = canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    if declared != actual:
        raise ValueError("algorithm-selection manifest hash mismatch")
    algorithms = payload.get("algorithms")
    anchors = payload.get("rule_anchors")
    selection = payload.get("selection_dataset")
    gate = payload.get("gate")
    if not all(isinstance(value, Mapping) for value in (algorithms, anchors, selection, gate)):
        raise ValueError("algorithm-selection manifest objects are incomplete")
    if set(algorithms) != {"ippo", "vrpo"} or set(anchors) != {"ippo", "vrpo"}:
        raise ValueError("algorithm selection requires IPPO and VRPO artifacts/anchors")
    if gate.get("direct_candidate") != "vrpo" or gate.get("direct_opponent") != "ippo":
        raise ValueError("direct selection orientation must be VRPO candidate vs IPPO baseline")
    if int(selection.get("base_deals", 0)) < 32:
        raise ValueError("algorithm selection requires at least 32 independent deal clusters")
    base = path.parent.resolve()
    resolved = {
        "ippo_checkpoint": _safe_file(base, algorithms["ippo"], label="IPPO checkpoint"),
        "vrpo_checkpoint": _safe_file(base, algorithms["vrpo"], label="VRPO checkpoint"),
        "ippo_anchor": _safe_file(base, anchors["ippo"], label="IPPO rule anchor"),
        "vrpo_anchor": _safe_file(base, anchors["vrpo"], label="VRPO rule anchor"),
    }
    return {"manifest": dict(payload), **resolved}


def _cluster_ci(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row.get("status") != "complete" or type(row.get("candidate_win")) is not bool:
            raise ValueError("head-to-head record is incomplete")
        grouped[str(row["deal_id"])].append(float(bool(row["candidate_win"])))
    if any(len(values) != 4 for values in grouped.values()):
        raise ValueError("each direct head-to-head deal must contain four seat rotations")
    values = [sum(grouped[key]) / 4.0 for key in sorted(grouped)]
    return {**clustered_mean_ci95(values), "cluster_values": values}


def _load_rule_anchor(path: Path, *, expected_checkpoint_sha256: str) -> dict[tuple[str, int], bool]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: dict[tuple[str, int], bool] = {}
    for row in rows:
        if row.get("opponent_id") != "rule":
            continue
        if row.get("status") != "complete" or row.get("candidate_checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError(f"rule-anchor identity/status mismatch: {path}")
        key = (str(row["deal_id"]), int(row["seat_rotation"]))
        if key in selected or type(row.get("candidate_win")) is not bool:
            raise ValueError(f"rule-anchor duplicate/malformed row: {path}")
        selected[key] = bool(row["candidate_win"])
    if len(selected) == 0:
        raise ValueError(f"rule-anchor has no rule records: {path}")
    return selected


def _paired_rule_difference(
    *, ippo_path: Path, vrpo_path: Path, ippo_sha256: str, vrpo_sha256: str,
) -> dict[str, object]:
    ippo = _load_rule_anchor(ippo_path, expected_checkpoint_sha256=ippo_sha256)
    vrpo = _load_rule_anchor(vrpo_path, expected_checkpoint_sha256=vrpo_sha256)
    if set(ippo) != set(vrpo):
        raise ValueError("IPPO/VRPO rule anchors do not have exact paired coverage")
    grouped: dict[str, list[float]] = defaultdict(list)
    for deal_id, rotation in sorted(ippo):
        grouped[deal_id].append(float(vrpo[(deal_id, rotation)]) - float(ippo[(deal_id, rotation)]))
    if any(len(values) != 4 for values in grouped.values()):
        raise ValueError("paired rule anchors require four rotations per deal")
    differences = [sum(grouped[key]) / 4.0 for key in sorted(grouped)]
    transformed = [(value + 1.0) / 2.0 for value in differences]
    ci = clustered_mean_ci95(transformed)
    low = None if ci["low"] is None else 2.0 * float(ci["low"]) - 1.0
    high = None if ci["high"] is None else 2.0 * float(ci["high"]) - 1.0
    return {
        "schema": PAIRED_DIFFERENCE_SCHEMA,
        "method": "paired-deal-mean bounded-empirical-Bernstein-95 via [-1,1] to [0,1] transform",
        "clusters": len(differences),
        "mean_difference_vrpo_minus_ippo": sum(differences) / len(differences),
        "ci95_low": low,
        "ci95_high": high,
        "cluster_differences": differences,
        "paired_rows": len(ippo),
    }


def _decision(direct: Mapping[str, object], rule: Mapping[str, object]) -> dict[str, object]:
    low = direct.get("low")
    high = direct.get("high")
    reason: str
    if isinstance(low, (int, float)) and float(low) > 0.5:
        selected, status = "vrpo", "select_vrpo_direct"
        reason = "VRPO direct head-to-head clustered CI lower bound is above 0.5"
    elif isinstance(high, (int, float)) and float(high) < 0.5:
        selected, status = "ippo", "select_ippo_direct"
        reason = "VRPO direct head-to-head clustered CI upper bound is below 0.5"
    else:
        rule_low = rule.get("ci95_low")
        rule_high = rule.get("ci95_high")
        if isinstance(rule_low, (int, float)) and float(rule_low) > 0.0:
            selected, status = "vrpo", "select_vrpo_paired_rule"
            reason = "direct head-to-head was inconclusive; paired rule-anchor difference favors VRPO"
        elif isinstance(rule_high, (int, float)) and float(rule_high) < 0.0:
            selected, status = "ippo", "select_ippo_paired_rule"
            reason = "direct head-to-head was inconclusive; paired rule-anchor difference favors IPPO"
        else:
            selected, status = "ippo", "retain_ippo_inconclusive"
            reason = "direct and paired rule-anchor intervals are inconclusive; retain the locked M08 IPPO baseline"
    body = {
        "schema": ALGORITHM_SELECTION_DECISION_SCHEMA,
        "status": status,
        "selected_algorithm": selected,
        "reason": reason,
        "direct_head_to_head": dict(direct),
        "paired_rule_anchor": dict(rule),
        "final_holdout_used": False,
        "champion_promotion_performed": False,
        "strength_claim": False,
        "created_at": _timestamp(),
    }
    return {**body, "decision_hash": canonical_hash(body)}


def run_algorithm_selection(manifest_path: str | Path, output_root: str | Path, *, device: str = "cuda:0") -> dict[str, object]:
    path = Path(manifest_path).resolve()
    validated = validate_selection_manifest(_load_object(path), path=path)
    manifest = validated["manifest"]
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("M09 algorithm selection requires a real Kaggle T4x2 environment")
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if any("t4" not in name.lower() for name in names):
        raise RuntimeError(f"M09 algorithm selection requires two Tesla T4 GPUs, got {names}")
    atomic_json(root / "runtime_environment.json", runtime_environment())
    atomic_json(root / "algorithm_selection_manifest.json", manifest)

    algorithms = manifest["algorithms"]
    ippo = load_v2_policy(validated["ippo_checkpoint"], device=device, expected_algorithm="ippo")
    vrpo = load_v2_policy(validated["vrpo_checkpoint"], device=device, expected_algorithm="vrpo")
    for label, artifact in (("ippo", ippo), ("vrpo", vrpo)):
        expected = algorithms[label]
        if artifact.sha256 != expected["sha256"] or artifact.iteration != int(expected["iteration"]):
            raise ValueError(f"{label} checkpoint runtime identity mismatch")
        if artifact.metadata.get("source_parent_commit") != expected["source_commit"]:
            raise ValueError(f"{label} source commit mismatch")
        if not artifact.training_data_provenance or not artifact.training_data_provenance.get("complete"):
            raise ValueError(f"{label} training provenance is incomplete")
    if ippo.protocol_version != vrpo.protocol_version or ippo.backend_identity_hash != vrpo.backend_identity_hash:
        raise ValueError("IPPO/VRPO runtime protocol/backend identities differ")

    selection = manifest["selection_dataset"]
    deals_manifest = build_bounded_selection_manifest(
        base_deals=int(selection["base_deals"]), seed_base=int(selection["seed_base"]),
        level_ranks=tuple(int(value) for value in selection["level_ranks"]),
        leader_seats=tuple(int(value) for value in selection["leader_seats"]),
        previous_results=tuple(selection["previous_results"]),
    )
    actual_dataset_hash = dataset_identity(deals_manifest, "selection")
    if actual_dataset_hash != selection["dataset_hash"]:
        raise ValueError("algorithm-selection deal dataset identity mismatch")

    games_path = root / "head_to_head_games.jsonl"
    games_path.unlink(missing_ok=True)
    records: list[dict[str, object]] = []
    deals = deals_for(deals_manifest, "selection")
    started = time.perf_counter()
    for deal_index, deal in enumerate(deals):
        for rotation in (0, 1, 2, 3):
            row = play_game(
                deal=deal, rotation=rotation, opponent_id="ippo", opponent_kind="v2",
                candidate=vrpo, opponent_v2=ippo, temperature=0.0, device=device,
            )
            append_jsonl(games_path, row)
            records.append(row)
        print(json.dumps({
            "event": "algorithm_selection_progress", "completed_clusters": deal_index + 1,
            "total_clusters": len(deals), "completed_games": len(records),
            "elapsed_seconds": time.perf_counter() - started,
        }, ensure_ascii=False), flush=True)
    expected_keys = {
        ("ippo", str(deal["deal_id"]), rotation)
        for deal in deals for rotation in (0, 1, 2, 3)
    }
    actual_keys = {(str(row["opponent_id"]), str(row["deal_id"]), int(row["seat_rotation"])) for row in records}
    if actual_keys != expected_keys or len(records) != len(expected_keys):
        raise RuntimeError("direct head-to-head matrix coverage is incomplete")
    if any(int(row.get("illegal_count", 0)) or int(row.get("truncated_count", 0)) for row in records):
        raise RuntimeError("direct head-to-head contains illegal or truncated games")

    direct = _cluster_ci(records)
    rule = _paired_rule_difference(
        ippo_path=validated["ippo_anchor"], vrpo_path=validated["vrpo_anchor"],
        ippo_sha256=ippo.sha256, vrpo_sha256=vrpo.sha256,
    )
    decision = _decision(direct, rule)
    summary = {
        "schema": ALGORITHM_SELECTION_SUMMARY_SCHEMA,
        "status": "complete",
        "comparison_id": manifest["comparison_id"],
        "candidate": vrpo.identity,
        "opponent": ippo.identity,
        "selection_dataset_hash": actual_dataset_hash,
        "games": len(records),
        "clusters": len(deals),
        "candidate_wins": sum(bool(row["candidate_win"]) for row in records),
        "candidate_win_rate": sum(bool(row["candidate_win"]) for row in records) / len(records),
        "illegal_count": 0,
        "truncated_count": 0,
        "direct_clustered_ci95": direct,
        "paired_rule_anchor": rule,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "formal_champion_promotion": False,
        "final_holdout": False,
        "strength_claim": False,
        "completed_at": _timestamp(),
    }
    atomic_json(root / "algorithm_selection_decision.json", decision)
    atomic_json(root / "algorithm_selection_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    summary = run_algorithm_selection(args.manifest, args.output_root, device=args.device)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
