# SPDX-License-Identifier: MIT
"""Two-GPU post-training development comparison. No training or final-holdout access."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Mapping, Sequence
from .common import append_jsonl, atomic_json, canonical_hash, sha256_file

ARMS = ("control_vs_rule", "candidate_vs_rule", "candidate_vs_control")
SCHEMA = "guandan-marvel-pilot-development-evaluation-v1"

def tasks_for(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    from guandan_marvel.evaluation.manifests import deals_for, validate_deals_manifest
    validate_deals_manifest(manifest)
    tasks = []
    for deal in deals_for(manifest, "selection"):
        for rotation in range(4):
            for arm in ARMS:
                tasks.append({"task_id": f"{arm}/{deal['deal_id']}/{rotation}",
                              "arm": arm, "deal": dict(deal), "rotation": rotation})
    return tasks

def check_records(records: Sequence[Mapping[str, object]], tasks, *, control_sha: str, candidate_sha: str):
    expected = {t["task_id"]: t for t in tasks}
    seen = set()
    hashes = {}
    for row in records:
        key = row.get("task_id")
        if key in seen or key not in expected:
            raise ValueError(f"duplicate or unplanned evaluation task: {key}")
        seen.add(key)
        task = expected[key]
        if row.get("status") != "complete" or row.get("illegal_count") != 0 or row.get("truncated_count") != 0:
            raise ValueError(f"incomplete/illegal/truncated evaluation: {key}")
        deal = task["deal"]
        for name, wanted in (("deal_id", deal["deal_id"]), ("deal_seed", deal["seed"]),
                             ("level_rank", deal["level_rank"]), ("leader_seat", deal["leader_seat"]),
                             ("seat_rotation", task["rotation"]), ("arm", task["arm"])):
            if row.get(name) != wanted:
                raise ValueError(f"evaluation task binding mismatch: {name} for {key}")
        sha = control_sha if task["arm"] == "control_vs_rule" else candidate_sha
        if row.get("candidate_checkpoint_sha256") != sha:
            raise ValueError("evaluation candidate identity changed")
        opponent_sha = control_sha if task["arm"] == "candidate_vs_control" else None
        if row.get("opponent_checkpoint_sha256") != opponent_sha:
            raise ValueError("evaluation opponent identity changed")
        physical = row.get("base_deal_sha256")
        if not isinstance(physical, str) or len(physical) != 64:
            raise ValueError("missing physical deal identity")
        if hashes.setdefault(deal["deal_id"], physical) != physical:
            raise ValueError("same deal differs across arm/rotation")
        if row.get("temperature") != 0.0:
            raise ValueError("development comparison requires fixed greedy temperature=0")
        if not isinstance(row.get("candidate_win"), bool):
            raise ValueError("candidate_win must be a real boolean")
        reward = row.get("candidate_team_reward")
        import math
        if not isinstance(reward,(int,float)) or not math.isfinite(reward) or not 0 < abs(reward) <= 3:
            raise ValueError("invalid raw terminal reward")
        if row["candidate_win"] != (reward > 0):
            raise ValueError("win/reward sign mismatch")
    if seen != set(expected):
        raise ValueError(f"missing evaluation tasks: {len(set(expected)-seen)}")


def summarize(records, tasks, *, control_sha, candidate_sha):
    from guandan_marvel.evaluation.stats import clustered_mean_ci95
    check_records(records, tasks, control_sha=control_sha, candidate_sha=candidate_sha)
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["arm"], row["deal_id"])].append(float(row["candidate_win"]))
    means = {key: sum(v)/4.0 for key, v in grouped.items()}
    by_arm = {}
    for arm in ARMS:
        subset = [r for r in records if r["arm"] == arm]
        clusters = [v for (a, _), v in means.items() if a == arm]
        by_arm[arm] = {"games": len(subset), "independent_deals": len(clusters),
                       "win_rate": sum(float(r["candidate_win"]) for r in subset)/len(subset),
                       "clustered_ci95": clustered_mean_ci95(clusters),
                       "raw_team_reward_mean": sum(float(r["candidate_team_reward"]) for r in subset)/len(subset)}
    deals = sorted({r["deal_id"] for r in records})
    differences = [means[("candidate_vs_rule", d)] - means[("control_vs_rule", d)] for d in deals]
    transformed = clustered_mean_ci95([(d+1.0)/2.0 for d in differences])
    paired = {"metric": "paired rule win-rate difference, per independent deal",
              "mean": sum(differences)/len(differences), "independent_deals": len(deals),
              "low": None if transformed["low"] is None else 2*transformed["low"]-1,
              "high": None if transformed["high"] is None else 2*transformed["high"]-1,
              "method": "bounded empirical-Bernstein on (difference+1)/2; inverse transformed"}
    ctrl_ci = by_arm["candidate_vs_control"]["clustered_ci95"]
    supported = paired["low"] is not None and paired["low"] > 0 and ctrl_ci["low"] is not None and ctrl_ci["low"] > 0.5
    return {"schema": SCHEMA, "status": "complete", "arms": by_arm, "paired_rule_improvement": paired,
            "verdict": "EARLY_IMPROVEMENT_SUPPORTED" if supported else "INCONCLUSIVE",
            "control_sha256": control_sha, "candidate_sha256": candidate_sha,
            "formal_promotion": False, "final_holdout_used": False, "fabledan": "NOT_RUN",
            "strength_claim": "development evidence only; does not establish superiority to FableDan"}


def run_shard(*, manifest_path: Path, control_path: Path, candidate_path: Path, output: Path, shard: int, shards: int, maximum_iteration: int = 50, parent_provenance: Path | None = None):
    import torch
    from guandan_marvel.evaluation.checkpoints import load_v2_policy
    from guandan_marvel.evaluation.host import play_game
    if shards != 2 or shard not in (0, 1) or torch.cuda.device_count() < 2:
        raise RuntimeError("pilot evaluation requires two explicit CUDA workers, no CPU/single-GPU fallback")
    torch.set_num_threads(1)
    torch.cuda.set_device(shard)
    device = f"cuda:{shard}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = tasks_for(manifest)
    control = load_v2_policy(control_path, device=device, expected_algorithm="ippo")
    candidate = load_v2_policy(candidate_path, device=device, expected_algorithm="ippo")
    if not control.metadata.get("source_hash") or control.metadata.get("source_hash") != candidate.metadata.get("source_hash"):
        raise ValueError("control/candidate source identities differ")
    if control.architecture != candidate.architecture:
        raise ValueError("control/candidate architectures differ")
    if maximum_iteration not in (50, 250) or candidate.iteration < 1 or candidate.iteration > maximum_iteration or control.iteration != 0:
        raise ValueError("unexpected pilot checkpoint counters")
    provenance = candidate.training_data_provenance or {}
    if not provenance.get("complete"):
        raise ValueError("candidate requires complete, aggregated training provenance")
    training_hashes = set(provenance.get("deal_hashes", []))
    initial_seeds = set(provenance.get("initial_env_seeds", []))
    if parent_provenance is not None:
        parent = json.loads(parent_provenance.read_text(encoding="utf-8"))
        training_hashes.update(parent["training_deal_hashes"])
        initial_seeds.update(parent["initial_env_seeds"])
    mine = tasks[shard::shards]
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"games.gpu{shard}.jsonl"
    if path.exists():
        raise FileExistsError("refusing to append new identities to an old evaluation shard")
    append_jsonl(output / f"events.gpu{shard}.jsonl", {"event": "start", "device": device,
                 "gpu": torch.cuda.get_device_name(shard), "planned_games": len(mine),
                 "control_sha256": control.sha256, "candidate_sha256": candidate.sha256,
                 "manifest_hash": manifest["hash"]})
    started = time.monotonic()
    try:
        for index, task in enumerate(mine, 1):
            arm = task["arm"]
            if task["deal"]["seed"] in initial_seeds:
                raise ValueError("development seed overlaps training initialization")
            artifact = control if arm == "control_vs_rule" else candidate
            row = play_game(deal=task["deal"], rotation=task["rotation"], opponent_id=arm,
                            opponent_kind="v2" if arm == "candidate_vs_control" else "rule",
                            candidate=artifact, opponent_v2=control if arm == "candidate_vs_control" else None,
                            temperature=0.0, device=device, max_steps=None)
            if row.get("base_deal_sha256") in training_hashes:
                raise ValueError("development physical deal overlaps actual training data")
            row.update({"task_id": task["task_id"], "arm": arm, "gpu_index": shard,
                        "evaluation_manifest_hash": manifest["hash"]})
            append_jsonl(path, row)
            if row.get("status") != "complete" or row.get("illegal_count") or row.get("truncated_count"):
                raise RuntimeError(f"invalid completed game: {task['task_id']}")
            if index % 8 == 0:
                print(f"[PILOT EVAL gpu{shard}] {index}/{len(mine)} games elapsed={time.monotonic()-started:.1f}s", flush=True)
        atomic_json(output / f"worker{shard}.json", {"status": "complete", "gpu_index": shard,
                    "games": len(mine), "elapsed_seconds": time.monotonic()-started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(shard)})
    except BaseException as error:
        atomic_json(output / f"worker{shard}.json", {"status": "failed", "error": repr(error), "gpu_index": shard})
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--maximum-iteration", type=int, choices=(50, 250), default=50)
    p.add_argument("--parent-provenance", type=Path)
    args = p.parse_args(argv)
    run_shard(manifest_path=args.manifest, control_path=args.control, candidate_path=args.candidate,
              output=args.output, shard=args.shard, shards=2, maximum_iteration=args.maximum_iteration, parent_provenance=args.parent_provenance)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
