"""Ability evaluation suite for trained GuanDan policies.

The suite is inference-only: it never trains, mutates checkpoints, or silently
finds historical artifacts. It runs one algorithm worker per explicit GPU and
reports game-level plus base-deal-clustered statistics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import shutil
import time
import traceback
import uuid
import zipfile
from typing import Any

from .evaluation import evaluate_profile
from .evaluation_stats import wilson_interval
from .execution_profiles import load_execution_profile

_UPDATE_RE = re.compile(r"_update_(\d+)_")


@dataclass(frozen=True)
class EvalWorkerSpec:
    algorithm: str
    candidate: str
    old_snapshot: str
    cross_snapshot: str
    device: str
    profile: str
    batch_size: int
    output_root: str
    seed: int


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _device_index(device: str) -> int | None:
    if device.startswith("cuda:") and device.split(":", 1)[1].isdigit():
        return int(device.split(":", 1)[1])
    return None


def _child_device(device: str) -> str:
    index = _device_index(device)
    if index is None:
        return device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    return "cuda"


def _extract_bundle(bundle: str | Path, destination: Path) -> Path:
    source = Path(bundle).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        return source
    if source.suffix.lower() != ".zip":
        raise ValueError("progress bundle must be a ZIP or directory")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        archive.extractall(destination)
    return destination


def _checkpoint_for(root: Path, algorithm: str, update: int) -> Path:
    scope = root / "workers" / algorithm / "remote_full" / algorithm
    matches = []
    for path in scope.glob("*.pt"):
        match = _UPDATE_RE.search(path.name)
        if match and int(match.group(1)) == update:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {algorithm} update {update} checkpoint, found {len(matches)}")
    return matches[0].resolve()


def _cluster_summary(records: list[dict[str, Any]], *, opponent: str, seed: int) -> dict[str, Any]:
    games = [row for row in records if row["opponent"] == opponent]
    wins = [int(row["winner_team"] == row["candidate_team"]) for row in games]
    groups: dict[tuple[int, str], list[int]] = {}
    for row, win in zip(games, wins):
        groups.setdefault((row["deal_group"], row["base_deal_sha256"]), []).append(win)
    group_values = [sum(values) / len(values) for values in groups.values()]
    rng = random.Random(seed)
    bootstrap = []
    if group_values:
        for _ in range(2000):
            sample = [group_values[rng.randrange(len(group_values))] for _ in group_values]
            bootstrap.append(sum(sample) / len(sample))
        bootstrap.sort()
    lower = bootstrap[int(0.025 * (len(bootstrap) - 1))] if bootstrap else None
    upper = bootstrap[int(0.975 * (len(bootstrap) - 1))] if bootstrap else None
    interval = wilson_interval(sum(wins), len(wins)) if wins else None
    return {
        "opponent": opponent, "games": len(games), "independent_deal_groups": len(groups),
        "candidate_wins": sum(wins), "candidate_win_rate": sum(wins) / len(wins) if wins else None,
        "game_level_wilson95": interval, "deal_cluster_bootstrap95": (lower, upper),
        "mean_token_steps": sum(row["token_steps"] for row in games) / len(games) if games else None,
        "mean_committed_steps": sum(row["committed_steps"] for row in games) / len(games) if games else None,
    }


def _run_one(label: str, candidate: Path, snapshot: Path, spec: EvalWorkerSpec, device: str, out: Path) -> dict[str, Any]:
    evidence = out / label / "summary.json"
    started = time.monotonic()
    result = evaluate_profile(
        profile=spec.profile, seed=spec.seed, candidate_checkpoint=candidate,
        snapshot_checkpoint=snapshot, evidence_path=evidence, device=device,
        batch_size=spec.batch_size,
    )
    records = [asdict(row) for row in result.records]
    summary = {
        "status": result.status, "label": label, "algorithm": spec.algorithm,
        "candidate": str(candidate), "snapshot": str(snapshot),
        "runtime_seconds": result.runtime_seconds, "wall_seconds": time.monotonic() - started,
        "completed_games_per_opponent": result.completed_games_per_opponent,
        "summary": result.summary,
        "clustered": {opponent: _cluster_summary(records, opponent=opponent, seed=spec.seed + index)
                      for index, opponent in enumerate(("random", "rule", "snapshot"))},
        "metadata": result.metadata,
    }
    _atomic_json(out / label / "ability_summary.json", summary)
    return summary


def _worker(spec: EvalWorkerSpec, result_queue) -> None:
    started = time.monotonic()
    out = Path(spec.output_root) / spec.algorithm
    try:
        device = _child_device(spec.device)
        candidate = Path(spec.candidate)
        old_snapshot = Path(spec.old_snapshot)
        cross_snapshot = Path(spec.cross_snapshot)
        reports = [
            _run_one(f"{spec.algorithm}_update102_vs_update100", candidate, old_snapshot, spec, device, out),
            _run_one(f"{spec.algorithm}_update102_vs_other_update102", candidate, cross_snapshot, spec, device, out),
        ]
        payload = {"status": "complete", "algorithm": spec.algorithm, "device": spec.device,
                   "reports": reports, "elapsed_seconds": time.monotonic() - started}
        _atomic_json(out / "result.json", payload)
        result_queue.put(payload)
    except BaseException as exc:
        payload = {"status": "failed", "algorithm": spec.algorithm, "device": spec.device,
                   "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started}
        _atomic_json(out / "result.json", payload)
        result_queue.put(payload)


def build_plan(*, profile: str = "remote_full", progress_bundle: str | Path = "",
               output_root: str | Path = "runs/evaluation-suite", batch_size: int = 16,
               device_ippo: str = "cuda:0", device_vrpo: str = "cuda:1") -> dict[str, Any]:
    resolved = load_execution_profile(Path(__file__).resolve().parents[1], profile)
    return {"format": "guandan-ability-evaluation-plan-v1", "profile": resolved,
            "progress_bundle": str(progress_bundle), "output_root": str(Path(output_root).resolve()),
            "batch_size": batch_size, "workers": {"ippo": device_ippo, "vrpo": device_vrpo},
            "matchups": ["self_old_snapshot", "cross_algorithm_update102"],
            "accepted": False, "kaggle_verified": False}


def _validate_pair(device_ippo: str, device_vrpo: str) -> None:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("evaluation suite requires two CUDA devices")
    indices = [_device_index(device_ippo), _device_index(device_vrpo)]
    if any(index is None for index in indices) or indices[0] == indices[1] or any(index >= torch.cuda.device_count() for index in indices):
        raise RuntimeError("evaluation workers require distinct visible cuda indices")


def run(*, execute: bool = False, profile: str = "remote_full", progress_bundle: str = "",
        output_root: str | Path = "runs/evaluation-suite", batch_size: int = 16,
        device_ippo: str = "cuda:0", device_vrpo: str = "cuda:1") -> dict[str, Any]:
    plan = build_plan(profile=profile, progress_bundle=progress_bundle, output_root=output_root,
                      batch_size=batch_size, device_ippo=device_ippo, device_vrpo=device_vrpo)
    root = Path(output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "plan.json", plan)
    if not execute:
        return {"status": "plan_only", **plan}
    if plan["profile"]["config"]["mode"] == "remote":
        _validate_pair(device_ippo, device_vrpo)
    if not progress_bundle:
        raise ValueError("evaluation requires an explicit progress bundle")
    bundle_root = _extract_bundle(progress_bundle, root / "progress")
    checkpoints = {
        "ippo100": _checkpoint_for(bundle_root, "ippo", 100), "ippo102": _checkpoint_for(bundle_root, "ippo", 102),
        "vrpo100": _checkpoint_for(bundle_root, "vrpo", 100), "vrpo102": _checkpoint_for(bundle_root, "vrpo", 102),
    }
    context = mp.get_context("spawn"); result_queue = context.Queue()
    specs = [
        EvalWorkerSpec("ippo", str(checkpoints["ippo102"]), str(checkpoints["ippo100"]), str(checkpoints["vrpo102"]), device_ippo, profile, batch_size, str(root), 2_000_000),
        EvalWorkerSpec("vrpo", str(checkpoints["vrpo102"]), str(checkpoints["vrpo100"]), str(checkpoints["ippo102"]), device_vrpo, profile, batch_size, str(root), 3_000_000),
    ]
    processes = [context.Process(target=_worker, args=(spec, result_queue), name=f"guandan-eval-{spec.algorithm}") for spec in specs]
    for process in processes: process.start()
    results = []
    try:
        for _ in processes: results.append(result_queue.get())
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive(): process.terminate(); process.join(timeout=5)
    status = "complete" if len(results) == 2 and all(row.get("status") == "complete" for row in results) else "incomplete"
    final = {"format": "guandan-ability-evaluation-v1", "status": status, "profile": profile,
             "checkpoints": {key: str(value) for key, value in checkpoints.items()},
             "workers": {row["algorithm"]: row for row in results}, "accepted": False, "kaggle_verified": False}
    _atomic_json(root / "result.json", final)
    return final


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GuanDan ability evaluation suite")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", default="remote_full", choices=("local_fast", "remote_full"))
    parser.add_argument("--progress-bundle", required=True)
    parser.add_argument("--output-root", default="runs/evaluation-suite")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device-ippo", default="cuda:0")
    parser.add_argument("--device-vrpo", default="cuda:1")
    args = parser.parse_args(argv)
    result = run(**vars(args)); print(json.dumps({key: result.get(key) for key in ("status", "accepted", "kaggle_verified")}, ensure_ascii=False))
    return 0 if result["status"] in {"plan_only", "complete"} else 2


if __name__ == "__main__": raise SystemExit(main())
