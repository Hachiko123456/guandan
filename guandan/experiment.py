"""Explicit two-device experiment runner for fresh training and safe resume.

This module is deliberately separate from the legacy acceptance workflow. It runs
IPPO and VRPO in independent spawned processes, binds each worker to one explicit
CUDA device, and never silently falls back from a requested dual-GPU run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import uuid
from typing import Any

from .execution_profiles import load_execution_profile


@dataclass(frozen=True)
class WorkerSpec:
    algorithm: str
    device: str
    seed: int
    output_root: str
    profile: str
    base_updates: int
    resume_updates: int
    resume_checkpoint: str | None = None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _device_index(device: str) -> int | None:
    if device == "cuda":
        return None
    if device.startswith("cuda:"):
        value = device.split(":", 1)[1]
        if value.isdigit():
            return int(value)
    return None


def _child_device(device: str) -> str:
    index = _device_index(device)
    if index is None:
        return device
    # A spawned worker sees one physical GPU as logical cuda:0. This must happen
    # before importing torch in the child process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    return "cuda"


def _worker(spec: WorkerSpec, result_queue) -> None:
    started = time.monotonic()
    worker_dir = Path(spec.output_root) / "workers" / spec.algorithm
    progress_path = worker_dir / "progress.json"

    def progress(event: dict[str, Any]) -> None:
        _atomic_json(progress_path, {
            "algorithm": spec.algorithm, "device": spec.device,
            "updated_at": datetime.now(timezone.utc).isoformat(), **event,
        })

    try:
        device = _child_device(spec.device)
        from .training.trainer import train

        progress({"status": "base_started" if spec.resume_checkpoint is None else "resume_started"})
        if spec.resume_checkpoint is None:
            base = train(
                algorithm=spec.algorithm, profile=spec.profile, device=device,
                updates=spec.base_updates, checkpoint_dir=worker_dir,
                seed=spec.seed, progress_callback=progress,
            )
            if base.status != "complete":
                raise RuntimeError(f"{spec.algorithm} base training incomplete: {base.reason}")
            resume = train(
                algorithm=spec.algorithm, profile=spec.profile, device=device,
                updates=spec.resume_updates, checkpoint_dir=worker_dir,
                seed=spec.seed, resume=base.checkpoint, progress_callback=progress,
            )
        else:
            resume = train(
                algorithm=spec.algorithm, profile=spec.profile, device=device,
                updates=spec.resume_updates, checkpoint_dir=worker_dir,
                seed=spec.seed, resume=spec.resume_checkpoint, progress_callback=progress,
            )
        if resume.status != "complete":
            raise RuntimeError(f"{spec.algorithm} resume training incomplete: {resume.reason}")
        result = {
            "status": "complete", "algorithm": spec.algorithm, "device": spec.device,
            "base_updates": spec.base_updates, "resume_updates": spec.resume_updates,
            "checkpoint": resume.checkpoint, "recovery_manifest": resume.recovery_manifest,
            "lifetime_updates": resume.lifetime_updates,
            "lifetime_token_steps": resume.lifetime_token_steps,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(worker_dir / "result.json", result)
        result_queue.put(result)
    except BaseException as exc:
        result = {
            "status": "failed", "algorithm": spec.algorithm, "device": spec.device,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(worker_dir / "result.json", result)
        result_queue.put(result)


def build_plan(*, profile: str = "remote_full", output_root: str | Path = "runs/dual-gpu",
               device_ippo: str = "cuda:0", device_vrpo: str = "cuda:1",
               base_updates: int | None = None, resume_updates: int | None = None) -> dict[str, Any]:
    resolved = load_execution_profile(Path(__file__).resolve().parents[1], profile)
    cfg = resolved["config"]["training"]
    return {
        "format": "guandan-dual-gpu-plan-v1", "profile": resolved,
        "output_root": str(Path(output_root).resolve()),
        "workers": {
            "ippo": {"algorithm": "ippo", "device": device_ippo, "seed": 10_000,
                     "base_updates": cfg["updates_per_algorithm"] if base_updates is None else base_updates,
                     "resume_updates": cfg["resume_updates"] if resume_updates is None else resume_updates},
            "vrpo": {"algorithm": "vrpo", "device": device_vrpo, "seed": 60_000,
                     "base_updates": cfg["updates_per_algorithm"] if base_updates is None else base_updates,
                     "resume_updates": cfg["resume_updates"] if resume_updates is None else resume_updates},
        },
        "dual_gpu_required": device_ippo.startswith("cuda") and device_vrpo.startswith("cuda"),
        "accepted": False, "kaggle_verified": False,
    }


def _validate_cuda_pair(plan: dict[str, Any]) -> None:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("dual-GPU execution requires at least two visible CUDA devices; no silent downgrade")
    indices = [_device_index(plan["workers"][name]["device"]) for name in ("ippo", "vrpo")]
    if any(index is None for index in indices) or indices[0] == indices[1]:
        raise ValueError("IPPO and VRPO must be assigned distinct explicit CUDA devices")
    if any(index >= torch.cuda.device_count() for index in indices):
        raise RuntimeError("requested CUDA device is not visible")


def run(*, execute: bool = False, profile: str = "remote_full", output_root: str | Path = "runs/dual-gpu",
        device_ippo: str = "cuda:0", device_vrpo: str = "cuda:1", base_updates: int | None = None,
        resume_updates: int | None = None, resume_ippo: str | None = None,
        resume_vrpo: str | None = None) -> dict[str, Any]:
    plan = build_plan(profile=profile, output_root=output_root, device_ippo=device_ippo,
                      device_vrpo=device_vrpo, base_updates=base_updates,
                      resume_updates=resume_updates)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "plan.json", plan)
    if not execute:
        return {"status": "plan_only", **plan}
    if plan["dual_gpu_required"]:
        _validate_cuda_pair(plan)
    start = time.monotonic()
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    specs = [
        WorkerSpec("ippo", device_ippo, 10_000, str(root), profile,
                   plan["workers"]["ippo"]["base_updates"], plan["workers"]["ippo"]["resume_updates"], resume_ippo),
        WorkerSpec("vrpo", device_vrpo, 60_000, str(root), profile,
                   plan["workers"]["vrpo"]["base_updates"], plan["workers"]["vrpo"]["resume_updates"], resume_vrpo),
    ]
    processes = [context.Process(target=_worker, args=(spec, result_queue), name=f"guandan-{spec.algorithm}") for spec in specs]
    for process in processes:
        process.start()
    results = []
    failed = False
    try:
        while len(results) < len(processes):
            item = result_queue.get()
            results.append(item)
            if item.get("status") != "complete":
                failed = True
                # Do not burn the remaining session on a sibling after one
                # algorithm has already failed. Its result is marked incomplete
                # below rather than being mistaken for a successful run.
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                break
    finally:
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    if failed and len(results) < len(processes):
        for spec in specs:
            if not any(item.get("algorithm") == spec.algorithm for item in results):
                results.append({"status": "incomplete", "algorithm": spec.algorithm,
                                "device": spec.device, "error": "sibling worker failed"})
    status = "complete" if len(results) == 2 and all(item.get("status") == "complete" for item in results) else "incomplete"
    output = {"format": "guandan-dual-gpu-run-v1", "status": status, "profile": profile,
              "workers": {item["algorithm"]: item for item in results},
              "elapsed_seconds": time.monotonic() - start, "accepted": False, "kaggle_verified": False}
    _atomic_json(root / "result.json", output)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Explicit two-GPU GuanDan training runner")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", default="remote_full", choices=("local_fast", "remote_full"))
    parser.add_argument("--output-root", default="runs/dual-gpu")
    parser.add_argument("--device-ippo", default="cuda:0")
    parser.add_argument("--device-vrpo", default="cuda:1")
    parser.add_argument("--base-updates", type=int)
    parser.add_argument("--resume-updates", type=int)
    parser.add_argument("--resume-ippo")
    parser.add_argument("--resume-vrpo")
    args = parser.parse_args(argv)
    result = run(**vars(args))
    print(json.dumps({key: result[key] for key in ("status", "output_root", "elapsed_seconds", "accepted", "kaggle_verified") if key in result}, ensure_ascii=False))
    return 0 if result["status"] in {"plan_only", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
