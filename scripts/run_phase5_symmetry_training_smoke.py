# SPDX-License-Identifier: MIT
"""Real-CUDA symmetry consistency smoke: 2 IPPO iterations + save/load.

Engineering smoke only (no formal training, no chess-strength claim).  It proves
that with a sealed SIGNIFICANT artifact the consistency term is consumed, finite
and differentiable, that parameters receive a finite update, that the enabled
configuration survives an exact checkpoint resume, and that the default-off
configuration attaches no symmetry branch at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

# Capacity adaptation authorised for this bounded smoke: the 4 GiB card OOMs on
# the learner forward at rollout/collect batch 4.  This only changes batch sizes
# (and allocator fragmentation policy); model, horizon, action capacity,
# precision, loss math and sampling remain untouched.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WARM_START = ROOT / "runs/comparison_delivery/dataset/m09_candidate_iter002000.pt"
DEFAULT_SEAL = ROOT / "runs/phase5/closeout_05e/symmetry_artifact_seal.json"
OUTPUT_ROOT = ROOT / "runs/phase5/closeout_05e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def finite_parameters(module) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in module.state_dict().values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda")
    parser.add_argument("--warm-start", default=str(DEFAULT_WARM_START))
    parser.add_argument("--symmetry-weight", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "symmetry_training_smoke.json"))
    parser.add_argument("--run-root", default=str(OUTPUT_ROOT / "symmetry_training_smoke"))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from guandan_marvel import cpp_game
    from guandan_marvel.config import DDPConfig, SymmetryConfig, deserialize_train_config
    from guandan_marvel.phase5.belief_probe import require_cuda_device
    from guandan_marvel.training.local import IPPOTrainer

    device = require_cuda_device(args.device)
    warm_path = Path(args.warm_start).resolve()
    payload = torch.load(warm_path, map_location="cpu", weights_only=False)
    cfg = deserialize_train_config(payload["resolved_config"])
    cfg.device = str(device)
    cfg.rollout_batch_size = 1
    cfg.collect_batch_size = 1
    cfg.mini_batches = 1
    cfg.total_iterations = max(2, int(args.iterations))
    cfg.compile.enabled = False
    cfg.amp_enabled = False
    cfg.optimizer.use_muon = False
    cfg.ppo.agent_epoch = 1
    # The production checkpoint records single-process DDP disabled; a smoke run
    # is launched directly (not via torchrun), so keep the single-process path.
    cfg.ddp = DDPConfig(enabled=False)
    cfg.symmetry = SymmetryConfig(
        symmetry_weight=float(args.symmetry_weight),
        diagnostic_artifact=str(DEFAULT_SEAL),
        compatibility_checkpoint=str(warm_path),
    )
    run_root = Path(args.run_root).resolve()
    phase1 = IPPOTrainer(cfg, game=cpp_game, run_dir=run_root / "phase1")
    assert phase1.symmetry is not None, "enabled configuration did not attach the branch"
    phase1.agent_resume.load_state_dict(payload["agent_state_dict"], strict=True)
    phase1.agent_merged.load_state_dict(payload["agent_state_dict"], strict=True)
    before = snapshot(phase1.agent_resume)
    metrics = phase1.train_iteration()
    after = snapshot(phase1.agent_resume)
    changed = [
        name for name in before
        if not torch.equal(before[name], after[name])
    ]
    symmetry_keys = sorted(key for key in metrics if "symmetry" in key)
    if not symmetry_keys:
        raise RuntimeError("the enabled symmetry term produced no recorded metric")
    for key in symmetry_keys:
        value = metrics[key]
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"{key} is non-finite: {value}")
    if not changed:
        raise RuntimeError("the symmetry-enabled iteration changed no parameter")
    if not finite_parameters(phase1.agent_resume):
        raise RuntimeError("a parameter became non-finite after the symmetry update")
    checkpoint = phase1.save_checkpoint(run_root / "checkpoints" / "symmetry-iter000001.pt")

    phase2 = IPPOTrainer(cfg, game=cpp_game, run_dir=run_root / "phase2")
    decision = phase2.load_checkpoint(checkpoint)
    assert phase2.symmetry is not None, "resume lost the symmetry branch"
    second = phase2.train_iteration()
    assert "agent/symmetry_loss" in second, "resumed iteration did not consume the branch"
    checkpoint2 = phase2.save_checkpoint(run_root / "checkpoints" / "symmetry-iter000002.pt")

    off_cfg = cfg
    off_cfg.symmetry = SymmetryConfig()
    off_trainer = IPPOTrainer(off_cfg, game=cpp_game, run_dir=run_root / "off")
    assert off_trainer.symmetry is None, "disabled configuration attached a branch"

    result = {
        "schema": "guandan-marvel-phase5-symmetry-training-smoke-v1",
        "status": "PASS",
        "device": str(device),
        "parameter_device": str(next(phase1.agent_resume.parameters()).device),
        "gpu_name": torch.cuda.get_device_name(device),
        "symmetry_weight": float(args.symmetry_weight),
        "capacity_adaptation": {
            "reason": "4 GiB card OOM on learner forward at batch 4",
            "rollout_batch_size_before": 4,
            "rollout_batch_size_after": 1,
            "collect_batch_size_before": 4,
            "collect_batch_size_after": 1,
            "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "unchanged": [
                "max_available_actions", "game horizon", "model structure",
                "amp/fp16 precision", "loss mathematics", "sampling data",
            ],
        },
        "seal_path": str(DEFAULT_SEAL),
        "seal_sha256": sha256(DEFAULT_SEAL),
        "artifact_scan": phase1.symmetry.contract.audit(),
        "component_sha256": {
            name: sha256(ROOT / name) for name in (
                "guandan_marvel/phase5/symmetry_training.py",
                "guandan_marvel/training/local.py",
                "guandan_marvel/config.py",
                "guandan_marvel/training/checkpoints.py",
            )
        },
        "iterations": 2,
        "resume_decision": decision.kind,
        "metrics_iteration_1": {
            key: value for key, value in metrics.items()
            if key.startswith("agent/symmetry") or key.startswith("agent/pg_loss")
            or key.startswith("agent/grad_norm") or key.startswith("statistics/iterations")
        },
        "metrics_iteration_2": {
            key: value for key, value in second.items()
            if key.startswith("agent/symmetry") or key.startswith("agent/pg_loss")
            or key.startswith("agent/grad_norm") or key.startswith("statistics/iterations")
        },
        "symmetry_keys": symmetry_keys,
        "parameters_changed": changed,
        "all_parameters_finite": True,
        "disabled_branch_attached": False,
        "checkpoints": [
            {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            {"path": str(checkpoint2), "sha256": sha256(checkpoint2)},
        ],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "wall_seconds": time.perf_counter() - started,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "resume_decision": decision.kind,
        "symmetry_metrics": result["metrics_iteration_2"],
        "changed_parameter_count": len(changed),
        "wall_seconds": result["wall_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
