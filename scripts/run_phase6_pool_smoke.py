# SPDX-License-Identifier: MIT
"""Bounded phase-6 GPU opponent-pool smoke (3 pre-registered configurations).

Run only with the project GPU Python and an explicit CUDA device:

    D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase6_pool_smoke.py --device cuda

The smoke is a local 4 GB adaptation: batch=1, one frozen opponent, 1-2
iterations per configuration, AMP FP16 with an explicit ``init_scale=1024``
(the production default 65536 overflowed FP16 gradients for this tiny model).
No parameter, loss formula or tolerance is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from guandan_marvel.config import (  # noqa: E402
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE, ModelConfig, OpponentPoolConfig,
    PPOConfig, TrainConfig,
)
from guandan_marvel.evaluation.checkpoints import freeze_merged_policy  # noqa: E402
from guandan_marvel.evaluation.common import file_sha256  # noqa: E402
from guandan_marvel.training.local import IPPOTrainer  # noqa: E402
from guandan_marvel.training.smoke import CppSmokeGame  # noqa: E402

PRE_REGISTERED_CONFIGURATIONS = (
    "A: pool off, 1 iteration, FP16 AMP (init_scale=1024)",
    "B: pool on with one frozen evaluation checkpoint, 1 iteration, FP16 AMP",
    "C: pool on, 2 iterations control vs load-after-iteration-1 resume",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cfg(game, *, pool=None, seed: int = 4242) -> TrainConfig:
    cfg = TrainConfig(
        game=game.spec, device="cuda", algo_name="ippo", seed=seed,
        run_name="phase6-pool-smoke",
        rollout_batch_size=1, collect_batch_size=1, mini_batches=1,
        max_sampler_bytes_per_rank=512 * 2**20, amp_enabled=True,
        amp_init_scale=1024.0, checkpoint_replay_buffer=False,
        model=ModelConfig(
            architecture="v1.4-macro-fuse", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
        ppo=PPOConfig(
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
        ),
        opponent_pool=OpponentPoolConfig() if pool is None else pool,
    )
    cfg.ppo.agent_epoch = 1
    cfg.optimizer.use_muon = False
    return cfg


def _param_snapshot(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _changed_keys(before, model) -> list[str]:
    after = model.state_dict()
    return [key for key in before if not torch.equal(before[key], after[key])]


def _pool_metrics(metrics) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if key.startswith("pool/")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument(
        "--out", default="runs/phase6/pool_smoke/pilot_pool.json",
    )
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("phase-6 pool smoke requires an explicit available CUDA device")
    game = CppSmokeGame(512)
    root = Path(args.out).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema": "guandan-marvel-phase6-pool-smoke-v1",
        "created_at": _timestamp(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "requested_device": args.device,
        "pre_registered_configurations": list(PRE_REGISTERED_CONFIGURATIONS),
        "all_local_adaptations": [
            "batch=1 for a 4 GB laptop GPU",
            "one frozen opponent resident (two tiny models)",
            "AMP FP16 with explicit init_scale=1024 (production default 65536)",
            "1-2 iterations per configuration; smoke only, not a formal gate",
        ],
        "budget": {
            "max_configurations": 3,
            "iterations_per_configuration": 2,
            "device": "cuda",
            "formal_gates_run": False,
        },
        "configurations": {},
    }

    started = time.perf_counter()
    # Configuration A: pool off.  This trainer also produces the opponent
    # evaluation checkpoint for configurations B/C.
    torch.cuda.reset_peak_memory_stats()
    a_started = time.perf_counter()
    cfg_a = _cfg(game)
    trainer_a = IPPOTrainer(cfg_a, game=game, run_dir=root / "config-a")
    assert trainer_a.device.type == "cuda"
    assert trainer_a.policy_router is None
    a_before = _param_snapshot(trainer_a.agent_resume)
    a_metrics = trainer_a.train_iteration()
    a_changed = _changed_keys(a_before, trainer_a.agent_resume)
    if not a_changed:
        raise RuntimeError("configuration A performed no optimizer update")
    report["configurations"]["A"] = {
        "pool_enabled": False,
        "device": str(trainer_a.device),
        "autocast_dtype": a_metrics["runtime/amp_autocast_dtype"],
        "scaler_enabled": a_metrics["runtime/amp_scaler_enabled"],
        "scaler_scale": a_metrics["runtime/amp_scaler_scale"],
        "amp_skipped_steps": a_metrics["runtime/amp_skipped_steps"],
        "learner_updates": a_metrics["statistics/learner_updates"],
        "changed_parameter_count": len(a_changed),
        "completed_games": trainer_a.completed_games_total,
        "wall_seconds": time.perf_counter() - a_started,
    }
    opponent_checkpoint = freeze_merged_policy(trainer_a, root / "opponent-eval.pt")
    opponent_sha = file_sha256(opponent_checkpoint)
    a_changed_count = len(a_changed)
    del trainer_a
    torch.cuda.empty_cache()

    entry = {
        "id": "smoke-opponent",
        "kind": "v2_evaluation",
        "checkpoint": str(opponent_checkpoint),
        "checkpoint_sha256": opponent_sha,
        "architecture": str(cfg_a.model.architecture),
        "protocol_version": str(cfg_a.game.protocol_version),
        "backend": "cpp",
        "resolved_config_hash": None,
        "source_hash": None,
        "policy_version": 0,
        "weight": 1.0,
        "active": True,
        "tags": ["recent"],
    }
    # The frozen checkpoint knows its own config/source identity; the pool
    # validates the artifact against the running runtime.
    from guandan_marvel.evaluation.checkpoints import load_v2_policy

    artifact = load_v2_policy(opponent_checkpoint, device="cpu", expected_algorithm="ippo")
    entry["resolved_config_hash"] = artifact.resolved_config_hash
    entry["source_hash"] = artifact.metadata.get("source_hash")
    pool = OpponentPoolConfig(enabled=True, entries=(entry,), sampling_seed=17)

    # Configuration B: one frozen opponent, one iteration.
    torch.cuda.reset_peak_memory_stats()
    b_started = time.perf_counter()
    cfg_b = _cfg(game, pool=pool)
    trainer_b = IPPOTrainer(cfg_b, game=game, run_dir=root / "config-b")
    assert trainer_b.device.type == "cuda"
    opponent_before = trainer_b.opponent_pool.parameter_hashes()
    b_metrics = trainer_b.train_iteration()
    opponent_after = trainer_b.policy_router.frozen_parameter_hashes()
    if opponent_before != opponent_after:
        raise RuntimeError("frozen opponent parameters changed during configuration B")
    b_peak = int(torch.cuda.max_memory_allocated())
    b_metrics_pool = _pool_metrics(b_metrics)
    if b_metrics_pool.get("pool/current_token_count", 0) <= 0:
        raise RuntimeError("configuration B produced no current-policy samples")
    if b_metrics_pool.get("pool/opponent_token_count", 0) <= 0:
        raise RuntimeError("configuration B produced no frozen-opponent samples")
    report["configurations"]["B"] = {
        "pool_enabled": True,
        "device": str(trainer_b.device),
        "manifest_hash": trainer_b.opponent_pool.manifest_hash,
        "opponent_checkpoint_sha256": opponent_sha,
        "frozen_parameters_unchanged": True,
        "autocast_dtype": b_metrics["runtime/amp_autocast_dtype"],
        "scaler_enabled": b_metrics["runtime/amp_scaler_enabled"],
        "scaler_scale": b_metrics["runtime/amp_scaler_scale"],
        "amp_skipped_steps": b_metrics["runtime/amp_skipped_steps"],
        "learner_updates": b_metrics["statistics/learner_updates"],
        "pool_metrics": b_metrics_pool,
        "peak_allocated_bytes": b_peak,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "completed_games": trainer_b.completed_games_total,
        "wall_seconds": time.perf_counter() - b_started,
    }
    del trainer_b
    torch.cuda.empty_cache()

    # Configuration C: pool-on control (2 iterations) vs resume after iteration 1.
    c_started = time.perf_counter()
    cfg_c = _cfg(game, pool=pool)
    control = IPPOTrainer(cfg_c, game=game, run_dir=root / "config-c-control")
    control.train_iteration()
    checkpoint = control.save_checkpoint(root / "config-c-iter1.pt")
    control.train_iteration()
    control_state = control.agent_resume.state_dict()

    resumed = IPPOTrainer(cfg_c, game=game, run_dir=root / "config-c-resume")
    decision = resumed.load_checkpoint(checkpoint)
    resumed.train_iteration()
    resumed_state = resumed.agent_resume.state_dict()
    exact = all(
        torch.equal(control_state[key], resumed_state[key]) for key in control_state
    )
    if decision.kind != "exact_resume" or not exact:
        raise RuntimeError(
            f"configuration C resume mismatch: kind={decision.kind}, equal={exact}"
        )
    report["configurations"]["C"] = {
        "pool_enabled": True,
        "resume_decision": decision.kind,
        "resumed_parameters_identical": exact,
        "manifest_hash": resumed.opponent_pool.manifest_hash,
        "completed_games_control": control.completed_games_total,
        "completed_games_resumed": resumed.completed_games_total,
        "checkpoint_sha256": file_sha256(checkpoint),
        "wall_seconds": time.perf_counter() - c_started,
    }
    report["total_wall_seconds"] = time.perf_counter() - started
    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    report["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    report["source_hash"] = control.source_hash
    report["resolved_config_hash"] = control.resolved_config_hash
    report["completed_work"] = "A/B/C smoke iterations"
    report["unexecuted_items"] = [
        "formal gate 1-5", "Kaggle", "dual T4/NCCL", "FableDan evaluation",
        "final holdout", "deployment search",
    ]
    report["config_a_changed_parameter_count"] = a_changed_count
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "device": report["gpu_name"],
        "configurations": list(report["configurations"]),
        "wall_seconds": report["total_wall_seconds"],
        "output": str(destination),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
