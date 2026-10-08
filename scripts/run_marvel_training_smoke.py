"""Run the bounded M03 IPPO/VRPO local semantic smoke profile."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.training.local import IPPOTrainer, VRPOTrainer
from guandan_marvel.training.smoke import PythonReferenceSmokeGame


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config(*, algorithm: str, game: PythonReferenceSmokeGame, root: Path, seed: int) -> TrainConfig:
    cfg = TrainConfig(
        game=game.spec,
        device="cpu",
        algo_name=algorithm,
        logging_dir=str(root / algorithm),
        run_name=f"m03-local-{algorithm}",
        seed=seed,
        rollout_batch_size=1,
        collect_batch_size=1,
        replay_buffer_scale=1,
        mini_batches=1,
        total_iterations=2,
        max_sampler_bytes_per_rank=64 * 2**20,
        amp_enabled=False,
        checkpoint_replay_buffer=True,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.ppo.agent_epoch = 2
    cfg.vrpo.aux_epoch = 2
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg


def run_algorithm(*, algorithm: str, game: PythonReferenceSmokeGame, root: Path, seed: int) -> dict[str, object]:
    cfg = build_config(algorithm=algorithm, game=game, root=root, seed=seed)
    trainer_cls = IPPOTrainer if algorithm == "ippo" else VRPOTrainer
    phase1 = trainer_cls(cfg, game=game, run_dir=root / algorithm / "phase1")
    first_metrics = phase1.train_iteration()
    checkpoint1 = phase1.save_checkpoint(root / algorithm / "checkpoints" / f"guandan-marvel-v2-{algorithm}-iter000001.pt")

    phase2 = trainer_cls(cfg, game=game, run_dir=root / algorithm / "phase2")
    decision = phase2.load_checkpoint(checkpoint1)
    second_metrics = phase2.train_iteration()
    checkpoint2 = phase2.save_checkpoint(root / algorithm / "checkpoints" / f"guandan-marvel-v2-{algorithm}-iter000002.pt")

    required_finite = [
        value for metrics in (first_metrics, second_metrics) for value in metrics.values()
        if isinstance(value, float)
    ]
    if not all(np.isfinite(required_finite)):
        raise FloatingPointError(f"{algorithm} smoke produced a non-finite metric")
    if phase2.iteration != 2 or phase2.durable_updates != 2 or phase2.completed_games_total != 2:
        raise AssertionError(f"{algorithm} did not complete two durable updates/games")
    return {
        "algorithm": algorithm,
        "iterations": phase2.iteration,
        "durable_updates": phase2.durable_updates,
        "learner_updates": phase2.learner_updates,
        "completed_games": phase2.completed_games_total,
        "total_env_steps": phase2.sampler.total_env_steps,
        "total_game_steps": phase2.sampler.total_game_steps,
        "trajectory_lengths": [
            int(first_metrics["statistics/trajectory_length_mean"]),
            int(second_metrics["statistics/trajectory_length_mean"]),
        ],
        "resume_decision": decision.kind,
        "merged_policy_identity": phase2.merged_policy_identity,
        "optimizer": second_metrics["runtime/optimizer"],
        "compile_enabled": second_metrics["runtime/compile_enabled"],
        "amp_enabled": second_metrics["runtime/amp_enabled"],
        "sampler_memory_bytes": int(phase2.memory_budget["sampler_bytes_per_rank"]),
        "replay_memory": phase2.replay_memory_budget,
        "first_metrics": first_metrics,
        "second_metrics": second_metrics,
        "checkpoints": [
            {"path": str(checkpoint1), "sha256": sha256(checkpoint1), "bytes": checkpoint1.stat().st_size},
            {"path": str(checkpoint2), "sha256": sha256(checkpoint2), "bytes": checkpoint2.stat().st_size},
        ],
        "metrics_jsonl": [str(phase1.metrics_path), str(phase2.metrics_path)],
        "source_hash": phase2.source_hash,
        "semantic_contract_hash": phase2.semantic_contract["hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    game = PythonReferenceSmokeGame(512)
    result = {
        "schema": "guandan-marvel-m03-local-smoke-v1",
        "date": "2026-10-08",
        "scope": "local semantic smoke only; no strength/performance/remote claim",
        "backend": "python_reference",
        "game_identifier": game.identifier,
        "default_protocol_horizon": 4096,
        "smoke_horizon": 512,
        "reduced_horizon_disclaimer": True,
        "profile": {
            "rollout_batch_size": 1,
            "collect_batch_size": 1,
            "replay_buffer_scale": 1,
            "mini_batches": 1,
            "agent_epoch": 2,
            "critic_aux_epoch": 2,
            "model": {
                "history_dim": 16, "feature_dim": 24, "decoder_layers": 1,
                "decoder_heads": 4, "state_layers": 1, "action_layers": 1,
                "fuse_layers": 1,
            },
            "optimizer": "AdamW",
            "muon": False,
            "compile": False,
            "amp": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "ippo": run_algorithm(algorithm="ippo", game=game, root=output, seed=3101),
        "vrpo": run_algorithm(algorithm="vrpo", game=game, root=output, seed=3201),
        "model_strength_claim": False,
        "remote_verified": False,
    }
    summary = output / "summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ippo_updates": result["ippo"]["durable_updates"],
        "ippo_games": result["ippo"]["completed_games"],
        "vrpo_updates": result["vrpo"]["durable_updates"],
        "vrpo_games": result["vrpo"]["completed_games"],
        "summary": str(summary),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
