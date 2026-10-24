"""Durable-checkpoint recovery worker for the phase-2 Profile C smoke.

Each mode runs in its own OS process so a resume is genuinely a new process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from guandan_marvel.config import (  # noqa: E402
    LEGACY_POLICY_REGULARIZER_MODE, MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    ModelConfig, PPOConfig, TrainConfig,
)
from guandan_marvel.training.local import IPPOTrainer  # noqa: E402
from guandan_marvel.training.smoke import CppSmokeGame  # noqa: E402


def build_config(seed: int) -> tuple[TrainConfig, CppSmokeGame]:
    game = CppSmokeGame(max_game_horizon=512)
    cfg = TrainConfig(
        game=game.spec,
        device="cpu",
        algo_name="ippo",
        run_name="phase2-macro-recovery",
        logging_dir="runs/marvel_v14_phase2/smoke/recovery/rank-runtime",
        seed=seed,
        rollout_batch_size=1,
        collect_batch_size=1,
        replay_buffer_scale=1,
        mini_batches=1,
        total_iterations=8,
        max_sampler_bytes_per_rank=256 * 2**20,
        amp_enabled=False,
        checkpoint_replay_buffer=True,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
        ppo=PPOConfig(
            agent_epoch=2,
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=LEGACY_POLICY_REGULARIZER_MODE,
        ),
    )
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg, game


def state_digest(state_dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        tensor = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def optimizer_digest(optimizer) -> str:
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    for param_key in sorted(state["state"], key=str):
        entry = state["state"][param_key]
        for name in sorted(entry):
            value = entry[name]
            digest.update(f"{param_key}:{name}".encode("utf-8"))
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tuple(tensor.shape)).encode("utf-8"))
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    for group in state["param_groups"]:
        digest.update(json.dumps(group, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def rng_digest() -> str:
    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().numpy().tobytes())
    digest.update(repr(np.random.get_state()).encode("utf-8"))
    digest.update(repr(random.getstate()).encode("utf-8"))
    return digest.hexdigest()


def snapshot(trainer) -> dict[str, object]:
    return {
        "iteration": int(trainer.iteration),
        "durable_updates": int(trainer.durable_updates),
        "learner_updates": int(trainer.learner_updates),
        "durable_learner_updates": int(trainer.durable_learner_updates),
        "completed_games_total": int(trainer.completed_games_total),
        "env_steps": int(trainer.sampler.total_env_steps),
        "game_steps": int(trainer.sampler.total_game_steps),
        "agent_resume_digest": state_digest(trainer.agent_resume.state_dict()),
        "agent_merged_digest": state_digest(trainer.agent_merged.state_dict()),
        "optimizer_digest": optimizer_digest(trainer.agent_optimizer),
        "rng_digest": rng_digest(),
        "training_semantics_profile": trainer.training_semantics_profile,
        "policy_regularizer_mode": trainer.policy_regularizer_mode,
        "last_metrics": {
            key: trainer.last_metrics.get(key)
            for key in (
                "macro/pg_loss", "macro/value_loss_last", "agent/grad_norm",
                "macro/clip_frac", "macro/legacy_token_kl", "macro/normalized_entropy",
            )
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "resume"), required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cfg, game = build_config(args.seed)
    trainer = IPPOTrainer(cfg, game=game, run_dir=Path(args.run_dir))
    checkpoint_path = Path(args.checkpoint)
    payload: dict[str, object] = {"mode": args.mode}
    if args.mode == "resume":
        decision = trainer.load_checkpoint(checkpoint_path)
        payload["resume_kind"] = decision.kind
        payload["state_on_load"] = snapshot(trainer)
    for _ in range(int(args.iters)):
        trainer.train_iteration()
    if args.mode == "train":
        trainer.save_checkpoint(checkpoint_path)
    payload["state"] = snapshot(trainer)
    payload["checkpoint_exists"] = checkpoint_path.is_file()
    trainer.close()
    Path(args.out).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
