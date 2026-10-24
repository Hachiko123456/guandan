"""Shared tiny evaluation-checkpoint fixture for phase-6 pool repair tests."""
from __future__ import annotations

from pathlib import Path

from guandan_marvel.config import (
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE, ModelConfig, PPOConfig, TrainConfig,
)
from guandan_marvel.evaluation.checkpoints import freeze_merged_policy
from guandan_marvel.evaluation.common import file_sha256
from guandan_marvel.training.local import IPPOTrainer
from guandan_marvel.training.smoke import CppSmokeGame


def tiny_config(game, *, seed: int = 707, run_name: str = "phase6-repair-fixture") -> TrainConfig:
    cfg = TrainConfig(
        game=game.spec, device="cpu", algo_name="ippo", seed=seed,
        run_name=run_name,
        rollout_batch_size=1, collect_batch_size=1, mini_batches=1,
        max_sampler_bytes_per_rank=512 * 2**20, amp_enabled=False,
        checkpoint_replay_buffer=False,
        model=ModelConfig(
            architecture="v1.4-macro-fuse", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
        ppo=PPOConfig(
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
        ),
    )
    cfg.ppo.agent_epoch = 1
    cfg.optimizer.use_muon = False
    return cfg


def make_tiny_eval_checkpoint(root: Path):
    game = CppSmokeGame(512)
    cfg = tiny_config(game)
    trainer = IPPOTrainer(cfg, game=game, run_dir=Path(root) / "seed-trainer")
    trainer.train_iteration()
    checkpoint = freeze_merged_policy(trainer, Path(root) / "opponent-eval.pt")
    entry = {
        "id": "opponent-a",
        "kind": "v2_evaluation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "architecture": str(cfg.model.architecture),
        "protocol_version": str(cfg.game.protocol_version),
        "backend": trainer.backend_name,
        "resolved_config_hash": trainer.resolved_config_hash,
        "source_hash": trainer.source_hash,
        "policy_version": 1,
        "weight": 1.0,
        "active": True,
        "tags": ["recent"],
    }
    return {"game": game, "cfg": cfg, "trainer": trainer, "checkpoint": checkpoint, "entry": entry}
