"""Phase-6 explicit CUDA contract: no CPU fallback, real FP16 scaler update."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from guandan_marvel.config import (
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE, ModelConfig, PPOConfig, TrainConfig,
)
from guandan_marvel.evaluation.checkpoints import freeze_merged_policy, load_v2_policy
from guandan_marvel.evaluation.agents import V2PolicySession
from guandan_marvel.training.local import IPPOTrainer
from guandan_marvel.training.smoke import CppSmokeGame

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable; GPU evidence is NOT RUN",
)


def _cuda_config(game, *, seed: int = 606) -> TrainConfig:
    cfg = TrainConfig(
        game=game.spec, device="cuda", algo_name="ippo", seed=seed,
        run_name="phase6-cuda",
        rollout_batch_size=1, collect_batch_size=1, mini_batches=1,
        max_sampler_bytes_per_rank=512 * 2**20, amp_enabled=True,
        # Explicit local 4 GB adaptation (recorded in config/semantics): the
        # production default initial scale (65536) overflowed FP16 gradients on
        # this tiny smoke model; the math/tolerances are unchanged.
        amp_init_scale=1024.0,
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


@pytest.mark.slow
def test_cuda_fp16_scaler_real_update_and_same_device_resume(tmp_path: Path):
    game = CppSmokeGame(512)
    cfg = _cuda_config(game)
    torch.cuda.reset_peak_memory_stats()
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "cuda")
    assert trainer.device.type == "cuda"
    assert next(trainer.agent_resume.parameters()).device.type == "cuda"
    assert trainer.grad_scaler.is_enabled()
    assert trainer.policy_router is None
    before = {key: value.detach().clone() for key, value in trainer.agent_resume.state_dict().items()}
    metrics = trainer.train_iteration()
    after = trainer.agent_resume.state_dict()
    changed = [key for key in before if not torch.equal(before[key], after[key])]
    assert changed, "an AMP optimizer step must change at least one parameter"
    assert all(bool(torch.isfinite(value).all()) for value in after.values())
    assert metrics["runtime/amp_enabled"] is True
    assert metrics["runtime/amp_autocast_dtype"] == str(torch.float16)
    assert metrics["runtime/amp_scaler_enabled"] is True
    assert metrics["runtime/amp_scaler_scale"] > 0.0
    assert metrics["statistics/learner_updates"] == 1
    peak = int(torch.cuda.max_memory_allocated())
    assert peak > 0
    checkpoint = trainer.save_checkpoint(tmp_path / "cuda-iter1.pt")
    scale = float(trainer.grad_scaler.get_scale())

    restored = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "cuda-restored")
    decision = restored.load_checkpoint(checkpoint)
    assert decision.kind == "exact_resume"
    assert restored.device.type == "cuda"
    assert float(restored.grad_scaler.get_scale()) == pytest.approx(scale)
    for key, value in trainer.agent_resume.state_dict().items():
        assert torch.equal(value, restored.agent_resume.state_dict()[key])


@pytest.mark.slow
def test_cuda_evaluation_checkpoint_inference_has_no_cpu_fallback(tmp_path: Path):
    game = CppSmokeGame(512)
    cfg = _cuda_config(game, seed=608)
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "cuda-eval")
    trainer.train_iteration()
    checkpoint = freeze_merged_policy(trainer, tmp_path / "eval.pt")
    artifact = load_v2_policy(checkpoint, device="cuda", expected_algorithm="ippo")
    assert next(artifact.model.parameters()).device.type == "cuda"
    from guandan_marvel import cpp_game

    env = cpp_game.EnvBatch(1, auto_reset=False, max_horizon=512)
    step = env.reset(11)
    session = V2PolicySession(artifact, seed=7, temperature=1.0, device="cuda")
    probabilities, value = session.observe_score_and_value(step)
    assert probabilities.device.type == "cuda"
    assert bool(torch.isfinite(probabilities).all())
    assert value is not None and bool(torch.isfinite(value).all())
    index = session.select_index(probabilities, step)
    assert 0 <= index < probabilities.numel()
