from __future__ import annotations
from types import SimpleNamespace

from guandan_marvel.config import GameSpec
from guandan_marvel.kaggle.profiles import build_train_config
from guandan_marvel.training.checkpoints import build_semantic_configuration
from guandan_marvel.training.local import LocalTrainerBase


def _game():
    return SimpleNamespace(
        identifier="GuandanMARVELV2Cpp", num_players=4, num_tokens=320, pad_token=0,
        num_channels=512, max_game_horizon=4096, max_observation_tokens=48,
        max_action_tokens=8, max_available_actions=256,
        protocol_version="GD-ACTION-0.2-MARVEL|GD-ENV-0.2-MARVEL|GD-ENCODING-0.2-MARVEL",
    )


def _schedule():
    return {
        "schema": "guandan-marvel-training-context-schedule-v1",
        "contexts": [
            {"level_rank": 3, "leader_seat": 0, "previous_result": None, "weight": 2},
            {"level_rank": 15, "leader_seat": 3, "previous_result": {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN", "tribute_cancelled": False}, "weight": 1},
        ],
    }


def test_full_profile_freezes_context_schedule_and_warm_start_hyperparameters(tmp_path):
    profile = {
        "schema": "guandan-marvel-kaggle-profile-v1", "mode": "full", "algorithm": "ippo",
        "ddp": {"launcher": "torchrun", "world_size": 2, "backend": "nccl", "require_cuda": True},
        "training": {
            "target_iterations": 2000, "checkpoint_interval": 500, "evaluation_interval": 0,
            "rollout_batch_size": 4, "collect_batch_size": 4, "replay_buffer_scale": 8,
            "mini_batches": 2, "max_sampler_bytes_per_rank": 4294967296, "seed": 470700,
            "agent_epoch": 4, "critic_aux_epoch": 4, "base_lr": 1e-4,
            "base_clip_coef": 0.01, "base_reg_coef": 0.05,
            "lr_stable_iterations": 500, "reg_stable_iterations": 500,
            "training_context_schedule": _schedule(),
            "model": {"architecture": "v1.3", "history_dim": 128, "feature_dim": 256, "decoder_layers": 4, "decoder_heads": 4, "state_layers": 3, "action_layers": 2, "fuse_layers": 2},
        },
    }
    cfg = build_train_config(profile, game=_game(), run_root=tmp_path)
    assert cfg.training_context_schedule == _schedule()
    assert cfg.ppo.base_lr == 1e-4
    assert cfg.ppo.base_clip_coef == 0.01
    assert cfg.ppo.base_reg_coef == 0.05
    semantic = build_semantic_configuration(cfg, algorithm="ippo")
    assert semantic["sampling_and_replay"]["training_context_schedule"] == _schedule()


class _Module:
    def __init__(self): self.loaded = None
    def load_state_dict(self, state, strict=True): self.loaded = (state, strict)


class _Merge:
    def __init__(self): self.iteration = None
    def set_reference_iteration(self, value): self.iteration = value


def test_warm_start_loads_policy_only_and_resets_training_state():
    fake = SimpleNamespace(
        algorithm="ippo", cfg=SimpleNamespace(model=SimpleNamespace(architecture="v1.3"), game=GameSpec.from_game(_game())),
        agent_resume=_Module(), agent_merged=_Module(), merge_updater=_Merge(),
        _require_healthy=lambda: None,
    )
    payload = {
        "schema": "guandan-marvel-evaluation-checkpoint-v1", "algorithm": "ippo",
        "architecture": "v1.3", "protocol_version": _game().protocol_version,
        "agent_state_dict": {"weight": 1}, "iteration": 5000,
        "resolved_config_hash": "a" * 64, "source_hash": "b" * 64,
    }
    result = LocalTrainerBase.load_warm_start_policy(fake, payload)
    assert fake.agent_resume.loaded == ({"weight": 1}, True)
    assert fake.agent_merged.loaded == ({"weight": 1}, True)
    assert fake.merge_updater.iteration == 0
    assert result["source_iteration"] == 5000
    assert result["optimizer_reset"] is True
    assert result["sampler_reset"] is True
