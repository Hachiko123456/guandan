"""Stage-1 DMC config/semantics registration tests.

These tests pin the accept/reject matrix from
``docs/dmc_prompts/01_dmc_semantics_config.md`` and the locked defaults from
the DMC migration plan.  They fail on the pre-DMC implementation, which neither
knows ``algo_name="dmc"`` nor the ``dmc_v1`` semantics profile.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from guandan_marvel.config import (
    DMC_ACTION_FEATURE_VERSION,
    DMC_CHECKPOINT_SCHEMA,
    DMC_TRAINING_SEMANTICS_PROFILE,
    LEGACY_TRAINING_SEMANTICS_PROFILE,
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
    SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE,
    DMCRuntimeConfig,
    GameSpec,
    PPOConfig,
    TrainConfig,
    deserialize_train_config,
)
from guandan_marvel.phase6.config_types import OpponentPoolConfig

GAME = GameSpec(
    identifier="GuandanDMC",
    num_players=4,
    num_tokens=320,
    pad_token=0,
    num_channels=512,
    max_game_horizon=4096,
    max_observation_tokens=48,
    max_action_tokens=8,
    max_available_actions=256,
)

DMC_DEFAULT_VALUES = {
    "buffer_capacity": 500_000,
    "batch_size": 1024,
    "games_per_cycle": 256,
    "gradient_replay_ratio": 1.0,
    "epsilon_start": 0.10,
    "epsilon_final": 0.02,
    "epsilon_decay_samples": 5_000_000,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "action_feature_version": "GD-DMC-ACTION-1",
    "checkpoint_schema": "dmc-v1",
}


def _dmc_ppo(**overrides) -> PPOConfig:
    return PPOConfig(training_semantics_profile=DMC_TRAINING_SEMANTICS_PROFILE, **overrides)


def _train(algo_name: str, ppo: PPOConfig, **overrides) -> TrainConfig:
    return TrainConfig(game=GAME, algo_name=algo_name, ppo=ppo, **overrides)


def test_dmc_accepts_dmc_v1_profile():
    cfg = _train("dmc", _dmc_ppo(), run_name="dmc-accept", logging_dir="/tmp/dmc-accept")
    assert cfg.algo_name == "dmc"
    assert cfg.ppo.training_semantics_profile == DMC_TRAINING_SEMANTICS_PROFILE
    assert cfg.dmc == DMCRuntimeConfig()


def test_dmc_requires_dmc_v1_profile():
    with pytest.raises(ValueError, match="algo_name=dmc requires training_semantics_profile='dmc_v1'"):
        _train("dmc", PPOConfig())


@pytest.mark.parametrize(
    "profile",
    [MACRO_DECISION_TRAINING_SEMANTICS_PROFILE, SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE],
)
def test_dmc_rejects_macro_profiles(profile):
    with pytest.raises(ValueError, match="algo_name=dmc requires training_semantics_profile='dmc_v1'"):
        _train("dmc", PPOConfig(training_semantics_profile=profile))


def test_dmc_rejects_target_kl():
    with pytest.raises(ValueError, match="target_kl"):
        _dmc_ppo(target_kl=0.02)


def test_dmc_rejects_policy_regularizer_mode():
    with pytest.raises(ValueError, match="does not use policy_regularizer_mode"):
        _dmc_ppo(policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE)


def test_dmc_rejects_opponent_pool():
    pool = OpponentPoolConfig(enabled=True, entries=({"path": "pool.pt"},))
    with pytest.raises(ValueError, match="opponent_pool requires algo_name=ippo"):
        _train("dmc", _dmc_ppo(), opponent_pool=pool)


def test_ippo_rejects_dmc_v1_profile():
    with pytest.raises(ValueError, match="only defined for algo_name='dmc'"):
        _train("ippo", _dmc_ppo())


def test_dmc_runtime_defaults_snapshot():
    assert asdict(DMCRuntimeConfig()) == DMC_DEFAULT_VALUES
    assert DMC_ACTION_FEATURE_VERSION == "GD-DMC-ACTION-1"
    assert DMC_CHECKPOINT_SCHEMA == "dmc-v1"
    # The pre-existing defaults stay untouched.
    defaults = TrainConfig(game=GAME, run_name="defaults", logging_dir="/tmp/defaults")
    assert defaults.algo_name == "vrpo"
    assert defaults.ppo.training_semantics_profile == LEGACY_TRAINING_SEMANTICS_PROFILE
    assert defaults.dmc == DMCRuntimeConfig()


def test_dmc_runtime_block_is_serialized_only_for_dmc():
    legacy = TrainConfig(game=GAME, run_name="legacy", logging_dir="/tmp/legacy")
    legacy_payload = legacy.to_dict()
    assert "dmc" not in legacy_payload
    restored_legacy = deserialize_train_config(legacy_payload)
    assert restored_legacy.dmc == DMCRuntimeConfig()

    dmc_cfg = _train("dmc", _dmc_ppo(), run_name="dmc-serialize", logging_dir="/tmp/dmc-serialize")
    dmc_payload = dmc_cfg.to_dict()
    assert dmc_payload["dmc"] == DMC_DEFAULT_VALUES
    restored = deserialize_train_config(dmc_payload)
    assert restored.algo_name == "dmc"
    assert restored.ppo.training_semantics_profile == DMC_TRAINING_SEMANTICS_PROFILE
    assert restored.dmc == DMCRuntimeConfig()
    assert restored == dmc_cfg
