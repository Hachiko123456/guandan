"""Phase-6 repair: opponent tokens never enter Profile B value loss or combos."""
from __future__ import annotations

import pytest
import torch

from guandan_marvel.config import (
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
    SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE,
    OpponentPoolConfig, TrainConfig, validate_opponent_pool_combinations,
)
from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.training.losses import compute_value_loss

from tests.marvel_training._phase6_pool_fixtures import tiny_config
from guandan_marvel.training.smoke import CppSmokeGame


def _rollout(pred, target):
    return TensorDict({
        "num_steps": torch.tensor([3], dtype=torch.long),
        "player_ids": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "policy_train_mask": torch.tensor([[True, False, True]], dtype=torch.bool),
        "on_policy_valid": torch.tensor([[True, False, True]], dtype=torch.bool),
    }, batch_size=[1]), torch.tensor([pred], dtype=torch.float32), torch.tensor([target], dtype=torch.float32)


def test_profile_b_value_loss_ignores_opponent_tokens():
    rollout, prediction, target = _rollout([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    baseline = compute_value_loss(
        rollout,
        player_state_value_pred=prediction,
        player_state_value_target=target,
    ).value_loss_mean
    assert float(baseline) == pytest.approx(0.5, abs=1e-6)
    hostile_prediction = prediction.clone()
    hostile_prediction[0, 1] = 100.0
    changed = compute_value_loss(
        rollout,
        player_state_value_pred=hostile_prediction,
        player_state_value_target=target,
    ).value_loss_mean
    assert torch.equal(baseline, changed)
    hostile_target = target.clone()
    hostile_target[0, 1] = -999.0
    changed_target = compute_value_loss(
        rollout,
        player_state_value_pred=prediction,
        player_state_value_target=hostile_target,
    ).value_loss_mean
    assert torch.equal(baseline, changed_target)


def test_pool_combination_contract_rejects_unmasked_losses():
    with pytest.raises(ValueError, match="training_semantics_profile"):
        validate_opponent_pool_combinations(
            enabled=True,
            training_semantics_profile=SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
            symmetry_weight=0.0,
            ntp_weight=0.0,
        )
    with pytest.raises(ValueError, match="symmetry"):
        validate_opponent_pool_combinations(
            enabled=True,
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
            symmetry_weight=0.1,
            ntp_weight=0.0,
        )
    with pytest.raises(ValueError, match="ntp_weight"):
        validate_opponent_pool_combinations(
            enabled=True,
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
            symmetry_weight=0.0,
            ntp_weight=0.1,
        )


def test_train_config_rejects_pool_with_profile_b_and_ntp():
    game = CppSmokeGame(512)
    cfg = tiny_config(game)
    entry = {
        "id": "x", "kind": "v2_evaluation", "checkpoint": "unused.pt",
        "checkpoint_sha256": "0" * 64,
        "architecture": "v1.4-macro-fuse",
        "protocol_version": str(game.protocol_version),
        "backend": "cpp", "resolved_config_hash": "", "source_hash": "",
        "policy_version": 0, "weight": 1.0, "active": True, "tags": [],
    }
    with pytest.raises(ValueError, match="training_semantics_profile"):
        TrainConfig(
            game=game.spec, algo_name="ippo",
            ppo=type(cfg.ppo)(
                training_semantics_profile=SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE,
                policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
            ),
            opponent_pool=OpponentPoolConfig(enabled=True, entries=(entry,)),
            max_sampler_bytes_per_rank=512 * 2**20,
        )
    with pytest.raises(ValueError, match="ntp_weight"):
        TrainConfig(
            game=game.spec, algo_name="ippo", ntp_weight=0.1,
            ppo=type(cfg.ppo)(
                training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
                policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
            ),
            opponent_pool=OpponentPoolConfig(enabled=True, entries=(entry,)),
            max_sampler_bytes_per_rank=512 * 2**20,
        )
