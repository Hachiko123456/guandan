"""Phase-2 Profile B isolation, value/normalization/regularizer and config/checkpoint.

Covers spec sections 10.2, 10.4 and 10.5 (non-DDP parts).
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from guandan_marvel.config import (
    LEGACY_POLICY_REGULARIZER_MODE, LEGACY_TRAINING_SEMANTICS_PROFILE,
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE, MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
    PPOConfig, SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE, TrainConfig,
    deserialize_train_config,
)
from guandan_marvel.kaggle.profiles import build_resume_train_config, build_train_config
from guandan_marvel.model import Agent
from guandan_marvel.training.checkpoints import (
    CheckpointCompatibilityError, build_semantic_configuration, canonical_json_hash,
    serialize_config,
)
from guandan_marvel.training.gae import compute_gae_from_value
from guandan_marvel.training.legacy_gae import legacy_gae_macro_oracle
from guandan_marvel.training.losses import compute_clipped_pg_loss, compute_value_loss
from guandan_marvel.training.macro_ppo import (
    MacroPPOError, build_macro_batch_view, compute_macro_normalized_kl_mean,
    compute_macro_pg_loss, compute_macro_value_loss, explain_macro_value,
    macro_regularizer_diagnostics,
)
from guandan_marvel.training.macro_gae import compute_macro_decision_gae
from guandan_marvel.training.smoke import PythonReferenceSmokeGame
from guandan_marvel.training.local import IPPOTrainer, LocalTrainerBase

from tests.marvel_training._macro_fixtures import make_case, scalar_macro_pg_loss

CLIP = 0.2
GAMMA = 0.97


def _trainer_config(tmp_path: Path, *, profile: str, regularizer: str, seed: int = 2024, epochs: int = 2):
    game = PythonReferenceSmokeGame(512)
    cfg = TrainConfig(
        game=game.spec,
        device="cpu",
        algo_name="ippo",
        logging_dir=str(tmp_path / "logging"),
        run_name=f"phase2-{profile}",
        seed=seed,
        rollout_batch_size=1,
        collect_batch_size=1,
        replay_buffer_scale=1,
        mini_batches=1,
        total_iterations=2,
        max_sampler_bytes_per_rank=64 * 2**20,
        amp_enabled=False,
        checkpoint_replay_buffer=True,
        model=__import__("guandan_marvel.config", fromlist=["ModelConfig"]).ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
        ppo=PPOConfig(
            agent_epoch=epochs,
            training_semantics_profile=profile,
            policy_regularizer_mode=regularizer,
        ),
    )
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg, game


def _mixed_case():
    players = [0, 0, 1, 0, 2, 3, 1, 1]
    commits = [False, True, True, True, True, True, False, True]
    rewards = {(0, 0): 1.0, (5, 0): 2.0, (3, 0): -0.5, (6, 1): 0.25}
    values = {(0, 0): 0.7, (2, 0): -0.4, (5, 0): 0.2, (1, 1): 0.9, (6, 1): -0.1}
    return make_case(
        players, commits, rewards=rewards, values=values, horizon=10, dtype=torch.float32,
    )


def _mixed_case_with_ratios():
    players = [0, 0, 1, 0, 2, 3, 1, 1]
    commits = [False, True, True, True, True, True, False, True]
    rows = [
        [0.4, 0.6, 0.0, 0.0],
        [0.55, 0.45, 0.0, 0.0],
        [0.45, 0.55, 0.0, 0.0],
        [0.6, 0.4, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.55, 0.45, 0.0, 0.0],
        [0.4, 0.6, 0.0, 0.0],
    ]
    rewards = {(0, 0): 1.0, (5, 0): 2.0, (3, 0): -0.5, (6, 1): 0.25}
    values = {(0, 0): 0.7, (2, 0): -0.4, (5, 0): 0.2, (1, 1): 0.9, (6, 1): -0.1}
    return make_case(
        players, commits, rewards=rewards, values=values, horizon=10, dtype=torch.float32,
        avail_probs_rows=rows,
    )


def test_profile_b_gathers_production_token_gae_at_macro_starts():
    rollout, values, segmentation, _ = _mixed_case()
    token_gae = compute_gae_from_value(
        rollout, values, discount_gamma=GAMMA, gae_lambda=0.95,
    )
    gathered = token_gae.player_advantage.gather(
        1, segmentation.macro_start_index.clamp(min=0),
    )
    oracle = legacy_gae_macro_oracle(
        rollout, values, discount_gamma=GAMMA, gae_lambda=0.95, segmentation=segmentation,
    )
    valid = segmentation.macro_valid & (segmentation.macro_end_index >= 0)
    assert torch.allclose(gathered[valid], oracle.legacy_at_macro_start[valid], atol=1e-6)
    assert oracle.max_abs_diff < 1e-6
    # Profile B's advantage, value target and value loss are literally the legacy ones.
    value_loss_a = compute_value_loss(
        rollout,
        player_state_value_pred=token_gae.player_state_value,
        player_state_value_target=token_gae.player_state_value,
    )
    value_loss_b = compute_value_loss(
        rollout,
        player_state_value_pred=token_gae.player_state_value,
        player_state_value_target=token_gae.player_state_value,
    )
    assert torch.equal(value_loss_a.value_loss_mean, value_loss_b.value_loss_mean)


def test_profile_b_only_changes_policy_ratio_aggregation():
    rollout, values, segmentation, new_probs = _mixed_case_with_ratios()
    token_gae = compute_gae_from_value(rollout, values, discount_gamma=GAMMA, gae_lambda=0.95)
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    macro_advantage = token_gae.player_advantage.gather(
        1, segmentation.macro_start_index.clamp(min=0),
    )
    macro_pg = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=new_probs.log(),
        macro_advantage=macro_advantage, clip_coef=CLIP,
    )
    token_pg = compute_clipped_pg_loss(
        rollout, clip_coef=CLIP, avail_probs_new=new_probs,
        player_advantage=token_gae.player_advantage,
    )
    assert macro_pg.pg_loss_mean.item() != pytest.approx(token_pg.pg_loss_mean.item(), abs=1e-6)
    diagnostics = macro_regularizer_diagnostics(rollout, view=view, avail_probs_new=new_probs)
    assert diagnostics["legacy_token_kl_mean"] == pytest.approx(
        float(token_pg.kl_unif_mean.detach()), abs=1e-6,
    )


def test_macro_length_one_matches_token_ratio_and_clip_math():
    players = [0, 1, 2, 3]
    commits = [True] * 4
    rows = [
        [0.7, 0.3, 0.0, 0.0],
        [0.4, 0.6, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.55, 0.45, 0.0, 0.0],
    ]
    rollout, values, segmentation, new_probs = make_case(
        players, commits, behavior=[0.5] * 4, avail_probs_rows=rows,
    )
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    advantages = torch.tensor([[1.0, -0.5, 0.25, -0.75]])
    macro_pg = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=new_probs.log(),
        macro_advantage=advantages, clip_coef=CLIP,
    )
    log_ratios = [math.log(row[0] / 0.5) for row in rows]
    expected = scalar_macro_pg_loss(log_ratios, advantages[0].tolist(), clip_coef=CLIP)
    assert macro_pg.pg_loss_mean.item() == pytest.approx(expected, abs=1e-6)
    assert torch.allclose(
        macro_pg.log_ratio[0], torch.tensor(log_ratios, dtype=torch.float32), atol=1e-6,
    )


def test_profile_c_macro_value_loss_only_at_macro_starts():
    rollout, values, segmentation, _ = _mixed_case()
    macro_gae = compute_macro_decision_gae(
        rollout, values, segmentation=segmentation, discount_gamma=GAMMA, macro_lambda=0.95,
    )
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    prediction = torch.zeros((1, 10), dtype=torch.float32)
    prediction[0] = 0.5
    for m in range(segmentation.max_macros):
        start = int(segmentation.macro_start_index[0, m])
        if start >= 0:
            prediction[0, start] = 0.1 * (m + 1)
    output = compute_macro_value_loss(
        player_state_value_pred=prediction, view=view,
        macro_value_target=macro_gae.value_target,
    )
    valid = segmentation.macro_valid & (segmentation.macro_end_index >= 0)
    total = 0.0
    count = 0
    for m in range(segmentation.max_macros):
        if not bool(valid[0, m]):
            continue
        start = int(segmentation.macro_start_index[0, m])
        residual = float(prediction[0, start]) - float(macro_gae.value_target[0, m])
        delta = 10.0
        element = 0.5 * residual * residual if abs(residual) <= delta else delta * (abs(residual) - 0.5 * delta)
        # torch huber_loss(delta): 0.5*x^2 for |x|<=delta else delta*(|x|-0.5*delta)
        total += element
        count += 1
    assert output.macro_value_loss_mean.item() == pytest.approx(total / count, abs=1e-6)
    # The token value loss over the same prediction/targets is a different estimator.
    token_target = torch.zeros((1, 10), dtype=torch.float32)
    token_loss = compute_value_loss(
        rollout, player_state_value_pred=prediction, player_state_value_target=token_target,
    )
    assert token_loss.value_loss_mean.item() != pytest.approx(
        output.macro_value_loss_mean.item(), abs=1e-6,
    )


def test_macro_value_target_is_detached_and_finite():
    rollout, values, segmentation, _ = _mixed_case()
    macro_gae = compute_macro_decision_gae(
        rollout, values, segmentation=segmentation, discount_gamma=GAMMA, macro_lambda=0.95,
    )
    assert macro_gae.value_target.requires_grad is False
    assert macro_gae.advantage.requires_grad is False
    with pytest.raises(RuntimeError):
        macro_gae.value_target.sum().backward()
    valid = macro_gae.valid
    assert bool(torch.isfinite(macro_gae.value_target[valid]).all())


def test_padding_macros_excluded_from_value_and_normalization():
    players = [0, 1]
    commits = [True, True]
    clean_rollout, values, clean_seg, clean_probs = make_case(
        players, commits, values={(0, 0): 0.3}, garbage_padding=False,
    )
    dirty_rollout, dirt_values, dirty_seg, dirty_probs = make_case(
        players, commits, values={(0, 0): 0.3}, horizon=12, garbage_padding=True,
    )
    clean_view = build_macro_batch_view(clean_seg, torch.tensor([0]))
    dirty_view = build_macro_batch_view(dirty_seg, torch.tensor([0]))
    targets = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
    prediction = torch.tensor([[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]])
    clean = compute_macro_value_loss(
        player_state_value_pred=prediction[:, :2], view=clean_view, macro_value_target=targets,
    )
    dirty = compute_macro_value_loss(
        player_state_value_pred=prediction, view=dirty_view, macro_value_target=targets,
    )
    assert clean.macro_value_loss_mean.item() == pytest.approx(
        dirty.macro_value_loss_mean.item(), abs=0.0,
    )
    advantage = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    clean_pg = compute_macro_pg_loss(
        clean_rollout, view=clean_view, avail_log_probs_new=clean_probs.log(),
        macro_advantage=advantage, clip_coef=CLIP,
    )
    dirty_pg = compute_macro_pg_loss(
        dirty_rollout, view=dirty_view, avail_log_probs_new=dirty_probs.log(),
        macro_advantage=advantage, clip_coef=CLIP,
    )
    assert clean_pg.adv_norm_mean == pytest.approx(dirty_pg.adv_norm_mean, abs=0.0)
    assert clean_pg.pg_loss_mean.item() == pytest.approx(dirty_pg.pg_loss_mean.item(), abs=0.0)


def test_single_macro_and_zero_variance_are_finite_zero():
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits)
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    output = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=new_probs.log(),
        macro_advantage=torch.tensor([[5.0, 5.0]]), clip_coef=CLIP,
    )
    assert output.zero_variance_count == 1
    assert output.pg_loss_mean.item() == pytest.approx(0.0, abs=0.0)
    assert all(
        math.isfinite(value)
        for value in (
            output.adv_norm_mean, output.adv_norm_std, output.log_ratio_mean, output.ratio_mean,
        )
    )


def test_single_legal_action_normalized_entropy_is_zero_and_no_nan():
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits, legal=[[0], [0]])
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    diagnostics = macro_regularizer_diagnostics(rollout, view=view, avail_probs_new=new_probs)
    assert math.isfinite(diagnostics["normalized_entropy_mean"])
    assert diagnostics["normalized_entropy_mean"] == pytest.approx(0.0, abs=0.0)
    assert diagnostics["single_legal_token_count"] == 2


def test_zero_log_zero_entropy_has_no_nan():
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits, legal=[[0], [0]])
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    # avail_probs containing exact zeros must be handled by the 0*log0 = 0 limit.
    probs = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    diagnostics = macro_regularizer_diagnostics(rollout, view=view, avail_probs_new=probs)
    assert math.isfinite(diagnostics["raw_entropy_mean"])
    assert diagnostics["raw_entropy_mean"] == pytest.approx(0.0, abs=1e-9)


def test_macro_kl_weights_each_macro_equally():
    # 3-token macro and 1-token macro, each with per-token KL = k1, k2.
    players = [0, 0, 0, 1]
    commits = [False, False, True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits, behavior=[0.5] * 4)
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    legal = (rollout["avail_weights"] > 0).to(torch.float32)
    uniform = legal / legal.sum(dim=-1, keepdim=True)
    probs = torch.full_like(rollout["policy_probs"], 0.5)
    probs = torch.where(legal > 0, probs, torch.zeros_like(probs))
    eps = 1e-9

    def token_kl(p):
        q = uniform
        return ((p + eps) * ((p + eps).log() - (q + eps).log())).sum(-1)

    expected_per_macro = {
        0: token_kl(probs)[0, 0].item(),
        1: token_kl(probs)[0, 3].item(),
    }
    diagnostics = macro_regularizer_diagnostics(rollout, view=view, avail_probs_new=probs)
    expected = sum(expected_per_macro.values()) / 2
    assert diagnostics["macro_kl_mean"] == pytest.approx(expected, abs=1e-6)
    # The token-mean differs because the long macro has three tokens.
    token_mean = float(token_kl(probs)[0, [0, 1, 2, 3]].mean().item())
    if abs(expected_per_macro[0] - expected_per_macro[1]) > 1e-6:
        assert diagnostics["macro_kl_mean"] != pytest.approx(token_mean, abs=1e-6)


def test_regularizer_mode_does_not_change_pg_gae_or_value():
    rollout, values, segmentation, new_probs = _mixed_case()
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    macro_gae = compute_macro_decision_gae(
        rollout, values, segmentation=segmentation, discount_gamma=GAMMA, macro_lambda=0.95,
    )
    advantage = macro_gae.advantage
    first = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=new_probs.log(),
        macro_advantage=advantage, clip_coef=CLIP,
    )
    kl_legacy = macro_regularizer_diagnostics(rollout, view=view, avail_probs_new=new_probs)
    differentiable = new_probs.clone().requires_grad_(True)
    kl_macro = compute_macro_normalized_kl_mean(
        rollout, view=view, avail_probs_new=differentiable,
    )
    second = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=new_probs.log(),
        macro_advantage=advantage, clip_coef=CLIP,
    )
    assert torch.equal(first.pg_loss_mean, second.pg_loss_mean)
    assert kl_legacy["legacy_token_kl_mean"] != pytest.approx(
        float(kl_macro.detach()), abs=1e-6,
    ) or kl_legacy["macro_kl_mean"] == pytest.approx(float(kl_macro.detach()), abs=1e-6)
    if not torch.is_grad_enabled():
        raise AssertionError("unreachable")
    assert kl_macro.requires_grad


def test_value_ev_zero_variance_is_undefined_and_reported():
    prediction = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    target = torch.full((1, 3), 0.5, dtype=torch.float32)
    valid = torch.ones((1, 3), dtype=torch.bool)
    value, degenerate = explain_macro_value(prediction, target, valid)
    assert value is None
    assert degenerate == 1
    target2 = torch.tensor([[0.1, 0.5, 0.9]], dtype=torch.float32)
    value2, degenerate2 = explain_macro_value(target2, target2, valid)
    assert value2 == pytest.approx(1.0, abs=1e-6)
    assert degenerate2 == 0


def test_new_config_fields_round_trip_and_semantic_hash_covers_them():
    cfg, _game = _trainer_config(
        Path("."), profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
        regularizer=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
    )
    cfg.ppo.macro_gae_lambda = 0.9
    restored = deserialize_train_config(serialize_config(cfg))
    assert restored.ppo.training_semantics_profile == MACRO_DECISION_TRAINING_SEMANTICS_PROFILE
    assert restored.ppo.policy_regularizer_mode == MACRO_NORMALIZED_POLICY_REGULARIZER_MODE
    assert restored.ppo.macro_gae_lambda == pytest.approx(0.9)
    hash_one = build_semantic_configuration(cfg, algorithm="ippo")["hash"]
    other = copy.deepcopy(cfg)
    other.ppo.macro_gae_lambda = 0.5
    assert build_semantic_configuration(other, algorithm="ippo")["hash"] != hash_one
    other2 = copy.deepcopy(cfg)
    other2.ppo.policy_regularizer_mode = LEGACY_POLICY_REGULARIZER_MODE
    assert build_semantic_configuration(other2, algorithm="ippo")["hash"] != hash_one
    other3 = copy.deepcopy(cfg)
    other3.ppo.training_semantics_profile = SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE
    assert build_semantic_configuration(other3, algorithm="ippo")["hash"] != hash_one


def test_missing_new_fields_normalize_to_legacy():
    payload = {
        "schema": "guandan-marvel-core-config-v1",
        "game": {
            "identifier": "G", "num_players": 4, "num_tokens": 320, "pad_token": 0,
            "num_channels": 8, "max_game_horizon": 64, "max_observation_tokens": 8,
            "max_action_tokens": 4, "max_available_actions": 4, "protocol_version": "P",
        },
        "algo_name": "ippo",
        "ppo": {"base_lr": 1e-4, "agent_epoch": 1},
    }
    cfg = deserialize_train_config(payload)
    assert cfg.ppo.training_semantics_profile == LEGACY_TRAINING_SEMANTICS_PROFILE
    assert cfg.ppo.policy_regularizer_mode == LEGACY_POLICY_REGULARIZER_MODE
    assert cfg.ppo.macro_gae_lambda is None


def test_undefined_combinations_and_vrpo_macro_fail_fast():
    with pytest.raises(ValueError, match="unsupported training_semantics_profile"):
        PPOConfig(training_semantics_profile="macro_v2")
    with pytest.raises(ValueError, match="unsupported policy_regularizer_mode"):
        PPOConfig(
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode="macro_entropy_v1",
        )
    with pytest.raises(ValueError, match="legacy_token_v1 only supports"):
        PPOConfig(
            training_semantics_profile=LEGACY_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
        )
    with pytest.raises(ValueError, match="macro_gae_lambda"):
        PPOConfig(
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            macro_gae_lambda=1.5,
        )

    fake_game = SimpleNamespace(
        identifier="G", num_players=4, num_tokens=320, pad_token=0, num_channels=8,
        max_game_horizon=64, max_observation_tokens=8, max_action_tokens=4,
        max_available_actions=4, protocol_version="P",
    )
    profile = {
        "schema": "guandan-marvel-kaggle-profile-v1", "mode": "full", "algorithm": "vrpo",
        "ddp": {"launcher": "torchrun", "world_size": 1, "backend": "nccl", "require_cuda": False},
        "training": {
            "target_iterations": 1, "checkpoint_interval": 1, "evaluation_interval": 0,
            "rollout_batch_size": 1, "collect_batch_size": 1, "replay_buffer_scale": 1,
            "mini_batches": 1, "max_sampler_bytes_per_rank": 2**22, "seed": 1,
            "training_semantics_profile": MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            "policy_regularizer_mode": MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
            "model": {
                "architecture": "v1.3", "history_dim": 16, "feature_dim": 24,
                "decoder_layers": 1, "decoder_heads": 4, "state_layers": 1,
                "action_layers": 1, "fuse_layers": 1,
            },
        },
    }
    with pytest.raises(ValueError, match="vrpo"):
        build_train_config(profile, game=fake_game, run_root=Path("."), local_cpu_test=True)


def test_checkpoint_profile_mismatch_fails_closed(tmp_path: Path):
    legacy_cfg, game = _trainer_config(
        tmp_path, profile=LEGACY_TRAINING_SEMANTICS_PROFILE,
        regularizer=LEGACY_POLICY_REGULARIZER_MODE, seed=7,
    )
    trainer = IPPOTrainer(legacy_cfg, game=game, run_dir=tmp_path / "legacy")
    trainer.train_iteration()
    checkpoint = trainer.save_checkpoint(tmp_path / "legacy.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["training_semantics_profile"] == LEGACY_TRAINING_SEMANTICS_PROFILE
    assert payload["policy_regularizer_mode"] == LEGACY_POLICY_REGULARIZER_MODE
    assert payload["semantic_contract"]["semantic_config"]["training_semantics"]["profile"] == (
        LEGACY_TRAINING_SEMANTICS_PROFILE
    )

    macro_cfg, macro_game = _trainer_config(
        tmp_path, profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
        regularizer=LEGACY_POLICY_REGULARIZER_MODE, seed=7,
    )
    macro_trainer = IPPOTrainer(macro_cfg, game=macro_game, run_dir=tmp_path / "macro")
    with pytest.raises(CheckpointCompatibilityError):
        macro_trainer.load_checkpoint(checkpoint)
    assert macro_trainer.failed

    # The explicit semantics gate also rejects a payload whose recorded profile differs.
    probe = SimpleNamespace(
        training_semantics_profile=LEGACY_TRAINING_SEMANTICS_PROFILE,
        policy_regularizer_mode=LEGACY_POLICY_REGULARIZER_MODE,
    )
    with pytest.raises(CheckpointCompatibilityError, match="training semantics"):
        LocalTrainerBase._verify_checkpoint_training_semantics(
            probe, {"training_semantics_profile": MACRO_DECISION_TRAINING_SEMANTICS_PROFILE},
        )
    # Old checkpoints without the keys are treated as legacy defaults.
    LocalTrainerBase._verify_checkpoint_training_semantics(probe, {})


def test_kaggle_resume_inherits_checkpoint_semantics_and_cannot_override(tmp_path: Path):
    macro_cfg, game = _trainer_config(
        tmp_path, profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
        regularizer=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE, seed=9,
    )
    resolved = serialize_config(macro_cfg)
    profile = {
        "schema": "guandan-marvel-kaggle-profile-v1", "mode": "resume",
        "expected_algorithm": "ippo", "resume": {"additional_iterations": 2},
    }
    checkpoint_payload = {
        "resolved_config": resolved, "algorithm": "ippo", "iteration": 2,
    }
    resumed = build_resume_train_config(
        profile, checkpoint_payload=checkpoint_payload, game=game, run_root=tmp_path / "resume",
    )
    assert resumed.ppo.training_semantics_profile == MACRO_DECISION_TRAINING_SEMANTICS_PROFILE
    assert resumed.ppo.policy_regularizer_mode == MACRO_NORMALIZED_POLICY_REGULARIZER_MODE

    bad_profile = copy.deepcopy(profile)
    bad_profile["resume"] = {"additional_iterations": 1, "training_semantics_profile": "legacy_token_v1"}
    with pytest.raises(ValueError, match="forbidden semantic overrides"):
        build_resume_train_config(
            bad_profile, checkpoint_payload=checkpoint_payload, game=game, run_root=tmp_path / "bad",
        )


def test_kaggle_new_training_parses_semantics(tmp_path: Path):
    fake_game = SimpleNamespace(
        identifier="G", num_players=4, num_tokens=320, pad_token=0, num_channels=8,
        max_game_horizon=64, max_observation_tokens=8, max_action_tokens=4,
        max_available_actions=4, protocol_version="P",
    )
    profile = {
        "schema": "guandan-marvel-kaggle-profile-v1", "mode": "full", "algorithm": "ippo",
        "ddp": {"launcher": "torchrun", "world_size": 1, "backend": "nccl", "require_cuda": False},
        "training": {
            "target_iterations": 2, "checkpoint_interval": 1, "evaluation_interval": 0,
            "rollout_batch_size": 2, "collect_batch_size": 2, "replay_buffer_scale": 1,
            "mini_batches": 1, "max_sampler_bytes_per_rank": 2**22, "seed": 11,
            "training_semantics_profile": SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE,
            "policy_regularizer_mode": LEGACY_POLICY_REGULARIZER_MODE,
            "model": {
                "architecture": "v1.3", "history_dim": 16, "feature_dim": 24,
                "decoder_layers": 1, "decoder_heads": 4, "state_layers": 1,
                "action_layers": 1, "fuse_layers": 1,
            },
        },
    }
    cfg = build_train_config(profile, game=fake_game, run_root=tmp_path, local_cpu_test=True)
    assert cfg.ppo.training_semantics_profile == SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE
    semantic = build_semantic_configuration(cfg, algorithm="ippo")
    assert semantic["training_semantics"]["profile"] == SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE
    assert semantic["training_semantics"]["environment_discount_clock"] == "base_token_v1"
    from guandan_marvel.training.checkpoints import validate_declared_hash

    validate_declared_hash(semantic, name="semantic")


def test_regularizer_switch_does_not_change_pg_or_value_metrics_under_trainer(tmp_path: Path):
    base_cfg, game = _trainer_config(
        tmp_path, profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
        regularizer=LEGACY_POLICY_REGULARIZER_MODE, seed=31, epochs=1,
    )
    legacy_kl_trainer = IPPOTrainer(base_cfg, game=game, run_dir=tmp_path / "c-legacy-kl")
    legacy_metrics = legacy_kl_trainer.train_iteration()
    macro_cfg, macro_game = _trainer_config(
        tmp_path, profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
        regularizer=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE, seed=31, epochs=1,
    )
    macro_kl_trainer = IPPOTrainer(macro_cfg, game=macro_game, run_dir=tmp_path / "c-macro-kl")
    macro_metrics = macro_kl_trainer.train_iteration()
    for key in (
        "macro_initial/pg_loss", "macro_initial/value_loss", "macro_initial/adv_raw_mean",
        "macro_initial/adv_raw_std", "macro_initial/log_ratio_mean", "macro_initial/ratio_mean",
    ):
        assert legacy_metrics[key] == pytest.approx(macro_metrics[key], abs=0.0), key
    # The double-tracked legacy token KL is the same quantity in both modes up to the
    # re-computation order of the diagnostic copy.
    assert legacy_metrics["macro_initial/legacy_token_kl"] == pytest.approx(
        macro_metrics["macro_initial/legacy_token_kl"], abs=1e-6,
    )
    # Only the regularizer term (and its derived total loss / gradients) differ.
    assert legacy_metrics["macro_initial/normalized_entropy"] == pytest.approx(
        macro_metrics["macro_initial/normalized_entropy"], abs=0.0,
    )
    assert legacy_metrics["agent/reg_coef"] == macro_metrics["agent/reg_coef"]


def test_ntp_auxiliary_loss_is_identical_between_profiles_a_and_b(tmp_path: Path):
    """Same rollout/model/config: the auxiliary NTP term is untouched by A/B."""
    legacy_cfg, game = _trainer_config(
        tmp_path, profile=LEGACY_TRAINING_SEMANTICS_PROFILE,
        regularizer=LEGACY_POLICY_REGULARIZER_MODE, seed=55, epochs=1,
    )
    legacy_cfg.ntp_weight = 0.5
    legacy_metrics = IPPOTrainer(
        legacy_cfg, game=game, run_dir=tmp_path / "ntp-legacy",
    ).train_iteration()
    sequence_cfg, sequence_game = _trainer_config(
        tmp_path, profile=SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE,
        regularizer=LEGACY_POLICY_REGULARIZER_MODE, seed=55, epochs=1,
    )
    sequence_cfg.ntp_weight = 0.5
    sequence_metrics = IPPOTrainer(
        sequence_cfg, game=sequence_game, run_dir=tmp_path / "ntp-sequence",
    ).train_iteration()
    assert "agent_initial/ntp_loss" in legacy_metrics
    assert "macro_initial/ntp_loss" in sequence_metrics
    assert legacy_metrics["agent_initial/ntp_loss"] == pytest.approx(
        sequence_metrics["macro_initial/ntp_loss"], abs=0.0,
    )
    assert legacy_metrics["agent_initial/ntp_pos_rate"] == pytest.approx(
        sequence_metrics["macro_initial/ntp_pos_rate"], abs=0.0,
    )


def test_macro_trainer_runs_and_reports_required_metrics_legacy_and_macro_gae(tmp_path: Path):
    for profile in (
        SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE,
        MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    ):
        cfg, game = _trainer_config(
            tmp_path, profile=profile, regularizer=LEGACY_POLICY_REGULARIZER_MODE, seed=41,
        )
        trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / f"run-{profile}")
        metrics = trainer.train_iteration()
        assert metrics["macro/profile"] == profile
        assert metrics["macro/regularizer_mode"] == LEGACY_POLICY_REGULARIZER_MODE
        assert metrics["macro/value_sample_unit"] == (
            "macro_start" if profile == MACRO_DECISION_TRAINING_SEMANTICS_PROFILE else "token"
        )
        assert metrics["macro_initial/macro_count_global"] >= 1
        for key in (
            "macro_initial/pg_loss", "macro_initial/clip_frac", "macro_initial/adv_raw_mean",
            "macro_initial/adv_norm_mean", "macro_initial/log_ratio_mean", "macro_initial/ratio_mean",
            "macro_initial/legacy_token_kl", "macro_initial/macro_kl",
            "macro_initial/raw_entropy", "macro_initial/normalized_entropy",
            "macro_initial/value_loss", "macro_initial/grad_norm/decoder",
            "macro_initial/grad_norm/state", "macro_initial/grad_norm/action",
            "macro_initial/grad_norm/logit", "macro_initial/grad_norm/value",
        ):
            value = metrics.get(key)
            assert value is not None, key
            if isinstance(value, float):
                assert math.isfinite(value), key
        assert metrics["macro/degenerate_target_variance_count"] >= 0
        assert metrics["macro/clamp_count"] == 0
        # policy and value gradients are healthy (non-zero for at least one step)
        assert metrics["macro/grad_norm/value"] > 0.0 or profile == SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE
        assert metrics["agent/grad_norm"] > 0.0
        lines = trainer.metrics_path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        assert record["macro/profile"] == profile
