"""Phase-6 opponent pool: default-off inertness, ownership, recovery, rejection."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from guandan_marvel.config import (
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE, MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    ModelConfig, OpponentPoolConfig, PPOConfig, TrainConfig,
)
from guandan_marvel.evaluation.checkpoints import freeze_merged_policy
from guandan_marvel.evaluation.common import file_sha256
from guandan_marvel.phase6.design import canonical_hash
from guandan_marvel.phase6.opponent_pool import (
    OpponentPool, OpponentPoolError, PoolPolicyRouter, SeatAssignment,
    build_manifest, load_manifest, write_manifest,
)
from guandan_marvel.training.checkpoints import CheckpointCompatibilityError
from guandan_marvel.training.local import IPPOTrainer
from guandan_marvel.training.smoke import CppSmokeGame

LEGACY_TRAJECTORY_KEYS = {
    "num_steps", "player_ids", "obs_tokens", "obs_channels", "avail_tokens",
    "avail_weights", "avail_is_commit", "policy_probs", "behavior_action_prob",
    "on_policy_valid", "action_taken", "rewards", "is_terminal",
}
POOL_KEYS = {
    "policy_owner", "policy_train_mask", "opponent_checkpoint_id",
    "opponent_policy_version", "assignment_id", "assigned_seats",
}


def _config(game, *, pool=None, seed: int = 707) -> TrainConfig:
    cfg = TrainConfig(
        game=game.spec, device="cpu", algo_name="ippo", seed=seed,
        run_name="phase6-pool-cpu",
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
        opponent_pool=OpponentPoolConfig() if pool is None else pool,
    )
    cfg.ppo.agent_epoch = 1
    cfg.optimizer.use_muon = False
    return cfg


@pytest.fixture(scope="module")
def pool_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase6-pool-fixture")
    game = CppSmokeGame(512)
    cfg = _config(game)
    trainer = IPPOTrainer(cfg, game=game, run_dir=root / "seed-trainer")
    trainer.train_iteration()
    opponent = freeze_merged_policy(trainer, root / "opponent-eval.pt")
    entry = {
        "id": "opponent-a",
        "kind": "v2_evaluation",
        "checkpoint": str(opponent),
        "checkpoint_sha256": file_sha256(opponent),
        "architecture": str(cfg.model.architecture),
        "protocol_version": str(cfg.game.protocol_version),
        "backend": trainer.backend_name,
        "resolved_config_hash": trainer.resolved_config_hash,
        "source_hash": trainer.source_hash,
        "policy_version": 4,
        "weight": 1.0,
        "active": True,
        "tags": ["recent"],
    }
    return {"root": root, "game": game, "entry": entry}


def _pool_config(entry, **kwargs) -> OpponentPoolConfig:
    return OpponentPoolConfig(enabled=True, entries=(entry,), sampling_seed=11, **kwargs)


def _trainer(fixture, name: str, *, pool=None, seed: int = 707) -> IPPOTrainer:
    cfg = _config(fixture["game"], pool=pool, seed=seed)
    return IPPOTrainer(cfg, game=fixture["game"], run_dir=fixture["root"] / name)


def _state_equals(left, right) -> bool:
    left_state, right_state = left.state_dict(), right.state_dict()
    return set(left_state) == set(right_state) and all(
        torch.equal(left_state[key], right_state[key]) for key in left_state
    )


def test_disabled_pool_is_inert_and_has_no_extra_forward(pool_fixture, monkeypatch, tmp_path):
    def _must_not_run(*args, **kwargs):
        raise AssertionError("pool router ran while the pool was disabled")

    monkeypatch.setattr(PoolPolicyRouter, "policy_probabilities", _must_not_run)
    monkeypatch.setattr(PoolPolicyRouter, "prepare", _must_not_run)
    trainer = _trainer(pool_fixture, "disabled")
    assert trainer.policy_router is None
    assert trainer.opponent_pool is None
    assert not (trainer.run_dir / "opponent_pool_manifest.json").exists()
    metrics = trainer.train_iteration()
    assert set(trainer.last_trajectory_keys) == LEGACY_TRAJECTORY_KEYS
    assert trainer.sampler.state_dict()["policy_router"] is None
    assert not (set(trainer.last_trajectory_keys) & POOL_KEYS)
    assert metrics["statistics/learner_updates"] == 1
    assert "pool/current_token_count" not in metrics
    # An explicitly disabled configuration is the same training run.
    reference = _trainer(pool_fixture, "disabled-reference")
    reference_metrics = reference.train_iteration()
    assert _state_equals(trainer.agent_resume, reference.agent_resume)
    assert metrics["statistics/learner_updates"] == reference_metrics["statistics/learner_updates"]


def test_pool_iteration_separates_owners_and_never_trains_the_opponent(pool_fixture):
    entry = pool_fixture["entry"]
    trainer = _trainer(pool_fixture, "pool-on", pool=_pool_config(entry))
    opponent_before = trainer.policy_router.frozen_parameter_hashes() if trainer.policy_router.sessions else {
        entry_id: hash_value for entry_id, hash_value in trainer.opponent_pool.parameter_hashes().items()
    }
    current_before = {key: value.clone() for key, value in trainer.agent_resume.state_dict().items()}
    metrics = trainer.train_iteration()
    assert set(trainer.last_trajectory_keys) == LEGACY_TRAJECTORY_KEYS | POOL_KEYS
    assert metrics["pool/enabled"] is True
    assert metrics["pool/current_token_count"] > 0
    assert metrics["pool/opponent_token_count"] > 0
    assert metrics["pool/current_macro_count_local"] > 0
    assert metrics["pool/frozen_parameters_unchanged"] is True
    assert metrics["pool/resident_model_count"] == 1
    current_after = trainer.agent_resume.state_dict()
    assert any(not torch.equal(current_before[key], current_after[key]) for key in current_before)
    opponent_after = trainer.policy_router.frozen_parameter_hashes()
    if opponent_before:
        assert opponent_after == opponent_before
    trainer._verify_frozen_excluded_from_optimizer(trainer.agent_optimizer)
    manifest_path = trainer.run_dir / "opponent_pool_manifest.json"
    assert manifest_path.is_file()
    manifest = load_manifest(manifest_path, validate_checkpoints=True)
    assert manifest["manifest_hash"] == trainer.opponent_pool.manifest_hash
    assignment = trainer.opponent_pool.assignment(slot=0)
    current_seats = [seat for seat, owner in enumerate(assignment.owners_by_seat) if owner == 0]
    assert len(current_seats) == 2
    assert len({seat % 2 for seat in current_seats}) == 1


def test_pool_exact_resume_matches_uninterrupted_continuation(pool_fixture):
    pool = _pool_config(pool_fixture["entry"])
    control = _trainer(pool_fixture, "control", pool=pool)
    control.train_iteration()
    checkpoint = control.save_checkpoint(pool_fixture["root"] / "pool-iter1.pt")
    control.train_iteration()

    restored = _trainer(pool_fixture, "restored", pool=pool)
    decision = restored.load_checkpoint(checkpoint)
    assert decision.kind == "exact_resume"
    restored.train_iteration()
    assert restored.iteration == control.iteration == 2
    assert restored.completed_games_total == control.completed_games_total
    assert _state_equals(restored.agent_resume, control.agent_resume)
    assert restored.sampler.state_dict()["policy_router"]["manifest_hash"] == (
        control.sampler.state_dict()["policy_router"]["manifest_hash"]
    )


def test_pool_checkpoint_rejects_missing_or_forged_state(pool_fixture):
    pool = _pool_config(pool_fixture["entry"])
    trainer = _trainer(pool_fixture, "checkpoint-source", pool=pool)
    trainer.train_iteration()
    checkpoint = trainer.save_checkpoint(pool_fixture["root"] / "pool-source.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    missing = copy.deepcopy(payload)
    del missing["opponent_pool"]
    # A failed load poisons a trainer, so every negative case uses a fresh one.
    with pytest.raises(CheckpointCompatibilityError, match="requires an opponent_pool"):
        _trainer(pool_fixture, "missing-block", pool=pool).load_checkpoint_payload(missing)

    forged_manifest = copy.deepcopy(payload)
    forged_manifest["opponent_pool"]["manifest_hash"] = "0" * 64
    with pytest.raises(CheckpointCompatibilityError, match="hash is inconsistent"):
        _trainer(pool_fixture, "forged-manifest", pool=pool).load_checkpoint_payload(
            forged_manifest
        )

    forged_router = copy.deepcopy(payload)
    forged_router["sampler_state"]["policy_router"]["state_hash"] = "1" * 64
    with pytest.raises(CheckpointCompatibilityError, match="hash"):
        _trainer(pool_fixture, "forged-router", pool=pool).load_checkpoint_payload(
            forged_router
        )

    forged_entries = copy.deepcopy(payload)
    forged_entries["opponent_pool"]["entries"][0]["checkpoint_sha256"] = "2" * 64
    forged_entries["opponent_pool"].pop("hash")
    forged_entries["opponent_pool"]["hash"] = canonical_hash(forged_entries["opponent_pool"])
    with pytest.raises(CheckpointCompatibilityError, match="entries differ"):
        _trainer(pool_fixture, "forged-entries", pool=pool).load_checkpoint_payload(
            forged_entries
        )

    disabled = _trainer(pool_fixture, "checkpoint-disabled")
    with pytest.raises(CheckpointCompatibilityError):
        disabled.load_checkpoint_payload(payload)
    with pytest.raises(CheckpointCompatibilityError, match="runtime pool is disabled"):
        _trainer(pool_fixture, "checkpoint-disabled-guard")._verify_opponent_pool_checkpoint(
            payload
        )


def test_pool_manifest_is_strict_and_rejects_incompatible_checkpoints(pool_fixture, tmp_path):
    entry = pool_fixture["entry"]
    manifest = build_manifest(entries=[entry], sampling_seed=11)
    path = write_manifest(manifest, tmp_path / "manifest.json")
    loaded = load_manifest(path, validate_checkpoints=True)
    assert loaded["manifest_hash"] == manifest["manifest_hash"]

    missing = copy.deepcopy(loaded)
    del missing["sampling"]
    write_manifest(missing, tmp_path / "missing.json")
    with pytest.raises(OpponentPoolError, match="missing keys"):
        load_manifest(tmp_path / "missing.json")

    extra = copy.deepcopy(loaded)
    extra["unexpected"] = True
    write_manifest(extra, tmp_path / "extra.json")
    with pytest.raises(OpponentPoolError, match="unknown keys"):
        load_manifest(tmp_path / "extra.json")

    changed = copy.deepcopy(loaded)
    changed["entries"][0]["checkpoint_sha256"] = "0" * 64
    write_manifest(changed, tmp_path / "changed.json")
    with pytest.raises(OpponentPoolError, match="SHA256 changed"):
        load_manifest(tmp_path / "changed.json")

    assert manifest["manifest_hash"] == build_manifest(
        entries=[entry], sampling_seed=11,
    )["manifest_hash"], "the manifest hash must not depend on wall-clock creation time"

    with pytest.raises(OpponentPoolError, match="architecture"):
        OpponentPool.from_config(
            _pool_config(entry), device="cpu",
            expected_architecture="v1.3",
            expected_protocol_version=str(pool_fixture["game"].protocol_version),
            expected_backend="cpp",
            expected_backend_identity_hash=None,
            model_feature_identity={
                "history_dim": 16, "feature_dim": 24, "decoder_layers": 1,
                "decoder_heads": 4, "state_layers": 1, "action_layers": 1,
                "fuse_layers": 1,
            },
            batch_slots=1,
        )


def test_sampling_assignment_rng_and_pool_state_recover_exactly(pool_fixture):
    entry = pool_fixture["entry"]
    config = _pool_config(entry)
    common = dict(
        device="cpu", expected_architecture="v1.4-macro-fuse",
        expected_protocol_version=str(pool_fixture["game"].protocol_version),
        expected_backend="cpp", expected_backend_identity_hash=None,
        model_feature_identity={
            "history_dim": 16, "feature_dim": 24, "decoder_layers": 1,
            "decoder_heads": 4, "state_layers": 1, "action_layers": 1,
            "fuse_layers": 1,
        },
        batch_slots=1,
    )
    first = OpponentPool.from_config(config, **common)
    second = OpponentPool.from_config(config, **common)
    sequence_first, sequence_second = [], []
    for episode in range(4):
        assignment_first = first.assign_episode(slot=0, episode_index=episode)
        assignment_second = second.assign_episode(slot=0, episode_index=episode)
        assert assignment_first.payload() == assignment_second.payload()
        sequence_first.append(assignment_first.payload())
        sequence_second.append(assignment_second.payload())
        with pytest.raises(OpponentPoolError, match="hot-switched"):
            first.assign_episode(slot=0, episode_index=episode + 1)
        first.close_episode(slot=0)
        second.close_episode(slot=0)
    assert sequence_first == sequence_second
    teams = [row["current_team"] for row in sequence_first]
    assert teams == [0, 1, 0, 1]
    current_seats = [
        tuple(seat for seat, owner in enumerate(row["owners_by_seat"]) if owner == 0)
        for row in sequence_first
    ]
    for seat in range(4):
        assert sum(seat in seats for seats in current_seats) == 2

    state = first.state_dict()
    restored = OpponentPool.from_config(config, **common)
    restored.load_state_dict(state)
    next_first = first.assign_episode(slot=0, episode_index=4)
    next_restored = restored.assign_episode(slot=0, episode_index=4)
    assert next_first.payload() == next_restored.payload()

    tampered = dict(state)
    tampered["draw_index"] = int(tampered["draw_index"]) + 1
    with pytest.raises(OpponentPoolError, match="state hash"):
        OpponentPool.from_config(config, **common).load_state_dict(tampered)

    wrong_manifest = dict(state)
    wrong_manifest["manifest_hash"] = "0" * 64
    wrong_manifest["hash"] = canonical_hash(
        {key: value for key, value in wrong_manifest.items() if key != "hash"}
    )
    with pytest.raises(OpponentPoolError, match="manifest differs"):
        OpponentPool.from_config(config, **common).load_state_dict(wrong_manifest)

    assignment_payload = dict(next_first.payload())
    assignment_payload["assignment_hash"] = "0" * 64
    with pytest.raises(OpponentPoolError, match="assignment hash"):
        SeatAssignment.from_payload(
            assignment_payload, expected_manifest_hash=first.manifest_hash,
        )


def test_warm_start_cannot_ignore_pool_assignments(pool_fixture):
    trainer = _trainer(pool_fixture, "warm-start", pool=_pool_config(pool_fixture["entry"]))
    with pytest.raises(ValueError, match="warm start cannot reconstruct"):
        trainer.load_warm_start_policy({"schema": "guandan-marvel-evaluation-checkpoint-v1"})
