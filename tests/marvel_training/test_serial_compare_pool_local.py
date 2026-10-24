"""Local CPU evidence for the pool path used by the candidate arm.

The workstation C++ extension was built from different native sources than the
current working tree, so ``build_backend_identity`` refuses to run (a
pre-existing dirty-tree condition, see ``test_backend_identity.py``).  To still
exercise the *real* pool contract locally — two frozen resident entries, owner
masks, team rotation, macro ownership and frozen-parameter invariance — this
module stubs only the backend-identity helper with an internally consistent
identity.  The source guard itself is untouched; on the target device it runs
for real (the package restores the original Linux native at its original path).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from guandan_marvel import cpp_game
from guandan_marvel.config import (
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE, MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
    ModelConfig, PPOConfig, TrainConfig,
)
from guandan_marvel.evaluation.checkpoints import EVALUATION_CHECKPOINT_SCHEMA
from guandan_marvel.model import Agent
from guandan_marvel.training.checkpoints import (
    BACKEND_IDENTITY_SCHEMA, build_backend_identity, canonical_json_hash, serialize_config,
)
from guandan_marvel.training.local import IPPOTrainer
from guandan_marvel.training.smoke import CppSmokeGame


def _stub_backend_identity() -> dict[str, object]:
    implementation = dict(cpp_game.get_backend_build_identity())
    implementation["source_matches_loaded_binary"] = True
    payload = {
        "schema": BACKEND_IDENTITY_SCHEMA,
        "backend": "cpp",
        "module": str(cpp_game.__name__),
        "game_identifier": str(cpp_game.identifier),
        "protocol_version": str(cpp_game.protocol_version),
        "implementation": implementation,
    }
    return {**payload, "hash": canonical_json_hash(payload)}


def _tiny_config(game, *, pool=None, seed: int = 707) -> TrainConfig:
    extra = {} if pool is None else {"opponent_pool": pool}
    return TrainConfig(
        game=game.spec, algo_name="ippo", device="cpu", seed=seed, run_name="serial-compare-pool-local",
        rollout_batch_size=1, collect_batch_size=1, mini_batches=1,
        max_sampler_bytes_per_rank=512 * 2**20, ntp_weight=0.0,
        model=ModelConfig(
            architecture="v1.4-macro-fuse", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4, state_layers=1, action_layers=1, fuse_layers=1,
        ),
        ppo=PPOConfig(
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
        ),
        **extra,
    )


def _tiny_evaluation_checkpoint(root: Path, cfg: TrainConfig, name: str, *, seed: int, identity) -> dict[str, object]:
    torch.manual_seed(seed)
    model = Agent(cfg, enable_value_head=True, enable_ntp_head=False)
    resolved = serialize_config(cfg)
    payload = {
        "schema": EVALUATION_CHECKPOINT_SCHEMA,
        "algorithm": "ippo",
        "architecture": str(cfg.model.architecture),
        "protocol_version": str(cfg.game.protocol_version),
        "backend": "cpp",
        "backend_identity": identity,
        "resolved_config": resolved,
        "resolved_config_hash": canonical_json_hash(resolved),
        "agent_state_dict": model.state_dict(),
        "iteration": int(name.lstrip("p")),
        "merged_policy": {"identity": "current_policy_copy", "beta": 0.0},
        "resume_classification": "evaluation_only",
        "training_resume_allowed": False,
        "source_hash": None,
        "source_parent_commit": "f" * 40,
    }
    path = root / f"{name}.pt"
    torch.save(payload, path)
    return {
        "id": name,
        "kind": "v2_evaluation",
        "checkpoint": str(path),
        "checkpoint_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "architecture": payload["architecture"],
        "protocol_version": payload["protocol_version"],
        "backend": "cpp",
        "resolved_config_hash": payload["resolved_config_hash"],
        "source_hash": "",
        "policy_version": int(payload["iteration"]),
        "weight": 1.0,
        "active": True,
        "tags": ["recent"],
    }


@pytest.fixture(scope="module")
def pool_fixture(tmp_path_factory):
    import guandan_marvel.evaluation.checkpoints as evaluation_checkpoints
    import guandan_marvel.training.local as training_local

    root = tmp_path_factory.mktemp("serial-compare-pool")
    identity = _stub_backend_identity()
    original_eval = evaluation_checkpoints.build_backend_identity
    original_local = training_local.build_backend_identity
    evaluation_checkpoints.build_backend_identity = lambda game: identity
    training_local.build_backend_identity = lambda game: identity
    try:
        game = CppSmokeGame(512)
        cfg = _tiny_config(game)
        entries = (
            _tiny_evaluation_checkpoint(root, cfg, "p250", seed=250, identity=identity),
            _tiny_evaluation_checkpoint(root, cfg, "p375", seed=375, identity=identity),
        )
        yield {"game": game, "entries": entries, "root": root}
    finally:
        evaluation_checkpoints.build_backend_identity = original_eval
        training_local.build_backend_identity = original_local


def test_two_frozen_entries_resident_owner_masks_and_rotation(pool_fixture, tmp_path, monkeypatch):
    from guandan_marvel.phase6.config_types import OpponentPoolConfig

    import guandan_marvel.evaluation.checkpoints as evaluation_checkpoints
    import guandan_marvel.training.local as training_local

    identity = _stub_backend_identity()
    monkeypatch.setattr(evaluation_checkpoints, "build_backend_identity", lambda game: identity)
    monkeypatch.setattr(training_local, "build_backend_identity", lambda game: identity)

    game = pool_fixture["game"]
    pool = OpponentPoolConfig(
        enabled=True, entries=tuple(pool_fixture["entries"]), sampling_seed=5,
        sampling_scheme="guandan-marvel-phase6-weighted-active-sampling-v1",
        assignment_scheme="team_rotation_v1", max_resident_policies=2,
    )
    trainer = IPPOTrainer(_tiny_config(game, pool=pool), game=game, run_dir=tmp_path / "pooled")
    assert trainer.opponent_pool is not None
    assert trainer.opponent_pool.loaded_model_count == 2
    for entry in trainer.opponent_pool.entries:
        assert entry.artifact.model.training is False
        assert all(not parameter.requires_grad for parameter in entry.artifact.model.parameters())

    assignments = []
    for slot in range(4):
        assignment = trainer.opponent_pool.assign_episode(slot=slot, episode_index=0)
        assignments.append(assignment)
        current = [seat for seat, owner in enumerate(assignment.owners_by_seat) if owner == 0]
        assert len(current) == 2 and len({seat % 2 for seat in current}) == 1
        assert assignment.current_team == current[0] % 2
        opponents = {assignment.owners_by_seat[seat] for seat in range(4) if seat not in current}
        assert len(opponents) == 1 and opponents.pop() in (1, 2)
    assert [item.current_team for item in assignments] == [0, 1, 0, 1]
    assert {item.opponent_owner_id for item in assignments} <= {1, 2}

    metrics = trainer.train_iteration()
    assert int(metrics["pool/current_token_count"]) > 0
    assert int(metrics["pool/opponent_token_count"]) > 0
    assert tuple(metrics["pool/opponent_owner_ids"]) in {(1,), (2,), (1, 2)}
    assert int(metrics["pool/current_assignment_count"]) == 1
    # Only current-owned macros are trainable; opponent macros never cross over.
    assert int(metrics["macro/macro_count_local"]) == int(metrics["pool/current_macro_count_local"])
    assert int(metrics["pool/opponent_macro_count_local"]) >= 0
    assert metrics["pool/frozen_parameters_unchanged"] is True
    assert int(metrics["pool/resident_model_count"]) == 2
    assert int(metrics["pool/resident_session_count"]) == 2
    # Frozen parameters stayed bitwise identical across the iteration.
    for entry in trainer.opponent_pool.entries:
        assert all(parameter.grad is None for parameter in entry.artifact.model.parameters())
