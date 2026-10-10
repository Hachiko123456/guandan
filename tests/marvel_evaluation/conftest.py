from __future__ import annotations

from pathlib import Path

import pytest
import torch

from guandan.model import PolicyValueNet
from guandan_marvel import cpp_game
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.evaluation.checkpoints import EVALUATION_CHECKPOINT_SCHEMA
from guandan_marvel.model import Agent
from guandan_marvel.runtime.sampler import TRAINING_DATA_PROVENANCE_SCHEMA
from guandan_marvel.training.checkpoints import (
    PRE_TRAINING_PROVENANCE_CHECKPOINT_SCHEMA,
    build_backend_identity,
    canonical_json_hash,
    serialize_config,
)


def _provenance(cfg: TrainConfig, *, deal_hashes=()) -> dict[str, object]:
    base = {
        "schema": TRAINING_DATA_PROVENANCE_SCHEMA,
        "domain": "training",
        "manifest_hash": cfg.training_data_manifest_hash,
        "generator_version": cfg.training_data_generator_version,
        "seed_semantics": "fixture actual deal hashes",
        "initial_env_seed": cfg.seed,
        "rollout_batch_size": cfg.rollout_batch_size,
        "deal_count": max(1, len(deal_hashes)),
        "deal_hashes": list(deal_hashes),
        "records_hash": canonical_json_hash(list(deal_hashes)),
        "complete": True,
    }
    return {**base, "hash": canonical_json_hash(base)}


@pytest.fixture
def v2_checkpoint_factory(tmp_path: Path):
    def create(name="candidate.pt", *, seed=17, old=False, mutate=0.0, deal_hashes=()):
        torch.manual_seed(seed)
        cfg = TrainConfig(
            game=cpp_game.GAME_SPEC,
            device="cpu",
            seed=seed,
            max_sampler_bytes_per_rank=64 * 1024 * 1024,
            rollout_batch_size=1,
            collect_batch_size=1,
            model=ModelConfig(
                architecture="v1.3", history_dim=8, feature_dim=8,
                decoder_layers=1, decoder_heads=1, state_layers=1,
                action_layers=1, fuse_layers=1, ffn_ratio=2,
            ),
        )
        model = Agent(cfg, enable_value_head=True).eval()
        state = model.state_dict()
        if mutate:
            key = next(iter(state))
            state[key] = state[key].clone()
            state[key].view(-1)[0] += float(mutate)
        resolved = serialize_config(cfg)
        source_base = {"schema": "guandan-marvel-source-fingerprint-v1", "components": {}}
        payload = {
            "schema": PRE_TRAINING_PROVENANCE_CHECKPOINT_SCHEMA if old else EVALUATION_CHECKPOINT_SCHEMA,
            "algorithm": "ippo",
            "architecture": "v1.3",
            "protocol_version": cpp_game.protocol_version,
            "backend": "cpp",
            "backend_identity": build_backend_identity(cpp_game),
            "resolved_config": resolved,
            "resolved_config_hash": canonical_json_hash(resolved),
            "agent_state_dict": state,
            "iteration": 3,
            "merged_policy": {"identity": "current_policy_copy", "beta": 0.0},
            "resume_classification": "evaluation_only",
            "training_resume_allowed": False,
            "source_fingerprint": {**source_base, "hash": canonical_json_hash(source_base)},
            "source_hash": canonical_json_hash(source_base),
            "source_parent_commit": "f" * 40,
        }
        if not old:
            payload["training_data_provenance"] = _provenance(cfg, deal_hashes=deal_hashes)
        path = tmp_path / name
        torch.save(payload, path)
        return path
    return create


@pytest.fixture
def v2_checkpoint(v2_checkpoint_factory) -> Path:
    return v2_checkpoint_factory()


@pytest.fixture
def second_v2_checkpoint(v2_checkpoint_factory) -> Path:
    return v2_checkpoint_factory("candidate-b.pt", seed=18, mutate=0.01)


@pytest.fixture
def old_v2_checkpoint(v2_checkpoint_factory) -> Path:
    return v2_checkpoint_factory("old-v4.pt", seed=19, old=True)


@pytest.fixture
def legacy_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(23)
    model = PolicyValueNet(hidden_dim=8)
    payload = {
        "policy_snapshot_version": 1,
        "model_state": model.state_dict(),
        "model_config": {
            "architecture": "mean-pooled-token-MLP",
            "hidden_dim": 8,
            "critic": "decentralized-V-shared-teams",
            "vocabulary": 256,
            "channels": 256,
        },
        "metadata": {
            "kind": "policy_snapshot",
            "source_checkpoint": "fixture",
            "source_git_commit": "0" * 40,
            "source_fingerprint": "fixture",
            "implementation_fingerprint": "fixture",
            "semantic_contract": {},
            "compatibility_schema_version": 1,
            "profile": "fixture",
            "algorithm": "ippo",
            "update": 1,
            "seed": 1,
            "model_version": "legacy-v1",
            "training_version": "legacy-v1",
            "protocol_versions": {
                "rules_version": "GD-RULES-0.1",
                "action_version": "GD-ACTION-0.1",
                "environment_version": "GD-ENV-0.1",
                "encoding_version": "GD-ENCODING-0.1",
            },
        },
        "training_deals": [],
    }
    path = tmp_path / "legacy-v1.pt"
    torch.save(payload, path)
    return path
