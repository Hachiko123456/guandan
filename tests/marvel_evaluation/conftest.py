from __future__ import annotations

from pathlib import Path

import pytest
import torch

from guandan.model import PolicyValueNet
from guandan_marvel import cpp_game
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.evaluation.checkpoints import EVALUATION_CHECKPOINT_SCHEMA
from guandan_marvel.training.checkpoints import build_backend_identity, canonical_json_hash, serialize_config
from guandan_marvel.model import Agent


@pytest.fixture
def v2_checkpoint(tmp_path: Path) -> Path:
    cfg = TrainConfig(
        game=cpp_game.GAME_SPEC,
        device="cpu",
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
    resolved = serialize_config(cfg)
    payload = {
        "schema": EVALUATION_CHECKPOINT_SCHEMA,
        "algorithm": "ippo",
        "architecture": "v1.3",
        "protocol_version": cpp_game.protocol_version,
        "backend": "cpp",
        "backend_identity": build_backend_identity(cpp_game),
        "resolved_config": resolved,
        "resolved_config_hash": canonical_json_hash(resolved),
        "agent_state_dict": model.state_dict(),
        "iteration": 3,
        "merged_policy": {"identity": "current_policy_copy", "beta": 0.0},
        "resume_classification": "evaluation_only",
        "training_resume_allowed": False,
    }
    path = tmp_path / "candidate.pt"
    torch.save(payload, path)
    return path


@pytest.fixture
def legacy_checkpoint(tmp_path: Path) -> Path:
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
