from __future__ import annotations

from pathlib import Path
import pytest
import torch

from guandan_marvel import cpp_game
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.evaluation.checkpoints import EVALUATION_CHECKPOINT_SCHEMA
from guandan_marvel.kaggle.common import canonical_hash
from guandan_marvel.model import Agent
from guandan_marvel.runtime.sampler import TRAINING_DATA_PROVENANCE_SCHEMA
from guandan_marvel.training.checkpoints import build_backend_identity, canonical_json_hash, serialize_config


@pytest.fixture
def m07_cpp_evaluation_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(771)
    cfg = TrainConfig(
        game=cpp_game.GAME_SPEC,
        device="cpu",
        seed=771,
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
    source = {"schema": "guandan-marvel-source-fingerprint-v1", "components": {}}
    source = {**source, "hash": canonical_json_hash(source)}
    provenance = {
        "schema": TRAINING_DATA_PROVENANCE_SCHEMA,
        "domain": "training",
        "manifest_hash": cfg.training_data_manifest_hash,
        "generator_version": cfg.training_data_generator_version,
        "seed_semantics": "M07 evaluation fixture",
        "initial_env_seed": cfg.seed,
        "rollout_batch_size": 1,
        "deal_count": 0,
        "deal_hashes": [],
        "records_hash": canonical_json_hash([]),
        "complete": True,
    }
    provenance = {**provenance, "hash": canonical_hash(provenance)}
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
        "iteration": 1,
        "merged_policy": {"identity": "current_policy_copy", "beta": 0.0},
        "training_data_provenance": provenance,
        "resume_classification": "evaluation_only",
        "training_resume_allowed": False,
        "source_fingerprint": source,
        "source_hash": source["hash"],
        "source_parent_commit": "f" * 40,
    }
    path = tmp_path / "m07-cpp-eval.pt"
    torch.save(payload, path)
    return path
