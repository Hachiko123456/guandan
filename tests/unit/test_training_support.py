from __future__ import annotations

import json
from pathlib import Path

import torch

from guandan.model import PolicyValueNet
from guandan.training.checkpoint import CURRENT_PROTOCOL_VERSIONS, load_checkpoint, save_checkpoint
from guandan.training.config import load_profile


def test_profiles_are_strict_and_have_expected_local_remote_budgets() -> None:
    local = load_profile("local_fast")
    remote = load_profile("remote_full")
    assert local.updates_per_algorithm == 5
    assert local.rollout_envs == 4
    assert remote.updates_per_algorithm == 100
    assert remote.rollout_envs == 8


def test_checkpoint_round_trip_and_resume_metadata(tmp_path: Path) -> None:
    model = PolicyValueNet(hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    metadata = save_checkpoint(path, model=model, optimizer=optimizer, profile="local_fast", algorithm="ippo", update=3, seed=7, protocol_versions=CURRENT_PROTOCOL_VERSIONS, git_commit="test-commit")
    assert metadata.update_count == 3
    restored = PolicyValueNet(hidden_dim=16)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    loaded = load_checkpoint(path, model=restored, optimizer=restored_optimizer, expected_profile="local_fast", expected_algorithm="ippo")
    assert loaded.metadata.update_count == 3
    assert loaded.metadata.seed_count == 7
    assert loaded.metadata.git_commit == "test-commit"


def test_checkpoint_rejects_protocol_mismatch(tmp_path: Path) -> None:
    model = PolicyValueNet(hidden_dim=8)
    path = tmp_path / "bad.pt"
    versions = dict(CURRENT_PROTOCOL_VERSIONS)
    versions["rules_version"] = "GD-RULES-bad"
    save_checkpoint(path, model=model, profile="local_fast", algorithm="ippo", protocol_versions=versions)
    try:
        load_checkpoint(path)
    except Exception as exc:
        assert "rules_version" in str(exc)
    else:
        raise AssertionError("incompatible protocol checkpoint was accepted")
