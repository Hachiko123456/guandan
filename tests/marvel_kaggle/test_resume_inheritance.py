from __future__ import annotations

import json
from pathlib import Path

import pytest

from guandan_marvel import cpp_game
from guandan_marvel.config import GameSpec
from guandan_marvel.kaggle.profiles import build_resume_train_config, build_train_config, require_strict_resume_decision
from guandan_marvel.model import Agent
from guandan_marvel.training.checkpoints import ResumeDecision, serialize_config


def _profiles():
    return json.loads(Path("configs/marvel_kaggle_profiles.json").read_text(encoding="utf-8"))["profiles"]


def test_full_checkpoint_rebuilds_resume_without_semantic_override(tmp_path):
    profiles = _profiles()
    original = build_train_config(profiles["full"], game=cpp_game, run_root=tmp_path / "full", local_cpu_test=True)
    payload = {
        "algorithm": original.algo_name,
        "iteration": 7,
        "resolved_config": serialize_config(original),
    }
    resumed = build_resume_train_config(
        profiles["resume"], checkpoint_payload=payload, game=cpp_game, run_root=tmp_path / "resume",
    )
    assert resumed.model == original.model
    assert resumed.rollout_batch_size == original.rollout_batch_size == 4
    assert resumed.collect_batch_size == original.collect_batch_size == 4
    assert resumed.replay_buffer_scale == original.replay_buffer_scale == 8
    assert resumed.mini_batches == original.mini_batches == 2
    assert resumed.ppo.agent_epoch == original.ppo.agent_epoch == 4
    assert resumed.vrpo.aux_epoch == original.vrpo.aux_epoch == 4
    assert resumed.optimizer == original.optimizer
    assert resumed.ddp == original.ddp
    assert resumed.seed == original.seed
    assert resumed.training_data_manifest_hash == original.training_data_manifest_hash
    assert resumed.total_iterations == 9
    model = Agent(resumed, enable_value_head=True)
    assert model.history_dim == 128
    assert model.feature_dim == 256
    assert model.decoder_n_layers == 4


def test_resume_profile_has_only_run_controls():
    resume = _profiles()["resume"]
    assert "training" not in resume and "algorithm" not in resume
    assert set(resume["resume"]) == {
        "additional_iterations", "checkpoint_interval", "evaluation_interval", "max_wall_seconds",
    }


def test_resume_rejects_forbidden_profile_override(tmp_path):
    profiles = _profiles()
    original = build_train_config(profiles["pilot"], game=cpp_game, run_root=tmp_path / "pilot", local_cpu_test=True)
    resume = json.loads(json.dumps(profiles["resume"]))
    resume["resume"]["model"] = {"history_dim": 1}
    with pytest.raises(ValueError, match="forbidden semantic overrides"):
        build_resume_train_config(
            resume,
            checkpoint_payload={"algorithm": original.algo_name, "iteration": 1, "resolved_config": serialize_config(original)},
            game=cpp_game,
            run_root=tmp_path / "resume",
        )


def test_only_explicit_run_control_compatible_resume_is_accepted():
    require_strict_resume_decision(ResumeDecision("exact_resume", ()))
    require_strict_resume_decision(ResumeDecision("compatible_resume", (
        "logging_dir: checkpoint='old' expected='new'",
        "total_iterations: checkpoint=1 expected=2",
    )))
    with pytest.raises(RuntimeError, match="non-run-control"):
        require_strict_resume_decision(ResumeDecision("compatible_resume", (
            "model.history_dim: checkpoint=32 expected=128",
        )))
