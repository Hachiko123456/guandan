from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from guandan_marvel.buffer import BufferConfig, TrajectoryBuffer
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.training.checkpoints import (
    CheckpointCompatibilityError, PRE_REPAIR_TRAINING_CHECKPOINT_SCHEMA,
    TRAINING_CHECKPOINT_SCHEMA, build_backend_identity, build_execution_fingerprint,
    build_semantic_configuration, build_semantic_contract, classify_resume,
    canonical_json_hash, compute_source_fingerprint, serialize_config,
)
from guandan_marvel.training.local import IPPOTrainer, VRPOTrainer
from guandan_marvel.training.smoke import PythonReferenceSmokeGame
from guandan_marvel.training.validation import CRITIC_ONLY_FIELDS


def _config(tmp_path: Path, *, algorithm: str, seed: int = 1001) -> tuple[TrainConfig, PythonReferenceSmokeGame]:
    game = PythonReferenceSmokeGame(512)
    cfg = TrainConfig(
        game=game.spec,
        device="cpu",
        algo_name=algorithm,
        logging_dir=str(tmp_path / "configured-logging"),
        run_name=f"m03-{algorithm}-test",
        seed=seed,
        rollout_batch_size=1,
        collect_batch_size=1,
        replay_buffer_scale=1,
        mini_batches=1,
        total_iterations=2,
        max_sampler_bytes_per_rank=64 * 2**20,
        amp_enabled=False,
        checkpoint_replay_buffer=True,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.ppo.agent_epoch = 2
    cfg.vrpo.aux_epoch = 2
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg, game


def _all_finite(metrics: dict[str, object]) -> bool:
    for value in metrics.values():
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
            return False
    return True


@pytest.mark.slow
def test_ippo_real_complete_trajectory_multiple_epochs_metrics_and_jsonl(tmp_path: Path) -> None:
    cfg, game = _config(tmp_path, algorithm="ippo", seed=1101)
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "ippo")
    metrics = trainer.train_iteration()
    assert trainer.iteration == trainer.durable_updates == 1
    assert trainer.learner_updates == 2
    assert trainer.last_policy_epochs == 2
    assert trainer.completed_games_total == 1
    assert 0 < trainer.last_trajectory_num_steps[0] < 512
    assert not (set(trainer.last_trajectory_keys) & CRITIC_ONLY_FIELDS)
    assert metrics["health/nonfinite_losses"] == metrics["health/nonfinite_gradients"] == 0
    assert metrics["runtime/optimizer"] == "adamw"
    assert metrics["runtime/compile_enabled"] is False
    assert metrics["runtime/amp_enabled"] is False
    assert metrics["runtime/merged_policy_identity"] == "current_policy_copy"
    assert _all_finite(metrics)
    lines = trainer.metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["schema"] == "guandan-marvel-metrics-v1"
    assert record["statistics/completed_games_total"] == 1


@pytest.mark.slow
def test_vrpo_real_updates_q_targets_replay_and_metrics(tmp_path: Path) -> None:
    cfg, game = _config(tmp_path, algorithm="vrpo", seed=1201)
    trainer = VRPOTrainer(cfg, game=game, run_dir=tmp_path / "vrpo")
    metrics = trainer.train_iteration()
    assert trainer.iteration == trainer.durable_updates == 1
    assert trainer.learner_updates == 4  # 2 actor epochs + 2 critic epochs
    assert trainer.last_policy_epochs == 2
    assert trainer.last_critic_batch_sources == ("fresh", "replay")
    assert len(trainer.replay_buffer) == 1
    assert metrics["buffer/replay_buffer_size"] == 1
    assert metrics["critic_initial/qv_lbd_loss"] >= 0
    assert metrics["critic_initial/qv_td1_loss"] >= 0
    assert metrics["critic/qv_lbd_loss"] >= 0
    assert metrics["memory/replay_buffer_bytes_estimate"] > 0
    assert not (set(trainer.last_trajectory_keys) & CRITIC_ONLY_FIELDS)
    assert _all_finite(metrics)


@pytest.mark.slow
def test_ippo_exact_checkpoint_resume_matches_uninterrupted_continuation(tmp_path: Path) -> None:
    cfg, game = _config(tmp_path, algorithm="ippo", seed=1301)
    control = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "control")
    control.train_iteration()
    checkpoint = control.save_checkpoint(tmp_path / "guandan-marvel-v2-ippo-iter000001.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema"] == TRAINING_CHECKPOINT_SCHEMA
    assert payload["merged_policy"]["identity"] == "current_policy_copy"
    assert payload["merged_policy"]["not_smoothed_when_beta_zero"] is True
    control_metrics = control.train_iteration()
    control_state = {key: value.detach().clone() for key, value in control.agent_resume.state_dict().items()}

    restored = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "restored")
    decision = restored.load_checkpoint(checkpoint)
    assert decision.kind == "exact_resume"
    restored_metrics = restored.train_iteration()
    assert restored.iteration == control.iteration == 2
    assert restored.durable_updates == control.durable_updates == 2
    assert restored.sampler.total_env_steps == control.sampler.total_env_steps
    assert restored.sampler.total_game_steps == control.sampler.total_game_steps
    assert all(torch.equal(control_state[key], restored.agent_resume.state_dict()[key]) for key in control_state)
    assert restored_metrics["agent/pg_loss"] == pytest.approx(control_metrics["agent/pg_loss"], abs=0.0)


@pytest.mark.slow
def test_vrpo_checkpoint_restores_replay_and_continues_update(tmp_path: Path) -> None:
    cfg, game = _config(tmp_path, algorithm="vrpo", seed=1401)
    trainer = VRPOTrainer(cfg, game=game, run_dir=tmp_path / "vrpo-source")
    trainer.train_iteration()
    checkpoint = trainer.save_checkpoint(tmp_path / "guandan-marvel-v2-vrpo-iter000001.pt")
    restored = VRPOTrainer(cfg, game=game, run_dir=tmp_path / "vrpo-restored")
    decision = restored.load_checkpoint(checkpoint)
    assert decision.kind == "exact_resume"
    assert len(restored.replay_buffer) == 1
    restored.train_iteration()
    assert restored.iteration == 2
    assert restored.durable_updates == 2
    assert len(restored.replay_buffer) == 1

    corrupt_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    replay_storage = corrupt_payload["replay_buffer_state"]["storage"]
    steps = int(replay_storage["num_steps"][0])
    replay_storage["is_terminal"][0, steps - 1] = False
    corrupt_path = tmp_path / "corrupt-replay.pt"
    torch.save(corrupt_payload, corrupt_path)
    rejected = VRPOTrainer(cfg, game=game, run_dir=tmp_path / "vrpo-corrupt")
    with pytest.raises(ValueError, match="real terminal"):
        rejected.load_checkpoint(corrupt_path)
    assert rejected.failed and rejected.sampler.failed
    with pytest.raises(RuntimeError):
        rejected.train_iteration()
    with pytest.raises(RuntimeError):
        rejected.save_checkpoint(tmp_path / "corrupt-must-not-save.pt")


def test_resume_classification_all_categories_and_legacy_rejection(tmp_path: Path) -> None:
    cfg, game = _config(tmp_path, algorithm="vrpo", seed=1451)
    semantic = build_semantic_configuration(cfg, algorithm="vrpo")
    contract = build_semantic_contract(
        semantic_config=semantic,
        sampler_contract="sampler-v2", sampler_state_schema="sampler-state-v1",
        buffer_schema="buffer-v2", q_target_contract="q-v1",
    )
    source_file = tmp_path / "source.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    source = compute_source_fingerprint(tmp_path, {"test": ("source.py",)})
    execution = build_execution_fingerprint("cpu")
    backend_identity = build_backend_identity(game)
    resolved = serialize_config(cfg)
    base = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "semantic_contract": contract,
        "backend": backend_identity["backend"],
        "backend_identity": backend_identity,
        "source_fingerprint": source,
        "source_hash": source["hash"],
        "execution_fingerprint": execution,
        "resolved_config": resolved,
        "resolved_config_hash": canonical_json_hash(resolved),
        "agent_resume_state_dict": {}, "agent_state_dict": {},
        "agent_optimizer_state_dict": {}, "critic_state_dict": {},
        "critic_optimizer_state_dict": {}, "replay_buffer_state": {},
        "iteration": 1, "durable_updates": 1, "counters": {},
        "rng_state": {}, "sampler_state": {}, "schedule_state": {},
        "merged_policy": {},
    }
    exact = classify_resume(
        base, expected_contract=contract, expected_source_fingerprint=source,
        expected_execution_fingerprint=execution, expected_backend_identity=backend_identity, expected_config=resolved,
        require_replay=True,
    )
    assert exact.kind == "exact_resume"

    compatible_cfg = copy.deepcopy(resolved)
    compatible_cfg["logging_dir"] = "different-log-dir"
    compatible_cfg["run_name"] = "different-run-name"
    compatible = classify_resume(
        base, expected_contract=contract, expected_source_fingerprint=source,
        expected_execution_fingerprint=execution, expected_backend_identity=backend_identity, expected_config=compatible_cfg,
        require_replay=True,
    )
    assert compatible.kind == "compatible_resume"
    assert all("logging_dir" in reason or "run_name" in reason for reason in compatible.reasons)

    warm = dict(base); warm.pop("replay_buffer_state")
    assert classify_resume(
        warm, expected_contract=contract, expected_source_fingerprint=source,
        expected_execution_fingerprint=execution, expected_backend_identity=backend_identity, expected_config=resolved,
        require_replay=True,
    ).kind == "warm_restart"

    changed_cfg = copy.deepcopy(cfg)
    changed_cfg.discount_gamma = 0.5
    changed_semantic = build_semantic_configuration(changed_cfg, algorithm="vrpo")
    changed_contract = build_semantic_contract(
        semantic_config=changed_semantic,
        sampler_contract="sampler-v2", sampler_state_schema="sampler-state-v1",
        buffer_schema="buffer-v2", q_target_contract="q-v1",
    )
    migration = classify_resume(
        base, expected_contract=changed_contract, expected_source_fingerprint=source,
        expected_execution_fingerprint=execution, expected_backend_identity=backend_identity, expected_config=serialize_config(changed_cfg),
        require_replay=True,
    )
    assert migration.kind == "migration"
    assert any("discount_gamma" in reason for reason in migration.reasons)

    assert classify_resume(
        base, expected_contract=contract, expected_source_fingerprint=source,
        expected_execution_fingerprint=execution, expected_backend_identity=backend_identity, expected_config=resolved,
        require_replay=True, for_evaluation=True,
    ).kind == "evaluation_only"
    old_m03 = {"schema": PRE_REPAIR_TRAINING_CHECKPOINT_SCHEMA}
    assert classify_resume(
        old_m03, expected_contract=contract, expected_source_fingerprint=source,
        expected_execution_fingerprint=execution, expected_backend_identity=backend_identity, expected_config=resolved,
        require_replay=False,
    ).kind == "migration"
    legacy = {"checkpoint_version": 2, "action_version": "GD-ACTION-0.1", "environment_version": "GD-ENV-0.1"}
    with pytest.raises(CheckpointCompatibilityError, match="legacy V1"):
        classify_resume(
            legacy, expected_contract=contract, expected_source_fingerprint=source,
            expected_execution_fingerprint=execution, expected_backend_identity=backend_identity, expected_config=resolved,
            require_replay=False,
        )


def test_incomplete_rollout_does_not_add_replay_or_durable_update(tmp_path: Path, monkeypatch) -> None:
    cfg, game = _config(tmp_path, algorithm="vrpo", seed=1501)
    trainer = VRPOTrainer(cfg, game=game, run_dir=tmp_path / "vrpo-incomplete")
    incomplete = trainer.replay_buffer.empty(batch_size=1)
    incomplete["num_steps"].fill_(1)
    incomplete["on_policy_valid"].fill_(True)
    incomplete["behavior_action_prob"].fill_(1.0)
    incomplete["avail_weights"][:, :, 0].fill_(1.0)
    monkeypatch.setattr(trainer.sampler, "collect", lambda **_kwargs: incomplete)
    with pytest.raises(ValueError, match="real terminal"):
        trainer.train_iteration()
    assert trainer.durable_updates == 0
    assert trainer.iteration == 0
    assert len(trainer.replay_buffer) == 0
    assert trainer.failed
    with pytest.raises(RuntimeError, match="durable"):
        trainer.save_checkpoint(tmp_path / "invalid.pt")


def test_memory_preflight_rejects_before_sampler_allocation(tmp_path: Path) -> None:
    cfg, game = _config(tmp_path, algorithm="ippo", seed=1601)
    cfg.max_sampler_bytes_per_rank = 1
    with pytest.raises(MemoryError, match="exceeds"):
        IPPOTrainer(cfg, game=game, run_dir=tmp_path / "too-small")
