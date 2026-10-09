from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.runtime.sampler import SamplerFailedStateError
from guandan_marvel.training.checkpoints import (
    CheckpointCompatibilityError, PRE_REPAIR_TRAINING_CHECKPOINT_SCHEMA,
    TrainingCheckpointStore, classify_resume, compute_source_fingerprint,
)
from guandan_marvel.training.local import IPPOTrainer, RUNTIME_SOURCE_COMPONENTS, VRPOTrainer
from guandan_marvel.training.smoke import PythonReferenceSmokeGame


def _config(
    root: Path,
    *,
    algorithm: str = "ippo",
    seed: int = 2101,
    gamma: float = 1.0,
    decoder_heads: int = 4,
    logging_suffix: str = "configured",
) -> tuple[TrainConfig, PythonReferenceSmokeGame]:
    game = PythonReferenceSmokeGame(512)
    cfg = TrainConfig(
        game=game.spec,
        device="cpu",
        algo_name=algorithm,
        logging_dir=str(root / logging_suffix),
        run_name=f"repair-{algorithm}",
        seed=seed,
        rollout_batch_size=1,
        collect_batch_size=1,
        replay_buffer_scale=1,
        mini_batches=1,
        total_iterations=3,
        max_sampler_bytes_per_rank=64 * 2**20,
        amp_enabled=False,
        checkpoint_replay_buffer=True,
        discount_gamma=gamma,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=decoder_heads,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.ppo.agent_epoch = 2
    cfg.vrpo.aux_epoch = 2
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg, game


@pytest.fixture(scope="module")
def ippo_iteration1_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("m03-repair-ippo")
    cfg, game = _config(root, algorithm="ippo", seed=2201)
    trainer = IPPOTrainer(cfg, game=game, run_dir=root / "source")
    trainer.train_iteration()
    path = trainer.save_checkpoint(root / "guandan-marvel-v2-ippo-iter000001-v3.pt")
    return root, cfg, game, path


def _assert_poisoned(trainer) -> None:
    assert trainer.failed
    assert trainer.iteration_status == "failed"
    assert trainer.sampler.failed
    with pytest.raises(RuntimeError, match="unusable"):
        trainer.train_iteration()
    with pytest.raises(RuntimeError):
        trainer.checkpoint_payload()
    with pytest.raises(RuntimeError):
        trainer.save_checkpoint(trainer.run_dir / "must-not-save.pt")
    with pytest.raises(SamplerFailedStateError):
        trainer.sampler.collect(agent_model=trainer.agent_model)


@pytest.mark.slow
def test_keyboard_interrupt_after_optimizer_step_poisoned_and_previous_checkpoint_recovers(
    ippo_iteration1_checkpoint,
) -> None:
    root, cfg, game, checkpoint = ippo_iteration1_checkpoint
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    trainer = IPPOTrainer(cfg, game=game, run_dir=root / "interrupt")
    assert trainer.load_checkpoint(checkpoint).kind == "exact_resume"
    before = {key: value.detach().clone() for key, value in trainer.agent_resume.state_dict().items()}
    original_step = trainer._optimizer_step
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        with pytest.raises(RuntimeError, match="active"):
            trainer.checkpoint_payload()
        result = original_step(*args, **kwargs)
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("injected after optimizer step")
        return result

    trainer._optimizer_step = interrupted
    with pytest.raises(KeyboardInterrupt, match="injected"):
        trainer.train_iteration()
    changed = sum(
        not torch.equal(before[key], trainer.agent_resume.state_dict()[key]) for key in before
    )
    assert changed > 0
    assert trainer.iteration == trainer.durable_updates == 1
    assert trainer.durable_learner_updates == 2
    assert trainer.learner_updates == 3
    assert trainer.failure["partial_learner_updates"] == 1
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash
    _assert_poisoned(trainer)

    recovered = IPPOTrainer(cfg, game=game, run_dir=root / "recovered")
    assert recovered.load_checkpoint(checkpoint).kind == "exact_resume"
    recovered.train_iteration()
    assert recovered.iteration == recovered.durable_updates == 2


@pytest.mark.slow
def test_runtime_error_mid_ippo_epoch_uses_same_fail_closed_path(ippo_iteration1_checkpoint) -> None:
    root, cfg, game, checkpoint = ippo_iteration1_checkpoint
    trainer = IPPOTrainer(cfg, game=game, run_dir=root / "runtime-error")
    trainer.load_checkpoint(checkpoint)
    original_step = trainer._optimizer_step

    def broken(*args, **kwargs):
        original_step(*args, **kwargs)
        raise RuntimeError("injected policy epoch failure")

    trainer._optimizer_step = broken
    with pytest.raises(RuntimeError, match="injected"):
        trainer.train_iteration()
    assert trainer.iteration == trainer.durable_updates == 1
    assert trainer.failure["partial_learner_updates"] == 1
    _assert_poisoned(trainer)


class _InterruptAfterForward(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    @property
    def decoder(self):
        return self.base.decoder

    @property
    def decoder_n_layers(self):
        return self.base.decoder_n_layers

    def forward(self, *args, **kwargs):
        result = self.base(*args, **kwargs)
        raise KeyboardInterrupt("injected after cached forward")


@pytest.mark.slow
def test_sampler_cached_forward_interrupt_poisoned_at_existing_checkpoint_boundary(
    ippo_iteration1_checkpoint,
) -> None:
    root, cfg, game, checkpoint = ippo_iteration1_checkpoint
    trainer = IPPOTrainer(cfg, game=game, run_dir=root / "cache-interrupt")
    trainer.load_checkpoint(checkpoint)
    trainer.agent_model = _InterruptAfterForward(trainer.agent_resume)
    with pytest.raises(KeyboardInterrupt, match="cached forward"):
        trainer.train_iteration()
    assert trainer.iteration == trainer.durable_updates == 1
    assert trainer.sampler.failed
    assert trainer.failure["partial_learner_updates"] == 0
    _assert_poisoned(trainer)


@pytest.mark.slow
def test_vrpo_interrupt_after_actor_before_critic_rejects_save_and_recovers(tmp_path: Path) -> None:
    cfg, game = _config(tmp_path, algorithm="vrpo", seed=2301)
    source = VRPOTrainer(cfg, game=game, run_dir=tmp_path / "vrpo-source")
    source.train_iteration()
    checkpoint = source.save_checkpoint(tmp_path / "guandan-marvel-v2-vrpo-iter000001-v3.pt")
    trainer = VRPOTrainer(cfg, game=game, run_dir=tmp_path / "vrpo-interrupt")
    trainer.load_checkpoint(checkpoint)
    critic_before = {key: value.detach().clone() for key, value in trainer.critic.state_dict().items()}
    original_step = trainer._optimizer_step
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        result = original_step(*args, **kwargs)
        calls += 1
        if calls == cfg.ppo.agent_epoch:
            raise KeyboardInterrupt("injected after actor before critic")
        return result

    trainer._optimizer_step = interrupted
    with pytest.raises(KeyboardInterrupt, match="before critic"):
        trainer.train_iteration()
    assert trainer.failure["stage"] == "policy_update"
    assert trainer.failure["partial_learner_updates"] == 2
    assert all(torch.equal(critic_before[key], trainer.critic.state_dict()[key]) for key in critic_before)
    _assert_poisoned(trainer)

    recovered = VRPOTrainer(cfg, game=game, run_dir=tmp_path / "vrpo-recovered")
    recovered.load_checkpoint(checkpoint)
    recovered.train_iteration()
    assert recovered.iteration == 2
    assert len(recovered.replay_buffer) == 1


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        ("gamma", "discount_gamma"),
        ("heads", "decoder_heads"),
        ("optimizer", "adam_beta1"),
        ("schedule", "agent_lr_decay_power"),
    ],
)
def test_real_semantic_config_changes_are_migration_and_failed_load_invalidates_instance(
    ippo_iteration1_checkpoint, mutation: str, expected_field: str,
) -> None:
    root, _cfg, game, checkpoint = ippo_iteration1_checkpoint
    cfg, _ = _config(root, algorithm="ippo", seed=2201)
    if mutation == "gamma":
        cfg.discount_gamma = 0.5
    elif mutation == "heads":
        cfg.model.decoder_heads = 2
    elif mutation == "optimizer":
        cfg.optimizer.adam_beta1 = 0.8
    elif mutation == "schedule":
        cfg.schedule.agent_lr_decay_power = 0.5
    trainer = IPPOTrainer(cfg, game=game, run_dir=root / f"semantic-{mutation}")
    with pytest.raises(CheckpointCompatibilityError, match=expected_field):
        trainer.load_checkpoint(checkpoint)
    _assert_poisoned(trainer)


def test_approved_logging_only_change_is_compatible_and_can_continue(ippo_iteration1_checkpoint) -> None:
    root, _cfg, game, checkpoint = ippo_iteration1_checkpoint
    cfg, _ = _config(root, algorithm="ippo", seed=2201, logging_suffix="different-log-dir")
    cfg.run_name = "different-run-name"
    trainer = IPPOTrainer(cfg, game=game, run_dir=root / "compatible")
    decision = trainer.load_checkpoint(checkpoint)
    assert decision.kind == "compatible_resume"
    assert any("logging_dir" in reason for reason in decision.reasons)
    trainer.train_iteration()
    assert trainer.iteration == 2


def test_payload_hash_tampering_rejected_and_instance_invalidated(ippo_iteration1_checkpoint, tmp_path: Path) -> None:
    root, cfg, game, checkpoint = ippo_iteration1_checkpoint
    payload = dict(TrainingCheckpointStore.load(checkpoint))
    payload["resolved_config"] = copy.deepcopy(payload["resolved_config"])
    payload["resolved_config"]["discount_gamma"] = 0.25
    tampered = tmp_path / "tampered-config-hash.pt"
    torch.save(payload, tampered)
    trainer = IPPOTrainer(cfg, game=game, run_dir=root / "tampered")
    with pytest.raises(CheckpointCompatibilityError, match="resolved_config_hash is inconsistent"):
        trainer.load_checkpoint(tampered)
    _assert_poisoned(trainer)


def test_source_fingerprint_inventory_and_component_change(tmp_path: Path) -> None:
    required = {
        "guandan_marvel/game.py", "guandan_marvel/model/transformer.py",
        "guandan_marvel/model/cache.py", "guandan_marvel/optim/common.py",
        "guandan_marvel/optim/muon.py", "guandan_marvel/training/schedules.py",
        "guandan/cards.py", "guandan/combos.py", "guandan/round.py", "guandan/state.py",
    }
    actual = {path for paths in RUNTIME_SOURCE_COMPONENTS.values() for path in paths}
    assert required <= actual

    (tmp_path / "rules").mkdir()
    (tmp_path / "model").mkdir()
    (tmp_path / "rules/engine.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "model/net.py").write_text("VALUE = 1\n", encoding="utf-8")
    components = {"rules": ("rules/engine.py",), "model": ("model/net.py",)}
    before = compute_source_fingerprint(tmp_path, components)
    (tmp_path / "model/net.py").write_text("VALUE = 2\n", encoding="utf-8")
    after = compute_source_fingerprint(tmp_path, components)
    assert before["hash"] != after["hash"]
    assert before["components"]["rules"]["hash"] == after["components"]["rules"]["hash"]
    assert before["components"]["model"]["hash"] != after["components"]["model"]["hash"]


def test_pre_repair_m03_checkpoint_is_migration_not_exact(ippo_iteration1_checkpoint) -> None:
    root, cfg, game, _checkpoint = ippo_iteration1_checkpoint
    old = Path("runs/M03_marvel_training_smoke/20261008_cpu_smoke_acceptance/ippo/checkpoints/guandan-marvel-v2-ippo-iter000001.pt")
    payload = TrainingCheckpointStore.load(old)
    assert payload["schema"] == PRE_REPAIR_TRAINING_CHECKPOINT_SCHEMA
    trainer = IPPOTrainer(cfg, game=game, run_dir=root / "old-m03")
    with pytest.raises(CheckpointCompatibilityError, match="pre-repair"):
        trainer.load_checkpoint(old)
    _assert_poisoned(trainer)
