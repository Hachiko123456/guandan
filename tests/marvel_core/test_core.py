from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import guandan
import guandan_marvel
from guandan_marvel.buffer import BufferConfig, TrajectoryBuffer
from guandan_marvel.config import CompileConfig, DDPConfig, OptimizerConfig, deserialize_train_config
from guandan_marvel.model import Agent, CentralCritic, TensorDict, TransformerConfig, TransformerModel
from guandan_marvel.model.trainable import TrainableModule
from guandan_marvel.runtime.checkpoints import CHECKPOINT_SCHEMA, CheckpointManager, CheckpointState, CheckpointStore
from guandan_marvel.runtime.compilation import ModuleCompiler, TorchCompileSettings
from guandan_marvel.runtime.distributed import DistributedRuntime
from guandan_marvel.runtime.logger import MetricSinks
from guandan_marvel.testing import FAKE_GAME_SPEC, fake_rollout, fake_train_config
from guandan_marvel.training.schedules import compute_agent_schedule


def test_independent_namespace_keeps_legacy_guandan_importable(tmp_path: Path) -> None:
    assert Path(guandan.__file__).name == "__init__.py"
    assert guandan_marvel.__name__ == "guandan_marvel"
    (tmp_path / "marvel.py").write_text("SENTINEL = 'reference-package'\n", encoding="utf-8")
    code = (
        "import json, marvel, guandan, guandan_marvel; "
        "print(json.dumps({'sentinel': marvel.SENTINEL, 'core': guandan_marvel.__version__, "
        "'guandan': guandan.__name__}))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd()), str(tmp_path)])
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"sentinel": "reference-package", "core": "0.1.0", "guandan": "guandan"}


def test_config_deserializes_nested_and_legacy_model_aliases() -> None:
    payload = {
        "game": dataclasses.asdict(FAKE_GAME_SPEC),
        "model": {
            "architecture": "v1.3",
            "history_embd": 16,
            "feature_embd": 24,
            "decoder_n_layers": 1,
            "decoder_n_heads": 4,
            "n_state_layers": 1,
            "n_action_layers": 1,
            "n_fuse_layers": 1,
        },
        "optimizer": {"use_muon": False, "weight_decay": 0.02},
        "compile": {"enabled": False, "backend": "eager"},
        "ddp": {"enabled": False},
    }
    cfg = deserialize_train_config(payload)
    assert cfg is not None
    assert cfg.game == FAKE_GAME_SPEC
    assert cfg.model.history_dim == 16
    assert cfg.model.feature_dim == 24
    assert cfg.optimizer.weight_decay == pytest.approx(0.02)
    assert cfg.use_compile is False
    assert deserialize_train_config(cfg) is cfg


def test_tensordict_validates_slices_clones_and_assignment() -> None:
    td = TensorDict({"x": torch.arange(12).reshape(3, 4), "y": torch.ones(3, 2)}, batch_size=[3])
    sliced = td[torch.tensor([2, 0])]
    assert sliced.batch_size == torch.Size([2])
    assert sliced["x"].tolist() == [[8, 9, 10, 11], [0, 1, 2, 3]]
    clone = td.clone()
    clone["x"][0, 0] = -1
    assert td["x"][0, 0].item() == 0
    td[torch.tensor([1])] = TensorDict({"x": torch.full((1, 4), 7), "y": torch.zeros(1, 2)}, batch_size=[1])
    assert td["x"][1].tolist() == [7, 7, 7, 7]
    with pytest.raises(ValueError, match="batch_size"):
        TensorDict({"bad": torch.zeros(2, 3)}, batch_size=[3])


def test_transformer_and_sparse_encoders_have_expected_shapes() -> None:
    transformer = TransformerModel(TransformerConfig(vocab_size=0, hidden_size=16, num_hidden_layers=1, num_attention_heads=4, intermediate_size=32, max_position_embeddings=8, pad_token_id=0))
    embeddings = torch.randn(2, 3, 16)
    output = transformer(inputs_embeds=embeddings, attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]), return_dict=True)
    assert output.last_hidden_state.shape == (2, 3, 16)

    agent = Agent(fake_train_config())
    mask = torch.tensor([[True, True, False], [True, False, False]])
    decoded = torch.randn(2, 3, agent.history_dim)
    channels = torch.randint(0, 2, (2, 3, agent.num_channels), dtype=torch.uint8)
    sparse_state = agent._encode_state_features(decoded, channels, mask, use_sparse=True)
    dense_state = agent._encode_state_features(decoded, channels, mask, use_sparse=False)
    assert sparse_state.shape == dense_state.shape == (2, 3, agent.feature_dim)
    assert torch.allclose(sparse_state, dense_state, atol=1e-5)

    tokens = torch.randint(1, agent.num_tokens, (2, 3, 4, 3))
    weights = torch.tensor([[[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], [[1, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]]], dtype=torch.float32)
    sparse_actions, legal = agent._encode_action_features(tokens, weights, mask, use_sparse=True)
    dense_actions, dense_legal = agent._encode_action_features(tokens, weights, mask, use_sparse=False)
    assert sparse_actions.shape == (2, 3, 4, agent.feature_dim)
    assert torch.equal(legal, dense_legal)
    assert torch.allclose(sparse_actions, dense_actions, atol=1e-5)


def test_fake_game_agent_and_critic_forward_shapes() -> None:
    cfg = fake_train_config()
    rollout = fake_rollout(batch_size=2, steps=3)
    agent = Agent(cfg, enable_value_head=True)
    critic = CentralCritic(cfg, output_mode="qvalue")
    agent_out = agent(rollout)
    critic_out = critic(rollout)
    assert agent_out is not None
    assert agent_out.avail_probs.shape == (2, 3, FAKE_GAME_SPEC.max_available_actions)
    assert agent_out.state_values.shape == (2, 3, FAKE_GAME_SPEC.num_players)
    assert critic_out.qvalues.shape == (2, 3, FAKE_GAME_SPEC.max_available_actions, FAKE_GAME_SPEC.num_players)
    valid = agent_out.mask
    assert torch.allclose(agent_out.avail_probs[valid].sum(-1), torch.ones_like(agent_out.avail_probs[valid].sum(-1)), atol=1e-5)


def _complete_batch(buffer: TrajectoryBuffer, count: int, marker: float) -> TensorDict:
    batch = buffer.empty(batch_size=count)
    batch["num_steps"].fill_(2)
    batch["rewards"].fill_(marker)
    return batch


def test_buffer_replay_and_state_round_trip() -> None:
    buffer = TrajectoryBuffer(3, BufferConfig(game=FAKE_GAME_SPEC, device="cpu"))
    buffer.add(_complete_batch(buffer, 2, 3.0))
    state = buffer.state_dict()
    restored = TrajectoryBuffer(3, BufferConfig(game=FAKE_GAME_SPEC, device="cpu"))
    restored.load_state_dict(state)
    assert len(restored) == 2
    assert torch.equal(restored.storage["rewards"], buffer.storage["rewards"])
    sample = restored.sample(2, keys=("num_steps", "rewards"))
    assert sample.batch_size == torch.Size([2])
    assert set(sample.keys()) == {"num_steps", "rewards"}

    transition = TensorDict({key: value[:1, 0] for key, value in buffer.storage.items() if key != "num_steps"}, batch_size=[1])
    accumulator = TrajectoryBuffer(1, BufferConfig(game=FAKE_GAME_SPEC, device="cpu"))
    for _ in range(FAKE_GAME_SPEC.max_game_horizon):
        accumulator.add_step(torch.tensor([0]), transition)
    with pytest.raises(OverflowError, match="horizon"):
        accumulator.add_step(torch.tensor([0]), transition)


class _TinyTrainable(TrainableModule):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.embed = nn.Embedding(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


def test_optimizer_groups_and_round_trip() -> None:
    model = _TinyTrainable()
    optimizer = model.configure_optimizer(lr=1e-3, optimizer=OptimizerConfig(use_muon=False, weight_decay=0.1))
    assert "proj.weight" in model.weight_decay_param_names
    assert all("norm" not in name and "embed" not in name and not name.endswith("bias") for name in model.weight_decay_param_names)
    loss = model(torch.randn(3, 4)).square().mean()
    loss.backward()
    optimizer.step()
    restored = _TinyTrainable()
    restored_optimizer = restored.configure_optimizer(lr=1e-3, optimizer=OptimizerConfig(use_muon=False, weight_decay=0.1))
    restored_optimizer.load_state_dict(optimizer.state_dict())
    assert len(restored_optimizer.state_dict()["state"]) > 0

    muon_model = _TinyTrainable()
    muon_optimizer = muon_model.configure_optimizer(lr=1e-3, optimizer=OptimizerConfig(use_muon=True))
    assert set(muon_optimizer.state_dict()["optimizers"]) == {"muon", "adamw"}


def test_versioned_checkpoint_round_trip_model_optimizer_buffer(tmp_path: Path) -> None:
    cfg = fake_train_config()
    model = Agent(cfg, enable_value_head=True)
    optimizer = model.configure_optimizer(lr=1e-3, optimizer=cfg.optimizer)
    buffer = TrajectoryBuffer(2, BufferConfig(game=FAKE_GAME_SPEC, device="cpu"))
    buffer.add(_complete_batch(buffer, 2, 5.0))
    optimization_loss = sum(parameter.square().mean() for parameter in model.parameters())
    optimization_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer_state_size = len(optimizer.state_dict()["state"])
    assert optimizer_state_size > 0
    state = CheckpointState({"agent_state_dict": model, "optimizer": optimizer, "replay": buffer, "iteration": 7})
    path = CheckpointManager(checkpoint_dir=tmp_path, config_payload=cfg).save_from_state(state)
    payload = CheckpointStore.load_checkpoint_payload_file(path, device="cpu")
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["train_config"]["schema"] == "guandan-marvel-core-config-v1"

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    buffer.reset()
    state["iteration"] = 0
    CheckpointManager(checkpoint_dir=tmp_path).load_into_state(path, state, device="cpu")
    assert state.iteration == 7
    assert len(buffer) == 2
    assert len(optimizer.state_dict()["state"]) == optimizer_state_size
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())

    loaded = CheckpointManager.load_model(path, model_cls=Agent, device="cpu", model_kwargs={"enable_value_head": True})
    assert isinstance(loaded, Agent)
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_compile_switch_is_explicit_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = nn.Linear(2, 2)
    disabled = ModuleCompiler(TorchCompileSettings(enabled=False, backend="eager", dynamic_shapes=False))
    assert disabled.compile(module) is module

    calls = {}
    def fake_compile(model, **kwargs):
        calls.update(kwargs)
        return model
    monkeypatch.setattr(torch, "compile", fake_compile)
    enabled = ModuleCompiler(TorchCompileSettings(enabled=True, backend="eager", dynamic_shapes=True))
    assert enabled.compile(module) is module
    assert calls == {"backend": "eager", "dynamic": True}
    with pytest.raises(ValueError, match="backend"):
        ModuleCompiler(TorchCompileSettings(enabled=True, backend="", dynamic_shapes=False)).compile(module)
    monkeypatch.setattr(torch, "compile", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        ModuleCompiler(TorchCompileSettings(enabled=True, backend="eager", dynamic_shapes=False)).compile(module)


def test_compile_eager_backend_executes_when_available() -> None:
    module = nn.Linear(2, 2)
    compiled = ModuleCompiler(TorchCompileSettings(enabled=True, backend="eager", dynamic_shapes=False)).compile(module)
    result = compiled(torch.ones(1, 2))
    assert result.shape == (1, 2)


def test_ddp_requires_explicit_valid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)
    runtime = DistributedRuntime.from_config(DDPConfig(enabled=False), device="cpu")
    assert runtime.context.enabled is False

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="explicitly disabled"):
        DistributedRuntime.from_config(DDPConfig(enabled=False), device="cpu")
    with pytest.raises(RuntimeError, match="incomplete"):
        DistributedRuntime.from_config(DDPConfig(enabled=True), device="cpu")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29599")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        DistributedRuntime.from_config(DDPConfig(enabled=True), device="cpu")


def test_logging_and_schedule_utilities_are_versioned_and_deterministic(tmp_path: Path) -> None:
    jsonl = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"
    sinks = MetricSinks(jsonl_path=jsonl, csv_path=csv_path)
    sinks.log({"loss": torch.tensor(1.25), "shape": [2, 3]}, step=4)
    record = json.loads(jsonl.read_text(encoding="utf-8"))
    assert record == {"loss": 1.25, "schema": "guandan-marvel-metrics-v1", "shape": [2, 3], "step": 4}
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "step,key,value"

    schedule = compute_agent_schedule(fake_train_config(), iteration=1000)
    assert schedule.lr_scale == pytest.approx(0.5)
    assert schedule.agent_lr == pytest.approx(2e-4)
