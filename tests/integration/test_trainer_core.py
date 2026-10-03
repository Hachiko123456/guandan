"""Real profile-sized low-level engine evidence; never a reduced-hand smoke substitute."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
import torch

from guandan.cards import Rank
from guandan.env_batch import GuandanEnvBatch
from guandan.environment_types import GameConfig
from guandan.model import PolicyValueNet
from guandan.training.collector import Deadline
from guandan.training.config import load_profile
from guandan.training.critic import CentralQCritic
from guandan.training import estimators, objectives
from guandan.training.runtime import BudgetExceeded, TRAIN_SEED_BASE
from guandan.training.trainer_core import TrainingEngine, TrainingEngineError


@pytest.fixture(autouse=True)
def single_torch_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def engine(algorithm="ippo", *, size=4, epochs=2, provider=None):
    torch.manual_seed(TRAIN_SEED_BASE)
    np.random.seed(TRAIN_SEED_BASE)
    batch = GuandanEnvBatch([GameConfig(level_rank=Rank.FIVE, seed=TRAIN_SEED_BASE + i) for i in range(size)])
    batch.reset(seeds=[TRAIN_SEED_BASE + i for i in range(size)])
    actor = PolicyValueNet(hidden_dim=16)
    critic = CentralQCritic(hidden_dim=16) if algorithm == "vrpo" else None
    parameters = list(actor.parameters()) + (list(critic.parameters()) if critic is not None else [])
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    return TrainingEngine(algorithm, batch, actor, optimizer, critic=critic, seed=TRAIN_SEED_BASE,
                          epochs=epochs, objective_provider=provider)


@pytest.mark.parametrize("algorithm", ["ippo", "vrpo"])
def test_local_fast_five_full_deal_updates_and_exactly_one_loaded_resume(algorithm):
    profile = load_profile("local_fast")
    assert profile.rollout_envs == 4
    assert profile.token_steps_per_update == 256
    assert profile.updates_per_algorithm == profile.checkpoint_interval == 5
    assert profile.resume_updates == 1
    deadline = Deadline(profile.max_hours)
    first = engine(algorithm)
    for env in first.envs.envs:
        hands = env.full_state()["round_state"]["hands"]
        assert [len(hand) for hand in hands] == [27] * 4
        assert sorted(card for hand in hands for card in hand) == list(range(108))
        assert env.config.level_rank == Rank.FIVE and env.config.previous_result is None
    digests = []
    for update in range(1, profile.updates_per_algorithm + 1):
        metrics = first.collect_and_update(profile.token_steps_per_update, deadline=deadline)
        assert metrics["status"] == "complete"
        assert metrics["update_count"] == update
        assert metrics["total_token_steps"] == update * 256
        assert metrics["token_steps"] == 256
        assert metrics["per_env_ticks"] == [64] * 4
        assert metrics["lifetime_ticks_per_env"] == [64 * update] * 4
        assert metrics["optimizer_steps"] == 2 * update
        assert np.isfinite(metrics["loss"]) and np.isfinite(metrics["gradient_norm"])
        assert metrics["target_kind"] == ("gae" if algorithm == "ippo" else "q_boost")
        assert all(record["hand_sizes"] == [27] * 4 for record in metrics["deal_records"])
        assert isinstance(metrics["terminal_snapshots"], list)
        digests.append(metrics["rollout_sha256"])
    assert first.total_token_steps == 1280 and first.update_count == 5
    assert len(set(digests)) == 5
    # Actual serialization/load of the whole engine at update5, not metadata
    # spoofing. BytesIO avoids writing artifacts outside this worker's scope.
    memory = BytesIO()
    torch.save(first.state_dict(), memory)
    memory.seek(0)
    loaded = torch.load(memory, weights_only=True)
    second = engine(algorithm)
    second.load_state_dict(loaded)
    assert second.update_count == 5 and second.total_token_steps == 1280
    assert second.envs.serialize() == first.envs.serialize()
    for left, right in zip(first.actor.parameters(), second.actor.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    metrics = second.collect_and_update(256, deadline=deadline)
    assert metrics["update_count"] == 6
    assert metrics["total_token_steps"] == 1536
    assert metrics["token_steps"] == 256
    assert metrics["lifetime_ticks_per_env"] == [384] * 4
    assert second.optimizer_steps == 12
    assert first.update_count == 5 and first.total_token_steps == 1280
    assert metrics["rollout_sha256"] not in digests
    assert np.isfinite(metrics["loss"]) and np.isfinite(metrics["gradient_norm"])


@pytest.mark.parametrize("algorithm", ["ippo", "vrpo"])
def test_public_estimator_objective_paths_and_no_duplicate_update(algorithm):
    calls = []
    provider = {}
    for name, module in (("compute_gae", estimators), ("compute_q_boost", estimators),
                         ("ippo_loss", objectives), ("vrpo_loss", objectives)):
        def spy(*args, _name=name, _fn=getattr(module, name), **kwargs):
            calls.append((_name, args[0].shape))
            return _fn(*args, **kwargs)
        provider[name] = spy
    value = engine(algorithm, size=1, epochs=1, provider=provider)
    before = deepcopy(value.actor.state_dict())
    q_before = deepcopy(value.critic.state_dict()) if value.critic is not None else None
    rollout = value.collect(4)
    with pytest.raises(TrainingEngineError, match="pending"):
        value.collect(4)
    metrics = value.update(rollout)
    with pytest.raises(TrainingEngineError, match="consumed"):
        value.update(rollout)
    assert value.update_count == 1 and value.total_token_steps == 4 and value.optimizer_steps == 1
    assert metrics["token_steps"] == 4
    expected_estimator = "compute_gae" if algorithm == "ippo" else "compute_q_boost"
    assert (expected_estimator, torch.Size([4, 1, 4])) in calls
    expected_objective = "ippo_loss" if algorithm == "ippo" else "vrpo_loss"
    assert sum(name == expected_objective for name, shape in calls) == 4
    assert not any(name == ("vrpo_loss" if algorithm == "ippo" else "ippo_loss") for name, _ in calls)
    assert any(not torch.equal(before[key], tensor) for key, tensor in value.actor.state_dict().items())
    if q_before is not None:
        assert any(not torch.equal(q_before[key], tensor) for key, tensor in value.critic.state_dict().items())


def test_restore_is_weights_only_safe_and_rejects_hyperparameters_before_loading():
    first = engine(size=1, epochs=1)
    first.collect_and_update(4)
    snapshot = first.state_dict()
    assert snapshot["extra_state"]["collector"]["ticks_per_env"] == [4]
    assert isinstance(snapshot["extra_state"]["numpy_rng_state"]["keys"], list)
    memory = BytesIO()
    torch.save(snapshot, memory)
    memory.seek(0)
    loaded = torch.load(memory, weights_only=True)
    second = engine(size=1, epochs=1)
    second.load_state_dict(loaded)
    expected = first.collect_and_update(4)
    actual = second.collect_and_update(4)
    assert expected["rollout_sha256"] == actual["rollout_sha256"]
    assert expected["loss"] == actual["loss"]
    for a, b in zip(first.actor.parameters(), second.actor.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key, value in (("gamma", 0.5), ("gae_lambda", 0.2), ("epochs", 9), ("max_grad_norm", 2)):
        incompatible = deepcopy(loaded)
        incompatible["extra_state"]["settings"][key] = value
        original = deepcopy(second.actor.state_dict())
        with pytest.raises(TrainingEngineError, match="settings"):
            second.load_state_dict(incompatible)
        for name, tensor in second.actor.state_dict().items():
            torch.testing.assert_close(tensor, original[name], atol=0, rtol=0)


def test_expired_budget_performs_no_update_and_no_counter_scaling():
    value = engine(size=1, epochs=1)
    deadline = Deadline(1, clock=lambda: 4000, started_at=0)
    with pytest.raises(BudgetExceeded):
        value.collect_and_update(256, deadline=deadline)
    assert value.update_count == value.optimizer_steps == value.total_token_steps == 0


def test_vrpo_requires_real_critic_and_optimizer_coverage():
    value = engine(size=1, epochs=1)
    with pytest.raises(TrainingEngineError, match="critic"):
        TrainingEngine("vrpo", value.envs, value.actor, value.optimizer)
    critic = CentralQCritic(hidden_dim=8)
    with pytest.raises(TrainingEngineError, match="optimizer"):
        TrainingEngine("vrpo", value.envs, value.actor, value.optimizer, critic=critic)
