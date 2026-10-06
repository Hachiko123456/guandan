from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from guandan import evaluation_agents
from guandan.evaluation import play_game
from guandan.evaluation_agents import RandomAgent, RuleAgent, SnapshotAgent, clear_snapshot_cache, make_agent
from guandan.environment import GameConfig, GuandanEnv
from guandan.model import PolicyValueNet
from guandan.training.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from guandan.training.runtime import file_sha256
from guandan.training.snapshot import export_policy_snapshot, load_policy_snapshot


@pytest.fixture(autouse=True)
def isolated_cache():
    clear_snapshot_cache()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    clear_snapshot_cache()
    torch.set_num_threads(previous_threads)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(812)
        model = torch.nn.ModuleDict({'actor': PolicyValueNet(hidden_dim=8)})
    save_checkpoint(
        path, model, update=5, seed=10000, git_commit=None,
        model_config={
            'architecture': 'mean-pooled-token-MLP', 'hidden_dim': 8,
            'critic': 'decentralized-V-shared-teams', 'vocabulary': 256, 'channels': 256,
        },
        training_state={'engine': {'collector': {'deal_records': [
            {'seed': 10000, 'deal_sha256': 'training-only'},
        ]}}},
    )
    return path


def obs():
    return GuandanEnv(GameConfig(seed=1, level_rank=5)).reset()


def test_random_and_rule_select_legal_observation_token():
    observation = obs()
    legal = set(observation.legal_next_tokens[observation.legal_next_mask].tolist())
    assert RandomAgent(7).select_token(observation) in legal
    assert RuleAgent().select_token(observation) in legal


def test_snapshot_accepts_current_checkpoint_and_actual_weights(checkpoint):
    agent = SnapshotAgent(checkpoint, seed=1)
    observation = obs()
    assert agent.select_token(observation) in set(observation.legal_next_tokens[observation.legal_next_mask].tolist())
    assert agent.metadata['sha256'] == file_sha256(checkpoint)
    loaded = load_checkpoint(checkpoint, map_location='cpu')
    for name, weight in agent.model.state_dict().items():
        assert torch.equal(weight, loaded.model_state['actor.' + name])


def test_policy_snapshot_has_actor_only_and_is_accepted(checkpoint, tmp_path):
    path = tmp_path / 'policy.snapshot.pt'
    result = export_policy_snapshot(checkpoint, path)
    assert result['policy_snapshot_version'] == 1
    payload = torch.load(path, weights_only=True)
    assert 'optimizer_state' not in payload
    loaded = load_policy_snapshot(path)
    assert loaded.metadata['kind'] == 'policy_snapshot'
    agent = SnapshotAgent(path, seed=3)
    assert agent.select_token(obs()) in set(obs().legal_next_tokens[obs().legal_next_mask].tolist())


def test_agent_factory_rejects_unknown_and_missing_snapshot():
    with pytest.raises(ValueError):
        make_agent('not-an-agent')
    with pytest.raises(ValueError):
        make_agent('snapshot')


def test_snapshot_missing_checkpoint_is_explicit(tmp_path):
    with pytest.raises(CheckpointError):
        SnapshotAgent(tmp_path / 'does-not-exist.pt')


def test_checkpoint_load_and_model_construction_happen_once(checkpoint, monkeypatch):
    loads = Mock(wraps=evaluation_agents.load_policy_snapshot)
    construct = Mock(wraps=PolicyValueNet)
    monkeypatch.setattr(evaluation_agents, 'load_policy_snapshot', loads)
    monkeypatch.setattr(evaluation_agents, 'PolicyValueNet', construct)
    first = SnapshotAgent(checkpoint, seed=7)
    second = make_agent('snapshot', checkpoint=checkpoint.resolve(), seed=19)
    greedy = SnapshotAgent(checkpoint, seed=23, device='cpu:0', deterministic=True)
    assert first is not second and first.model is second.model is greedy.model
    assert loads.call_count == construct.call_count == 1
    assert first.device == torch.device('cpu')
    assert greedy.device == torch.device('cpu:0')
    assert first.metadata['seed'] == 7 and second.metadata['seed'] == 19
    assert greedy.metadata['deterministic'] is greedy.deterministic is True
    assert not first.model.training
    assert all(not parameter.requires_grad for parameter in first.model.parameters())


def test_concurrent_wrappers_only_load_and_construct_once(checkpoint, monkeypatch):
    loads = Mock(wraps=evaluation_agents.load_policy_snapshot)
    construct = Mock(wraps=PolicyValueNet)
    monkeypatch.setattr(evaluation_agents, 'load_policy_snapshot', loads)
    monkeypatch.setattr(evaluation_agents, 'PolicyValueNet', construct)
    with ThreadPoolExecutor(max_workers=4) as pool:
        agents = list(pool.map(lambda seed: SnapshotAgent(checkpoint, seed=seed), range(8)))
    assert loads.call_count == construct.call_count == 1
    assert all(agent.model is agents[0].model for agent in agents)
    assert len({id(agent.generator) for agent in agents}) == 8


def test_batch_forward_and_rng_wrappers_are_independent(checkpoint):
    first = SnapshotAgent(checkpoint, seed=7)
    second = SnapshotAgent(checkpoint, seed=19)
    observations = [obs(), GuandanEnv(GameConfig(seed=2, level_rank=5)).reset()]
    before = second.generator.get_state().clone()
    batch = SnapshotAgent.select_tokens_batch([(first, observations[0]), (second, observations[1])])
    assert len(batch) == 2
    assert all(token in set(observation.legal_next_tokens[observation.legal_next_mask].tolist()) for token, observation in zip(batch, observations))
    assert not torch.equal(before, second.generator.get_state())
    replay = SnapshotAgent(checkpoint, seed=7)
    assert replay.model is first.model


def test_clear_forces_reload_but_existing_agents_remain_usable(checkpoint, monkeypatch):
    loads = Mock(wraps=evaluation_agents.load_policy_snapshot)
    monkeypatch.setattr(evaluation_agents, 'load_policy_snapshot', loads)
    old = SnapshotAgent(checkpoint, seed=7)
    state = old.generator.get_state().clone()
    clear_snapshot_cache()
    fresh = SnapshotAgent(checkpoint, seed=7)
    assert loads.call_count == 2
    assert old.model is not fresh.model
    assert torch.equal(state, old.generator.get_state())
    assert torch.equal(old.logits(obs()), fresh.logits(obs()))


@pytest.mark.parametrize('rotation', [0, 1, 2, 3])
def test_play_game_baseline_preserves_complete_trace_and_seat_seed_semantics(checkpoint, rotation):
    options = dict(
        opponent='snapshot', deal_group=2, seat_rotation=rotation, seed=1000002,
        candidate_checkpoint=checkpoint, snapshot_checkpoint=checkpoint,
    )
    row = play_game(**options)
    assert row.done and not row.truncated
    assert row.logical_to_physical == [(seat + rotation) % 4 for seat in range(4)]
    for logical, physical in enumerate(row.logical_to_physical):
        expected_seed = 22_002_000 + rotation * 10 + logical
        assert row.policy_seeds_by_seat[physical] == expected_seed
        assert row.agent_metadata[physical]['seed'] == expected_seed
    assert row.token_steps > 0 and row.committed_steps > 0


def test_play_game_rejects_incomplete_environment_and_agent_errors(checkpoint):
    class Broken:
        metadata = {'kind': 'broken'}
        def select_token(self, observation):
            assert observation.private_hand_card_ids is None
            raise RuntimeError('deliberate model failure')

    with pytest.raises(RuntimeError, match='deliberate model failure'):
        play_game(opponent='rule', deal_group=0, seat_rotation=0, seed=1000000,
                  candidate_checkpoint=checkpoint, snapshot_checkpoint=checkpoint,
                  agent_factory=lambda *a, **kw: Broken())

    class BadEnvironment(GuandanEnv):
        def step(self, token_id):
            raise RuntimeError('environment failure')

    with pytest.raises(RuntimeError, match='environment failure'):
        play_game(opponent='random', deal_group=0, seat_rotation=0, seed=1000000,
                  candidate_checkpoint=checkpoint, snapshot_checkpoint=checkpoint,
                  env_factory=BadEnvironment)
