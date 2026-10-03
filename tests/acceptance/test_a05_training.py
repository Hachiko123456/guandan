"""A05 real profile gate: no one-step smoke can satisfy this module."""
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest
import torch

from guandan.training.checkpoint import load_checkpoint, IncompatibleCheckpointError
from guandan.training.runtime import Deadline, TRAIN_SEED_BASE, resolved_profile, write_evidence, file_sha256
from guandan.training.trainer import train


@pytest.fixture(scope='module')
def completed_training(tmp_path_factory):
    config = resolved_profile()
    settings = config['config']['training']
    directory = Path(os.environ['GUANDAN_EVIDENCE_DIR']) if 'GUANDAN_EVIDENCE_DIR' in os.environ else tmp_path_factory.mktemp('a05_real_profile')
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for index, algorithm in enumerate(config['config']['algorithms']):
        budget = Deadline(settings['max_hours'])
        seed = TRAIN_SEED_BASE + index * 50_000
        first = train(algorithm=algorithm, profile=config['name'], seed=seed,
                      checkpoint_dir=directory, hidden_dim=32, deadline=budget)
        assert first.status == 'complete', asdict(first)
        assert first.updates == settings['updates_per_algorithm']
        assert first.total_token_steps == settings['updates_per_algorithm'] * settings['token_steps_per_update']
        assert first.profile_counts_match
        assert first.checkpoint is not None and Path(first.checkpoint).is_file()
        resumed = train(algorithm=algorithm, profile=config['name'], seed=seed,
                        checkpoint_dir=directory, hidden_dim=32, resume=first.checkpoint, deadline=budget)
        assert resumed.status == 'complete', asdict(resumed)
        results[algorithm] = {'base': asdict(first), 'resume': asdict(resumed)}
    summary = {'scope': 'local-training-pipeline-not-model-strength', 'status': 'complete',
               'profile': config, 'algorithms': results, 'accepted': False}
    write_evidence(directory / 'training_counts.json', summary)
    return config, results


@pytest.mark.parametrize('algorithm', ['ippo', 'vrpo'])
def test_actual_base_and_resume_profile_counts(completed_training, algorithm):
    profile, runs = completed_training
    settings = profile['config']['training']
    base, resume = runs[algorithm]['base'], runs[algorithm]['resume']
    assert base['updates'] == settings['updates_per_algorithm']
    assert base['total_token_steps'] == settings['updates_per_algorithm'] * settings['token_steps_per_update']
    assert resume['updates'] == settings['resume_updates']
    assert resume['total_token_steps'] == settings['resume_updates'] * settings['token_steps_per_update']
    assert resume['lifetime_updates'] == base['updates'] + resume['updates']
    assert resume['lifetime_token_steps'] == base['total_token_steps'] + resume['total_token_steps']
    assert resume['start_update'] == base['updates']
    assert resume['resumed_from'] == base['checkpoint']
    assert resume['resume_sha256'] == file_sha256(base['checkpoint'])
    for execution in (base, resume):
        assert execution['profile_counts_match']
        assert len(execution['metrics']) == execution['updates']
        for update in execution['metrics']:
            assert update['token_steps'] == settings['token_steps_per_update']
            assert update['rollout_envs'] == settings['rollout_envs']
            assert update['per_env_ticks'] == [settings['token_steps_per_update'] // settings['rollout_envs']] * settings['rollout_envs']
            assert len(update['rollout_sha256']) == 64
            assert update['target_kind'] == ('gae' if algorithm == 'ippo' else 'q_boost')
            assert torch.isfinite(torch.tensor(update['gradient_norm']))
            assert torch.isfinite(torch.tensor(update['loss']))
        assert Path(execution['evidence']).is_file()
        assert execution['runtime']['model_architecture'].startswith('mean-pooled-token-MLP')


@pytest.mark.parametrize('algorithm', ['ippo', 'vrpo'])
def test_saved_resume_state_versions_optimizer_and_counters(completed_training, algorithm):
    _, runs = completed_training
    base = load_checkpoint(runs[algorithm]['base']['checkpoint'], expected_algorithm=algorithm)
    resumed = load_checkpoint(runs[algorithm]['resume']['checkpoint'], expected_algorithm=algorithm)
    for checkpoint in (base, resumed):
        assert checkpoint.optimizer_state
        state = checkpoint.training_state['engine']
        assert state['update_count'] == checkpoint.metadata.update_count
        assert state['collector']['total_token_steps'] > 0
        assert len(state['collector']['envs']) == 4
        assert state['torch_rng_state'].dtype == torch.uint8
        assert checkpoint.model_config['architecture'] == 'mean-pooled-token-MLP'
    assert any(not torch.equal(base.model_state[key], resumed.model_state[key]) for key in base.model_state)
    opposite = 'vrpo' if algorithm == 'ippo' else 'ippo'
    with pytest.raises(IncompatibleCheckpointError):
        load_checkpoint(runs[algorithm]['base']['checkpoint'], expected_algorithm=opposite)


def test_distinct_vrpo_and_ippo_critic_objective(completed_training):
    _, runs = completed_training
    ippo = load_checkpoint(runs['ippo']['base']['checkpoint'])
    vrpo = load_checkpoint(runs['vrpo']['base']['checkpoint'])
    assert not any(key.startswith('critic.') for key in ippo.model_state)
    assert any(key.startswith('critic.') for key in vrpo.model_state)
    assert ippo.model_config['critic'] != vrpo.model_config['critic']
    assert len(runs['ippo']['base']['metrics']) == len(runs['vrpo']['base']['metrics']) == 5
