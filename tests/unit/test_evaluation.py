from __future__ import annotations

import json

import pytest

from guandan.environment import GuandanEnv
from guandan.evaluation import (
    EvaluationFailure, EvaluationIncomplete, _assignment, evaluate_profile, play_game,
)
from guandan.training.runtime import BudgetExceeded, Deadline, resolved_profile
from guandan.model import PolicyValueNet
from guandan.training.checkpoint import save_checkpoint
from guandan.training.snapshot import export_policy_snapshot
import torch


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
        training_state={'engine': {'collector': {'deal_records': []}}},
    )
    return path


@pytest.mark.parametrize('rotation', [0, 1, 2, 3])
@pytest.mark.parametrize('opponent', ['random', 'rule', 'snapshot'])
def test_assignments_rotate_agents_not_the_underlying_deal(rotation, opponent):
    mapping, agents = _assignment(rotation, opponent)
    assert mapping == [(rotation + i) % 4 for i in range(4)]
    assert agents[rotation] == agents[(rotation + 2) % 4] == 'candidate'
    assert agents[(rotation + 1) % 4] == agents[(rotation + 3) % 4] == opponent


@pytest.mark.parametrize('rotation', [True, False, 4, -1, 0.5, '1'])
def test_rotation_validation(rotation):
    with pytest.raises(ValueError):
        _assignment(rotation, 'random')


def test_missing_model_explicit_failure_without_project_status(tmp_path):
    out = tmp_path / 'evidence/report.json'
    with pytest.raises(EvaluationFailure) as caught:
        evaluate_profile(candidate_checkpoint=tmp_path / 'missing.pt', snapshot_checkpoint=tmp_path / 'missing2.pt', evidence_path=out)
    assert caught.value.report['status'] == 'failed'
    assert caught.value.report['completed'] == 0
    assert not out.exists()
    assert json.loads(out.with_name('report_failure.json').read_text())['completed'] == 0


def test_expired_profile_budget_is_incomplete_before_loading_models(tmp_path):
    with pytest.raises(EvaluationIncomplete) as caught:
        evaluate_profile(max_hours=1e-12, evidence_path=tmp_path / 'out/report.json')
    assert caught.value.report['status'] == 'incomplete'
    assert caught.value.report['completed'] == 0


@pytest.mark.parametrize('hours', [0, -1, float('nan'), float('inf'), True])
def test_invalid_budget_never_scales_or_runs(hours):
    with pytest.raises(ValueError):
        evaluate_profile(max_hours=hours)


def test_batch_evaluation_preserves_counts_and_accepts_policy_artifact(checkpoint, tmp_path, monkeypatch):
    resolved = resolved_profile()
    resolved['config']['evaluation'] = {
        'deal_groups_per_pairing': 1, 'seat_rotations': [0],
        'opponents': ['random', 'rule', 'snapshot'], 'max_hours': 0.5,
    }
    monkeypatch.setattr('guandan.evaluation.resolved_profile', lambda profile='local_fast': resolved)
    policy_snapshot = tmp_path / 'opponent.snapshot.pt'
    export_policy_snapshot(checkpoint, policy_snapshot)
    result = evaluate_profile(
        candidate_checkpoint=checkpoint, snapshot_checkpoint=policy_snapshot,
        evidence_path=tmp_path / 'report.json', batch_size=2,
    )
    assert result.status == 'complete'
    assert result.total_games == 3
    assert result.completed_games_per_opponent == {'random': 1, 'rule': 1, 'snapshot': 1}
    assert result.metadata['batch_size'] == 2
    assert (tmp_path / 'report.json').exists()


def test_batch_size_validation():
    with pytest.raises(ValueError):
        evaluate_profile(batch_size=0)
    with pytest.raises(ValueError):
        evaluate_profile(batch_size=True)


def test_agent_error_bubbles_not_fallback_and_receives_observation_only(checkpoint):
    class Broken:
        metadata = {'kind': 'broken'}
        def select_token(self, observation):
            assert observation.private_hand_card_ids is None
            assert not hasattr(observation, 'round_state')
            raise RuntimeError('deliberate model failure')

    with pytest.raises(RuntimeError, match='deliberate model failure'):
        play_game(opponent='rule', deal_group=0, seat_rotation=0, seed=1000000,
                  candidate_checkpoint=checkpoint, snapshot_checkpoint=checkpoint,
                  agent_factory=lambda *a, **kw: Broken())


def test_deadline_checked_inside_game_not_only_between_games(checkpoint):
    class Tripwire:
        calls = 0
        def check(self):
            self.calls += 1
            if self.calls == 4:
                raise BudgetExceeded('elapsed mid-game')

    deadline = Tripwire()
    with pytest.raises(BudgetExceeded):
        play_game(opponent='rule', deal_group=0, seat_rotation=0, seed=1000000,
                  candidate_checkpoint=checkpoint, snapshot_checkpoint=checkpoint, deadline=deadline)
    assert deadline.calls == 4


def test_actual_step_error_bubbles_and_no_replacement_agent(checkpoint):
    class BadEnvironment(GuandanEnv):
        def step(self, token_id):
            raise RuntimeError('environment failure')

    with pytest.raises(RuntimeError, match='environment failure'):
        play_game(opponent='random', deal_group=0, seat_rotation=0, seed=1000000,
                  candidate_checkpoint=checkpoint, snapshot_checkpoint=checkpoint,
                  env_factory=BadEnvironment)
