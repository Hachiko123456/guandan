"""A06 failure paths, real per-token budgets, canonical config and safe agents."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from guandan.evaluation import (EvaluationFailure, EvaluationIncomplete, _assignment,
                                evaluate_profile, play_game, reviewed_checkpoints)
from guandan.evaluation_agents import make_agent
from guandan.training.runtime import BudgetExceeded, Deadline, resolved_profile


@pytest.mark.parametrize('rotation', [0,1,2,3])
@pytest.mark.parametrize('opponent', ['random','rule','snapshot'])
def test_assignments_rotate_agents_not_the_underlying_deal(rotation, opponent):
    mapping, agents = _assignment(rotation,opponent)
    assert mapping == [(rotation+i)%4 for i in range(4)]
    assert agents[rotation] == agents[(rotation+2)%4] == 'candidate'
    assert agents[(rotation+1)%4] == agents[(rotation+3)%4] == opponent


@pytest.mark.parametrize('rotation', [True, False, 4, -1, 0.5, '1'])
def test_rotation_validation(rotation):
    with pytest.raises(ValueError): _assignment(rotation,'random')


def test_missing_model_explicit_failure_without_any_completed_games(tmp_path):
    out = tmp_path/'evidence/report.json'
    with pytest.raises(EvaluationFailure) as caught:
        evaluate_profile(candidate_checkpoint=tmp_path/'missing.pt',snapshot_checkpoint=tmp_path/'missing2.pt',evidence_path=out)
    assert caught.value.report['status'] == 'failed'
    assert caught.value.report['completed'] == 0
    assert not out.exists()
    assert json.loads(out.with_name('report_failure.json').read_text())['completed'] == 0


def test_expired_profile_budget_is_incomplete_before_loading_models(tmp_path):
    with pytest.raises(EvaluationIncomplete) as caught:
        evaluate_profile(max_hours=1e-12,evidence_path=tmp_path/'out/report.json')
    assert caught.value.report['status'] == 'incomplete'
    assert caught.value.report['completed'] == 0


@pytest.mark.parametrize('hours', [0, -1, float('nan'), float('inf'), True])
def test_invalid_budget_never_scales_or_runs(hours):
    with pytest.raises(ValueError): evaluate_profile(max_hours=hours)


def test_canonical_runner_profile_mismatch_fails(monkeypatch):
    wrong=resolved_profile()
    wrong['config']['evaluation']['deal_groups_per_pairing']=1
    monkeypatch.setenv('GUANDAN_PROFILE','local_fast')
    monkeypatch.setenv('GUANDAN_RESOLVED_PROFILE_JSON',json.dumps(wrong))
    with pytest.raises(ValueError): evaluate_profile()


def test_unknown_opponent_and_seed_no_fallback():
    with pytest.raises(ValueError): _assignment(0,'typo')
    with pytest.raises(ValueError):
        play_game(opponent='random',deal_group=0,seat_rotation=0,seed=10000,
                  candidate_checkpoint='no.pt',snapshot_checkpoint='no2.pt')


def test_agent_error_bubbles_not_fallback_and_receives_observation_only():
    sources=reviewed_checkpoints()
    class Broken:
        metadata={'kind':'broken'}
        def select_token(self, obs):
            assert obs.private_hand_card_ids is None
            assert not hasattr(obs,'round_state')
            raise RuntimeError('deliberate model failure')
    with pytest.raises(RuntimeError,match='deliberate model failure'):
        play_game(opponent='rule',deal_group=0,seat_rotation=0,seed=1000000,
                  candidate_checkpoint=sources['candidate'],snapshot_checkpoint=sources['snapshot'],
                  agent_factory=lambda *a, **kw:Broken())


def test_deadline_checked_inside_game_not_only_between_games():
    sources=reviewed_checkpoints()
    class Tripwire:
        calls=0
        def check(self):
            self.calls+=1
            if self.calls==4: raise BudgetExceeded('elapsed mid-game')
    deadline=Tripwire()
    with pytest.raises(BudgetExceeded):
        play_game(opponent='rule',deal_group=0,seat_rotation=0,seed=1000000,
                  candidate_checkpoint=sources['candidate'],snapshot_checkpoint=sources['snapshot'],deadline=deadline)
    assert deadline.calls==4


def test_actual_step_error_bubbles_and_no_replacement_agent():
    from guandan.environment import GuandanEnv
    sources=reviewed_checkpoints()
    class BadEnvironment(GuandanEnv):
        def step(self, token_id): raise RuntimeError('environment failure')
    with pytest.raises(RuntimeError,match='environment failure'):
        play_game(opponent='random',deal_group=0,seat_rotation=0,seed=1000000,
                  candidate_checkpoint=sources['candidate'],snapshot_checkpoint=sources['snapshot'],env_factory=BadEnvironment)
