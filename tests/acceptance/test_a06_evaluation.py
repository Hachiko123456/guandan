"""A06 acceptance uses ONE shared real 48-game run, not labels or fake records."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from guandan.evaluation import evaluate_profile, play_game, reviewed_checkpoints
from guandan.evaluation_stats import summarize_games
from guandan.training.runtime import EVALUATION_SEED_BASE, file_sha256, resolved_profile

pytestmark = pytest.mark.acceptance


@pytest.fixture(scope='module')
def actual_evaluation(tmp_path_factory):
    profile = resolved_profile()
    output = Path(os.environ['GUANDAN_EVIDENCE_DIR']) if 'GUANDAN_EVIDENCE_DIR' in os.environ else tmp_path_factory.mktemp('a06_real_profile')
    result = evaluate_profile(profile=profile['name'], evidence_path=output / 'evaluation_summary.json')
    return profile, result, output


@pytest.mark.parametrize('opponent', ['random','rule','snapshot'])
def test_each_pairing_completes_profile_games(actual_evaluation, opponent):
    profile, result, output = actual_evaluation
    cfg = profile['config']['evaluation']
    target = cfg['deal_groups_per_pairing'] * len(cfg['seat_rotations'])
    records = [record for record in result.records if record.opponent == opponent]
    assert len(records) == target == result.completed_games_per_opponent[opponent]
    assert result.status == 'complete'
    assert result.total_games == target*len(cfg['opponents'])
    assert {(r.deal_group,r.seat_rotation) for r in records} == {
        (group,rot) for group in range(cfg['deal_groups_per_pairing']) for rot in cfg['seat_rotations']}
    for r in records:
        assert r.done is True and r.truncated is False and r.terminal
        assert sorted(r.ranking) == [1,2,3,4]
        assert r.ranking.index(1)%2 == r.winner_team
        assert r.rewards == [1.0 if seat%2 == r.winner_team else -1.0 for seat in range(4)]
        assert sum(r.rewards) == 0
        assert r.team_rewards == [1.0 if team == r.winner_team else -1.0 for team in range(2)]
        assert sum(r.decisions_by_seat) == r.token_steps >= r.committed_steps > 0
        assert r.initial_leader == 0
        assert r.logical_to_physical == [(seat+r.seat_rotation)%4 for seat in range(4)]
        assert r.policies_by_seat[r.seat_rotation] == r.policies_by_seat[(r.seat_rotation+2)%4] == 'candidate'
        assert r.policies_by_seat[(r.seat_rotation+1)%4] == r.policies_by_seat[(r.seat_rotation+3)%4] == opponent
        saved = output / 'games' / f'{opponent}_{r.deal_group:04d}_{r.seat_rotation}.json'
        assert json.loads(saved.read_text()) == asdict(r)


def test_fixed_deals_are_not_redealt_or_co_rotated(actual_evaluation):
    profile, result, _ = actual_evaluation
    hashes = []
    for group in range(profile['config']['evaluation']['deal_groups_per_pairing']):
        rows = [r for r in result.records if r.deal_group == group]
        assert len({r.seed for r in rows}) == 1
        assert len({r.base_deal_sha256 for r in rows}) == 1
        # The primary candidate really observes all four DIFFERENT base hands;
        # co-rotating both hands and agents would incorrectly make this size 1.
        assert len({r.policy_hand_sha256 for r in rows}) == 4
        hashes.append(rows[0].base_deal_sha256)
    assert len(set(hashes)) == profile['config']['evaluation']['deal_groups_per_pairing']


def test_snapshot_is_actual_frozen_checkpoint_and_no_train_eval_overlap(actual_evaluation):
    _, result, _ = actual_evaluation
    paths = reviewed_checkpoints()
    assert result.metadata['candidate']['sha256'] == file_sha256(paths['candidate'])
    assert result.metadata['snapshot_opponent']['sha256'] == file_sha256(paths['snapshot'])
    assert result.metadata['candidate']['sha256'] != result.metadata['snapshot_opponent']['sha256']
    assert result.metadata['candidate']['update'] == 6
    assert result.metadata['snapshot_opponent']['update'] == 5
    for row in result.records:
        assert row.seed >= EVALUATION_SEED_BASE
        for seat, role in enumerate(row.policies_by_seat):
            agent = row.agent_metadata[seat]
            if role in {'candidate', 'snapshot'}:
                expected = result.metadata['candidate' if role == 'candidate' else 'snapshot_opponent']
                assert agent['sha256'] == expected['sha256']
                assert agent['kind'] == 'snapshot'
            else:
                assert agent['kind'] == role
    assert result.metadata['source_evidence']['a05_report_sha256']


def test_statistics_use_actual_completed_records_and_correlated_group_metadata(actual_evaluation):
    profile, result, output = actual_evaluation
    expected = summarize_games([asdict(r) for r in result.records])
    assert result.summary == expected
    assert expected['overall']['completed'] == result.total_games
    for opponent, summary in expected['per_opponent'].items():
        assert summary['wins']+summary['losses'] == summary['completed']
        assert summary['mean_team_reward'] == 2*summary['win_rate']-1
        for rotation in profile['config']['evaluation']['seat_rotations']:
            assert summary['by_rotation'][str(rotation)]['completed'] == profile['config']['evaluation']['deal_groups_per_pairing']
        assert summary['confidence']['cluster_adjusted'] is False
        assert summary['confidence']['independent_deal_groups'] == profile['config']['evaluation']['deal_groups_per_pairing']
        assert 'not independent' in summary['confidence']['warning']
    saved = json.loads((output/'evaluation_summary.json').read_text())
    assert saved['summary'] == result.summary
    assert saved['metadata']['resolved_profile'] == profile
    assert result.runtime_seconds < profile['config']['evaluation']['max_hours']*3600


def test_cross_process_replay_matches_real_game_despite_pythonhashseed(actual_evaluation, tmp_path):
    _, result, _ = actual_evaluation
    expected = next(r for r in result.records if r.opponent == 'random' and r.deal_group == 0 and r.seat_rotation == 0)
    sources = reviewed_checkpoints()
    destination = tmp_path/'replay.json'
    code = '''import json,sys,torch
from dataclasses import asdict
from pathlib import Path
from guandan.evaluation import play_game
torch.set_num_threads(1)
row=play_game(opponent='random',deal_group=0,seat_rotation=0,seed=int(sys.argv[1]),candidate_checkpoint=sys.argv[2],snapshot_checkpoint=sys.argv[3])
Path(sys.argv[4]).write_text(json.dumps(asdict(row)),encoding='utf8')
'''
    subprocess.run([sys.executable,'-B','-c',code,str(expected.seed),sources['candidate'],sources['snapshot'],str(destination)],
                   cwd=Path(__file__).resolve().parents[2],env={**os.environ,'PYTHONHASHSEED':'928471'},check=True,timeout=180)
    assert json.loads(destination.read_text()) == asdict(expected)
