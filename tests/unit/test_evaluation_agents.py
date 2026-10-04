from __future__ import annotations
from pathlib import Path
import pytest
from guandan.evaluation import _snapshot_path
from guandan.evaluation_agents import RandomAgent,RuleAgent,SnapshotAgent,make_agent
from guandan.environment import GameConfig,GuandanEnv


def obs():
    return GuandanEnv(GameConfig(seed=1,level_rank=5)).reset()

def test_random_and_rule_select_legal_observation_token():
    o=obs(); legal=set(o.legal_next_tokens[o.legal_next_mask].tolist())
    assert RandomAgent(7).select_token(o) in legal
    assert RuleAgent().select_token(o) in legal

def test_snapshot_requires_real_checkpoint_and_loads_actual_weights():
    agent=SnapshotAgent(_snapshot_path(),seed=1)
    o=obs(); assert agent.select_token(o) in set(o.legal_next_tokens[o.legal_next_mask].tolist())
    assert agent.metadata['sha256'] and agent.metadata['algorithm']=='ippo'

def test_agent_factory_rejects_unknown_and_missing_snapshot():
    with pytest.raises(ValueError): make_agent('not-an-agent')
    with pytest.raises(ValueError): make_agent('snapshot')

def test_snapshot_missing_checkpoint_is_explicit():
    with pytest.raises(Exception): SnapshotAgent(Path('does-not-exist.pt'))
