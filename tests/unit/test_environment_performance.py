import pytest
from guandan.environment import GuandanEnv,GameConfig
from guandan.cards import Rank
from guandan.state import RoundState


def test_action_cache_does_not_serialize_history(monkeypatch):
    env=GuandanEnv(GameConfig(seed=1,level_rank=Rank.FIVE))
    obs=env.reset()
    expected=obs.legal_next_tokens.copy()
    def forbidden(*a,**kw):raise AssertionError('serialize not a cache key')
    monkeypatch.setattr(RoundState,'serialize',forbidden)
    assert (env.observe().legal_next_tokens==expected).all()


def test_candidate_cache_refresh_after_actor_or_hand_mutation():
    env=GuandanEnv(GameConfig(seed=52,level_rank=Rank.FIVE));env.reset()
    p=env._protocol
    for seat in (1,3,2,0):
        p.round_state.active_seat=seat
        actual=p._candidate_sequences()
        fresh=p.clone();fresh.__dict__.pop('_cache_key',None);fresh.__dict__.pop('_cache_paths',None)
        assert actual==fresh._candidate_sequences()
    p.round_state.hands[0],p.round_state.hands[2]=p.round_state.hands[2],p.round_state.hands[0]
    actual=p._candidate_sequences()
    fresh=p.clone();fresh.__dict__.pop('_cache_key',None);fresh.__dict__.pop('_cache_paths',None)
    assert actual==fresh._candidate_sequences()
