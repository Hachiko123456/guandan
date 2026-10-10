from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import guandan_marvel.evaluation.evaluator as evaluator_module
from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent, load_v2_policy
from guandan_marvel.evaluation.evaluator import EvaluationRunner
from guandan_marvel.evaluation.manifests import build_deals_manifest
from guandan_marvel.evaluation.pool import OpponentPoolStore
from guandan_marvel.evaluation.promotion import ChampionStore, PromotionPolicy


def _record(**kw):
    deal=kw['deal'];rotation=kw['rotation'];candidate=kw['candidate'];opponent=kw.get('opponent_v2');legacy=kw.get('legacy');team=rotation%2
    return {'schema':'guandan-marvel-evaluation-game-v1','status':'complete','opponent_id':kw['opponent_id'],'opponent_kind':kw['opponent_kind'],'deal_id':deal['deal_id'],'deal_seed':deal['seed'],'level_rank':deal['level_rank'],'leader_seat':deal['leader_seat'],'base_deal_sha256':hashlib.sha256(deal['deal_id'].encode()).hexdigest(),'seat_rotation':rotation,'logical_to_physical':[(i+rotation)%4 for i in range(4)],'policies_by_seat':['candidate','x','candidate','x'],'candidate_team':team,'winner_team':team,'candidate_win':True,'ranking':[1,2,3,4] if team==0 else [2,1,4,3],'outcome_class':'HEAD_LAST','rewards':[1.0,-1.0,1.0,-1.0] if team==0 else [-1.0,1.0,-1.0,1.0],'candidate_team_reward':1.0,'token_steps':10,'committed_steps':4,'engine_finish_token_step':10,'engine_finish_committed_step':4,'decisions_by_seat':[1,1,1,1],'illegal_count':0,'truncated_count':0,'runtime_seconds':0.001,'actions_per_second':10000.0,'action_trace_sha256':'a'*64,'candidate_checkpoint_sha256':candidate.sha256,'opponent_checkpoint_sha256':opponent.sha256 if opponent else (legacy.sha256 if legacy else None),'temperature':0.0,'backend':'cpp','protocol_version':candidate.protocol_version}


def test_path_cache_detects_checkpoint_replacement(v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,tmp_path):
    a=load_v2_policy(v2_checkpoint);b=load_v2_policy(second_v2_checkpoint);legacy=load_legacy_v1_opponent(legacy_checkpoint)
    store=ChampionStore(tmp_path/'champ.json');store.bootstrap(b.identity)
    pool=OpponentPoolStore(tmp_path/'pool.json');pool.initialize();pool.add_entry({'id':'b','kind':'champion','checkpoint':str(b.path),'policy_identity':b.identity,'weight':1.0})
    runner=EvaluationRunner(output_dir=tmp_path/'out',champion_store=store,opponent_pool=pool,legacy_checkpoint=legacy.path)
    first=runner._load_v2(v2_checkpoint);Path(v2_checkpoint).write_bytes(Path(second_v2_checkpoint).read_bytes());second=runner._load_v2(v2_checkpoint)
    assert first.sha256!=second.sha256 and second.sha256==load_v2_policy(v2_checkpoint).sha256


def test_pool_failure_leaves_identifiable_pending_transaction_not_success(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,monkeypatch):
    candidate=load_v2_policy(v2_checkpoint);champion=load_v2_policy(second_v2_checkpoint);legacy=load_legacy_v1_opponent(legacy_checkpoint)
    store=ChampionStore(tmp_path/'champ.json');store.bootstrap(champion.identity)
    pool=OpponentPoolStore(tmp_path/'pool.json');pool.initialize();pool.add_entry({'id':'old','kind':'champion','checkpoint':str(champion.path),'policy_identity':champion.identity,'weight':1.0})
    runner=EvaluationRunner(output_dir=tmp_path/'run',champion_store=store,opponent_pool=pool,legacy_checkpoint=legacy.path,promotion_policy=PromotionPolicy(),provenance_policy='formal')
    manifest=build_deals_manifest(counts={'training':1,'selection':32,'final':1},bases={'training':100,'selection':1000,'final':2000},level_ranks=(5,),leader_seats=(0,))
    monkeypatch.setattr(evaluator_module,'play_game',_record)
    monkeypatch.setattr(pool,'add_entry',lambda *a,**k: (_ for _ in ()).throw(OSError('pool publish failed')))
    result=runner.run_selection(deals_manifest=manifest,candidate_checkpoint=candidate.path)
    assert result['promotion_decision']['status']=='promote'
    assert result['status']=='partial_failure' and 'pool publish failed' in result['error']
    assert result['promotion_transaction']['state']=='partial_failure'
    repaired=store.load()
    assert repaired['champion']['checkpoint_sha256']==candidate.sha256
    assert repaired['pool_sync']['status']=='pending'
    assert pool.load()['revision']==1


def test_overlap_rejection_happens_before_any_game_or_inference_callback(tmp_path,old_v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,monkeypatch):
    candidate=load_v2_policy(old_v2_checkpoint);champion=load_v2_policy(second_v2_checkpoint);legacy=load_legacy_v1_opponent(legacy_checkpoint)
    store=ChampionStore(tmp_path/'champ.json');store.bootstrap(champion.identity)
    pool=OpponentPoolStore(tmp_path/'pool.json');pool.initialize();pool.add_entry({'id':'old','kind':'champion','checkpoint':str(champion.path),'policy_identity':champion.identity,'weight':1.0})
    runner=EvaluationRunner(output_dir=tmp_path/'run',champion_store=store,opponent_pool=pool,legacy_checkpoint=legacy.path,promotion_policy=PromotionPolicy(mode='smoke',min_clusters=1),provenance_policy='smoke_only')
    seed=int(candidate.metadata['training_seed']);manifest=build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':100,'selection':seed,'final':200},level_ranks=(5,),leader_seats=(0,))
    calls={'n':0}
    def should_not_run(**kw):calls['n']+=1;return _record(**kw)
    monkeypatch.setattr(evaluator_module,'play_game',should_not_run)
    with pytest.raises(ValueError,match='known checkpoint training seed'):
        runner.run_selection(deals_manifest=manifest,candidate_checkpoint=candidate.path)
    assert calls['n']==0 and not (tmp_path/'run/games.jsonl').exists()
