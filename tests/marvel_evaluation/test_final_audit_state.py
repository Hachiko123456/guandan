from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Barrier, Thread

import pytest

import guandan_marvel.evaluation.evaluator as evaluator_module
from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent, load_v2_policy
from guandan_marvel.evaluation.evaluator import EvaluationRunner
from guandan_marvel.evaluation.manifests import FinalHoldoutAudit, build_deals_manifest, dataset_identity
from guandan_marvel.evaluation.pool import OpponentPoolStore
from guandan_marvel.evaluation.promotion import ChampionStore, PromotionPolicy
from guandan_marvel.evaluation.common import payload_hash


def _manifest():
    return build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':100,'selection':200,'final':300},level_ranks=(5,),leader_seats=(0,))


def _setup(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint):
    candidate=load_v2_policy(v2_checkpoint);champion=load_v2_policy(second_v2_checkpoint);legacy=load_legacy_v1_opponent(legacy_checkpoint)
    store=ChampionStore(tmp_path/'champion.json');store.bootstrap(champion.identity)
    pool=OpponentPoolStore(tmp_path/'pool.json');pool.initialize();pool.add_entry({'id':'champ','kind':'champion','checkpoint':str(champion.path),'policy_identity':champion.identity,'weight':1.0})
    runner=EvaluationRunner(output_dir=tmp_path/'final-run',champion_store=store,opponent_pool=pool,legacy_checkpoint=legacy.path,promotion_policy=PromotionPolicy(),provenance_policy='formal')
    return candidate,champion,legacy,runner


def _fake_record(**kw):
    deal=kw['deal'];rotation=kw['rotation'];candidate=kw['candidate'];opponent=kw.get('opponent_v2');legacy=kw.get('legacy')
    candidate_team=rotation%2;winner=candidate_team
    digest=hashlib.sha256(str(deal['deal_id']).encode()).hexdigest()
    return {'schema':'guandan-marvel-evaluation-game-v1','status':'complete','opponent_id':kw['opponent_id'],'opponent_kind':kw['opponent_kind'],'deal_id':deal['deal_id'],'deal_seed':deal['seed'],'level_rank':deal['level_rank'],'leader_seat':deal['leader_seat'],'base_deal_sha256':digest,'seat_rotation':rotation,'logical_to_physical':[(i+rotation)%4 for i in range(4)],'policies_by_seat':['candidate','x','candidate','x'],'candidate_team':candidate_team,'winner_team':winner,'candidate_win':True,'ranking':[1,2,3,4] if winner==0 else [2,1,4,3],'outcome_class':'HEAD_LAST','rewards':[1.0,-1.0,1.0,-1.0] if winner==0 else [-1.0,1.0,-1.0,1.0],'candidate_team_reward':1.0,'token_steps':10,'committed_steps':4,'engine_finish_token_step':10,'engine_finish_committed_step':4,'decisions_by_seat':[1,1,1,1],'illegal_count':0,'truncated_count':0,'runtime_seconds':0.01,'actions_per_second':1000.0,'action_trace_sha256':'a'*64,'candidate_checkpoint_sha256':candidate.sha256,'opponent_checkpoint_sha256':opponent.sha256 if opponent else (legacy.sha256 if legacy else None),'temperature':0.0,'backend':'cpp','protocol_version':candidate.protocol_version}


def _audit(tmp_path,manifest,candidate_sha):
    audit=FinalHoldoutAudit(tmp_path/'audit.json',deals_manifest=manifest);audit.initialize();audit.lock_candidate(candidate_sha);return audit


def _reserve(audit,manifest,*,evaluation_id,candidate_sha,plan='p',config='c'):
    return audit.reserve(
        evaluation_id=evaluation_id,candidate_sha256=candidate_sha,plan_identity_hash=plan,config_hash=config,
        actual_dataset_identity=dataset_identity(manifest,'final'),
        actual_generator_version=manifest['generator_version'],
        actual_deal_count=len(manifest['datasets']['final']),
    )


def test_consumed_and_candidate_mismatch_reject_before_zero_game_callbacks(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,monkeypatch):
    manifest=_manifest();candidate,champion,legacy,runner=_setup(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint)
    audit=_audit(tmp_path,manifest,candidate.sha256);_reserve(audit,manifest,evaluation_id='used',candidate_sha=candidate.sha256);audit.mark_running(evaluation_id='used');audit.finish(evaluation_id='used',status='consumed',evaluation_hash='e')
    calls={'n':0}
    def count(**kw):calls['n']+=1;return _fake_record(**kw)
    monkeypatch.setattr(evaluator_module,'play_game',count)
    with pytest.raises(ValueError,match='not unused'):
        runner.run_final_audit(deals_manifest=manifest,candidate_checkpoint=v2_checkpoint,final_audit=audit,evaluation_id='again')
    assert calls['n']==0 and not (tmp_path/'final-run/cluster_summary.json').exists()

    other_dir=tmp_path/'mismatch';other_dir.mkdir();candidate,champion,legacy,runner2=_setup(other_dir,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint)
    audit2=_audit(other_dir,manifest,champion.sha256)
    with pytest.raises(ValueError,match='candidate identity mismatch'):
        runner2.run_final_audit(deals_manifest=manifest,candidate_checkpoint=v2_checkpoint,final_audit=audit2,evaluation_id='mismatch')
    assert calls['n']==0


def test_concurrent_final_reservation_allows_exactly_one(tmp_path,v2_checkpoint):
    manifest=_manifest();candidate=load_v2_policy(v2_checkpoint);audit=_audit(tmp_path,manifest,candidate.sha256)
    barrier=Barrier(3);out=[]
    def reserve(name):
        barrier.wait()
        try:_reserve(audit,manifest,evaluation_id=name,candidate_sha=candidate.sha256,plan='p'+name);out.append(('ok',name))
        except Exception as e:out.append(('error',type(e).__name__))
    threads=[Thread(target=reserve,args=(name,)) for name in ('a','b')]
    [t.start() for t in threads];barrier.wait();[t.join() for t in threads]
    assert sum(item[0]=='ok' for item in out)==1 and sum(item[0]=='error' for item in out)==1


def test_first_game_failure_marks_failed_and_retry_rejects_without_more_games(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,monkeypatch):
    manifest=_manifest();candidate,_,_,runner=_setup(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint);audit=_audit(tmp_path,manifest,candidate.sha256)
    calls={'n':0}
    def fail(**kw):calls['n']+=1;raise RuntimeError('first game failed')
    monkeypatch.setattr(evaluator_module,'play_game',fail)
    result=runner.run_final_audit(deals_manifest=manifest,candidate_checkpoint=v2_checkpoint,final_audit=audit,evaluation_id='first')
    assert result['status']=='incomplete' and calls['n']==1 and audit.load()['state']=='failed'
    with pytest.raises(ValueError,match='not unused'):
        runner.run_final_audit(deals_manifest=manifest,candidate_checkpoint=v2_checkpoint,final_audit=audit,evaluation_id='retry')
    assert calls['n']==1


def test_keyboard_interrupt_and_plan_write_failure_leave_terminal_audit_state(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,monkeypatch):
    manifest=_manifest();candidate,_,_,runner=_setup(tmp_path/'interrupt',v2_checkpoint,second_v2_checkpoint,legacy_checkpoint);audit=_audit(tmp_path/'interrupt',manifest,candidate.sha256)
    monkeypatch.setattr(evaluator_module,'play_game',lambda **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):runner.run_final_audit(deals_manifest=manifest,candidate_checkpoint=v2_checkpoint,final_audit=audit,evaluation_id='interrupt')
    assert audit.load()['state']=='interrupted'

    base=tmp_path/'write';candidate,_,_,runner=_setup(base,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint);audit=_audit(base,manifest,candidate.sha256)
    real=evaluator_module.atomic_write_json
    def fail_plan(path,payload,**kwargs):
        if Path(path).name=='evaluation_plan.json':raise OSError('write failed')
        return real(path,payload,**kwargs)
    monkeypatch.setattr(evaluator_module,'atomic_write_json',fail_plan)
    with pytest.raises(OSError,match='write failed'):runner.run_final_audit(deals_manifest=manifest,candidate_checkpoint=v2_checkpoint,final_audit=audit,evaluation_id='write')
    assert audit.load()['state']=='failed'


def test_rejected_repeat_never_overwrites_old_summary(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,monkeypatch):
    manifest=_manifest();candidate,_,_,runner=_setup(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint);audit=_audit(tmp_path,manifest,candidate.sha256)
    _reserve(audit,manifest,evaluation_id='used',candidate_sha=candidate.sha256);audit.mark_running(evaluation_id='used');audit.finish(evaluation_id='used',status='consumed',evaluation_hash='e')
    runner.output_dir.mkdir(parents=True);summary=runner.output_dir/'cluster_summary.json';summary.write_bytes(b'old-summary')
    monkeypatch.setattr(evaluator_module,'play_game',lambda **kw: pytest.fail('must not play'))
    with pytest.raises(ValueError,match='not unused'):runner.run_final_audit(deals_manifest=manifest,candidate_checkpoint=v2_checkpoint,final_audit=audit,evaluation_id='again')
    assert summary.read_bytes()==b'old-summary'


def test_final_manifest_must_match_audit_before_materialize_forward_or_game(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,monkeypatch):
    manifest_a=build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':100,'selection':200,'final':760030},level_ranks=(5,),leader_seats=(0,))
    manifest_b=build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':100,'selection':200,'final':760040},level_ranks=(5,),leader_seats=(0,))
    assert dataset_identity(manifest_a,'final')!=dataset_identity(manifest_b,'final')
    candidate,_,_,runner=_setup(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint);audit_a=_audit(tmp_path,manifest_a,candidate.sha256)
    audit_before=audit_a.path.read_bytes();existing=runner.output_dir/'existing-result.json';runner.output_dir.mkdir(parents=True);existing.write_bytes(b'unchanged')
    calls={'materialize':0,'forward':0,'game':0}
    import guandan_marvel.evaluation.provenance as provenance_module
    from guandan_marvel.evaluation.agents import V2PolicySession
    real_materialize=provenance_module.materialize_deal_identity;real_forward=V2PolicySession.observe_and_score
    def materialize(row):calls['materialize']+=1;return real_materialize(row)
    def forward(self,step):calls['forward']+=1;return real_forward(self,step)
    def game(**kw):calls['game']+=1;raise RuntimeError(f"unexpected final seed {kw['deal']['seed']}")
    monkeypatch.setattr(provenance_module,'materialize_deal_identity',materialize);monkeypatch.setattr(V2PolicySession,'observe_and_score',forward);monkeypatch.setattr(evaluator_module,'play_game',game)
    with pytest.raises(ValueError,match='dataset identity mismatch'):
        runner.run_final_audit(deals_manifest=manifest_b,candidate_checkpoint=v2_checkpoint,final_audit=audit_a,evaluation_id='wrong-dataset')
    assert calls=={'materialize':0,'forward':0,'game':0}
    assert audit_a.path.read_bytes()==audit_before and audit_a.load()['state']=='unused'
    assert existing.read_bytes()==b'unchanged'
    assert {p.name for p in runner.output_dir.iterdir()}=={'existing-result.json'}


def test_correct_audit_and_manifest_complete_and_consume(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,monkeypatch):
    manifest=_manifest();candidate,_,_,runner=_setup(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint);audit=_audit(tmp_path,manifest,candidate.sha256)
    monkeypatch.setattr(evaluator_module,'play_game',_fake_record)
    result=runner.run_final_audit(deals_manifest=manifest,candidate_checkpoint=v2_checkpoint,final_audit=audit,evaluation_id='correct')
    state=audit.load()
    assert result['status']=='complete' and result['games_recorded']==20
    assert state['state']=='consumed' and state['active_evaluation']['dataset_identity']==dataset_identity(manifest,'final')


def test_timestamp_only_change_preserves_dataset_identity_and_cannot_reuse_consumed_holdout(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,monkeypatch):
    manifest=_manifest();candidate,_,_,runner=_setup(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint);audit=_audit(tmp_path,manifest,candidate.sha256)
    _reserve(audit,manifest,evaluation_id='used',candidate_sha=candidate.sha256);audit.mark_running(evaluation_id='used');audit.finish(evaluation_id='used',status='consumed',evaluation_hash='e')
    changed=__import__('copy').deepcopy(manifest);changed['created_at']='2099-01-01T00:00:00+00:00';changed['hash']=payload_hash(changed)
    assert dataset_identity(changed,'final')==dataset_identity(manifest,'final')
    calls={'n':0};monkeypatch.setattr(evaluator_module,'play_game',lambda **kw:calls.__setitem__('n',calls['n']+1))
    with pytest.raises(ValueError,match='not unused'):
        runner.run_final_audit(deals_manifest=changed,candidate_checkpoint=v2_checkpoint,final_audit=audit,evaluation_id='timestamp')
    assert calls['n']==0 and audit.load()['state']=='consumed'


def test_unused_audit_a_cannot_bypass_consumed_dataset_b(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint,monkeypatch):
    manifest_a=build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':100,'selection':200,'final':760030},level_ranks=(5,),leader_seats=(0,))
    manifest_b=build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':100,'selection':200,'final':760040},level_ranks=(5,),leader_seats=(0,))
    candidate,_,_,runner=_setup(tmp_path,v2_checkpoint,second_v2_checkpoint,legacy_checkpoint)
    audit_a=FinalHoldoutAudit(tmp_path/'audit-a.json',deals_manifest=manifest_a);audit_a.initialize();audit_a.lock_candidate(candidate.sha256)
    audit_b=FinalHoldoutAudit(tmp_path/'audit-b.json',deals_manifest=manifest_b);audit_b.initialize();audit_b.lock_candidate(candidate.sha256);_reserve(audit_b,manifest_b,evaluation_id='used-b',candidate_sha=candidate.sha256);audit_b.mark_running(evaluation_id='used-b');audit_b.finish(evaluation_id='used-b',status='consumed',evaluation_hash='e')
    before_a=audit_a.path.read_bytes();before_b=audit_b.path.read_bytes();calls={'n':0}
    monkeypatch.setattr(evaluator_module,'play_game',lambda **kw:calls.__setitem__('n',calls['n']+1))
    with pytest.raises(ValueError,match='dataset identity mismatch'):
        runner.run_final_audit(deals_manifest=manifest_b,candidate_checkpoint=v2_checkpoint,final_audit=audit_a,evaluation_id='borrow-a')
    assert calls['n']==0 and audit_a.path.read_bytes()==before_a and audit_b.path.read_bytes()==before_b


def test_audit_object_and_persisted_identity_conflict_rejected_inside_lock(tmp_path,v2_checkpoint):
    manifest_a=build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':100,'selection':200,'final':760030},level_ranks=(5,),leader_seats=(0,))
    manifest_b=build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':100,'selection':200,'final':760040},level_ranks=(5,),leader_seats=(0,))
    candidate=load_v2_policy(v2_checkpoint);audit_a=_audit(tmp_path,manifest_a,candidate.sha256)
    other=FinalHoldoutAudit(tmp_path/'other.json',deals_manifest=manifest_b);other.initialize();other.lock_candidate(candidate.sha256)
    audit_a.path.write_bytes(other.path.read_bytes());before=audit_a.path.read_bytes()
    with pytest.raises(ValueError,match='audit object and persisted audit'):
        _reserve(audit_a,manifest_a,evaluation_id='conflict',candidate_sha=candidate.sha256)
    assert audit_a.path.read_bytes()==before
