"""Machine verification for final manifest/audit dataset binding."""
from __future__ import annotations
import argparse,copy,hashlib,json,tempfile
from pathlib import Path
from threading import Barrier,Thread
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import guandan_marvel.evaluation.evaluator as evaluator_module
import guandan_marvel.evaluation.provenance as provenance_module
from guandan_marvel.evaluation.agents import V2PolicySession
from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent,load_v2_policy
from guandan_marvel.evaluation.common import payload_hash
from guandan_marvel.evaluation.evaluator import EvaluationRunner
from guandan_marvel.evaluation.manifests import FinalHoldoutAudit,build_deals_manifest,dataset_identity
from guandan_marvel.evaluation.pool import OpponentPoolStore
from guandan_marvel.evaluation.promotion import ChampionStore,PromotionPolicy

def fhash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(**kw):
 d=kw['deal'];r=kw['rotation'];c=kw['candidate'];o=kw.get('opponent_v2');l=kw.get('legacy');team=r%2
 return {'schema':'guandan-marvel-evaluation-game-v1','status':'complete','opponent_id':kw['opponent_id'],'opponent_kind':kw['opponent_kind'],'deal_id':d['deal_id'],'deal_seed':d['seed'],'level_rank':d['level_rank'],'leader_seat':d['leader_seat'],'base_deal_sha256':hashlib.sha256(d['deal_id'].encode()).hexdigest(),'seat_rotation':r,'logical_to_physical':[(i+r)%4 for i in range(4)],'policies_by_seat':['candidate','x','candidate','x'],'candidate_team':team,'winner_team':team,'candidate_win':True,'ranking':[1,2,3,4] if team==0 else [2,1,4,3],'outcome_class':'HEAD_LAST','rewards':[1.,-1.,1.,-1.] if team==0 else [-1.,1.,-1.,1.],'candidate_team_reward':1.,'token_steps':10,'committed_steps':4,'engine_finish_token_step':10,'engine_finish_committed_step':4,'decisions_by_seat':[1,1,1,1],'illegal_count':0,'truncated_count':0,'runtime_seconds':.01,'actions_per_second':1000.,'action_trace_sha256':'a'*64,'candidate_checkpoint_sha256':c.sha256,'opponent_checkpoint_sha256':o.sha256 if o else (l.sha256 if l else None),'temperature':0.,'backend':'cpp','protocol_version':c.protocol_version}
def setup(base,candidate_path,champion_path,legacy_path):
 c=load_v2_policy(candidate_path);h=load_v2_policy(champion_path);l=load_legacy_v1_opponent(legacy_path);cs=ChampionStore(base/'champion.json');cs.bootstrap(h.identity);pool=OpponentPoolStore(base/'pool.json');pool.initialize();pool.add_entry({'id':'h','kind':'champion','checkpoint':str(h.path),'policy_identity':h.identity,'weight':1.});runner=EvaluationRunner(output_dir=base/'final-run',champion_store=cs,opponent_pool=pool,legacy_checkpoint=l.path,promotion_policy=PromotionPolicy(),provenance_policy='formal');return c,runner
def audit(path,manifest,sha):
 a=FinalHoldoutAudit(path,deals_manifest=manifest);a.initialize();a.lock_candidate(sha);return a
def reserve(a,m,eid,sha):return a.reserve(evaluation_id=eid,candidate_sha256=sha,plan_identity_hash='p'+eid,config_hash='c',actual_dataset_identity=dataset_identity(m,'final'),actual_generator_version=m['generator_version'],actual_deal_count=len(m['datasets']['final']))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--champion',type=Path,required=True);ap.add_argument('--legacy',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 A=build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':100,'selection':200,'final':760030},level_ranks=(5,),leader_seats=(0,));B=build_deals_manifest(counts={'training':1,'selection':1,'final':1},bases={'training':100,'selection':200,'final':760040},level_ranks=(5,),leader_seats=(0,));assert dataset_identity(A,'final')!=dataset_identity(B,'final')
 with tempfile.TemporaryDirectory() as td:
  td=Path(td);candidate,runner=setup(td/'mismatch',args.candidate,args.champion,args.legacy);auditA=audit(td/'audit-a.json',A,candidate.sha256);before=auditA.path.read_bytes();runner.output_dir.mkdir(parents=True);sentinel=runner.output_dir/'existing.json';sentinel.write_bytes(b'old');counts={'materialize':0,'forward':0,'game':0};realmat=provenance_module.materialize_deal_identity;realfwd=V2PolicySession.observe_and_score;realgame=evaluator_module.play_game
  def mat(row):counts['materialize']+=1;return realmat(row)
  def fwd(self,step):counts['forward']+=1;return realfwd(self,step)
  def game(**kw):counts['game']+=1;return record(**kw)
  provenance_module.materialize_deal_identity=mat;V2PolicySession.observe_and_score=fwd;evaluator_module.play_game=game
  error=None
  try:runner.run_final_audit(deals_manifest=B,candidate_checkpoint=args.candidate,final_audit=auditA,evaluation_id='mismatch')
  except Exception as exc:error=f'{type(exc).__name__}: {exc}'
  finally:provenance_module.materialize_deal_identity=realmat;V2PolicySession.observe_and_score=realfwd;evaluator_module.play_game=realgame
  mismatch={'error':error,'counts':counts,'audit_before_sha256':hashlib.sha256(before).hexdigest(),'audit_after_sha256':fhash(auditA.path),'audit_unchanged':auditA.path.read_bytes()==before,'output_files':sorted(p.name for p in runner.output_dir.iterdir()),'sentinel_unchanged':sentinel.read_bytes()==b'old'}
  # Correct A flow.
  candidate,good=setup(td/'good',args.candidate,args.champion,args.legacy);goodAudit=audit(td/'good-audit.json',A,candidate.sha256);evaluator_module.play_game=record
  try:goodResult=good.run_final_audit(deals_manifest=A,candidate_checkpoint=args.candidate,final_audit=goodAudit,evaluation_id='good')
  finally:evaluator_module.play_game=realgame
  correct={'status':goodResult['status'],'games':goodResult['games_recorded'],'audit_state':goodAudit.load()['state'],'dataset_identity':goodAudit.load()['active_evaluation']['dataset_identity']}
  # Timestamp-only identity and consumed rejection.
  changed=copy.deepcopy(A);changed['created_at']='2099-01-01T00:00:00+00:00';changed['hash']=payload_hash(changed);timestamp={'same_dataset_identity':dataset_identity(changed,'final')==dataset_identity(A,'final')};calls={'n':0};evaluator_module.play_game=lambda **kw:calls.__setitem__('n',calls['n']+1)
  try:good.run_final_audit(deals_manifest=changed,candidate_checkpoint=args.candidate,final_audit=goodAudit,evaluation_id='timestamp')
  except Exception as exc:timestamp['error']=f'{type(exc).__name__}: {exc}'
  finally:evaluator_module.play_game=realgame
  timestamp['game_callbacks']=calls['n']
  # Borrow unused A audit for consumed B.
  borrowBase=td/'borrow';candidate,borrowRunner=setup(borrowBase,args.candidate,args.champion,args.legacy);unusedA=audit(borrowBase/'unused-a.json',A,candidate.sha256);consumedB=audit(borrowBase/'consumed-b.json',B,candidate.sha256);reserve(consumedB,B,'used-b',candidate.sha256);consumedB.mark_running(evaluation_id='used-b');consumedB.finish(evaluation_id='used-b',status='consumed',evaluation_hash='e');ba=unusedA.path.read_bytes();bb=consumedB.path.read_bytes();bcalls={'n':0};evaluator_module.play_game=lambda **kw:bcalls.__setitem__('n',bcalls['n']+1)
  try:borrowRunner.run_final_audit(deals_manifest=B,candidate_checkpoint=args.candidate,final_audit=unusedA,evaluation_id='borrow')
  except Exception as exc:borrow_error=f'{type(exc).__name__}: {exc}'
  finally:evaluator_module.play_game=realgame
  borrow={'error':borrow_error,'game_callbacks':bcalls['n'],'audit_a_unchanged':unusedA.path.read_bytes()==ba,'audit_b_unchanged':consumedB.path.read_bytes()==bb,'audit_a_state':unusedA.load()['state'],'audit_b_state':consumedB.load()['state']}
  # Object/file conflict.
  conflict=td/'conflict';conflict.mkdir();obj=audit(conflict/'audit.json',A,candidate.sha256);other=audit(conflict/'other.json',B,candidate.sha256);obj.path.write_bytes(other.path.read_bytes());ob=obj.path.read_bytes()
  try:reserve(obj,A,'conflict',candidate.sha256)
  except Exception as exc:conflict_error=f'{type(exc).__name__}: {exc}'
  conflict_result={'error':conflict_error,'bytes_unchanged':obj.path.read_bytes()==ob}
  # Concurrent reservation.
  concurrent=audit(td/'concurrent.json',A,candidate.sha256);barrier=Barrier(3);out=[]
  def work(name):
   barrier.wait()
   try:reserve(concurrent,A,name,candidate.sha256);out.append('ok')
   except Exception:out.append('error')
  ts=[Thread(target=work,args=(n,)) for n in ('x','y')];[t.start() for t in ts];barrier.wait();[t.join() for t in ts]
  report={'schema':'guandan-marvel-m06-final-binding-verification-v1','dataset_a_identity':dataset_identity(A,'final'),'dataset_b_identity':dataset_identity(B,'final'),'mismatch':mismatch,'correct':correct,'timestamp_only':timestamp,'borrow_audit':borrow,'object_file_conflict':conflict_result,'concurrent':{'successes':out.count('ok'),'rejections':out.count('error')},'passes':bool(error and 'dataset identity mismatch' in error and counts=={'materialize':0,'forward':0,'game':0} and mismatch['audit_unchanged'] and mismatch['output_files']==['existing.json'] and correct['status']=='complete' and correct['audit_state']=='consumed' and timestamp['same_dataset_identity'] and timestamp['game_callbacks']==0 and borrow['game_callbacks']==0 and borrow['audit_a_unchanged'] and borrow['audit_b_unchanged'] and conflict_result['bytes_unchanged'] and out.count('ok')==1 and out.count('error')==1)}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False));return 0 if report['passes'] else 1
if __name__=='__main__':raise SystemExit(main())
