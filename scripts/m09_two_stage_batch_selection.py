from __future__ import annotations
import argparse,json,os,shutil,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
from typing import Mapping
from guandan_marvel.evaluation import PromotionPolicy,build_evaluation_plan,load_legacy_v1_opponent,load_v2_policy
from guandan_marvel.evaluation.common import atomic_write_json,payload_hash
from guandan_marvel.evaluation.dual_gpu import DeterministicWorkerGroup,WorkerCommand
from guandan_marvel.evaluation.provenance import verify_data_isolation
from guandan_marvel.evaluation.resumable import ResumableEvaluationStore,build_resumable_evaluation_plan
from guandan_marvel.kaggle.common import canonical_hash,sha256_file

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def ts(): return datetime.now(timezone.utc).isoformat()
def find_candidate(root,iteration):
 found=[]
 for p in root.rglob(f'ippo_iter{iteration:06d}.pt'):
  if p.is_file():
   try:
    a=load_v2_policy(p,device='cpu',expected_algorithm='ippo')
    if a.iteration==iteration: found.append((p,a))
   except Exception: pass
 hashes={a.sha256 for _,a in found}
 if len(hashes)!=1: raise RuntimeError(f'expected one identity for iter{iteration}, found={[(str(p),a.sha256) for p,a in found]}')
 return sorted(found,key=lambda x:(len(str(x[0])),str(x[0])))[0]
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--working-root',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);a=p.parse_args(argv)
 control=load(a.control);base=a.control.parent.resolve();out=a.working_root.resolve()/'guandan-marvel-stage1-selection';out.mkdir(parents=True,exist_ok=True)
 champion_path=base/control['champion']['path'];legacy_path=base/control['legacy']['path'];deals_path=base/control['deals']['path'];worker=base/control['worker']['path']
 champion=load_v2_policy(champion_path,device='cpu',expected_algorithm='ippo');legacy=load_legacy_v1_opponent(legacy_path);deals=load(deals_path)
 candidates=[]
 for iteration in control['candidate_iterations']:
  candidates.append(find_candidate(a.input_root.resolve(),int(iteration)))
 rows=[]
 for candidate_path,candidate in candidates:
  root=out/f'iter{candidate.iteration:06d}';root.mkdir(parents=True,exist_ok=True)
  pool={'id':'champion-generation-1','kind':'champion','checkpoint':str(champion_path),'checkpoint_sha256':champion.sha256,'policy_identity':champion.identity,'weight':1.0}
  isolation=verify_data_isolation(artifacts={x.sha256:x for x in (candidate,champion)}.values(),deals_manifest=deals,domain='selection',policy='formal')
  policy=PromotionPolicy(mode='formal',ci95_low_threshold=.52,min_clusters=int(control['selection_clusters']),rule_min_win_rate=.4,random_min_win_rate=.7)
  plan=build_evaluation_plan(mode='selection',deals_manifest=deals,deal_domain='selection',candidate_identity=candidate.identity,champion_identity=champion.identity,legacy_identity=legacy.identity,pool_sample=pool,temperature=0.0,data_isolation=isolation,promotion_policy=policy,evaluation_id=f'two-stage-selection-iter{candidate.iteration:06d}')
  plan=build_resumable_evaluation_plan(base_plan=plan,materialized_deals=isolation['materialized_deals']);atomic_write_json(root/'evaluation_plan.json',plan);atomic_write_json(root/'deals_manifest.json',deals)
  context={'schema':'guandan-marvel-m09-formal-worker-context-v1','plan_hash':plan['hash'],'plan_path':str(root/'evaluation_plan.json'),'deals_manifest_path':str(root/'deals_manifest.json'),'candidate_checkpoint':str(candidate_path),'champion_checkpoint':str(champion_path),'pool_checkpoint':str(champion_path),'legacy_checkpoint':str(legacy_path),'output_dir':str(root)}
  cp=root/'worker_context.json';atomic_write_json(cp,context)
  commands=[WorkerCommand(i,i,(sys.executable,'-u',str(worker),'--context',str(cp),'--worker-index',str(i),'--world-size','2','--device','cuda:0')) for i in range(2)]
  DeterministicWorkerGroup(output_dir=root,plan=plan,commands=commands,base_env=os.environ,heartbeat_seconds=20).run();result=ResumableEvaluationStore(root,plan).finalize();summary=load(root/'evaluation_summary.json')['cluster_summary'];per=summary['per_opponent']
  illegal=sum(int(x['eval/illegal_count']) for x in per.values());truncated=sum(int(x['eval/truncated_count']) for x in per.values());rule=per['rule'];champ=per['champion'];random=per['random'];legacy_row=per['legacy']
  eligible=illegal==0 and truncated==0 and float(random['eval/win_rate'])>=.7 and float(legacy_row['eval/win_rate'])>=.7
  score=[float(rule['eval/clustered_ci95_low']),float(rule['eval/win_rate']),-float(rule['eval/seat_rotation_gap']),float(champ['eval/clustered_ci95_low']),float(champ['eval/win_rate']),-candidate.iteration]
  rows.append({'iteration':candidate.iteration,'candidate_path':str(candidate_path),'candidate_sha256':candidate.sha256,'eligible':eligible,'score':score,'rule_win_rate':rule['eval/win_rate'],'rule_ci95_low':rule['eval/clustered_ci95_low'],'rule_seat_rotation_gap':rule['eval/seat_rotation_gap'],'champion_win_rate':champ['eval/win_rate'],'champion_ci95_low':champ['eval/clustered_ci95_low'],'random_win_rate':random['eval/win_rate'],'legacy_win_rate':legacy_row['eval/win_rate'],'illegal':illegal,'truncated':truncated,'evaluation_hash':result['evaluation_hash'],'summary_hash':result['summary_hash']})
 eligible=[r for r in rows if r['eligible']]
 if not eligible: raise RuntimeError('no stage1 candidate passed health/random/legacy floors')
 selected=max(eligible,key=lambda r:tuple(r['score']));anchor=out/'stage2_anchor.pt';shutil.copy2(selected['candidate_path'],anchor)
 body={'schema':'guandan-marvel-stage1-selection-decision-v1','created_at':ts(),'policy':control['selection_policy'],'candidates':rows,'selected':selected,'stage2_anchor':{'path':anchor.name,'sha256':sha256_file(anchor),'bytes':anchor.stat().st_size},'final_holdout_used':False};decision={**body,'hash':canonical_hash(body)};atomic_write_json(out/'stage1_selection_decision.json',decision)
 archive=a.working_root.resolve()/'stage1-selection-results.zip';tmp=archive.with_suffix('.zip.tmp')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
  for path in sorted(out.rglob('*')):
   if path.is_file():z.write(path,path.relative_to(out).as_posix())
 with zipfile.ZipFile(tmp) as z:
  if z.testzip():raise RuntimeError('selection ZIP CRC failure')
 os.replace(tmp,archive);print(json.dumps({'status':'complete','selected':selected,'anchor':str(anchor),'archive':str(archive),'decision_hash':decision['hash']},indent=2,ensure_ascii=False),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
