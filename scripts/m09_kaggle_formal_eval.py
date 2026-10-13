"""Run one M09 formal selection/champion evaluation from prior Kaggle notebook outputs."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys,time,zipfile
from pathlib import Path

def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()

def load(path):
 x=json.loads(Path(path).read_text(encoding='utf-8'))
 if not isinstance(x,dict):raise ValueError(f'JSON object required: {path}')
 return x

def safe_file(base,spec,label):
 p=(base/str(spec['path'])).resolve()
 if not p.is_relative_to(base.resolve()) or not p.is_file() or sha256_file(p)!=spec['sha256']:raise RuntimeError(f'{label} missing/hash mismatch')
 return p

def find_source(input_root,expected):
 found=[]
 for p in input_root.rglob('source_manifest.json'):
  try:
   x=load(p)
   root=p.parent
   if x.get('git_commit')==expected['source_commit'] and x.get('manifest_hash')==expected['source_manifest_hash'] and (root/'guandan_marvel').is_dir():
    binaries=list((root/'nanopokerlib').glob('_guandan_cpp*.so'))
    if binaries and sha256_file(binaries[0])==expected['runtime_binary_sha256']:found.append((root,binaries[0]))
  except Exception:pass
 if not found:raise RuntimeError('matching built r2 source output not found; add the main-training notebook output as input')
 identities={(load(root/'source_manifest.json')['manifest_hash'],sha256_file(binary)) for root,binary in found}
 if len(identities)!=1:raise RuntimeError('conflicting source outputs found')
 return sorted(found,key=lambda x:(len(str(x[0])),str(x[0])))[0][0]

def find_recovery(input_root,expected,working):
 dirs=[]
 for p in input_root.rglob('recovery_manifest.json'):
  try:
   x=load(p)
   if x.get('source_commit')==expected['source_commit'] and x.get('iteration')==expected['candidate_iteration'] and x.get('checkpoint_manifest_hash')==expected['candidate_checkpoint_manifest_hash']:dirs.append(p.parent)
  except Exception:pass
 if dirs:return sorted(dirs,key=str)[0]
 for p in input_root.rglob('guandan-marvel-recovery*.zip'):
  try:
   with zipfile.ZipFile(p) as z:x=json.loads(z.read('recovery_manifest.json'))
   if x.get('source_commit')==expected['source_commit'] and x.get('iteration')==expected['candidate_iteration'] and x.get('checkpoint_manifest_hash')==expected['candidate_checkpoint_manifest_hash']:
    target=working/'materialized-recovery';target.mkdir(parents=True,exist_ok=False);z=zipfile.ZipFile(p);z.extractall(target);z.close();return target
  except Exception:pass
 raise RuntimeError('matching iter500 recovery not found; add the segment0500 notebook output as input')

def find_champion(input_root,expected):
 found=[]
 for p in input_root.rglob('iter000100.pt'):
  if p.is_file() and sha256_file(p)==expected['initial_champion_sha256']:found.append(p)
 if not found:raise RuntimeError('initial iter100 champion not found; add the r2 IPPO pilot notebook output as input')
 return sorted(found,key=str)[0]

def export_candidate(training_path,destination):
 import torch
 payload=torch.load(training_path,map_location='cpu',weights_only=False)
 required=('algorithm','architecture','protocol_version','backend','backend_identity','resolved_config','resolved_config_hash','agent_state_dict','iteration','merged_policy','source_fingerprint','source_hash','source_parent_commit','training_data_provenance')
 if any(k not in payload for k in required):raise ValueError('training checkpoint lacks evaluation export fields')
 body={k:payload[k] for k in required}
 body.update(schema='guandan-marvel-evaluation-checkpoint-v1',resume_classification='evaluation_only',training_resume_allowed=False)
 destination.parent.mkdir(parents=True,exist_ok=True);tmp=destination.with_suffix('.pt.tmp');torch.save(body,tmp);os.replace(tmp,destination);return destination

def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--control-manifest',type=Path,required=True);ap.add_argument('--input-root',type=Path,default=Path('/kaggle/input'));ap.add_argument('--working-root',type=Path,default=Path('/kaggle/working'));a=ap.parse_args(argv)
 control=load(a.control_manifest);base=a.control_manifest.parent.resolve();expected=control['expected'];legacy=safe_file(base,control['legacy_checkpoint'],'legacy checkpoint')
 source=find_source(a.input_root.resolve(),expected);sys.path.insert(0,str(source));os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['GUANDAN_MARVEL_SOURCE_ROOT']=str(source);os.environ['GUANDAN_MARVEL_SOURCE_COMMIT']=expected['source_commit']
 from guandan.state import PreviousHandResult
 from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent,load_v2_policy
 from guandan_marvel.evaluation.common import atomic_write_json,payload_hash
 from guandan_marvel.evaluation.evaluator import EvaluationRunner
 from guandan_marvel.evaluation.manifests import build_deals_manifest
 from guandan_marvel.evaluation.pool import OpponentPoolStore
 from guandan_marvel.evaluation.promotion import ChampionStore,PromotionPolicy
 from guandan_marvel.kaggle.bootstrap import validate_recovery_directory
 recovery=find_recovery(a.input_root.resolve(),expected,a.working_root.resolve());rm=validate_recovery_directory(recovery);candidate_manifest=recovery/rm['checkpoint'];cm=load(candidate_manifest);rank0=next(x for x in cm['rank_shards'] if int(x['rank'])==0);training=candidate_manifest.parent/rank0['shard']
 champion_input=find_champion(a.input_root.resolve(),expected)
 output=a.working_root.resolve()/'guandan-marvel-formal-eval-iter0500';output.mkdir(parents=True,exist_ok=True);candidate=export_candidate(training,output/'candidates/ippo_iter000500.pt');champion_path=output/'champion_state/champions/generation000001_iter000100.pt';champion_path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(champion_input,champion_path)
 candidate_artifact=load_v2_policy(candidate,device='cuda:0',expected_algorithm='ippo');champion_artifact=load_v2_policy(champion_path,device='cuda:0',expected_algorithm='ippo');legacy_artifact=load_legacy_v1_opponent(legacy)
 deals=build_deals_manifest(counts={'training':0,'selection':32,'final':0},bases={'training':999990000,'selection':1000000000,'final':1500000000},level_ranks=(3,4,5,6,7,8,9,10,11,12,13,14,15),leader_seats=(0,1,2,3),previous_results=(None,PreviousHandResult((1,3,2,4),0,'DOUBLE_DOWN'),PreviousHandResult((1,2,3,4),0,'HEAD_THIRD'),PreviousHandResult((1,2,4,3),0,'HEAD_LAST'),PreviousHandResult((1,2,3,4),0,'HEAD_THIRD',True)))
 atomic_write_json(output/'deals_manifest.json',deals);cs=ChampionStore(output/'champion_state/champion_manifest.json');champ=cs.bootstrap(champion_artifact.identity,reason='M09 iter100 selected IPPO initial champion');pool=OpponentPoolStore(output/'champion_state/opponent_pool_manifest.json');pool.initialize(sampling_seed=20261014);pool.add_entry({'id':'champion-generation-1','kind':'champion','checkpoint':str(champion_artifact.path),'checkpoint_sha256':champion_artifact.sha256,'policy_identity':champion_artifact.identity,'weight':1.0,'source_champion_manifest_hash':champ['hash']});pool.add_entry({'id':'rule-anchor-v1','kind':'rule','weight':0.25,'version':'v2-public-row-heuristic-v1'});pool.add_entry({'id':'legacy-v1-anchor','kind':'legacy_v1','checkpoint':str(legacy_artifact.path),'checkpoint_sha256':legacy_artifact.sha256,'legacy_identity':legacy_artifact.identity,'weight':0.25})
 runner=EvaluationRunner(output_dir=output/'selection_run',champion_store=cs,opponent_pool=pool,legacy_checkpoint=legacy,device='cuda:0',temperature=0.0,promotion_policy=PromotionPolicy(mode='formal',ci95_low_threshold=0.52,min_clusters=32,rule_min_win_rate=0.40,random_min_win_rate=0.70),provenance_policy='formal');started=time.perf_counter();result=runner.run_selection(deals_manifest=deals,candidate_checkpoint=candidate)
 portable={'schema':'guandan-marvel-m09-portable-champion-state-v1','candidate_iteration':500,'evaluation_result':result,'champion_manifest':cs.load(),'opponent_pool_manifest':pool.recover(),'candidate_checkpoint':{'path':'candidates/ippo_iter000500.pt','sha256':sha256_file(candidate)},'initial_champion_checkpoint':{'path':'champion_state/champions/generation000001_iter000100.pt','sha256':sha256_file(champion_path)},'elapsed_seconds':time.perf_counter()-started,'final_holdout_used':False};atomic_write_json(output/'formal_evaluation_summary.json',portable)
 zpath=a.working_root.resolve()/'guandan-marvel-formal-eval-iter0500.zip';tmp=zpath.with_suffix('.zip.tmp');
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
  for p in sorted(output.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(output).as_posix())
 os.replace(tmp,zpath);print(json.dumps({'status':'complete','promotion':result.get('promotion_decision'),'zip':str(zpath),'zip_bytes':zpath.stat().st_size,'zip_sha256':sha256_file(zpath)},indent=2,ensure_ascii=False),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
