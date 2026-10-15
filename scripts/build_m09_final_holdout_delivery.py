"""Build the frozen iter2000 one-shot M09 final holdout delivery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from guandan.state import PreviousHandResult
from guandan_marvel.evaluation.common import payload_hash
from guandan_marvel.evaluation.manifests import build_deals_manifest, dataset_identity
from guandan_marvel.kaggle.common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from guandan_marvel.kaggle.discovery import DATASET_MANIFEST_NAME, validate_dataset_manifest
from guandan_marvel.kaggle.m09_unified_delivery import _overlay_init
from guandan_marvel.kaggle.package import SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, validate_source_archive

DELIVERY_ID = "guandan-marvel-m09-final-holdout-iter2000-v1"
CONTROL_NAME = "m09_final_holdout_control.json"
NOTEBOOK_NAME = "guandan_marvel_m09_final_holdout.ipynb"


def _timestamp(): return datetime.now(timezone.utc).isoformat()


def _extract_member(archive: Path, member: str, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        data = bundle.read(member)
    destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(data)


NOTEBOOK = r'''from pathlib import Path
import hashlib,importlib.util,json,os,shutil,subprocess,sys,threading,time,traceback
from IPython.display import FileLink,display
INPUT=Path('/kaggle/input');WORKING=Path('/kaggle/working');ERROR=None;EXIT=1;START=time.monotonic();STOP=threading.Event();STAGE={'name':'startup'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def stage(name,**extra):
 STAGE['name']=name;print(json.dumps({'event':'final_bootstrap_stage','stage':name,'elapsed_seconds':round(time.monotonic()-START,3),**extra},sort_keys=True),flush=True)
def heartbeat():
 while not STOP.wait(10):print(json.dumps({'event':'final_bootstrap_heartbeat','stage':STAGE['name'],'elapsed_seconds':round(time.monotonic()-START,3)},sort_keys=True),flush=True)
def stream(cmd,*,cwd,env,label):
 print(f'[{label}] RUN:',cmd,flush=True);p=subprocess.Popen([str(x) for x in cmd],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 for line in p.stdout:print(line.rstrip(),flush=True)
 code=p.wait();print(f'[{label}] exit={code}',flush=True);return code
threading.Thread(target=heartbeat,daemon=True).start();stage('dataset_discovery')
try:
 controls=sorted(INPUT.rglob('m09_final_holdout_control.json'))
 if len(controls)!=1:raise RuntimeError(f'exactly one final control required: {controls}')
 control_path=controls[0].resolve();base=control_path.parent;control=json.loads(control_path.read_text(encoding='utf-8'))
 import hashlib as _h
 body={k:v for k,v in control.items() if k!='hash'}
 if control.get('schema')!='guandan-marvel-m09-final-holdout-control-v1' or control.get('hash')!=_h.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest():raise RuntimeError('final control hash mismatch')
 stage('control_validation',files=len(control['files']))
 for row in control['files']:
  p=(base/row['path']).resolve()
  if not p.is_relative_to(base) or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise RuntimeError(f'final control file mismatch: {row}')
 dataset_path=base/'guandan_marvel_dataset.json';bootstrap_path=base/'guandan_marvel_bootstrap.py'
 spec=importlib.util.spec_from_file_location('final_bootstrap',bootstrap_path);boot=importlib.util.module_from_spec(spec);sys.modules[spec.name]=boot;spec.loader.exec_module(boot)
 empty=WORKING/'final-empty-input';empty.mkdir(exist_ok=True);stage('source_materialization');prepared=boot.prepare_inputs(dataset_path,input_root=empty,working_root=WORKING);source=Path(prepared['source_root'])
 overlay=WORKING/'m09-final-overlay';shutil.copytree(base/'control_overlay',overlay,dirs_exist_ok=True);sys.path.insert(0,str(overlay));sys.path.insert(1,str(source))
 profile=base/control['acceptance_profiles']['path'];target=source/'configs/acceptance_profiles.json';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(profile,target)
 runtime=base/control['runtime_binary']['path'];runtime_target=source/control['runtime_binary']['restore_relative_path'];runtime_target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(runtime,runtime_target)
 env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONPATH']=str(overlay)+os.pathsep+str(source);stage('native_import')
 if stream([sys.executable,'-u','-c',"import guandan_marvel.cpp_game as g;print(g.get_backend_build_identity(),flush=True)"],cwd=source,env=env,label='native-import')!=0:raise RuntimeError('native import failed')
 stage('one_shot_final_holdout');EXIT=stream([sys.executable,'-u',str(overlay/'scripts/m09_final_holdout.py'),'--control',str(control_path),'--working-root',str(WORKING)],cwd=source,env=env,label='final-holdout');STOP.set()
except BaseException as e:
 STOP.set();ERROR=f'{type(e).__name__}: {e}';print(ERROR,flush=True);traceback.print_exc()
OUTPUT=WORKING/'guandan-marvel-m09-final-holdout.zip'
if OUTPUT.is_file():display(FileLink(str(OUTPUT)))
if ERROR is not None:raise RuntimeError(ERROR)
if EXIT!=0:raise RuntimeError(f'final holdout failed exit={EXIT}; do not rerun this holdout without audit review')
'''


def render_notebook():
    return {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"},"accelerator":"GPU"},"cells":[{"cell_type":"markdown","metadata":{},"source":["# M09 One-Shot Final Holdout — Frozen IPPO iter2000\n","该 Notebook 只允许 Run All 一次。候选已冻结；不要添加训练 Output，不要修改 candidate/deals/control。结果不得用于重新选择 checkpoint。\n"]},{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":NOTEBOOK.splitlines(keepends=True)}]}


def build(args):
    output=args.output.resolve(); repo=args.repository.resolve(); source_ds=args.source_dataset.resolve(); continuation=args.continuation.resolve()
    if output.exists():shutil.rmtree(output)
    dataset=output/'dataset';dataset.mkdir(parents=True)
    for name in (SOURCE_ARCHIVE_NAME,SOURCE_MANIFEST_NAME,'guandan_marvel_bootstrap.py','acceptance_profiles.json','legacy_v1_anchor.pt'):
        src=(source_ds/name) if (source_ds/name).is_file() else (repo/'configs/acceptance_profiles.json' if name=='acceptance_profiles.json' else repo/'runs/a07probe/local_fast/ippo/ippo_update_000001_90453d2696194fc2b755820939dcbed2_periodic.pt')
        shutil.copy2(src,dataset/name)
    validate_source_archive(dataset/SOURCE_ARCHIVE_NAME)
    _extract_member(continuation,'candidates/ippo_iter002000.pt',dataset/'ippo_iter002000_final_candidate.pt')
    _extract_member(continuation,'champion_state/champions/generation000001_iter000100.pt',dataset/'ippo_iter000100_champion.pt')
    with zipfile.ZipFile(continuation) as z:
        recovery=z.read('training/training_recovery.committed.zip')
    with zipfile.ZipFile(__import__('io').BytesIO(recovery)) as rz:
        rm=json.loads(rz.read('recovery_manifest.json'));runtime=rm['runtime_binary'];(dataset/'runtime').mkdir();(dataset/'runtime'/Path(runtime['archive_path']).name).write_bytes(rz.read(runtime['archive_path']))
    deals=build_deals_manifest(counts={'training':0,'selection':0,'final':65},bases={'training':999990000,'selection':1000000000,'final':1500000000},level_ranks=(3,4,5,6,7,8,9,10,11,12,13,14,15),leader_seats=(0,1,2,3),previous_results=(None,PreviousHandResult((1,3,2,4),0,'DOUBLE_DOWN'),PreviousHandResult((1,2,3,4),0,'HEAD_THIRD'),PreviousHandResult((1,2,4,3),0,'HEAD_LAST'),PreviousHandResult((1,2,3,4),0,'HEAD_THIRD',True)))
    atomic_json(dataset/'final_deals_manifest.json',deals)
    freeze_body={'schema':'guandan-marvel-m09-final-candidate-freeze-v1','candidate_iteration':2000,'candidate_checkpoint_sha256':sha256_file(dataset/'ippo_iter002000_final_candidate.pt'),'training_plan_terminal_iteration':2000,'selection_evaluation_hash':'314b472d87013837089d3aa8c976b7c3f48f09212d78bd1520ab4076d8cdd161','selection_decision_hash':'3fb2d21394ff5145f1996269c8075d19fd7c834356a9991c66944e2195ec6e0e','selection_status':'inconclusive','freeze_reason':'plan-terminal iter2000 checkpoint; highest candidate-vs-champion point estimate; all health/rule/random floors passed; frozen before final holdout','frozen_at':_timestamp(),'final_holdout_run':False}
    freeze={**freeze_body,'hash':canonical_hash(freeze_body)};atomic_json(dataset/'final_candidate_freeze.json',freeze)
    overlay=dataset/'control_overlay';(overlay/'guandan_marvel/evaluation').mkdir(parents=True);(overlay/'scripts').mkdir();(overlay/'guandan_marvel/__init__.py').write_text(_overlay_init(),encoding='utf-8');(overlay/'guandan_marvel/evaluation/__init__.py').write_text(_overlay_init(),encoding='utf-8');shutil.copy2(repo/'scripts/m09_final_holdout.py',overlay/'scripts/m09_final_holdout.py')
    source_manifest=json.loads((dataset/SOURCE_MANIFEST_NAME).read_text(encoding='utf-8'));runtime_path=next((dataset/'runtime').iterdir())
    specs={
      'candidate_checkpoint':dataset/'ippo_iter002000_final_candidate.pt','champion_checkpoint':dataset/'ippo_iter000100_champion.pt','legacy_checkpoint':dataset/'legacy_v1_anchor.pt','deals_manifest':dataset/'final_deals_manifest.json','candidate_freeze':dataset/'final_candidate_freeze.json'}
    control_files=[*specs.values(),dataset/'acceptance_profiles.json',runtime_path,overlay/'scripts/m09_final_holdout.py',overlay/'guandan_marvel/__init__.py',overlay/'guandan_marvel/evaluation/__init__.py']
    rows=relative_file_manifest(dataset,control_files)
    control_body={'schema':'guandan-marvel-m09-final-holdout-control-v1','created_at':_timestamp(),'evaluation_id':'m09-final-iter2000-v1','source_identity':{'source_commit':source_manifest['git_commit'],'source_archive_sha256':sha256_file(dataset/SOURCE_ARCHIVE_NAME),'source_content_manifest_hash':source_manifest['manifest_hash']},'candidate_checkpoint':{'path':specs['candidate_checkpoint'].relative_to(dataset).as_posix(),'bytes':specs['candidate_checkpoint'].stat().st_size,'sha256':sha256_file(specs['candidate_checkpoint'])},'champion_checkpoint':{'path':specs['champion_checkpoint'].relative_to(dataset).as_posix(),'bytes':specs['champion_checkpoint'].stat().st_size,'sha256':sha256_file(specs['champion_checkpoint'])},'legacy_checkpoint':{'path':'legacy_v1_anchor.pt','bytes':specs['legacy_checkpoint'].stat().st_size,'sha256':sha256_file(specs['legacy_checkpoint'])},'deals_manifest':{'path':'final_deals_manifest.json','bytes':specs['deals_manifest'].stat().st_size,'sha256':sha256_file(specs['deals_manifest']),'manifest_hash':deals['hash'],'dataset_hash':dataset_identity(deals,'final')},'candidate_freeze':{'path':'final_candidate_freeze.json','bytes':specs['candidate_freeze'].stat().st_size,'sha256':sha256_file(specs['candidate_freeze']),'freeze_hash':freeze['hash']},'acceptance_profiles':{'path':'acceptance_profiles.json','sha256':sha256_file(dataset/'acceptance_profiles.json')},'runtime_binary':{'path':runtime_path.relative_to(dataset).as_posix(),'sha256':sha256_file(runtime_path),'restore_relative_path':runtime['restore_relative_path']},'final_clusters':65,'total_games':1300,'heartbeat_seconds':20,'files':rows,'claims':{'candidate_frozen':True,'final_holdout_one_shot':True,'final_holdout_run':False,'m09_accepted':False}}
    control={**control_body,'hash':payload_hash(control_body)};atomic_json(dataset/CONTROL_NAME,control)
    base_manifest=json.loads((source_ds/DATASET_MANIFEST_NAME).read_text(encoding='utf-8'));base_manifest.update({'created_at':_timestamp(),'mode':'preflight','profiles':{'preflight':{'schema':'guandan-marvel-kaggle-profile-v1','mode':'preflight','required_gpu_count':2,'required_gpu_name_contains':'T4','telemetry_interval_seconds':5}},'recovery':{'required':False,'auto_discovery':False},'m09':{'role':'final_holdout_iter2000','final_control':{'path':CONTROL_NAME,'sha256':sha256_file(dataset/CONTROL_NAME),'hash':control['hash']},'final_holdout_generated':True},'claims':{'remote_ready':True,'remote_verified':False,'candidate_frozen':True,'final_holdout_run':False,'m09_accepted':False,'strength_claim':False}});atomic_json(dataset/DATASET_MANIFEST_NAME,base_manifest);validate_dataset_manifest(base_manifest,manifest_path=dataset/DATASET_MANIFEST_NAME)
    atomic_json(output/NOTEBOOK_NAME,render_notebook());(output/'README_FINAL_HOLDOUT.md').write_text('# One-shot final holdout\n\nCreate one private Dataset from dataset/, import the Notebook, choose T4 x2, add only this Dataset, and Run All once. Do not rerun or modify inputs.\n',encoding='utf-8')
    upload=output/'guandan-marvel-m09-final-holdout-upload.zip';tmp=upload.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
      for p in sorted(dataset.rglob('*')):
       if p.is_file():z.write(p,p.relative_to(dataset).as_posix())
    with zipfile.ZipFile(tmp) as z:
      if (bad:=z.testzip()):raise ValueError(bad)
    tmp.replace(upload)
    manifest={'schema':'guandan-marvel-m09-final-holdout-delivery-v1','delivery_id':DELIVERY_ID,'created_at':_timestamp(),'dataset':'dataset','notebook':NOTEBOOK_NAME,'control_hash':control['hash'],'candidate_sha256':control['candidate_checkpoint']['sha256'],'final_dataset_hash':control['deals_manifest']['dataset_hash'],'total_games':1300,'upload':{'path':upload.name,'bytes':upload.stat().st_size,'sha256':sha256_file(upload)},'final_holdout_run':False,'m09_accepted':False};manifest['hash']=canonical_hash(manifest);atomic_json(output/'delivery_manifest.json',manifest);return manifest


def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-dataset',type=Path,required=True);p.add_argument('--continuation',type=Path,required=True);a=p.parse_args(argv);print(json.dumps(build(a),indent=2,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
