from __future__ import annotations
import hashlib,json,os,shutil,sys,zipfile
from pathlib import Path

SCHEMA='guandan-continuation500-plan-v1'
BASE_SOURCE={'archive_sha256':'5afaf81a5e76e70e56c6f871898313df4962fabe3f9cef0818931a2c53b15a2a',
 'content_manifest_hash':'6574388f92df8457731215f1e08707e5b5b9368bb1d46c165bb3aa2be2997c2a',
 'git_commit':'fa4a7af3227dd8d76663bd9cf2b0e3a659d70373','core_source_hash':'0492d679f3ccea7c6388676463f6b266e178201498285ecc84e84de20d1657ef'}
BASE_RECOVERY='8ae3e5348010d65f092f1d7e0bc808f29faf82e73aebbd155bc12791617d0e21'
ANCHOR_SHA='965c60603b6aa8b5fe107d00c1a998e1389b64d104c4610386786b9487849f5f'
ORIGIN_SHA='2f453083cf818759ad142dad26358c85fa217661da7d23a8e4eb54736dbfaed8'
RUNTIME_SOURCE_PATH='/kaggle/working/guandan_t4x2_combo_v1/source/guandan-marvel-source'
EXPECTED_PYTHON='3.13.15'
EXPECTED_BACKEND_HASH='6e6875fdd2f754a6fbd5eef0b6cb540d29fa5456727d9eaab34cedc8852167ba'
V4_PAYLOAD_SHA='0b018993dfa09528ed6c5dd225a825d9d7867a2f1df42963bdb1344ed5dd7347'


def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w',encoding='utf-8')as f:f.write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def append(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8')as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def verify_hash_record(d,key='manifest_hash'):
 if d.get(key)!=canonical({k:v for k,v in d.items()if k!=key}):raise ValueError('invalid '+key)
def safe_path(root,name):
 root=Path(root).resolve();raw=root/name
 if not name or any(c in name for c in ('\\',':','\0')) or raw.is_symlink():raise ValueError('unsafe path '+repr(name))
 p=raw.resolve()
 if not p.is_relative_to(root)or p==root:raise ValueError('path escapes owned root')
 return p

def extract_safe(archive,destination,max_bytes=8*1024**3):
 destination=Path(destination).resolve()
 if destination.exists():raise FileExistsError(destination)
 with zipfile.ZipFile(archive)as z:
  infos=z.infolist();names=[i.filename.casefold()for i in infos]
  if len(names)!=len(set(names))or sum(i.file_size for i in infos)>max_bytes:raise ValueError('duplicate/oversize archive')
  for i in infos:
   if i.orig_filename!=i.filename:raise ValueError('normalized archive path is forbidden')
   safe_path(destination,i.filename)
   if ((i.external_attr>>16)&0o170000)==0o120000:raise ValueError('archive symlink')
  destination.mkdir(parents=True)
  z.extractall(destination)
 return destination

def validate_plan(plan,*,testing=False):
 verify_hash_record(plan,'plan_hash')
 if plan.get('schema')!=SCHEMA:raise ValueError('unsupported continuation plan')
 if plan.get('test_only'):
  if not testing:raise ValueError('test plan forbidden in production')
 else:
  if (plan.get('start_iteration'),plan.get('target_iteration'),plan.get('collect_global'),plan.get('horizon'))!=(250,500,40,1024):raise ValueError('production cap/collect/horizon changed')
  if plan['base_source']!=BASE_SOURCE or plan['base_recovery_sha256']!=BASE_RECOVERY or plan['anchor_sha256']!=ANCHOR_SHA:raise ValueError('production parent/source changed')
  if (plan.get('runtime_source_path'),plan.get('expected_python_version'),plan.get('expected_backend_hash'))!=(RUNTIME_SOURCE_PATH,EXPECTED_PYTHON,EXPECTED_BACKEND_HASH):raise ValueError('original absolute runtime/Python/backend identity changed')
  if plan['wall_seconds_cap']!=14400 or plan['evaluation_reserve_seconds']!=5400:raise ValueError('wall budget changed')
  if plan.get('fabledan')!='NOT_RUN'or plan.get('final_holdout')!='NOT_TOUCHED':raise ValueError('evaluation boundary changed')
 if not 0<=plan['start_iteration']<plan['target_iteration'] or plan['collect_global']<=0:raise ValueError('invalid iteration/count contract')
 return plan

def checkpoint_info(recovery,plan,*,verify_files=True,testing=False):
 validate_plan(plan,testing=testing);root=Path(recovery).resolve();record=read_json(root/'recovery_manifest.json');verify_hash_record(record)
 source=plan['base_source']
 for key,wanted in [('source_archive_sha256',source['archive_sha256']),('source_content_manifest_hash',source['content_manifest_hash']),('source_commit',source['git_commit'])]:
  if record.get(key)!=wanted:raise ValueError('recovery source mismatch: '+key)
 for row in record['files']:
  path=safe_path(root,row['path'])
  if not path.is_file()or path.stat().st_size!=row['bytes']or(verify_files and sha(path)!=row['sha256']):raise ValueError('recovery member mismatch: '+row['path'])
 ddp=read_json(safe_path(root,record['checkpoint']));verify_hash_record(ddp)
 if ddp['manifest_hash']!=record['checkpoint_manifest_hash']or not ddp.get('complete')or ddp['world_size']!=2:raise ValueError('incomplete DDP checkpoint')
 i=int(ddp['iteration'])
 if int(record['iteration'])!=i:raise ValueError('recovery iteration header disagrees with DDP checkpoint')
 games=int(ddp['global_counters']['completed_games_total'])
 if not plan['start_iteration']<=i<=plan['target_iteration']or games!=i*plan['collect_global']:raise ValueError('actual iteration/games outside continuation contract')
 for key in ('agent_merged_state_hash','agent_resume_state_hash','source_hash','semantic_contract_hash'):
  if len({r[key]for r in ddp['rank_shards']})!=1:raise ValueError('rank disagreement: '+key)
 if ddp['source_hash']!=source['core_source_hash']:raise ValueError('core source mismatch')
 saved=read_json(root/'artifacts/pilot_run_contract.json')
 if i>plan['start_iteration']or saved.get('schema')=='guandan-continuation500-contract-v1':
  if saved!=run_contract(plan):raise ValueError('recovery belongs to another continuation contract')
 else:
  if saved.get('pilot_manifest_hash')!=plan['v4_manifest_hash']or saved.get('global_collect_batch_size')!=plan['collect_global']:raise ValueError('original V4 contract mismatch')
 cfg=read_json(root/'artifacts/resolved_config.json')['train_config']
 if cfg['collect_batch_size']!=plan['collect_global']or cfg['game']['max_game_horizon']!=plan['horizon']:raise ValueError('recovered collection/game contract changed')
 if not plan.get('test_only'):
  if cfg['rollout_batch_size']!=48 or cfg['ntp_weight']!=.02 or (cfg['ppo']['base_clip_coef'],cfg['ppo']['base_reg_coef'],cfg['ppo']['base_lr'])!=(.2,.01,.0001):raise ValueError('learning recipe changed')
 return record,ddp,cfg

def run_contract(plan):
 return {'schema':'guandan-continuation500-contract-v1','plan_hash':plan['plan_hash'],
    'start_iteration':plan['start_iteration'],'target_iteration':plan['target_iteration'],
    'global_collect_batch_size':plan['collect_global'],'anchor_sha256':plan['anchor_sha256'],
    'base_source':plan['base_source'],'driver_archive_sha256':plan.get('driver_archive_sha256')}

def restore_native(recovery,source,record):
 binary=record['runtime_binary'];src=safe_path(recovery,binary['archive_path']);target=safe_path(source,binary['restore_relative_path'])
 if target.parent!=Path(source).resolve()/'nanopokerlib'or sha(src)!=binary['sha256']:raise ValueError('invalid native runtime binding')
 if target.exists():
  if sha(target)!=binary['sha256']:raise ValueError('conflicting native; never silently recompile')
  return target # never truncate an already mapped shared library
 shutil.copy2(src,target)
 return target

def export_policy(recovery,destination,plan,*,testing=False):
 import torch
 from guandan_marvel.kaggle.checkpoints import load_rank_checkpoint_for_resume,_state_dict_hash
 from guandan_marvel.evaluation.checkpoints import _atomic_torch_save,EVALUATION_CHECKPOINT_SCHEMA
 record,ddp,cfg=checkpoint_info(recovery,plan,verify_files=False,testing=testing)
 payload,checked=load_rank_checkpoint_for_resume(safe_path(recovery,record['checkpoint']).parent,rank=0,expected_world_size=2,expected_source_commit=plan['base_source']['git_commit'])
 if _state_dict_hash(payload['agent_state_dict'])!=ddp['rank_shards'][0]['agent_merged_state_hash']:raise ValueError('exported weights do not match training checkpoint')
 names=('algorithm','architecture','protocol_version','backend','backend_identity','resolved_config','resolved_config_hash','agent_state_dict','iteration','source_fingerprint','source_hash','source_parent_commit')
 policy={name:payload[name]for name in names}
 policy.update({'schema':EVALUATION_CHECKPOINT_SCHEMA,'training_data_provenance':ddp['training_data_provenance'],
     'distributed_training':{'world_size':2,'checkpoint_manifest_hash':ddp['manifest_hash'],'global_counters':ddp['global_counters']},
     'merged_policy':{'identity':'current_policy_copy','beta':0.,'frozen_for_evaluation':True},'resume_classification':'evaluation_only','training_resume_allowed':False})
 _atomic_torch_save(policy,Path(destination))
 return Path(destination)

def retire_raw(root,owner):
 root=Path(root).resolve();owner=Path(owner).resolve();target=(root/'checkpoints').resolve()
 if not target.is_relative_to(owner)or target==owner:raise ValueError('out-of-workspace raw retirement')
 if not(root/'guandan-marvel-recovery.zip').is_file():raise ValueError('cannot retire the only recovery')
 if target.exists():shutil.rmtree(target)
 atomic_json(root/'raw_retention.json',{'retired':True,'reason':'verified compressed recovery retained'})


def verify_recovery_zip(bundle,plan,*,testing=False):
    """Validate a durable bundle without making another multi-GB disk copy."""
    validate_plan(plan,testing=testing)
    with zipfile.ZipFile(bundle)as z:
        infos=z.infolist()
        if len(infos)!=len({i.filename.casefold()for i in infos})or sum(i.file_size for i in infos)>8*1024**3:raise ValueError('unsafe recovery container')
        rec=json.loads(z.read('recovery_manifest.json'));verify_hash_record(rec)
        if {i.filename for i in infos}!={'recovery_manifest.json',*(r['path']for r in rec['files'])}:raise ValueError('undeclared recovery files')
        for row in rec['files']:
            h=hashlib.sha256();size=0
            with z.open(row['path'])as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b);size+=len(b)
            if size!=row['bytes']or h.hexdigest()!=row['sha256']:raise ValueError('incomplete recovery member')
        ddp=json.loads(z.read(rec['checkpoint']));verify_hash_record(ddp)
        if ddp['manifest_hash']!=rec['checkpoint_manifest_hash']or not ddp['complete']or ddp['world_size']!=2:raise ValueError('incomplete DDP recovery')
        for key,wanted in [('source_archive_sha256',plan['base_source']['archive_sha256']),('source_content_manifest_hash',plan['base_source']['content_manifest_hash']),('source_commit',plan['base_source']['git_commit'])]:
            if rec[key]!=wanted:raise ValueError('recovery/source mismatch')
        i=int(ddp['iteration'])
        if int(rec['iteration'])!=i:raise ValueError('recovery iteration header mismatch')
        if not plan['start_iteration']<=i<=plan['target_iteration']or ddp['global_counters']['completed_games_total']!=i*plan['collect_global']:raise ValueError('recovery counters invalid')
        if ddp['source_hash']!=plan['base_source']['core_source_hash']:raise ValueError('recovery core changed')
        if json.loads(z.read('artifacts/pilot_run_contract.json'))!=run_contract(plan):raise ValueError('wrong continuation recovery contract')
        return rec,ddp
