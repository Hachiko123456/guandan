from __future__ import annotations
import argparse,json,shutil,zipfile,sys,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from guandan_marvel.kaggle.budget import build_completed_game_budget_contract
from guandan_marvel.kaggle.common import atomic_json,canonical_hash,relative_file_manifest,sha256_file
from guandan_marvel.kaggle.discovery import DATASET_MANIFEST_NAME,validate_dataset_manifest
from guandan_marvel.kaggle.package import SOURCE_ARCHIVE_NAME,SOURCE_MANIFEST_NAME,build_source_archive,validate_source_archive
DELIVERY_ID='guandan-marvel-m09-two-stage-stage2-warm-start-v1'; PLAN='marvel_m09_two_stage_continuation_plan.json'; LOCK='marvel_m08_locked_config.json'; ACCEPT='acceptance_profiles.json'; CONTROL='m09_two_stage_stage2_control.json'; NOTEBOOK='guandan_marvel_m09_two_stage_stage2.ipynb'
def ts(): return datetime.now(timezone.utc).isoformat()
def previous_contexts():
    return [None,
      {'ranking':[1,3,2,4],'winner_team':0,'outcome_class':'DOUBLE_DOWN','tribute_cancelled':False},
      {'ranking':[1,2,3,4],'winner_team':0,'outcome_class':'HEAD_THIRD','tribute_cancelled':False},
      {'ranking':[1,2,4,3],'winner_team':0,'outcome_class':'HEAD_LAST','tribute_cancelled':True}]
def context_schedule():
    rows=[]
    ranks=list(range(3,16))
    for rank in ranks:
      for leader in range(4):
        for previous in previous_contexts():
          row={'level_rank':rank,'leader_seat':leader,'weight':1}
          row['previous_result']=previous
          rows.append(row)
    return {'schema':'guandan-marvel-training-context-schedule-v1','version':1,'contexts':rows}
def notebook():
    code=r'''from pathlib import Path
import hashlib,json,os,subprocess,sys,threading,time,traceback
from IPython.display import FileLink,display
INPUT=Path('/kaggle/input');WORK=Path('/kaggle/working');ERROR=None;EXIT=1;START=time.monotonic();STOP=threading.Event();STAGE={'name':'startup'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def stage(n,**x):
 STAGE['name']=n;print(json.dumps({'event':'stage2_bootstrap_stage','stage':n,'elapsed_seconds':round(time.monotonic()-START,3),**x},sort_keys=True),flush=True)
def hb():
 while not STOP.wait(10): print(json.dumps({'event':'stage2_bootstrap_heartbeat','stage':STAGE['name'],'elapsed_seconds':round(time.monotonic()-START,3)},sort_keys=True),flush=True)
def stream(cmd,cwd,env):
 print('[stage2] RUN:',cmd,flush=True);p=subprocess.Popen([str(x) for x in cmd],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 for line in p.stdout: print(line.rstrip(),flush=True)
 return p.wait()
threading.Thread(target=hb,daemon=True).start()
try:
 stage('dataset_discovery'); controls=sorted(INPUT.rglob('m09_two_stage_stage2_control.json'))
 if len(controls)!=1: raise RuntimeError(f'exactly one stage2 control required: {controls}')
 cp=controls[0].resolve(); base=cp.parent; dataset=base/'guandan_marvel_dataset.json'; control=json.loads(cp.read_text(encoding='utf-8'))
 if control.get('manifest_hash')!=hashlib.sha256(json.dumps({k:v for k,v in control.items() if k!='manifest_hash'},sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest(): raise RuntimeError('stage2 control hash mismatch')
 spec=__import__('importlib.util').util.spec_from_file_location('stage2_bootstrap',base/'guandan_marvel_bootstrap.py');boot=__import__('importlib.util').util.module_from_spec(spec);sys.modules[spec.name]=boot;spec.loader.exec_module(boot)
 stage('source_materialization'); prepared=boot.prepare_inputs(dataset,input_root=INPUT,working_root=WORK); source=Path(prepared['source_root'])
 env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',GUANDAN_MARVEL_SOURCE_ROOT=str(source),GUANDAN_MARVEL_SOURCE_COMMIT=control['source_identity']['source_commit'],GUANDAN_MARVEL_SOURCE_ARCHIVE_SHA256=control['source_identity']['source_archive_sha256'],GUANDAN_MARVEL_SOURCE_CONTENT_MANIFEST_HASH=control['source_identity']['source_content_manifest_hash']);env['PYTHONPATH']=str(source)+os.pathsep+env.get('PYTHONPATH','')
 stage('native_build'); build_env=dict(env);build_env['GUANDAN_BUILD_TIMEOUT_SECONDS']='900'
 if stream([sys.executable,'-u','setup.py','build_ext','--inplace'],source,build_env)!=0: raise RuntimeError('native build failed')
 stage('native_import');
 if stream([sys.executable,'-u','-c','import guandan_marvel.cpp_game as g;print(g.get_backend_build_identity(),flush=True)'],source,env)!=0: raise RuntimeError('native import failed')
 stage('warm_start_training'); work=WORK/'stage2-training-work'; EXIT=stream([sys.executable,'-u','-m','guandan_marvel.kaggle.runner','--dataset-manifest',str(dataset),'--input-root',str(INPUT),'--working-root',str(work),'--source-root',str(source)],source,env)
 if EXIT!=0: raise RuntimeError(f'stage2 training failed with exit={EXIT}')
 out=work/'guandan-marvel-output'
 for p in [out/'guandan-marvel-output.zip',out/'guandan-marvel-recovery.zip']:
  if p.is_file(): display(FileLink(str(p)))
 stage('complete')
except BaseException as e:
 ERROR=f'{type(e).__name__}: {e}';print(ERROR,flush=True);traceback.print_exc()
finally: STOP.set()
if ERROR: raise RuntimeError(ERROR)
if EXIT!=0: raise RuntimeError(f'stage2 failed with exit={EXIT}')
'''
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'},'accelerator':'GPU'},'cells':[{'cell_type':'markdown','metadata':{},'source':['# Guandan MARVEL 阶段二 warm-start\n','此 Notebook 从阶段一 selection 选出的 policy checkpoint 开始，重置 optimizer/replay，使用新的上下文分布训练。\n']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(keepends=True)}]}
def build(output,source_dataset,candidate):
 output=Path(output).resolve(); source_dataset=Path(source_dataset).resolve(); candidate=Path(candidate).resolve()
 if output.exists(): shutil.rmtree(output)
 output.mkdir(parents=True); dataset=output/'dataset';dataset.mkdir()
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 source_build=build_source_archive(ROOT,dataset/SOURCE_ARCHIVE_NAME,commit=source_commit,allow_dirty=True)
 atomic_json(dataset/SOURCE_MANIFEST_NAME,source_build['source_manifest'])
 with zipfile.ZipFile(dataset/SOURCE_ARCHIVE_NAME) as source_zip:
  (dataset/'guandan_marvel_bootstrap.py').write_bytes(source_zip.read('guandan-marvel-source/guandan_marvel/kaggle/bootstrap.py'))
 for n in (LOCK,ACCEPT): shutil.copy2(source_dataset/n,dataset/n)
 plan=json.loads((ROOT/'configs'/PLAN).read_text(encoding='utf-8')); shutil.copy2(ROOT/'configs'/PLAN,dataset/PLAN)
 shutil.copy2(candidate,dataset/'warm_start_policy.pt')
 source=validate_source_archive(dataset/SOURCE_ARCHIVE_NAME); sm=json.loads((dataset/SOURCE_MANIFEST_NAME).read_text(encoding='utf-8')); identity={'source_commit':source['git_commit'],'source_archive_sha256':source['archive_sha256'],'source_content_manifest_hash':sm['manifest_hash']}
 schedule=context_schedule(); training={'target_iterations':2000,'checkpoint_interval':500,'evaluation_interval':0,'max_wall_seconds':42000,'checkpoint_keep_last':4,'evaluation_keep_last':0,'rollout_batch_size':4,'collect_batch_size':4,'replay_buffer_scale':8,'mini_batches':2,'max_sampler_bytes_per_rank':4294967296,'seed':470700,'agent_epoch':4,'base_lr':0.0001,'base_clip_coef':0.01,'base_reg_coef':0.05,'lr_stable_iterations':500,'reg_stable_iterations':500,'critic_aux_epoch':4,'amp_enabled':False,'compile_enabled':False,'model':{'architecture':'v1.3','history_dim':128,'feature_dim':256,'decoder_layers':4,'decoder_heads':4,'state_layers':3,'action_layers':2,'fuse_layers':2},'training_context_schedule':schedule}
 control_body={'schema':'guandan-marvel-two-stage-control-v1','version':1,'delivery_id':DELIVERY_ID,'created_at':ts(),'plan_id':plan['plan_id'],'plan_hash':canonical_hash(plan),'source_identity':identity,'stage2':{'id':'stage2_context_coverage_warm_start','target_iterations':2000,'base_candidate':{'path':'warm_start_policy.pt','sha256':sha256_file(dataset/'warm_start_policy.pt'),'bytes':candidate.stat().st_size},'training_context_schedule_hash':canonical_hash(schedule)},'claims':{'warm_start':True,'optimizer_reset':True,'replay_reset':True,'final_holdout_generated':False,'m09_accepted':False}}
 control={**control_body,'manifest_hash':canonical_hash(control_body)};atomic_json(dataset/CONTROL,control)
 profile={'schema':'guandan-marvel-kaggle-profile-v1','mode':'full','algorithm':'ippo','required_gpu_count':2,'required_gpu_name_contains':'T4','ddp':{'launcher':'torchrun','world_size':2,'backend':'nccl','require_cuda':True},'training':training,'warm_start':{'schema':'guandan-marvel-warm-start-input-v1','path':'warm_start_policy.pt','sha256':sha256_file(dataset/'warm_start_policy.pt'),'bytes':candidate.stat().st_size},'evaluation':{'base_deals':1,'seed_base':1000000000},'completed_game_budget':build_completed_game_budget_contract(comparison_id='m09-ippo-stage2-context-v1',algorithm='ippo',target_completed_games=8000,global_collect_batch_size=4,expected_iterations=2000,selection_dataset_hash=control['plan_hash']),'telemetry_interval_seconds':5,'m09':{'role':'two_stage_stage2_context_coverage','plan':{'path':PLAN,'sha256':sha256_file(dataset/PLAN),'plan_hash':control['plan_hash']},'locked_config':{'path':LOCK,'sha256':sha256_file(dataset/LOCK)}}}
 files=[dataset/n for n in (PLAN,LOCK,ACCEPT,'warm_start_policy.pt')]; control['control_files']=relative_file_manifest(dataset,files); control['manifest_hash']=canonical_hash({k:v for k,v in control.items() if k!='manifest_hash'}); atomic_json(dataset/CONTROL,control)
 manifest={'schema':'guandan-marvel-kaggle-dataset-v2','package_id':sm.get('package_id',DELIVERY_ID),'created_at':ts(),'mode':'full','source':{'archive':SOURCE_ARCHIVE_NAME,'archive_sha256':identity['source_archive_sha256'],'manifest':SOURCE_MANIFEST_NAME,'manifest_sha256':sha256_file(dataset/SOURCE_MANIFEST_NAME),'git_commit':identity['source_commit'],'content_manifest_hash':identity['source_content_manifest_hash']},'bootstrap':{'path':'guandan_marvel_bootstrap.py','sha256':sha256_file(dataset/'guandan_marvel_bootstrap.py')},'profiles':{'full':profile},'recovery':{'required':False,'auto_discovery':False},'m09':{'role':'two_stage_stage2_context_coverage','plan':{'path':PLAN,'sha256':sha256_file(dataset/PLAN),'plan_hash':control['plan_hash']},'locked_config':{'path':LOCK,'sha256':sha256_file(dataset/LOCK)},'control':{'path':CONTROL,'sha256':sha256_file(dataset/CONTROL),'manifest_hash':control['manifest_hash']}},'claims':{'stage2_warm_start':True,'final_holdout_generated':False,'m09_accepted':False}}
 atomic_json(dataset/DATASET_MANIFEST_NAME,manifest);validate_dataset_manifest(manifest,manifest_path=dataset/DATASET_MANIFEST_NAME);atomic_json(output/NOTEBOOK,notebook());(output/'README_STAGE2_UPLOAD.md').write_text('# 阶段二上传\n\n上传 dataset/ 全部文件，导入 Notebook，GPU T4 x2，只添加本 Dataset，Run All。阶段二从 warm_start_policy.pt 初始化，optimizer/replay 重置；不运行 final holdout。\n',encoding='utf-8')
 body={'schema':'guandan-marvel-m09-two-stage-stage2-delivery-v1','delivery_id':DELIVERY_ID,'created_at':ts(),'dataset':'dataset','notebook':NOTEBOOK,'control_hash':control['manifest_hash'],'base_candidate_sha256':sha256_file(dataset/'warm_start_policy.pt'),'final_holdout_generated':False,'m09_accepted':False};atomic_json(output/'delivery_manifest.json',{**body,'manifest_hash':canonical_hash(body)})
 upload=output/'guandan-marvel-m09-two-stage-stage2-upload.zip';tmp=upload.with_suffix('.zip.tmp')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
  for path in sorted(dataset.rglob('*')):
   if path.is_file():z.write(path,path.relative_to(dataset).as_posix())
 with zipfile.ZipFile(tmp) as z:
  if z.testzip():raise ValueError('stage2 upload ZIP CRC failure')
 tmp.replace(upload)
 return {**body,'upload':{'path':str(upload),'bytes':upload.stat().st_size,'sha256':sha256_file(upload)}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-dataset',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.output,a.source_dataset,a.candidate),indent=2,ensure_ascii=False))
