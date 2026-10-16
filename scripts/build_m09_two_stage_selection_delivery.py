from __future__ import annotations
import argparse,hashlib,json,shutil,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from guandan.state import PreviousHandResult
from guandan_marvel.evaluation.manifests import build_deals_manifest,dataset_identity
from guandan_marvel.kaggle.common import atomic_json,canonical_hash,relative_file_manifest,sha256_file
from guandan_marvel.kaggle.package import SOURCE_ARCHIVE_NAME,SOURCE_MANIFEST_NAME,validate_source_archive
DELIVERY='guandan-marvel-m09-two-stage-stage1-selection-v1';CONTROL='m09_two_stage_selection_control.json';NOTEBOOK='guandan_marvel_m09_two_stage_selection.ipynb'
def ts():return datetime.now(timezone.utc).isoformat()
def notebook():
 code=r'''from pathlib import Path
import hashlib,importlib.util,json,os,shutil,subprocess,sys,threading,time,traceback
from IPython.display import FileLink,display
INPUT=Path('/kaggle/input');WORK=Path('/kaggle/working');ERROR=None;EXIT=1;STOP=threading.Event();START=time.monotonic();STAGE={'name':'startup'}
def stage(n,**x):STAGE['name']=n;print(json.dumps({'event':'selection_bootstrap_stage','stage':n,'elapsed_seconds':round(time.monotonic()-START,3),**x},sort_keys=True),flush=True)
def hb():
 while not STOP.wait(10):print(json.dumps({'event':'selection_bootstrap_heartbeat','stage':STAGE['name'],'elapsed_seconds':round(time.monotonic()-START,3)},sort_keys=True),flush=True)
def stream(cmd,cwd,env):
 print('[selection] RUN:',cmd,flush=True);p=subprocess.Popen([str(x) for x in cmd],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 for line in p.stdout:print(line.rstrip(),flush=True)
 return p.wait()
threading.Thread(target=hb,daemon=True).start()
try:
 stage('dataset_discovery');controls=sorted(INPUT.rglob('m09_two_stage_selection_control.json'))
 if len(controls)!=1:raise RuntimeError(f'exactly one selection control required: {controls}')
 cp=controls[0].resolve();base=cp.parent;control=json.loads(cp.read_text(encoding='utf-8'));dataset=base/'guandan_marvel_dataset.json'
 spec=importlib.util.spec_from_file_location('selection_bootstrap',base/'guandan_marvel_bootstrap.py');boot=importlib.util.module_from_spec(spec);sys.modules[spec.name]=boot;spec.loader.exec_module(boot)
 empty=WORK/'selection-empty';empty.mkdir(exist_ok=True);stage('source_materialization');prepared=boot.prepare_inputs(dataset,input_root=empty,working_root=WORK);source=Path(prepared['source_root'])
 env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',GUANDAN_MARVEL_SOURCE_ROOT=str(source),GUANDAN_MARVEL_SOURCE_COMMIT=control['source_identity']['source_commit']);env['PYTHONPATH']=str(source)+os.pathsep+env.get('PYTHONPATH','')
 stage('native_build');build_env=dict(env);build_env['GUANDAN_BUILD_TIMEOUT_SECONDS']='900'
 if stream([sys.executable,'-u','setup.py','build_ext','--inplace'],source,build_env)!=0:raise RuntimeError('native build failed')
 stage('batch_selection');EXIT=stream([sys.executable,'-u',str(base/control['selection_script']['path']),'--control',str(cp),'--input-root',str(INPUT),'--working-root',str(WORK),'--source-root',str(source)],source,env)
 if EXIT!=0:raise RuntimeError(f'selection failed exit={EXIT}')
 for p in [WORK/'stage1-selection-results.zip',WORK/'guandan-marvel-stage1-selection/stage2_anchor.pt',WORK/'guandan-marvel-stage1-selection/stage1_selection_decision.json']:
  if p.is_file():display(FileLink(str(p)))
 stage('complete')
except BaseException as e:ERROR=f'{type(e).__name__}: {e}';print(ERROR,flush=True);traceback.print_exc()
finally:STOP.set()
if ERROR:raise RuntimeError(ERROR)
'''
 return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'},'accelerator':'GPU'},'cells':[{'cell_type':'markdown','metadata':{},'source':['# Stage1 batch selection\n','Add this private selection Dataset and the completed Stage1 Notebook Output. The run evaluates iter2000/3000/4000/5000 on one fresh selection dataset and exports one stage2_anchor.pt.\n']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(keepends=True)}]}
def extract(archive,member,dst):
 with zipfile.ZipFile(archive) as z:dst.write_bytes(z.read(member))
def build(output,source_dataset,baseline_archive):
 output=Path(output).resolve();source_dataset=Path(source_dataset).resolve();baseline_archive=Path(baseline_archive).resolve()
 if output.exists():shutil.rmtree(output)
 output.mkdir(parents=True);d=output/'dataset';d.mkdir()
 for n in (SOURCE_ARCHIVE_NAME,SOURCE_MANIFEST_NAME,'guandan_marvel_bootstrap.py','acceptance_profiles.json'):shutil.copy2(source_dataset/n,d/n)
 extract(baseline_archive,'candidate/ippo_iter002000.pt',d/'ippo_iter002000.pt');extract(baseline_archive,'champion/champions/generation000001_iter000100.pt',d/'ippo_iter000100_champion.pt');extract(baseline_archive,'legacy/legacy_v1_anchor.pt',d/'legacy_v1_anchor.pt')
 shutil.copy2(ROOT/'scripts/m09_two_stage_batch_selection.py',d/'m09_two_stage_batch_selection.py');shutil.copy2(ROOT/'scripts/m09_unified_formal_worker.py',d/'m09_unified_formal_worker.py')
 prev=(None,PreviousHandResult((1,3,2,4),0,'DOUBLE_DOWN',False),PreviousHandResult((1,2,3,4),0,'HEAD_THIRD',False),PreviousHandResult((1,2,4,3),0,'HEAD_LAST',True))
 deals=build_deals_manifest(counts={'training':0,'selection':65,'final':0},bases={'training':1800000000,'selection':1810000000,'final':1820000000},previous_results=prev);atomic_json(d/'selection_deals_manifest.json',deals)
 source=validate_source_archive(d/SOURCE_ARCHIVE_NAME);sm=json.loads((d/SOURCE_MANIFEST_NAME).read_text(encoding='utf-8'));identity={'source_commit':source['git_commit'],'source_archive_sha256':source['archive_sha256'],'source_content_manifest_hash':sm['manifest_hash']}
 body={'schema':'guandan-marvel-two-stage-selection-control-v1','created_at':ts(),'source_identity':identity,'candidate_iterations':[2000,3000,4000,5000],'selection_clusters':65,'selection_dataset_hash':dataset_identity(deals,'selection'),'champion':{'path':'ippo_iter000100_champion.pt','sha256':sha256_file(d/'ippo_iter000100_champion.pt')},'legacy':{'path':'legacy_v1_anchor.pt','sha256':sha256_file(d/'legacy_v1_anchor.pt')},'deals':{'path':'selection_deals_manifest.json','sha256':sha256_file(d/'selection_deals_manifest.json'),'hash':deals['hash']},'worker':{'path':'m09_unified_formal_worker.py','sha256':sha256_file(d/'m09_unified_formal_worker.py')},'selection_script':{'path':'m09_two_stage_batch_selection.py','sha256':sha256_file(d/'m09_two_stage_batch_selection.py')},'selection_policy':{'primary':['rule_ci95_low','rule_win_rate'],'secondary':['negative_rule_seat_rotation_gap','champion_ci95_low','champion_win_rate'],'health':['illegal_zero','truncated_zero'],'floors':{'random':.7,'legacy':.7},'tie_break':'earlier_iteration'},'final_holdout_used':False}
 control={**body,'manifest_hash':canonical_hash(body)};atomic_json(d/CONTROL,control)
 manifest={'schema':'guandan-marvel-kaggle-dataset-v2','package_id':sm['package_id'],'created_at':ts(),'mode':'preflight','source':{'archive':SOURCE_ARCHIVE_NAME,'archive_sha256':identity['source_archive_sha256'],'manifest':SOURCE_MANIFEST_NAME,'manifest_sha256':sha256_file(d/SOURCE_MANIFEST_NAME),'git_commit':identity['source_commit'],'content_manifest_hash':identity['source_content_manifest_hash']},'bootstrap':{'path':'guandan_marvel_bootstrap.py','sha256':sha256_file(d/'guandan_marvel_bootstrap.py')},'profiles':{'preflight':{'schema':'guandan-marvel-kaggle-profile-v1','mode':'preflight','required_gpu_count':2,'required_gpu_name_contains':'T4','telemetry_interval_seconds':5}},'recovery':{'required':False,'auto_discovery':False},'m09':{'role':'two_stage_stage1_selection','final_holdout_generated':False},'claims':{'selection_only':True,'final_holdout_generated':False,'m09_accepted':False}}
 atomic_json(d/'guandan_marvel_dataset.json',manifest);atomic_json(output/NOTEBOOK,notebook());(output/'README_SELECTION_UPLOAD.md').write_text('# Stage1 selection upload\n\nCreate one private Dataset from dataset/. Import the Notebook, choose T4 x2, add this Dataset plus the completed Stage1 Notebook Output, and Run All. Download stage1-selection-results.zip and stage2_anchor.pt.\n',encoding='utf-8')
 upload=output/'guandan-marvel-m09-stage1-selection-upload.zip';tmp=upload.with_suffix('.tmp')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
  for p in sorted(d.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(d).as_posix())
 tmp.replace(upload);return {'delivery_id':DELIVERY,'control_hash':control['manifest_hash'],'selection_dataset_hash':body['selection_dataset_hash'],'upload':str(upload)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-dataset',type=Path,required=True);p.add_argument('--baseline-archive',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.output,a.source_dataset,a.baseline_archive),indent=2,ensure_ascii=False))
