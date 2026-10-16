from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT))

from guandan_marvel.kaggle.budget import build_completed_game_budget_contract
from guandan_marvel.kaggle.common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from guandan_marvel.kaggle.discovery import DATASET_MANIFEST_NAME, validate_dataset_manifest
from guandan_marvel.kaggle.package import SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, validate_source_archive

DELIVERY_ID = "guandan-marvel-m09-two-stage-stage1-iter5000-v3"
SCHEMA = "guandan-marvel-m09-two-stage-stage1-delivery-v1"
CONTROL_NAME = "m09_two_stage_control.json"
NOTEBOOK_NAME = "guandan_marvel_m09_two_stage_stage1.ipynb"
README_NAME = "README_STAGE1_UPLOAD.md"
PLAN_NAME = "marvel_m09_two_stage_continuation_plan.json"
LOCKED_NAME = "marvel_m08_locked_config.json"
MAIN_PLAN_NAME = "marvel_m09_main_training_plan.json"
RECOVERY_NAME = "guandan-marvel-recovery.zip"
ACCEPTANCE_NAME = "acceptance_profiles.json"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def notebook() -> dict[str, object]:
    code = r'''from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, threading, time, traceback, zipfile
from IPython.display import FileLink, display

INPUT_ROOT=Path('/kaggle/input'); WORKING_ROOT=Path('/kaggle/working')
START=time.monotonic(); STOP=threading.Event(); STAGE={'name':'startup'}; EXIT=1; ERROR=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def stage(name, **extra):
    STAGE['name']=name
    print(json.dumps({'event':'stage1_bootstrap_stage','stage':name,'elapsed_seconds':round(time.monotonic()-START,3),**extra},sort_keys=True,ensure_ascii=False),flush=True)

def heartbeat():
    while not STOP.wait(10):
        print(json.dumps({'event':'stage1_bootstrap_heartbeat','stage':STAGE['name'],'elapsed_seconds':round(time.monotonic()-START,3)},sort_keys=True),flush=True)

def stream(command, *, cwd, env, label):
    print(f'[{label}] RUN:',command,flush=True)
    p=subprocess.Popen([str(x) for x in command],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout: print(line.rstrip(),flush=True)
    code=p.wait(); print(f'[{label}] exit={code}',flush=True); return code

def export_candidate(training_path, destination):
    import torch
    payload=torch.load(training_path,map_location='cpu',weights_only=False)
    required=('algorithm','architecture','protocol_version','backend','backend_identity','resolved_config','resolved_config_hash','agent_state_dict','iteration','merged_policy','source_fingerprint','source_hash','source_parent_commit','training_data_provenance')
    missing=[key for key in required if key not in payload]
    if missing: raise RuntimeError(f'evaluation export fields missing: {missing}')
    body={key:payload[key] for key in required}
    body.update(schema='guandan-marvel-evaluation-checkpoint-v1',resume_classification='evaluation_only',training_resume_allowed=False)
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix('.pt.tmp'); torch.save(body,tmp); os.replace(tmp,destination)
    return destination

def zip_tree(root, destination):
    tmp=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file(): z.write(path,path.relative_to(root).as_posix())
    with zipfile.ZipFile(tmp) as z:
        if z.testzip(): raise RuntimeError('stage1 artifact ZIP CRC failure')
    os.replace(tmp,destination)

threading.Thread(target=heartbeat,daemon=True).start()
try:
    stage('dataset_discovery')
    controls=sorted(INPUT_ROOT.rglob('m09_two_stage_control.json'))
    if len(controls)!=1: raise RuntimeError(f'exactly one stage1 control is required: {controls}')
    control_path=controls[0].resolve(); base=control_path.parent
    dataset_path=base/'guandan_marvel_dataset.json'
    control=json.loads(control_path.read_text(encoding='utf-8'))
    if control.get('schema')!='guandan-marvel-two-stage-control-v1': raise RuntimeError('stage1 control schema mismatch')
    if control.get('manifest_hash')!=canonical({k:v for k,v in control.items() if k!='manifest_hash'}): raise RuntimeError('stage1 control hash mismatch')
    for row in control['control_files']:
        p=(base/row['path']).resolve()
        if not p.is_relative_to(base) or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            raise RuntimeError(f'control file mismatch: expected={row}, observed={{"exists":p.is_file(),"bytes":p.stat().st_size if p.is_file() else None,"sha256":sha(p) if p.is_file() else None}}')
    recovery_path=(base/control['recovery']['path']).resolve()
    recovery_directory=(base/Path(control['recovery']['path']).stem).resolve()
    if recovery_path.is_file():
        with zipfile.ZipFile(recovery_path) as recovery_zip:
            recovery_manifest=json.loads(recovery_zip.read('recovery_manifest.json'))
            bad=recovery_zip.testzip()
            if bad: raise RuntimeError(f'recovery ZIP CRC failure: {bad}')
        recovery_input_kind='zip'
    elif recovery_directory.is_dir() and (recovery_directory/'recovery_manifest.json').is_file():
        recovery_manifest=json.loads((recovery_directory/'recovery_manifest.json').read_text(encoding='utf-8'))
        recovery_input_kind='directory'
    else:
        raise RuntimeError(f'recovery input is missing: expected ZIP={recovery_path} or directory={recovery_directory}')
    print(json.dumps({'event':'stage1_recovery_input_detected','kind':recovery_input_kind,'path':str(recovery_path if recovery_input_kind=='zip' else recovery_directory)},sort_keys=True),flush=True)
    declared=recovery_manifest.get('manifest_hash')
    if declared!=canonical({k:v for k,v in recovery_manifest.items() if k!='manifest_hash'}): raise RuntimeError('recovery inner manifest hash mismatch')
    if int(recovery_manifest.get('iteration',-1))!=int(control['recovery']['iteration']) or declared!=control['recovery']['manifest_hash']:
        raise RuntimeError(f'recovery identity mismatch: iteration={recovery_manifest.get("iteration")}, manifest_hash={declared}')
    stage('source_materialization')
    spec=__import__('importlib.util').util.spec_from_file_location('stage1_bootstrap',base/'guandan_marvel_bootstrap.py')
    boot=__import__('importlib.util').util.module_from_spec(spec); sys.modules[spec.name]=boot; spec.loader.exec_module(boot)
    prepared=boot.prepare_inputs(dataset_path,input_root=INPUT_ROOT,working_root=WORKING_ROOT)
    source=Path(prepared['source_root']); recovery=Path(prepared['recovery_root'])
    if not recovery.is_dir(): raise RuntimeError('iter2000 recovery was not materialized')
    env=os.environ.copy(); env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',GUANDAN_MARVEL_SOURCE_ROOT=str(source),GUANDAN_MARVEL_SOURCE_COMMIT=control['source_identity']['source_commit'])
    env['GUANDAN_MARVEL_SOURCE_ARCHIVE_SHA256']=control['source_identity']['source_archive_sha256']; env['GUANDAN_MARVEL_SOURCE_CONTENT_MANIFEST_HASH']=control['source_identity']['source_content_manifest_hash']
    env['PYTHONPATH']=str(source)+os.pathsep+env.get('PYTHONPATH','')
    stage('native_import')
    if stream([sys.executable,'-u','-c','import guandan_marvel.cpp_game as g; print(g.get_backend_build_identity(),flush=True)'],cwd=source,env=env,label='native-import')!=0: raise RuntimeError('native import failed')
    stage('training_iter2000_to_5000')
    work=WORKING_ROOT/'stage1-training-work'
    command=[sys.executable,'-u','-m','guandan_marvel.kaggle.runner','--dataset-manifest',str(dataset_path),'--input-root',str(INPUT_ROOT),'--working-root',str(work),'--source-root',str(source),'--recovery-root',str(recovery)]
    EXIT=stream(command,cwd=source,env=env,label='stage1-training')
    if EXIT!=0: raise RuntimeError(f'stage1 training failed with exit={EXIT}')
    output=work/'guandan-marvel-output'; latest=json.loads((output/'latest_run.json').read_text(encoding='utf-8')); run_root=output/latest['path']
    stage('milestone_export')
    artifacts=WORKING_ROOT/'stage1-artifacts'; candidates=artifacts/'candidates'; candidates.mkdir(parents=True,exist_ok=True)
    milestones=[int(x) for x in control['stage1']['milestone_iterations']]
    rows=[]
    for iteration in milestones:
        checkpoint=run_root/'checkpoints'/f'iter{iteration:06d}'/'rank000.pt'
        if not checkpoint.is_file(): raise RuntimeError(f'milestone checkpoint missing: {checkpoint}')
        candidate=export_candidate(checkpoint,candidates/f'ippo_iter{iteration:06d}.pt')
        rows.append({'iteration':iteration,'checkpoint_sha256':sha(candidate),'bytes':candidate.stat().st_size,'path':candidate.relative_to(artifacts).as_posix()})
    manifest={'schema':'guandan-marvel-stage1-milestone-manifest-v1','plan_id':control['plan_id'],'source_recovery_iteration':2000,'target_iteration':5000,'candidates':rows,'training_run':latest,'created_at':timestamp()}
    manifest['hash']=canonical({k:v for k,v in manifest.items() if k!='hash'}); (artifacts/'stage1_milestone_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
    zip_tree(artifacts,WORKING_ROOT/'stage1-milestone-candidates.zip')
    stage('complete',candidates=len(rows),recovery=str(output/'guandan-marvel-recovery.zip'))
    print(json.dumps({'stage1_candidates':str(WORKING_ROOT/'stage1-milestone-candidates.zip'),'recovery':str(output/'guandan-marvel-recovery.zip'),'output':str(output/'guandan-marvel-output.zip'),'manifest':manifest},indent=2,sort_keys=True,ensure_ascii=False),flush=True)
except BaseException as error:
    ERROR=f'{type(error).__name__}: {error}'; print(ERROR,flush=True); traceback.print_exc()
finally:
    STOP.set()

for path in [WORKING_ROOT/'stage1-milestone-candidates.zip',WORKING_ROOT/'stage1-training-work/guandan-marvel-output/guandan-marvel-recovery.zip',WORKING_ROOT/'stage1-training-work/guandan-marvel-output/guandan-marvel-output.zip']:
    if path.is_file(): display(FileLink(str(path)))
if ERROR is not None: raise RuntimeError(ERROR)
if EXIT!=0: raise RuntimeError(f'stage1 failed with exit={EXIT}')
'''
    return {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}, 'accelerator': 'GPU'},
        'cells': [
            {'cell_type':'markdown','metadata':{},'source':[
                '# Guandan MARVEL 两阶段方案：阶段一 iter2000 → iter5000\n',
                '本 Notebook 只执行阶段一 exact resume。它不会运行 final holdout，也不会修改旧 M09 记录。训练完成后会导出 iter3000、iter4000、iter5000 三个轻量候选 checkpoint，供后续 selection evaluation 和阶段二 warm-start 使用。\n'
            ]},
            {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(keepends=True)},
        ]
    }


def build(repository: Path, output: Path, *, source_dataset: Path, continuation: Path) -> dict[str, object]:
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True)
    dataset=output/'dataset'; dataset.mkdir()
    source_dataset=source_dataset.resolve(); continuation=continuation.resolve()
    if not continuation.is_file(): raise FileNotFoundError(continuation)
    with zipfile.ZipFile(continuation) as incoming:
        names=set(incoming.namelist())
        if 'recovery_manifest.json' not in names:
            member='training/training_recovery.committed.zip'
            if member not in names: raise ValueError('input ZIP is neither a recovery nor a unified continuation with committed recovery')
            extracted=output/'_iter2000_recovery.tmp.zip'; extracted.write_bytes(incoming.read(member)); continuation=extracted
    source_archive=source_dataset/SOURCE_ARCHIVE_NAME; source_manifest=source_dataset/SOURCE_MANIFEST_NAME; bootstrap=source_dataset/'guandan_marvel_bootstrap.py'
    if not source_archive.is_file() or not source_manifest.is_file() or not bootstrap.is_file(): raise FileNotFoundError('source dataset must contain source ZIP, source manifest, and bootstrap')
    source_check=validate_source_archive(source_archive)
    recovery_sha=sha256_file(continuation)
    # Validate the recovery ZIP before embedding it; the bootstrap/runtime will validate again remotely.
    with zipfile.ZipFile(continuation) as z:
        payload=json.loads(z.read('recovery_manifest.json'))
        if payload.get('iteration') != 2000: raise ValueError(f'expected iter2000 recovery, got {payload.get("iteration")}')
        source_manifest_payload=json.loads(source_manifest.read_text(encoding='utf-8'))
        expected_recovery_identity=(source_check['git_commit'],source_check['archive_sha256'],source_manifest_payload['manifest_hash'])
        observed_recovery_identity=(payload.get('source_commit'),payload.get('source_archive_sha256'),payload.get('source_content_manifest_hash'))
        if observed_recovery_identity != expected_recovery_identity:
            raise ValueError(f'recovery/source identity mismatch: observed={observed_recovery_identity}, expected={expected_recovery_identity}')
    # Copy shared source and plans.
    for path in (source_archive, source_manifest, bootstrap): shutil.copy2(path,dataset/path.name)
    for name in (PLAN_NAME, LOCKED_NAME, MAIN_PLAN_NAME, ACCEPTANCE_NAME):
        src=(repository/'configs'/name) if (repository/'configs'/name).is_file() else (source_dataset/name)
        if not src.is_file(): raise FileNotFoundError(src)
        shutil.copy2(src,dataset/name)
    shutil.copy2(continuation,dataset/RECOVERY_NAME)
    if continuation.name == '_iter2000_recovery.tmp.zip': continuation.unlink(missing_ok=True)
    plan=json.loads((dataset/PLAN_NAME).read_text(encoding='utf-8'))
    source_manifest_payload=json.loads((dataset/SOURCE_MANIFEST_NAME).read_text(encoding='utf-8'))
    source_identity={'source_commit':source_check['git_commit'],'source_archive_sha256':source_check['archive_sha256'],'source_content_manifest_hash':source_manifest_payload['manifest_hash']}
    control_files=relative_file_manifest(dataset,[dataset/name for name in (PLAN_NAME,LOCKED_NAME,MAIN_PLAN_NAME,ACCEPTANCE_NAME)])
    control_body={
        'schema':'guandan-marvel-two-stage-control-v1','version':1,'delivery_id':DELIVERY_ID,'created_at':timestamp(),
        'plan_id':plan['plan_id'],'plan_hash':canonical_hash(plan),'source_identity':source_identity,
        'stage1':plan['stage1'],'stage2':plan['stage2'],'selection_policy':plan['selection_policy'],
        'recovery':{'path':RECOVERY_NAME,'archive_sha256_local':recovery_sha,'iteration':2000,'manifest_hash':payload['manifest_hash'],'checkpoint_manifest_hash':payload['checkpoint_manifest_hash']},'control_files':control_files,
        'claims':{'stage1_training_only':True,'stage1_milestones_exported':True,'stage2_ready_after_selection':False,'final_holdout_generated':False,'m09_accepted':False}
    }
    control={**control_body,'manifest_hash':canonical_hash(control_body)}; atomic_json(dataset/CONTROL_NAME,control)
    dataset_payload={
        'schema':'guandan-marvel-kaggle-dataset-v2','package_id':source_manifest_payload.get('package_id',DELIVERY_ID),'created_at':timestamp(),'mode':'resume',
        'source':{'archive':SOURCE_ARCHIVE_NAME,'archive_sha256':source_identity['source_archive_sha256'],'manifest':SOURCE_MANIFEST_NAME,'manifest_sha256':sha256_file(dataset/SOURCE_MANIFEST_NAME),'git_commit':source_identity['source_commit'],'content_manifest_hash':source_identity['source_content_manifest_hash']},
        'bootstrap':{'path':'guandan_marvel_bootstrap.py','sha256':sha256_file(dataset/'guandan_marvel_bootstrap.py')},
        'profiles':{'resume':{
            'schema':'guandan-marvel-kaggle-profile-v1','mode':'resume','expected_algorithm':'ippo','required_gpu_count':2,'required_gpu_name_contains':'T4','ddp':{'launcher':'torchrun','world_size':2},
            'resume':{'target_iterations':5000,'checkpoint_interval':500,'evaluation_interval':0,'max_wall_seconds':42000,'checkpoint_keep_last':8,'evaluation_keep_last':0},
            'evaluation':{'base_deals':1,'seed_base':1000000000},'telemetry_interval_seconds':5,
            'completed_game_budget':build_completed_game_budget_contract(comparison_id='m09-ippo-two-stage-stage1-v1',algorithm='ippo',target_completed_games=20000,global_collect_batch_size=4,expected_iterations=5000,selection_dataset_hash=control['plan_hash']),
            'm09':{'role':'two_stage_stage1_iter5000','plan':{'path':PLAN_NAME,'sha256':sha256_file(dataset/PLAN_NAME),'plan_hash':control['plan_hash']},'locked_config':{'path':LOCKED_NAME,'sha256':sha256_file(dataset/LOCKED_NAME)},'final_holdout_generated':False}
        }},
        'recovery':{'required':True,'auto_discovery':False,'selector':'embedded_iter2000_recovery'},
        'm09':{'role':'two_stage_stage1_iter5000','plan':{'path':PLAN_NAME,'sha256':sha256_file(dataset/PLAN_NAME),'plan_hash':control['plan_hash']},'locked_config':{'path':LOCKED_NAME,'sha256':sha256_file(dataset/LOCKED_NAME)},'control':{'path':CONTROL_NAME,'sha256':sha256_file(dataset/CONTROL_NAME),'manifest_hash':control['manifest_hash']},'final_holdout_generated':False},
        'claims':{'remote_ready':True,'stage1_training_complete':False,'stage2_ready_after_selection':False,'final_holdout_generated':False,'m09_accepted':False,'strength_claim':False}
    }
    atomic_json(dataset/DATASET_MANIFEST_NAME,dataset_payload); validate_dataset_manifest(dataset_payload,manifest_path=dataset/DATASET_MANIFEST_NAME)
    atomic_json(output/NOTEBOOK_NAME,notebook())
    (output/README_NAME).write_text(f'''# 阶段一：iter2000 → iter5000 Kaggle 上传教程\n\n本交付只运行阶段一 exact resume。起点为 iter2000 / 8000 games，终点为 iter5000 / 20000 games。\n\n## 上传\n\n1. 在 Kaggle 创建一个新的私有 Dataset。\n2. 上传 `dataset/` 目录中的全部文件，不要改名、不要只上传 ZIP。\n3. 导入 `{NOTEBOOK_NAME}`。\n4. Add Input 只添加这个阶段一 Dataset；不再添加旧 production Dataset，也不要另加旧 recovery。\n5. Accelerator 选择 GPU T4 x2。\n6. Run All。\n\n## 运行结果\n\nNotebook 完成后下载：\n\n- `stage1-milestone-candidates.zip`；\n- `guandan-marvel-recovery.zip`；\n- `guandan-marvel-output.zip`。\n\n`stage1-milestone-candidates.zip` 应包含 iter3000、iter4000、iter5000。阶段一不运行 final holdout。\n\n## 下一步\n\n先把候选包下载回本地，完成 selection evaluation，选择最佳 checkpoint 后，再使用新的 Stage2 builder 生成阶段二 warm-start Dataset。不要把 iter5000 自动当成阶段二起点。\n''',encoding='utf-8',newline='\n')
    files=[p for p in output.rglob('*') if p.is_file()]
    body={'schema':SCHEMA,'delivery_id':DELIVERY_ID,'created_at':timestamp(),'dataset':'dataset','notebook':NOTEBOOK_NAME,'control_hash':control['manifest_hash'],'source_identity':source_identity,'stage1_target_iteration':5000,'stage1_target_completed_games':20000,'files':relative_file_manifest(output,files),'stage2_ready_after_selection':False,'final_holdout_generated':False,'m09_accepted':False}
    atomic_json(output/'delivery_manifest.json',{**body,'manifest_hash':canonical_hash(body)})
    upload=output/'guandan-marvel-m09-two-stage-stage1-v3-upload.zip'; temp=upload.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for path in sorted(dataset.rglob('*')):
            if path.is_file(): z.write(path,path.relative_to(dataset).as_posix())
    with zipfile.ZipFile(temp) as z:
        if z.testzip(): raise ValueError('stage1 upload ZIP CRC failure')
    temp.replace(upload)
    return json.loads((output/'delivery_manifest.json').read_text(encoding='utf-8'))


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--source-dataset',type=Path,required=True); p.add_argument('--continuation',type=Path,required=True); args=p.parse_args()
    result=build(ROOT,args.output.resolve(),source_dataset=args.source_dataset,continuation=args.continuation)
    print(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
