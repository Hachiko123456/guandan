"""Build the self-contained opaque continuation upload; never trains or uploads."""
from __future__ import annotations
import ast,base64,hashlib,io,json,shutil,zipfile
from pathlib import Path
from .common import (SCHEMA,BASE_SOURCE,BASE_RECOVERY,ANCHOR_SHA,ORIGIN_SHA,V4_PAYLOAD_SHA,RUNTIME_SOURCE_PATH,EXPECTED_PYTHON,EXPECTED_BACKEND_HASH,canonical,sha,read_json,atomic_json)
PAYLOAD='guandan_continue500_payload_v1.bin';NOTEBOOK='guandan_continue500_v1.ipynb'
MAGIC=b'GUANDAN-CONTINUATION500-V1\n'

BOOTSTRAP=r'''from pathlib import Path
import base64,hashlib,io,json,os,sys,time,traceback,zipfile
STARTED=time.monotonic()
EXPECTED_PAYLOAD_SHA256=__PAYLOAD_HASH__
EXPECTED_MANIFEST_HASH=__MANIFEST_HASH__
PAYLOAD_PATH=None  # optional exact path if multiple identical inputs are mounted
RECOVERY_ZIP=None  # optional latest_recovery.zip produced by THIS continuation package
RUN_ROOT=Path('/kaggle/working/guandan_continue500_v1')

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def unpack_zip(z,destination,flat=False,max_bytes=512*1024**2):
    infos=z.infolist();names=[i.filename.casefold()for i in infos]
    if len(names)!=len(set(names))or sum(i.file_size for i in infos)>max_bytes:raise RuntimeError('duplicate/oversized input archive')
    for i in infos:
        n=i.orig_filename;p=(destination/n).resolve()
        if n!=i.filename or any(c in n for c in ('\\',':','\0'))or not p.is_relative_to(destination.resolve())or p==destination.resolve()or ((i.external_attr>>16)&0o170000)==0o120000:raise RuntimeError('unsafe archive path')
        if flat and ('/'in n or i.is_dir()):raise RuntimeError('flat input files required')
    z.extractall(destination)

print('[CONTINUE500 BOOTSTRAP] verifying input before any training',flush=True)
try:
    if RUN_ROOT.exists():raise RuntimeError('RUN_ROOT already exists; download it and choose a NEW path / fresh kernel')
    candidates=[Path(PAYLOAD_PATH)]if PAYLOAD_PATH is not None else list(Path('/kaggle/input').rglob('guandan_continue500_payload_v1.bin'))
    matches=[p for p in candidates if p.is_file()and not p.is_symlink()and p.stat().st_size<=512*1024**2 and file_sha(p)==EXPECTED_PAYLOAD_SHA256]
    if len(matches)!=1:raise RuntimeError('Expected exactly one matching continuation500 .bin; do not mix V4/old pilot uploads: '+str(candidates))
    raw=matches[0].read_bytes();magic=b'GUANDAN-CONTINUATION500-V1\n'
    if not raw.startswith(magic):raise RuntimeError('wrong opaque transport version')
    data=base64.b64decode(raw[len(magic):].strip(),validate=True)
    with zipfile.ZipFile(io.BytesIO(data))as z:
        meta=json.loads(z.read('continuation_manifest.json'))
        if meta['manifest_hash']!=EXPECTED_MANIFEST_HASH or canonical({k:v for k,v in meta.items()if k!='manifest_hash'})!=meta['manifest_hash']:raise RuntimeError('Notebook/payload manifest mismatch')
        names={i.filename for i in z.infolist()}
        if names!={'continuation_manifest.json',*(r['path']for r in meta['files'])}:raise RuntimeError('undeclared continuation inputs')
        for r in meta['files']:
            b=z.read(r['path'])
            if len(b)!=r['bytes']or hashlib.sha256(b).hexdigest()!=r['sha256']:raise RuntimeError('input mismatch: '+r['path'])
        RUN_ROOT.mkdir(parents=True);dataset=RUN_ROOT/'input';dataset.mkdir()
        unpack_zip(z,dataset,flat=True)
    driver=RUN_ROOT/'driver';driver.mkdir()
    with zipfile.ZipFile(dataset/'driver.zip')as z:unpack_zip(z,driver,max_bytes=4*1024**2)
    sys.path.insert(0,str(driver))
    from continuation500.pipeline import run_pipeline
    result=run_pipeline(dataset=dataset,run_root=RUN_ROOT,recovery_zip=None if RECOVERY_ZIP is None else Path(RECOVERY_ZIP),started_at=STARTED)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    from IPython.display import display,FileLink,HTML
    display(HTML((RUN_ROOT/'artifacts/report.html').read_text(encoding='utf-8')))
    display(FileLink(result['results_zip']))
    if (RUN_ROOT/'artifacts/latest_recovery.zip').is_file():display(FileLink(str(RUN_ROOT/'artifacts/latest_recovery.zip')))
    if result['status']not in ('complete','budget_stopped'):raise RuntimeError('Stopped safely at '+str(result.get('failed_phase'))+': '+str(result.get('error')))
except BaseException as error:
    diagnostics=RUN_ROOT.parent/(RUN_ROOT.name+'_bootstrap');diagnostics.mkdir(parents=True,exist_ok=True)
    (diagnostics/'bootstrap_error.json').write_text(json.dumps({'status':'failed','error':repr(error),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-STARTED},ensure_ascii=False,indent=2),encoding='utf-8')
    raise
'''


def build(repository,output,base_inputs,validation_record=None):
    repo,out,inputs=Path(repository).resolve(),Path(output).resolve(),Path(base_inputs).resolve()
    build_dir=out.parent/(out.name+'_build')
    if out.exists()or build_dir.exists():raise FileExistsError('new output paths required; never overwrite old deliveries')
    checks={'v4_payload.bin':V4_PAYLOAD_SHA,'base_recovery.zip':BASE_RECOVERY,'anchor_iter250.pt':ANCHOR_SHA}
    for name,digest in checks.items():
        if sha(inputs/name)!=digest:raise ValueError('wrong base input '+name)
    data_dir=build_dir/'dataset';data_dir.mkdir(parents=True)
    for name in checks:shutil.copy2(inputs/name,data_dir/name)
    raw=(inputs/'v4_payload.bin').read_bytes()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(raw.split(b'\n',1)[1].strip(),validate=True)))as z:
        v4= json.loads(z.read('pilot_manifest.json'))
        profile=json.loads(z.read(v4['profiles']['precheck']))
        contexts=profile['training']['training_context_schedule']['contexts']
    # Pure metadata generation; frozen runtime later validates exactly these files.
    from guandan_marvel.evaluation.manifests import build_deals_manifest
    from guandan.state import PreviousHandResult
    previous=[];seen=set()
    for c in contexts:
        key=canonical(c.get('previous_result'))
        if key in seen:continue
        seen.add(key);p=c.get('previous_result')
        previous.append(None if p is None else PreviousHandResult(tuple(p['ranking']),p['winner_team'],p['outcome_class'],p.get('tribute_cancelled',False)))
    if len(previous)<4:raise ValueError('preflight must include the original tribute contexts')
    for name,count,base in [('preflight',4,1_822_200_000),('monitor',32,1_822_210_000),('final',128,1_822_220_000)]:
        m=build_deals_manifest(counts={'training':0,'selection':count,'final':0},
            bases={'training':0,'selection':base,'final':2_000_000_000},previous_results=tuple(previous))
        atomic_json(data_dir/(name+'_deals.json'),m)
    driver_dir=repo/'tools/continuation500';driver_files=[]
    with zipfile.ZipFile(data_dir/'driver.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1)as z:
        for p in sorted(driver_dir.glob('*.py')):
            if p.name=='delivery.py':continue
            b=p.read_bytes();ast.parse(b.decode('utf-8-sig'));name='continuation500/'+p.name
            z.writestr(name,b);driver_files.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    plan={'schema':SCHEMA,'id':'guandan-v4-stateful-continue250-to500-20261022-v1',
       'start_iteration':250,'target_iteration':500,'collect_global':40,'horizon':1024,'monitor_iteration':375,
       'wall_seconds_cap':14400,'evaluation_reserve_seconds':5400,
       'runtime_source_path':RUNTIME_SOURCE_PATH,'expected_python_version':EXPECTED_PYTHON,'expected_backend_hash':EXPECTED_BACKEND_HASH,
       'base_source':BASE_SOURCE,'base_recovery_sha256':BASE_RECOVERY,'anchor_sha256':ANCHOR_SHA,
       'origin_control_sha256':ORIGIN_SHA,'v4_payload_sha256':V4_PAYLOAD_SHA,'v4_manifest_hash':v4['manifest_hash'],
       'driver_archive_sha256':sha(data_dir/'driver.zip'),'driver_files':driver_files,
       'learning_recipe':'preserve exact V4 tensors/optimizer/sampler/RNG/scaler; only non-semantic absolute target/logging changes',
       'fabledan':'NOT_RUN','final_holdout':'NOT_TOUCHED','test_only':False}
    plan['plan_hash']=canonical(plan);atomic_json(data_dir/'plan.json',plan)
    if validation_record is not None:
        validation=read_json(validation_record)
        for row in validation['source_files']:
            if sha(repo/row['path'])!=row['sha256']:raise ValueError('stale validation '+row['path'])
        atomic_json(data_dir/'LOCAL_VALIDATION.json',validation)
    files=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(data_dir.iterdir())]
    meta={'schema':'guandan-continuation500-delivery-v1','plan_hash':plan['plan_hash'],'files':files,
          'training_start':250,'training_target':500,'total_games_target':20000,'new_games_cap':10000,
          'wall_seconds_cap':14400,'fabledan':'NOT_RUN','final_holdout':'NOT_TOUCHED'}
    meta['manifest_hash']=canonical(meta);atomic_json(data_dir/'continuation_manifest.json',meta)
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True)as z:
        for p in sorted(data_dir.iterdir()):z.write(p,p.name,compress_type=zipfile.ZIP_STORED if p.suffix=='.zip'else zipfile.ZIP_DEFLATED)
    payload=MAGIC+base64.b64encode(stream.getvalue())+b'\n'
    if len(payload)>512*1024**2 or zipfile.is_zipfile(io.BytesIO(payload)):raise ValueError('invalid/oversize opaque payload')
    out.mkdir();(out/PAYLOAD).write_bytes(payload)
    payload_sha=sha(out/PAYLOAD)
    code=BOOTSTRAP.replace('__PAYLOAD_HASH__',repr(payload_sha)).replace('__MANIFEST_HASH__',repr(meta['manifest_hash']));ast.parse(code)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'kaggle':{'accelerator':'nvidiaTeslaT4','isInternetEnabled':False}},
        'cells':[{'id':'instructions','cell_type':'markdown','metadata':{},'source':['# 掼蛋 v4 完整状态续训：iter250 → iter500\n',
            '**Dataset只上传 `'+PAYLOAD+'`；Notebook单独导入。无需另外上传旧v4包/旧结果ZIP。**\n',
            'GPU必须T4×2；Internet可关闭。保留原优化器/采样器/RNG，不从iter50重训。\n',
            '先双卡真实评估冒烟，再2次更新/保存+恢复1次更新，再评估新导出模型；全部通过才长训。\n',
            '目标绝对500、累计20000局，本次最多新增10000局；中间375做快评，500完整评估。\n',
            '总安全上限4小时，保留90分钟终点评估预算；不足则保存实际进度，同包恢复，不自动超500。\n',
            '已达500的恢复只补评估、零新增更新。外部平台故障无法绝对避免，已知错误会前置检查。\n']},
            {'id':'continue500-run','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    atomic_json(out/NOTEBOOK,notebook);shutil.copy2(repo/'docs/t4x2_continue500.md',out/'README_KAGGLE.md')
    if validation_record is not None:shutil.copy2(data_dir/'LOCAL_VALIDATION.json',out/'LOCAL_VALIDATION.json')
    verification={'schema':'guandan-continuation500-verification-v1','status':'BUILT_TARGET_PREFLIGHT_PENDING',
       'payload':PAYLOAD,'payload_bytes':len(payload),'payload_sha256':payload_sha,'notebook':NOTEBOOK,'notebook_sha256':sha(out/NOTEBOOK),
       'manifest_hash':meta['manifest_hash'],'plan_hash':plan['plan_hash'],'frozen_v4_source_sha256':BASE_SOURCE['archive_sha256'],
       'base_recovery_sha256':BASE_RECOVERY,'anchor_policy_sha256':ANCHOR_SHA,'new_training_launched':False}
    atomic_json(out/'DELIVERY_VERIFICATION.json',verification)
    return verification
