"""Offline transport/identity/bootstrap QA. Never invokes the training pipeline."""
from __future__ import annotations
import argparse,ast,base64,hashlib,io,json,subprocess,sys,uuid,warnings,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.continuation500.common import read_json,sha,canonical,atomic_json,validate_plan,BASE_SOURCE,BASE_RECOVERY,ANCHOR_SHA
from tools.continuation500.delivery import PAYLOAD,NOTEBOOK,MAGIC


def verify(delivery):
    delivery=Path(delivery).resolve();repo=Path(__file__).resolve().parents[1]
    verification=read_json(delivery/'DELIVERY_VERIFICATION.json')
    assert sha(delivery/PAYLOAD)==verification['payload_sha256']
    assert sha(delivery/NOTEBOOK)==verification['notebook_sha256']
    raw=(delivery/PAYLOAD).read_bytes();assert raw.startswith(MAGIC)
    assert not zipfile.is_zipfile(io.BytesIO(raw))
    z=zipfile.ZipFile(io.BytesIO(base64.b64decode(raw[len(MAGIC):].strip(),validate=True)))
    meta=json.loads(z.read('continuation_manifest.json'))
    assert meta['manifest_hash']==canonical({k:v for k,v in meta.items()if k!='manifest_hash'})==verification['manifest_hash']
    assert set(z.namelist())=={'continuation_manifest.json',*(r['path']for r in meta['files'])}
    for row in meta['files']:
        b=z.read(row['path']);assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256']
    plan=validate_plan(json.loads(z.read('plan.json')))
    assert plan['plan_hash']==verification['plan_hash']
    assert hashlib.sha256(z.read('base_recovery.zip')).hexdigest()==BASE_RECOVERY
    assert hashlib.sha256(z.read('anchor_iter250.pt')).hexdigest()==ANCHOR_SHA
    import torch
    anchor=torch.load(io.BytesIO(z.read('anchor_iter250.pt')),map_location='cpu',weights_only=True)
    identity=anchor['backend_identity'];implementation=identity['implementation']
    from pathlib import PurePosixPath
    assert plan['runtime_source_path']==str(PurePosixPath(implementation['extension_path']).parent.parent)
    assert plan['expected_python_version']==implementation['python_version']
    assert plan['expected_backend_hash']==identity['hash']
    with zipfile.ZipFile(io.BytesIO(z.read('driver.zip')))as driver:
        assert set(driver.namelist())=={r['path']for r in plan['driver_files']}
        for row in plan['driver_files']:
            b=driver.read(row['path']);assert hashlib.sha256(b).hexdigest()==row['sha256']
            assert b==(repo/'tools'/row['path']).read_bytes();ast.parse(b.decode('utf8'))
    v4=z.read('v4_payload.bin')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(v4.split(b'\n',1)[1].strip(),validate=True)))as frozen:
        original=frozen.read('guandan-marvel-source.zip')
        assert hashlib.sha256(original).hexdigest()==BASE_SOURCE['archive_sha256']
        with zipfile.ZipFile(io.BytesIO(original))as source:
            sm=json.loads(source.read('guandan-marvel-source/source_manifest.json'))
            assert sm['manifest_hash']==BASE_SOURCE['content_manifest_hash']
            for r in sm['files']:
                b=source.read('guandan-marvel-source/'+r['path'])
                assert len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
        parent=json.loads(frozen.read('parent_run.json'))
    from guandan_marvel.kaggle.pilot_evaluation import tasks_for
    seeds=set();deal_counts={}
    for name,count in [('preflight',4),('monitor',32),('final',128)]:
        m=json.loads(z.read(name+'_deals.json'));tasks=tasks_for(m)
        assert len(tasks)==count*4*3
        current={t['deal']['seed']for t in tasks};assert not current&seeds and not current&set(parent['initial_env_seeds'])
        seeds.update(current);deal_counts[name]=count
    notebook=read_json(delivery/NOTEBOOK)
    code=''.join(next(c['source']for c in notebook['cells']if c['cell_type']=='code'));ast.parse(code)
    assert verification['payload_sha256']in code and meta['manifest_hash']in code
    try:
        import nbformat
        nbformat.validate(nbformat.from_dict(notebook))
    except ImportError:pass
    # The actual bootstrap executes in isolated children. Only local paths and
    # the final call boundary are replaced; all transport/driver checks run.
    qa=delivery.parent/('artifact_qa_'+uuid.uuid4().hex[:8]);qa.mkdir()
    harness=qa/'harness.py'
    harness.write_text(HARNESS,encoding='utf8')
    cases=[]
    for case in ('valid','wrong_payload_hash','wrong_manifest_hash','missing_payload','reused_root','wrong_magic','zip_traversal','zip_duplicate'):
        run=qa/case;run.mkdir()
        proc=subprocess.run([sys.executable,'-X','utf8',str(harness),str(delivery),str(run),case],
            cwd=qa,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=120)
        (run/'console.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf8')
        assert proc.returncode==0,case+': '+proc.stdout[-2000:]+proc.stderr[-2000:]
        cases.append({'case':case,'status':'PASS'})
    result={'schema':'guandan-continuation500-artifact-qa-v1','status':'PASS','payload_sha256':sha(delivery/PAYLOAD),
        'notebook_sha256':sha(delivery/NOTEBOOK),'plan_hash':plan['plan_hash'],'source_files_checked':len(sm['files']),
        'driver_files_checked':len(plan['driver_files']),'deals':deal_counts,'bootstrap_cases':cases,
        'original_v4_source_byte_identical':True,'original_absolute_native_path_and_python_bound':True,'full_state_from_iter250_not_iter50':True,
        'opaque_transport_not_auto_unzip':True,'simulated_handoff_only':True,'training_updates_launched':0,
        'kaggle_t4x2_execution':'NOT_RUN_TARGET_PREFLIGHT_REQUIRED','details':str(qa)}
    atomic_json(delivery/'ARTIFACT_QA.json',result)
    verification.update(status='LOCAL_VERIFIED_TARGET_PREFLIGHT_PENDING',artifact_qa='ARTIFACT_QA.json')
    atomic_json(delivery/'DELIVERY_VERIFICATION.json',verification)
    return result


HARNESS=r"""import sys,json,hashlib,io,base64,zipfile,types,warnings
from pathlib import Path
D,R,case=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
notebook=json.loads((D/'guandan_continue500_v1.ipynb').read_text(encoding='utf8'))
code=''.join(next(c['source']for c in notebook['cells']if c['cell_type']=='code'))
payload=D/'guandan_continue500_payload_v1.bin';root=R/'working'
if case=='wrong_payload_hash':
    code=code.replace(next(l for l in code.splitlines()if l.startswith('EXPECTED_PAYLOAD_SHA256=')), 'EXPECTED_PAYLOAD_SHA256='+repr('0'*64))
if case=='wrong_manifest_hash':
    code=code.replace(next(l for l in code.splitlines()if l.startswith('EXPECTED_MANIFEST_HASH=')), 'EXPECTED_MANIFEST_HASH='+repr('0'*64))
if case=='missing_payload':payload=R/'missing.bin'
if case=='reused_root':root.mkdir()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
if case in ('wrong_magic','zip_traversal','zip_duplicate'):
    if case=='wrong_magic':raw=b'WRONG-VERSION\n'
    else:
        name='../escape'if case=='zip_traversal'else'plan.json';b=b'x'
        meta={'files':[{'path':name,'bytes':1,'sha256':hashlib.sha256(b).hexdigest()}]};meta['manifest_hash']=canonical(meta)
        stream=io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with zipfile.ZipFile(stream,'w')as z:
                z.writestr('continuation_manifest.json',json.dumps(meta));z.writestr(name,b)
                if case=='zip_duplicate':z.writestr(name,b)
        raw=b'GUANDAN-CONTINUATION500-V1\n'+base64.b64encode(stream.getvalue())+b'\n'
        code=code.replace(next(l for l in code.splitlines()if l.startswith('EXPECTED_MANIFEST_HASH=')),'EXPECTED_MANIFEST_HASH='+repr(meta['manifest_hash']))
    payload=R/'bad.bin';payload.write_bytes(raw)
    code=code.replace(next(l for l in code.splitlines()if l.startswith('EXPECTED_PAYLOAD_SHA256=')),'EXPECTED_PAYLOAD_SHA256='+repr(hashlib.sha256(raw).hexdigest()))
code=code.replace('PAYLOAD_PATH=None', 'PAYLOAD_PATH=Path('+repr(str(payload))+')')
code=code.replace("RUN_ROOT=Path('/kaggle/working/guandan_continue500_v1')",'RUN_ROOT=Path('+repr(str(root))+')')
# Explicit dry-run boundary. The real driver itself is imported and verified.
code=code.replace('    from continuation500.pipeline import run_pipeline','    from continuation500.pipeline import verify_driver_files\n    verify_driver_files(json.loads((dataset/"plan.json").read_text()))\n    run_pipeline=handoff')
calls=[]
def handoff(**kwargs):
    calls.append(kwargs)
    assert kwargs['dataset'].is_dir()and kwargs['recovery_zip']is None
    a=kwargs['run_root']/'artifacts';a.mkdir();(a/'report.html').write_text('dry-run only',encoding='utf8')
    target=kwargs['run_root']/'dry_run_results.zip';target.write_bytes(b'dry-run')
    return {'status':'complete','results_zip':str(target),'simulated_handoff':True}
display=types.ModuleType('IPython.display');display.display=lambda *a:None;display.HTML=lambda x:x;display.FileLink=lambda x:x
sys.modules['IPython.display']=display
try:exec(compile(code,'generated_notebook','exec'),{'handoff':handoff,'__name__':'__main__'})
except Exception as e:
    if case=='valid':raise
    assert not calls
    expected={'wrong_payload_hash':'exactly one','wrong_manifest_hash':'manifest mismatch','missing_payload':'exactly one',
        'reused_root':'already exists','wrong_magic':'transport version','zip_traversal':'unsafe archive path','zip_duplicate':'duplicate/oversized'}[case]
    assert expected.lower()in str(e).lower(),repr(e)
    assert not (R/'escape').exists()
else:
    assert case=='valid'and len(calls)==1
print('BOOTSTRAP QA PASS',case)
"""

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delivery',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.delivery),ensure_ascii=False,indent=2))
