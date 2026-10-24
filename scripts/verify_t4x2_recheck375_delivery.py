"""Offline final-package identity and real-bootstrap dry-run QA. Zero GPU games."""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import io
import json
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path, PurePosixPath
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.recheck375.common import SOURCE, atomic_json, canonical, read_json, sha, validate_plan
from tools.recheck375.delivery import MAGIC, NOTEBOOK, PAYLOAD
from tools.recheck375.protocol import NEW_ARMS, audit_cache, schedule, tasks_for


def verify(delivery):
    import torch
    delivery = Path(delivery).resolve()
    repo = Path(__file__).resolve().parents[1]
    verification = read_json(delivery / "DELIVERY_VERIFICATION.json")
    assert sha(delivery / PAYLOAD) == verification["payload_sha256"]
    assert sha(delivery / NOTEBOOK) == verification["notebook_sha256"]
    raw = (delivery / PAYLOAD).read_bytes()
    assert raw.startswith(MAGIC) and not zipfile.is_zipfile(io.BytesIO(raw))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(raw[len(MAGIC):].strip(), validate=True))) as bundle:
        meta = json.loads(bundle.read("delivery_manifest.json"))
        assert meta["manifest_hash"] == canonical({k: v for k, v in meta.items() if k != "manifest_hash"}) == verification["manifest_hash"]
        assert set(bundle.namelist()) == {"delivery_manifest.json", *(r["path"] for r in meta["files"])}
        for row in meta["files"]:
            data = bundle.read(row["path"])
            assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
        plan = validate_plan(json.loads(bundle.read("plan.json")))
        assert plan["plan_hash"] == verification["plan_hash"]
        assert hashlib.sha256(bundle.read("frozen_source.zip")).hexdigest() == SOURCE["archive_sha256"]
        with zipfile.ZipFile(io.BytesIO(bundle.read("frozen_source.zip"))) as source:
            manifest = json.loads(source.read("guandan-marvel-source/source_manifest.json"))
            assert manifest["manifest_hash"] == SOURCE["content_manifest_hash"]
            for row in manifest["files"]:
                data = source.read("guandan-marvel-source/" + row["path"])
                assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
        with zipfile.ZipFile(io.BytesIO(bundle.read("driver.zip"))) as driver:
            assert set(driver.namelist()) == {r["path"] for r in plan["driver_files"]}
            for row in plan["driver_files"]:
                data = driver.read(row["path"])
                assert hashlib.sha256(data).hexdigest() == row["sha256"]
                assert data == (repo / "tools" / row["path"]).read_bytes()
                ast.parse(data.decode("utf8"))
                assert b"IPPOTrainer(" not in data and b"torch.distributed.run" not in data
        for name, spec in plan["models"].items():
            data = bundle.read(spec["path"])
            assert hashlib.sha256(data).hexdigest() == spec["sha256"]
            header = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            assert header["training_resume_allowed"] is False and header["iteration"] == spec["iteration"]
            assert not {"sampler_state", "agent_optimizer_state_dict", "optimizer_state_dict"} & set(header)
            backend = header["backend_identity"]
            assert backend["hash"] == plan["backend_hash"]
            native = backend["implementation"]
            assert str(PurePosixPath(native["extension_path"]).parent.parent) == plan["runtime_source_path"]
            assert native["python_version"] == plan["python"]
            assert hashlib.sha256(bundle.read("frozen_native.so")).hexdigest() == native["extension_sha256"]
        deals = json.loads(bundle.read("development_deals.json"))
        raw_cache = [json.loads(line) for line in bundle.read("cached_prior_games.jsonl").splitlines() if line.strip()]
        unique, cache_audit = audit_cache(raw_cache, deals, plan, json.loads(bundle.read("prior_endpoint_summary.json")))
        jobs = schedule(tasks_for(deals, NEW_ARMS, plan))
        assert [len(j) for j in jobs] == [384, 384]
        for job in jobs:
            assert all(sum(t["arm"] == arm for t in job) == 128 for arm in NEW_ARMS)
    notebook = read_json(delivery / NOTEBOOK)
    code = "".join(next(c["source"] for c in notebook["cells"] if c["cell_type"] == "code"))
    ast.parse(code)
    assert verification["payload_sha256"] in code and meta["manifest_hash"] in code
    try:
        import nbformat
        nbformat.validate(nbformat.from_dict(notebook))
    except ImportError:
        pass
    qa = delivery.parent / ("artifact_qa_" + uuid.uuid4().hex[:8])
    qa.mkdir()
    (qa / "harness.py").write_text(HARNESS, encoding="utf8")
    cases = []
    for case in ("valid", "budget_stop", "wrong_payload_hash", "wrong_manifest_hash", "missing_payload", "reused_root", "wrong_magic", "zip_traversal", "zip_duplicate"):
        run = qa / case
        run.mkdir()
        result = subprocess.run([sys.executable, "-X", "utf8", str(qa / "harness.py"), str(delivery), str(run), case],
                                cwd=qa, capture_output=True, text=True, encoding="utf8", errors="replace", timeout=120)
        (run / "console.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf8")
        assert result.returncode == 0, case + ": " + result.stdout[-1000:] + result.stderr[-2000:]
        cases.append({"case": case, "status": "PASS"})
    result = {"schema": "guandan-recheck375-artifact-qa-v1", "status": "PASS",
              "payload_sha256": sha(delivery / PAYLOAD), "notebook_sha256": sha(delivery / NOTEBOOK),
              "source_files_checked": len(manifest["files"]), "driver_files_checked": len(plan["driver_files"]),
              "original_absolute_backend_and_model_bytes_preserved": True,
              "active_training_jobs_or_optimizer_states_present": False, "new_training_games": 0, "gpu_evaluation_launched_by_qa": False,
              "cache_audit": cache_audit, "sharding": "each GPU:128 games per arm,384 new unique games total",
              "bootstrap_cases": cases, "scope": "actual input decode/hash/extract and real driver import, simulated final handoff only",
              "target_t4x2_runtime_preflight": "PENDING", "details": str(qa)}
    atomic_json(delivery / "ARTIFACT_QA.json", result)
    verification.update(status="LOCAL_VERIFIED_TARGET_PREFLIGHT_PENDING", artifact_qa="ARTIFACT_QA.json")
    atomic_json(delivery / "DELIVERY_VERIFICATION.json", verification)
    return result


HARNESS = r"""import sys,json,hashlib,io,base64,zipfile,types,warnings
from pathlib import Path
D,R,case=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
notebook=json.loads((D/'guandan_recheck375_v1.ipynb').read_text(encoding='utf8'))
code=''.join(next(c['source']for c in notebook['cells']if c['cell_type']=='code'))
payload=D/'guandan_recheck375_payload_v1.bin';root=R/'working'
if case=='wrong_payload_hash':code=code.replace(next(l for l in code.splitlines()if l.startswith('EXPECTED_PAYLOAD_SHA256 =')),'EXPECTED_PAYLOAD_SHA256 = '+repr('0'*64))
if case=='wrong_manifest_hash':code=code.replace(next(l for l in code.splitlines()if l.startswith('EXPECTED_MANIFEST_HASH =')),'EXPECTED_MANIFEST_HASH = '+repr('0'*64))
if case=='missing_payload':payload=R/'missing.bin'
if case=='reused_root':root.mkdir()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
if case in ('wrong_magic','zip_traversal','zip_duplicate'):
    if case=='wrong_magic':raw=b'WRONG-VERSION\n'
    else:
        name='../escape'if case=='zip_traversal'else'plan.json';data=b'x'
        meta={'files':[{'path':name,'bytes':1,'sha256':hashlib.sha256(data).hexdigest()}]};meta['manifest_hash']=canonical(meta)
        stream=io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with zipfile.ZipFile(stream,'w')as z:
                z.writestr('delivery_manifest.json',json.dumps(meta));z.writestr(name,data)
                if case=='zip_duplicate':z.writestr(name,data)
        raw=b'GUANDAN-RECHECK375-EVALONLY-V1\n'+base64.b64encode(stream.getvalue())+b'\n'
        code=code.replace(next(l for l in code.splitlines()if l.startswith('EXPECTED_MANIFEST_HASH =')),'EXPECTED_MANIFEST_HASH = '+repr(meta['manifest_hash']))
    payload=R/'bad.bin';payload.write_bytes(raw)
    code=code.replace(next(l for l in code.splitlines()if l.startswith('EXPECTED_PAYLOAD_SHA256 =')),'EXPECTED_PAYLOAD_SHA256 = '+repr(hashlib.sha256(raw).hexdigest()))
code=code.replace('PAYLOAD_PATH = None','PAYLOAD_PATH = Path('+repr(str(payload))+')')
code=code.replace("RUN_ROOT = Path('/kaggle/working/guandan_recheck375_v1')",'RUN_ROOT = Path('+repr(str(root))+')')
code=code.replace('    from recheck375.pipeline import run_pipeline','    from recheck375.common import verify_driver, validate_plan\n    verify_driver(validate_plan(json.loads((dataset/"plan.json").read_text())))\n    run_pipeline = handoff')
calls=[]
def handoff(**kwargs):
    calls.append(kwargs)
    assert kwargs['dataset'].is_dir()and kwargs['resume_results']is None
    a=kwargs['run_root']/'artifacts';a.mkdir();(a/'report.html').write_text('dry run; no GPU/no training',encoding='utf8')
    target=kwargs['run_root']/'dry_run_results.zip';target.write_bytes(b'dry-run')
    return {'status':'budget_stopped'if case=='budget_stop'else'complete','results_zip':str(target),'new_training_games':0,'optimizer_updates':0,'simulated_handoff':True}
display=types.ModuleType('IPython.display');display.display=lambda *a:None;display.HTML=lambda x:x;display.FileLink=lambda x:x
sys.modules['IPython.display']=display
try:exec(compile(code,'generated_notebook','exec'),{'handoff':handoff,'__name__':'__main__'})
except Exception as e:
    if case in ('valid','budget_stop'):raise
    assert not calls
    expected={'wrong_payload_hash':'exactly one','wrong_manifest_hash':'manifest mismatch','missing_payload':'exactly one',
              'reused_root':'already exists','wrong_magic':'transport version','zip_traversal':'unsafe ZIP member','zip_duplicate':'duplicate/oversize'}[case]
    assert expected.lower()in str(e).lower(),repr(e)
    assert not (R/'escape').exists()
else:assert case in ('valid','budget_stop')and len(calls)==1
print('BOOTSTRAP QA PASS',case)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.delivery), ensure_ascii=False, indent=2))
