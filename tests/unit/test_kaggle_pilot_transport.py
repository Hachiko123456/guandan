from __future__ import annotations
import ast
import base64
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pilot_transport_hotfix", ROOT/"scripts/repackage_t4x2_pilot_transport.py")
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


def zip_bytes(files):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    return stream.getvalue()


@pytest.fixture
def original(tmp_path):
    source=zip_bytes({"guandan-marvel-source/setup.py":b"# mocked build only\n",
                      "guandan-marvel-source/marker.txt":b"unchanged source bytes"})
    files={"guandan-marvel-source.zip":source,"warm_start_policy.pt":b"fixture only; no model loaded",
           "precheck_profile.json":b'{"target":2}',"resume_check_profile.json":b'{"target":3}',
           "train_profile.json":b'{"target":50}',"development_deals.json":b'{"final":0}'}
    manifest={"source":{"archive_sha256":transport._transport_hash(source),"git_commit":"a"*40},
              "files":[{"path":name,"bytes":len(data),"sha256":transport._transport_hash(data)} for name,data in files.items()]}
    manifest["manifest_hash"]=transport._transport_canonical_hash(manifest)
    files["pilot_manifest.json"]=json.dumps(manifest).encode()
    code_tree=ast.parse((ROOT/"guandan_marvel/kaggle/pilot_delivery.py").read_text(encoding="utf-8"))
    node=next(n for n in code_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="BOOTSTRAP" for t in n.targets))
    code=ast.literal_eval(node.value).replace("EXPECTED_PILOT_MANIFEST_HASH = None",f"EXPECTED_PILOT_MANIFEST_HASH = {manifest['manifest_hash']!r}",1)
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[{"cell_type":"markdown","source":["old"]},
        {"cell_type":"code","source":code.splitlines(keepends=True),"execution_count":None,"outputs":[]}]}
    root=tmp_path/"v2";root.mkdir()
    (root/"pilot-upload.zip").write_bytes(zip_bytes(files))
    (root/"guandan_t4x2_pilot.ipynb").write_text(json.dumps(notebook),encoding="utf-8")
    return {"root":root,"files":files,"manifest":manifest,"notebook":notebook,"code":code}


@pytest.fixture
def packaged(original,tmp_path):
    output=tmp_path/"v3"
    metadata=transport.repackage(original["root"],output)
    return output,metadata


def test_payload_is_not_zip_and_preserves_exact_v2_archive(original,packaged):
    output,metadata=packaged
    data=(output/transport.TRANSPORT_FILENAME).read_bytes()
    assert not zipfile.is_zipfile(io.BytesIO(data))
    assert data.startswith(transport.TRANSPORT_MAGIC)
    decoded=base64.b64decode(data[len(transport.TRANSPORT_MAGIC):].strip(),validate=True)
    assert decoded==(original["root"]/"pilot-upload.zip").read_bytes()
    assert metadata["training_source_and_config_changed"] is False
    assert metadata["inner_pilot_manifest_hash"]==original["manifest"]["manifest_hash"]


def test_round_trip_verifies_every_file(original,packaged,tmp_path):
    output,meta=packaged
    run=tmp_path/"working/pilot"
    dataset=transport.materialize_pilot_transport(output,run,meta["payload_sha256"],meta["inner_pilot_manifest_hash"])
    for name,data in original["files"].items():assert (dataset/name).read_bytes()==data
    report=json.loads((run.parent/(run.name+"_bootstrap")/"bootstrap_diagnostics.json").read_text(encoding="utf-8"))
    assert report["status"]=="complete"
    assert report["actual_payload_sha256"]==meta["payload_sha256"]


def test_reproduces_v2_failure_if_inner_zip_is_expanded(original,tmp_path):
    dataset=tmp_path/"platform-expanded";dataset.mkdir()
    for name,data in original["files"].items():
        if name!="guandan-marvel-source.zip":(dataset/name).write_bytes(data)
    (dataset/"guandan-marvel-source").mkdir()
    code=original["code"].replace("DATASET_DIR = None",f"DATASET_DIR = Path({str(dataset)!r})",1)
    code=code.replace("RUN_ROOT = Path('/kaggle/working/guandan_t4x2_pilot')",f"RUN_ROOT = Path({str(tmp_path/'v2-run')!r})",1)
    with pytest.raises(RuntimeError,match="Input identity failure"):exec(compile(code,"v2_reproduction","exec"),{})
    assert not (tmp_path/"v2-run").exists()
    with pytest.raises(RuntimeError,match="missing_file"):
        transport.verify_unpacked_pilot(dataset,original["manifest"]["manifest_hash"],tmp_path/"diagnostics")
    report=json.loads((tmp_path/"diagnostics/bootstrap_diagnostics.json").read_text(encoding="utf-8"))
    assert report["errors"][0]["expanded_source_directory_present"] is True


@pytest.mark.parametrize("mode",["missing","modified","directory"])
def test_actual_input_failure_has_expected_and_actual_diagnostics(original,tmp_path,mode):
    dataset=tmp_path/"dataset";dataset.mkdir()
    for name,data in original["files"].items():(dataset/name).write_bytes(data)
    path=dataset/"guandan-marvel-source.zip"
    if mode=="missing":path.unlink()
    elif mode=="directory":path.unlink();path.mkdir()
    else:path.write_bytes(b"changed bytes")
    with pytest.raises(RuntimeError):transport.verify_unpacked_pilot(dataset,original["manifest"]["manifest_hash"],tmp_path/"diag")
    report=json.loads((tmp_path/"diag/bootstrap_diagnostics.json").read_text(encoding="utf-8"))
    row=report["errors"][0]
    assert row["expected_sha256"]==original["manifest"]["files"][0]["sha256"]
    if mode=="modified":assert row["actual_sha256"]==transport._transport_hash(b"changed bytes")
    else:assert row["is_file"] is False


def test_tampered_payload_rejected_before_extraction(packaged,tmp_path):
    output,meta=packaged
    payload=output/transport.TRANSPORT_FILENAME
    payload.write_bytes(payload.read_bytes()+b"tampered")
    run=tmp_path/"working/pilot"
    with pytest.raises(RuntimeError,match="matching"):
        transport.materialize_pilot_transport(output,run,meta["payload_sha256"],meta["inner_pilot_manifest_hash"])
    assert not (run.parent/(run.name+"_dataset")).exists()
    report=json.loads((run.parent/(run.name+"_bootstrap")/"bootstrap_diagnostics.json").read_text(encoding="utf-8"))
    assert report["candidate_details"][0]["sha256"]!=meta["payload_sha256"]


def test_renamed_zip_is_not_accepted(original,tmp_path):
    payload=tmp_path/transport.TRANSPORT_FILENAME
    payload.write_bytes((original["root"]/"pilot-upload.zip").read_bytes())
    with pytest.raises(ValueError,match="renaming"):
        transport.materialize_pilot_transport(tmp_path,tmp_path/"work/pilot",transport._transport_file_hash(payload),original["manifest"]["manifest_hash"])


def test_mismatched_manifest_rejected(packaged,tmp_path):
    output,meta=packaged
    with pytest.raises(ValueError,match="manifest identity"):
        transport.materialize_pilot_transport(output,tmp_path/"work/pilot",meta["payload_sha256"],"0"*64)


@pytest.mark.parametrize("name",["../escape","/absolute","C:/escape","dir\\escape"])
def test_unsafe_zip_paths_rejected(original,name):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w") as z:
        for key,value in original["files"].items():z.writestr(key,value)
        info=zipfile.ZipInfo("placeholder");info.filename=name
        z.writestr(info,b"untrusted")
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as z:
        with pytest.raises(ValueError,match="unsafe"):
            transport._verify_transport_container(z,original["manifest"]["manifest_hash"])


def test_input_safety_cap_is_not_bypassed(original,monkeypatch):
    monkeypatch.setattr(transport,"MAX_EXTRACTED_BYTES",1)
    with zipfile.ZipFile(original["root"]/"pilot-upload.zip") as z:
        with pytest.raises(ValueError,match="size"):transport._verify_transport_container(z,original["manifest"]["manifest_hash"])


def test_existing_work_is_not_overwritten(packaged,tmp_path):
    output,meta=packaged
    run=tmp_path/"work/pilot"
    existing=run.parent/(run.name+"_dataset");existing.mkdir(parents=True)
    (existing/"keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):transport.materialize_pilot_transport(output,run,meta["payload_sha256"],meta["inner_pilot_manifest_hash"])
    assert (existing/"keep.txt").read_text(encoding="utf-8")=="keep"


def test_ambiguous_payload_requires_explicit_selection(packaged,tmp_path):
    output,meta=packaged
    copy=output/"second";copy.mkdir()
    (copy/transport.TRANSPORT_FILENAME).write_bytes((output/transport.TRANSPORT_FILENAME).read_bytes())
    with pytest.raises(RuntimeError,match="multiple"):
        transport.materialize_pilot_transport(output,tmp_path/"work/pilot",meta["payload_sha256"],meta["inner_pilot_manifest_hash"])
    result=transport.materialize_pilot_transport(output,tmp_path/"work/pilot2",meta["payload_sha256"],meta["inner_pilot_manifest_hash"],output/transport.TRANSPORT_FILENAME)
    assert (result/"pilot_manifest.json").exists()


def test_repackage_refuses_overwrite(original,packaged):
    output,_=packaged
    with pytest.raises(FileExistsError):transport.repackage(original["root"],output)


def test_notebook_keeps_build_training_and_recovery_tail_unchanged(original,packaged):
    output,meta=packaged
    notebook=json.loads((output/"guandan_t4x2_pilot_v3.ipynb").read_text(encoding="utf-8"))
    code="".join(notebook["cells"][1]["source"])
    ast.parse(code)
    assert code[code.index("if RUN_ROOT.exists():"):]==original["code"][original["code"].index("if RUN_ROOT.exists():"):]
    assert meta["payload_sha256"] in code and meta["inner_pilot_manifest_hash"] in code
    assert "PILOT_PAYLOAD_PATH" in code and "T4" in "".join(notebook["cells"][0]["source"])


def test_generated_notebook_end_to_end_input_handoff_no_training(original,packaged,tmp_path,monkeypatch):
    output,meta=packaged
    notebook=json.loads((output/"guandan_t4x2_pilot_v3.ipynb").read_text(encoding="utf-8"))
    code="".join(notebook["cells"][1]["source"])
    run=tmp_path/"work/pilot_v3"
    code=code.replace("Path('/kaggle/input')",f"Path({str(output)!r})")
    code=code.replace("RUN_ROOT = Path('/kaggle/working/guandan_t4x2_pilot_v3')",f"RUN_ROOT = Path({str(run)!r})",1)
    calls=[]
    fake_pilot=types.ModuleType("guandan_marvel.kaggle.pilot")
    def fake_run(**kwargs):
        calls.append(kwargs)
        target=kwargs["output"];target.mkdir()
        (target/"report.html").write_text("<p>mock only</p>",encoding="utf-8")
        return {"status":"complete","simulation":True}
    fake_pilot.run_pilot=fake_run
    monkeypatch.setitem(sys.modules,"guandan_marvel.kaggle.pilot",fake_pilot)
    fake_torch=types.ModuleType("torch")
    fake_torch.cuda=types.SimpleNamespace(device_count=lambda:2,get_device_name=lambda i:"Tesla T4")
    monkeypatch.setitem(sys.modules,"torch",fake_torch)
    display=types.ModuleType("IPython.display")
    display.display=lambda *a:None;display.HTML=lambda x:x;display.FileLink=lambda x:x
    monkeypatch.setitem(sys.modules,"IPython.display",display)
    import subprocess,os
    monkeypatch.setattr(subprocess,"run",lambda *a,**k:types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(sys,"path",list(sys.path))
    for key in ("PYTHONPATH","PYTHONUNBUFFERED","GUANDAN_MARVEL_SOURCE_COMMIT","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","NCCL_DEBUG","TORCH_DISTRIBUTED_DEBUG","TORCH_NCCL_ASYNC_ERROR_HANDLING"):
        monkeypatch.setenv(key,os.environ.get(key,""))
    exec(compile(code,"generated_v3_notebook","exec"),{})
    assert len(calls)==1
    dataset=calls[0]["dataset"]
    for name,data in original["files"].items():assert (dataset/name).read_bytes()==data
    assert calls[0]["recovery"] is None and calls[0]["started_at"]>0
    assert (run.parent/(run.name+"_bootstrap")/"bootstrap_diagnostics.json").exists()
