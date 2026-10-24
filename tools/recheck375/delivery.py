"""Create the self-contained eval-only upload. Does not train, infer or upload."""
from __future__ import annotations
import ast
import base64
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from .common import (BACKEND_HASH, DEVELOPMENT_HASH, MODEL_SHAS, PROTOCOL, PYTHON_VERSION,
                     RUNTIME_PATH, SCHEMA, SOURCE, TORCH_VERSION, atomic_json, canonical,
                     read_json, sha, state_hash, validate_plan)
from .protocol import CACHED_ARMS, NEW_ARMS, audit_cache

PAYLOAD = "guandan_recheck375_payload_v1.bin"
NOTEBOOK = "guandan_recheck375_v1.ipynb"
MAGIC = b"GUANDAN-RECHECK375-EVALONLY-V1\n"


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
    return stream.getvalue()


BOOTSTRAP = r"""from pathlib import Path
import base64, hashlib, io, json, sys, time, traceback, zipfile
STARTED = time.monotonic()
EXPECTED_PAYLOAD_SHA256 = __PAYLOAD_SHA__
EXPECTED_MANIFEST_HASH = __MANIFEST_SHA__
PAYLOAD_PATH = None  # optional exact mounted path; normally leave None
RESUME_RESULTS_ZIP = None  # optional THIS-package results ZIP, or its Kaggle-extracted directory
RUN_ROOT = Path('/kaggle/working/guandan_recheck375_v1')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def unpack(bundle, root, *, flat=False, limit=256*1024**2):
    members = bundle.infolist()
    if len({i.filename.casefold() for i in members}) != len(members) or sum(i.file_size for i in members) > limit:
        raise RuntimeError('duplicate/oversize input archive')
    for i in members:
        name = i.orig_filename
        target = (root/name).resolve()
        if name != i.filename or any(c in name for c in ('\\', ':', '\0')) or target == root.resolve() or not target.is_relative_to(root.resolve()) or ((i.external_attr >> 16) & 0o170000) == 0o120000:
            raise RuntimeError('unsafe ZIP member')
        if flat and ('/' in name or i.is_dir()):
            raise RuntimeError('flat input files required')
    bundle.extractall(root)

print('[RECHECK375] EVALUATION ONLY; authorized new training games = 0', flush=True)
try:
    if RUN_ROOT.exists():
        raise RuntimeError('RUN_ROOT already exists; preserve old outputs and choose a NEW RUN_ROOT/fresh session')
    candidates = [Path(PAYLOAD_PATH)] if PAYLOAD_PATH is not None else list(Path('/kaggle/input').rglob('guandan_recheck375_payload_v1.bin'))
    matches = [p for p in candidates if p.is_file() and not p.is_symlink() and p.stat().st_size <= 256*1024**2 and sha(p) == EXPECTED_PAYLOAD_SHA256]
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one matching recheck375 payload; do not mix continuation500/old pilot files: '+str(candidates))
    raw = matches[0].read_bytes()
    magic = b'GUANDAN-RECHECK375-EVALONLY-V1\n'
    if not raw.startswith(magic):
        raise RuntimeError('wrong opaque transport version')
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(raw[len(magic):].strip(), validate=True))) as bundle:
        meta = json.loads(bundle.read('delivery_manifest.json'))
        if meta['manifest_hash'] != EXPECTED_MANIFEST_HASH or canonical({k:v for k,v in meta.items() if k!='manifest_hash'}) != meta['manifest_hash']:
            raise RuntimeError('Notebook/payload manifest mismatch')
        if {i.filename for i in bundle.infolist()} != {'delivery_manifest.json', *(r['path'] for r in meta['files'])}:
            raise RuntimeError('undeclared payload files')
        for row in meta['files']:
            data = bundle.read(row['path'])
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise RuntimeError('input hash mismatch: '+row['path'])
        RUN_ROOT.mkdir(parents=True)
        dataset = RUN_ROOT/'input'
        dataset.mkdir()
        unpack(bundle, dataset, flat=True)
    driver = RUN_ROOT/'driver'
    driver.mkdir()
    with zipfile.ZipFile(dataset/'driver.zip') as bundle:
        unpack(bundle, driver, limit=4*1024**2)
    sys.path.insert(0, str(driver))
    from recheck375.pipeline import run_pipeline
    result = run_pipeline(dataset=dataset, run_root=RUN_ROOT, resume_results=RESUME_RESULTS_ZIP, started_at=STARTED)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    from IPython.display import FileLink, HTML, display
    display(HTML((RUN_ROOT/'artifacts/report.html').read_text(encoding='utf8')))
    display(FileLink(result['results_zip']))
    if result['status'] == 'budget_stopped':
        print('[RECHECK375] BUDGET_STOPPED: partial verified games saved; resume with RESUME_RESULTS_ZIP', flush=True)
    elif result['status'] != 'complete':
        raise RuntimeError('Stopped safely at '+str(result.get('failed_phase'))+': '+str(result.get('error')))
except BaseException as error:
    diagnostics = RUN_ROOT.parent/(RUN_ROOT.name+'_bootstrap')
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics/'bootstrap_error.json').write_text(json.dumps({'status':'failed', 'error':repr(error), 'traceback':traceback.format_exc(), 'training_updates':0}, ensure_ascii=False, indent=2), encoding='utf8')
    raise
"""


def build(repo, output, prior_results, base_inputs, validation_record=None):
    import torch
    repo, output = Path(repo).resolve(), Path(output).resolve()
    staging = output.with_name(output.name + "_build")
    if output.exists() or staging.exists():
        raise FileExistsError("never overwrite an old delivery")
    staging.mkdir(parents=True)
    dataset = staging / "dataset"
    dataset.mkdir()
    prior_results = Path(prior_results).resolve()
    with zipfile.ZipFile(prior_results) as prior:
        result = json.loads(prior.read("continuation_result.json"))
        if result["status"] != "complete" or result["training"]["actual_iteration"] != 500 or result["training"]["completed_games_total"] != 20000:
            raise ValueError("requires the completed 250->500 run")
        old_plan = json.loads(prior.read("plan.json"))
        if old_plan["base_source"] != SOURCE:
            raise ValueError("different original training core")
        model_files = {"p250": "anchor_iter250.pt", "p375": "model_iter000375.pt", "p500": "model_iter000500.pt"}
        models = {}
        backend = None
        for name, member in model_files.items():
            data = prior.read(member)
            if hashlib.sha256(data).hexdigest() != MODEL_SHAS[name]:
                raise ValueError("wrong user checkpoint: " + name)
            header = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            if header["schema"] != "guandan-marvel-evaluation-checkpoint-v1" or header.get("training_resume_allowed") is not False:
                raise ValueError("payload must contain evaluation-only policies, not training states")
            if header["iteration"] != int(name[1:]) or header["source_hash"] != SOURCE["core_source_hash"]:
                raise ValueError("wrong model iteration/source")
            if header["backend_identity"]["hash"] != BACKEND_HASH:
                raise ValueError("wrong original backend")
            if not all(torch.isfinite(t).all() for t in header["agent_state_dict"].values()):
                raise ValueError("nonfinite checkpoint")
            backend = header["backend_identity"]["implementation"]
            target = dataset / f"policy_{name[1:]}.pt"
            target.write_bytes(data)
            models[name] = {"path": target.name, "sha256": MODEL_SHAS[name], "iteration": header["iteration"],
                            "state_hash": state_hash(header["agent_state_dict"]), "training_resume_allowed": False}
        if backend["python_version"] != PYTHON_VERSION or str(Path(backend["extension_path"]).parent.parent).replace("\\", "/") != RUNTIME_PATH:
            raise ValueError("original absolute runtime/Python binding changed")
        recovery_bytes = prior.read("latest_recovery.zip")
        if hashlib.sha256(recovery_bytes).hexdigest() != result["recovery"]["sha256"]:
            raise ValueError("corrupt latest recovery")
        with zipfile.ZipFile(io.BytesIO(recovery_bytes)) as recovery:
            rec = json.loads(recovery.read("recovery_manifest.json"))
            binary = rec["runtime_binary"]
            native = recovery.read(binary["archive_path"])
            if hashlib.sha256(native).hexdigest() != binary["sha256"] or binary["sha256"] != backend["extension_sha256"]:
                raise ValueError("native binary differs from model identity")
            (dataset / "frozen_native.so").write_bytes(native)
        manifest = json.loads(prior.read("evaluation/final_500/manifest.json"))
        if manifest["hash"] != DEVELOPMENT_HASH:
            raise ValueError("different final development deals")
        raw_records = []
        for phase in ("anchor_final_baseline", "final_500"):
            for rank in (0, 1):
                raw_records.extend(json.loads(line) for line in prior.read(f"evaluation/{phase}/games.gpu{rank}.jsonl").splitlines() if line.strip())
        prior_summary = result["evaluations"]["final_500"]
    raw = (Path(base_inputs) / "v4_payload.bin").read_bytes()
    if hashlib.sha256(raw).hexdigest() != "0b018993dfa09528ed6c5dd225a825d9d7867a2f1df42963bdb1344ed5dd7347":
        raise ValueError("wrong original source input")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(raw.split(b"\n", 1)[1].strip(), validate=True))) as original:
        source_bytes = original.read("guandan-marvel-source.zip")
        if hashlib.sha256(source_bytes).hexdigest() != SOURCE["archive_sha256"]:
            raise ValueError("original source archive changed")
        (dataset / "frozen_source.zip").write_bytes(source_bytes)
        (dataset / "parent_provenance.json").write_bytes(original.read("parent_run.json"))
    atomic_json(dataset / "development_deals.json", manifest)
    atomic_json(dataset / "prior_endpoint_summary.json", prior_summary)
    (dataset / "cached_prior_games.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_records), encoding="utf8")
    source_files = {}
    driver_rows = []
    for path in sorted((repo / "tools/recheck375").glob("*.py")):
        if path.name == "delivery.py":
            continue
        data = path.read_bytes()
        ast.parse(data.decode("utf-8-sig"))
        name = "recheck375/" + path.name
        source_files[name] = data
        driver_rows.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (dataset / "driver.zip").write_bytes(zip_bytes(source_files))
    plan = {"schema": SCHEMA, "id": "same-set-recheck375-evalonly-20261022-v1", "test_only": False,
            "training_authorized": False, "new_training_games_cap": 0, "optimizer_updates_cap": 0,
            "protocol": PROTOCOL, "rotations": [0, 1], "horizon": 1024, "temperature": 0.0,
            "deal_count": 128, "development_hash": DEVELOPMENT_HASH, "source": SOURCE,
            "runtime_source_path": RUNTIME_PATH, "python": PYTHON_VERSION, "torch": TORCH_VERSION, "backend_hash": BACKEND_HASH,
            "models": models, "native": {"path": "frozen_native.so", "sha256": binary["sha256"], "restore_relative_path": binary["restore_relative_path"]},
            "prior_results_sha256": sha(prior_results), "prior_training_plan_hash": result["plan_hash"],
            "cached_arms": list(CACHED_ARMS), "new_arms": list(NEW_ARMS),
            "new_unique_strength_games_cap": 768, "cache_replay_games": 24, "smoke_new_games": 24,
            "wall_seconds_cap": 5400, "fabledan": "NOT_RUN", "final_holdout": "NOT_TOUCHED",
            "driver_archive_sha256": sha(dataset / "driver.zip"), "driver_files": driver_rows}
    plan["plan_hash"] = canonical(plan)
    validate_plan(plan)
    _, cache_audit = audit_cache(raw_records, manifest, plan, prior_summary)
    atomic_json(dataset / "build_cache_audit.json", cache_audit)
    atomic_json(dataset / "plan.json", plan)
    if validation_record is not None:
        validation = read_json(validation_record)
        for row in validation["source_files"]:
            if sha(repo / row["path"]) != row["sha256"]:
                raise ValueError("stale local validation: " + row["path"])
        atomic_json(dataset / "LOCAL_VALIDATION.json", validation)
    meta = {"schema": "guandan-recheck375-delivery-v1", "plan_hash": plan["plan_hash"], "training_authorized": False,
            "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(dataset.iterdir())]}
    meta["manifest_hash"] = canonical(meta)
    atomic_json(dataset / "delivery_manifest.json", meta)
    payload = MAGIC + base64.b64encode(zip_bytes({p.name: p.read_bytes() for p in dataset.iterdir()})) + b"\n"
    if len(payload) > 256 * 1024**2 or zipfile.is_zipfile(io.BytesIO(payload)):
        raise ValueError("invalid/oversize opaque upload")
    output.mkdir()
    (output / PAYLOAD).write_bytes(payload)
    code = BOOTSTRAP.replace("__PAYLOAD_SHA__", repr(sha(output / PAYLOAD))).replace("__MANIFEST_SHA__", repr(meta["manifest_hash"]))
    ast.parse(code)
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": False}},
                "cells": [{"id": "scope", "cell_type": "markdown", "metadata": {}, "source": [
                    "# iter375同集复测：只评估，不训练\n",
                    "Dataset只上传 `" + PAYLOAD + "`，GPU选择T4×2，Internet可关闭。\n",
                    "复用250/500终点结果，新增375对rule/250/500共768场唯一对局；另24场旧结果重放校验。\n",
                    "两个唯一队伍分配，逐arm双卡均衡。先缓存重放，再375小冒烟，全部通过才跑余下对局。\n",
                    "硬上限90分钟；出错/超时保留逐局进度，可用本包结果ZIP恢复未完成任务。无FableDan/final holdout。\n"]},
                          {"id": "evaluate-only", "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)}]}
    atomic_json(output / NOTEBOOK, notebook)
    shutil.copy2(repo / "docs/t4x2_recheck375.md", output / "README_KAGGLE.md")
    if validation_record is not None:
        shutil.copy2(dataset / "LOCAL_VALIDATION.json", output / "LOCAL_VALIDATION.json")
    verification = {"schema": "guandan-recheck375-verification-v1", "status": "BUILT_TARGET_PREFLIGHT_PENDING",
                    "payload": PAYLOAD, "payload_bytes": len(payload), "payload_sha256": sha(output / PAYLOAD),
                    "notebook": NOTEBOOK, "notebook_sha256": sha(output / NOTEBOOK), "manifest_hash": meta["manifest_hash"],
                    "plan_hash": plan["plan_hash"], "training_authorized": False, "new_training_games": 0,
                    "source_archive_sha256": SOURCE["archive_sha256"], "new_strength_games": 768, "validation_replays": 24}
    atomic_json(output / "DELIVERY_VERIFICATION.json", verification)
    return verification
