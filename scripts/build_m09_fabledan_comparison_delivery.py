# SPDX-License-Identifier: MIT
"""Build the M09 vs FableDan Kaggle comparison delivery (self-contained dataset)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.comparison.plan import (
    BOOTCAMP_FINAL_CANDIDATE_ARM, BOOTCAMP_FINAL_CONTROL_KEY, CONTROL_SCHEMA, DELIVERY_ID,
    build_comparison_deals, default_arms, expected_games,
)
from guandan_marvel.evaluation.common import payload_hash
from guandan_marvel.evaluation.manifests import dataset_identity, validate_deals_manifest
from guandan_marvel.kaggle.bootstrap import BOOTSTRAP_NAME, DATASET_MANIFEST_NAME, DATASET_MANIFEST_SCHEMA
from guandan_marvel.kaggle.common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from guandan_marvel.kaggle.discovery import validate_dataset_manifest
from guandan_marvel.kaggle.package import SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, build_source_archive, validate_source_archive

DEFAULT_SOURCE_COMMIT = "e5c82e7c27fb366800e9722c9e8e4720df7db01f"
CANDIDATE_MEMBER = "candidate/ippo_iter002000.pt"
CHAMPION_MEMBER = "champion/champions/generation000001_iter000100.pt"
CANDIDATE_SHA256 = "af0df1eaeb67c88e467ee78bf6b0494c49d000f5733d655ee81b40351d4ca9c8"
CHAMPION_SHA256 = "b00a2c178e8b3e99d80418ae283a785285639b8172a0d96232beef7233b9d41c"
CANDIDATE_TARGET = "m09_candidate_iter002000.pt"
CHAMPION_TARGET = "m09_champion_iter000100.pt"
BOOTCAMP_FINAL_TARGET = "bootcamp_final_policy.pt"
DEALS_NAME = "comparison_deals_manifest.json"
CONTROL_NAME = "comparison_control.json"
FABLEDAN_WEIGHTS_NAME = "fabledan_weights.npz"
NOTEBOOK_NAME = "guandan_marvel_comparison.ipynb"
UPLOAD_NAME = "m09-fabledan-comparison-upload.zip"
SOFT_TIME_LIMIT_SECONDS = 30600
VENDOR_FILES = ("cards.py", "combos.py", "encode.py", "model_np.py")
OVERLAY_INIT = "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
VENDOR_INIT = '"""Vendored FableDan NumPy inference modules (built into the overlay)."""\n'
DETERMINISTIC_CREATED_AT = "2026-10-16T00:00:00+00:00"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deterministic_source_archive(repository: Path, destination: Path, *, commit: str) -> tuple[dict[str, object], str]:
    """Build the source archive with pinned timestamps so rebuilds are byte-identical."""
    result = build_source_archive(repository, destination, commit=commit, allow_dirty=True)
    manifest = dict(result["source_manifest"])
    manifest["created_at"] = DETERMINISTIC_CREATED_AT
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = canonical_hash(body)
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(destination) as source_archive, zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as target_archive:
        for info in sorted(source_archive.infolist(), key=lambda item: item.filename):
            data = source_archive.read(info.filename)
            if info.filename == "guandan-marvel-source/source_manifest.json":
                data = manifest_bytes
            entry = zipfile.ZipInfo(info.filename, date_time=(2026, 10, 10, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            target_archive.writestr(entry, data)
    with zipfile.ZipFile(temporary) as archive:
        if bad := archive.testzip():
            raise ValueError(f"source archive CRC failure: {bad}")
    temporary.replace(destination)
    validate_source_archive(destination)
    return manifest, sha256_file(destination)


def _extract_member(archive: Path, member: str, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        data = bundle.read(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


NOTEBOOK = r'''from pathlib import Path
import hashlib,importlib.util,json,os,shutil,subprocess,sys,threading,time,traceback,zipfile
from IPython.display import FileLink,display
INPUT=Path('/kaggle/input');WORKING=Path('/kaggle/working');ERROR=None;EXIT=1;START=time.monotonic();STOP=threading.Event();STAGE={'name':'startup'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def stage(name,**extra):
 STAGE['name']=name;print(json.dumps({'event':'comparison_bootstrap_stage','stage':name,'elapsed_seconds':round(time.monotonic()-START,3),**extra},sort_keys=True),flush=True)
def heartbeat():
 while not STOP.wait(10):print(json.dumps({'event':'comparison_bootstrap_heartbeat','stage':STAGE['name'],'elapsed_seconds':round(time.monotonic()-START,3)},sort_keys=True),flush=True)
def stream(cmd,*,cwd,env,label):
 print(f'[{label}] RUN:',cmd,flush=True);p=subprocess.Popen([str(x) for x in cmd],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 for line in p.stdout:print(f'[{label}] '+line.rstrip(),flush=True)
 code=p.wait();print(f'[{label}] exit={code}',flush=True);return code
def parallel(jobs,*,cwd):
 procs=[]
 for cmd,jenv,label in jobs:
  print(f'[{label}] RUN:',cmd,flush=True)
  procs.append((subprocess.Popen([str(x) for x in cmd],cwd=cwd,env=jenv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1),label))
 def pump(proc,label):
  for line in proc.stdout:print(f'[{label}] '+line.rstrip(),flush=True)
 threads=[threading.Thread(target=pump,args=(proc,label),daemon=True) for proc,label in procs]
 for t in threads:t.start()
 for t in threads:t.join()
 return [proc.wait() for proc,_ in procs]
threading.Thread(target=heartbeat,daemon=True).start();stage('dataset_discovery')
try:
 controls=sorted(INPUT.rglob('comparison_control.json'))
 if len(controls)!=1:raise RuntimeError(f'exactly one comparison control required: {controls}')
 control_path=controls[0].resolve();base=control_path.parent;control=json.loads(control_path.read_text(encoding='utf-8'))
 import hashlib as _h
 body={k:v for k,v in control.items() if k!='hash'}
 if control.get('schema')!='guandan-marvel-comparison-control-v1' or control.get('hash')!=_h.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest():raise RuntimeError('comparison control hash mismatch')
 stage('control_validation',files=len(control['files']))
 mismatches=[]
 for row in control['files']:
  p=(base/row['path']).resolve()
  if row['path']=='guandan-marvel-source.zip' and not p.is_file():
   candidates=sorted(c for c in base.rglob('source_manifest.json') if c.is_file() and c.parent.resolve()!=base)
   if candidates:
    print(json.dumps({'event':'comparison_control_note','path':row['path'],'found':[str(c.relative_to(base)) for c in candidates[:3]],'note':'source zip absent but an unpacked source manifest exists; the bootstrap validates the directory input instead'},sort_keys=True),flush=True)
    continue
  actual_bytes=p.stat().st_size if p.is_file() else None
  actual_sha=sha(p) if p.is_file() else None
  if not p.is_relative_to(base) or not p.is_file() or actual_bytes!=row['bytes'] or actual_sha!=row['sha256']:
   mismatches.append({'path':row['path'],'expected_bytes':row['bytes'],'expected_sha256':row['sha256'],'actual_bytes':actual_bytes,'actual_sha256':actual_sha,'is_file':p.is_file(),'is_dir':p.is_dir()})
 if mismatches:
  print(json.dumps({'event':'comparison_control_mismatch_report','control_created_at':control.get('created_at'),'count':len(mismatches),'items':mismatches[:10]},sort_keys=True,indent=1),flush=True)
  raise RuntimeError('comparison control file mismatch: recreate the Kaggle dataset entirely from the current m09-fabledan-comparison-upload.zip; do not mix files from older deliveries')
 manifests=sorted(INPUT.rglob('guandan_marvel_dataset.json'))
 if len(manifests)!=1:raise RuntimeError(f'exactly one dataset manifest required (attach only the comparison dataset): {manifests}')
 dataset_path=manifests[0].resolve();dataset=json.loads(dataset_path.read_text(encoding='utf-8'))
 bootstrap_spec=dataset['bootstrap'];bootstrap_path=(dataset_path.parent/bootstrap_spec['path']).resolve()
 if not bootstrap_path.is_file() or sha(bootstrap_path)!=bootstrap_spec['sha256']:raise RuntimeError('dataset bootstrap hash mismatch')
 spec=importlib.util.spec_from_file_location('comparison_bootstrap',bootstrap_path);boot=importlib.util.module_from_spec(spec);sys.modules[spec.name]=boot;spec.loader.exec_module(boot)
 empty=WORKING/'comparison-empty-input';empty.mkdir(exist_ok=True);stage('source_materialization')
 prepared=boot.prepare_inputs(dataset_path,input_root=empty,working_root=WORKING);source=Path(prepared['source_root'])
 overlay=WORKING/'comparison-overlay'
 if overlay.exists():shutil.rmtree(overlay)
 shutil.copytree(base/'overlay',overlay)
 env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONPATH']=str(overlay)+os.pathsep+str(source);env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
 ext=sorted((source/'nanopokerlib').glob('_guandan_cpp*.so'))+sorted((source/'nanopokerlib').glob('_guandan_cpp*.pyd'))
 if not ext:
  stage('native_build')
  if stream([sys.executable,'-u','setup.py','build_ext','--inplace'],cwd=source,env=env,label='native-build')!=0:raise RuntimeError('native build failed')
 stage('native_import')
 if stream([sys.executable,'-u','-c','import guandan_marvel.cpp_game as g;print(g.get_backend_build_identity(),flush=True)'],cwd=source,env=env,label='native-import')!=0:raise RuntimeError('native import failed')
 stage('adapter_validation')
 if stream([sys.executable,'-u',str(overlay/'scripts/compare_validate_adapter.py'),'--control',str(control_path),'--working-root',str(WORKING),'--device','cuda:0'],cwd=source,env=env,label='adapter-validation')!=0:raise RuntimeError('adapter validation failed; refusing to run the comparison')
 stage('comparison_workers')
 workers=int(control.get('workers',2));soft=float(control.get('soft_time_limit_seconds',0));jobs=[]
 try:
  import torch as _torch
  gpu_count=max(1,int(_torch.cuda.device_count()))
 except Exception:
  gpu_count=1
 print(json.dumps({'event':'comparison_worker_plan','workers':workers,'gpus':gpu_count,'soft_time_limit_seconds':soft},sort_keys=True),flush=True)
 for index in range(workers):
  worker_env=dict(env);worker_env['CUDA_VISIBLE_DEVICES']=str(index%gpu_count)
  jobs.append(([sys.executable,'-u',str(overlay/'scripts/compare_run_worker.py'),'--control',str(control_path),'--working-root',str(WORKING),'--worker-index',str(index),'--workers',str(workers),'--device','cuda:0','--soft-time-limit',str(soft)],worker_env,f'worker{index}'))
 codes=parallel(jobs,cwd=source)
 if any(code!=0 for code in codes):raise RuntimeError(f'comparison workers failed: {codes}')
 stage('summary')
 summary_env=dict(env);summary_env['CUDA_VISIBLE_DEVICES']=''
 if stream([sys.executable,'-u',str(overlay/'scripts/compare_summarize.py'),'--control',str(control_path),'--working-root',str(WORKING)],cwd=source,env=summary_env,label='summary')!=0:raise RuntimeError('comparison summary failed')
 STOP.set();EXIT=0
except BaseException as e:
 STOP.set();ERROR=f'{type(e).__name__}: {e}';print(ERROR,flush=True);traceback.print_exc()
OUTPUT=WORKING/'m09-fabledan-comparison-output.zip'
if OUTPUT.is_file():display(FileLink(str(OUTPUT)))
if ERROR is not None:raise RuntimeError(ERROR)
if EXIT!=0:raise RuntimeError('comparison run did not complete')
'''


def render_notebook() -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "accelerator": "GPU",
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# M09 vs FableDan — same-arena comparison (T4 x2)\n",
                "一次性运行：256 副无进贡牌的主对局 + 锚点臂（同批牌）。适配器校验不通过会拒绝开跑。\n",
                "默认 4 个 worker（每张 T4 两个进程）以重叠 GPU 推理与 FableDan CPU 推理；预计 2–3 小时。\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": NOTEBOOK.splitlines(keepends=True)},
        ],
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    repository = args.repository.resolve()
    m09_archive = args.m09_archive.resolve()
    fabledan_weights = args.fabledan_weights.resolve()
    fabledan_src = args.fabledan_src.resolve()
    if output.exists():
        shutil.rmtree(output)
    dataset = output / "dataset"
    dataset.mkdir(parents=True)

    source_manifest, source_archive_sha256 = _deterministic_source_archive(
        repository, dataset / SOURCE_ARCHIVE_NAME, commit=args.source_commit,
    )
    (dataset / SOURCE_MANIFEST_NAME).write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    with zipfile.ZipFile(dataset / SOURCE_ARCHIVE_NAME) as archive:
        (dataset / BOOTSTRAP_NAME).write_bytes(archive.read("guandan-marvel-source/guandan_marvel/kaggle/bootstrap.py"))

    candidate_path = dataset / CANDIDATE_TARGET
    champion_path = dataset / CHAMPION_TARGET
    _extract_member(m09_archive, CANDIDATE_MEMBER, candidate_path)
    _extract_member(m09_archive, CHAMPION_MEMBER, champion_path)
    if sha256_file(candidate_path) != CANDIDATE_SHA256:
        raise ValueError("extracted candidate checkpoint hash differs from the frozen M09 candidate")
    if sha256_file(champion_path) != CHAMPION_SHA256:
        raise ValueError("extracted champion checkpoint hash differs from the frozen M09 champion")

    bootcamp_final_spec = None
    if args.bootcamp_final_checkpoint is not None:
        bootcamp_source = args.bootcamp_final_checkpoint.resolve()
        if not bootcamp_source.is_file():
            raise FileNotFoundError(f"bootcamp final checkpoint not found: {bootcamp_source}")
        bootcamp_payload = torch.load(bootcamp_source, map_location="cpu", weights_only=False)
        if not isinstance(bootcamp_payload, dict) or bootcamp_payload.get("schema") != "guandan-marvel-evaluation-checkpoint-v1":
            raise ValueError("bootcamp final must be a guandan-marvel-evaluation-checkpoint-v1 artifact")
        bootcamp_target = dataset / BOOTCAMP_FINAL_TARGET
        shutil.copy2(bootcamp_source, bootcamp_target)
        bootcamp_final_spec = {
            "path": BOOTCAMP_FINAL_TARGET,
            "bytes": bootcamp_target.stat().st_size,
            "sha256": sha256_file(bootcamp_target),
            "iteration": int(bootcamp_payload.get("iteration", -1)),
        }

    shutil.copy2(fabledan_weights, dataset / FABLEDAN_WEIGHTS_NAME)
    vendor = dataset / "overlay" / "fabledan_vendor"
    vendor.mkdir(parents=True)
    (vendor / "__init__.py").write_text(VENDOR_INIT, encoding="utf-8", newline="\n")
    for name in VENDOR_FILES:
        shutil.copy2(fabledan_src / name, vendor / name)

    comparison_pkg = dataset / "overlay" / "guandan_marvel" / "comparison"
    comparison_pkg.mkdir(parents=True)
    (dataset / "overlay" / "guandan_marvel" / "__init__.py").write_text(OVERLAY_INIT, encoding="utf-8", newline="\n")
    for path in sorted((repository / "guandan_marvel" / "comparison").glob("*.py")):
        shutil.copy2(path, comparison_pkg / path.name)
    scripts_dir = dataset / "overlay" / "scripts"
    scripts_dir.mkdir(parents=True)
    for name in ("compare_validate_adapter.py", "compare_run_worker.py", "compare_summarize.py"):
        shutil.copy2(repository / "scripts" / name, scripts_dir / name)

    deals = build_comparison_deals()
    deals["created_at"] = DETERMINISTIC_CREATED_AT
    deals["hash"] = payload_hash(deals)
    validate_deals_manifest(deals)
    deals_path = dataset / DEALS_NAME
    atomic_json(deals_path, deals)

    arms = default_arms(with_bootcamp_final=bootcamp_final_spec is not None)
    total_games = sum(int(arm["deals"]) * 4 for arm in arms)

    dataset_manifest_body = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "package_id": str(source_manifest["package_id"]),
        "mode": "preflight",
        "created_at": DETERMINISTIC_CREATED_AT,
        "source": {
            "archive": SOURCE_ARCHIVE_NAME,
            "archive_sha256": source_archive_sha256,
            "manifest": SOURCE_MANIFEST_NAME,
            "manifest_sha256": sha256_file(dataset / SOURCE_MANIFEST_NAME),
            "git_commit": str(source_manifest["git_commit"]),
            "content_manifest_hash": str(source_manifest["manifest_hash"]),
        },
        "bootstrap": {"path": BOOTSTRAP_NAME, "sha256": sha256_file(dataset / BOOTSTRAP_NAME)},
        "recovery": {"required": False, "auto_discovery": False},
        "profiles": {
            "preflight": {
                "schema": "guandan-marvel-kaggle-profile-v1",
                "mode": "preflight",
                "required_gpu_count": 2,
                "required_gpu_name_contains": "T4",
                "telemetry_interval_seconds": 5,
            },
        },
        "claims": {
            "remote_ready": True,
            "remote_verified": False,
            "informal_comparison": True,
            "formal_acceptance": False,
        },
    }
    atomic_json(dataset / DATASET_MANIFEST_NAME, dataset_manifest_body)

    control_files = [
        path for path in sorted(dataset.rglob("*"))
        if path.is_file() and path.name != CONTROL_NAME
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    control_body = {
        "schema": CONTROL_SCHEMA,
        "created_at": DETERMINISTIC_CREATED_AT,
        "comparison_id": DELIVERY_ID,
        "scope": {
            "arena": "guandan-marvel C++ protocol GD-ACTION/ENV/ENCODING-0.2-MARVEL",
            "tribute": "none (all deals use previous_result=None)",
            "temperature": 0.0,
            "guest_candidate_scope": "FableDan gen_moves intersection only",
            "informal": True,
            "primary_endpoint": BOOTCAMP_FINAL_CANDIDATE_ARM if bootcamp_final_spec is not None else "h2h",
        },
        "source_identity": {
            "git_commit": str(source_manifest["git_commit"]),
            "package_id": str(source_manifest["package_id"]),
            "source_archive_sha256": source_archive_sha256,
            "source_manifest_sha256": sha256_file(dataset / SOURCE_MANIFEST_NAME),
            "source_content_manifest_hash": str(source_manifest["manifest_hash"]),
        },
        "candidate": {
            "path": CANDIDATE_TARGET, "bytes": candidate_path.stat().st_size,
            "sha256": CANDIDATE_SHA256, "iteration": 2000,
        },
        "champion": {
            "path": CHAMPION_TARGET, "bytes": champion_path.stat().st_size,
            "sha256": CHAMPION_SHA256, "iteration": 100,
        },
        "fabledan_weights": {
            "path": FABLEDAN_WEIGHTS_NAME, "bytes": (dataset / FABLEDAN_WEIGHTS_NAME).stat().st_size,
            "sha256": sha256_file(dataset / FABLEDAN_WEIGHTS_NAME),
        },
        "deals_manifest": {
            "path": DEALS_NAME, "bytes": deals_path.stat().st_size, "sha256": sha256_file(deals_path),
            "manifest_hash": deals["hash"], "dataset_hash": dataset_identity(deals, "final"),
            "deal_count": len(deals["datasets"]["final"]), "seed_base": 1600000000,
        },
        "arms": arms,
        "temperature": 0.0,
        "workers": 4,
        "heartbeat_seconds": 20,
        "soft_time_limit_seconds": SOFT_TIME_LIMIT_SECONDS,
        "worker_deep_verify": False,
        "validation": {"deals": 2, "arms": ["h2h", "fabledan_vs_rule"]},
        "total_games": total_games,
        "files": relative_file_manifest(dataset, control_files),
        "claims": {
            "informal_comparison": True,
            "formal_acceptance": False,
            "tribute_free_scope": True,
            "pre_registered_plan": True,
        },
    }
    if bootcamp_final_spec is not None:
        control_body[BOOTCAMP_FINAL_CONTROL_KEY] = bootcamp_final_spec
    control = {**control_body, "hash": payload_hash(control_body)}
    control_path = dataset / CONTROL_NAME
    atomic_json(control_path, control)

    validate_dataset_manifest(dataset_manifest_body, manifest_path=dataset / DATASET_MANIFEST_NAME)

    atomic_json(output / NOTEBOOK_NAME, render_notebook())
    (output / "README_COMPARISON.md").write_text(
        "# M09 vs FableDan comparison delivery\n\n"
        "1. Create one private Kaggle Dataset from `dataset/` (keep file names). Never mix files from older deliveries.\n"
        "   Rebuilds of this delivery are byte-deterministic: the control hash is the version fingerprint.\n"
        "2. Import `guandan_marvel_comparison.ipynb`, choose **GPU T4 x2**, attach only this Dataset, Run All once.\n"
        "3. The adapter validation stage must pass before the comparison starts; the run is sharded and resumable.\n"
        "4. Four workers (two per T4) overlap GPU inference with FableDan CPU inference; watch periodic heartbeats.\n"
        "5. Download `m09-fabledan-comparison-output.zip` from the notebook output.\n\n"
        f"Total planned games: {total_games} (tribute-free deals only).\n",
        encoding="utf-8", newline="\n",
    )

    upload = output / UPLOAD_NAME
    temporary = upload.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(dataset.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                info = zipfile.ZipInfo(path.relative_to(dataset).as_posix(), date_time=(2026, 10, 10, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(temporary) as archive:
        if bad := archive.testzip():
            raise ValueError(f"upload CRC failure: {bad}")
    temporary.replace(upload)

    manifest = {
        "schema": "guandan-marvel-comparison-delivery-v1",
        "delivery_id": DELIVERY_ID,
        "created_at": _timestamp(),
        "dataset": "dataset",
        "notebook": NOTEBOOK_NAME,
        "control_hash": control["hash"],
        "candidate_sha256": CANDIDATE_SHA256,
        "champion_sha256": CHAMPION_SHA256,
        "bootcamp_final_sha256": None if bootcamp_final_spec is None else bootcamp_final_spec["sha256"],
        "fabledan_weights_sha256": control["fabledan_weights"]["sha256"],
        "deals_dataset_hash": control["deals_manifest"]["dataset_hash"],
        "total_games": total_games,
        "soft_time_limit_seconds": SOFT_TIME_LIMIT_SECONDS,
        "upload": {"path": upload.name, "bytes": upload.stat().st_size, "sha256": sha256_file(upload)},
        "claims": {"informal_comparison": True, "formal_acceptance": False},
    }
    manifest["hash"] = canonical_hash(manifest)
    atomic_json(output / "delivery_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--m09-archive", type=Path, required=True)
    parser.add_argument("--fabledan-weights", type=Path, required=True)
    parser.add_argument("--fabledan-src", type=Path, required=True)
    parser.add_argument("--bootcamp-final-checkpoint", type=Path, default=None)
    parser.add_argument("--source-commit", type=str, default=DEFAULT_SOURCE_COMMIT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
