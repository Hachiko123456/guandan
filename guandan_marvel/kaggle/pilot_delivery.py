# SPDX-License-Identifier: MIT
"""Frozen, offline-capable T4x2 50-iteration pilot delivery (no training at build time)."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import shutil
import zipfile
from .common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from .package import build_source_archive, validate_source_archive, SOURCE_ARCHIVE_NAME
from .phase4_delivery import _training_block, validate_production_warm_start

PILOT_SCHEMA = "guandan-marvel-t4x2-learning-pilot-v1"
PILOT_ID = "guandan-marvel-v14-t4x2-pilot-20261021"
MANIFEST_NAME = "pilot_manifest.json"
CAP_ITERATIONS = 50
COLLECT_GAMES = 20
GLOBAL_WALL_SECONDS = 12600


def make_profiles():
    training = deepcopy(_training_block())
    training.update({"rollout_batch_size": 16, "collect_batch_size": COLLECT_GAMES,
                     "trim_completed_padding": True, "target_iterations": 2,
                     "checkpoint_interval": 10, "evaluation_interval": 0,
                     "max_wall_seconds": 1800, "max_wall_seconds_per_session": 1800,
                     "checkpoint_keep_last": 1, "evaluation_keep_last": 1})
    # Hyperparameters/architecture are intentionally NOT the unverified 14-item proposal.
    common = {"name": PILOT_ID, "algorithm": "ippo", "mode": "full",
              "ddp": {"world_size": 2, "backend": "nccl", "require_cuda": True,
                      "find_unused_parameters": False},
              "heartbeat_interval_seconds": 5, "telemetry_interval_seconds": 5,
              "pilot_observability": {"enabled": True, "cpu_threads": 1, "maximum_iteration": 50}}
    start = {**deepcopy(common), "training": training}
    profiles = {"precheck": start}
    for name, target in (("resume_check", 3), ("train", CAP_ITERATIONS)):
        profiles[name] = {**deepcopy(common), "mode": "resume", "expected_algorithm": "ippo",
                          "resume": {"target_iterations": target, "checkpoint_interval": 10,
                                     "evaluation_interval": 0, "checkpoint_keep_last": 1,
                                     "evaluation_keep_last": 1, "max_wall_seconds": 10800}}
    return profiles


def validate_manifest(root: str | Path):
    root = Path(root).resolve()
    d = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if d.get("schema") == "guandan-marvel-t4x2-combo-pilot-v1":
        from .combo_delivery import validate_combo_manifest
        return validate_combo_manifest(root)
    if d.get("schema") != PILOT_SCHEMA:
        raise ValueError("unsupported pilot manifest")
    if d.get("manifest_hash") != canonical_hash({k: v for k, v in d.items() if k != "manifest_hash"}):
        raise ValueError("pilot manifest hash mismatch")
    for row in d["files"]:
        raw_path = root / str(row["path"])
        if raw_path.parent != root or raw_path.is_symlink():
            raise ValueError("pilot inputs must be flat regular files")
        path = raw_path.resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError("unsafe/missing pilot input")
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"pilot input content mismatch: {row['path']}")
    from guandan_marvel.evaluation.manifests import validate_deals_manifest, deals_for
    evaluation = json.loads((root / "development_deals.json").read_text(encoding="utf-8"))
    validate_deals_manifest(evaluation)
    if (evaluation["hash"] != d["evaluation_manifest_hash"] or len(deals_for(evaluation, "selection")) != 32
            or deals_for(evaluation, "training") or deals_for(evaluation, "final")
            or d["expected_evaluation_games"] != 384):
        raise ValueError("pilot development/final dataset contract mismatch")
    source = d["source"]
    if sha256_file(root / SOURCE_ARCHIVE_NAME) != source["archive_sha256"]:
        raise ValueError("source identity mismatch")
    if d["training_cap"] != {"absolute_iterations": 50, "completed_games": 1000, "global_collect_batch_size": 20}:
        raise ValueError("pilot training cap must stay 50 iterations / 1000 actual games")
    if d["wall_seconds_cap"] != GLOBAL_WALL_SECONDS:
        raise ValueError("unexpected global wall budget")
    profiles = {name: json.loads((root / filename).read_text(encoding="utf-8")) for name, filename in d["profiles"].items()}
    if profiles != make_profiles():
        raise ValueError("pilot profiles differ from the versioned baseline/performance recipe")
    from guandan_marvel.buffer import trajectory_memory_budget
    from guandan_marvel.game import GAME_SPEC
    budget = trajectory_memory_budget(GAME_SPEC, rollout_batch_size=16, collect_batch_size=20, world_size=2)
    if budget != d["sampler_budget"]:
        raise ValueError("sampler budget drift")
    return d


BOOTSTRAP = r"""from pathlib import Path
import hashlib, json, os, subprocess, sys, time, zipfile
PILOT_STARTED = time.monotonic()
EXPECTED_PILOT_MANIFEST_HASH = None
# Bundled build dependencies, no network setup required. Kaggle must provide two T4 GPUs.
DATASET_DIR = None  # optionally set Path('/kaggle/input/your-dataset')
RECOVERY_ZIP = None  # optional path to matching prior pilot guandan-marvel-recovery.zip
RUN_ROOT = Path('/kaggle/working/guandan_t4x2_pilot')
if DATASET_DIR is None:
    candidates = list(Path('/kaggle/input').rglob('pilot_manifest.json'))
    if len(candidates) == 0:
        archives = list(Path('/kaggle/input').rglob('pilot-upload.zip'))
        if len(archives) == 1:
            unpack = RUN_ROOT.parent/(RUN_ROOT.name+'_dataset')
            if unpack.exists(): raise RuntimeError(f'{unpack} already exists; choose a new RUN_ROOT')
            unpack.mkdir(parents=True)
            with zipfile.ZipFile(archives[0]) as z:
                for info in z.infolist():
                    if '/' in info.orig_filename or chr(92) in info.orig_filename or ':' in info.orig_filename or info.orig_filename in ('.','..'):
                        raise RuntimeError('Expected a flat pilot upload ZIP with safe names')
                    if ((info.external_attr>>16)&0o170000)==0o120000: raise RuntimeError('Symlink in input ZIP')
                z.extractall(unpack)
            candidates=[unpack/'pilot_manifest.json']
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one pilot Dataset, found {candidates}; set DATASET_DIR explicitly')
    DATASET_DIR = candidates[0].parent
DATASET_DIR = Path(DATASET_DIR)
meta = json.loads((DATASET_DIR/'pilot_manifest.json').read_text())
if EXPECTED_PILOT_MANIFEST_HASH is not None and meta['manifest_hash'] != EXPECTED_PILOT_MANIFEST_HASH:
    raise RuntimeError('This Notebook and Dataset belong to different pilot builds; attach the matching input')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
for row in meta['files']:
    p=(DATASET_DIR/row['path']).resolve()
    if not p.is_relative_to(DATASET_DIR.resolve()) or not p.is_file() or sha(p)!=row['sha256']:
        raise RuntimeError(f'Input identity failure: {row}')
body={k:v for k,v in meta.items() if k!='manifest_hash'}
actual=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
if actual!=meta['manifest_hash']: raise RuntimeError('Pilot manifest hash mismatch')
if RUN_ROOT.exists(): raise RuntimeError(f'{RUN_ROOT} already exists; preserve/download it, choose a NEW RUN_ROOT')
RUN_ROOT.mkdir(parents=True)
source_parent=RUN_ROOT/'source'
with zipfile.ZipFile(DATASET_DIR/'guandan-marvel-source.zip') as z:
    for info in z.infolist():
        p=(source_parent/info.filename).resolve()
        if not p.is_relative_to(source_parent.resolve()) or chr(92) in info.orig_filename or ':' in info.orig_filename or ((info.external_attr>>16)&0o170000)==0o120000:
            raise RuntimeError('Unsafe source member')
    z.extractall(source_parent)
source=source_parent/'guandan-marvel-source'
sys.path.insert(0,str(source))
os.environ['PYTHONPATH']=str(source)
os.environ['PYTHONUNBUFFERED']='1'
os.environ['GUANDAN_MARVEL_SOURCE_COMMIT']=meta['source']['git_commit']
# One CPU thread per GPU rank prevents BLAS/OpenMP oversubscription.
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['NCCL_DEBUG']='INFO'
os.environ['TORCH_DISTRIBUTED_DEBUG']='INFO'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1'
if sys.version_info < (3, 12): raise RuntimeError('This source requires Python >= 3.12; do not install into an incompatible runtime silently')
import torch
if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
    raise RuntimeError('Select GPU T4 x2 before running this Notebook')
if RECOVERY_ZIP is not None:
    from guandan_marvel.kaggle.pilot import restore_runtime_from_recovery
    restore_runtime_from_recovery(Path(RECOVERY_ZIP), source, meta, RUN_ROOT/'recovery_input')
else:
    with (RUN_ROOT/'native_build.log').open('w') as log:
        proc=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=source,stdout=log,stderr=subprocess.STDOUT,timeout=660)
    print((RUN_ROOT/'native_build.log').read_text()[-10000:])
    if proc.returncode: raise RuntimeError('Native build failed; see native_build.log')
from guandan_marvel.kaggle.pilot import run_pilot
result=run_pilot(dataset=DATASET_DIR, output=RUN_ROOT/'artifacts', source=source,
                 recovery=None if RECOVERY_ZIP is None else RUN_ROOT/'recovery_input', started_at=PILOT_STARTED)
print(json.dumps(result,ensure_ascii=False,indent=2))
from IPython.display import display, HTML, FileLink
display(HTML((RUN_ROOT/'artifacts'/'report.html').read_text()))
display(FileLink(str(RUN_ROOT/'guandan_t4x2_pilot_results.zip')))
if result['status'] == 'failed': raise RuntimeError(result.get('error'))
"""


def render_notebook(expected_manifest_hash: str | None = None):
    code = BOOTSTRAP.replace("EXPECTED_PILOT_MANIFEST_HASH = None", f"EXPECTED_PILOT_MANIFEST_HASH = {expected_manifest_hash!r}", 1)
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": False}},
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "id": "instructions", "source": [
                    "# 掼蛋 v1.4 T4×2 短训实验\n",
                    "先把 pilot-upload.zip 上传为 Kaggle Dataset，再导入此 Notebook 并 Add Input。\n",
                    "Settings 中明确选择 **GPU T4 x2**；Internet 可关闭。Run All 执行 2 iter 预检、恢复到 3，再继续到绝对 50 iter（总计 1000 局），最后用双卡对照评估。\n",
                    "总预算 3.5 小时，保存恢复与评估预留时间；不足时保留可恢复状态，不伪称完成。不会继续到 250 iter，不使用 final holdout，不对 FableDan 宣称胜出。\n",
                    "关闭或异常前下载 outputs，至少保留结果 ZIP。\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "id": "run-pilot", "source": code.splitlines(keepends=True)}]}


def build_delivery(repository: str | Path, output: str | Path, warm_start: str | Path, *, validation_record: str | Path | None = None):
    import torch
    from guandan_marvel.buffer import trajectory_memory_budget
    from guandan_marvel.game import GAME_SPEC
    from guandan_marvel.evaluation.manifests import build_deals_manifest
    repo, out, warm = Path(repository).resolve(), Path(output).resolve(), Path(warm_start).resolve()
    if out.exists():
        raise FileExistsError("choose a new output path; historical deliveries are never overwritten")
    if not warm.is_file():
        raise FileNotFoundError(warm)
    identity = validate_production_warm_start(torch.load(warm, map_location="cpu", weights_only=False), source=str(warm))
    dataset = out / "dataset"
    dataset.mkdir(parents=True)
    source = build_source_archive(repo, dataset / SOURCE_ARCHIVE_NAME, allow_dirty=True)
    validate_source_archive(dataset / SOURCE_ARCHIVE_NAME)
    shutil.copy2(warm, dataset / "warm_start_policy.pt")
    for name, profile in make_profiles().items():
        atomic_json(dataset / f"{name}_profile.json", profile)
    evaluation = build_deals_manifest(counts={"training": 0, "selection": 32, "final": 0},
                                     bases={"training": 0, "selection": 1_800_210_000, "final": 2_000_000_000},
                                     level_ranks=tuple(range(3,16)), leader_seats=(0,1,2,3), previous_results=(None,))
    atomic_json(dataset / "development_deals.json", evaluation)
    if validation_record is not None:
        validation = json.loads(Path(validation_record).read_text(encoding="utf-8"))
        for row in validation.get("source_files", []):
            candidate = (repo / row["path"]).resolve()
            if not candidate.is_relative_to(repo) or sha256_file(candidate) != row["sha256"]:
                raise ValueError(f"local validation record does not describe current source: {row['path']}")
        validation["delivery_source_archive_sha256"] = source["archive_sha256"]
        validation["kaggle_t4x2_executed"] = False
        atomic_json(dataset / "LOCAL_VALIDATION.json", validation)
        atomic_json(out / "LOCAL_VALIDATION.json", validation)
    budget = trajectory_memory_budget(GAME_SPEC, rollout_batch_size=16, collect_batch_size=20, world_size=2)
    body = {"schema": PILOT_SCHEMA, "pilot_id": PILOT_ID,
            "training_cap": {"absolute_iterations": 50, "completed_games": 1000, "global_collect_batch_size": 20},
            "wall_seconds_cap": GLOBAL_WALL_SECONDS, "evaluation_reserve_seconds": 3600,
            "source": {"archive_sha256": source["archive_sha256"],
                       "content_manifest_hash": source["source_manifest"]["manifest_hash"],
                       "git_commit": source["source_manifest"]["git_commit"],
                       "workspace_snapshot": source["source_manifest"]["workspace_snapshot"]},
            "warm_start": {"file": "warm_start_policy.pt", "sha256": sha256_file(warm), "identity": identity},
            "profiles": {name: f"{name}_profile.json" for name in make_profiles()},
            "sampler_budget": budget, "evaluation_manifest_hash": evaluation["hash"],
            "expected_evaluation_games": 32*4*3, "evaluation_role": "development_only",
            "fabledan": "NOT_RUN", "final_holdout": "NOT_TOUCHED",
            "files": relative_file_manifest(dataset, [p for p in dataset.iterdir() if p.is_file()]),
            "remote_verified": False}
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    atomic_json(dataset / MANIFEST_NAME, manifest)
    validate_manifest(dataset)
    atomic_json(out / "guandan_t4x2_pilot.ipynb", render_notebook(manifest["manifest_hash"]))
    guide = repo / "docs/t4x2_learning_pilot.md"
    if guide.is_file():
        shutil.copy2(guide, out / "README_KAGGLE.md")
    upload = out / "pilot-upload.zip"
    with zipfile.ZipFile(upload, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for path in sorted(dataset.iterdir()):
            z.write(path, path.name)
    with zipfile.ZipFile(upload) as z:
        if z.testzip():
            raise ValueError("upload CRC verification failed")
    return {"pilot_manifest_hash": manifest["manifest_hash"], "upload": str(upload),
            "upload_sha256": sha256_file(upload), "upload_bytes": upload.stat().st_size,
            "notebook": str(out / "guandan_t4x2_pilot.ipynb"), "remote_verified": False}
