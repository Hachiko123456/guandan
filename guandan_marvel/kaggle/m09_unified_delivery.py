# SPDX-License-Identifier: MIT
"""Build the single-Dataset/single-Notebook M09 Unified Pipeline delivery."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Mapping

from .common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from .discovery import DATASET_MANIFEST_NAME, validate_dataset_manifest
from .package import SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, validate_source_archive
from .unified_pipeline import AUTO_CONTINUATION_SCHEMA, validate_recovery_archive

UNIFIED_DELIVERY_ID = "guandan-marvel-m09-unified-pipeline-v1"
UNIFIED_DELIVERY_SCHEMA = "guandan-marvel-m09-unified-delivery-v1"
UNIFIED_VALIDATION_SCHEMA = "guandan-marvel-m09-unified-delivery-validation-v1"
UNIFIED_CONTROL_SCHEMA = "guandan-marvel-m09-unified-control-v1"
UNIFIED_CONTROL_NAME = "m09_unified_control.json"
UNIFIED_NOTEBOOK_NAME = "guandan_marvel_m09_unified_pipeline.ipynb"
UNIFIED_README_NAME = "README_统一流水线.md"
INITIAL_RECOVERY_NAME = "m09_initial_iter0500_recovery.seed.zip"
CHAMPION_SEED_NAME = "ippo_iter000100_champion_seed.pt"
LEGACY_NAME = "legacy_v1_anchor.pt"
ACCEPTANCE_NAME = "acceptance_profiles.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _overlay_init() -> str:
    return "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"


NOTEBOOK_BOOTSTRAP = r'''from pathlib import Path
import hashlib, importlib.util, json, os, shutil, subprocess, sys, threading, time, traceback, zipfile
from IPython.display import FileLink, display
INPUT_ROOT=Path('/kaggle/input'); WORKING_ROOT=Path('/kaggle/working')
PIPELINE_EXIT=1; PIPELINE_ERROR=None
BOOTSTRAP_STARTED=time.monotonic(); BOOTSTRAP_STAGE={'name':'startup'}; BOOTSTRAP_STOP=threading.Event()

def stage(name, **extra):
    BOOTSTRAP_STAGE['name']=name
    print(json.dumps({'event':'unified_bootstrap_stage','stage':name,'elapsed_seconds':round(time.monotonic()-BOOTSTRAP_STARTED,3),**extra},sort_keys=True,ensure_ascii=False),flush=True)

def bootstrap_heartbeat():
    while not BOOTSTRAP_STOP.wait(10.0):
        print(json.dumps({'event':'unified_bootstrap_heartbeat','stage':BOOTSTRAP_STAGE['name'],'elapsed_seconds':round(time.monotonic()-BOOTSTRAP_STARTED,3)},sort_keys=True),flush=True)

stage('startup')
BOOTSTRAP_THREAD=threading.Thread(target=bootstrap_heartbeat,name='unified-bootstrap-heartbeat',daemon=True); BOOTSTRAP_THREAD.start()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def stream(command, *, cwd, env, label):
    print(f'[{label}] RUN:',command,flush=True)
    process=subprocess.Popen([str(x) for x in command],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout: print(line.rstrip(),flush=True)
    code=process.wait(); print(f'[{label}] exit={code}',flush=True); return code

try:
    stage('dataset_discovery')
    manifests=sorted(INPUT_ROOT.rglob('guandan_marvel_dataset.json'))
    controls=sorted(INPUT_ROOT.rglob('m09_unified_control.json'))
    if len(manifests)!=1 or len(controls)!=1:
        raise RuntimeError(f'exactly one Unified Dataset is required: manifests={manifests}, controls={controls}')
    DATASET_PATH=manifests[0].resolve(); CONTROL_PATH=controls[0].resolve(); DATASET=json.loads(DATASET_PATH.read_text(encoding='utf-8')); CONTROL=json.loads(CONTROL_PATH.read_text(encoding='utf-8'))
    if CONTROL.get('schema')!='guandan-marvel-m09-unified-control-v1' or CONTROL.get('manifest_hash')!=canonical({k:v for k,v in CONTROL.items() if k!='manifest_hash'}):
        raise RuntimeError('Unified control manifest identity mismatch')
    base=CONTROL_PATH.parent.resolve()
    stage('control_validation',control_files=len(CONTROL['control_files']))
    for row in CONTROL['control_files']:
        path=(base/row['path']).resolve()
        if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise RuntimeError(f'Unified control file mismatch: {row}')
    bootstrap_spec=DATASET['bootstrap']; bootstrap_path=(DATASET_PATH.parent/bootstrap_spec['path']).resolve()
    if sha(bootstrap_path)!=bootstrap_spec['sha256']: raise RuntimeError('Dataset bootstrap hash mismatch')
    spec=importlib.util.spec_from_file_location('m09_dataset_bootstrap',bootstrap_path); BOOTSTRAP=importlib.util.module_from_spec(spec); sys.modules[spec.name]=BOOTSTRAP; spec.loader.exec_module(BOOTSTRAP)
    empty=WORKING_ROOT/'m09-unified-empty-input'; empty.mkdir(parents=True,exist_ok=True)
    stage('source_materialization')
    prepared=BOOTSTRAP.prepare_inputs(DATASET_PATH,input_root=empty,working_root=WORKING_ROOT)
    SOURCE_ROOT=Path(prepared['source_root'])
    stage('control_overlay_materialization')
    overlay_source=DATASET_PATH.parent/'control_overlay'; OVERLAY_ROOT=WORKING_ROOT/'m09-unified-control-overlay'
    if OVERLAY_ROOT.exists(): shutil.rmtree(OVERLAY_ROOT)
    shutil.copytree(overlay_source,OVERLAY_ROOT)
    sys.path.insert(0,str(OVERLAY_ROOT)); sys.path.insert(1,str(SOURCE_ROOT))
    ENV=os.environ.copy(); ENV['PYTHONUNBUFFERED']='1'; ENV['PYTHONDONTWRITEBYTECODE']='1'; ENV['PYTHONPATH']=str(OVERLAY_ROOT)+os.pathsep+str(SOURCE_ROOT)+(os.pathsep+ENV['PYTHONPATH'] if ENV.get('PYTHONPATH') else '')
    ENV['GUANDAN_MARVEL_SOURCE_ROOT']=str(SOURCE_ROOT); ENV['GUANDAN_MARVEL_SOURCE_COMMIT']=str(CONTROL['source_identity']['source_commit'])
    from guandan_marvel.kaggle.unified_pipeline import discover_latest_recovery, materialize_recovery_candidate
    stage('recovery_discovery')
    selected=discover_latest_recovery(INPUT_ROOT,source_identity=CONTROL['source_identity'])
    stage('recovery_materialization',kind=selected.kind,iteration=int(selected.manifest.get('iteration',-1)))
    RECOVERY_ROOT=materialize_recovery_candidate(selected,WORKING_ROOT/'guandan-marvel-recovery',copy_directory=False)
    recovery_manifest=json.loads((RECOVERY_ROOT/'recovery_manifest.json').read_text(encoding='utf-8'))
    restored=False; runtime_binary=recovery_manifest.get('runtime_binary')
    if isinstance(runtime_binary,dict):
        source_binary=RECOVERY_ROOT/runtime_binary['archive_path']; target_binary=SOURCE_ROOT/runtime_binary['restore_relative_path']; target_binary.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source_binary,target_binary); restored=True
    if not restored:
        stage('native_build')
        build_env=dict(ENV); build_env['GUANDAN_BUILD_TIMEOUT_SECONDS']='600'
        if stream([sys.executable,'-u','setup.py','build_ext','--inplace'],cwd=SOURCE_ROOT,env=build_env,label='native-build')!=0: raise RuntimeError('native build failed')
    stage('native_import')
    if stream([sys.executable,'-u','-c',"import guandan_marvel.cpp_game as g; print(g.get_backend_build_identity(),flush=True)"],cwd=SOURCE_ROOT,env=ENV,label='native-import')!=0: raise RuntimeError('native import failed')
    stage('unified_pipeline')
    script=OVERLAY_ROOT/'scripts/m09_unified_pipeline.py'
    command=[sys.executable,'-u',str(script),'--dataset-manifest',str(DATASET_PATH),'--control-manifest',str(CONTROL_PATH),'--input-root',str(INPUT_ROOT),'--working-root',str(WORKING_ROOT),'--source-root',str(SOURCE_ROOT),'--recovery-root',str(RECOVERY_ROOT)]
    PIPELINE_EXIT=stream(command,cwd=SOURCE_ROOT,env=ENV,label='unified-pipeline')
    BOOTSTRAP_STOP.set(); BOOTSTRAP_THREAD.join(timeout=2.0)
except BaseException as error:
    BOOTSTRAP_STOP.set(); BOOTSTRAP_THREAD.join(timeout=2.0)
    PIPELINE_ERROR=f'{type(error).__name__}: {error}'
    print(PIPELINE_ERROR,flush=True); traceback.print_exc()
'''

NOTEBOOK_RESULTS = r'''OUTPUT=WORKING_ROOT/'guandan-marvel-unified-output'
if OUTPUT.is_dir():
    latest=OUTPUT/'latest_run.json'
    if latest.is_file(): print(latest.read_text(encoding='utf-8'),flush=True)
    links=[]
    for path in [OUTPUT/'guandan-marvel-continuation.zip',OUTPUT/'pipeline_state.json',OUTPUT/'latest_run.json',OUTPUT/'remote_c_result.json']:
        if path.is_file(): links.append(path)
    links.extend(sorted(OUTPUT.glob('committed_training/*/training_recovery.committed.zip')))
    links.extend(sorted(OUTPUT.glob('evaluation/*/evaluation_summary.json')))
    links.extend(sorted(OUTPUT.glob('evaluation/*/single_dual_consistency.json')))
    for path in links: display(FileLink(str(path)))
else:
    print('Unified output directory was not created.',flush=True)
if PIPELINE_ERROR is not None: raise RuntimeError(PIPELINE_ERROR)
if PIPELINE_EXIT!=0: raise RuntimeError(f'Unified pipeline failed recoverably with exit={PIPELINE_EXIT}; download the links above and rerun with this Notebook Output as Input.')
'''


def render_unified_notebook() -> dict[str, object]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}, "accelerator": "GPU"},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Guandan MARVEL M09 Unified Pipeline\n",
                "长期只使用一个 Unified Dataset、这个 Notebook，以及上一段 Notebook Output。当前 v1 先通过 GUANDAN_M09_REMOTE_VALIDATION=A/B/C 做小预算 Remote A/B/C；production Segment1000 默认被 gate。验证通过并更新同一 Dataset 的版本化 control 后，Run All 才会自动发现最新 recovery 和下一 Segment。双 GPU formal 默认关闭；final holdout 永不由此 Notebook 生成。\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": NOTEBOOK_BOOTSTRAP.splitlines(keepends=True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": NOTEBOOK_RESULTS.splitlines(keepends=True)},
        ],
    }


def build_unified_delivery(
    repository: str | Path,
    delivery_root: str | Path,
    *,
    source_dataset_root: str | Path,
    initial_recovery: str | Path,
    champion_seed: str | Path,
    legacy_checkpoint: str | Path,
) -> dict[str, object]:
    repo = Path(repository).resolve()
    output = Path(delivery_root).resolve()
    source_dataset = Path(source_dataset_root).resolve()
    initial_recovery = Path(initial_recovery).resolve()
    champion_seed = Path(champion_seed).resolve()
    legacy_checkpoint = Path(legacy_checkpoint).resolve()
    if output.exists():
        shutil.rmtree(output)
    dataset = output / "dataset"
    dataset.mkdir(parents=True)

    for name in (SOURCE_ARCHIVE_NAME, SOURCE_MANIFEST_NAME, "guandan_marvel_bootstrap.py"):
        shutil.copy2(source_dataset / name, dataset / name)
    source_manifest = json.loads((dataset / SOURCE_MANIFEST_NAME).read_text(encoding="utf-8"))
    source_validation = validate_source_archive(dataset / SOURCE_ARCHIVE_NAME)
    source_identity = {
        "source_commit": source_manifest["git_commit"],
        "source_archive_sha256": source_validation["archive_sha256"],
        "source_content_manifest_hash": source_manifest["manifest_hash"],
    }

    plan_source = repo / "configs/marvel_m09_main_training_plan.json"
    locked_source = repo / "configs/marvel_m08_locked_config.json"
    acceptance_source = repo / "configs/acceptance_profiles.json"
    copies = {
        "marvel_m09_main_training_plan.json": plan_source,
        "marvel_m08_locked_config.json": locked_source,
        ACCEPTANCE_NAME: acceptance_source,
        INITIAL_RECOVERY_NAME: initial_recovery,
        CHAMPION_SEED_NAME: champion_seed,
        LEGACY_NAME: legacy_checkpoint,
    }
    for name, source in copies.items():
        shutil.copy2(source, dataset / name)

    overlay = dataset / "control_overlay"
    (overlay / "guandan_marvel/kaggle").mkdir(parents=True)
    (overlay / "guandan_marvel/evaluation").mkdir(parents=True)
    (overlay / "scripts").mkdir(parents=True)
    (overlay / "guandan_marvel/__init__.py").write_text(_overlay_init(), encoding="utf-8", newline="\n")
    (overlay / "guandan_marvel/kaggle/__init__.py").write_text(_overlay_init(), encoding="utf-8", newline="\n")
    (overlay / "guandan_marvel/evaluation/__init__.py").write_text(_overlay_init(), encoding="utf-8", newline="\n")
    overlay_sources = {
        "guandan_marvel/kaggle/unified_pipeline.py": repo / "guandan_marvel/kaggle/unified_pipeline.py",
        "guandan_marvel/evaluation/resumable.py": repo / "guandan_marvel/evaluation/resumable.py",
        "guandan_marvel/evaluation/dual_gpu.py": repo / "guandan_marvel/evaluation/dual_gpu.py",
        "guandan_marvel/evaluation/provenance.py": repo / "guandan_marvel/evaluation/provenance.py",
        "scripts/m09_unified_pipeline.py": repo / "scripts/m09_unified_pipeline.py",
        "scripts/m09_unified_formal_worker.py": repo / "scripts/m09_unified_formal_worker.py",
    }
    for relative, source in overlay_sources.items():
        shutil.copy2(source, overlay / relative)

    main_plan = json.loads((dataset / "marvel_m09_main_training_plan.json").read_text(encoding="utf-8"))
    policy = dict(main_plan["champion_plan"]["promotion_policy"])
    control_file_paths = [
        dataset / "marvel_m09_main_training_plan.json",
        dataset / "marvel_m08_locked_config.json",
        dataset / ACCEPTANCE_NAME,
        dataset / CHAMPION_SEED_NAME,
        dataset / LEGACY_NAME,
        *[path for path in overlay.rglob("*") if path.is_file()],
    ]
    control_files = relative_file_manifest(dataset, control_file_paths)
    control_body: dict[str, object] = {
        "schema": UNIFIED_CONTROL_SCHEMA,
        "version": 1,
        "delivery_id": UNIFIED_DELIVERY_ID,
        "created_at": _timestamp(),
        "source_identity": source_identity,
        "main_plan_hash": main_plan["plan_hash"],
        "main_training_plan": {"path": "marvel_m09_main_training_plan.json", "sha256": sha256_file(dataset / "marvel_m09_main_training_plan.json")},
        "locked_config": {"path": "marvel_m08_locked_config.json", "sha256": sha256_file(dataset / "marvel_m08_locked_config.json")},
        "acceptance_profiles": {"path": ACCEPTANCE_NAME, "sha256": sha256_file(dataset / ACCEPTANCE_NAME)},
        "initial_recovery": {"path": INITIAL_RECOVERY_NAME, "sha256": sha256_file(dataset / INITIAL_RECOVERY_NAME), "iteration": 500},
        "champion_seed": {"path": CHAMPION_SEED_NAME, "sha256": sha256_file(dataset / CHAMPION_SEED_NAME), "iteration": 100},
        "legacy_checkpoint": {"path": LEGACY_NAME, "sha256": sha256_file(dataset / LEGACY_NAME)},
        "pipeline_runner": {"path": "control_overlay/scripts/m09_unified_pipeline.py", "sha256": sha256_file(overlay / "scripts/m09_unified_pipeline.py")},
        "formal_worker": {"path": "control_overlay/scripts/m09_unified_formal_worker.py", "sha256": sha256_file(overlay / "scripts/m09_unified_formal_worker.py")},
        "control_files": control_files,
        "schedule": [
            {"segment_id": "segment0500", "target_iteration": 500, "target_completed_games": 2_000},
            {"segment_id": "segment1000", "target_iteration": 1_000, "target_completed_games": 4_000},
            {"segment_id": "segment1500", "target_iteration": 1_500, "target_completed_games": 6_000},
            {"segment_id": "segment2000", "target_iteration": 2_000, "target_completed_games": 8_000},
        ],
        "evaluation": {"formal_workers_default": 1, "dual_gpu_formal_remote_validated": False, "sharding": "task_index_mod_world_size", "resume": True},
        "remote_validation": {"production_enabled": False, "default_mode": "A", "smoke_target_iteration": 502, "smoke_target_completed_games": 2008, "selection_clusters": 2, "modes": ["A", "B", "C"]},
        "promotion_policy": {
            "ci95_low_threshold": float(policy["ci95_low_threshold"]),
            "min_clusters": int(policy["min_clusters"]),
            "rule_min_win_rate": float(policy["rule_min_win_rate"]),
            "random_min_win_rate": float(policy["random_min_win_rate"]),
        },
        "auto_continuation": {"schema": AUTO_CONTINUATION_SCHEMA, "enabled": False, "max_segments_per_run": 1, "final_holdout_allowed": False},
        "monitoring": {"heartbeat_seconds": 20, "gpu_source": "nvidia-smi", "stdout_stderr_forwarded": True},
        "claims": {"remote_ready": True, "remote_validated": False, "final_holdout_generated": False, "m09_accepted": False, "strength_claim": False},
    }
    control = {**control_body, "manifest_hash": canonical_hash(control_body)}
    atomic_json(dataset / UNIFIED_CONTROL_NAME, control)

    base_dataset = json.loads((source_dataset / DATASET_MANIFEST_NAME).read_text(encoding="utf-8"))
    base_dataset["created_at"] = _timestamp()
    base_dataset["mode"] = "preflight"
    base_dataset["profiles"] = {"preflight": {"schema": "guandan-marvel-kaggle-profile-v1", "mode": "preflight", "required_gpu_count": 2, "required_gpu_name_contains": "T4", "telemetry_interval_seconds": 5}}
    base_dataset["recovery"] = {"required": False, "auto_discovery": False, "unified_selector": "highest_compatible_iteration"}
    base_dataset["m09"] = {
        "role": "unified_pipeline",
        "plan": {"path": "marvel_m09_main_training_plan.json", "sha256": sha256_file(dataset / "marvel_m09_main_training_plan.json"), "plan_hash": main_plan["plan_hash"]},
        "locked_config": {"path": "marvel_m08_locked_config.json", "sha256": sha256_file(dataset / "marvel_m08_locked_config.json")},
        "unified_control": {"path": UNIFIED_CONTROL_NAME, "sha256": sha256_file(dataset / UNIFIED_CONTROL_NAME), "manifest_hash": control["manifest_hash"]},
        "final_holdout_generated": False,
    }
    base_dataset["claims"] = {"remote_ready": True, "remote_verified": False, "unified_pipeline_remote_validated": False, "m09_accepted": False, "strength_claim": False}
    atomic_json(dataset / DATASET_MANIFEST_NAME, base_dataset)
    validate_dataset_manifest(base_dataset, manifest_path=dataset / DATASET_MANIFEST_NAME)

    notebook = render_unified_notebook()
    atomic_json(output / UNIFIED_NOTEBOOK_NAME, notebook)
    (output / UNIFIED_README_NAME).write_text(_readme(), encoding="utf-8", newline="\n")
    validation = validate_unified_delivery(output)
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "delivery_manifest.json"]
    body = {
        "schema": UNIFIED_DELIVERY_SCHEMA,
        "delivery_id": UNIFIED_DELIVERY_ID,
        "created_at": _timestamp(),
        "dataset": "dataset",
        "notebook": UNIFIED_NOTEBOOK_NAME,
        "source_identity": source_identity,
        "control_manifest_hash": control["manifest_hash"],
        "main_plan_hash": main_plan["plan_hash"],
        "validation": validation,
        "files": relative_file_manifest(output, files),
        "remote_validated": False,
        "final_holdout_generated": False,
        "m09_accepted": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    atomic_json(output / "delivery_manifest.json", manifest)
    return manifest



def _readme() -> str:
    return f"""# M09 Unified Pipeline v1

长期入口只有：

1. `dataset/` 创建的一个私有 Kaggle Dataset；
2. `{UNIFIED_NOTEBOOK_NAME}`；
3. 上一次这个 Notebook 的 Output。

当前 control 将正式 Segment1000 置于 gate 后。不要直接无变量 Run All 启动 production。按顺序完成：

- Remote A：直接 Run All（v1 在 production gate 未打开时默认 A；显式 `GUANDAN_M09_REMOTE_VALIDATION=A` 等价），只跑 iter500→502 / 2000→2008 games 和 2-cluster smoke evaluation；
- Remote B 首次：设置 `GUANDAN_M09_REMOTE_VALIDATION=B`、`GUANDAN_M09_EVAL_FAIL_AFTER=8`、`GUANDAN_M09_EVAL_FAIL_WORKER=0`；
- Remote B 恢复：添加首次 B Output，保留 `GUANDAN_M09_REMOTE_VALIDATION=B`，移除 failure injection；
- Remote C：添加已完成的 A Output，设置 `GUANDAN_M09_REMOTE_VALIDATION=C`，自动比较同一 plan 的单/双 worker 输出。

这些变量是外部运行请求，不需要修改 Notebook、target iteration 或 training/evaluation 代码。A/B/C 通过后，应更新**同一个 Kaggle Dataset** 的版本化 control（`production_enabled=true`）；随后日常只添加 Unified Dataset + 上一 Output 并 Run All，自动选择 1000/1500/2000。

训练 durable commit 完成且 DDP 完全退出后，才启动独立 evaluation worker。评估失败会保留 committed recovery 和合法 JSONL shards；相同 Notebook + 上一 Output 会跳过训练并补缺失任务。

双 GPU formal production 默认仍为单 worker，直到 Remote C 通过并发布新 control。自动连续 Segment 默认关闭。本交付不生成 final holdout，不表示 Remote A/B/C 已通过，也不表示 M09 accepted 或模型达到 E1/E2/E3。
"""



def validate_unified_delivery(delivery_root: str | Path) -> dict[str, object]:
    root = Path(delivery_root).resolve()
    dataset = root / "dataset"
    notebooks = sorted(root.glob("*.ipynb"))
    dataset_manifests = sorted(root.rglob(DATASET_MANIFEST_NAME))
    controls = sorted(root.rglob(UNIFIED_CONTROL_NAME))
    if notebooks != [root / UNIFIED_NOTEBOOK_NAME]:
        raise ValueError(f"Unified delivery must contain exactly one Notebook: {notebooks}")
    if dataset_manifests != [dataset / DATASET_MANIFEST_NAME] or controls != [dataset / UNIFIED_CONTROL_NAME]:
        raise ValueError("Unified delivery must contain exactly one Dataset/control entry")
    dataset_payload = json.loads((dataset / DATASET_MANIFEST_NAME).read_text(encoding="utf-8"))
    validate_dataset_manifest(dataset_payload, manifest_path=dataset / DATASET_MANIFEST_NAME)
    if dataset_payload.get("m09", {}).get("role") != "unified_pipeline":
        raise ValueError("Unified Dataset role mismatch")
    source = validate_source_archive(dataset / SOURCE_ARCHIVE_NAME)
    control = json.loads((dataset / UNIFIED_CONTROL_NAME).read_text(encoding="utf-8"))
    if control.get("schema") != UNIFIED_CONTROL_SCHEMA or control.get("manifest_hash") != canonical_hash({key: value for key, value in control.items() if key != "manifest_hash"}):
        raise ValueError("Unified control manifest schema/hash mismatch")
    for row in control["control_files"]:
        path = (dataset / str(row["path"])).resolve()
        if not path.is_relative_to(dataset) or not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Unified control file mismatch: {row['path']}")
    recovery = validate_recovery_archive(dataset / INITIAL_RECOVERY_NAME)
    if recovery["iteration"] != 500 or recovery["sha256"] != control["initial_recovery"]["sha256"]:
        raise ValueError("Unified initial recovery is not the verified iter500 seed")
    for key in ("champion_seed", "legacy_checkpoint", "formal_worker", "pipeline_runner"):
        spec = control[key]
        if sha256_file(dataset / str(spec["path"])) != spec["sha256"]:
            raise ValueError(f"Unified control identity mismatch: {key}")
    notebook = json.loads((root / UNIFIED_NOTEBOOK_NAME).read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("Unified Notebook structure is invalid")
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{UNIFIED_NOTEBOOK_NAME}#cell{index}")
    required = (
        "discover_latest_recovery", "m09_unified_pipeline.py", "subprocess.Popen",
        "guandan-marvel-continuation.zip", "training_recovery.committed.zip",
        "PIPELINE_EXIT", "FileLink", "INPUT_ROOT",
    )
    missing = [marker for marker in required if marker not in code]
    if missing:
        raise ValueError(f"Unified Notebook behavior markers missing: {missing}")
    if control["evaluation"]["formal_workers_default"] != 1 or control["evaluation"]["dual_gpu_formal_remote_validated"] is not False:
        raise ValueError("dual-GPU formal evaluation must remain non-default before Remote C")
    remote_validation = control.get("remote_validation") or {}
    if remote_validation.get("production_enabled") is not False or remote_validation.get("default_mode") != "A" or int(remote_validation.get("smoke_target_iteration", 0)) != 502:
        raise ValueError("Unified delivery must gate production Segment1000 behind staged remote smoke")
    if control["auto_continuation"] != {"schema": AUTO_CONTINUATION_SCHEMA, "enabled": False, "max_segments_per_run": 1, "final_holdout_allowed": False}:
        raise ValueError("auto-continuation must remain versioned and disabled by default")
    forbidden = [path for path in root.rglob("*") if path.is_file() and "final_holdout" in path.name.lower()]
    if forbidden:
        raise ValueError(f"Unified delivery generated forbidden final holdout artifacts: {forbidden}")
    return {
        "schema": UNIFIED_VALIDATION_SCHEMA,
        "valid": True,
        "delivery_id": UNIFIED_DELIVERY_ID,
        "dataset_count": 1,
        "notebook_count": 1,
        "source_commit": source["git_commit"],
        "source_archive_sha256": source["archive_sha256"],
        "control_manifest_hash": control["manifest_hash"],
        "initial_recovery_sha256": recovery["sha256"],
        "champion_seed_sha256": control["champion_seed"]["sha256"],
        "notebook_syntax_valid": True,
        "formal_workers_default": 1,
        "dual_gpu_formal_remote_validated": False,
        "auto_continuation_enabled": False,
        "final_holdout_generated": False,
        "production_enabled": False,
        "remote_validated": False,
        "m09_accepted": False,
    }


__all__ = [
    "UNIFIED_CONTROL_NAME", "UNIFIED_DELIVERY_ID", "UNIFIED_NOTEBOOK_NAME",
    "build_unified_delivery", "render_unified_notebook", "validate_unified_delivery",
]
