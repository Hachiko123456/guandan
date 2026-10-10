# SPDX-License-Identifier: MIT
"""Build and validate the M07 Kaggle Dataset/Notebook delivery without uploading it."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Iterable, Mapping
import zipfile

from .common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from .discovery import (
    BOOTSTRAP_NAME, DATASET_MANIFEST_NAME, DATASET_MANIFEST_SCHEMA, SOURCE_MANIFEST_SCHEMA,
    SUPPORTED_MODES, validate_dataset_manifest,
)

PACKAGE_SCHEMA = "guandan-marvel-kaggle-delivery-v2"
PACKAGE_ID = "guandan-marvel-t4x2-20261010"
SOURCE_ARCHIVE_NAME = "guandan-marvel-source.zip"
SOURCE_MANIFEST_NAME = "source_manifest.json"
NOTEBOOK_NAME = "guandan_marvel_t4x2.ipynb"
README_NAME = "README_中文.md"
DELIVERY_MANIFEST_NAME = "delivery_manifest.json"
SOURCE_PREFIXES = ("guandan/", "guandan_marvel/", "nanopokerlib/")
SOURCE_ROOT_FILES = frozenset({
    "setup.py", "pyproject.toml", "MANIFEST.in", "README.md", "configs/marvel_kaggle_profiles.json",
})
BLOCKED_PARTS = frozenset({".git", "runs", "logs", "checkpoints", "wandb", "__pycache__", "build"})
BLOCKED_SUFFIXES = frozenset({".pt", ".pth", ".pyd", ".so", ".dll", ".dylib", ".key", ".pem"})


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, *args: str, binary: bool = False):
    result = subprocess.check_output(["git", *args], cwd=root)
    return result if binary else result.decode("utf-8").strip()


def _selected(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or BLOCKED_PARTS.intersection(path.parts):
        return False
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    return name in SOURCE_ROOT_FILES or any(name.startswith(prefix) for prefix in SOURCE_PREFIXES)


def _source_snapshot(root: Path, *, commit: str | None, allow_dirty: bool) -> tuple[str, bool, dict[str, bytes]]:
    root = root.resolve()
    head = _git(root, "rev-parse", "HEAD")
    if len(head) != 40:
        raise ValueError("source export requires a full Git commit")
    status = _git(root, "status", "--porcelain")
    if status and not allow_dirty:
        raise ValueError("source export requires a clean worktree unless --allow-dirty is explicit")
    if commit is not None and commit != "HEAD":
        head = _git(root, "rev-parse", commit)
    workspace_snapshot = bool(status and allow_dirty and commit in {None, "HEAD"})
    data: dict[str, bytes] = {}
    if workspace_snapshot:
        names = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", binary=True).decode("utf-8").split("\0")
        for name in filter(None, names):
            normalized = name.replace("\\", "/")
            if _selected(normalized):
                path = root / name
                if path.is_file():
                    data[normalized] = path.read_bytes()
    else:
        names = _git(root, "ls-tree", "-r", "--name-only", "-z", head, binary=True).decode("utf-8").split("\0")
        for name in filter(None, names):
            if _selected(name):
                mode = _git(root, "ls-tree", head, "--", name).split()[0]
                if mode not in {"100644", "100755"}:
                    raise ValueError(f"unsupported source entry mode for {name}: {mode}")
                data[name] = _git(root, "show", f"{head}:{name}", binary=True)
    required = {
        "setup.py", "pyproject.toml", "guandan_marvel/LICENSE", "guandan_marvel/NOTICE.md",
        "nanopokerlib/LICENSE", "nanopokerlib/NOTICE.md", "guandan_marvel/kaggle/runner.py",
        "guandan_marvel/kaggle/worker.py", "configs/marvel_kaggle_profiles.json",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"source export is incomplete: {missing}")
    return head, workspace_snapshot, data



def _bundled_pybind11() -> tuple[str, dict[str, bytes]]:
    distribution = importlib.metadata.distribution("pybind11")
    version = distribution.version
    package_root = Path(distribution.locate_file("pybind11")).resolve()
    data: dict[str, bytes] = {}
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix.lower() not in {".pyc", ".pyo"}:
            data[f"pybind11/{path.relative_to(package_root).as_posix()}"] = path.read_bytes()
    license_candidates = [item for item in distribution.files or [] if "LICENSE" in str(item).upper()]
    if not license_candidates:
        raise FileNotFoundError("installed pybind11 distribution does not expose its license")
    license_path = Path(distribution.locate_file(license_candidates[0])).resolve()
    data["third_party_licenses/pybind11-LICENSE"] = license_path.read_bytes()
    return version, data

def build_source_archive(
    repository: str | Path,
    destination: str | Path,
    *,
    commit: str | None = "HEAD",
    allow_dirty: bool = False,
) -> dict[str, object]:
    root = Path(repository).resolve()
    destination = Path(destination).resolve()
    git_commit, workspace_snapshot, data = _source_snapshot(root, commit=commit, allow_dirty=allow_dirty)
    pybind11_version, pybind11_data = _bundled_pybind11()
    overlap = sorted(set(data) & set(pybind11_data))
    if overlap:
        raise ValueError(f"bundled dependency collides with repository source: {overlap}")
    data.update(pybind11_data)
    files = [
        {
            "path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "origin": "bundled_pybind11" if name.startswith(("pybind11/", "third_party_licenses/pybind11")) else "git",
        }
        for name, content in sorted(data.items())
    ]
    body = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "package_id": PACKAGE_ID,
        "git_commit": git_commit,
        "workspace_snapshot": workspace_snapshot,
        "source_root": "guandan-marvel-source",
        "created_at": _timestamp(),
        "files": files,
        "licenses": [
            "guandan_marvel/LICENSE", "guandan_marvel/NOTICE.md",
            "nanopokerlib/LICENSE", "nanopokerlib/NOTICE.md",
            "third_party_licenses/pybind11-LICENSE",
        ],
        "bundled_dependencies": [
            {"name": "pybind11", "version": pybind11_version, "purpose": "offline native extension build"}
        ],
        "uploaded": False,
        "remote_verified": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for name, content in sorted(data.items()):
                info = zipfile.ZipInfo(f"guandan-marvel-source/{name}", date_time=(2026, 10, 10, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
            info = zipfile.ZipInfo("guandan-marvel-source/source_manifest.json", date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive": str(destination),
        "archive_sha256": sha256_file(destination),
        "archive_bytes": destination.stat().st_size,
        "source_manifest": manifest,
    }


NOTEBOOK_BOOTSTRAP = r'''from pathlib import Path
import hashlib, importlib.util, json, os, subprocess, sys, threading, time, traceback, zipfile

INPUT_ROOT = Path('/kaggle/input')
WORKING_ROOT = Path('/kaggle/working')
BOOTSTRAP_ROOT = WORKING_ROOT / 'guandan-marvel-bootstrap'
BOOTSTRAP_ROOT.mkdir(parents=True, exist_ok=True)
BOOTSTRAP_LOG = BOOTSTRAP_ROOT / 'bootstrap.log'
BOOTSTRAP_PROGRESS = BOOTSTRAP_ROOT / 'bootstrap_progress.json'
BOOTSTRAP_FAILURE = BOOTSTRAP_ROOT / 'bootstrap_failure.json'
BOOTSTRAP_DIAGNOSTICS = WORKING_ROOT / 'guandan-marvel-bootstrap-diagnostics.zip'
BOOTSTRAP_STARTED = time.monotonic()
BOOTSTRAP_LOCK = threading.Lock()
MODE = None


def _atomic_json(path, payload):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, target)


def _append_log(text):
    with BOOTSTRAP_LOG.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(str(text).rstrip('\n') + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def bootstrap_event(stage, message, *, status='running', **extra):
    elapsed = time.monotonic() - BOOTSTRAP_STARTED
    line = f'[Guandan bootstrap] stage={stage} status={status} elapsed={elapsed:.1f}s {message}'
    payload = {
        'schema': 'guandan-marvel-bootstrap-progress-v1',
        'stage': stage,
        'status': status,
        'message': str(message),
        'elapsed_seconds': elapsed,
        'mode': MODE,
        'pid': os.getpid(),
        **extra,
    }
    with BOOTSTRAP_LOCK:
        print(line, flush=True)
        _append_log(line)
        _atomic_json(BOOTSTRAP_PROGRESS, payload)
    return payload


def _write_diagnostics_zip():
    temporary = BOOTSTRAP_DIAGNOSTICS.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (BOOTSTRAP_LOG, BOOTSTRAP_PROGRESS, BOOTSTRAP_FAILURE):
            if path.is_file():
                archive.write(path, path.name)
    os.replace(temporary, BOOTSTRAP_DIAGNOSTICS)


def bootstrap_failed(error):
    payload = {
        'schema': 'guandan-marvel-bootstrap-failure-v1',
        'stage': json.loads(BOOTSTRAP_PROGRESS.read_text(encoding='utf-8')).get('stage') if BOOTSTRAP_PROGRESS.is_file() else 'unknown',
        'error_type': type(error).__name__,
        'error': str(error),
        'elapsed_seconds': time.monotonic() - BOOTSTRAP_STARTED,
        'mode': MODE,
        'traceback': traceback.format_exc(),
    }
    _atomic_json(BOOTSTRAP_FAILURE, payload)
    bootstrap_event('failed', f'{type(error).__name__}: {error}', status='failed')
    _write_diagnostics_zip()


def run_with_heartbeat(command, *, cwd, env=None, label, timeout_seconds, allowed_returncodes=(0,)):
    command = [str(value) for value in command]
    process_env = os.environ.copy() if env is None else dict(env)
    process_env['PYTHONUNBUFFERED'] = '1'
    stop = threading.Event()
    started = time.monotonic()
    bootstrap_event(label, 'process starting', command=command, timeout_seconds=float(timeout_seconds))

    def heartbeat():
        while not stop.wait(5.0):
            bootstrap_event(
                label,
                'process still running',
                process_elapsed_seconds=time.monotonic() - started,
                timeout_seconds=float(timeout_seconds),
            )

    thread = threading.Thread(target=heartbeat, name=f'{label}-heartbeat', daemon=True)
    thread.start()
    try:
        completed = subprocess.run(
            command,
            cwd=Path(cwd),
            env=process_env,
            check=False,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as error:
        bootstrap_event(
            label,
            f'process timed out after {float(timeout_seconds):.0f}s',
            status='failed',
            process_elapsed_seconds=time.monotonic() - started,
        )
        raise RuntimeError(
            f'{label} timed out after {float(timeout_seconds):.0f}s; see {BOOTSTRAP_LOG}'
        ) from error
    finally:
        stop.set()
        thread.join(timeout=2.0)
    if completed.returncode not in set(allowed_returncodes):
        raise RuntimeError(f'{label} failed with exit code {completed.returncode}')
    bootstrap_event(
        label,
        f'process completed exit={completed.returncode}',
        status='complete',
        process_elapsed_seconds=time.monotonic() - started,
    )
    return completed


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


bootstrap_event('start', 'Notebook bootstrap started')
try:
    bootstrap_event('dataset_discovery', 'searching for the unique Dataset manifest')
    manifests = sorted(INPUT_ROOT.rglob('guandan_marvel_dataset.json'))
    if len(manifests) != 1:
        raise RuntimeError(f'必须且只能发现一个 guandan_marvel_dataset.json，实际: {manifests}')
    DATASET_MANIFEST_PATH = manifests[0].resolve()
    bootstrap_event('dataset_validation', f'loading {DATASET_MANIFEST_PATH}')
    DATASET = json.loads(DATASET_MANIFEST_PATH.read_text(encoding='utf-8'))
    if DATASET.get('schema') != 'guandan-marvel-kaggle-dataset-v2':
        raise RuntimeError('Dataset manifest schema 不受支持')
    MODE = str(DATASET.get('mode'))
    bootstrap_spec = DATASET.get('bootstrap') or {}
    bootstrap_relative = Path(str(bootstrap_spec.get('path', '')))
    if bootstrap_relative.is_absolute() or '..' in bootstrap_relative.parts:
        raise RuntimeError('bootstrap 路径不安全')
    BOOTSTRAP_PATH = (DATASET_MANIFEST_PATH.parent / bootstrap_relative).resolve()
    if not BOOTSTRAP_PATH.is_relative_to(DATASET_MANIFEST_PATH.parent.resolve()) or not BOOTSTRAP_PATH.is_file():
        raise RuntimeError('Dataset bootstrap 文件缺失')
    if sha256_file(BOOTSTRAP_PATH) != bootstrap_spec.get('sha256'):
        raise RuntimeError('Dataset bootstrap SHA-256 不匹配')
    bootstrap_event('bootstrap_import', f'importing {BOOTSTRAP_PATH}')
    module_spec = importlib.util.spec_from_file_location('guandan_marvel_dataset_bootstrap', BOOTSTRAP_PATH)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError('无法加载 Dataset bootstrap')
    BOOTSTRAP = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = BOOTSTRAP
    module_spec.loader.exec_module(BOOTSTRAP)
    bootstrap_event('input_materialization', 'validating and copying source/recovery into /kaggle/working')
    PREPARED = BOOTSTRAP.prepare_inputs(
        DATASET_MANIFEST_PATH, input_root=INPUT_ROOT, working_root=WORKING_ROOT,
    )
    SOURCE_ROOT = Path(PREPARED['source_root'])
    RECOVERY_PATH = None if PREPARED['recovery_root'] is None else Path(PREPARED['recovery_root'])
    ENV = os.environ.copy()
    ENV['PYTHONPATH'] = str(SOURCE_ROOT) + (os.pathsep + ENV['PYTHONPATH'] if ENV.get('PYTHONPATH') else '')
    ENV['PYTHONUNBUFFERED'] = '1'
    ENV['GUANDAN_MARVEL_SOURCE_ROOT'] = str(SOURCE_ROOT)
    ENV['GUANDAN_MARVEL_SOURCE_COMMIT'] = DATASET['source']['git_commit']
    ENV['GUANDAN_MARVEL_SOURCE_ARCHIVE_SHA256'] = DATASET['source']['archive_sha256']
    ENV['GUANDAN_MARVEL_SOURCE_CONTENT_MANIFEST_HASH'] = DATASET['source']['content_manifest_hash']
    if not PREPARED['runtime_binary_restored']:
        BUILD_ENV = dict(ENV)
        BUILD_ENV['GUANDAN_BUILD_TIMEOUT_SECONDS'] = '600'
        run_with_heartbeat(
            [sys.executable, '-u', 'setup.py', 'build_ext', '--inplace'],
            cwd=SOURCE_ROOT,
            env=BUILD_ENV,
            label='native_build',
            timeout_seconds=660,
        )
    else:
        bootstrap_event('native_build', 'restored verified native binary from recovery; build skipped', status='complete')
    run_with_heartbeat(
        [sys.executable, '-u', '-c', "import guandan_marvel.cpp_game as g; print(g.get_backend_build_identity(), flush=True)"],
        cwd=SOURCE_ROOT,
        env=ENV,
        label='native_import',
        timeout_seconds=120,
    )
    bootstrap_event('ready_for_runner', 'bootstrap complete; training runner may start', status='complete')
    print(json.dumps({
        'mode': MODE,
        'dataset_manifest': str(DATASET_MANIFEST_PATH),
        'source_commit': DATASET['source']['git_commit'],
        'source_input_kind': PREPARED['source_input_kind'],
        'source_archive_sha256_verified': PREPARED['source_archive_sha256_verified'],
        'recovery_input_kind': PREPARED['recovery_input_kind'],
        'recovery': None if RECOVERY_PATH is None else str(RECOVERY_PATH),
        'bootstrap_log': str(BOOTSTRAP_LOG),
    }, ensure_ascii=False, indent=2), flush=True)
except BaseException as error:
    bootstrap_failed(error)
    raise
'''

NOTEBOOK_RUN = r'''profile = DATASET['profiles'][MODE]
run_control = profile.get('training') or profile.get('resume') or {}
runner_timeout_seconds = float(run_control.get('max_wall_seconds', 600)) + 900.0
command = [
    sys.executable, '-u', '-m', 'guandan_marvel.kaggle.runner',
    '--dataset-manifest', str(DATASET_MANIFEST_PATH),
    '--input-root', str(INPUT_ROOT),
    '--working-root', str(WORKING_ROOT),
    '--source-root', str(SOURCE_ROOT),
]
if RECOVERY_PATH is not None:
    command += ['--recovery-root', str(RECOVERY_PATH)]
print('RUN:', command, flush=True)
completed = run_with_heartbeat(
    command,
    cwd=SOURCE_ROOT,
    env=ENV,
    label='runner',
    timeout_seconds=runner_timeout_seconds,
    allowed_returncodes=(0, 2),
)
if completed.returncode == 2:
    print('达到时间预算，最近 durable recovery 已保留。', flush=True)
'''

NOTEBOOK_RESULTS = r'''from IPython.display import FileLink, display
OUTPUT_ROOT = WORKING_ROOT / 'guandan-marvel-output'
latest = json.loads((OUTPUT_ROOT / 'latest_run.json').read_text(encoding='utf-8'))
print(json.dumps(latest, ensure_ascii=False, indent=2))
for name in ('guandan-marvel-output.zip', 'guandan-marvel-recovery.zip'):
    path = OUTPUT_ROOT / name
    if path.is_file():
        display(FileLink(str(path)))
'''


def render_notebook() -> dict[str, object]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown", "metadata": {},
                "source": [
                    "# Guandan MARVEL T4×2 DDP\n",
                    "上传且只添加一个本交付的模式 Dataset，Notebook 会从 manifest 选择模式并 fail-closed 自动发现 source/recovery。训练使用 `torchrun --standalone --nproc_per_node=2`；M07 本地交付不等于真实 T4×2 已验证。\n",
                ],
            },
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": NOTEBOOK_BOOTSTRAP.splitlines(keepends=True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": NOTEBOOK_RUN.splitlines(keepends=True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": NOTEBOOK_RESULTS.splitlines(keepends=True)},
        ],
    }


def validate_source_archive(path: str | Path) -> dict[str, object]:
    archive_path = Path(path).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"source ZIP CRC failure: {bad}")
        names = archive.namelist()
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe source ZIP member: {name}")
        manifest_name = "guandan-marvel-source/source_manifest.json"
        if names.count(manifest_name) != 1:
            raise ValueError("source ZIP must contain exactly one internal source_manifest.json")
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
            raise ValueError("source manifest schema mismatch")
        declared = manifest.get("manifest_hash")
        actual = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
        if declared != actual:
            raise ValueError("source manifest hash mismatch")
        declared_names = set()
        for row in manifest["files"]:
            name = f"guandan-marvel-source/{row['path']}"
            declared_names.add(name)
            data = archive.read(name)
            if len(data) != int(row["bytes"]) or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"source member mismatch: {name}")
        extras = set(names) - declared_names - {manifest_name}
        if extras:
            raise ValueError(f"undeclared source ZIP members: {sorted(extras)}")
        runner_source = archive.read("guandan-marvel-source/guandan_marvel/kaggle/runner.py").decode("utf-8")
        if '"--standalone", "--nproc_per_node=2"' not in runner_source:
            raise ValueError("packaged runner does not use torchrun --standalone --nproc_per_node=2")
        worker_source = archive.read("guandan-marvel-source/guandan_marvel/kaggle/worker.py").decode("utf-8")
        if "world_size != 2" not in worker_source:
            raise ValueError("packaged worker does not fail closed outside world_size=2")
        progress_source = archive.read("guandan-marvel-source/guandan_marvel/kaggle/progress.py").decode("utf-8")
        if "[M07 heartbeat]" not in progress_source or "heartbeat_age_seconds" not in progress_source:
            raise ValueError("packaged runtime lacks Notebook-visible heartbeat diagnostics")
    return {
        "schema": "guandan-marvel-source-validation-v1",
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "git_commit": manifest["git_commit"],
        "workspace_snapshot": manifest["workspace_snapshot"],
        "file_count": len(manifest["files"]),
        "valid": True,
    }


def validate_notebook(path: str | Path) -> dict[str, object]:
    notebook_path = Path(path).resolve()
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    if payload.get("nbformat") != 4 or not isinstance(payload.get("cells"), list):
        raise ValueError("invalid notebook structure")
    code = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"] if cell.get("cell_type") == "code")
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{notebook_path}#cell{index}")
    forbidden = ("<path>", "TODO", "CHANGE_ME", "SOURCE_DATASET_HINT", "RESUME_IPPO", "RESUME_VRPO")
    found = [item for item in forbidden if item.lower() in notebook_path.read_text(encoding="utf-8").lower()]
    if found:
        raise ValueError(f"notebook contains forbidden placeholders: {found}")
    required = (
        "guandan_marvel_dataset.json", "nproc_per_node=2", "build_ext", "--dataset-manifest",
        "guandan-marvel-recovery.zip", "Run All" if False else "guandan-marvel-output.zip",
        "GUANDAN_BUILD_TIMEOUT_SECONDS", "bootstrap_progress.json", "process still running",
        "guandan-marvel-bootstrap-diagnostics.zip",
    )
    missing = [item for item in required if item not in notebook_path.read_text(encoding="utf-8")]
    if missing:
        raise ValueError(f"notebook is missing required behavior markers: {missing}")
    return {
        "schema": "guandan-marvel-notebook-validation-v1",
        "path": str(notebook_path), "code_cells": sum(cell.get("cell_type") == "code" for cell in payload["cells"]),
        "syntax_valid": True, "placeholders_absent": True, "valid": True,
    }


def _readme() -> str:
    return """# 掼蛋 MARVEL Kaggle T4×2 使用说明\n\n本目录是 **M07 remote-ready 本地交付**。它没有自动上传 Kaggle，也没有把本地检查伪装成真实 T4×2 验证；真实远端吞吐、显存、双卡效率和 pilot 结论属于 M08。\n\n## 1. 选择模式 Dataset\n\n在 `datasets` 下只选择一个目录创建 Kaggle Dataset：\n\n- `preflight`：推荐先运行，只检查源码、C++ 扩展、两张 T4、NCCL 和输入；\n- `pilot`：短 DDP 训练、durable checkpoint 和短评估；\n- `resume`：从另一个 Kaggle Dataset 中的 `guandan-marvel-recovery.zip` 精确恢复；\n- `full`：长训练。只有你明确选择 `datasets/full` 时才会启动，不是默认；\n- `eval_only`：从 recovery 中的冻结/训练 checkpoint 做只读短评估。\n\n不要把多个模式目录合并成同一个 Kaggle Dataset；Notebook 发现多个 `guandan_marvel_dataset.json` 会 fail closed。\n\n## 2. 创建 Kaggle Dataset\n\n1. 在 Kaggle 新建私有 Dataset。\n2. 上传所选 `datasets/<mode>` 目录内的全部四个文件（Dataset manifest、bootstrap、source manifest、source ZIP）。\n3. `resume`/`eval_only` 还要把上一次下载的 `guandan-marvel-recovery.zip` 建成第二个 Dataset。\n4. 不要改文件名、路径、manifest 或 hash。\n\n## 3. 添加 Notebook 并运行\n\n1. 导入 `guandan_marvel_t4x2.ipynb`。\n2. Add Input：只添加一个模式 Dataset；恢复模式再添加一个 recovery Dataset。\n3. Accelerator 选择 **GPU T4 x2**。\n4. Internet 可关闭；源码从 Dataset 解压，本地编译 pybind 扩展。\n5. 点击 **Run All**，不需要修改任何代码、绝对路径、checkpoint 名称或设备参数。\n\n首次 source 构建会从 bootstrap 开始立即打印阶段日志，并每 5 秒打印 heartbeat。native build 内层硬超时为 600 秒，Notebook 外层硬超时为 660 秒；超时或异常会在 `/kaggle/working/guandan-marvel-bootstrap/` 写进度、日志和失败 JSON，并生成 `/kaggle/working/guandan-marvel-bootstrap-diagnostics.zip`。不要继续运行长期无日志的旧 Notebook。\n\n训练模式固定使用 `torchrun --standalone --nproc_per_node=2`。rank 0 写主日志、manifest、checkpoint 和 bundle；每个 rank 另有诊断日志。\n\n## 4. 下载 output\n\n运行完成后下载：\n\n- `/kaggle/working/guandan-marvel-output/guandan-marvel-output.zip`；\n- `/kaggle/working/guandan-marvel-output/guandan-marvel-recovery.zip`。\n\n后者用于下一次 `resume` 或 `eval_only`。正常结束、预算停止和可捕获异常会尽力保留最近 durable checkpoint；Kaggle 平台硬杀无法保证 finally 执行，因此 checkpoint 会周期生成。\n\n## 5. 语义边界\n\n- `preflight`/`pilot` 只证明管线和恢复流程，不证明模型强度；\n- M07 `accepted` 只表示 package 构建和本地验证完成，即 `remote_ready`；\n- 真实双 T4 是否达到 1.5×、显存峰值和 full 参数必须由 M08 的下载证据确认；\n- `evaluation_summary.json` 是有限管线评估，不是 formal selection 或 final holdout；\n- Notebook 遇到多个 source/recovery、hash 不匹配、GPU 数量不符或恢复 world size 不符时会拒绝运行。\n"""


def build_delivery(
    repository: str | Path,
    delivery_root: str | Path,
    *,
    commit: str | None = "HEAD",
    allow_dirty: bool = False,
) -> dict[str, object]:
    repository = Path(repository).resolve()
    delivery_root = Path(delivery_root).resolve()
    if delivery_root.exists():
        shutil.rmtree(delivery_root)
    delivery_root.mkdir(parents=True)
    profiles_payload = json.loads((repository / "configs/marvel_kaggle_profiles.json").read_text(encoding="utf-8"))
    if profiles_payload.get("schema") != "guandan-marvel-kaggle-profiles-v1":
        raise ValueError("unsupported Kaggle profiles schema")
    source_build = build_source_archive(
        repository, delivery_root / SOURCE_ARCHIVE_NAME, commit=commit, allow_dirty=allow_dirty,
    )
    source_manifest = source_build["source_manifest"]
    atomic_json(delivery_root / SOURCE_MANIFEST_NAME, source_manifest)
    source_manifest_sha = sha256_file(delivery_root / SOURCE_MANIFEST_NAME)
    with zipfile.ZipFile(delivery_root / SOURCE_ARCHIVE_NAME) as archive:
        bootstrap_bytes = archive.read(f"guandan-marvel-source/guandan_marvel/kaggle/bootstrap.py")
    bootstrap_path = delivery_root / BOOTSTRAP_NAME
    bootstrap_path.write_bytes(bootstrap_bytes)
    bootstrap_sha = sha256_file(bootstrap_path)
    for mode in sorted(SUPPORTED_MODES):
        dataset_root = delivery_root / "datasets" / mode
        dataset_root.mkdir(parents=True)
        shutil.copy2(delivery_root / SOURCE_ARCHIVE_NAME, dataset_root / SOURCE_ARCHIVE_NAME)
        shutil.copy2(delivery_root / SOURCE_MANIFEST_NAME, dataset_root / SOURCE_MANIFEST_NAME)
        shutil.copy2(bootstrap_path, dataset_root / BOOTSTRAP_NAME)
        profile = profiles_payload["profiles"][mode]
        dataset = {
            "schema": DATASET_MANIFEST_SCHEMA,
            "package_id": PACKAGE_ID,
            "created_at": _timestamp(),
            "mode": mode,
            "source": {
                "archive": SOURCE_ARCHIVE_NAME,
                "archive_sha256": source_build["archive_sha256"],
                "manifest": SOURCE_MANIFEST_NAME,
                "manifest_sha256": source_manifest_sha,
                "git_commit": source_manifest["git_commit"],
                "content_manifest_hash": source_manifest["manifest_hash"],
            },
            "bootstrap": {"path": BOOTSTRAP_NAME, "sha256": bootstrap_sha},
            "profiles": {mode: profile},
            "recovery": {"required": mode in {"resume", "eval_only"}, "auto_discovery": True},
            "claims": {"package_ready": True, "remote_ready": True, "remote_verified": False, "model_strength": False},
        }
        atomic_json(dataset_root / DATASET_MANIFEST_NAME, dataset)
        validate_dataset_manifest(dataset, manifest_path=dataset_root / DATASET_MANIFEST_NAME)
    (delivery_root / NOTEBOOK_NAME).write_text(
        json.dumps(render_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (delivery_root / README_NAME).write_text(_readme(), encoding="utf-8", newline="\n")
    (delivery_root / SOURCE_ARCHIVE_NAME).unlink()
    (delivery_root / SOURCE_MANIFEST_NAME).unlink()
    bootstrap_path.unlink()
    validation = validate_delivery(delivery_root)
    files = [path for path in delivery_root.rglob("*") if path.is_file() and path.name != DELIVERY_MANIFEST_NAME]
    body = {
        "schema": PACKAGE_SCHEMA,
        "package_id": PACKAGE_ID,
        "created_at": _timestamp(),
        "source_commit": source_manifest["git_commit"],
        "workspace_snapshot": source_manifest["workspace_snapshot"],
        "default_recommended_mode": "preflight",
        "modes": sorted(SUPPORTED_MODES),
        "notebook": NOTEBOOK_NAME,
        "readme": README_NAME,
        "files": relative_file_manifest(delivery_root, files),
        "validation": validation,
        "uploaded": False,
        "remote_ready": True,
        "remote_verified": False,
    }
    manifest = {**body, "manifest_hash": canonical_hash(body)}
    atomic_json(delivery_root / DELIVERY_MANIFEST_NAME, manifest)
    return manifest


def validate_delivery(delivery_root: str | Path) -> dict[str, object]:
    root = Path(delivery_root).resolve()
    manifest_verified = False
    delivery_manifest_path = root / DELIVERY_MANIFEST_NAME
    if delivery_manifest_path.is_file():
        delivery_manifest = json.loads(delivery_manifest_path.read_text(encoding="utf-8"))
        if delivery_manifest.get("schema") != PACKAGE_SCHEMA:
            raise ValueError("delivery manifest schema mismatch")
        declared_hash = delivery_manifest.get("manifest_hash")
        actual_hash = canonical_hash({key: value for key, value in delivery_manifest.items() if key != "manifest_hash"})
        if declared_hash != actual_hash:
            raise ValueError("delivery manifest hash mismatch")
        declared_files = {row["path"]: row for row in delivery_manifest.get("files", [])}
        actual_files = {
            path.relative_to(root).as_posix(): path
            for path in root.rglob("*") if path.is_file() and path.name != DELIVERY_MANIFEST_NAME
        }
        if set(declared_files) != set(actual_files):
            raise ValueError("delivery manifest file set mismatch")
        for name, path in actual_files.items():
            row = declared_files[name]
            if int(row["bytes"]) != path.stat().st_size or row["sha256"] != sha256_file(path):
                raise ValueError(f"delivery file hash mismatch: {name}")
        manifest_verified = True
    notebook = validate_notebook(root / NOTEBOOK_NAME)
    source_validations = []
    dataset_validations = []
    for mode in sorted(SUPPORTED_MODES):
        dataset_root = root / "datasets" / mode
        manifest_path = dataset_root / DATASET_MANIFEST_NAME
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_dataset_manifest(payload, manifest_path=manifest_path)
        if payload["mode"] != mode:
            raise ValueError(f"dataset folder/mode mismatch: {mode}")
        source_validations.append(validate_source_archive(dataset_root / SOURCE_ARCHIVE_NAME))
        dataset_validations.append({
            "mode": mode,
            "manifest": manifest_path.relative_to(root).as_posix(),
            "valid": True,
        })
    commits = {row["git_commit"] for row in source_validations}
    archive_hashes = {row["archive_sha256"] for row in source_validations}
    if len(commits) != 1 or len(archive_hashes) != 1:
        raise ValueError("mode datasets do not share one source identity")
    return {
        "schema": "guandan-marvel-delivery-validation-v2",
        "valid": True,
        "delivery_manifest_verified": manifest_verified,
        "notebook": {**notebook, "path": NOTEBOOK_NAME},
        "datasets": dataset_validations,
        "source_commit": next(iter(commits)),
        "source_archive_sha256": next(iter(archive_hashes)),
        "source_archive_file_count": source_validations[0]["file_count"],
        "remote_verified": False,
    }


__all__ = [
    "DELIVERY_MANIFEST_NAME", "NOTEBOOK_NAME", "PACKAGE_ID", "README_NAME",
    "SOURCE_ARCHIVE_NAME", "SOURCE_MANIFEST_NAME", "build_delivery", "build_source_archive",
    "render_notebook", "validate_delivery", "validate_notebook", "validate_source_archive",
]
