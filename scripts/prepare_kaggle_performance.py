"""Generate a hash-pinned, opt-in Kaggle short-test notebook. No old packager edits."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NEW_FILES = ("guandan/performance_gate.py", "scripts/run_local_performance.py",
             "scripts/prepare_kaggle_performance.py", "docs/gpu_performance_workflow.md",
             "tests/unit/test_performance_gate.py")

CONFIG = '''# Default is plan-only. This Notebook NEVER launches a long training run.
EXECUTE = False
SOURCE_ZIP = ''  # exact ZIP or expanded source root containing SOURCE_MANIFEST.json under /kaggle/input
PROGRESS_BUNDLE = ''  # blank=auto-discover one trained ZIP/expanded directory; None=explicit fresh seeded run
BUNDLE_SHA256 = ''  # optional SHA256 for the original progress ZIP, not a directory
LOCAL_REPORT = ''  # optional optimized local result.json for release gating
REFERENCE_REPORT = ''  # optional same-machine reference local result.json
BUDGET_SECONDS = 600  # total from bootstrap start, includes extraction/import/CUDA initialization
'''

DISCOVERY = r'''import hashlib
import zipfile
from pathlib import Path, PurePosixPath


def _discovery_check(check_budget):
    if check_budget is not None:
        check_budget()


def _discovery_sha256(path, check_budget):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            _discovery_check(check_budget)
            digest.update(chunk)
    return digest.hexdigest()


def _discovery_inside(path, input_root):
    root = Path(input_root).expanduser().resolve()
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists() or not candidate.is_relative_to(root):
        raise ValueError('Input path must exist under /kaggle/input: ' + str(path))
    return candidate


def _discovery_unique(paths):
    result = []
    seen = set()
    for path in paths:
        resolved = Path(path).resolve()
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def _discovery_scan_paths(input_root, check_budget, max_entries):
    root = Path(input_root).resolve()
    count = 0
    for path in root.rglob('*'):
        _discovery_check(check_budget)
        count += 1
        if count > max_entries:
            raise ValueError(f'Input scan exceeds max_entries={max_entries}; set an explicit path or remove unrelated Datasets')
        if path.is_symlink():
            continue
        yield path


def _discovery_expanded_sources(input_root, expected_manifest_sha256, check_budget, max_entries):
    candidates = []
    for path in _discovery_scan_paths(input_root, check_budget, max_entries):
        if path.is_file() and path.name == 'SOURCE_MANIFEST.json':
            if _discovery_sha256(path, check_budget) == expected_manifest_sha256:
                candidates.append(path.parent)
    return _discovery_unique(candidates)


def _discovery_source_zips(input_root, expected_source_sha256, check_budget, max_entries):
    candidates = []
    for path in _discovery_scan_paths(input_root, check_budget, max_entries):
        if not path.is_file():
            continue
        # Kaggle may rename an uploaded ZIP or place it under a versioned path;
        # identify it by ZIP signature plus the pinned whole-file SHA, not name.
        with path.open('rb') as stream:
            signature = stream.read(4)
        _discovery_check(check_budget)
        if signature != b'PK\x03\x04':
            continue
        if _discovery_sha256(path, check_budget) == expected_source_sha256:
            candidates.append(path)
    return _discovery_unique(candidates)


def _discovery_progress_zip(path, check_budget, max_entries):
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > max_entries:
                raise ValueError(f'Progress ZIP has more than max_entries={max_entries}')
            found = set()
            for info in infos:
                _discovery_check(check_budget)
                rel = PurePosixPath(info.filename)
                if rel.is_absolute() or '..' in rel.parts or '\\' in info.filename or ':' in info.filename:
                    continue
                parts = rel.parts
                for index in range(len(parts) - 1):
                    if parts[index:index + 2] == ('workers', 'ippo'):
                        found.add('ippo')
                    if parts[index:index + 2] == ('workers', 'vrpo'):
                        found.add('vrpo')
            return found == {'ippo', 'vrpo'}
    except (OSError, zipfile.BadZipFile):
        return False


def _discovery_expanded_progress(input_root, check_budget, max_entries):
    candidates = []
    for path in _discovery_scan_paths(input_root, check_budget, max_entries):
        if path.is_dir() and path.name == 'workers':
            if (path / 'ippo').is_dir() and (path / 'vrpo').is_dir():
                candidates.append(path.parent)
    return _discovery_unique(candidates)


def _discovery_progress_zips(input_root, check_budget, max_entries):
    candidates = []
    for path in _discovery_scan_paths(input_root, check_budget, max_entries):
        if path.is_file() and path.suffix.lower() == '.zip' and _discovery_progress_zip(path, check_budget, max_entries):
            candidates.append(path)
    return _discovery_unique(candidates)


def _discovery_one(label, candidates):
    candidates = _discovery_unique(candidates)
    if not candidates:
        raise ValueError(f'Auto-discovery found no valid {label}; set an explicit path or add the correct Dataset')
    if len(candidates) > 1:
        listing = '\n'.join('  - ' + str(path) for path in candidates)
        raise ValueError(f'Auto-discovery found multiple {label} candidates; refusing to guess:\n{listing}')
    return str(candidates[0])


def _discovery_explicit_source(value, input_root, expected_manifest_sha256, check_budget, max_entries):
    path = _discovery_inside(value, input_root)
    if path.is_file():
        return str(path)
    if (path / 'SOURCE_MANIFEST.json').is_file():
        return str(path)
    candidates = _discovery_expanded_sources(path, expected_manifest_sha256, check_budget, max_entries)
    return _discovery_one('expanded source', candidates)


def _discovery_explicit_progress(value, input_root, check_budget, max_entries):
    path = _discovery_inside(value, input_root)
    if path.is_file():
        if path.suffix.lower() != '.zip':
            raise ValueError('Explicit progress path must be a ZIP or directory containing workers/')
        return str(path)
    if (path / 'workers' / 'ippo').is_dir() and (path / 'workers' / 'vrpo').is_dir():
        return str(path)
    candidates = _discovery_expanded_progress(path, check_budget, max_entries)
    return _discovery_one('expanded progress', candidates)


def discover_inputs(source_value, progress_value, input_root, check_budget, source_sha256, manifest_sha256, *, max_entries=20000):
    'Find exactly one hash-pinned source and at most one trained progress bundle.'
    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Input root does not exist: ' + str(root))
    if source_value in ('', None):
        source_candidates = _discovery_source_zips(root, source_sha256, check_budget, max_entries)
        source_candidates += _discovery_expanded_sources(root, manifest_sha256, check_budget, max_entries)
        source = _discovery_one('source', source_candidates)
        source_mode = 'auto'
    else:
        source = _discovery_explicit_source(source_value, root, manifest_sha256, check_budget, max_entries)
        source_mode = 'explicit'
    if progress_value is None:
        progress, progress_mode = None, 'fresh_seeded'
    elif progress_value == '':
        progress_candidates = _discovery_expanded_progress(root, check_budget, max_entries)
        progress_candidates += _discovery_progress_zips(root, check_budget, max_entries)
        progress_candidates = _discovery_unique(progress_candidates)
        if len(progress_candidates) > 1:
            listing = '\n'.join('  - ' + str(path) for path in progress_candidates)
            raise ValueError(f'Auto-discovery found multiple progress candidates; refusing to guess:\n{listing}')
        if not progress_candidates:
            raise ValueError('Auto-discovery found no valid progress; set PROGRESS_BUNDLE=None for an explicit fresh seeded run')
        progress = str(progress_candidates[0])
        progress_mode = 'auto'
    else:
        progress = _discovery_explicit_progress(progress_value, root, check_budget, max_entries)
        progress_mode = 'explicit'
    return {'source': source, 'progress': progress, 'source_mode': source_mode, 'progress_mode': progress_mode}
'''

RUN = r'''import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zipfile

EXPECTED_SOURCE_SHA256 = '__SOURCE_SHA256__'
EXPECTED_MANIFEST_SHA256 = '__MANIFEST_SHA256__'
STARTED = time.monotonic()
if type(EXECUTE) is not bool:
    raise ValueError('EXECUTE must be bool')
if isinstance(BUDGET_SECONDS, bool) or not isinstance(BUDGET_SECONDS, (int, float)) or not math.isfinite(BUDGET_SECONDS) or not 0 < BUDGET_SECONDS <= 600:
    raise ValueError('BUDGET_SECONDS must be finite, positive, <=600')
DEADLINE = STARTED + BUDGET_SECONDS
RUN_ROOT = Path('/kaggle/working') / ('guandan-performance-' + uuid.uuid4().hex[:10])
RUN_ROOT.mkdir(parents=True, exist_ok=False)
process = None


def remaining():
    seconds = DEADLINE - time.monotonic()
    if seconds <= 0:
        raise TimeoutError('Whole-notebook short-test budget exhausted')
    return seconds


def input_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(Path('/kaggle/input').resolve()) or not path.exists():
        raise ValueError('Use an existing /kaggle/input path: ' + str(value))
    return path


try:
    if not EXECUTE:
        print(json.dumps({'status':'plan_only', 'shorttest_only':True, 'execute':False,
                          'budget_seconds':BUDGET_SECONDS, 'source_sha256':EXPECTED_SOURCE_SHA256,
                          'workloads':['inference','short-training','short-evaluation'],
                          'comparisons':['single T4','dual T4'], 'recommended':False}, indent=2))
    else:
        discovered = discover_inputs(
            SOURCE_ZIP, PROGRESS_BUNDLE, Path('/kaggle/input').resolve(), remaining,
            EXPECTED_SOURCE_SHA256, EXPECTED_MANIFEST_SHA256,
        )
        SOURCE_ZIP = discovered['source']
        PROGRESS_BUNDLE = discovered['progress']
        print('INPUT_DISCOVERY:', json.dumps(discovered, ensure_ascii=False, indent=2), flush=True)
        source = input_path(SOURCE_ZIP)
        unpack = RUN_ROOT / 'source'
        if source.is_file():
            hasher = hashlib.sha256()
            with source.open('rb') as stream:
                for part in iter(lambda: stream.read(1024**2), b''):
                    remaining()
                    hasher.update(part)
            if hasher.hexdigest() != EXPECTED_SOURCE_SHA256:
                raise ValueError('Wrong source ZIP SHA256')
            with zipfile.ZipFile(source) as archive:
                infos = archive.infolist()
                if len(infos) > 10000 or sum(x.file_size for x in infos) > 128*1024**2:
                    raise ValueError('Source ZIP exceeds limits')
                seen = set()
                for item in infos:
                    relative = PurePosixPath(item.filename)
                    key = item.filename.casefold()
                    if relative.is_absolute() or '..' in relative.parts or chr(92) in item.filename or ':' in item.filename or key in seen:
                        raise ValueError('Unsafe source entry')
                    seen.add(key)
                    target = unpack.joinpath(*relative.parts)
                    if not target.resolve().is_relative_to(unpack.resolve()):
                        raise ValueError('Source escapes output')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as src, target.open('xb') as dst:
                        for part in iter(lambda: src.read(1024**2), b''):
                            remaining()
                            dst.write(part)
            source_root = unpack / 'guandan'
        elif source.is_dir():
            source_root = source
            if not (source_root / 'SOURCE_MANIFEST.json').is_file():
                raise ValueError('Expanded SOURCE_ZIP must directly contain SOURCE_MANIFEST.json (not workers/)')
        else:
            raise ValueError('Source must be ZIP or expanded source directory')
        manifest_file = source_root / 'SOURCE_MANIFEST.json'
        remaining()
        if manifest_file.is_symlink() or manifest_file.stat().st_size > 4*1024**2:
            raise ValueError('Unsafe source manifest')
        raw_manifest = manifest_file.read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != EXPECTED_MANIFEST_SHA256:
            raise ValueError('Wrong expanded source manifest SHA256')
        manifest = json.loads(raw_manifest)
        files = manifest['files']
        if not isinstance(files, dict) or len(files) > 10000:
            raise ValueError('Invalid file manifest')
        actual = set()
        for file in source_root.rglob('*'):
            remaining()
            if file.is_symlink():
                raise ValueError('Linked source entry not allowed')
            if file.is_file():
                actual.add(file.relative_to(source_root).as_posix())
        if actual != set(files) | {'SOURCE_MANIFEST.json'}:
            raise ValueError('Source manifest file-set mismatch')
        total = 0
        for name, expected in files.items():
            remaining()
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or chr(92) in name or ':' in name:
                raise ValueError('Unsafe source manifest path')
            path = source_root.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(source_root.resolve()):
                raise ValueError('Source manifest escapes input')
            total += path.stat().st_size
            if total > 128*1024**2:
                raise ValueError('Expanded source exceeds size bound')
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                for part in iter(lambda: stream.read(1024**2), b''):
                    remaining()
                    hasher.update(part)
            if hasher.hexdigest() != expected:
                raise ValueError('Source file SHA256 mismatch: '+name)
        command = [sys.executable, '-B', '-u', '-m', 'guandan.performance_gate', '--mode', 'remote',
                   '--execute', '--output', str(RUN_ROOT / 'results'),
                   '--budget-seconds', str(max(0, remaining()-3))]
        for flag, value in [('--bundle', PROGRESS_BUNDLE), ('--reference-report', REFERENCE_REPORT), ('--local-report', LOCAL_REPORT)]:
            if value:
                command.extend([flag, str(input_path(value))])
        if BUNDLE_SHA256:
            command.extend(['--bundle-sha256', BUNDLE_SHA256])
        if remaining() <= 3:
            raise TimeoutError('No time left for the gate')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        with (RUN_ROOT / 'console.log').open('w') as log:
            process = subprocess.Popen(command, cwd=source_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            offset = 0
            while process.poll() is None:
                process_timeout = min(2, remaining())
                try:
                    process.wait(timeout=process_timeout)
                except subprocess.TimeoutExpired:
                    pass
                with (RUN_ROOT / 'console.log').open() as read:
                    read.seek(offset)
                    text = read.read()
                    offset = read.tell()
                    if text:
                        print(text, end='', flush=True)
        result_file = RUN_ROOT / 'results/result.json'
        if result_file.exists():
            print(result_file.read_text())
        if process.returncode:
            raise RuntimeError(f'Short-test incomplete/failed, exit={process.returncode}; inspect result and logs')
finally:
    if process is not None and process.poll() is None:
        # Parent handles SIGTERM by cleaning up its independently owned worker groups.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
    # Bounded copy of small reports only; do not rezip datasets/checkpoints/source.
    if EXECUTE:
        with zipfile.ZipFile(str(RUN_ROOT)+'.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in RUN_ROOT.rglob('*'):
                if path.is_file() and 'source' not in path.relative_to(RUN_ROOT).parts and path.suffix in ('.json','.jsonl','.log') and path.stat().st_size <= 8*1024**2:
                    archive.write(path, path.relative_to(RUN_ROOT).as_posix())
        print('Reports:', str(RUN_ROOT)+'.zip')
'''


def render_notebook(source_sha256, manifest_sha256):
    if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in (source_sha256, manifest_sha256)):
        raise ValueError("invalid source SHA256")
    intro = "# T4 单卡/双卡限时性能验证\n默认只生成计划，不训练长任务。先本地验证，再用一次最多600秒的Kaggle短测；没有收益不推荐长跑。每条lane全局工作量完全相同。"
    cells = []
    execution = DISCOVERY + '\n' + RUN.replace("__SOURCE_SHA256__", source_sha256).replace("__MANIFEST_SHA256__", manifest_sha256)
    for index, (kind, text) in enumerate((("markdown", intro), ("code", CONFIG), ("code", execution))):
        cell = {"cell_type": kind, "id": f"performance-{index}", "metadata": {}, "source": text.splitlines(True)}
        if kind == "code":
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}}}


def build(destination, root=ROOT):
    """Snapshot explicit source files, including reviewed uncommitted changes.

    Do not claim the git commit is the source identity: the pinned archive SHA is.
    No uploading and no mutation of old releases, packagers or working source.
    """
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new output directory")
    from guandan.deployment.provenance import DEPLOYMENT_REQUIRED_FILES
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    names = set(DEPLOYMENT_REQUIRED_FILES) | set(NEW_FILES)
    for name in filter(None, tracked):
        path = Path(name)
        if path.parts[0] in ("guandan", "scripts", "notebooks") and path.suffix == ".py":
            names.add(name)
        if path.parts[0] == "configs" and path.suffix == ".json":
            names.add(name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip())
    data = {}
    for name in sorted(names):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Unsafe or missing source file: {name}")
        data[name] = path.read_bytes()
    manifest = {"format": "guandan-source-v1", "git_commit": commit,
                "files": {name:hashlib.sha256(value).hexdigest() for name,value in data.items()},
                "profile": "remote_full", "uploaded": False, "accepted": False, "kaggle_verified": False}
    data["SOURCE_MANIFEST.json"] = json.dumps(manifest, sort_keys=True, indent=2).encode()
    destination.mkdir(parents=True)
    archive_path = destination / "guandan-performance-source.zip"
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(data.items()):
            info = zipfile.ZipInfo("guandan/"+name, (2026,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    source_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(data["SOURCE_MANIFEST.json"]).hexdigest()
    metadata = {"source_sha256": source_sha, "source_manifest_sha256": manifest_sha, "git_commit_provenance_only": commit, "dirty_source": dirty,
                "status": "pending_remote", "recommended": False, "execute_default": False,
                "shorttest_only": True}
    (destination / "guandan_performance.ipynb").write_text(json.dumps(render_notebook(source_sha, manifest_sha), ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "PACKAGE_METADATA.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (destination / "gpu_performance_workflow.md").write_bytes(data["docs/gpu_performance_workflow.md"])
    return metadata


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description="Build pinned shorttest-only notebook; never run/upload")
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
