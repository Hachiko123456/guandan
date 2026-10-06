"""Bounded progress-bundle extraction and checkpoint integrity checks for evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

MAX_UNPACKED = 2 * 1024**3
MAX_FILES = 10000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_progress(source, destination):
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        return source
    if not source.is_file() or source.suffix.lower() != '.zip':
        raise ValueError('progress input must be an existing ZIP or the expanded directory containing workers/')
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('extraction destination exists; use a new evaluation output directory')
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES or sum(x.file_size for x in infos) > MAX_UNPACKED:
            raise ValueError('progress ZIP exceeds extraction limits')
        seen = set()
        for item in infos:
            name = item.orig_filename
            path = PurePosixPath(name)
            key = name.rstrip('/').casefold()
            mode = (item.external_attr >> 16) & 0xffff
            if (not key or key in seen or chr(92) in name or ':' in name or path.is_absolute()
                    or '..' in path.parts or name.rstrip('/') != path.as_posix()
                    or stat.S_ISLNK(mode) or item.flag_bits & 1):
                raise ValueError(f'unsafe or duplicate progress ZIP entry: {name!r}')
            seen.add(key)
        destination.mkdir(parents=True)
        for item in infos:
            target = destination.joinpath(*PurePosixPath(item.filename).parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError('progress ZIP path escapes output')
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
    return destination


def checkpoint_for(root, algorithm, update):
    root = Path(root).resolve()
    scope = root / 'workers' / algorithm / 'remote_full' / algorithm
    files = list(scope.glob(f'{algorithm}_update_{update:06d}_*.pt'))
    if len(files) != 1:
        raise ValueError(f'expected one {algorithm} update{update} under {scope}; found {len(files)}')
    checkpoint = files[0]
    manifest_path = checkpoint.with_suffix('.recovery.json')
    if checkpoint.is_symlink() or manifest_path.is_symlink() or not checkpoint.resolve().is_relative_to(root) or not manifest_path.resolve().is_relative_to(root):
        raise ValueError('checkpoint/manifest escapes progress input')
    if not manifest_path.is_file() or manifest_path.stat().st_size > 1024**2:
        raise ValueError('missing or oversized checkpoint recovery manifest')
    m = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (m.get('checkpoint_filename') != checkpoint.name or m.get('algorithm') != algorithm
            or type(m.get('durable_updates')) is not int or m['durable_updates'] != update):
        raise ValueError('checkpoint manifest identity mismatch')
    if m.get('checkpoint_sha256') != sha256(checkpoint):
        raise ValueError('checkpoint SHA256 mismatch')
    return checkpoint.resolve()


def prepare_checkpoints(bundle, destination):
    root = extract_progress(bundle, destination)
    return {f'{alg}{update}': checkpoint_for(root, alg, update)
            for alg in ('ippo','vrpo') for update in (100,102)}



