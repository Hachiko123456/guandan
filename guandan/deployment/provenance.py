"""Source provenance works in either a Git checkout or a verified source export."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

MANIFEST = 'SOURCE_MANIFEST.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_provenance(root: Path) -> dict:
    root = Path(root).resolve()
    if (root / MANIFEST).is_file():
        manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
        if manifest.get('format') != 'guandan-source-v1' or not re.fullmatch('[0-9a-f]{40}', manifest.get('git_commit','')):
            raise ValueError('invalid source manifest/version/commit')
        for name, digest in manifest['files'].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
                raise ValueError(f'source manifest hash/path mismatch: {name}')
        return {'git_commit': manifest['git_commit'], 'source_kind': 'verified_export',
                'source_manifest_sha256': sha256(root / MANIFEST), 'git_worktree_clean': True}
    proc = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True)
    status = subprocess.run(['git','status','--porcelain'],cwd=root,capture_output=True,text=True,check=True)
    commit = proc.stdout.strip()
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('invalid source commit')
    return {'git_commit':commit,'source_kind':'git','git_worktree_clean':not bool(status.stdout.strip()),
            'source_manifest_sha256':None}
