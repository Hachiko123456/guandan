"""Offline source export, no network/upload. Only committed tracked files are packaged."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

from .provenance import MANIFEST


def build_package(root: Path, output: Path) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    status = subprocess.check_output(['git','status','--porcelain'],cwd=root).strip()
    if status:
        raise ValueError('commit/review source changes before exporting a clean deployment package')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip()
    files = subprocess.check_output(['git','ls-tree','-r','--name-only','-z',commit],cwd=root).decode().split('\0')
    manifest = {'format':'guandan-source-v1','git_commit':commit,'files':{},'profile':'remote_full',
                'kaggle_verified':False,'accepted':False}
    data = {}
    blocked = {'.git', 'runs','checkpoints','logs','wandb','__pycache__'}
    for name in filter(None, files):
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or blocked.intersection(path.parts) or name.startswith('project_status/history/'):
            continue
        if path.suffix.lower() in {'.pt','.pth','.pem','.key'} or path.name in {'.env','kaggle.json', MANIFEST}:
            raise ValueError(f'sensitive/derived tracked file is not allowed in source export: {name}')
        mode = subprocess.check_output(['git','ls-tree',commit,'--',name],cwd=root).decode().split()[0]
        if mode not in {'100644','100755'}:
            raise ValueError(f'unsupported linked/submodule source: {name}')
        content = subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=root)
        manifest['files'][name] = hashlib.sha256(content).hexdigest()
        data[name] = content
    profile = json.loads(data['configs/acceptance_profiles.json'])
    if profile['version'] != 'profiles-0.2':
        raise ValueError('deployment requires profiles-0.2')
    output.parent.mkdir(parents=True, exist_ok=True)
    # ZIP with fixed metadata, so equal source commits yield reproducible bytes.
    with output.open('xb') as raw:
        with zipfile.ZipFile(raw, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
            data[MANIFEST] = json.dumps(manifest,sort_keys=True,indent=2).encode('utf-8')
            for name in sorted(data):
                info = zipfile.ZipInfo('guandan/'+name,date_time=(2026,1,1,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data[name])
    return {'archive':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'git_commit':commit,'file_count':len(data),'uploaded':False,'kaggle_verified':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Export committed pure-Python source; never upload')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args(argv)
    print(json.dumps(build_package(args.root,args.output),indent=2))


if __name__=='__main__':main()
