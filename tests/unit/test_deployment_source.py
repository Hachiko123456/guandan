"""Portable committed source export and provenance tests. No Kaggle/network use."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from guandan.deployment.package import build_package
from guandan.deployment.provenance import source_provenance


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args]).decode('utf-8').strip()


def fixture_repo(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    git(root,'init','-q')
    git(root,'config','user.name','Test Fixture')
    git(root,'config','user.email','fixture@example.invalid')
    (root/'configs').mkdir()
    (root/'guandan').mkdir()
    (root/'configs/acceptance_profiles.json').write_text(json.dumps({'version':'profiles-0.2'}))
    (root/'guandan/__init__.py').write_text('VALUE = 17\n')
    git(root,'add','.')
    git(root,'commit','-qm','fixture source')
    return root


def test_source_archive_is_reproducible_and_gitless_verifiable(tmp_path):
    root=fixture_repo(tmp_path)
    first=build_package(root,tmp_path/'first.zip')
    second=build_package(root,tmp_path/'second.zip')
    assert first['sha256']==second['sha256']
    assert first['uploaded'] is False and first['kaggle_verified'] is False
    target=tmp_path/'unpacked'
    with zipfile.ZipFile(first['archive']) as archive:
        assert all('/.git/' not in name for name in archive.namelist())
        archive.extractall(target)
    exported=target/'guandan'
    metadata=source_provenance(exported)
    assert metadata['git_commit']==git(root,'rev-parse','HEAD')
    assert metadata['source_kind']=='verified_export'
    assert not (exported/'.git').exists()
    assert (exported/'configs/acceptance_profiles.json').is_file()
    with pytest.raises(FileExistsError):
        build_package(root,tmp_path/'first.zip')


def test_export_rejects_uncommitted_inputs_and_tamper(tmp_path):
    root=fixture_repo(tmp_path)
    (root/'extra.py').write_text('unreviewed = True')
    with pytest.raises(ValueError,match='commit'):
        build_package(root,tmp_path/'invalid.zip')
    git(root,'add','extra.py');git(root,'commit','-qm','review extra')
    result=build_package(root,tmp_path/'valid.zip')
    with zipfile.ZipFile(result['archive']) as archive:
        archive.extractall(tmp_path/'unpacked')
    exported=tmp_path/'unpacked/guandan'
    (exported/'extra.py').write_text('tampered = True')
    with pytest.raises(ValueError,match='mismatch'):
        source_provenance(exported)


@pytest.mark.parametrize('filename',['secret.pem','kaggle.json','.env','model.pt'])
def test_secret_and_checkpoint_files_are_never_exported(tmp_path, filename):
    root=fixture_repo(tmp_path)
    (root/filename).write_text('synthetic sentinel not a real secret')
    git(root,'add','.');git(root,'commit','-qm','synthetic unsafe fixture')
    with pytest.raises(ValueError,match='sensitive'):
        build_package(root,tmp_path/'unsafe.zip')


def test_manifest_path_escape_is_rejected(tmp_path):
    root=tmp_path/'exported';root.mkdir()
    outside=tmp_path/'outside.py';outside.write_text('VALUE = 1')
    manifest={'format':'guandan-source-v1','git_commit':'a'*40,
              'files':{'../outside.py':hashlib.sha256(outside.read_bytes()).hexdigest()}}
    (root/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='mismatch'):
        source_provenance(root)


def test_git_checkout_provenance_is_actual_commit_and_dirty_flag(tmp_path):
    root=fixture_repo(tmp_path)
    clean=source_provenance(root)
    assert clean['source_kind']=='git' and clean['git_worktree_clean'] is True
    (root/'dirty').write_text('x')
    assert source_provenance(root)['git_worktree_clean'] is False
