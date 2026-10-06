import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from guandan.evaluation_inputs import extract_progress, checkpoint_for


@pytest.mark.parametrize('name', ['../outside','/absolute', 'a\\b', 'C:/file', 'a/../file'])
def test_unsafe_zip_rejected_before_extraction(tmp_path, name):
    z=tmp_path/'input.zip'
    with zipfile.ZipFile(z,'w') as archive:
        info=zipfile.ZipInfo('placeholder')
        info.filename=name  # bypass Windows ZipInfo separator normalization
        archive.writestr(info,b'x')
    with pytest.raises(ValueError):
        extract_progress(z,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_zip_and_expanded_supported(tmp_path):
    z=tmp_path/'input.zip'
    with zipfile.ZipFile(z,'w') as archive:
        archive.writestr('workers/',b'')
        archive.writestr('workers/a.json',b'{}')
    root=extract_progress(z,tmp_path/'out')
    assert (root/'workers/a.json').read_bytes()==b'{}'
    assert extract_progress(root,tmp_path/'unused')==root
    with pytest.raises(FileExistsError):
        extract_progress(z,root)


def test_checkpoint_digest_identity_and_ambiguity(tmp_path):
    p=tmp_path/'workers/ippo/remote_full/ippo/ippo_update_000102_example.pt'
    p.parent.mkdir(parents=True)
    p.write_bytes(b'checkpoint bytes')
    m=dict(checkpoint_filename=p.name,algorithm='ippo',durable_updates=102,
           checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    p.with_suffix('.recovery.json').write_text(json.dumps(m))
    assert checkpoint_for(tmp_path,'ippo',102)==p
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA256'):
        checkpoint_for(tmp_path,'ippo',102)
    (p.parent/'ippo_update_000102_other.pt').write_bytes(b'x')
    with pytest.raises(ValueError,match='expected one'):
        checkpoint_for(tmp_path,'ippo',102)
