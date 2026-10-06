"""Build a clean source ZIP and ONE importable Run-All notebook, offline only."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = '''# 一份 Notebook；只添加本次源码 Dataset，启用两张 GPU 后 Run All。
EXECUTE = True  # False 只生成计划，不启动训练
SOURCE_DATASET_HINT = ''  # 源码 Dataset 的相对 /kaggle/input 路径
INSTALL_DEPS = False  # 缺依赖时才显式改 True
RESUME_IPPO = ''  # 可选：明确指向本次 semantic-compatible checkpoint 的 /kaggle/input 路径
RESUME_VRPO = ''  # 可选：明确指向本次 semantic-compatible checkpoint 的 /kaggle/input 路径
RESUME_BUNDLE = ''  # 可选：本 runner 生成的 progress ZIP 或已展开目录；不会自动猜测
BASE_UPDATES = None  # None=remote_full profile；短实验可显式设置
RESUME_UPDATES = None
'''


RUN = '''import os
import json
import uuid
import shutil
import subprocess
import threading
import time

EXPERIMENT_ROOT = WORKING_ROOT / 'guandan-dual-gpu'
PROGRESS_ZIP = WORKING_ROOT / 'guandan-dual-gpu-progress.zip'

def resolve_resume(bundle, algorithm):
    if not bundle:
        return None
    source = Path(bundle).resolve()
    if not source.is_relative_to(INPUT_ROOT.resolve()) or not source.exists():
        raise ValueError('RESUME_BUNDLE 必须位于 /kaggle/input 且存在')
    root = WORKING_ROOT / 'guandan-resume-bundle'
    if source.is_file() and source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as archive:
            archive.extractall(root)
    elif source.is_dir():
        root = source
    else:
        raise ValueError('RESUME_BUNDLE 必须是 ZIP 或已展开目录')
    candidates = list((root / 'workers' / algorithm).rglob('*.pt'))
    if not candidates:
        raise ValueError(f'progress bundle 中找不到 {algorithm} checkpoint')
    def update_number(path):
        import re
        matches = re.findall(r'_update_(\\d+)_', path.name)
        return int(matches[-1]) if matches else -1
    return str(max(candidates, key=lambda item: (update_number(item), item.name)).resolve())

if RESUME_BUNDLE:
    RESUME_IPPO = RESUME_IPPO or resolve_resume(RESUME_BUNDLE, 'ippo')
    RESUME_VRPO = RESUME_VRPO or resolve_resume(RESUME_BUNDLE, 'vrpo')

command = [sys.executable, '-u', '-B', '-m', 'guandan.experiment',
           '--profile', 'remote_full', '--output-root', str(EXPERIMENT_ROOT),
           '--device-ippo', 'cuda:0', '--device-vrpo', 'cuda:1']
if BASE_UPDATES is not None:
    command += ['--base-updates', str(BASE_UPDATES)]
if RESUME_UPDATES is not None:
    command += ['--resume-updates', str(RESUME_UPDATES)]
if RESUME_IPPO:
    command += ['--resume-ippo', str(Path(RESUME_IPPO).resolve())]
if RESUME_VRPO:
    command += ['--resume-vrpo', str(Path(RESUME_VRPO).resolve())]
if EXECUTE:
    command.append('--execute')
print('COMMAND:', command)
log_root = WORKING_ROOT / 'guandan-dual-gpu-logs'
log_root.mkdir(exist_ok=True)
log_path = log_root / (uuid.uuid4().hex + '.log')
stop_heartbeat = threading.Event()

def heartbeat():
    while not stop_heartbeat.wait(60):
        print(f'[notebook] dual-GPU runner alive; log={log_path}', flush=True)

thread = threading.Thread(target=heartbeat, daemon=True)
thread.start()
returncode = 0
try:
    with log_path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=WORK_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
finally:
    stop_heartbeat.set()
if EXPERIMENT_ROOT.exists():
    archive_base = str(PROGRESS_ZIP.with_suffix(''))
    shutil.make_archive(archive_base, 'zip', root_dir=EXPERIMENT_ROOT)
print('returncode:', returncode)
print('experiment result:', EXPERIMENT_ROOT / 'result.json')
print('progress ZIP:', PROGRESS_ZIP)
print('log:', log_path)
if returncode not in (0, 2):
    raise RuntimeError(f'dual-GPU runner failed; log={log_path}')
'''


SUMMARY = '''result_path = EXPERIMENT_ROOT / 'result.json'
if result_path.is_file():
    result = json.loads(result_path.read_text(encoding='utf-8'))
    print(json.dumps({key: result.get(key) for key in
                      ('status', 'elapsed_seconds', 'accepted', 'kaggle_verified')},
                     ensure_ascii=False, indent=2))
if PROGRESS_ZIP.is_file():
    from IPython.display import FileLink, display
    display(FileLink(str(PROGRESS_ZIP)))
'''



def render_notebook(metadata, template):
    intro = '''# 掼蛋双 GPU 训练 Notebook

本 Notebook 只绑定本次源码 ZIP，并通过显式 spawn worker 使用 T4 x2：

- IPPO -> cuda:0
- VRPO -> cuda:1

训练 checkpoint、worker 日志、system telemetry 和 progress ZIP 均由新的 dual-GPU runner 生成。
旧的 progress ZIP 不会被自动猜测；如需恢复，必须在 CONFIG 中分别填写 RESUME_IPPO/RESUME_VRPO。
训练状态的 resume 以 semantic contract 为准，源码 commit 仅作为 provenance。

默认 remote_full 是运行管线规模，不代表模型强度或 Kaggle 验收；任何 GPU/会话失败都保持 incomplete/failed，绝不自动降级到单卡。
'''
    bootstrap = template
    for name, key in [('COMMIT', 'source_git_commit'), ('ZIP_SHA256', 'source_zip_sha256'),
                      ('MANIFEST_SHA256', 'source_manifest_sha256'), ('CHECKER_SHA256', 'checker_sha256')]:
        bootstrap = bootstrap.replace(f'__{name}__', metadata[key])
    cells=[]
    for index,(kind,text) in enumerate([('markdown',intro),('code',CONFIG),('code',bootstrap),
                                       ('code',RUN),('code',SUMMARY)]):
        cell={'id':f'unified-{index}', 'cell_type':kind,'metadata':{},'source':text.splitlines(True)}
        if kind=='code':
            ast.parse(text)
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    return {'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python','version':'3.12'}},'cells':cells}


def build(destination):
    from guandan.deployment.package import build_package
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('use a new release directory; preserve old source/checkpoint pairs')
    # Builder refuses dirty source before creating any deployment bundle.
    package = build_package(ROOT, destination/'guandan-source.zip')
    with zipfile.ZipFile(package['archive']) as bundle:
        manifest = bundle.read('guandan/SOURCE_MANIFEST.json')
        checker = bundle.read('guandan/scripts/kaggle_environment_check.py')
        template = bundle.read('guandan/notebooks/kaggle_unified_bootstrap.py').decode('utf-8-sig')
    metadata = {'source_git_commit': package['git_commit'], 'source_zip_sha256':package['sha256'],
                'source_manifest_sha256':hashlib.sha256(manifest).hexdigest(),
                'checker_sha256':hashlib.sha256(checker).hexdigest(),
                'profile':'remote_full', 'profile_version':'profiles-0.2',
                'accepted':False,'uploaded':False,'kaggle_verified':False}
    notebook=render_notebook(metadata, template)
    (destination/'guandan_all_in_one.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'PACKAGE_METADATA.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (destination/'使用说明.txt').write_text('''只需要这两个文件：
1. guandan-source.zip：上传为私有Dataset，然后添加到Notebook。
2. guandan_all_in_one.ipynb：导入Notebook，启用两张GPU，点击Run All。

Notebook 使用 guandan.experiment：IPPO 固定 cuda:0，VRPO 固定 cuda:1，不会静默降级单卡。
默认执行 remote_full 训练规模；accepted=false、kaggle_verified=false 是正确状态，不代表模型强度。
如果会话中断，下载 guandan-dual-gpu-progress.zip；下次挂载后在 CONFIG 中填写 RESUME_BUNDLE，
或分别填写 RESUME_IPPO/RESUME_VRPO。不会自动猜测旧 progress ZIP。
resume 只接受当前 semantic contract；旧 checkpoint 不会被训练恢复路径隐式迁移。
所有 worker 日志、telemetry、checkpoint 和 result 都位于 /kaggle/working/guandan-dual-gpu。
不要自动上传，不改规则、不用FableDan、不宣称强度。
''',encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.output),ensure_ascii=False,indent=2))
