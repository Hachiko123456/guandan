"""Build the inference-only Kaggle evaluation package and notebook."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = '''# 评估 Notebook：不训练，只评估已有 progress bundle。
EXECUTE = True
SOURCE_DATASET_HINT = ''
INSTALL_DEPS = False
PROGRESS_BUNDLE = ''  # /kaggle/input 下的 guandan-dual-gpu-progress.zip
EVAL_PROFILE = 'remote_full'  # quick 预筛查可改为 local_fast
BATCH_SIZE = 16
'''
RUN = '''import json
import shutil
import subprocess
import threading
import time
import uuid

if not PROGRESS_BUNDLE:
    raise ValueError('请在 CONFIG 中填写 PROGRESS_BUNDLE，指向 /kaggle/input 下的训练 progress ZIP')
OUTPUT_ROOT = WORKING_ROOT / 'guandan-ability-evaluation'
RESULT_ZIP = WORKING_ROOT / 'guandan-ability-evaluation.zip'
command = [sys.executable, '-u', '-B', '-m', 'guandan.evaluation_suite',
           '--progress-bundle', str(Path(PROGRESS_BUNDLE).resolve()),
           '--profile', EVAL_PROFILE, '--batch-size', str(BATCH_SIZE),
           '--output-root', str(OUTPUT_ROOT), '--execute',
           '--device-ippo', 'cuda:0', '--device-vrpo', 'cuda:1']
print('COMMAND:', command)
log_root = WORKING_ROOT / 'guandan-ability-evaluation-logs'; log_root.mkdir(exist_ok=True)
log_path = log_root / (uuid.uuid4().hex + '.log')
stop = threading.Event()
def heartbeat():
    while not stop.wait(60):
        print(f'[notebook] evaluation suite alive; log={log_path}', flush=True)
threading.Thread(target=heartbeat, daemon=True).start()
returncode = 0
try:
    with log_path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=WORK_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            returncode = process.wait()
finally:
    stop.set()
if OUTPUT_ROOT.exists():
    shutil.make_archive(str(RESULT_ZIP.with_suffix('')), 'zip', root_dir=OUTPUT_ROOT)
print('returncode:', returncode)
print('result:', OUTPUT_ROOT / 'result.json')
print('result ZIP:', RESULT_ZIP)
print('log:', log_path)
if returncode not in (0, 2):
    raise RuntimeError(f'evaluation suite failed: {log_path}')
'''
SUMMARY = '''result_path = OUTPUT_ROOT / 'result.json'
if result_path.is_file():
    print(json.dumps(json.loads(result_path.read_text(encoding='utf-8')), ensure_ascii=False, indent=2))
if RESULT_ZIP.is_file():
    from IPython.display import FileLink, display
    display(FileLink(str(RESULT_ZIP)))
'''

def render_notebook(metadata, template):
    intro = '''# 掼蛋模型能力评估 Notebook\n\n本 Notebook 不训练模型，只读取双 GPU 训练生成的 progress ZIP。\n\n评估内容：\n- IPPO update102 vs IPPO update100\n- VRPO update102 vs VRPO update100\n- IPPO update102 vs VRPO update102\n- VRPO update102 vs IPPO update102\n\n输出包括每个对手的胜率、平均 token/commit steps、按 base deal 聚类的 bootstrap 区间和完整逐局证据。\n'''
    bootstrap = template
    for name, key in [('COMMIT','source_git_commit'),('ZIP_SHA256','source_zip_sha256'),('MANIFEST_SHA256','source_manifest_sha256'),('CHECKER_SHA256','checker_sha256')]:
        bootstrap = bootstrap.replace(f'__{name}__', metadata[key])
    cells=[]
    for index,(kind,text) in enumerate([('markdown',intro),('code',CONFIG),('code',bootstrap),('code',RUN),('code',SUMMARY)]):
        cell={'id':f'evaluation-{index}','cell_type':kind,'metadata':{},'source':text.splitlines(True)}
        if kind=='code':
            ast.parse(text); cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}},'cells':cells}

def build(destination):
    from guandan.deployment.package import build_package
    destination=Path(destination).resolve()
    if destination.exists(): raise FileExistsError('use a new release directory')
    package=build_package(ROOT,destination/'guandan-source.zip')
    with zipfile.ZipFile(package['archive']) as bundle:
        manifest=bundle.read('guandan/SOURCE_MANIFEST.json')
        checker=bundle.read('guandan/scripts/kaggle_environment_check.py')
        template=bundle.read('guandan/notebooks/kaggle_unified_bootstrap.py').decode('utf-8-sig')
    metadata={'source_git_commit':package['git_commit'],'source_zip_sha256':package['sha256'],'source_manifest_sha256':hashlib.sha256(manifest).hexdigest(),'checker_sha256':hashlib.sha256(checker).hexdigest(),'profile':'remote_full','profile_version':'profiles-0.2','accepted':False,'uploaded':False,'kaggle_verified':False}
    notebook=render_notebook(metadata,template)
    (destination/'guandan_evaluation.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'PACKAGE_METADATA.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'使用说明.txt').write_text('''1. 上传 guandan-source.zip 为私有 Dataset。\n2. 上传/挂载本次训练生成的 guandan-dual-gpu-progress.zip。\n3. 导入 guandan_evaluation.ipynb，在 CONFIG 填写 PROGRESS_BUNDLE。\n4. 启用 T4 x2 后 Run All。\n5. 默认 remote_full；先预筛查可改 EVAL_PROFILE=local_fast。\n''',encoding='utf-8')
    return metadata

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); print(json.dumps(build(parser.parse_args().output),ensure_ascii=False,indent=2))
