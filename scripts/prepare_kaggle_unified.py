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

CONFIG = '''# 一份 Notebook；首次只添加源码 Dataset，点击 Run All 即按顺序执行。
EXECUTE = True  # False 只看计划，不训练
SOURCE_DATASET_HINT = ""  # 自动匹配本次源码；多个匹配时填写相对 /kaggle/input 路径
RESUME_BUNDLE = ""  # 自动找唯一进度包；多份时显式填写 /kaggle/input/... 路径
INSTALL_DEPS = False  # 缺依赖时显式改 True（仅 wheel，不自动替换规则/算法）
TOTAL_SESSION_HOURS = 10  # 整个流水线共用，不是每个算法10小时；还须考虑已用会话时间
SAVE_MARGIN_SECONDS = 300
CHECKPOINT_SECONDS = 600

import time
# 同一个 kernel 重跑 Run All 不重新获得10小时；新会话才重新开始计时。
if "GUANDAN_NOTEBOOK_STARTED" not in globals():
    GUANDAN_NOTEBOOK_STARTED = time.monotonic()
'''

RUN = '''import os
import uuid

PIPELINE_ROOT = WORKING_ROOT / "guandan-unified"
PROGRESS_ZIP = WORKING_ROOT / "guandan-unified-progress.zip"


def find_progress():
    if RESUME_BUNDLE:
        path = Path(RESUME_BUNDLE).resolve()
        if not path.is_relative_to(INPUT_ROOT.resolve()) or not path.exists():
            raise ValueError("RESUME_BUNDLE必须在 /kaggle/input")
        return path
    candidates = []
    for path in INPUT_ROOT.rglob("*-progress.zip"):
        with zipfile.ZipFile(path) as archive:
            if "WORKFLOW_BUNDLE.json" not in archive.namelist():
                continue
            if archive.getinfo("WORKFLOW_BUNDLE.json").file_size > 16*1024**2:
                raise ValueError("进度包manifest过大")
            metadata = json.loads(archive.read("WORKFLOW_BUNDLE.json"))
        if metadata.get("source_git_commit") == EXPECTED_COMMIT:
            candidates.append(path)
    for path in INPUT_ROOT.rglob("WORKFLOW_BUNDLE.json"):
        if path.stat().st_size > 16*1024**2:
            raise ValueError("进度包manifest过大")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("source_git_commit") == EXPECTED_COMMIT:
            candidates.append(path.parent)
    if len(candidates) > 1:
        raise ValueError("有多个相符进度包，请设置 RESUME_BUNDLE；不会猜测最新文件")
    return candidates[0] if candidates else None


remaining_seconds = TOTAL_SESSION_HOURS*3600 - (time.monotonic()-GUANDAN_NOTEBOOK_STARTED)
if remaining_seconds <= SAVE_MARGIN_SECONDS:
    raise RuntimeError("本Notebook整体时间预算已用尽，请先保存输出，在新会话恢复")
command = [sys.executable, "-u", "-B", "-m", "guandan.deployment.workflow",
           "--output-root", str(PIPELINE_ROOT), "--session-hours", str(remaining_seconds/3600),
           "--save-margin-seconds", str(SAVE_MARGIN_SECONDS),
           "--checkpoint-seconds", str(CHECKPOINT_SECONDS)]
if EXECUTE:
    command.append("--execute")
    if PIPELINE_ROOT.exists():
        if RESUME_BUNDLE:
            raise RuntimeError("已有本机进度不能被输入包覆盖；请用新会话恢复该包")
        print("继续当前working内经过验证的进度。")
    else:
        progress = find_progress()
        if progress:
            command += ["--resume-bundle", str(progress)]
            print("恢复进度包：", progress)
        else:
            # 不误把旧源码进度丢弃后悄悄重新训练。
            if list(INPUT_ROOT.rglob("*-progress.zip")) or list(INPUT_ROOT.rglob("WORKFLOW_BUNDLE.json")):
                raise RuntimeError("存在进度包但源码commit不匹配。请重新挂载生成该进度的同一版本源码ZIP。")
            print("没有进度包，从IPPO开始；随后自动VRPO和评估。")

log_root = WORKING_ROOT / "guandan-unified-logs"
log_root.mkdir(exist_ok=True)
log_path = log_root / (uuid.uuid4().hex + ".log")
print("COMMAND:", command)
# stdout实时显示并落盘。每60秒心跳不代表新的更新数或训练完成。
import threading
stop_heartbeat = threading.Event()

def heartbeat():
    while not stop_heartbeat.wait(60):
        print(f"[notebook] 子进程运行中，累计 {int(time.monotonic()-GUANDAN_NOTEBOOK_STARTED)} 秒；日志 {log_path}", flush=True)

thread = threading.Thread(target=heartbeat, daemon=True)
thread.start()
try:
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=WORK_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except KeyboardInterrupt:
                # 给子进程软停止请求，留时间保存；不伪装强杀可恢复。
                process.terminate()
                returncode = process.wait()
                raise
finally:
    stop_heartbeat.set()
print("returncode:", returncode)
print("workflow summary:", PIPELINE_ROOT / "workflow.json")
print("progress ZIP:", PROGRESS_ZIP)
print("log:", log_path)
if returncode not in (0, 2):
    raise RuntimeError(f"流水线显式失败，停止后续阶段。日志: {log_path}")
if returncode == 2:
    print("未全部完成：下载进度ZIP；新Kaggle会话挂载同一源码ZIP和进度ZIP，再用本Notebook Run All。")
'''

SUMMARY = '''summary_path = PIPELINE_ROOT / "workflow.json"
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps({key: summary.get(key) for key in
                     ("status", "active", "error", "elapsed_seconds", "accepted", "kaggle_verified",
                      "genuine_new_kaggle_session_verified")}, ensure_ascii=False, indent=2))
if PROGRESS_ZIP.is_file():
    from IPython.display import FileLink, display
    # 仅生成Notebook本地下载链接，不上传、不创建Dataset。
    display(FileLink(str(PROGRESS_ZIP)))
'''


def render_notebook(metadata, template):
    intro = '''# 掼蛋统一训练 Notebook：一次 Run All 自动串联

只导入这一个 Notebook，添加配套 `guandan-source.zip` Dataset 后启用GPU并 Run All：

环境检查 → A04 smoke → IPPO 100更新 → 保存/重载追加2更新 → VRPO同样流程 → 两算法A06评估。
每算法评估3000局，目标不会因时间不够缩减；这是管线验证，不是模型强度证明。

**整个流程共用一个时间预算，不能承诺一次Kaggle会话一定跑完。**
保存 `guandan-unified-progress.zip` 后，新会话只需挂载该包和同一源码ZIP，再 Run All。
已完成训练/评估会核对checkpoint与真实记录后跳过；中断的算法评估保留记录但从头重跑，
不会拼接缺失局数。同会话checkpoint重载不是“新Kaggle会话恢复验收”，accepted始终false。

默认EXECUTE=True会训练；只看计划请改False。代码不会自动上传Kaggle或重新创建远程会话。
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
2. guandan_all_in_one.ipynb：导入Notebook，启用GPU，点击Run All。

不要混用之前的ZIP；本Notebook绑定这次commit/hash。
自动执行IPPO100+2、VRPO100+2、两算法各3000局评估。默认不安装依赖。
全流程共用10小时时间预算（还须结合你会话已使用的时间），保存余量5分钟。
如未完成，保存 /kaggle/working/guandan-unified-progress.zip 和日志。
新会话挂载同一源码ZIP和进度ZIP，用同一个Notebook Run All继续。
已完成评估会跳过；不完整的算法评估保留记录但从头重跑。
同会话重载≠新Kaggle会话验收；所有accepted仍false。
不要自动上传，不改规则、不用FableDan、不宣称强度。
''',encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.output),ensure_ascii=False,indent=2))
