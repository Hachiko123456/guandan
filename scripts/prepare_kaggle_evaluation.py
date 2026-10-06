"""Build the inference-only Kaggle evaluation package and notebook."""
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

CONFIG = '''# 掼蛋评估 Notebook：不训练，只评估已有训练权重。
# 首次运行建议保持 AUTO_CONTINUE_AFTER_WARMUP=False，只预跑并估时。
EXECUTE = True
SOURCE_DATASET_HINT = ''  # 可选：源码 Dataset 在 /kaggle/input 下的相对路径
INSTALL_DEPS = False
PROGRESS_BUNDLE = ''  # 必填：/kaggle/input/.../guandan-dual-gpu-progress.zip，或含 workers/ 的展开目录
EVAL_PROFILE = 'local_fast'  # 预筛查；正式评估改为 remote_full
WARMUP_GAMES_PER_OPPONENT = 10  # 独立 seed，预跑不计正式统计，可非 4 的倍数
TARGET_GAMES_PER_OPPONENT = 100  # formal 目标，必须是 4 的倍数以平衡 seat rotation
AUTO_CONTINUE_AFTER_WARMUP = False  # False: warmup_complete 正常退出，不自动跑 formal
MAX_ESTIMATED_MINUTES = 60
BATCH_SIZE = 16
'''

RUN = r'''import json
from pathlib import Path
import math
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta

INPUT_ROOT = Path('/kaggle/input').resolve()
RUN_ID = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
OUTPUT_ROOT = WORKING_ROOT / f'guandan-evaluation-{RUN_ID}'
RESULT_ZIP = WORKING_ROOT / f'guandan-evaluation-{RUN_ID}.zip'
LOG_ROOT = WORKING_ROOT / f'guandan-evaluation-logs-{RUN_ID}'


def _validate_config():
    if not isinstance(EXECUTE, bool) or not isinstance(AUTO_CONTINUE_AFTER_WARMUP, bool):
        raise ValueError('EXECUTE 和 AUTO_CONTINUE_AFTER_WARMUP 必须是 True/False')
    for name, value in [('WARMUP_GAMES_PER_OPPONENT', WARMUP_GAMES_PER_OPPONENT),
                        ('TARGET_GAMES_PER_OPPONENT', TARGET_GAMES_PER_OPPONENT),
                        ('BATCH_SIZE', BATCH_SIZE)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} 必须是正整数')
    if TARGET_GAMES_PER_OPPONENT % 4:
        raise ValueError('TARGET_GAMES_PER_OPPONENT 必须是 4 的倍数；warmup 可以不是 4 的倍数')
    if isinstance(MAX_ESTIMATED_MINUTES, bool) or not isinstance(MAX_ESTIMATED_MINUTES, (int, float)) or not math.isfinite(MAX_ESTIMATED_MINUTES) or MAX_ESTIMATED_MINUTES <= 0:
        raise ValueError('MAX_ESTIMATED_MINUTES 必须为正数')
    if EVAL_PROFILE not in ('local_fast', 'remote_full'):
        raise ValueError('EVAL_PROFILE 必须为 local_fast 或 remote_full')
    if not PROGRESS_BUNDLE:
        if not EXECUTE:
            return None
        raise ValueError('请填写 PROGRESS_BUNDLE；不会自动猜测旧 progress')
    progress_path = Path(PROGRESS_BUNDLE).expanduser().resolve()
    if not progress_path.is_relative_to(INPUT_ROOT):
        raise ValueError(f'PROGRESS_BUNDLE 必须位于 /kaggle/input：{progress_path}')
    is_zip = progress_path.is_file() and progress_path.suffix.lower() == '.zip'
    is_expanded = progress_path.is_dir() and (progress_path / 'workers').is_dir()
    if not (is_zip or is_expanded):
        raise ValueError(f'PROGRESS_BUNDLE 必须是 ZIP，或包含 workers/ 的展开进度目录：{progress_path}')
    return progress_path


def _config_payload(progress_path):
    warmup_tasks = 2 * 4
    formal_tasks = 2 * 4
    return {
        'run_id': RUN_ID,
        'profile': EVAL_PROFILE,
        'warmup_games_per_opponent': WARMUP_GAMES_PER_OPPONENT,
        'target_games_per_opponent': TARGET_GAMES_PER_OPPONENT,
        'auto_continue_after_warmup': AUTO_CONTINUE_AFTER_WARMUP,
        'max_estimated_minutes': MAX_ESTIMATED_MINUTES,
        'batch_size': BATCH_SIZE,
        'warmup_total_games': warmup_tasks * WARMUP_GAMES_PER_OPPONENT,
        'formal_total_games': formal_tasks * TARGET_GAMES_PER_OPPONENT,
        'warmup_tasks': warmup_tasks,
        'formal_tasks': formal_tasks,
        'progress_bundle': None if progress_path is None else str(progress_path),
        'devices': {'ippo': 'cuda:0', 'vrpo': 'cuda:1'},
        'eta_rule': 'two-worker wall time uses the slower worker; do not add worker durations',
    }


def _command(progress_path):
    command = [
        sys.executable, '-u', '-B', '-m', 'guandan.evaluation_suite',
        '--progress-bundle', '' if progress_path is None else str(progress_path), '--profile', EVAL_PROFILE,
        '--batch-size', str(BATCH_SIZE), '--output-root', str(OUTPUT_ROOT),
        '--warmup-games-per-opponent', str(WARMUP_GAMES_PER_OPPONENT),
        '--target-games-per-opponent', str(TARGET_GAMES_PER_OPPONENT),
        '--max-estimated-minutes', str(MAX_ESTIMATED_MINUTES),
        '--device-ippo', 'cuda:0', '--device-vrpo', 'cuda:1',
    ]
    if AUTO_CONTINUE_AFTER_WARMUP:
        command.append('--auto-continue-after-warmup')
    if EXECUTE:
        command.append('--execute')
    return command


progress_path = _validate_config()
config = _config_payload(progress_path)
command = _command(progress_path)
print('RUN_ID:', RUN_ID, flush=True)
print('EVALUATION_CONFIG:', json.dumps(config, ensure_ascii=False), flush=True)
print('COMMAND:', command, flush=True)
print('计划：warmup 两张 GPU 共 8 个任务 /', config['warmup_total_games'], '局；formal 共 8 个任务 /', config['formal_total_games'], '局', flush=True)
print('ETA 规则：取 cuda:0 与 cuda:1 两个 worker 中较慢者的墙钟时间，不把两张卡时间相加。', flush=True)

if not EXECUTE:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / 'plan.json').write_text(json.dumps({'status': 'plan_only', 'command': command, **config}, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT_ROOT / 'evaluation_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('EXECUTE=False：仅生成 plan/evaluation_config，不启动 evaluation_suite。', flush=True)
else:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / 'notebook.log'
    stop = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not stop.wait(30):
            progress = OUTPUT_ROOT / 'live' / 'progress.json'
            try:
                payload = json.loads(progress.read_text(encoding='utf-8'))
                age = max(0, time.time() - progress.stat().st_mtime)
                print('[notebook] progress_age_seconds=', round(age, 1),
                      'status=', payload.get('status'),
                      'warmup=', str(payload.get('warmup_completed_games',0))+'/'+str(payload.get('warmup_target_games',0)),
                      'formal=', str(payload.get('formal_completed_games',0))+'/'+str(payload.get('formal_target_games',0)),
                      'ETA_seconds=', payload.get('estimated_remaining_seconds'),
                      'finish=', payload.get('estimated_finish'), 'log=', log_path, flush=True)
                if age > 120:
                    print('[warning] 超过 120 秒没有 progress.json 更新；心跳不等于局数推进。', flush=True)
            except (OSError, ValueError) as exc:
                print('[notebook] elapsed_seconds=', round(time.monotonic() - started, 1),
                      'progress=unavailable', type(exc).__name__, 'ETA=unknown', flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    returncode = None
    try:
        with log_path.open('w', encoding='utf-8') as log:
            log.write('COMMAND: ' + json.dumps(command, ensure_ascii=False) + '\n')
            log.flush()
            with subprocess.Popen(command, cwd=WORK_ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                  errors='replace', bufsize=1) as process:
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
    finally:
        stop.set()
        thread.join(timeout=2)
        # 可捕获的停止、异常和正常退出都先压缩已有文件；硬杀/断电不保证 finally 执行。
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        if OUTPUT_ROOT.is_dir():
            (OUTPUT_ROOT / 'evaluation_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
            if log_path.is_file():
                shutil.copyfile(log_path, OUTPUT_ROOT / 'notebook.log')
            shutil.make_archive(str(RESULT_ZIP.with_suffix('')), 'zip', root_dir=OUTPUT_ROOT)
            print('result ZIP:', RESULT_ZIP, flush=True)
        print('returncode:', returncode, '; log:', log_path, flush=True)

    result_path = OUTPUT_ROOT / 'result.json'
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        print('RESULT:', json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        status = result.get('status')
        if status == 'warmup_complete':
            print('预跑已正常完成。尚未运行 formal；查看 ETA 后，可改 AUTO_CONTINUE_AFTER_WARMUP=True 重新 Run All。', flush=True)
        elif status == 'stopped_by_budget':
            print('预算保护停止，不是模型加载失败；请查看停止原因、估时和实际已完成局数。', flush=True)
        elif status == 'complete':
            print('本次配置的正式评估已完成；不等于模型能力验收通过。', flush=True)
    else:
        print('result.json 尚未生成；请查看 live/progress.json 和 worker 日志。', flush=True)
    if returncode not in (0, 2):
        raise RuntimeError(f'evaluation suite failed: {log_path}')
    if returncode == 2:
        print('评估未全部完成或因预算停止；请根据 result.json 区分 stopped_by_budget / failed。', flush=True)
'''

SUMMARY = '''result_path = OUTPUT_ROOT / 'result.json'
config_path = OUTPUT_ROOT / 'evaluation_config.json'
print('OUTPUT_ROOT:', OUTPUT_ROOT)
print('RESULT_ZIP:', RESULT_ZIP)
if config_path.is_file():
    print('EVALUATION_CONFIG:')
    print(config_path.read_text(encoding='utf-8'))
if result_path.is_file():
    print('RESULT:')
    print(json.dumps(json.loads(result_path.read_text(encoding='utf-8')), ensure_ascii=False, indent=2))
elif not EXECUTE:
    print('plan_only：只生成计划，没有评估或 result.json。')
else:
    print('当前没有 result.json；不能宣称评估完成，请检查 worker 日志。')
if RESULT_ZIP.is_file():
    from IPython.display import FileLink, display
    display(FileLink(str(RESULT_ZIP)))
'''


def render_notebook(metadata: dict, template: str) -> dict:
    """Render an AST-valid notebook from pinned source metadata and templates."""
    intro = '''# 掼蛋模型能力评估 Notebook

本 Notebook **只评估已有权重，不训练模型**，调用源码包中的 `guandan.evaluation_suite`。

必须在 Kaggle 添加两个 Dataset：本交付目录的 `guandan-source.zip`，以及训练后下载的 `guandan-dual-gpu-progress.zip` 或包含 `workers/` 的展开 progress 目录。CONFIG 中必须填写 `/kaggle/input/...` 路径；不会自动猜测旧 progress。

warmup 默认每个对手 10 局，使用独立 seed，不计入 formal 统计，可以不是 4 的倍数；formal target 必须是 4 的倍数以平衡 seat rotation。`AUTO_CONTINUE_AFTER_WARMUP=False` 时，`warmup_complete` 是正常退出，不是卡住，也不代表 formal 已完成。

评估矩阵是两个 GPU worker 各自执行 random、rule、own-old、cross 四类任务：warmup 共 8 个任务/80 局，formal100 共 8 个任务/800 局。ETA 取两个 worker 中较慢者的墙钟时间，不把两张卡时间相加。

首次保持 `EVAL_PROFILE='local_fast'` 做预筛查；确认速度后可改为 `remote_full` 正式运行。profile 只选择运行时兼容策略，局数始终由 target 显式指定，不会从 100 自动扩大为 1000。`stopped_by_budget` 表示预算保护停止；不会请求交互 input。

每次运行都使用新的 `/kaggle/working/guandan-evaluation-<run_id>/`；可捕获退出会在 finally 中压缩已写入文件。硬杀/断电不保证 ZIP 生成，但 working 目录中已经写入的 games、live 和日志仍可能保留。
'''
    bootstrap = template
    for name, key in [('COMMIT', 'source_git_commit'), ('ZIP_SHA256', 'source_zip_sha256'),
                      ('MANIFEST_SHA256', 'source_manifest_sha256'), ('CHECKER_SHA256', 'checker_sha256')]:
        bootstrap = bootstrap.replace(f'__{name}__', metadata[key])
    cells = []
    for index, (kind, text) in enumerate([('markdown', intro), ('code', CONFIG), ('code', bootstrap), ('code', RUN), ('code', SUMMARY)]):
        cell = {'id': f'evaluation-{index}', 'cell_type': kind, 'metadata': {}, 'source': text.splitlines(True)}
        if kind == 'code':
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                         'language_info': {'name': 'python', 'version': '3.12'}}, 'cells': cells}


def build(destination: str | Path) -> dict:
    from guandan.deployment.package import build_package

    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('use a new release directory')
    package = build_package(ROOT, destination / 'guandan-source.zip')
    with zipfile.ZipFile(package['archive']) as bundle:
        manifest = bundle.read('guandan/SOURCE_MANIFEST.json')
        checker = bundle.read('guandan/scripts/kaggle_environment_check.py')
        template = bundle.read('guandan/notebooks/kaggle_unified_bootstrap.py').decode('utf-8-sig')
        tutorial = bundle.read('guandan/docs/kaggle_evaluation_tutorial.md').decode('utf-8-sig')
    metadata = {'source_git_commit': package['git_commit'], 'source_zip_sha256': package['sha256'],
                'source_manifest_sha256': hashlib.sha256(manifest).hexdigest(),
                'checker_sha256': hashlib.sha256(checker).hexdigest(), 'profile': 'local_fast',
                'profile_version': 'profiles-0.2', 'accepted': False, 'uploaded': False, 'kaggle_verified': False}
    destination.mkdir(parents=True, exist_ok=True)
    notebook = render_notebook(metadata, template)
    (destination / 'guandan_evaluation.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (destination / 'PACKAGE_METADATA.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (destination / '使用说明.txt').write_text(tutorial, encoding='utf-8')
    (destination / 'kaggle_evaluation_tutorial.md').write_text(tutorial, encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export the committed inference-only Kaggle evaluation package')
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False, indent=2))
