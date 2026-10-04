# %% [markdown]
# # 掼蛋 A07：只读来源、可写会话、显式执行
#
# 普通 Python 源码模板；不含密钥、不联网上传、不自动安装或训练。
# 将审阅后的离线 source export 解压到输入目录。检查点 bundle 是另一个
# 输入目录，不要把它作为源码安装，也不要把 source 与 output 放在同一目录。
# CLI: python notebooks/kaggle_training.py --action plan --profile remote_full

# %%
from pathlib import Path
import os
import subprocess
import sys

# 编辑这些路径；在 notebook 单元执行时 __file__ 可能不存在。
SOURCE_ROOT = Path(os.environ.get("GUANDAN_SOURCE_ROOT", "/kaggle/input/REPLACE_SOURCE/guandan"))
WORK_ROOT = Path("/kaggle/working/guandan-source")
OUTPUT_ROOT = Path("/kaggle/working/guandan")
RESUME_MANIFEST = Path("/kaggle/input/REPLACE_CHECKPOINT_BUNDLE/manifest.json")
CANDIDATE_CHECKPOINT = Path("/kaggle/input/REPLACE_CANDIDATE/candidate.pt")
SNAPSHOT_CHECKPOINT = Path("/kaggle/input/REPLACE_SNAPSHOT/snapshot.pt")

# %% [markdown]
# ## 1. 安装源码是显式的复制操作，不是上传
# 下列函数只在调用时执行。默认不安装依赖；需要时手动设置 install_deps=True。
# dry_run=True 仅验证 source manifest 并展示计划，目标目录必须尚不存在。

# %%
def bootstrap_source(*, install_deps=False, dry_run=True):
    import runpy

    checker = SOURCE_ROOT / "scripts" / "kaggle_environment_check.py"
    api = runpy.run_path(str(checker))
    return api["install_source"](SOURCE_ROOT, WORK_ROOT, install_deps=install_deps, dry_run=dry_run)


# 先审阅计划，再手动复制：
# print(bootstrap_source(dry_run=True))
# print(bootstrap_source(dry_run=False))

# %% [markdown]
# ## 2. 计划、硬件检查和冒烟
# plan/system 不执行训练。smoke/train/evaluate 必须显式 execute=True。
# 每次使用新的 Python 进程，防止旧挂载的 guandan 留在 notebook 模块缓存。

# %%
def session_command(action="plan", *, execute=False, algorithm="ippo", resume=None,
                    candidate_checkpoint=None, snapshot_checkpoint=None, root=None):
    source = Path(root) if root is not None else WORK_ROOT
    command = [sys.executable, "-B", str(source / "scripts" / "kaggle_entry.py"),
               "--action", action, "--profile", "remote_full", "--algorithm", algorithm,
               "--output-root", str(OUTPUT_ROOT), "--session-hours", "10",
               "--save-margin-seconds", "300", "--checkpoint-seconds", "600"]
    if execute:
        command.append("--execute")
    if resume is not None:
        command.extend(["--resume", str(resume)])
    if candidate_checkpoint is not None:
        command.extend(["--candidate-checkpoint", str(candidate_checkpoint)])
    if snapshot_checkpoint is not None:
        command.extend(["--snapshot-checkpoint", str(snapshot_checkpoint)])
    return command


def run_session(action="plan", **kwargs):
    command = session_command(action, **kwargs)
    return subprocess.run(command, check=False).returncode


# 手动运行，不要一次取消所有示例的注释：
# run_session("plan")
# run_session("system")
# run_session("smoke", execute=True)

# %% [markdown]
# ## 3. 新训练 / 新会话恢复 / A06 评估（分别显式运行）
# resume 指向当前挂载下的 manifest.json，而非旧机器的绝对路径。
# main-owned session 负责时间预算、信号处理、检查点和证据；这里不重复实现。
# 每个算法独立执行，ippo 与 vrpo 均有各自结果，单次示例不代表完整验收。

# %%
# run_session("train", execute=True, algorithm="ippo")
# run_session("train", execute=True, algorithm="vrpo")
# run_session("train", execute=True, algorithm="ippo", resume=RESUME_MANIFEST)
# run_session("evaluate", execute=True, candidate_checkpoint=CANDIDATE_CHECKPOINT,
#             snapshot_checkpoint=SNAPSHOT_CHECKPOINT)

# %% [markdown]
# ## 4. 普通脚本入口
# 无参数采用 main-owned CLI 的安全默认 plan；退出码原样返回。
# Kaggle 输出的保留/新会话重新挂载由用户操作，本模板从不上传。

# %%
def main(argv=None):
    import runpy

    root = Path(__file__).resolve().parents[1] if "__file__" in globals() else WORK_ROOT
    entry = runpy.run_path(str(root / "scripts" / "kaggle_entry.py"))
    return entry["main"](sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
