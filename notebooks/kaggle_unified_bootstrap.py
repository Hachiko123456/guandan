# 支持单 ZIP Dataset 和已展开的 Dataset，不要求上传文件夹。
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile

# 与本交付包绑定：不要混用旧 Notebook 和新 ZIP。
EXPECTED_COMMIT = '__COMMIT__'
EXPECTED_ZIP_SHA256 = '__ZIP_SHA256__'
EXPECTED_MANIFEST_SHA256 = '__MANIFEST_SHA256__'
EXPECTED_CHECKER_SHA256 = '__CHECKER_SHA256__'

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ROOT = WORKING_ROOT / "guandan"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _find_uploaded_source(input_root, hint=""):
    base = Path(input_root).resolve()
    scope = (base / hint).resolve() if hint else base
    if not base.is_dir() or not scope.exists() or not scope.is_relative_to(base):
        raise RuntimeError("请先把源码 ZIP 所在的 Dataset 添加到 Notebook；hint 必须位于 /kaggle/input")
    manifests = list(scope.rglob("SOURCE_MANIFEST.json")) if scope.is_dir() else []
    expanded = [p.parent.resolve() for p in manifests
                if p.resolve().is_relative_to(base)
                and p.stat().st_size <= 1024 * 1024
                and _sha(p) == EXPECTED_MANIFEST_SHA256]
    if len(expanded) == 1:
        return "expanded", expanded[0]
    if len(expanded) > 1:
        raise RuntimeError("匹配源码不唯一，请设置 SOURCE_DATASET_HINT 选择一个 Dataset")
    archives = list(scope.rglob("*.zip")) if scope.is_dir() else [scope]
    matches = [p.resolve() for p in archives
               if p.is_file() and p.resolve().is_relative_to(base)
               and p.stat().st_size <= 64 * 1024 * 1024
               and _sha(p) == EXPECTED_ZIP_SHA256]
    if len(matches) != 1:
        raise RuntimeError("找不到唯一匹配的 guandan-source.zip；请添加本次新 ZIP，不要使用旧包")
    return "zip", matches[0]


def _load_pinned_checker(kind, source):
    # 执行前先对入口字节验 SHA，ZIP 内部的其余文件仍由安装器完整验证。
    if kind == "zip":
        with zipfile.ZipFile(source) as bundle:
            raw = bundle.read("guandan/scripts/kaggle_environment_check.py")
            manifest_raw = bundle.read("guandan/SOURCE_MANIFEST.json")
        origin = str(source) + "!/guandan/scripts/kaggle_environment_check.py"
    else:
        raw = (source / "scripts/kaggle_environment_check.py").read_bytes()
        manifest_raw = (source / "SOURCE_MANIFEST.json").read_bytes()
        origin = str(source / "scripts/kaggle_environment_check.py")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_CHECKER_SHA256:
        raise RuntimeError("环境检查脚本 SHA256 不匹配，拒绝执行")
    if hashlib.sha256(manifest_raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("manifest SHA256 不匹配")
    metadata = json.loads(manifest_raw)
    if metadata["git_commit"] != EXPECTED_COMMIT:
        raise RuntimeError("源码 commit 不匹配")
    namespace = {"__name__": "kaggle_zip_bootstrap", "__file__": origin}
    exec(compile(raw, origin, "exec"), namespace)
    return namespace


SOURCE_KIND, SOURCE_INPUT = _find_uploaded_source(INPUT_ROOT, SOURCE_DATASET_HINT)
checker = _load_pinned_checker(SOURCE_KIND, SOURCE_INPUT)
if not WORKING_ROOT.is_dir():
    raise RuntimeError("缺少 /kaggle/working；本 Notebook 应在 Kaggle 运行")
# 使用全新目录，重跑也不会覆盖旧源码或检查点。
WORK_ROOT = Path(tempfile.mkdtemp(prefix="guandan-bootstrap-", dir=WORKING_ROOT)) / "source"
installer = checker["install_source_archive" if SOURCE_KIND == "zip" else "install_source"]
print("INPUT_KIND =", SOURCE_KIND)
print("SOURCE_INPUT =", SOURCE_INPUT)
print("WORK_ROOT =", WORK_ROOT)
for is_dry_run in (True, False):
    install = installer(SOURCE_INPUT, WORK_ROOT,
                        install_deps=INSTALL_DEPS, dry_run=is_dry_run,
                        enforce_kaggle_paths=True)
    print(json.dumps(install, ensure_ascii=False, indent=2))
    if install.get("dependency_returncode") not in (None, 0):
        raise RuntimeError("依赖安装失败，不能继续训练")
ENTRY = WORK_ROOT / "scripts/kaggle_entry.py"
