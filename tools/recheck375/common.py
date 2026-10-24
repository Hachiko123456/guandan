"""Pure I/O, source binding and hard evaluation-only authorization checks."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import shutil
import sys
import zipfile
from pathlib import Path

SCHEMA = "guandan-recheck375-plan-v1"
PROTOCOL = "two_unique_team_assignments_v1"
SOURCE = {
    "archive_sha256": "5afaf81a5e76e70e56c6f871898313df4962fabe3f9cef0818931a2c53b15a2a",
    "content_manifest_hash": "6574388f92df8457731215f1e08707e5b5b9368bb1d46c165bb3aa2be2997c2a",
    "git_commit": "fa4a7af3227dd8d76663bd9cf2b0e3a659d70373",
    "core_source_hash": "0492d679f3ccea7c6388676463f6b266e178201498285ecc84e84de20d1657ef",
}
RUNTIME_PATH = "/kaggle/working/guandan_t4x2_combo_v1/source/guandan-marvel-source"
BACKEND_HASH = "6e6875fdd2f754a6fbd5eef0b6cb540d29fa5456727d9eaab34cedc8852167ba"
PYTHON_VERSION = "3.13.15"
TORCH_VERSION = "2.11.0+cu128"
MODEL_SHAS = {
    "p250": "965c60603b6aa8b5fe107d00c1a998e1389b64d104c4610386786b9487849f5f",
    "p375": "d523e557975ec10b4e6e8347d49a0a9226c188a30bb22cf9088bb78ac16374ee",
    "p500": "2139e12728e64c6dba63de34e9a2673b9fc15b8f4251c3e019edb8c2cb80ba42",
}
DEVELOPMENT_HASH = "5fbfb160bab60be8c158d9d4bceb5c5381d83bb4ff691eb53575200176ffea16"


def canonical(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_path(root, name):
    root = Path(root).resolve()
    if not name or any(c in name for c in ("\\", ":", "\0")):
        raise ValueError("unsafe member path")
    target = (root / name).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("member escapes owned root")
    return target


def extract_safe(archive, target, *, max_bytes=128 * 1024**2):
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len({i.filename.casefold() for i in members}) != len(members):
            raise ValueError("duplicate ZIP members")
        if sum(i.file_size for i in members) > max_bytes:
            raise ValueError("oversize ZIP")
        for member in members:
            if member.orig_filename != member.filename:
                raise ValueError("normalized ZIP member")
            safe_path(target, member.filename)
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("ZIP symlink")
        target.mkdir(parents=True)
        bundle.extractall(target)
    return target


def verify_hash(value, key="hash"):
    if value.get(key) != canonical({k: v for k, v in value.items() if k != key}):
        raise ValueError("invalid " + key)


def validate_plan(plan, *, testing=False):
    verify_hash(plan, "plan_hash")
    if plan.get("schema") != SCHEMA:
        raise ValueError("wrong evaluation plan schema")
    if plan.get("test_only") and not testing:
        raise ValueError("local test plan is forbidden in production")
    if plan.get("training_authorized") is not False or plan.get("new_training_games_cap") != 0 or plan.get("optimizer_updates_cap") != 0:
        raise ValueError("this package authorizes ZERO training")
    if plan.get("fabledan") != "NOT_RUN" or plan.get("final_holdout") != "NOT_TOUCHED":
        raise ValueError("evaluation boundary changed")
    if plan.get("protocol") != PROTOCOL or plan.get("rotations") != [0, 1] or plan.get("horizon") != 1024:
        raise ValueError("unique-assignment/horizon contract changed")
    if not plan.get("test_only"):
        if plan.get("source") != SOURCE or plan.get("runtime_source_path") != RUNTIME_PATH:
            raise ValueError("frozen source/absolute path changed")
        if (plan.get("python"), plan.get("torch"), plan.get("backend_hash")) != (PYTHON_VERSION, TORCH_VERSION, BACKEND_HASH):
            raise ValueError("original environment identity changed")
        if plan.get("development_hash") != DEVELOPMENT_HASH or plan.get("deal_count") != 128:
            raise ValueError("must use exactly the original endpoint development set")
        if plan.get("temperature") != 0.0 or (plan.get("new_unique_strength_games_cap"), plan.get("cache_replay_games"), plan.get("smoke_new_games")) != (768, 24, 24):
            raise ValueError("fixed evaluation-only budget changed")
        if plan.get("new_arms") != ["p375_vs_rule", "p375_vs_p250", "p375_vs_p500"] or plan.get("cached_arms") != ["p250_vs_rule", "p500_vs_rule", "p500_vs_p250"]:
            raise ValueError("fixed comparison matrix changed")
        if plan.get("wall_seconds_cap") != 5400:
            raise ValueError("90-minute safety cap changed")
        for name, digest in MODEL_SHAS.items():
            if plan["models"][name]["sha256"] != digest:
                raise ValueError("checkpoint identity changed: " + name)
    return plan


def verify_driver(plan):
    root = Path(__file__).resolve().parents[1]
    for row in plan["driver_files"]:
        path = safe_path(root, row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("driver identity changed: " + row["path"])


def verify_input_files(dataset):
    dataset = Path(dataset)
    meta = read_json(dataset / "delivery_manifest.json")
    verify_hash(meta, "manifest_hash")
    if {p.name for p in dataset.iterdir()} != {"delivery_manifest.json", *(r["path"] for r in meta["files"])}:
        raise ValueError("undeclared input files")
    for row in meta["files"]:
        path = safe_path(dataset, row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("corrupt input: " + row["path"])
    return meta


def install_runtime(dataset, run_root, plan, *, audit_only=False):
    """Use the exact old binary and absolute path, without altering policy headers."""
    dataset, run_root = Path(dataset), Path(run_root)
    archive = dataset / "frozen_source.zip"
    if sha(archive) != plan["source"]["archive_sha256"]:
        raise ValueError("wrong frozen source archive")
    staged = extract_safe(archive, run_root / "source_staging") / "guandan-marvel-source"

    def check_source(source):
        manifest = read_json(source / "source_manifest.json")
        verify_hash(manifest, "manifest_hash")
        if manifest["manifest_hash"] != plan["source"]["content_manifest_hash"]:
            raise ValueError("source content manifest changed")
        for row in manifest["files"]:
            path = safe_path(source, row["path"])
            if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                raise ValueError("frozen runtime file changed: " + row["path"])

    check_source(staged)
    if audit_only:
        source = staged
    else:
        if platform.system() != "Linux" or platform.python_version() != plan["python"]:
            raise RuntimeError(f"requires original Linux/Python {plan['python']}; got {platform.system()}/{platform.python_version()}")
        source = Path(plan["runtime_source_path"])
        if source.resolve() != source:
            raise ValueError("original absolute source path is redirected")
        if source.exists():
            check_source(source)
        else:
            source.parent.mkdir(parents=True, exist_ok=True)
            staged.rename(source)
        check_source(source)
    native = plan["native"]
    native_input = safe_path(dataset, native["path"])
    if sha(native_input) != native["sha256"]:
        raise ValueError("native binary changed")
    target = safe_path(source, native["restore_relative_path"])
    if target.parent != source / "nanopokerlib":
        raise ValueError("unsafe native restore path")
    if target.exists():
        if sha(target) != native["sha256"]:
            raise ValueError("conflicting original native; never overwrite a loaded shared library")
    else:
        shutil.copy2(native_input, target)
    loaded = sys.modules.get("guandan_marvel")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
        raise RuntimeError("restart kernel: another Guandan runtime is already imported")
    sys.path.insert(0, str(source))
    os.environ["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), str(source)])
    os.environ["GUANDAN_MARVEL_SOURCE_COMMIT"] = plan["source"]["git_commit"]
    os.environ["PYTHONUNBUFFERED"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    atomic_json(run_root / "source_binding.json", {"source": str(source), "original_path": plan["runtime_source_path"], "audit_only": audit_only})
    return source


def state_hash(state):
    import torch
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
