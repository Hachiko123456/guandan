"""Explicit runtime inspection and source-export bootstrap, with no uploads.

Public API: inspect_environment(), require_remote_runtime(system), and
install_source(source_root, dest_root, install_deps=False, dry_run=False).
Importing this module does not import NumPy, Torch, or guandan, inspect hardware,
install packages, write files, or start a session.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import runpy
import shutil
import subprocess
import sys
import tomllib


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_MARKERS = ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_KERNEL_INTEGRATIONS")
SOURCE_MANIFEST = "SOURCE_MANIFEST.json"


def inspect_environment() -> dict:
    """Return measured local facts. Kaggle-looking markers are NOT proof."""
    system = {
        "schema_version": 1,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "python_version_info": list(sys.version_info[:3]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": None,
        "torch_status": "missing",
        "torch_error": None,
        "cuda_version": None,
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "kaggle": {
            "input_dir": str(KAGGLE_INPUT),
            "working_dir": str(KAGGLE_WORKING),
            "input_exists": KAGGLE_INPUT.is_dir(),
            "working_exists": KAGGLE_WORKING.is_dir(),
            "working_writable": KAGGLE_WORKING.is_dir() and os.access(KAGGLE_WORKING, os.W_OK),
            # Marker names only: never dump arbitrary environment values/secrets.
            "markers_present": [name for name in KAGGLE_MARKERS if os.environ.get(name)],
        },
        "accepted": False,
        "kaggle_verified": False,
    }
    try:
        torch = importlib.import_module("torch")
        system["torch"] = str(torch.__version__)
        system["cuda_version"] = torch.version.cuda
        system["cuda_available"] = bool(torch.cuda.is_available())
        system["gpu_count"] = int(torch.cuda.device_count()) if system["cuda_available"] else 0
        for index in range(system["gpu_count"]):
            properties = torch.cuda.get_device_properties(index)
            system["gpus"].append({
                "index": index,
                "name": str(torch.cuda.get_device_name(index)),
                "total_memory_bytes": int(properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
            })
        system["torch_status"] = "ok"
    except ModuleNotFoundError as exc:
        system["torch_status"] = "missing" if exc.name == "torch" else "error"
        system["torch_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        system["torch_status"] = "error"
        system["torch_error"] = f"{type(exc).__name__}: {exc}"
    return system


def require_remote_runtime(system: dict) -> None:
    """Fail closed on unsupported runtime facts; this is not remote acceptance."""
    errors = []
    version = system.get("python_version_info", [])
    if (not isinstance(version, (list, tuple)) or len(version) < 2
            or any(type(part) is not int for part in version)
            or tuple(version[:2]) < (3, 12)):
        errors.append("Python >=3.12 required")
    if system.get("torch_status") != "ok":
        errors.append("a working PyTorch import is required")
    if not system.get("cuda_available") or not system.get("gpus") or system.get("gpu_count", 0) < 1:
        errors.append("CUDA with an observable GPU is required; CPU fallback is not remote_full")
    kaggle = system.get("kaggle", {})
    if not (kaggle.get("input_exists") and kaggle.get("working_exists") and kaggle.get("working_writable")):
        errors.append("standard /kaggle/input and writable /kaggle/working directories are required")
    if errors:
        raise RuntimeError("; ".join(errors))


def _root(source_root) -> Path:
    root = Path(source_root).expanduser().resolve()
    for name in ("pyproject.toml", "configs/acceptance_profiles.json", "guandan/__init__.py"):
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"incomplete source root: {name}")
    return root


def _requirements(root: Path) -> list[str]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8-sig"))["project"]
    requirements = project.get("dependencies", [])
    if not isinstance(requirements, list) or any(not isinstance(value, str) or not value.strip() for value in requirements):
        raise ValueError("invalid project runtime dependencies")
    return requirements


def _install_command(root: Path) -> list[str]:
    # Binary wheels only: neither a C++ compiler nor a source package build is
    # ever an implicit fallback. Keep the preinstalled CUDA Torch if satisfied.
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--only-binary=:all:", *_requirements(root)]


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _export_files(root: Path) -> tuple[dict, list[str]]:
    """Validate copy boundaries; canonical provenance validation stays upstream."""
    manifest_path = root / SOURCE_MANIFEST
    if not manifest_path.is_file():
        raise ValueError("install_source requires an extracted source export with SOURCE_MANIFEST.json; export with guandan.deployment.package first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source manifest files must be a nonempty mapping")
    names = [*files, SOURCE_MANIFEST]
    for name in names:
        relative = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name or relative.is_absolute()
                or ".." in relative.parts or str(relative) != name):
            raise ValueError(f"unsafe source manifest path: {name!r}")
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"source file missing or outside source root: {name}")
        if any(_is_link(parent) for parent in (path, *path.parents) if parent.is_relative_to(root)):
            raise ValueError(f"linked source file is not portable: {name}")
    required = {"pyproject.toml", "configs/acceptance_profiles.json", "guandan/__init__.py",
                "guandan/deployment/provenance.py", "guandan/deployment/session.py",
                "scripts/kaggle_entry.py", "scripts/kaggle_environment_check.py"}
    if not required.issubset(files):
        raise ValueError(f"incomplete source export: {sorted(required.difference(files))}")
    return manifest, names


def install_source(source_root, dest_root, install_deps=False, dry_run=False) -> dict:
    """Copy a verified extracted export to a NEW writable source directory.

    Source is read-only input; checkpoint bundles are not installation sources.
    Existing destinations are never merged, overwritten, or removed. dry_run
    validates and returns the proposed commands without copying or invoking pip.
    No archive extraction, upload, credentials, Git clone, or build is performed.
    """
    source = _root(source_root)
    destination = Path(dest_root).expanduser().resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("source and destination must be separate, non-nested directories")
    if destination.is_relative_to(KAGGLE_INPUT.resolve()):
        raise ValueError("/kaggle/input is read-only input, not an installation destination")
    if destination.exists():
        raise FileExistsError(f"destination already exists; choose a fresh source directory: {destination}")
    manifest, names = _export_files(source)
    # Invoke the main-owned implementation rather than duplicating provenance.
    # This is an explicitly requested source installation, never an import side effect.
    provenance = runpy.run_path(str(source / "guandan/deployment/provenance.py"))["source_provenance"]
    source_evidence = provenance(source)
    command = _install_command(source) if install_deps else None
    report = {
        "status": "planned" if dry_run else "copied", "source_root": str(source),
        "dest_root": str(destination), "source": source_evidence,
        "files": sorted(names), "file_count": len(names),
        "install_deps": bool(install_deps), "dry_run": bool(dry_run),
        "commands": [] if command is None else [command], "dependency_returncode": None,
        "uploaded": False, "accepted": False, "kaggle_verified": False,
    }
    if dry_run:
        return report
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    provenance(destination)
    if command is not None:
        completed = subprocess.run(command, cwd=destination, check=False)
        report["dependency_returncode"] = completed.returncode
        report["status"] = "installed" if completed.returncode == 0 else "dependency_install_failed"
    return report


def check_dependencies(root: Path) -> dict:
    """Opt-in import/metadata diagnostic; no pip or network and no auto-fix."""
    packages = []
    # Import names for the runtime requirements in this project's pyproject.
    for name in ("numpy", "torch", "tqdm", "tyro", "psutil"):
        try:
            importlib.import_module(name)
            packages.append({"name": name, "version": importlib.metadata.version(name), "status": "ok"})
        except Exception as exc:
            packages.append({"name": name, "status": "missing_or_broken", "error": f"{type(exc).__name__}: {exc}"})
    return {"requirements": _requirements(root), "packages": packages,
            "version_constraints_evaluated": False,
            "status": "ok" if all(row["status"] == "ok" for row in packages) else "missing_or_broken"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--show-system", action="store_true", help="runtime report (default); no session execution")
    actions.add_argument("--check-dependencies", action="store_true", help="explicit runtime dependency import check")
    actions.add_argument("--source-root", type=Path, help="extracted source export to copy; requires --dest-root")
    parser.add_argument("--dest-root", type=Path)
    parser.add_argument("--install-deps", action="store_true", help="explicit wheel-only pip installation after copying")
    parser.add_argument("--dry-run", action="store_true", help="validate copy/install plan without changing files")
    parser.add_argument("--require-remote", action="store_true", help="require Python/CUDA and local Kaggle markers")
    args = parser.parse_args(argv)
    if args.source_root is None and (args.dest_root is not None or args.install_deps or args.dry_run):
        parser.error("--dest-root/--install-deps/--dry-run require --source-root")
    if args.source_root is not None and args.dest_root is None:
        parser.error("--source-root requires --dest-root")
    try:
        if args.source_root is not None:
            report = install_source(args.source_root, args.dest_root, args.install_deps, args.dry_run)
            code = report["dependency_returncode"] or 0
        elif args.check_dependencies:
            report = check_dependencies(_root(Path(__file__).resolve().parents[1]))
            code = 0 if report["status"] == "ok" else 1
        else:
            report = inspect_environment()
            code = 0
        if args.require_remote:
            system = report if "python_version_info" in report else inspect_environment()
            require_remote_runtime(system)
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "kaggle_verified": False}
        code = 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
