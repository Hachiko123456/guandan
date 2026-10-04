"""Source provenance for Git checkouts and verified deployment exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

MANIFEST = "SOURCE_MANIFEST.json"
_MANIFEST_KEYS = frozenset({
    "format", "git_commit", "files", "profile", "uploaded", "accepted", "kaggle_verified",
})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

# Keep this aligned with the installer’s minimum source-root contract.  The
# package builder and a verified extracted export must both contain these files.
DEPLOYMENT_REQUIRED_FILES = frozenset({
    "pyproject.toml",
    "configs/acceptance_profiles.json",
    "guandan/__init__.py",
    "guandan/deployment/__init__.py",
    "guandan/deployment/provenance.py",
    "guandan/deployment/package.py",
    "guandan/deployment/control.py",
    "guandan/deployment/session.py",
    "guandan/deployment/workflow.py",
    "notebooks/kaggle_unified_bootstrap.py",
    "scripts/kaggle_entry.py",
    "scripts/kaggle_environment_check.py",
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _manifest_path_error(name: object) -> ValueError:
    return ValueError(f"source manifest hash/path mismatch: {name!r}")


def _validate_manifest_name(name: object) -> str:
    if not isinstance(name, str) or not name or name == MANIFEST:
        raise _manifest_path_error(name)
    if "\\" in name or ":" in name:
        raise _manifest_path_error(name)
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or name != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise _manifest_path_error(name)
    return name


def _read_verified_manifest(root: Path) -> tuple[dict, str]:
    if not root.is_dir():
        raise ValueError(f"source export root is not a directory: {root}")
    manifest_path = root / MANIFEST
    if _is_link(manifest_path) or not manifest_path.is_file():
        raise ValueError("verified source export requires a regular SOURCE_MANIFEST.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid SOURCE_MANIFEST.json") from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("invalid source manifest schema/keys")
    if manifest["format"] != "guandan-source-v1":
        raise ValueError("invalid source manifest format")
    commit = manifest["git_commit"]
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise ValueError("invalid source manifest commit")
    if manifest["profile"] != "remote_full":
        raise ValueError("invalid source manifest profile")
    for field in ("uploaded", "accepted", "kaggle_verified"):
        if type(manifest[field]) is not bool or manifest[field] is not False:
            raise ValueError(f"source manifest {field} must be false")

    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("source manifest files must be a nonempty mapping")
    normalized: dict[str, str] = {}
    for raw_name, digest in files.items():
        name = _validate_manifest_name(raw_name)
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"source manifest hash/path mismatch: {name!r}")
        normalized[name] = digest

    actual: set[str] = set()
    for path in root.rglob("*"):
        if _is_link(path):
            raise ValueError(f"linked source file is not portable: {path.relative_to(root).as_posix()}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported source entry: {path.relative_to(root).as_posix()}")
        name = path.relative_to(root).as_posix()
        if name != MANIFEST:
            actual.add(name)

    expected = set(normalized)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"source manifest file-set mismatch: missing={missing}, extra={extra}")
    missing_required = sorted(DEPLOYMENT_REQUIRED_FILES - actual)
    if missing_required:
        raise ValueError(f"incomplete source export: {missing_required}")

    for name in sorted(normalized):
        path = root / name
        if not path.is_file() or _is_link(path) or sha256(path) != normalized[name]:
            raise ValueError(f"source manifest hash/path mismatch: {name}")
    return manifest, sha256(manifest_path)


def source_provenance(root: Path) -> dict:
    root = Path(root).resolve()
    if (root / MANIFEST).is_file() or (root / MANIFEST).is_symlink():
        manifest, manifest_digest = _read_verified_manifest(root)
        return {
            "git_commit": manifest["git_commit"],
            "source_kind": "verified_export",
            "source_manifest_sha256": manifest_digest,
            "git_worktree_clean": True,
        }

    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    )
    commit = proc.stdout.strip()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError("invalid source commit")
    return {
        "git_commit": commit,
        "source_kind": "git",
        "git_worktree_clean": not bool(status.stdout.strip()),
        "source_manifest_sha256": None,
    }
