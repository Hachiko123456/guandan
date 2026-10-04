"""Offline source export, no network/upload. Only committed tracked files are packaged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

from .provenance import DEPLOYMENT_REQUIRED_FILES, MANIFEST

_PROFILE_VERSION = "profiles-0.2"
_REMOTE_FULL_REQUIRED = {
    "mode": "remote",
    "device": "cuda",
    "algorithms": ["ippo", "vrpo"],
    "training": {
        "updates_per_algorithm": 100,
        "rollout_envs": 8,
        "token_steps_per_update": 1024,
        "checkpoint_interval": 10,
        "resume_updates": 2,
        "max_hours": 12,
    },
    "evaluation": {
        "deal_groups_per_pairing": 250,
        "seat_rotations": [0, 1, 2, 3],
        "opponents": ["random", "rule", "snapshot"],
        "max_hours": 12,
    },
}
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _validate_remote_profile(raw: bytes) -> None:
    try:
        profile = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("deployment requires a valid acceptance profile") from exc
    if not isinstance(profile, dict) or profile.get("version") != _PROFILE_VERSION:
        raise ValueError(f"deployment requires {_PROFILE_VERSION}")
    remote = profile.get("profiles", {}).get("remote_full") if isinstance(profile.get("profiles"), dict) else None
    if not isinstance(remote, dict):
        raise ValueError("deployment requires profiles.remote_full")
    for key, expected in _REMOTE_FULL_REQUIRED.items():
        if remote.get(key) != expected:
            raise ValueError(f"deployment remote_full configuration mismatch: {key}")


def build_package(root: Path, output: Path) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip()
    if status:
        raise ValueError("commit/review source changes before exporting a clean deployment package")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError("deployment requires a full 40-character source commit")
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "-z", commit], cwd=root
    ).decode().split("\0")
    manifest = {
        "format": "guandan-source-v1",
        "git_commit": commit,
        "files": {},
        "profile": "remote_full",
        "uploaded": False,
        "accepted": False,
        "kaggle_verified": False,
    }
    data: dict[str, bytes] = {}
    blocked = {".git", "runs", "checkpoints", "logs", "wandb", "__pycache__"}
    for name in filter(None, files):
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or "\\" in name
            or ".." in path.parts
            or blocked.intersection(path.parts)
            or name.startswith("project_status/history/")
        ):
            continue
        if path.suffix.lower() in {".pt", ".pth", ".pem", ".key"} or path.name in {".env", "kaggle.json", MANIFEST}:
            raise ValueError(f"sensitive/derived tracked file is not allowed in source export: {name}")
        mode = subprocess.check_output(["git", "ls-tree", commit, "--", name], cwd=root).decode().split()[0]
        if mode not in {"100644", "100755"}:
            raise ValueError(f"unsupported linked/submodule source: {name}")
        content = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        manifest["files"][name] = hashlib.sha256(content).hexdigest()
        data[name] = content

    missing_required = sorted(DEPLOYMENT_REQUIRED_FILES - data.keys())
    if missing_required:
        raise ValueError(f"incomplete source export: {missing_required}")
    _validate_remote_profile(data["configs/acceptance_profiles.json"])

    output.parent.mkdir(parents=True, exist_ok=True)
    # Fixed ZIP metadata keeps equal source commits byte-reproducible within the
    # supported Python/zlib toolchain.  No upload or network operation occurs.
    with output.open("xb") as raw:
        with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            data[MANIFEST] = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
            for name in sorted(data):
                info = zipfile.ZipInfo(f"guandan/{name}", date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data[name])
    return {
        "archive": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "git_commit": commit,
        "file_count": len(data),
        "uploaded": False,
        "kaggle_verified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export committed pure-Python source; never upload")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    print(json.dumps(build_package(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
