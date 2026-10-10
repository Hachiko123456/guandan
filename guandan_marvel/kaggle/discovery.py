# SPDX-License-Identifier: MIT
"""Fail-closed discovery for Kaggle input datasets, source archives, and recovery bundles."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping
import zipfile

from .common import canonical_hash, sha256_file

DATASET_MANIFEST_SCHEMA = "guandan-marvel-kaggle-dataset-v1"
SOURCE_MANIFEST_SCHEMA = "guandan-marvel-source-manifest-v1"
RECOVERY_MANIFEST_SCHEMA = "guandan-marvel-recovery-manifest-v1"
DATASET_MANIFEST_NAME = "guandan_marvel_dataset.json"
RECOVERY_BUNDLE_NAMES = frozenset({"guandan-marvel-recovery.zip", "guandan-marvel-partial.zip"})
SUPPORTED_MODES = frozenset({"preflight", "pilot", "resume", "full", "eval_only"})


@dataclass(frozen=True, slots=True)
class DatasetInput:
    manifest_path: Path
    manifest: dict[str, object]
    source_archive: Path
    source_manifest: Path


@dataclass(frozen=True, slots=True)
class RecoveryInput:
    bundle_path: Path
    manifest: dict[str, object]


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _safe_relative(base: Path, relative: object, *, label: str) -> Path:
    value = Path(str(relative))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{label} must be a safe relative path: {relative!r}")
    resolved = (base / value).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"{label} escapes its dataset directory: {relative!r}")
    return resolved


def validate_dataset_manifest(payload: Mapping[str, object], *, manifest_path: Path | None = None) -> None:
    if payload.get("schema") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("unsupported Kaggle dataset manifest schema")
    mode = payload.get("mode")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported Kaggle mode {mode!r}")
    package_id = str(payload.get("package_id", ""))
    if not package_id:
        raise ValueError("dataset manifest package_id is required")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("dataset manifest source object is required")
    for name in ("archive", "archive_sha256", "manifest", "manifest_sha256", "git_commit"):
        if not str(source.get(name, "")):
            raise ValueError(f"dataset source.{name} is required")
    for name in ("archive_sha256", "manifest_sha256"):
        digest = str(source[name]).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"dataset source.{name} must be SHA-256")
    commit = str(source["git_commit"]).lower()
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("dataset source.git_commit must be a full commit SHA")
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or mode not in profiles:
        raise ValueError(f"dataset manifest must include profiles.{mode}")
    profile = profiles[mode]
    if not isinstance(profile, Mapping) or profile.get("schema") != "guandan-marvel-kaggle-profile-v1":
        raise ValueError(f"profiles.{mode} is missing a versioned profile")
    if profile.get("mode") != mode:
        raise ValueError(f"profiles.{mode}.mode mismatch")
    if mode in {"pilot", "resume", "full"}:
        ddp = profile.get("ddp")
        if not isinstance(ddp, Mapping):
            raise ValueError("training profiles require a DDP object")
        if int(ddp.get("world_size", 0)) != 2 or ddp.get("launcher") != "torchrun":
            raise ValueError("training profiles must require torchrun world_size=2")
        training = profile.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("training profile object is required")
        for key in ("rollout_batch_size", "collect_batch_size"):
            if int(training.get(key, 0)) <= 0 or int(training[key]) % 2:
                raise ValueError(f"training.{key} must be positive and divisible by 2")
    recovery = payload.get("recovery")
    if not isinstance(recovery, Mapping):
        raise ValueError("dataset manifest recovery object is required")
    if mode == "resume" and not bool(recovery.get("required", False)):
        raise ValueError("resume dataset manifest must require recovery")
    if mode == "eval_only" and not bool(recovery.get("required", False)):
        raise ValueError("eval_only dataset manifest must require a recovery/checkpoint input")
    if manifest_path is not None:
        base = manifest_path.parent.resolve()
        archive = _safe_relative(base, source["archive"], label="source.archive")
        external_manifest = _safe_relative(base, source["manifest"], label="source.manifest")
        if not archive.is_file() or not external_manifest.is_file():
            raise FileNotFoundError("dataset source archive or external source manifest is missing")
        if sha256_file(archive) != source["archive_sha256"]:
            raise ValueError("dataset source archive SHA-256 mismatch")
        if sha256_file(external_manifest) != source["manifest_sha256"]:
            raise ValueError("dataset external source manifest SHA-256 mismatch")
        source_payload = _load_json(external_manifest)
        if source_payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
            raise ValueError("external source manifest schema mismatch")
        if source_payload.get("git_commit") != source["git_commit"]:
            raise ValueError("dataset/source manifest commit mismatch")
        if source_payload.get("package_id") != package_id:
            raise ValueError("dataset/source manifest package_id mismatch")


def discover_dataset(input_root: str | Path) -> DatasetInput:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle input root does not exist: {root}")
    candidates = sorted(path for path in root.rglob(DATASET_MANIFEST_NAME) if path.is_file())
    if len(candidates) != 1:
        rendered = [str(path) for path in candidates]
        raise RuntimeError(
            "exactly one guandan_marvel_dataset.json is required; "
            f"found {len(candidates)}: {rendered}"
        )
    manifest_path = candidates[0].resolve()
    payload = _load_json(manifest_path)
    validate_dataset_manifest(payload, manifest_path=manifest_path)
    source = payload["source"]
    archive = _safe_relative(manifest_path.parent, source["archive"], label="source.archive")
    source_manifest = _safe_relative(manifest_path.parent, source["manifest"], label="source.manifest")
    return DatasetInput(manifest_path, payload, archive, source_manifest)


def _read_recovery_manifest(bundle: Path) -> dict[str, object]:
    with zipfile.ZipFile(bundle) as archive:
        names = [name for name in archive.namelist() if name.rstrip("/") == "recovery_manifest.json"]
        if len(names) != 1:
            raise ValueError(f"recovery bundle must contain one root recovery_manifest.json: {bundle}")
        payload = json.loads(archive.read(names[0]).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"recovery manifest must be an object: {bundle}")
        for row in payload.get("files", []):
            name = str(row.get("path", ""))
            pure = Path(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe recovery member path: {name}")
            try:
                info = archive.getinfo(name)
            except KeyError as error:
                raise ValueError(f"recovery member is missing: {name}") from error
            digest = __import__("hashlib").sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if int(row.get("bytes", -1)) != info.file_size or row.get("sha256") != digest.hexdigest():
                raise ValueError(f"recovery member hash mismatch: {name}")
    if not isinstance(payload, dict) or payload.get("schema") != RECOVERY_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported recovery manifest in {bundle}")
    declared = payload.get("manifest_hash")
    actual = canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    if declared != actual:
        raise ValueError(f"recovery manifest hash mismatch: {bundle}")
    return payload


def discover_recovery(
    input_root: str | Path,
    *,
    source_commit: str,
    source_archive_sha256: str,
    required: bool,
) -> RecoveryInput | None:
    root = Path(input_root).resolve()
    bundles = sorted(
        path.resolve() for path in root.rglob("*.zip")
        if path.is_file() and path.name in RECOVERY_BUNDLE_NAMES
    )
    matches: list[RecoveryInput] = []
    errors: list[str] = []
    for bundle in bundles:
        try:
            payload = _read_recovery_manifest(bundle)
        except Exception as error:
            errors.append(f"{bundle}: {type(error).__name__}: {error}")
            continue
        if payload.get("source_commit") == source_commit and payload.get("source_archive_sha256") == source_archive_sha256:
            matches.append(RecoveryInput(bundle, payload))
    if len(matches) > 1:
        raise RuntimeError(f"multiple matching recovery bundles found; refusing to guess: {[str(x.bundle_path) for x in matches]}")
    if len(matches) == 1:
        return matches[0]
    if required:
        raise RuntimeError(
            "a unique matching recovery bundle is required but none was found; "
            f"scanned={len(bundles)}, invalid={errors}"
        )
    return None


__all__ = [
    "DATASET_MANIFEST_NAME", "DATASET_MANIFEST_SCHEMA", "DatasetInput", "RecoveryInput",
    "SOURCE_MANIFEST_SCHEMA", "SUPPORTED_MODES", "discover_dataset", "discover_recovery",
    "validate_dataset_manifest",
]
