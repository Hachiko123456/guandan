# SPDX-License-Identifier: MIT
"""Stdlib-only Kaggle input resolver shared by Notebook bootstrap and runtime."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Mapping
import zipfile

DATASET_MANIFEST_SCHEMA = "guandan-marvel-kaggle-dataset-v2"
SOURCE_MANIFEST_SCHEMA = "guandan-marvel-source-manifest-v1"
RECOVERY_MANIFEST_SCHEMA = "guandan-marvel-recovery-manifest-v3"
DATASET_MANIFEST_NAME = "guandan_marvel_dataset.json"
BOOTSTRAP_NAME = "guandan_marvel_bootstrap.py"
RECOVERY_BUNDLE_NAMES = frozenset({"guandan-marvel-recovery.zip", "guandan-marvel-partial.zip"})
SUPPORTED_MODES = frozenset({"preflight", "pilot", "resume", "full", "eval_only"})


@dataclass(frozen=True, slots=True)
class SourceInput:
    kind: str
    path: Path
    content_root: Path | None
    content_manifest: dict[str, object]
    content_manifest_hash: str
    archive_sha256: str | None
    archive_sha256_verified: bool


@dataclass(frozen=True, slots=True)
class DatasetInput:
    manifest_path: Path
    manifest: dict[str, object]
    source_manifest_path: Path
    source: SourceInput


@dataclass(frozen=True, slots=True)
class RecoveryInput:
    kind: str
    path: Path
    content_root: Path | None
    manifest: dict[str, object]
    content_manifest_hash: str
    archive_sha256: str | None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        raise ValueError(f"{label} escapes its input directory: {relative!r}")
    return resolved


def _validate_declared_manifest(payload: Mapping[str, object], *, schema: str, label: str) -> None:
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported {label} schema {payload.get('schema')!r}")
    declared = payload.get("manifest_hash")
    actual = canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    if declared != actual:
        raise ValueError(f"{label} manifest hash mismatch")


def _path_has_symlink(root: Path, path: Path) -> bool:
    current = path
    root = root.resolve()
    while current != root:
        if current.is_symlink():
            return True
        current = current.parent
    return root.is_symlink()


def _validate_directory_files(
    root: Path,
    manifest: Mapping[str, object],
    *,
    manifest_name: str,
    allowed_extras: set[str] | None = None,
) -> None:
    root = root.resolve()
    declared: set[str] = set()
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("content manifest files must be a list")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("content manifest file row must be an object")
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe content path: {relative}")
        normalized = relative.as_posix()
        if normalized in declared:
            raise ValueError(f"duplicate content path: {normalized}")
        declared.add(normalized)
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"content path escapes root: {relative}")
        if _path_has_symlink(root, root / relative):
            raise ValueError(f"symlinks are forbidden in unpacked input: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"content file is missing: {relative}")
        if path.stat().st_size != int(row.get("bytes", -1)) or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"content file hash mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != manifest_name
    }
    extras = sorted(actual - declared - set(allowed_extras or ()))
    missing = sorted(declared - actual)
    if extras or missing:
        raise ValueError(f"unpacked content file set mismatch: extras={extras}, missing={missing}")


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT((info.external_attr >> 16) & 0xFFFF) == stat.S_IFLNK


def _validate_zip_files(
    archive_path: Path,
    manifest: Mapping[str, object],
    *,
    content_prefix: str,
    manifest_member: str,
) -> None:
    declared = {f"{content_prefix}/{row['path']}" for row in manifest["files"]}
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        names: set[str] = set()
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe ZIP member: {info.filename}")
            if _zip_member_is_symlink(info):
                raise ValueError(f"ZIP symlink is forbidden: {info.filename}")
            if info.is_dir():
                continue
            names.add(info.filename)
        if names != declared | {manifest_member}:
            raise ValueError(
                "ZIP content file set mismatch: "
                f"extras={sorted(names - declared - {manifest_member})}, "
                f"missing={sorted(declared - names)}"
            )
        internal = json.loads(archive.read(manifest_member).decode("utf-8"))
        if internal.get("manifest_hash") != manifest.get("manifest_hash"):
            raise ValueError("ZIP/internal source manifest identity mismatch")
        for row in manifest["files"]:
            name = f"{content_prefix}/{row['path']}"
            info = archive.getinfo(name)
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if info.file_size != int(row["bytes"]) or digest.hexdigest() != row["sha256"]:
                raise ValueError(f"ZIP member hash mismatch: {name}")


def validate_dataset_structure(payload: Mapping[str, object]) -> None:
    if payload.get("schema") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("unsupported Kaggle dataset manifest schema")
    if payload.get("mode") not in SUPPORTED_MODES:
        raise ValueError(f"unsupported Kaggle mode {payload.get('mode')!r}")
    if not str(payload.get("package_id", "")):
        raise ValueError("dataset manifest package_id is required")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("dataset manifest source object is required")
    for name in ("archive", "archive_sha256", "manifest", "manifest_sha256", "git_commit", "content_manifest_hash"):
        if not str(source.get(name, "")):
            raise ValueError(f"dataset source.{name} is required")
    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or not str(bootstrap.get("path", "")) or not str(bootstrap.get("sha256", "")):
        raise ValueError("dataset manifest bootstrap path/hash is required")


def _source_identity_matches(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("schema", "package_id", "git_commit", "manifest_hash"))


def resolve_dataset_input(manifest_path: str | Path) -> DatasetInput:
    manifest_path = Path(manifest_path).resolve()
    payload = _load_json(manifest_path)
    validate_dataset_structure(payload)
    base = manifest_path.parent.resolve()
    source_spec = payload["source"]
    external_manifest_path = _safe_relative(base, source_spec["manifest"], label="source.manifest")
    if not external_manifest_path.is_file() or _path_has_symlink(base, external_manifest_path):
        raise FileNotFoundError("dataset external source manifest is missing or unsafe")
    if sha256_file(external_manifest_path) != source_spec["manifest_sha256"]:
        raise ValueError("dataset external source manifest SHA-256 mismatch")
    expected = _load_json(external_manifest_path)
    _validate_declared_manifest(expected, schema=SOURCE_MANIFEST_SCHEMA, label="source")
    if expected.get("package_id") != payload.get("package_id") or expected.get("git_commit") != source_spec["git_commit"]:
        raise ValueError("dataset/source manifest package or commit mismatch")
    if expected.get("manifest_hash") != source_spec["content_manifest_hash"]:
        raise ValueError("dataset/source content manifest identity mismatch")

    zip_path = _safe_relative(base, source_spec["archive"], label="source.archive")
    zip_input: SourceInput | None = None
    if zip_path.exists():
        if zip_path.is_symlink() or not zip_path.is_file():
            raise ValueError("source archive path is not a regular file")
        actual_archive_sha = sha256_file(zip_path)
        if actual_archive_sha != source_spec["archive_sha256"]:
            raise ValueError("dataset source archive SHA-256 mismatch")
        source_root_name = str(expected.get("source_root", ""))
        if not source_root_name or "/" in source_root_name or "\\" in source_root_name:
            raise ValueError("source manifest source_root is invalid")
        _validate_zip_files(
            zip_path, expected, content_prefix=source_root_name,
            manifest_member=f"{source_root_name}/source_manifest.json",
        )
        zip_input = SourceInput(
            "zip", zip_path, None, dict(expected), str(expected["manifest_hash"]),
            actual_archive_sha, True,
        )

    directory_inputs: list[SourceInput] = []
    conflicting_manifests: list[str] = []
    for candidate in sorted(base.rglob("source_manifest.json")):
        if not candidate.is_file() or _path_has_symlink(base, candidate):
            continue
        try:
            candidate_payload = _load_json(candidate)
        except Exception:
            continue
        if candidate_payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
            continue
        if not _source_identity_matches(candidate_payload, expected):
            conflicting_manifests.append(str(candidate))
            continue
        root = candidate.parent.resolve()
        try:
            _validate_declared_manifest(candidate_payload, schema=SOURCE_MANIFEST_SCHEMA, label="source")
            allowed = {
                DATASET_MANIFEST_NAME, BOOTSTRAP_NAME, str(source_spec["archive"]),
            } if root == base else None
            _validate_directory_files(
                root, candidate_payload, manifest_name="source_manifest.json",
                allowed_extras=allowed,
            )
        except FileNotFoundError:
            # The external copy beside a retained ZIP is identity metadata, not
            # an unpacked source tree unless all declared files are present.
            continue
        directory_inputs.append(SourceInput(
            "directory", root, root, dict(candidate_payload),
            str(candidate_payload["manifest_hash"]), None, False,
        ))
    if conflicting_manifests:
        raise RuntimeError(f"conflicting unpacked source manifests found: {conflicting_manifests}")
    identities = {item.content_manifest_hash for item in directory_inputs}
    if len(identities) > 1:
        raise RuntimeError(f"multiple different unpacked source inputs found: {[str(x.path) for x in directory_inputs]}")
    if zip_input is not None and directory_inputs:
        if any(item.content_manifest_hash != zip_input.content_manifest_hash for item in directory_inputs):
            raise RuntimeError("source ZIP and unpacked directory have different content identities")
        # Identical ZIP+directory are one logical input; prefer the verified
        # directory to avoid relying on archive retention.
        selected = directory_inputs[0]
        selected = SourceInput(
            selected.kind, selected.path, selected.content_root, selected.content_manifest,
            selected.content_manifest_hash, zip_input.archive_sha256, True,
        )
    elif directory_inputs:
        selected = directory_inputs[0]
    elif zip_input is not None:
        selected = zip_input
    else:
        raise FileNotFoundError(
            "neither the declared source ZIP nor a validated unpacked source directory was found"
        )
    return DatasetInput(manifest_path, payload, external_manifest_path, selected)


def discover_dataset(input_root: str | Path) -> DatasetInput:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle input root does not exist: {root}")
    candidates = sorted(path for path in root.rglob(DATASET_MANIFEST_NAME) if path.is_file())
    if len(candidates) != 1:
        raise RuntimeError(
            "exactly one guandan_marvel_dataset.json is required; "
            f"found {len(candidates)}: {[str(path) for path in candidates]}"
        )
    return resolve_dataset_input(candidates[0])


def _recovery_zip_manifest(bundle: Path) -> dict[str, object]:
    with zipfile.ZipFile(bundle) as archive:
        infos = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        for info in infos.values():
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts or _zip_member_is_symlink(info):
                raise ValueError(f"unsafe recovery ZIP member: {info.filename}")
        if "recovery_manifest.json" not in infos:
            raise ValueError(f"recovery ZIP lacks root recovery_manifest.json: {bundle}")
        payload = json.loads(archive.read("recovery_manifest.json").decode("utf-8"))
        _validate_declared_manifest(payload, schema=RECOVERY_MANIFEST_SCHEMA, label="recovery")
        declared = {str(row["path"]) for row in payload.get("files", [])}
        if set(infos) != declared | {"recovery_manifest.json"}:
            raise ValueError("recovery ZIP file set differs from its manifest")
        for row in payload.get("files", []):
            name = str(row["path"])
            info = infos[name]
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if info.file_size != int(row["bytes"]) or digest.hexdigest() != row["sha256"]:
                raise ValueError(f"recovery ZIP member hash mismatch: {name}")
    return payload


def validate_recovery_directory(root: str | Path) -> dict[str, object]:
    root = Path(root).resolve()
    manifest_path = root / "recovery_manifest.json"
    if not manifest_path.is_file() or _path_has_symlink(root, manifest_path):
        raise FileNotFoundError("unpacked recovery manifest is missing or unsafe")
    payload = _load_json(manifest_path)
    _validate_declared_manifest(payload, schema=RECOVERY_MANIFEST_SCHEMA, label="recovery")
    _validate_directory_files(root, payload, manifest_name="recovery_manifest.json")
    return payload


def discover_recovery(
    input_root: str | Path,
    *,
    source_commit: str,
    source_archive_sha256: str,
    source_content_manifest_hash: str,
    required: bool,
) -> RecoveryInput | None:
    root = Path(input_root).resolve()
    candidates: list[RecoveryInput] = []
    for bundle in sorted(root.rglob("*.zip")):
        if bundle.name not in RECOVERY_BUNDLE_NAMES:
            continue
        if bundle.is_symlink():
            raise ValueError(f"recovery ZIP symlink is forbidden: {bundle}")
        payload = _recovery_zip_manifest(bundle)
        candidates.append(RecoveryInput(
            "zip", bundle.resolve(), None, payload, str(payload["manifest_hash"]), sha256_file(bundle),
        ))
    for manifest_path in sorted(root.rglob("recovery_manifest.json")):
        if _path_has_symlink(root, manifest_path):
            raise ValueError(f"unpacked recovery symlink is forbidden: {manifest_path}")
        payload = validate_recovery_directory(manifest_path.parent)
        candidates.append(RecoveryInput(
            "directory", manifest_path.parent.resolve(), manifest_path.parent.resolve(),
            payload, str(payload["manifest_hash"]), None,
        ))
    matches = [item for item in candidates if (
        item.manifest.get("source_commit") == source_commit
        and item.manifest.get("source_archive_sha256") == source_archive_sha256
        and item.manifest.get("source_content_manifest_hash") == source_content_manifest_hash
    )]
    nonmatching = [str(item.path) for item in candidates if item not in matches]
    if candidates and not matches and required:
        raise RuntimeError(
            "recovery inputs were found but belong to a different source package: "
            f"{nonmatching}; use the recovery produced by this exact Dataset"
        )
    identities = {item.content_manifest_hash for item in matches}
    if len(identities) > 1:
        raise RuntimeError(f"multiple different matching recovery inputs found: {[str(x.path) for x in matches]}")
    if not matches:
        if required:
            raise RuntimeError("a unique matching recovery ZIP or unpacked recovery directory is required but none was found")
        return None
    # ZIP+directory with the same manifest are one logical input. Prefer the
    # directory, but retain observed archive SHA when a ZIP is also present.
    directories = [item for item in matches if item.kind == "directory"]
    zips = [item for item in matches if item.kind == "zip"]
    selected = directories[0] if directories else zips[0]
    archive_sha = zips[0].archive_sha256 if zips else None
    return RecoveryInput(
        selected.kind, selected.path, selected.content_root, selected.manifest,
        selected.content_manifest_hash, archive_sha,
    )


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts or _zip_member_is_symlink(info):
                raise ValueError(f"unsafe ZIP member: {info.filename}")
        archive.extractall(destination)


def _copy_readonly_tree(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"input source root cannot be a symlink: {source}")
    shutil.copytree(source, destination, symlinks=False)




def materialize_recovery_input(recovery: RecoveryInput, destination: str | Path) -> Path:
    target = Path(destination).resolve()
    if target.exists():
        shutil.rmtree(target)
    if recovery.kind == "directory":
        _copy_readonly_tree(recovery.path, target)
    else:
        target.mkdir(parents=True)
        _safe_extract(recovery.path, target)
    validate_recovery_directory(target)
    return target

def _copy_manifest_tree(
    source: Path, destination: Path, manifest: Mapping[str, object], *, manifest_name: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / manifest_name, destination / manifest_name)
    for row in manifest["files"]:
        relative = Path(str(row["path"]))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)

def prepare_inputs(
    dataset_manifest_path: str | Path,
    *,
    input_root: str | Path,
    working_root: str | Path,
) -> dict[str, object]:
    dataset = resolve_dataset_input(dataset_manifest_path)
    working = Path(working_root).resolve()
    working.mkdir(parents=True, exist_ok=True)
    source_target = working / "guandan-marvel-source"
    if source_target.exists():
        shutil.rmtree(source_target)
    if dataset.source.kind == "directory":
        _copy_manifest_tree(
            dataset.source.path, source_target, dataset.source.content_manifest,
            manifest_name="source_manifest.json",
        )
    else:
        with tempfile.TemporaryDirectory(prefix="guandan-source-", dir=working) as temp_name:
            temp = Path(temp_name)
            _safe_extract(dataset.source.path, temp)
            extracted = temp / str(dataset.source.content_manifest["source_root"])
            _validate_directory_files(extracted, dataset.source.content_manifest, manifest_name="source_manifest.json")
            shutil.move(str(extracted), source_target)
    _validate_directory_files(source_target, dataset.source.content_manifest, manifest_name="source_manifest.json")

    source_spec = dataset.manifest["source"]
    recovery_required = bool((dataset.manifest.get("recovery") or {}).get("required", False))
    recovery = discover_recovery(
        input_root,
        source_commit=str(source_spec["git_commit"]),
        source_archive_sha256=str(source_spec["archive_sha256"]),
        source_content_manifest_hash=str(source_spec["content_manifest_hash"]),
        required=recovery_required,
    )
    recovery_target: Path | None = None
    restored_binary = False
    if recovery is not None:
        recovery_target = materialize_recovery_input(
            recovery, working / "guandan-marvel-recovery",
        )
        recovery_manifest = validate_recovery_directory(recovery_target)
        runtime_binary = recovery_manifest.get("runtime_binary")
        if isinstance(runtime_binary, Mapping):
            source_binary = _safe_relative(recovery_target, runtime_binary["archive_path"], label="runtime_binary.archive_path")
            target_binary = _safe_relative(source_target, runtime_binary["restore_relative_path"], label="runtime_binary.restore_relative_path")
            if sha256_file(source_binary) != runtime_binary["sha256"]:
                raise ValueError("recovery runtime binary hash mismatch")
            target_binary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_binary, target_binary)
            restored_binary = True
    return {
        "dataset_manifest": str(dataset.manifest_path),
        "mode": dataset.manifest["mode"],
        "source_root": str(source_target),
        "source_input_kind": dataset.source.kind,
        "source_content_manifest_hash": dataset.source.content_manifest_hash,
        "source_archive_sha256": source_spec["archive_sha256"],
        "source_archive_sha256_verified": dataset.source.archive_sha256_verified,
        "source_archive_observed_sha256": dataset.source.archive_sha256,
        "recovery_root": None if recovery_target is None else str(recovery_target),
        "recovery_input_kind": None if recovery is None else recovery.kind,
        "recovery_archive_observed_sha256": None if recovery is None else recovery.archive_sha256,
        "runtime_binary_restored": restored_binary,
    }


__all__ = [
    "BOOTSTRAP_NAME", "DATASET_MANIFEST_NAME", "DATASET_MANIFEST_SCHEMA", "DatasetInput",
    "RECOVERY_BUNDLE_NAMES", "RECOVERY_MANIFEST_SCHEMA", "RecoveryInput", "SOURCE_MANIFEST_SCHEMA",
    "SUPPORTED_MODES", "SourceInput", "canonical_hash", "discover_dataset", "discover_recovery",
    "materialize_recovery_input", "prepare_inputs", "resolve_dataset_input", "sha256_file", "validate_dataset_structure",
    "validate_recovery_directory",
]
