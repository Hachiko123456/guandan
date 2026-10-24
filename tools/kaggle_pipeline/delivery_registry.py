"""Validated delivery publication and conservative local cleanup planning.

This module is deliberately local-only.  Publication reads and hashes an
explicit delivery, while cleanup merely describes paths that a caller may
choose to remove; it never deletes anything.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping


CURRENT_SCHEMA = "guandan-kaggle-current-delivery-v1"
VERIFICATION_SCHEMA = "guandan-serial-compare-delivery-verification-v1"
QA_SCHEMA = "guandan-serial-compare-artifact-qa-v1"
CLEANUP_SCHEMA = "guandan-kaggle-delivery-cleanup-plan-v1"
PAYLOAD_NAME = "guandan_serial_compare_payload_v1.bin"
NOTEBOOK_NAME = "guandan_serial_compare_v1.ipynb"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DELIVERY_NAME = re.compile(r"delivery_v([0-9]+)")
_BUILD_NAME = re.compile(r"delivery_v([0-9]+)_build")


class RegistryError(RuntimeError):
    """A delivery or registry record failed closed validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RegistryError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any], *, omitted: str = "hash") -> str:
    body = {key: item for key, item in value.items() if key != omitted}
    try:
        encoded = json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RegistryError("record is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RegistryError(f"{label} must be a JSON object")
    return value


def _require_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise RegistryError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_symlink_chain(path: Path, parent: Path, label: str) -> None:
    """Reject symlinks from ``parent`` down to ``path`` without following them."""
    try:
        relative = path.relative_to(parent)
    except ValueError as error:
        raise RegistryError(f"{label} escapes its allowed root") from error
    cursor = parent
    for part in relative.parts:
        cursor = cursor / part
        try:
            if cursor.is_symlink():
                raise RegistryError(f"{label} may not use symlinks: {cursor}")
        except OSError as error:
            raise RegistryError(f"cannot inspect {label}: {cursor}") from error


def _repo_path(repo: Path, value: str | os.PathLike[str], label: str,
               *, must_exist: bool = True) -> Path:
    raw = Path(value)
    lexical = raw if raw.is_absolute() else repo / raw
    lexical = Path(os.path.abspath(lexical))
    _reject_symlink_chain(lexical, repo, label)
    try:
        resolved = lexical.resolve(strict=must_exist)
    except (OSError, RuntimeError) as error:
        raise RegistryError(f"{label} does not exist: {lexical}") from error
    if not _inside(resolved, repo):
        raise RegistryError(f"{label} escapes the repository")
    return resolved


def _relative(repo: Path, path: Path, label: str) -> str:
    if not _inside(path, repo):
        raise RegistryError(f"{label} escapes the repository")
    value = path.relative_to(repo).as_posix()
    if not value or value == "." or value.startswith("../"):
        raise RegistryError(f"invalid repository-relative {label}")
    return value


def _regular_file(path: Path, delivery: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    _reject_symlink_chain(lexical, delivery, label)
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RegistryError(f"missing {label}: {path}") from error
    if not _inside(resolved, delivery) or not resolved.is_file() or resolved.is_symlink():
        raise RegistryError(f"{label} must be a regular file inside the delivery")
    return resolved


def _delivery_file(delivery: Path, value: Any, expected: str, label: str) -> Path:
    if value != expected:
        raise RegistryError(f"DELIVERY_VERIFICATION has an unexpected {label} name")
    return _regular_file(delivery / expected, delivery, label)


def _validate_verification(delivery: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    verification_path = _regular_file(
        delivery / "DELIVERY_VERIFICATION.json", delivery, "DELIVERY_VERIFICATION",
    )
    verification = _read_json(verification_path, "DELIVERY_VERIFICATION")
    if verification.get("schema") != VERIFICATION_SCHEMA:
        raise RegistryError("unsupported DELIVERY_VERIFICATION schema")
    if not isinstance(verification.get("status"), str) or not verification["status"]:
        raise RegistryError("DELIVERY_VERIFICATION has no status")
    if any(word in verification["status"].upper() for word in ("FAIL", "ERROR", "INVALID")):
        raise RegistryError("DELIVERY_VERIFICATION records a failed delivery")

    files = {
        "payload": _delivery_file(
            delivery, verification.get("payload"), PAYLOAD_NAME, "payload",
        ),
        "notebook": _delivery_file(
            delivery, verification.get("notebook"), NOTEBOOK_NAME, "notebook",
        ),
        "plan": _regular_file(delivery / "plan.json", delivery, "plan"),
        "manifest": _regular_file(
            delivery / "DELIVERY_MANIFEST.json", delivery, "DELIVERY_MANIFEST",
        ),
    }
    expected_size = verification.get("payload_bytes")
    if type(expected_size) is not int or expected_size < 0:
        raise RegistryError("DELIVERY_VERIFICATION has an invalid payload size")
    if files["payload"].stat().st_size != expected_size:
        raise RegistryError("payload size differs from DELIVERY_VERIFICATION")

    actual_payload = _sha256(files["payload"])
    actual_notebook = _sha256(files["notebook"])
    if actual_payload != _require_hex(verification.get("payload_sha256"), "payload_sha256"):
        raise RegistryError("payload hash differs from DELIVERY_VERIFICATION")
    if actual_notebook != _require_hex(verification.get("notebook_sha256"), "notebook_sha256"):
        raise RegistryError("notebook hash differs from DELIVERY_VERIFICATION")

    plan = _read_json(files["plan"], "plan")
    plan_hash = _require_hex(verification.get("plan_hash"), "plan_hash")
    if plan.get("plan_hash") != plan_hash:
        raise RegistryError("plan identity differs from DELIVERY_VERIFICATION")
    if plan_hash != _canonical_hash(plan, omitted="plan_hash"):
        raise RegistryError("plan hash is internally inconsistent")

    manifest = _read_json(files["manifest"], "DELIVERY_MANIFEST")
    manifest_hash = _require_hex(verification.get("manifest_hash"), "manifest_hash")
    if manifest.get("manifest_hash") != manifest_hash:
        raise RegistryError("manifest identity differs from DELIVERY_VERIFICATION")
    if manifest_hash != _canonical_hash(manifest, omitted="manifest_hash"):
        raise RegistryError("DELIVERY_MANIFEST hash is internally inconsistent")
    if manifest.get("plan_hash") != plan_hash:
        raise RegistryError("DELIVERY_MANIFEST is not bound to the plan")
    return verification, files


def _resolve_qa_path(repo: Path, delivery: Path, verification: Mapping[str, Any],
                     qa_path: str | os.PathLike[str] | None) -> Path:
    if qa_path is not None:
        explicit = _repo_path(repo, qa_path, "artifact QA")
        declared = verification.get("artifact_qa")
        if declared is not None:
            if not isinstance(declared, str) or not declared:
                raise RegistryError("DELIVERY_VERIFICATION artifact_qa is invalid")
            candidate = Path(declared)
            if not candidate.is_absolute():
                candidate = delivery / candidate
            if _repo_path(repo, candidate, "declared artifact QA") != explicit:
                raise RegistryError("explicit QA differs from DELIVERY_VERIFICATION")
        return explicit
    declared = verification.get("artifact_qa")
    if declared is not None:
        if not isinstance(declared, str) or not declared:
            raise RegistryError("DELIVERY_VERIFICATION artifact_qa is invalid")
        candidate = Path(declared)
        if not candidate.is_absolute():
            candidate = delivery / candidate
        return _repo_path(repo, candidate, "artifact QA")
    return _repo_path(repo, delivery / "ARTIFACT_QA.json", "artifact QA")


def _qa_delivery(repo: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise RegistryError("artifact QA is not bound to a delivery directory")
    return _repo_path(repo, value, "artifact QA delivery")


def _validate_qa(qa: Mapping[str, Any], *, repo: Path, delivery: Path,
                 payload_sha256: str, notebook_sha256: str, plan_hash: str,
                 payload_bytes: int, label: str) -> None:
    if qa.get("schema") != QA_SCHEMA:
        raise RegistryError(f"unsupported {label} schema")
    if qa.get("status") != "PASS":
        raise RegistryError(f"{label} status is not PASS")
    if _qa_delivery(repo, qa.get("delivery")) != delivery:
        raise RegistryError(f"{label} belongs to another delivery")
    payload = qa.get("payload")
    notebook = qa.get("notebook")
    if not isinstance(payload, Mapping) or payload.get("sha256") != payload_sha256:
        raise RegistryError(f"{label} payload binding differs")
    if payload.get("bytes") != payload_bytes:
        raise RegistryError(f"{label} payload size binding differs")
    if not isinstance(notebook, Mapping) or notebook.get("sha256") != notebook_sha256:
        raise RegistryError(f"{label} notebook binding differs")
    if qa.get("plan_hash") != plan_hash:
        raise RegistryError(f"{label} plan binding differs")
    declared = _require_hex(qa.get("hash"), f"{label} hash")
    if declared != _canonical_hash(qa):
        raise RegistryError(f"{label} hash is internally inconsistent")


def _default_verifier(delivery: Path, *, repo: Path) -> dict[str, Any]:
    try:
        from scripts.verify_t4x2_serial_compare_delivery import verify
        return verify(delivery, repo=repo)
    except RegistryError:
        raise
    except Exception as error:  # verifier failures are part of registry validation
        raise RegistryError(f"serial-compare verifier failed: {error}") from error


def _run_verifier(verifier: Callable[..., Any] | None, delivery: Path,
                  repo: Path) -> dict[str, Any]:
    selected = verifier or _default_verifier
    try:
        parameters = inspect.signature(selected).parameters
        accepts_repo = "repo" in parameters or any(
            item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
        )
        result = selected(delivery, repo=repo) if accepts_repo else selected(delivery)
    except RegistryError:
        raise
    except Exception as error:
        raise RegistryError(f"serial-compare verifier failed: {error}") from error
    if not isinstance(result, dict):
        raise RegistryError("serial-compare verifier did not return a record")
    return result


def _delivery_id(delivery: Path, manifest_hash: str, payload_sha256: str) -> str:
    match = _DELIVERY_NAME.fullmatch(delivery.name)
    version = f"v{int(match.group(1))}" if match else re.sub(
        r"[^a-z0-9]+", "-", delivery.name.lower(),
    ).strip("-")[:32]
    if not version:
        version = "delivery"
    identity = hashlib.sha256(
        (delivery.name + "\0" + manifest_hash + "\0" + payload_sha256).encode("utf-8")
    ).hexdigest()[:16]
    return f"serial-compare-{version}-{identity}"


def inspect_delivery(repo_root: str | os.PathLike[str],
                     delivery: str | os.PathLike[str],
                     qa_path: str | os.PathLike[str] | None = None,
                     verifier: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Validate an explicit delivery and return an unpublished registry record."""
    repo = Path(repo_root).resolve(strict=True)
    delivery_path = _repo_path(repo, delivery, "delivery")
    if not delivery_path.is_dir():
        raise RegistryError("delivery must be an explicit directory")
    verification, files = _validate_verification(delivery_path)
    qa_file = _resolve_qa_path(repo, delivery_path, verification, qa_path)
    if not qa_file.is_file() or qa_file.is_symlink():
        raise RegistryError("artifact QA must be a regular file")

    payload_sha256 = _sha256(files["payload"])
    notebook_sha256 = _sha256(files["notebook"])
    plan_sha256 = _sha256(files["plan"])
    plan_hash = str(verification["plan_hash"])
    manifest_hash = str(verification["manifest_hash"])
    saved_qa = _read_json(qa_file, "artifact QA")
    _validate_qa(
        saved_qa, repo=repo, delivery=delivery_path,
        payload_sha256=payload_sha256, notebook_sha256=notebook_sha256,
        plan_hash=plan_hash, payload_bytes=files["payload"].stat().st_size,
        label="artifact QA",
    )
    actual_qa = _run_verifier(verifier, delivery_path, repo)
    _validate_qa(
        actual_qa, repo=repo, delivery=delivery_path,
        payload_sha256=payload_sha256, notebook_sha256=notebook_sha256,
        plan_hash=plan_hash, payload_bytes=files["payload"].stat().st_size,
        label="actual verifier result",
    )

    return {
        "schema": CURRENT_SCHEMA,
        "delivery_id": _delivery_id(delivery_path, manifest_hash, payload_sha256),
        "delivery_dir": _relative(repo, delivery_path, "delivery path"),
        "payload": files["payload"].name,
        "payload_sha256": payload_sha256,
        "notebook": files["notebook"].name,
        "notebook_sha256": notebook_sha256,
        "plan_sha256": plan_sha256,
        "manifest_hash": manifest_hash,
        "artifact_qa_path": _relative(repo, qa_file, "artifact QA path"),
        "artifact_qa_hash": _sha256(qa_file),
    }


def _timestamp(clock: Callable[[], Any] | None) -> str:
    value = (clock or (lambda: datetime.now(timezone.utc)))()
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise RegistryError("publication clock must return a timezone-aware datetime")
        return value.isoformat()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise RegistryError("publication clock returned an invalid timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise RegistryError("publication timestamp must include a timezone")
        return value
    raise RegistryError("publication clock must return datetime or ISO timestamp")


def _encoded(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _install_immutable(path: Path, data: bytes) -> bool:
    """Atomically install a complete history file without replacing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _registry_root(repo: Path, registry_root: str | os.PathLike[str] | None) -> Path:
    value = registry_root if registry_root is not None else Path("runs/kaggle_delivery")
    root = _repo_path(repo, value, "registry root", must_exist=False)
    _reject_symlink_chain(root, repo, "registry root")
    return root


def publish(repo_root: str | os.PathLike[str], delivery: str | os.PathLike[str],
            qa_path: str | os.PathLike[str],
            registry_root: str | os.PathLike[str] | None = None,
            verifier: Callable[..., Any] | None = None,
            clock: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Validate and atomically publish current plus immutable history."""
    repo = Path(repo_root).resolve(strict=True)
    inspected = inspect_delivery(repo, delivery, qa_path=qa_path, verifier=verifier)
    root = _registry_root(repo, registry_root)
    history_dir = root / "history"
    history_path = history_dir / f"{inspected['delivery_id']}.json"
    current_path = root / "current.json"

    record = {**inspected, "published_at": _timestamp(clock)}
    if history_path.exists():
        if history_path.is_symlink() or not history_path.is_file():
            raise RegistryError("history target is not an immutable regular file")
        existing = _read_json(history_path, "delivery history")
        existing_identity = {key: value for key, value in existing.items() if key != "published_at"}
        if existing_identity != inspected or not isinstance(existing.get("published_at"), str):
            raise RegistryError("delivery history identity collision")
        record = existing
    else:
        installed = _install_immutable(history_path, _encoded(record))
        if not installed:
            existing = _read_json(history_path, "delivery history")
            existing_identity = {key: value for key, value in existing.items() if key != "published_at"}
            if existing_identity != inspected or not isinstance(existing.get("published_at"), str):
                raise RegistryError("delivery history identity collision")
            record = existing

    _atomic_replace(current_path, _encoded(record))
    return record


def _validate_current_shape(record: Mapping[str, Any]) -> None:
    required = {
        "schema", "delivery_id", "delivery_dir", "payload", "payload_sha256",
        "notebook", "notebook_sha256", "plan_sha256", "manifest_hash",
        "artifact_qa_path", "artifact_qa_hash", "published_at",
    }
    if set(record) != required or record.get("schema") != CURRENT_SCHEMA:
        raise RegistryError("unsupported or incomplete current delivery record")
    for key in ("delivery_id", "delivery_dir", "payload", "notebook",
                "artifact_qa_path", "published_at"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise RegistryError(f"current delivery has invalid {key}")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", str(record["delivery_id"])) is None:
        raise RegistryError("current delivery_id is unsafe")
    for key in ("payload_sha256", "notebook_sha256", "plan_sha256",
                "manifest_hash", "artifact_qa_hash"):
        _require_hex(record.get(key), key)
    try:
        parsed = datetime.fromisoformat(str(record["published_at"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise RegistryError("current delivery has an invalid published_at") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RegistryError("current delivery published_at lacks a timezone")


def resolve_current(repo_root: str | os.PathLike[str],
                    registry_root: str | os.PathLike[str] | None = None
                    ) -> tuple[Path, dict[str, Any]]:
    """Resolve current and recheck every persisted hash and QA binding."""
    repo = Path(repo_root).resolve(strict=True)
    root = _registry_root(repo, registry_root)
    current_path = _repo_path(repo, root / "current.json", "current delivery pointer")
    if current_path.is_symlink() or not current_path.is_file():
        raise RegistryError("current delivery pointer is not a regular file")
    record = _read_json(current_path, "current delivery pointer")
    _validate_current_shape(record)

    delivery = _repo_path(repo, record["delivery_dir"], "current delivery")
    if not delivery.is_dir():
        raise RegistryError("current delivery is not a directory")
    verification, files = _validate_verification(delivery)
    if files["payload"].name != record["payload"] or _sha256(files["payload"]) != record["payload_sha256"]:
        raise RegistryError("current payload changed")
    if files["notebook"].name != record["notebook"] or _sha256(files["notebook"]) != record["notebook_sha256"]:
        raise RegistryError("current notebook changed")
    if _sha256(files["plan"]) != record["plan_sha256"]:
        raise RegistryError("current plan changed")
    if verification["manifest_hash"] != record["manifest_hash"]:
        raise RegistryError("current manifest identity changed")

    qa_file = _repo_path(repo, record["artifact_qa_path"], "current artifact QA")
    if not qa_file.is_file() or qa_file.is_symlink() or _sha256(qa_file) != record["artifact_qa_hash"]:
        raise RegistryError("current artifact QA changed")
    qa = _read_json(qa_file, "current artifact QA")
    _validate_qa(
        qa, repo=repo, delivery=delivery,
        payload_sha256=str(record["payload_sha256"]),
        notebook_sha256=str(record["notebook_sha256"]),
        plan_hash=str(verification["plan_hash"]),
        payload_bytes=files["payload"].stat().st_size,
        label="current artifact QA",
    )
    expected_id = _delivery_id(delivery, str(record["manifest_hash"]), str(record["payload_sha256"]))
    if record["delivery_id"] != expected_id:
        raise RegistryError("current delivery_id is not bound to the delivery identity")
    return delivery, record


def _tree_bytes(path: Path) -> int:
    total = 0
    for root, directories, files in os.walk(path, topdown=True, followlinks=False):
        root_path = Path(root)
        directories[:] = [
            name for name in directories if not (root_path / name).is_symlink()
        ]
        for name in files:
            item = root_path / name
            if item.is_symlink():
                continue
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError as error:
                raise RegistryError(f"cannot compute bytes for {item}") from error
    return total


def _referenced_delivery_values(value: Any) -> list[str]:
    found: list[str] = []
    if not isinstance(value, Mapping):
        return found
    for key, item in value.items():
        if key in {"delivery", "delivery_dir"} and isinstance(item, str):
            found.append(item)
        if isinstance(item, Mapping):
            found.extend(_referenced_delivery_values(item))
    return found


def cleanup_plan(repo_root: str | os.PathLike[str],
                 delivery_parent: str | os.PathLike[str],
                 current_record: Mapping[str, Any],
                 pipeline_root: str | os.PathLike[str],
                 keep_previous: int = 1) -> dict[str, Any]:
    """Return a version-based cleanup plan.  This function never deletes."""
    if type(keep_previous) is not int or keep_previous < 0:
        raise RegistryError("keep_previous must be a non-negative integer")
    repo = Path(repo_root).resolve(strict=True)
    parent = _repo_path(repo, delivery_parent, "delivery parent")
    if not parent.is_dir():
        raise RegistryError("delivery parent is not a directory")
    pipeline = _repo_path(repo, pipeline_root, "pipeline root", must_exist=False)
    if pipeline.exists() and not pipeline.is_dir():
        raise RegistryError("pipeline root is not a directory")

    current_value = current_record.get("delivery_dir")
    if not isinstance(current_value, str):
        raise RegistryError("current record has no delivery_dir")
    current = _repo_path(repo, current_value, "current delivery")
    if current.parent != parent or _DELIVERY_NAME.fullmatch(current.name) is None:
        raise RegistryError("current delivery is outside the explicit delivery parent")

    deliveries: dict[int, Path] = {}
    builds: dict[int, Path] = {}
    try:
        children = list(parent.iterdir())
    except OSError as error:
        raise RegistryError("cannot inspect delivery parent") from error
    for child in children:
        delivery_match = _DELIVERY_NAME.fullmatch(child.name)
        build_match = _BUILD_NAME.fullmatch(child.name)
        if delivery_match is None and build_match is None:
            continue
        if child.is_symlink():
            try:
                target = child.resolve(strict=False)
            except OSError as error:
                raise RegistryError(f"cannot inspect symlink candidate: {child}") from error
            if not _inside(target, parent):
                raise RegistryError(f"delivery cleanup target escapes its parent: {child}")
            raise RegistryError(f"delivery cleanup target may not be a symlink: {child}")
        resolved = child.resolve(strict=True)
        if not _inside(resolved, parent) or resolved.parent != parent or not resolved.is_dir():
            raise RegistryError(f"unsafe delivery cleanup target: {child}")
        version = int((delivery_match or build_match).group(1))
        destination = deliveries if delivery_match is not None else builds
        if version in destination:
            raise RegistryError(f"duplicate delivery version: {version}")
        destination[version] = resolved

    current_version = int(_DELIVERY_NAME.fullmatch(current.name).group(1))
    if deliveries.get(current_version) != current:
        raise RegistryError("current delivery is not present under its parent")
    previous = sorted((version for version in deliveries if version < current_version), reverse=True)
    retained_versions = {current_version, *previous[:keep_previous]}
    reasons: dict[int, set[str]] = {current_version: {"current"}}
    for version in previous[:keep_previous]:
        reasons.setdefault(version, set()).add("previous")

    active_runs: list[str] = []
    if pipeline.exists():
        try:
            run_dirs = sorted(pipeline.iterdir(), key=lambda item: item.name)
        except OSError as error:
            raise RegistryError("cannot inspect pipeline runs") from error
        for run_dir in run_dirs:
            if run_dir.is_symlink() or not run_dir.is_dir():
                continue
            run_json = run_dir / "run.json"
            if not run_json.exists():
                continue
            if run_json.is_symlink() or not run_json.is_file():
                raise RegistryError(f"unsafe pipeline run record: {run_json}")
            state = _read_json(run_json, "pipeline run")
            for reference in _referenced_delivery_values(state):
                try:
                    referenced = _repo_path(repo, reference, "pipeline delivery reference")
                except RegistryError as error:
                    raise RegistryError(f"unsafe delivery reference in {run_json}") from error
                if referenced.parent != parent:
                    continue
                match = _DELIVERY_NAME.fullmatch(referenced.name)
                if match is None:
                    continue
                version = int(match.group(1))
                if deliveries.get(version) == referenced:
                    retained_versions.add(version)
                    reasons.setdefault(version, set()).add("active_run")
                    active_runs.append(_relative(repo, run_json, "pipeline run record"))

    candidate_paths: list[tuple[int, str, Path]] = []
    retained_paths: list[tuple[int, str, Path]] = []
    for version, delivery_path in deliveries.items():
        target = retained_paths if version in retained_versions else candidate_paths
        target.append((version, "delivery", delivery_path))
        build = builds.get(version)
        if build is not None:
            target.append((version, "build", build))

    def row(item: tuple[int, str, Path], *, retained: bool) -> dict[str, Any]:
        version, kind, path = item
        result: dict[str, Any] = {
            "absolute_path": str(path),
            "relative_path": _relative(repo, path, "cleanup target"),
            "version": version,
            "kind": kind,
            "bytes": _tree_bytes(path),
        }
        if retained:
            result["reasons"] = sorted(reasons.get(version, {"paired_with_retained"}))
        return result

    candidates = [row(item, retained=False) for item in candidate_paths]
    retained = [row(item, retained=True) for item in retained_paths]
    candidates.sort(key=lambda item: (item["relative_path"], item["absolute_path"]))
    retained.sort(key=lambda item: (item["relative_path"], item["absolute_path"]))
    return {
        "schema": CLEANUP_SCHEMA,
        "delivery_parent": _relative(repo, parent, "delivery parent"),
        "current_delivery": _relative(repo, current, "current delivery"),
        "keep_previous": keep_previous,
        "active_run_records": sorted(set(active_runs)),
        "retained": retained,
        "candidates": candidates,
        "candidate_bytes": sum(int(item["bytes"]) for item in candidates),
    }


__all__ = [
    "RegistryError", "inspect_delivery", "publish", "resolve_current",
    "cleanup_plan",
]
