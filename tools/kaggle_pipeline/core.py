"""Prepare immutable deliveries, submit once and fetch version-pinned output."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid
import zipfile

from tools.serial_compare.common import atomic_json, canonical, read_json, sha
from tools.serial_compare.delivery import NOTEBOOK, PAYLOAD
from .results import verify_results, ResultError
from .event_log import EventLog, sanitize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/kaggle_pipeline.json"


class PipelineError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = read_json(path)
    if config.get("schema") != "guandan-kaggle-pipeline-config-v1":
        raise PipelineError("unsupported pipeline configuration")
    config.setdefault("registry_root", "runs/kaggle_delivery")
    config.setdefault("delivery_parent", "runs/t4x2_serial_compare_delivery_20261023")
    config.setdefault("heartbeat_seconds", 300)
    config.setdefault("keep_previous_deliveries", 1)
    for field in ("account_file", "output_root", "registry_root", "delivery_parent"):
        config[field] = str((ROOT / config[field]).resolve())
    if config.get("accelerator") != "NvidiaTeslaT4":
        raise PipelineError("this delivery requires NvidiaTeslaT4 (T4 x2)")
    for field in ("timeout_seconds", "poll_seconds", "heartbeat_seconds", "dataset_wait_seconds"):
        value = config.get(field)
        if type(value) is not int or value <= 0:
            raise PipelineError("invalid positive integer: " + field)
    if config.get("enable_internet") is not False:
        raise PipelineError("current deliveries require the offline runtime")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,16}", config.get("experiment", "")):
        raise PipelineError("experiment must be a short lowercase slug")
    if type(config.get("keep_previous_deliveries")) is not int or config["keep_previous_deliveries"] < 0:
        raise PipelineError("keep_previous_deliveries must be a non-negative integer")
    return config


def credentials(config: dict) -> dict:
    path = Path(config["account_file"])
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        token = data.get("api_token", "")
        username, key = data.get("username", ""), data.get("key", "")
        if not all(isinstance(value, str) for value in (username, key, token)):
            raise ValueError()
        if token:
            if key or not token.startswith("KGAT_") or len(token) <= 5 or token != token.strip() or any(c.isspace() for c in token):
                raise ValueError()
            if username and not re.fullmatch(r"[A-Za-z0-9_-]+", username):
                raise ValueError()
            return {"username": username, "key": "", "api_token": token, "auth_method": "api_token"}
        if key.startswith("KGAT_"):
            raise ValueError()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", username) or not key.strip() or key != key.strip():
            raise ValueError()
    except (OSError, ValueError, TypeError, AttributeError):
        raise PipelineError(f"Configure api_token, or legacy username/key, in {path}; do not mix the two formats.") from None
    return {"username": username, "key": key, "api_token": None, "auth_method": "legacy"}


def account(config: dict) -> tuple[str, str]:
    values = credentials(config)
    return values["username"], values["key"]


def doctor(config: dict) -> dict:
    try:
        sdk = version("kaggle")
    except PackageNotFoundError:
        sdk = None
    try:
        values = credentials(config)
        configured = True
    except PipelineError:
        configured = False
    return {
        "status": "READY" if configured and sdk == "2.2.5" else "ACCOUNT_REQUIRED" if not configured else "SDK_REQUIRED",
        "account_file": config["account_file"], "account_configured": configured,
        "auth_method": values["auth_method"] if configured else None,
        "kaggle_version": sdk, "accelerator": config["accelerator"],
        "default_delivery": None, "remote_calls": False,
    }


def check_connection(config: dict, *, backend=None) -> dict:
    """User-invoked authenticated reads only; cache resolved owner, never token output."""
    values = credentials(config)
    if backend is None:
        from .backend import KaggleBackend
        backend = KaggleBackend(values["username"], values["key"], api_token=values["api_token"])
    checked = backend.check_connection()
    if checked.get("dataset_read") is not True or checked.get("kernel_read") is not True:
        raise PipelineError("Kaggle Dataset/Notebook read access was not confirmed")
    username = checked.get("username")
    if checked.get("authenticated") is not True or not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise PipelineError("Kaggle did not confirm an active account")
    if values["username"] and values["username"] != username:
        raise PipelineError("authenticated owner differs from the configured username")
    if values["api_token"] and not values["username"]:
        # Do not overwrite a credential file the user changed during the request.
        current = credentials(config)
        if current != values:
            raise PipelineError("account configuration changed during connection check; rerun check")
        atomic_json(Path(config["account_file"]), {"api_token": values["api_token"], "username": username})
    return {"status": "CONNECTED", "authenticated": True, "username": username,
            "auth_method": values["auth_method"], "dataset_read": checked.get("dataset_read") is True,
            "kernel_read": checked.get("kernel_read") is True, "remote_mutations": False,
            "training_started": False}


@contextmanager
def run_lock(folder: Path):
    """OS releases the advisory lock even if the client is terminated."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise PipelineError("another process owns this run; retry after it exits") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def save(folder: Path, state: dict) -> None:
    state["updated_at"] = now()
    atomic_json(folder / "run.json", state)


def _log(folder: Path, state: dict) -> EventLog:
    return EventLog(folder, str(state["run_id"]))


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(sanitize(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def locate(config: dict, run_id: str | None = None) -> Path:
    root = Path(config["output_root"])
    if run_id is None:
        if not (root / "latest.json").is_file():
            raise PipelineError("No prepared new delivery. First use prepare --delivery <new-delivery-directory>.")
        run_id = read_json(root / "latest.json")["run_id"]
    if not re.fullmatch(r"[a-z0-9-]{8,64}", run_id):
        raise PipelineError("invalid run_id")
    folder = root / run_id
    if not (folder / "run.json").is_file():
        raise PipelineError("unknown run_id")
    return folder


def local_status(config: dict, run_id: str | None = None) -> dict:
    folder = locate(config, run_id)
    state = read_json(folder / "run.json")
    return {**state, "remote_calls": False, "logs": {
        "events": str(folder / "events.jsonl"), "console": str(folder / "console.log"),
        "failure": str(folder / "failure.json"),
    }}


def current_delivery(config: dict) -> tuple[Path, dict]:
    from .delivery_registry import resolve_current
    return resolve_current(ROOT, config["registry_root"])


def publish_delivery(config: dict, delivery: Path, qa: Path) -> dict:
    from .delivery_registry import publish
    return publish(ROOT, delivery, qa, config["registry_root"])


def prepare_published(config: dict, *, dry_run: bool = False) -> dict:
    delivery, record = current_delivery(config)
    qa = read_json(ROOT / record["artifact_qa_path"])
    return prepare(config, delivery, dry_run=dry_run, verifier=lambda _delivery: qa)


def prune_deliveries(config: dict, *, keep_previous: int | None = None,
                     execute: bool = False) -> dict:
    from .delivery_registry import cleanup_plan
    _, current = current_delivery(config)
    retained = config["keep_previous_deliveries"] if keep_previous is None else keep_previous
    if type(retained) is not int or retained < 0:
        raise PipelineError("keep_previous must be a non-negative integer")
    plan = cleanup_plan(
        ROOT, config["delivery_parent"], current, config["output_root"],
        keep_previous=retained,
    )
    if not execute:
        return {**plan, "executed": False, "remote_resources_deleted": False}
    parent = Path(config["delivery_parent"]).resolve()
    removed = []
    for row in plan["candidates"]:
        target = Path(row["absolute_path"])
        resolved = target.resolve(strict=True)
        if resolved.parent != parent or target.is_symlink() or not resolved.is_dir():
            raise PipelineError("cleanup target changed after planning; nothing further was removed")
        shutil.rmtree(resolved)
        removed.append({"relative_path": row["relative_path"], "bytes": row["bytes"]})
    return {**plan, "executed": True, "removed": removed,
            "removed_bytes": sum(int(row["bytes"]) for row in removed),
            "remote_resources_deleted": False, "recoverable": False}


def verify_delivery(delivery: Path) -> dict:
    # This verifies the actual payload in isolation and its workspace identity.
    from scripts.verify_t4x2_serial_compare_delivery import verify
    return verify(delivery)


def prepare(config: dict, delivery: Path | None, *, dry_run: bool = False,
            verifier=None) -> dict:
    if delivery is None:
        if dry_run:
            return {"status": "DRY_RUN", "delivery": None, "remote_calls": False,
                    "next": "prepare --delivery <new-delivery-directory>", "settings": public_settings(config)}
        raise PipelineError("an explicit new delivery directory is required")
    delivery = delivery.resolve()
    if not delivery.is_dir():
        raise PipelineError("delivery directory does not exist")
    qa = (verifier or verify_delivery)(delivery)
    if qa.get("status") != "PASS":
        raise PipelineError("delivery verification failed")
    plan = read_json(delivery / "plan.json")
    hashes = {name: sha(delivery / name) for name in (PAYLOAD, NOTEBOOK, "plan.json")}
    # Reserve startup/shutdown margin beyond the bound experiment budget.
    budget = plan.get("budgets", plan.get("budget", {}))
    total = budget.get("total_seconds")
    if total is not None and config["timeout_seconds"] < total + 300:
        raise PipelineError("platform timeout must cover the plan budget plus 300 seconds")
    if dry_run:
        return {"status": "DRY_RUN", "delivery": str(delivery), "hashes": hashes,
                "remote_calls": False, "settings": public_settings(config)}
    run_id = config["experiment"] + "-" + canonical({"hashes": hashes, "settings": public_settings(config)})[:24]
    folder = Path(config["output_root"]) / run_id
    with run_lock(Path(config["output_root"])):
        return _prepare_snapshot(config, delivery, folder, run_id, hashes, plan, qa)


def _prepare_snapshot(config: dict, delivery: Path, folder: Path, run_id: str,
                      hashes: dict, plan: dict, qa: dict) -> dict:
    if (folder / "run.json").is_file():
        state = read_json(folder / "run.json")
        verify_staging(folder, state)
        if state["hashes"] != hashes or state["settings"] != public_settings(config):
            raise PipelineError("run identity collision")
        atomic_json(Path(config["output_root"]) / "latest.json", {"run_id": run_id})
        return state
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "dataset").mkdir()
    (folder / "kernel").mkdir()
    for name, dest in ((PAYLOAD, folder / "dataset" / PAYLOAD),
                       (NOTEBOOK, folder / "kernel" / NOTEBOOK),
                       ("plan.json", folder / "plan.json")):
        shutil.copyfile(delivery / name, dest)
        if sha(dest) != hashes[name]:
            raise PipelineError("delivery changed while preparing; do not submit")
    atomic_json(folder / "validation.json", qa)
    state = {
        "schema": "guandan-kaggle-pipeline-run-v1", "run_id": run_id,
        "status": "PREPARED", "created_at": now(), "delivery": str(delivery),
        "hashes": hashes, "payload_bytes": (folder / "dataset" / PAYLOAD).stat().st_size,
        "plan_hash": plan["plan_hash"], "settings": public_settings(config),
        "owner": None, "dataset_ref": None, "dataset_version": None,
        "kernel_ref": None, "kernel_version": None, "dataset_attempted": False,
        "kernel_attempted": False,
    }
    # Kaggle normalizes Notebook JSON on upload; reconcile by source semantics.
    from .backend import notebook_digest
    state["notebook_digest"] = notebook_digest(folder / "kernel" / NOTEBOOK)
    atomic_json(folder / "delivery.json", {
        "schema": "guandan-kaggle-run-delivery-v1",
        "run_id": run_id,
        "delivery": str(delivery),
        "hashes": hashes,
        "plan_hash": plan["plan_hash"],
        "payload_bytes": state["payload_bytes"],
    })
    save(folder, state)
    log = _log(folder, state)
    log.event("prepare", "validated", status="PREPARED", details={
        "delivery": str(delivery), "hashes": hashes,
        "payload_bytes": state["payload_bytes"], "plan_hash": plan["plan_hash"],
    })
    log.console("PREPARED", details={"run_id": run_id, "delivery": str(delivery)})
    atomic_json(Path(config["output_root"]) / "latest.json", {"run_id": run_id})
    return state


def public_settings(config: dict) -> dict:
    return {k: config[k] for k in ("accelerator", "timeout_seconds", "poll_seconds", "heartbeat_seconds",
                                  "dataset_wait_seconds", "enable_internet", "dataset_license")}


def verify_staging(folder: Path, state: dict) -> None:
    for name, target in ((PAYLOAD, folder / "dataset" / PAYLOAD),
                         (NOTEBOOK, folder / "kernel" / NOTEBOOK), ("plan.json", folder / "plan.json")):
        if not target.is_file() or target.is_symlink() or sha(target) != state["hashes"][name]:
            raise PipelineError("prepared input changed; prepare a new run")
    for child, allowed in (("dataset", {PAYLOAD, "dataset-metadata.json"}),
                           ("kernel", {NOTEBOOK, "kernel-metadata.json"})):
        if any(p.name not in allowed or not p.is_file() for p in (folder / child).iterdir()):
            raise PipelineError("unexpected file in upload staging")


def metadata(folder: Path, state: dict, username: str) -> None:
    if state.get("owner") not in (None, username):
        raise PipelineError("this run belongs to another Kaggle account")
    slug = state["run_id"]
    state.update(owner=username, dataset_ref=f"{username}/{slug}-input", kernel_ref=f"{username}/{slug}")
    atomic_json(folder / "dataset/dataset-metadata.json", {
        "id": state["dataset_ref"], "title": slug + " input",
        "licenses": [{"name": state["settings"]["dataset_license"]}],
    })
    atomic_json(folder / "kernel/kernel-metadata.json", {
        "id": state["kernel_ref"], "title": slug, "code_file": NOTEBOOK,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "machine_shape": state["settings"]["accelerator"],
        "enable_internet": False, "dataset_sources": [state["dataset_ref"]],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    })
    save(folder, state)


def backend_for(config: dict, state: dict):
    values = credentials(config)
    username = values["username"]
    if values["api_token"] and not username:
        username = check_connection(config)["username"]
    if state.get("owner") not in (None, username):
        raise PipelineError("this run belongs to another Kaggle account")
    from .backend import KaggleBackend
    return username, KaggleBackend(username, values["key"], api_token=values["api_token"])


def _dataset_matches(info: dict, state: dict) -> None:
    if info.get("is_private") is not True:
        raise PipelineError("remote Dataset privacy could not be verified")
    files = info.get("files", [])
    if len(files) != 1 or files[0].get("name") != PAYLOAD or files[0].get("bytes") != state["payload_bytes"]:
        raise PipelineError("remote Dataset file set differs from this run")
    if info.get("version") != 1:
        raise PipelineError("unique Dataset has an unexpected version")


def submit(config: dict, run_id: str | None = None, *, backend=None, username=None,
           sleep=time.sleep, emit=lambda value: None) -> dict:
    folder = locate(config, run_id)
    with run_lock(folder):
        state = read_json(folder / "run.json")
        log = _log(folder, state)
        operation = "validate_staging"
        try:
            verify_staging(folder, state)
            if backend is None:
                username, backend = backend_for(config, state)
            if not username:
                raise PipelineError("missing owner")
            metadata(folder, state, username)
            if state.get("kernel_version"):
                log.event("submit", "already_submitted", status=state["status"],
                          details={"kernel_ref": state.get("kernel_ref"), "version": state.get("kernel_version")})
                return state
            operation = "dataset_reconciliation"
            # Kaggle's dataset-detail endpoint returns 403 for non-existent
            # datasets (live-API verified 2026-10-25), so absence is probed
            # via the account listing first; the detail endpoint is only
            # queried for datasets that are known to exist.
            if backend.dataset_slug_exists(state["dataset_ref"]):
                info = backend.dataset_info(state["dataset_ref"])
                if not state["dataset_attempted"]:
                    raise PipelineError("Dataset slug collision; prepare a new run")
            else:
                info = None
            if info is None:
                if state["dataset_attempted"]:
                    raise PipelineError("previous Dataset upload outcome is unknown; retry later to reconcile, no automatic duplicate upload")
                state.update(status="DATASET_UPLOADING", dataset_attempted=True)
                save(folder, state)
                started = time.monotonic()
                log.event("dataset_upload", "intent_persisted", status=state["status"], details={
                    "dataset_ref": state["dataset_ref"], "bytes": state["payload_bytes"],
                    "payload_sha256": state["hashes"][PAYLOAD],
                })
                log.console("DATASET_UPLOADING", details={"bytes": state["payload_bytes"]})
                emit({"status": state["status"], "run_id": state["run_id"]})
                operation = "create_dataset"
                try:
                    response_dataset = backend.create_dataset(folder / "dataset")
                except Exception:
                    log.event("dataset_upload", "remote_outcome_unknown", status="UNKNOWN",
                              elapsed_seconds=time.monotonic() - started,
                              details={"dataset_ref": state["dataset_ref"]})
                    raise
                upload = {"schema": "guandan-kaggle-upload-v1", "dataset_ref": state["dataset_ref"],
                          "bytes": state["payload_bytes"], "payload_sha256": state["hashes"][PAYLOAD],
                          "elapsed_seconds": time.monotonic() - started, "result": response_dataset}
                atomic_json(folder / "records/dataset/upload.json", sanitize(upload))
                log.event("dataset_upload", "remote_call_returned", status="PROCESSING",
                          elapsed_seconds=upload["elapsed_seconds"], details=upload)
            deadline = time.monotonic() + state["settings"]["dataset_wait_seconds"]
            previous_dataset_status = None
            operation = "dataset_processing"
            while True:
                info = backend.dataset_info(state["dataset_ref"])
                dataset_status = None if info is None else info.get("status", "unknown")
                if dataset_status != previous_dataset_status:
                    row = {"time": now(), "status": dataset_status, "dataset_ref": state["dataset_ref"]}
                    _append_jsonl(folder / "records/dataset/processing.jsonl", row)
                    log.event("dataset_processing", "status_changed", status=dataset_status, details=row)
                    previous_dataset_status = dataset_status
                if info and info.get("ready"):
                    _dataset_matches(info, state)
                    break
                if info and info.get("status", "").lower() in ("error", "failed"):
                    raise PipelineError("Dataset processing failed; GPU job was not submitted")
                if time.monotonic() >= deadline:
                    raise PipelineError("Dataset processing timed out; rerun to resume polling")
                sleep(state["settings"]["poll_seconds"])
            state.update(status="DATASET_READY", dataset_version=info["version"])
            save(folder, state)
            log.console("DATASET_READY", details={"version": info["version"]})
            operation = "kernel_reconciliation"
            # Same 403-for-absent behaviour as the dataset detail endpoint:
            # probe the account listing before querying kernel details.
            remote = (
                backend.kernel_info(state["kernel_ref"])
                if backend.kernel_slug_exists(state["kernel_ref"])
                else None
            )
            if state["kernel_attempted"]:
                log.event("kernel_submission", "reconciliation_started", details={"kernel_ref": state["kernel_ref"]})
                if not remote:
                    raise PipelineError("previous GPU submission outcome is unknown; retry later, no automatic resubmission")
                if remote.get("version") != 1 or remote.get("code_sha256") != state["notebook_digest"]:
                    raise PipelineError("ambiguous remote kernel identity; no automatic resubmission")
                if remote.get("is_private") is not True:
                    raise PipelineError("remote Notebook privacy could not be verified")
                if remote.get("dataset_sources") not in ([state["dataset_ref"]], [state["dataset_ref"] + "/1"]):
                    raise PipelineError("remote Notebook inputs do not match this experiment")
                response = remote
                log.event("kernel_submission", "reconciled", status=remote.get("status"), details={
                    "kernel_ref": state["kernel_ref"], "version": remote["version"],
                })
            else:
                if remote is not None:
                    raise PipelineError("Notebook slug collision; prepare a new run")
                state.update(status="SUBMITTING", kernel_attempted=True)
                save(folder, state)
                log.event("kernel_submission", "intent_persisted", status="SUBMITTING",
                          details={"kernel_ref": state["kernel_ref"], "dataset_ref": state["dataset_ref"]})
                operation = "push_kernel"
                try:
                    response = backend.push_kernel(folder / "kernel", state["settings"]["accelerator"], state["settings"]["timeout_seconds"])
                except Exception:
                    log.event("kernel_submission", "remote_outcome_unknown", status="UNKNOWN",
                              details={"kernel_ref": state["kernel_ref"]})
                    raise
            if type(response.get("version")) is not int or response["version"] != 1:
                raise PipelineError("unexpected Notebook version; submission needs reconciliation")
            state.update(status="SUBMITTED", kernel_version=response["version"],
                         url=f"https://www.kaggle.com/code/{state['kernel_ref']}?scriptVersionId={response.get('version_id', '')}" if response.get("version_id") else f"https://www.kaggle.com/code/{state['kernel_ref']}/{response['version']}",
                         submitted_at=now())
            save(folder, state)
            submission = {"schema": "guandan-kaggle-submission-v1", "kernel_ref": state["kernel_ref"],
                          "kernel_version": state["kernel_version"], "url": state["url"],
                          "dataset_ref": state["dataset_ref"], "submitted_at": state["submitted_at"]}
            atomic_json(folder / "records/kernel/submission.json", submission)
            log.event("kernel_submission", "remote_call_returned", status="SUBMITTED", details=submission)
            log.console("SUBMITTED", details={"version": state["kernel_version"], "url": state["url"]})
            emit({"status": "SUBMITTED", "run_id": state["run_id"], "url": state["url"]})
            return state
        except Exception as error:
            outcome = "outcome_unknown" if operation in {"create_dataset", "push_kernel"} else "definite_failure"
            log.failure("submit", operation, error, outcome, True,
                        f".\\scripts\\upload_kaggle.ps1 -Watch -RunId {state['run_id']}")
            log.console("ERROR", details={"operation": operation, "outcome": outcome})
            raise


def status(config: dict, run_id: str | None = None, *, backend=None) -> dict:
    folder = locate(config, run_id)
    with run_lock(folder):
        state = read_json(folder / "run.json")
        if not state.get("kernel_version") or state["status"] in ("VERIFIED", "FETCHED_INCOMPLETE"):
            return state
        if backend is None:
            _, backend = backend_for(config, state)
        remote = backend.kernel_status(state["kernel_ref"], state["kernel_version"])
        previous_remote = state.get("remote_status")
        state.update(status="REMOTE_" + remote.upper(), remote_status=remote)
        save(folder, state)
        if remote != previous_remote:
            log = _log(folder, state)
            row = {"time": now(), "status": remote, "kernel_ref": state["kernel_ref"],
                   "kernel_version": state["kernel_version"]}
            _append_jsonl(folder / "records/kernel/status.jsonl", row)
            log.event("kernel_status", "status_changed", status=remote, details=row)
            log.console(state["status"], details={"version": state["kernel_version"]})
        return state


def fetch(config: dict, run_id: str | None = None, *, backend=None) -> dict:
    folder = locate(config, run_id)
    with run_lock(folder):
        state = read_json(folder / "run.json")
        if not state.get("kernel_version"):
            raise PipelineError("this run has not been submitted")
        if backend is None:
            _, backend = backend_for(config, state)
        remote = backend.kernel_status(state["kernel_ref"], state["kernel_version"])
        if remote not in ("complete", "error", "cancelled"):
            raise PipelineError("remote output is not final yet; use watch")
        output = folder / ("outputs-" + uuid.uuid4().hex[:8])
        output.mkdir()
        log = _log(folder, state)
        log.event("download", "started", details={"kernel_ref": state["kernel_ref"],
                                                   "version": state["kernel_version"]})
        try:
            backend.download_outputs(state["kernel_ref"], state["kernel_version"], output)
        except Exception as error:
            log.failure("download", "download_outputs", error, "definite_failure", True,
                        f".\\scripts\\upload_kaggle.ps1 -Watch -RunId {state['run_id']}")
            raise
        try:
            verification = verify_results(output, state["plan_hash"])
        except (ResultError, OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile) as error:
            verification = {"status": "FETCHED_INCOMPLETE", "integrity": "FAIL", "reason": type(error).__name__}
        if remote != "complete":
            verification["status"] = "FETCHED_INCOMPLETE"
        state.update(status=verification["status"], remote_status=remote, output_dir=str(output), verification=verification)
        atomic_json(output / "PIPELINE_VERIFICATION.json", verification)
        save(folder, state)
        log.event("verification", "completed", status=verification["status"], details=verification)
        log.console(verification["status"], details={"output": str(output),
                                                      "verdict": verification.get("verdict")})
        return state


def watch(config: dict, run_id: str | None = None, *, backend=None,
          sleep=time.sleep, emit=lambda value: None) -> dict:
    folder = locate(config, run_id)
    run_id = folder.name
    if backend is None:
        _, backend = backend_for(config, read_json(folder / "run.json"))
    previous = None
    last_heartbeat = time.monotonic()
    while True:
        state = status(config, run_id, backend=backend)
        if state["status"] != previous:
            emit({"status": state["status"], "run_id": run_id})
            previous = state["status"]
            last_heartbeat = time.monotonic()
        elif time.monotonic() - last_heartbeat >= float(state["settings"].get("heartbeat_seconds", 300)):
            log = _log(folder, state)
            log.event("watch", "heartbeat", status=state["status"], details={
                "kernel_ref": state.get("kernel_ref"), "version": state.get("kernel_version"),
            })
            log.console(state["status"], "heartbeat")
            last_heartbeat = time.monotonic()
        if state["status"] in ("VERIFIED", "FETCHED_INCOMPLETE"):
            return state
        if not state.get("kernel_version"):
            raise PipelineError("run has no submitted version; use run to submit or reconcile")
        if state.get("remote_status") in ("complete", "error", "cancelled"):
            return fetch(config, run_id, backend=backend)
        sleep(state["settings"]["poll_seconds"])


def build_delivery(config: dict) -> Path:
    values = config.get("build")
    if not isinstance(values, dict) or not all(values.get(k) for k in ("output", "model_dir", "base_inputs")):
        raise PipelineError("build requires explicit output, model_dir and base_inputs under config.build; existing deliveries are never guessed")
    output = (ROOT / values["output"]).resolve()
    # The existing builder loads policy tensors; keep Torch in the training env.
    python = ROOT / ".venv-gpu" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise PipelineError("existing model build environment .venv-gpu is missing")
    if output.exists() or output.with_name(output.name + "_build").exists():
        raise PipelineError("build output exists; use --delivery for that new package, never overwrite it")
    command = [str(python), str(ROOT / "scripts/build_t4x2_serial_compare.py"), "--output", str(output),
               "--model-dir", str(ROOT / values["model_dir"]), "--base-inputs", str(ROOT / values["base_inputs"])]
    if values.get("validation_record"):
        command += ["--validation-record", str(ROOT / values["validation_record"])]
    logs = Path(config["output_root"]) / ("build-" + uuid.uuid4().hex[:12])
    logs.mkdir(parents=True)
    with (logs / "stdout.log").open("w", encoding="utf-8") as stdout, (logs / "stderr.log").open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise PipelineError(f"delivery build failed; inspect local build logs in {logs}")
    return output
