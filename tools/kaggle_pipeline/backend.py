"""Bounded, lazy adapter for kaggle==2.2.5. No CLI or account-file access.

Environment/stdio changes are serialized between adapter instances. Callers must
not concurrently manipulate process environment or use the SDK outside this adapter.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import threading


_SDK_LOCK = threading.RLock()
_REF = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+\Z")


class BackendError(RuntimeError):
    """A safe operation name and optional HTTP status, never a server message."""

    def __init__(self, operation: str, status_code: int | None = None):
        allowed = {"initialize", "dataset_info", "create_dataset", "kernel_info",
                   "push_kernel", "kernel_status", "download_outputs", "check_connection"}
        self.operation = operation if operation in allowed else "initialize"
        self.status_code = status_code if type(status_code) is int and 100 <= status_code <= 599 else None
        suffix = f" (HTTP {self.status_code})" if self.status_code is not None else ""
        super().__init__(f"Kaggle {self.operation} failed{suffix}")


def _status_code(exc) -> int | None:
    value = getattr(getattr(exc, "response", None), "status_code", None)
    if value is None:
        value = getattr(exc, "status_code", getattr(exc, "status", None))
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    return value if type(value) is int and 100 <= value <= 599 else None


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _token(value) -> str:
    value = getattr(value, "name", value)
    return value.strip().lower().rsplit(".", 1)[-1] if isinstance(value, str) else "unknown"


def _kernel_state(value) -> str:
    state = _token(_field(value, "status", value))
    return {
        "queued": "queued", "pending": "queued", "new_script": "queued",
        "running": "running", "complete": "complete", "completed": "complete",
        "error": "error", "failed": "error", "cancelled": "cancelled",
        "canceled": "cancelled", "cancel_requested": "running",
        "cancel_acknowledged": "cancelled",
    }.get(state, "unknown")


def _version(value) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("Invalid version")
    return value


def _ref(value: str) -> tuple[str, str]:
    if not isinstance(value, str) or not _REF.fullmatch(value):
        raise ValueError("Invalid reference")
    return tuple(value.split("/"))


def _request(service: str, name: str, **fields):
    # kagglesdk itself does not have kaggle's package-level auto-authentication.
    module_name = {"datasets": "dataset_api_service", "kernels": "kernels_api_service"}[service]
    module = importlib.import_module(f"kagglesdk.{service}.types.{module_name}")
    request = getattr(module, name)()
    for key, value in fields.items():
        setattr(request, key, value)
    return request


def notebook_digest(path: Path) -> str:
    """Hash notebook meaning across SDK push/pull JSON normalization.

    Preserve code, Markdown, cell order/IDs, and non-execution metadata. Ignore
    outputs, execution counts and known execution-only metadata at either level.
    """
    notebook = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("Invalid notebook")

    def clean_metadata(container):
        metadata = container.get("metadata")
        if isinstance(metadata, dict):
            for field in ("execution", "ExecuteTime", "executionInfo", "widgets", "trusted"):
                metadata.pop(field, None)

    clean_metadata(notebook)
    for cell in notebook["cells"]:
        if not isinstance(cell, dict):
            raise ValueError("Invalid notebook cell")
        source = cell.get("source", "")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError("Invalid notebook source")
        cell["source"] = source
        clean_metadata(cell)
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    encoded = json.dumps(notebook, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _no_access_token():
    return None, None


@contextmanager
def _without_ambient_tokens(*, load_sdk: bool):
    # Importing kagglesdk defines helpers but does not authenticate. Patch it
    # BEFORE importing kaggle, whose package __init__ authenticates immediately.
    if load_sdk:
        importlib.import_module("kagglesdk")
    names = ("kagglesdk.kaggle_env", "kagglesdk", "kagglesdk.kaggle_http_client",
             "kaggle.api.kaggle_api_extended")
    saved = {}
    package = sys.modules.get("kagglesdk")
    package_reader = getattr(package, "get_access_token_from_env", None)
    try:
        for name in names:
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "get_access_token_from_env"):
                saved[name] = module, module.get_access_token_from_env
                module.get_access_token_from_env = _no_access_token
        yield
    finally:
        for name in names:
            if name in saved:
                module, reader = saved[name]
                module.get_access_token_from_env = reader
            else:
                # A first kaggle import copies the temporarily disabled helper
                # from kagglesdk. Restore that NEW alias to its original reader.
                module = sys.modules.get(name)
                if (module is not None and package_reader is not None
                        and getattr(module, "get_access_token_from_env", None) is _no_access_token):
                    module.get_access_token_from_env = package_reader


def _new_api(username: str, key: str, api_token: str | None = None):
    # The FIRST import runs kaggle.api.authenticate(). The surrounding scope
    # disables default-file/env token helpers and supplies an empty config dir.
    # Token mode must not let package-level auto-auth try OAuth/home credentials.
    # Bootstrap that disposable package singleton with synthetic legacy values;
    # the actual adapter below uses ONLY its explicit bearer configuration.
    if api_token is not None:
        bootstrap = {name: os.environ.get(name) for name in ("KAGGLE_USERNAME", "KAGGLE_KEY")}
        try:
            os.environ.update(KAGGLE_USERNAME="adapter-bootstrap", KAGGLE_KEY="adapter-bootstrap")
            sdk = importlib.import_module("kaggle.api.kaggle_api_extended")
        finally:
            for name, value in bootstrap.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    else:
        sdk = importlib.import_module("kaggle.api.kaggle_api_extended")

    class ExplicitApi(sdk.KaggleApi):
        MAX_UPLOAD_RESUME_ATTEMPTS = 1

        def _load_config(self):
            if api_token is None:
                self.config_values = {"username": username, "key": key}
            else:
                self.config_values = {"token": api_token, "username": username,
                                      "auth_method": str(sdk.AuthMethod.ACCESS_TOKEN)}

        def authenticate(self):
            if api_token is None:
                return super().authenticate()
            # Ambient-token discovery and legacy/OAuth fallback are forbidden.
            self._load_config()
            self._authenticated = True

        def with_retry(self, func, *args, **kwargs):
            return func

        def dataset_status(self, *args, **kwargs):
            # SDK create uses this probe to detect an existing dataset by
            # catching HTTPError as absence.  Live-API verified 2026-10-25:
            # Kaggle answers 404 *or 403* here for a dataset this token
            # cannot see — including one that does not exist.  Both must pass
            # through as the original HTTPError so the SDK proceeds to the
            # create request; any other status still fails closed as a
            # BackendError the SDK cannot swallow.
            try:
                return super().dataset_status(*args, **kwargs)
            except Exception as exc:
                if _status_code(exc) in (403, 404):
                    raise
                raise BackendError("create_dataset", _status_code(exc)) from None

        def upload_complete(self, path, url, quiet, resume=False):
            # SDK implementation retries PUT ten times and swallows exceptions.
            # One streaming PUT, no resume/retry, and no swallowed failures.
            if resume:
                raise ValueError("Upload resume disabled")
            with open(path, "rb") as source, sdk.requests.Session() as session:
                response = session.put(url, data=source, timeout=300, allow_redirects=False)
                response.raise_for_status()
                if not self._is_upload_successful(response):
                    raise ValueError("Incomplete upload")
            return sdk.ResumableUploadResult.COMPLETE

    api = ExplicitApi()
    api.args = []  # Do not inherit SDK class-level CLI/verbose settings.
    api.authenticate()
    return api


class KaggleBackend:
    def __init__(self, username: str, key: str, *, api=None, api_token: str | None = None):
        if not isinstance(username, str) or not isinstance(key, str):
            raise BackendError("initialize")
        if api_token is None:
            if not username.strip() or not key.strip():
                raise BackendError("initialize")
        elif (not isinstance(api_token, str) or not api_token.startswith("KGAT_")
              or not api_token[5:] or any(char.isspace() for char in api_token) or key
              or (username and not re.fullmatch(r"[A-Za-z0-9_-]+", username))):
            raise BackendError("initialize")
        self._username = username
        self._expected_username = username
        self._key = key
        self._api_token = api_token
        self._api = api

    @contextmanager
    def _operation(self, operation):
        failure = None
        # Do not retain SDK output in memory: it may contain credentials and can
        # be arbitrarily large. Restore even when authentication calls sys.exit.
        try:
            with _SDK_LOCK, tempfile.TemporaryDirectory(prefix="kaggle-adapter-") as config_dir:
                saved = {k: v for k, v in os.environ.items() if k.startswith("KAGGLE_") or k in ("VERBOSE", "VERBOSE_OUTPUT")}
                log_level = logging.root.manager.disable
                try:
                    for name in saved:
                        os.environ.pop(name, None)
                    os.environ["KAGGLE_CONFIG_DIR"] = config_dir
                    if self._api_token is None:
                        os.environ.update(KAGGLE_USERNAME=self._username, KAGGLE_KEY=self._key)
                    logging.disable(logging.CRITICAL)
                    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
                        with _without_ambient_tokens(load_sdk=self._api is None):
                            if self._api is None:
                                self._api = (_new_api(self._username, self._key) if self._api_token is None
                                             else _new_api(self._username, self._key, self._api_token))
                            if (self._api_token is not None and not self._username
                                    and operation not in {"check_connection", "initialize"}):
                                self._resolve_username(self._api)
                            yield self._api
                finally:
                    for name in list(os.environ):
                        if name.startswith("KAGGLE_") or name in ("VERBOSE", "VERBOSE_OUTPUT"):
                            os.environ.pop(name, None)
                    os.environ.update(saved)
                    logging.disable(log_level)
        except (Exception, SystemExit) as exc:
            failure = BackendError(operation, exc.status_code if isinstance(exc, BackendError) else _status_code(exc))
        # contextlib.__exit__ still handles the thrown exception here; clear
        # its implicit context after raise attaches it, before propagation.
        if failure is not None:
            try:
                raise failure from None
            finally:
                failure.__context__ = None

    def _resolve_username(self, api) -> str:
        if self._api_token is not None:
            username = api._introspect_token(self._api_token)
            if (not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", username)
                    or (self._expected_username and username != self._expected_username)):
                raise ValueError("Invalid token identity")
            api.config_values["username"] = username
            self._username = username
        return self._username

    def resolve_username(self) -> str:
        """Resolve and verify token identity without credential discovery/fallback."""
        with self._operation("check_connection") as api:
            return self._resolve_username(api)

    def check_connection(self) -> dict:
        """Check identity and one page of each read service; never submit work."""
        with self._operation("check_connection") as api:
            username = self._resolve_username(api)
            # dataset_list in SDK 2.2.5 has no page_size keyword.
            response = api.dataset_list_with_response(user=username, page_size=1)
            datasets = _field(response, "datasets")
            kernels = api.kernels_list(user=username, page_size=1)
            if not isinstance(datasets, list) or not isinstance(kernels, list):
                raise ValueError("Invalid read listing")
            return {"authenticated": True, "username": username,
                    "auth_method": "api_token" if self._api_token is not None else "legacy",
                    "dataset_read": True, "datasets_visible": len(datasets),
                    "kernel_read": True, "kernels_visible": len(kernels)}

    def dataset_slug_exists(self, ref: str) -> bool:
        """Definitive membership probe for one of this account's datasets.

        Kaggle's dataset-detail endpoint answers HTTP 403 both for foreign and
        for *non-existent* datasets (verified against the live API on
        2026-10-25), so absence cannot be distinguished from an authorization
        failure there.  The account listing includes private datasets and
        answers membership definitively; it is the correct existence probe
        before ``create_dataset``.
        """
        with self._operation("dataset_info") as api:
            owner, _slug = _ref(ref)
            if owner != self._username:
                return False
            page_token: str | None = None
            seen: set[str] = set()
            while True:
                response = api.dataset_list_with_response(
                    user=self._username, page_size=100, page_token=page_token,
                )
                for item in _field(response, "datasets") or []:
                    if _field(item, "ref") == ref or _field(item, "id") == ref:
                        return True
                page_token = _field(response, "next_page_token")
                if not page_token or (page_token in seen):
                    return False
                seen.add(page_token)

    def kernel_slug_exists(self, ref: str) -> bool:
        """Membership probe for one of this account's kernels (same rationale).

        The listing is single-page (accounts with >100 kernels must extend
        pagination before a false absence could trigger a fail-closed create
        collision, never a silent duplicate).
        """
        with self._operation("kernel_info") as api:
            owner, _slug = _ref(ref)
            if owner != self._username:
                return False
            kernels = api.kernels_list(user=self._username, page_size=100)
            if not isinstance(kernels, list):
                raise ValueError("Invalid kernel listing")
            return any(
                _field(item, "ref") == ref or _field(item, "id") == ref
                for item in kernels
            )

    def dataset_info(self, ref: str) -> dict | None:
        with self._operation("dataset_info") as api:
            owner, slug = _ref(ref)
            request = _request("datasets", "ApiGetDatasetRequest", owner_slug=owner, dataset_slug=slug)
            with api.build_kaggle_client() as client:
                try:
                    detail = client.datasets.dataset_api_client.get_dataset(request)
                except Exception as exc:
                    if _status_code(exc) == 404:
                        return None
                    raise
            if detail is None:
                raise ValueError("Missing dataset detail")
            state = _token(api.dataset_status(ref))
            # Fixed vocabulary prevents arbitrary response text entering logs.
            state = state if state in {"ready", "failed", "deleted", "reprocessing", "not_yet_persisted", "blobs_received", "blobs_decompressed", "blobs_copied_to_sds", "individual_blobs_compressed"} else "unknown"
            version = _field(detail, "current_version_number")
            version = _version(version) if version else None
            files, seen, page = [], set(), None
            # Processing datasets may have no listable files/version yet.
            # Preserve known detail fields and let the orchestrator wait.
            while state == "ready":
                response = api.dataset_list_files(f"{ref}/{version}" if version else ref, page_token=page, page_size=100)
                if response is None or _field(response, "error_message"):
                    raise ValueError("Invalid file listing")
                for item in _field(response, "files", []) or []:
                    name, size = _field(item, "name"), _field(item, "total_bytes")
                    if not isinstance(name, str) or type(size) is not int or size < 0:
                        raise ValueError("Invalid file entry")
                    files.append({"name": name, "bytes": size})
                page = _field(response, "next_page_token")
                if not page:
                    break
                if not isinstance(page, str) or page in seen:
                    raise ValueError("Invalid pagination")
                seen.add(page)
            private = _field(detail, "is_private")
            return {"status": state, "ready": state == "ready", "version": version,
                    "files": files, "is_private": private if type(private) is bool else None}

    @staticmethod
    def _purge_resumable_upload_state(folder: Path) -> None:
        """Remove kaggle 2.2.5 resumable-upload state for this folder's files.

        The SDK persists upload state under the per-user temp directory and
        deliberately keeps it after a failure, then requests ``resume=True`` on
        the next attempt.  This adapter's ``upload_complete`` refuses resume
        (every create is a deliberate full upload), so stale state from a
        failed attempt would otherwise abort every retry with "Upload resume
        disabled".  State file names mirror the upload path exactly as the
        SDK derives them, for both the relative and absolute spelling.
        """
        state_dir = os.path.join(tempfile.gettempdir(), ".kaggle-uploads")
        roots = dict.fromkeys([Path(folder), Path(folder).resolve()])
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                for spelling in (str(path), str(path.resolve())):
                    name = "%s.json" % spelling.replace(os.path.sep, "_").replace(":", "_")
                    try:
                        os.unlink(os.path.join(state_dir, name))
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        raise BackendError("create_dataset") from None

    def create_dataset(self, folder: Path) -> dict:
        with self._operation("create_dataset") as api:
            folder = Path(folder).resolve()
            self._purge_resumable_upload_state(folder)
            metadata = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
            ref = metadata["id"]
            _ref(ref)
            response = api.dataset_create_new(str(folder), public=False, quiet=True,
                                              convert_to_csv=False, dir_mode="zip")
            if response is None or _field(response, "error") or _token(_field(response, "status")) in {"error", "failed"}:
                raise ValueError("Dataset creation failed")
            return {"ref": ref, "url": f"https://www.kaggle.com/datasets/{ref}", "is_private": True}

    def kernel_info(self, ref: str) -> dict | None:
        with self._operation("kernel_info") as api:
            owner, slug = _ref(ref)
            # kernels_pull drops current_version_number from the saved metadata;
            # use the exact same official get_kernel endpoint without local files.
            request = _request("kernels", "ApiGetKernelRequest", user_name=owner, kernel_slug=slug)
            with api.build_kaggle_client() as client:
                try:
                    detail = client.kernels.kernels_api_client.get_kernel(request)
                except Exception as exc:
                    if _status_code(exc) == 404:
                        return None
                    raise
            metadata = _field(detail, "metadata")
            version = _version(_field(metadata, "current_version_number"))
            result = {"version": version, "status": _kernel_state(api.kernels_status(f"{ref}/{version}"))}
            private = _field(metadata, "is_private")
            result["is_private"] = private if type(private) is bool else None
            sources = _field(metadata, "dataset_data_sources")
            if isinstance(sources, list) and all(isinstance(source, str) for source in sources):
                result["dataset_sources"] = list(sources)
            shape = _field(metadata, "machine_shape")
            if isinstance(shape, str):
                result["machine_shape"] = shape
            with tempfile.TemporaryDirectory(prefix="kaggle-code-") as directory:
                code_path = Path(api.kernels_pull(f"{ref}/{version}", path=directory, metadata=False, quiet=True))
                if not code_path.resolve().is_relative_to(Path(directory).resolve()):
                    raise ValueError("Invalid pulled code path")
                result["code_sha256"] = notebook_digest(code_path)
            return result

    def push_kernel(self, folder: Path, accelerator: str, timeout_seconds: int) -> dict:
        with self._operation("push_kernel") as api:
            if not isinstance(accelerator, str) or not accelerator.strip():
                raise ValueError("Invalid accelerator")
            _version(timeout_seconds)
            metadata = json.loads((Path(folder) / "kernel-metadata.json").read_text(encoding="utf-8"))
            ref = metadata["id"]
            _ref(ref)
            if metadata.get("is_private", True) not in (True, "true"):
                raise ValueError("Kernel must be private")
            response = api.kernels_push(str(folder), timeout=str(timeout_seconds), acc=accelerator)
            if response is None or _field(response, "error") or any(
                _field(response, name) for name in ("invalid_tags", "invalid_dataset_sources",
                    "invalid_competition_sources", "invalid_kernel_sources", "invalid_model_sources")
            ):
                raise ValueError("Kernel submission failed")
            return {"version": _version(_field(response, "version_number")),
                    "url": f"https://www.kaggle.com/code/{ref}"}

    def kernel_status(self, ref: str, version: int) -> str:
        with self._operation("kernel_status") as api:
            _ref(ref)
            return _kernel_state(api.kernels_status(f"{ref}/{_version(version)}"))

    def download_outputs(self, ref: str, version: int, folder: Path) -> None:
        with self._operation("download_outputs") as api:
            _ref(ref)
            pinned = f"{ref}/{_version(version)}"
            Path(folder).mkdir(parents=True, exist_ok=True)
            # SDK 2.2.5 parses owner/slug/N and sends version_label='vN',
            # including every recursive output page. Never fall back to latest.
            api.kernels_output(pinned, path=str(folder), force=True, quiet=True)
