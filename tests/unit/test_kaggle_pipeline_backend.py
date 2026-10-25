"""Offline adapter contracts. Fake services only; no account or Kaggle requests."""
from __future__ import annotations

from contextlib import nullcontext
import builtins
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from types import SimpleNamespace as NS

import pytest

from tools.kaggle_pipeline import backend as b


REF = "test-owner/unique-experiment"
KEY = "synthetic-secret-key"
TOKEN = "KGAT_synthetic-offline-token"
NOTEBOOK = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
    {"cell_type": "code", "metadata": {}, "source": ["print('测试')\n"],
     "outputs": [], "execution_count": None},
]}


class HttpFailure(RuntimeError):
    def __init__(self, code):
        super().__init__(f"untrusted body contains {KEY}")
        self.response = NS(status_code=code)


class FakeApi:
    def __init__(self):
        self.calls = []
        self.errors = {}
        self.dataset = NS(current_version_number=1, is_private=True)
        self.dataset_state = "ready"
        self.kernel = NS(metadata=NS(current_version_number=1, is_private=True,
            dataset_data_sources=[REF], machine_shape="NvidiaTeslaT4x2"))
        self.state = "complete"
        self.pages = {None: NS(files=[NS(name="payload.zip", total_bytes=42)], next_page_token=None)}
        self.create_response = NS(error=None, status="ok")
        self.push_response = NS(error=None, version_number=1)

    def record(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        # Deliberately noisy, including lazy reads of the supplied key.
        print(os.environ.get("KAGGLE_KEY"))
        print(KEY, file=sys.stderr)
        logging.getLogger("kaggle.fake").warning(KEY)
        if method in self.errors:
            raise self.errors[method]

    def build_kaggle_client(self):
        self.record("build_kaggle_client")
        return nullcontext(NS(datasets=NS(dataset_api_client=NS(get_dataset=self.get_dataset)),
                              kernels=NS(kernels_api_client=NS(get_kernel=self.get_kernel))))

    def get_dataset(self, request):
        self.record("get_dataset", request)
        return self.dataset

    def get_kernel(self, request):
        self.record("get_kernel", request)
        return self.kernel

    def dataset_status(self, dataset):
        self.record("dataset_status", dataset)
        return self.dataset_state

    def dataset_list_files(self, dataset, page_token=None, page_size=20):
        self.record("dataset_list_files", dataset, page_token=page_token, page_size=page_size)
        return self.pages[page_token]

    def dataset_create_new(self, folder, **kwargs):
        self.record("dataset_create_new", folder, **kwargs)
        return self.create_response

    def kernels_push(self, folder, timeout=None, acc=None, no_run=False):
        self.record("kernels_push", folder, timeout=timeout, acc=acc, no_run=no_run)
        return self.push_response

    def kernels_pull(self, kernel, path, metadata=False, quiet=True):
        self.record("kernels_pull", kernel, path=path, metadata=metadata, quiet=quiet)
        code = Path(path) / "original.ipynb"
        code.write_bytes(json.dumps(NOTEBOOK).encode("utf-8"))
        return str(code)

    def kernels_status(self, kernel):
        self.record("kernels_status", kernel)
        return NS(status=self.state)

    def kernels_output(self, kernel, path, **kwargs):
        self.record("kernels_output", kernel, path=path, **kwargs)
        Path(path, "result.json").write_text("{}", encoding="utf-8")
        return [str(Path(path, "result.json"))], None


class FakeTokenApi(FakeApi):
    def __init__(self):
        super().__init__()
        self.config_values = {"token": TOKEN, "username": "", "auth_method": "ACCESS_TOKEN"}
        self.identity = "explicit-user"
        self.datasets = [NS(untrusted=TOKEN)]
        self.kernels = []

    def _introspect_token(self, token):
        self.record("introspect", token)
        return self.identity

    def dataset_list_with_response(self, **kwargs):
        self.record("dataset_list", **kwargs)
        return NS(datasets=self.datasets)

    def kernels_list(self, **kwargs):
        self.record("kernels_list", **kwargs)
        return self.kernels


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Network is forbidden in adapter tests")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    # Unit fakes need neither kaggle nor kagglesdk installed.
    monkeypatch.setattr(b, "_request", lambda service, name, **fields: NS(**fields))
    api = FakeApi()
    folder = tmp_path / "staging"
    folder.mkdir()
    (folder / "dataset-metadata.json").write_text(json.dumps({"id": REF, "is_private": False}))
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": REF, "is_private": True}))
    return b.KaggleBackend("explicit-user", KEY, api=api), api, folder


def test_import_and_constructor_are_lazy():
    # Separate interpreter gives an actual clean import without authenticating.
    script = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('kaggle', 'kagglesdk'):
        raise AssertionError('eager SDK import')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools.kaggle_pipeline.backend import KaggleBackend
KaggleBackend('fake-user', 'fake-key')
KaggleBackend('', '', api_token='KGAT_synthetic-offline-token')
assert 'kaggle' not in sys.modules
assert 'kagglesdk' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == completed.stderr == ""


@pytest.mark.parametrize("username,key,token", [
    ("", "", ""), ("", "", "KGAT_"), ("", "", "legacy-token"),
    ("", "", " KGAT_value"), ("", "", "KGAT_value\n"), ("", "", 123),
    ("", KEY, TOKEN), ("explicit-user", KEY, TOKEN), ("", " ", TOKEN),
    ("bad/owner", "", TOKEN), (None, "", TOKEN), ("", None, TOKEN),
    ("", "", None), ("explicit-user", "", None),
])
def test_invalid_and_mixed_credentials_are_rejected(username, key, token):
    with pytest.raises(b.BackendError, match="^Kaggle initialize failed$"):
        b.KaggleBackend(username, key, api_token=token)


@pytest.mark.parametrize("expected_owner", ["", "explicit-user"])
def test_connection_checks_identity_and_bounded_reads(expected_owner):
    api = FakeTokenApi()
    backend = b.KaggleBackend(expected_owner, "", api_token=TOKEN, api=api)
    result = backend.check_connection()
    assert result == {"authenticated": True, "username": "explicit-user", "auth_method": "api_token",
                      "dataset_read": True, "datasets_visible": 1, "kernel_read": True, "kernels_visible": 0}
    assert api.calls == [("introspect", (TOKEN,), {}),
                         ("dataset_list", (), {"user": "explicit-user", "page_size": 1}),
                         ("kernels_list", (), {"user": "explicit-user", "page_size": 1})]
    assert api.config_values["username"] == backend._username == "explicit-user"
    assert TOKEN not in json.dumps(result)
    backend.kernel_status(REF, 1)
    assert len([call for call in api.calls if call[0] == "introspect"]) == 1


def test_legacy_connection_check_uses_only_bounded_reads():
    api = FakeTokenApi()
    backend = b.KaggleBackend("explicit-user", KEY, api=api)
    assert backend.check_connection()["auth_method"] == "legacy"
    assert [call[0] for call in api.calls] == ["dataset_list", "kernels_list"]


def test_token_operations_resolve_missing_owner_and_reuse_cached_owner():
    api = FakeTokenApi()
    backend = b.KaggleBackend("", "", api_token=TOKEN, api=api)
    assert backend.kernel_status(REF, 1) == "complete"
    assert backend._username == "explicit-user"
    assert backend.kernel_status(REF, 1) == "complete"
    assert [call[0] for call in api.calls] == ["introspect", "kernels_status", "kernels_status"]
    api.calls.clear()
    cached = b.KaggleBackend("explicit-user", "", api_token=TOKEN, api=api)
    assert cached.kernel_status(REF, 1) == "complete"
    assert [call[0] for call in api.calls] == ["kernels_status"]


@pytest.mark.parametrize("identity", [None, "", "other-owner", "unsafe/owner", TOKEN + "\n"])
def test_resolve_username_rejects_invalid_identity_and_expected_owner_mismatch(identity):
    api = FakeTokenApi()
    api.identity = identity
    backend = b.KaggleBackend("explicit-user", "", api_token=TOKEN, api=api)
    with pytest.raises(b.BackendError, match="^Kaggle check_connection failed$") as caught:
        backend.resolve_username()
    assert [call[0] for call in api.calls] == ["introspect"]
    assert backend._username == "explicit-user"
    assert KEY not in "".join(traceback.format_exception(caught.value))
    assert TOKEN not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__context__ is None


@pytest.mark.parametrize("failure", [None, "introspect", "dataset_list", "kernels_list"])
def test_token_connection_restores_environment_helpers_and_redacts_failures(monkeypatch, capsys, caplog, failure):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "ambient-synthetic-token")
    monkeypatch.setenv("KAGGLE_USERNAME", "ambient-user")
    monkeypatch.setenv("KAGGLE_KEY", "ambient-key")
    before, disabled = dict(os.environ), logging.root.manager.disable
    api = FakeTokenApi()
    original = api.record
    def record(*args, **kwargs):
        assert not {"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"} & set(os.environ)
        assert Path(os.environ["KAGGLE_CONFIG_DIR"]).is_dir()
        print(TOKEN)
        return original(*args, **kwargs)
    api.record = record
    if failure:
        api.errors[failure] = HttpFailure(401)
    backend = b.KaggleBackend("", "", api_token=TOKEN, api=api)
    if failure:
        with pytest.raises(b.BackendError, match="^Kaggle check_connection failed \\(HTTP 401\\)$") as caught:
            backend.check_connection()
        assert KEY not in "".join(traceback.format_exception(caught.value))
        assert caught.value.__context__ is None
    else:
        assert backend.check_connection()["authenticated"] is True
    assert dict(os.environ) == before
    assert logging.root.manager.disable == disabled
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert TOKEN not in caplog.text and KEY not in caplog.text


@pytest.mark.parametrize("field", ["datasets", "kernels"])
def test_connection_does_not_treat_missing_listings_as_success(field):
    api = FakeTokenApi()
    setattr(api, field, None)
    with pytest.raises(b.BackendError, match="^Kaggle check_connection failed$"):
        b.KaggleBackend("", "", api_token=TOKEN, api=api).check_connection()


def test_dataset_info_exact_files_privacy_and_pagination(harness):
    backend, api, _ = harness
    api.pages[None].next_page_token = "next"
    api.pages["next"] = NS(files=[NS(name="nested/data.bin", total_bytes=0)], next_page_token="")
    result = backend.dataset_info(REF)
    assert result == {"status": "ready", "ready": True, "version": 1,
        "files": [{"name": "payload.zip", "bytes": 42}, {"name": "nested/data.bin", "bytes": 0}],
        "is_private": True}
    requests = [call for call in api.calls if call[0] == "dataset_list_files"]
    assert [call[1][0] for call in requests] == [f"{REF}/1", f"{REF}/1"]
    assert [call[2]["page_token"] for call in requests] == [None, "next"]
    detail = next(call[1][0] for call in api.calls if call[0] == "get_dataset")
    assert (detail.owner_slug, detail.dataset_slug) == tuple(REF.split("/"))


@pytest.mark.parametrize("state,ready", [("READY", True), ("reprocessing", False),
    ("failed", False), ("blobs_received", False), (KEY, False)])
def test_dataset_readiness_is_explicit(harness, state, ready):
    backend, api, _ = harness
    api.dataset_state = state
    result = backend.dataset_info(REF)
    assert result["ready"] is ready
    assert KEY not in result["status"]


@pytest.mark.parametrize("private", [True, False, None])
def test_dataset_privacy_is_not_invented(harness, private):
    backend, api, _ = harness
    api.dataset.is_private = private
    assert backend.dataset_info(REF)["is_private"] is private


@pytest.mark.parametrize("version", [None, 0, 1])
def test_pending_dataset_does_not_list_unavailable_files(harness, version):
    backend, api, _ = harness
    api.dataset_state = "blobs_received"
    api.dataset.current_version_number = version
    api.errors["dataset_list_files"] = HttpFailure(404)
    assert backend.dataset_info(REF) == {"status": "blobs_received", "ready": False,
        "version": version or None, "files": [], "is_private": True}
    assert not any(call[0] == "dataset_list_files" for call in api.calls)


@pytest.mark.parametrize("method,lookup", [("get_dataset", "dataset_info"), ("get_kernel", "kernel_info")])
@pytest.mark.parametrize("code", [404, 401, 403, 429, 500, None])
def test_none_only_on_definitive_resource_404(harness, method, lookup, code):
    backend, api, _ = harness
    api.errors[method] = HttpFailure(code) if code else TimeoutError(KEY)
    if code == 404:
        assert getattr(backend, lookup)(REF) is None
    else:
        with pytest.raises(b.BackendError) as caught:
            getattr(backend, lookup)(REF)
        assert caught.value.status_code == code
        assert KEY not in "".join(traceback.format_exception(caught.value))
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


@pytest.mark.parametrize("method,lookup", [("dataset_status", "dataset_info"),
    ("dataset_list_files", "dataset_info"), ("kernels_status", "kernel_info"), ("kernels_pull", "kernel_info")])
def test_secondary_404_does_not_claim_resource_absent(harness, method, lookup):
    backend, api, _ = harness
    api.errors[method] = HttpFailure(404)
    with pytest.raises(b.BackendError, match="HTTP 404"):
        getattr(backend, lookup)(REF)


def test_private_creation_and_no_raw_response(harness):
    backend, api, folder = harness
    api.create_response.url = KEY
    result = backend.create_dataset(folder)
    assert result == {"ref": REF, "url": f"https://www.kaggle.com/datasets/{REF}", "is_private": True}
    call = next(call for call in api.calls if call[0] == "dataset_create_new")
    assert call[2] == {"public": False, "quiet": True, "convert_to_csv": False, "dir_mode": "zip"}


def test_create_purges_stale_resumable_upload_state(harness, monkeypatch, tmp_path):
    """Stale SDK upload state must not deadlock retries ("Upload resume disabled")."""
    import tempfile
    backend, api, folder = harness
    state_dir = tmp_path / ".kaggle-uploads"
    state_dir.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    stale = []
    for spelling in (str(folder), str(Path(folder).resolve())):
        for name in Path(folder).rglob("*"):
            if name.is_file():
                info = state_dir / ("%s.json" % (str(name) if spelling == str(folder) else str(name.resolve())).replace(os.path.sep, "_").replace(":", "_"))
                info.write_text("{}", encoding="utf-8")
                stale.append(info)
    assert stale
    backend.create_dataset(folder)
    assert all(not info.exists() for info in stale)


@pytest.mark.parametrize("mutation", ["create_dataset", "push_kernel"])
@pytest.mark.parametrize("failure", ["exception", "body_error", "empty_response"])
def test_mutation_failures_are_redacted_and_not_retried(harness, mutation, failure):
    backend, api, folder = harness
    sdk_method = "dataset_create_new" if mutation == "create_dataset" else "kernels_push"
    response_attr = "create_response" if mutation == "create_dataset" else "push_response"
    if failure == "exception":
        api.errors[sdk_method] = HttpFailure(503)
    elif failure == "body_error":
        getattr(api, response_attr).error = KEY
    else:
        setattr(api, response_attr, None)
    with pytest.raises(b.BackendError) as caught:
        if mutation == "create_dataset":
            backend.create_dataset(folder)
        else:
            backend.push_kernel(folder, "NvidiaTeslaT4x2", 300)
    assert KEY not in "".join(traceback.format_exception(caught.value))
    assert len([call for call in api.calls if call[0] == sdk_method]) == 1


def test_kernel_read_only_reconciliation_binds_pinned_code_and_metadata(harness, tmp_path):
    backend, api, _ = harness
    path = tmp_path / "original.ipynb"
    path.write_text(json.dumps(NOTEBOOK), encoding="utf-8")
    result = backend.kernel_info(REF)
    assert result == {"version": 1, "status": "complete", "is_private": True,
        "dataset_sources": [REF], "machine_shape": "NvidiaTeslaT4x2", "code_sha256": b.notebook_digest(path)}
    assert [call[1][0] for call in api.calls if call[0] in ("kernels_status", "kernels_pull")] == [f"{REF}/1"] * 2
    assert not any(call[0] in ("kernels_push", "dataset_create_new") for call in api.calls)


def test_kernel_push_explicit_accelerator_and_timeout(harness):
    backend, api, folder = harness
    assert backend.push_kernel(folder, "NvidiaTeslaT4x2", 123) == {
        "version": 1, "url": f"https://www.kaggle.com/code/{REF}"}
    assert next(call for call in api.calls if call[0] == "kernels_push")[2] == {
        "timeout": "123", "acc": "NvidiaTeslaT4x2", "no_run": False}


def test_public_kernel_rejected_before_submission(harness):
    backend, api, folder = harness
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": REF, "is_private": False}))
    with pytest.raises(b.BackendError):
        backend.push_kernel(folder, "NvidiaTeslaT4x2", 123)
    assert not any(call[0] == "kernels_push" for call in api.calls)


@pytest.mark.parametrize("raw,expected", [("QUEUED", "queued"), ("running", "running"),
    (NS(name="COMPLETE"), "complete"), ("KernelWorkerStatus.ERROR", "error"),
    ("CANCEL_REQUESTED", "running"), ("CANCEL_ACKNOWLEDGED", "cancelled"),
    ("canceled", "cancelled"), ("NEW_SCRIPT", "queued"), (KEY, "unknown"), (None, "unknown")])
def test_version_pinned_status_normalization(harness, raw, expected):
    backend, api, _ = harness
    api.state = raw
    assert backend.kernel_status(REF, 7) == expected
    assert api.calls[-1][1] == (f"{REF}/7",)


def test_output_is_pinned_and_not_silently_downgraded(harness, tmp_path):
    backend, api, _ = harness
    folder = tmp_path / "outputs"
    assert backend.download_outputs(REF, 7, folder) is None
    assert (folder / "result.json").read_text() == "{}"
    assert api.calls[-1] == ("kernels_output", (f"{REF}/7",), {"path": str(folder), "force": True, "quiet": True})
    api.errors["kernels_output"] = HttpFailure(404)
    with pytest.raises(b.BackendError, match="HTTP 404"):
        backend.download_outputs(REF, 7, folder)
    assert [call[1][0] for call in api.calls if call[0] == "kernels_output"] == [f"{REF}/7"] * 2


@pytest.mark.parametrize("version", [None, 0, -1, True, "1", 1.0])
def test_invalid_version_never_requests_latest(harness, tmp_path, version):
    backend, api, _ = harness
    with pytest.raises(b.BackendError):
        backend.download_outputs(REF, version, tmp_path / "output")
    with pytest.raises(b.BackendError):
        backend.kernel_status(REF, version)
    assert not api.calls


@pytest.mark.parametrize("failure", [None, HttpFailure(401), SystemExit(KEY)])
def test_credentials_are_scoped_and_output_suppressed(harness, monkeypatch, capsys, caplog, failure):
    backend, api, _ = harness
    ambient = {"KAGGLE_USERNAME": "ambient", "KAGGLE_KEY": "ambient-key",
               "KAGGLE_API_TOKEN": "ambient-token", "KAGGLE_API_V1_TOKEN": "ambient-token-file",
               "KAGGLE_CONFIG_DIR": "ambient-config", "KAGGLE_API_ENVIRONMENT": "LOCALHOST",
               "KAGGLE_KERNEL_RUN_TYPE": "Interactive", "VERBOSE": "true"}
    for name, value in ambient.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)
    disabled = logging.root.manager.disable
    original = api.kernels_status
    def status(kernel):
        assert os.environ["KAGGLE_USERNAME"] == "explicit-user"
        assert os.environ["KAGGLE_KEY"] == KEY
        assert not set(ambient).difference({"KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"}) & set(os.environ)
        assert Path(os.environ["KAGGLE_CONFIG_DIR"]).is_dir()
        return original(kernel)
    api.kernels_status = status
    if failure:
        api.errors["kernels_status"] = failure
        with pytest.raises(b.BackendError):
            backend.kernel_status(REF, 1)
    else:
        assert backend.kernel_status(REF, 1) == "complete"
    assert dict(os.environ) == before
    assert logging.root.manager.disable == disabled
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert KEY not in caplog.text


def test_notebook_digest_matches_sdk_semantics_but_detects_changed_code(tmp_path):
    original = json.loads(json.dumps(NOTEBOOK))
    original["cells"][0].update(outputs=[{"text": "old output"}], execution_count=37)
    original["cells"][0]["metadata"]["execution"] = {"started": "yesterday"}
    original["cells"][0]["metadata"]["trusted"] = True
    original["metadata"]["widgets"] = {"old": "state"}
    first, pulled = tmp_path / "original.ipynb", tmp_path / "pulled.ipynb"
    first.write_text(json.dumps(original, indent=4, ensure_ascii=False), encoding="utf-8")
    normalized = json.loads(json.dumps(NOTEBOOK))
    normalized["cells"][0]["source"] = "".join(normalized["cells"][0]["source"])
    pulled.write_text(json.dumps(normalized), encoding="utf-8")
    assert hashlib.sha256(first.read_bytes()).digest() != hashlib.sha256(pulled.read_bytes()).digest()
    assert b.notebook_digest(first) == b.notebook_digest(pulled)
    normalized["cells"][0]["source"] += "print('changed')"
    pulled.write_text(json.dumps(normalized), encoding="utf-8")
    assert b.notebook_digest(first) != b.notebook_digest(pulled)


def test_installed_sdk_contract_offline(monkeypatch, tmp_path):
    if importlib.util.find_spec("kaggle") is None:
        pytest.skip("Optional SDK not installed; run with .venv-kaggle")
    # Real initialization/import and request types, with no network, no disk key.
    monkeypatch.setenv("KAGGLE_API_TOKEN", "must-not-introspect")
    monkeypatch.setenv("KAGGLE_API_V1_TOKEN", str(tmp_path / "must-not-read-token"))
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "ambient-config"))
    original_open = builtins.open
    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            path = Path(file)
            if path.name == "must-not-read-token":
                pytest.fail("Ambient token must never be read")
            if path.name == "kaggle.json":
                assert path.resolve().parent == Path(os.environ["KAGGLE_CONFIG_DIR"]).resolve()
                assert path.parent.name.startswith("kaggle-adapter-")
        return original_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded_open)
    backend = b.KaggleBackend("explicit-user", KEY)
    with backend._operation("initialize") as api:
        assert api.config_values == {"username": "explicit-user", "key": KEY, "auth_method": "LEGACY_API_KEY"}
        assert api.args == []
        assert api.MAX_UPLOAD_RESUME_ATTEMPTS == 1
        def fail_once():
            raise HttpFailure(429)
        assert api.with_retry(fail_once) is fail_once
        # Real types reject nonexistent fields, verifying inspected signatures.
        request = b._request("kernels", "ApiGetKernelRequest", user_name="test-owner", kernel_slug="unique-experiment", version_label="v7")
        assert request.version_label == "v7"
        dataset = b._request("datasets", "ApiGetDatasetRequest", owner_slug="test-owner", dataset_slug="unique-experiment")
        assert dataset.owner_slug == "test-owner"
        import kaggle
        assert kaggle.__version__ == "2.2.5"
        assert api.parse_kernel_string(f"{REF}/7") == ("test-owner", "unique-experiment", "7")
        assert api._kernel_version_label("7") == "v7"


@pytest.mark.parametrize("active,identity,expected_owner,valid", [
    (True, "explicit-user", "", True),
    (True, "explicit-user", "explicit-user", True),
    (False, "explicit-user", "", False),
    (True, "", "", False),
    (True, "other-user", "explicit-user", False),
])
def test_real_sdk_token_bearer_introspection_and_bounded_reads(monkeypatch, active, identity, expected_owner, valid):
    if importlib.util.find_spec("kaggle") is None:
        pytest.skip("Optional SDK not installed; run with .venv-kaggle")
    import requests
    import kagglesdk
    from kagglesdk import kaggle_http_client
    calls = []
    def forbidden(*args, **kwargs):
        pytest.fail("Explicit token must never fall back to credentials")
    monkeypatch.setattr(kagglesdk.KaggleCredentials, "load", forbidden)
    monkeypatch.setattr(kaggle_http_client, "_get_apikey_creds", forbidden)
    def send(session, request, **kwargs):
        operation = request.url.rsplit("/", 1)[-1]
        calls.append((operation, request.headers.get("Authorization"), json.loads(request.body)))
        payload = {
            "IntrospectToken": {"active": active, "username": identity},
            "ListDatasets": {"datasets": [], "nextPageToken": "must-not-follow"},
            "ListKernels": {"kernels": [], "nextPageToken": "must-not-follow"},
            "GetKernelSessionStatus": {"status": "COMPLETE"},
        }.get(operation)
        if payload is None:
            pytest.fail("Unexpected SDK operation")
        response = requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "application/json"
        response._content = json.dumps(payload).encode("utf-8")
        response.url, response.request = request.url, request
        return response
    monkeypatch.setattr(requests.sessions.Session, "send", send)
    backend = b.KaggleBackend(expected_owner, "", api_token=TOKEN)
    with backend._operation("initialize") as api:
        api.authenticate()
        config = dict(api.config_values)
        authenticated = api._authenticated
    assert calls == []  # Initialization installs bearer auth without introspection.
    assert authenticated is True
    assert config == {"token": TOKEN, "username": expected_owner, "auth_method": "ACCESS_TOKEN"}
    if valid:
        assert backend.check_connection() == {
            "authenticated": True, "username": "explicit-user", "auth_method": "api_token",
            "dataset_read": True, "datasets_visible": 0, "kernel_read": True, "kernels_visible": 0}
        assert backend.kernel_status(REF, 1) == "complete"
        assert [call[0] for call in calls] == ["IntrospectToken", "ListDatasets", "ListKernels", "GetKernelSessionStatus"]
        for _, _, body in calls[1:3]:
            assert body["user"] == "explicit-user"
            assert body["pageSize"] == 1
        assert backend._api.config_values["username"] == "explicit-user"
    else:
        with pytest.raises(b.BackendError, match="^Kaggle check_connection failed$"):
            backend.check_connection()
        assert [call[0] for call in calls] == ["IntrospectToken"]
    assert calls[0][2] == {"token": TOKEN}
    assert all(call[1] == "Bearer " + TOKEN for call in calls)


def test_lazy_initialization_failure_is_redacted_and_restores_environment(monkeypatch, capsys):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "ambient-synthetic-token")
    before = dict(os.environ)
    def broken(username, key):
        assert username == "explicit-user" and key == KEY
        assert os.environ["KAGGLE_KEY"] == KEY
        assert "KAGGLE_API_TOKEN" not in os.environ
        print(key)
        print(key, file=sys.stderr)
        raise SystemExit(key)
    monkeypatch.setattr(b, "_new_api", broken)
    backend = b.KaggleBackend("explicit-user", KEY)
    with pytest.raises(b.BackendError, match="^Kaggle kernel_status failed$") as caught:
        backend.kernel_status(REF, 1)
    assert caught.value.__context__ is None
    assert backend._api is None
    assert dict(os.environ) == before
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.fixture
def installed_backend():
    if importlib.util.find_spec("kaggle") is None:
        pytest.skip("Optional SDK not installed; run with .venv-kaggle")
    backend = b.KaggleBackend("explicit-user", KEY)
    with backend._operation("initialize"):
        pass
    return backend


def test_real_sdk_push_pull_status_output_all_use_official_fields(installed_backend, monkeypatch, tmp_path):
    backend = installed_backend
    api = backend._api
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelResponse, ApiKernelMetadata, ApiKernelBlob, ApiSaveKernelResponse,
        ApiGetKernelSessionStatusResponse, ApiListKernelSessionOutputResponse,
    )
    from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus
    folder = tmp_path / "staging"
    folder.mkdir()
    original = json.loads(json.dumps(NOTEBOOK))
    original["cells"][0].update(outputs=[{"text": "discard me"}], execution_count=8)
    (folder / "original.ipynb").write_text(json.dumps(original, indent=2), encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": REF, "code_file": "original.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_internet": False}))
    saved, pulls, statuses, pages = [], [], [], []
    def save(request):
        saved.append(request)
        response = ApiSaveKernelResponse()
        response.version_number = 1
        return response
    def pull(request):
        pulls.append(request)
        response = ApiGetKernelResponse()
        response.metadata = metadata = ApiKernelMetadata()
        metadata.current_version_number = 1
        metadata.is_private = True
        metadata.dataset_data_sources = [REF]
        metadata.machine_shape = "NvidiaTeslaT4x2"
        response.blob = blob = ApiKernelBlob()
        blob.source = saved[0].text
        blob.language = "python"
        blob.kernel_type = "notebook"
        return response
    def status(request):
        statuses.append(request)
        response = ApiGetKernelSessionStatusResponse()
        response.status = KernelWorkerStatus.COMPLETE
        return response
    def output(request):
        pages.append(request)
        response = ApiListKernelSessionOutputResponse()
        response.next_page_token = "second" if len(pages) == 1 else ""
        return response
    service = NS(save_kernel=save, get_kernel=pull, get_kernel_session_status=status,
                 list_kernel_session_output=output)
    monkeypatch.setattr(api, "build_kaggle_client", lambda: nullcontext(NS(kernels=NS(kernels_api_client=service))))
    assert backend.push_kernel(folder, "NvidiaTeslaT4x2", 123)["version"] == 1
    assert len(saved) == 1
    assert saved[0].is_private is True and saved[0].enable_internet is False
    assert saved[0].session_timeout_seconds == 123
    assert saved[0].machine_shape == "NvidiaTeslaT4x2"
    pushed = json.loads(saved[0].text)
    assert isinstance(pushed["cells"][0]["source"], str)
    assert pushed["cells"][0]["outputs"] == []
    info = backend.kernel_info(REF)
    assert info["code_sha256"] == b.notebook_digest(folder / "original.ipynb")
    assert info["is_private"] is True
    assert info["dataset_sources"] == [REF]
    assert [request.version_label for request in pulls] == ["", "v1"]
    assert [request.version_label for request in statuses] == ["v1"]
    backend.download_outputs(REF, 7, tmp_path / "outputs")
    assert len(pages) == 2
    assert [request.version_label for request in pages] == ["v7", "v7"]


@pytest.mark.parametrize("preflight_code,expected_commits", [(404, 1), (403, 1), (500, 0)])
def test_real_sdk_create_treats_404_and_403_preflight_as_absent(installed_backend, monkeypatch, tmp_path, preflight_code, expected_commits):
    import requests
    backend = installed_backend
    api = backend._api
    folder = tmp_path / "dataset"
    folder.mkdir()
    (folder / "dataset-metadata.json").write_text(json.dumps({"id": REF, "title": "Unique Experiment",
        "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
    commits = []
    def status(request):
        raise requests.HTTPError(KEY, response=NS(status_code=preflight_code))
    def create(request):
        commits.append(request)
        raise requests.HTTPError(KEY, response=NS(status_code=429))
    service = NS(get_dataset_status=status, create_dataset=create)
    monkeypatch.setattr(api, "build_kaggle_client", lambda: nullcontext(NS(datasets=NS(dataset_api_client=service))))
    monkeypatch.setattr(api, "upload_files", lambda *args, **kwargs: None)
    with pytest.raises(b.BackendError) as caught:
        backend.create_dataset(folder)
    assert caught.value.status_code == (429 if expected_commits else preflight_code)
    assert KEY not in "".join(traceback.format_exception(caught.value))
    assert len(commits) == expected_commits
    if commits:
        assert commits[0].is_private is True


def test_real_sdk_dataset_details_include_privacy_and_version(installed_backend, monkeypatch):
    from kagglesdk.datasets.types.dataset_api_service import (
        ApiDataset, ApiDatasetFile, ApiGetDatasetStatusResponse, ApiListDatasetFilesResponse,
    )
    from kagglesdk.datasets.types.dataset_enums import DatabundleVersionStatus
    backend = installed_backend
    detail = ApiDataset()
    detail.is_private = True
    detail.current_version_number = 1
    status = ApiGetDatasetStatusResponse()
    status.status = DatabundleVersionStatus.READY
    listing = ApiListDatasetFilesResponse()
    file = ApiDatasetFile()
    file.name = "payload.zip"
    file.total_bytes = 42
    listing.dataset_files = [file]
    listed_versions = []
    def files(request):
        listed_versions.append(request.dataset_version_number)
        return listing
    service = NS(get_dataset=lambda request: detail, get_dataset_status=lambda request: status,
                 list_dataset_files=files)
    monkeypatch.setattr(backend._api, "build_kaggle_client", lambda: nullcontext(NS(datasets=NS(dataset_api_client=service))))
    assert backend.dataset_info(REF) == {"status": "ready", "ready": True, "version": 1,
        "files": [{"name": "payload.zip", "bytes": 42}], "is_private": True}
    assert listed_versions == [1]


def test_upload_complete_is_the_sdk_original_not_an_override(installed_backend):
    """Live diagnosis 2026-10-25: a custom single-PUT upload_complete (no
    resume, timeout, no redirects) hung to timeout against the real blob
    endpoint and deadlocked retries via persisted upload state.  The adapter
    must keep kaggle 2.2.5's own resumable implementation."""
    api = installed_backend._api
    assert "upload_complete" not in vars(type(api))
    assert callable(api.upload_complete)


@pytest.mark.parametrize("auth_mode", ["legacy", "api_token"])
@pytest.mark.parametrize("already_imported", [False, True])
@pytest.mark.parametrize("failure_stage", ["none", "operation", "initialize"])
def test_default_disk_tokens_are_never_read_or_introspected_and_helpers_restore(already_imported, failure_stage, auth_mode):
    if importlib.util.find_spec("kaggle") is None:
        pytest.skip("Optional SDK not installed; run with .venv-kaggle")
    # Each child starts with a CLEAN kaggle import cache. All home credential
    # readers and network paths are replaced before any Kaggle initialization.
    script = r'''
import builtins
from contextlib import ExitStack
import importlib
import os
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch

class Forbidden(BaseException):
    pass

network_calls, disk_reads, token_calls = [], [], []
def network_trap(*args, **kwargs):
    network_calls.append(True)
    raise Forbidden("Network is forbidden")
socket.socket.connect = network_trap
socket.create_connection = network_trap
original_read_text = Path.read_text
def guarded_read_text(path, *args, **kwargs):
    if path.name in ("access_token", "access_token.txt", "kaggle.json"):
        raise Forbidden("Real credential file reads forbidden")
    return original_read_text(path, *args, **kwargs)
Path.read_text = guarded_read_text
original_open = builtins.open
def guarded_open(file, *args, **kwargs):
    if isinstance(file, (str, os.PathLike)):
        path = Path(file)
        if path.name in ("access_token", "access_token.txt"):
            raise Forbidden("Real token file reads forbidden")
        if path.name == "kaggle.json" and not path.parent.name.startswith("kaggle-adapter-"):
            raise Forbidden("Real account file reads forbidden")
    return original_open(file, *args, **kwargs)
builtins.open = guarded_open
import requests
requests.sessions.Session.send = network_trap
import kagglesdk
env = importlib.import_module("kagglesdk.kaggle_env")
http = importlib.import_module("kagglesdk.kaggle_http_client")
def fake_default_token_file(path):
    # Never inspects the actual home path or touches the filesystem.
    disk_reads.append(True)
    return "synthetic-home-token"
env._get_access_token_from_file = fake_default_token_file
def keyfile_trap(*args, **kwargs):
    raise Forbidden("Implicit disk API key forbidden")
http._get_apikey_creds = keyfile_trap
kagglesdk.KaggleCredentials.load = keyfile_trap
from tools.kaggle_pipeline import backend as b
assert "kaggle" not in sys.modules
already_imported = sys.argv[1] == "True"
failure_stage = sys.argv[2]
auth_mode = sys.argv[3]
if already_imported:
    # Populate the cache independently of the adapter. Guard import-time auth
    # with test-only disabled helpers and fake credentials in a temporary dir.
    with tempfile.TemporaryDirectory(prefix="kaggle-adapter-bootstrap-") as folder:
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"KAGGLE_USERNAME": "bootstrap-user",
                "KAGGLE_KEY": "bootstrap-key", "KAGGLE_CONFIG_DIR": folder}, clear=True))
            for module in (env, kagglesdk, http):
                stack.enter_context(patch.object(module, "get_access_token_from_env", lambda: (None, None)))
            importlib.import_module("kaggle")
    assert "kaggle" in sys.modules

def sentinel(name):
    def reader():
        token_calls.append(name)
        # Use the real lookup algorithm with its disk reader safely replaced.
        return original_lookup()
    return reader
original_lookup = env.get_access_token_from_env
names = ("kagglesdk.kaggle_env", "kagglesdk", "kagglesdk.kaggle_http_client",
         "kaggle.api.kaggle_api_extended")
originals = {}
for name in names:
    if name in sys.modules:
        module = sys.modules[name]
        module.get_access_token_from_env = reader = sentinel(name)
        originals[name] = reader
expected_new_alias = originals["kagglesdk"]
before_environment = dict(os.environ)
token = "KGAT_synthetic-offline-token" if auth_mode == "api_token" else None
backend = b.KaggleBackend("explicit-user", "" if token else "explicit-synthetic-key", api_token=token)
initialized_configs = []
real_new_api = b._new_api
def new_api(username, key, api_token=None):
    api = real_new_api(username, key, api_token)
    initialized_configs.append(dict(api.config_values))
    if failure_stage == "initialize":
        raise RuntimeError("synthetic initialization failure")
    return api
b._new_api = new_api
expected_config = {"username": "explicit-user", "key": "explicit-synthetic-key",
                   "auth_method": "LEGACY_API_KEY"}
expected_auth = ("explicit-user", "explicit-synthetic-key")
if token:
    expected_config = {"username": "explicit-user", "token": token, "auth_method": "ACCESS_TOKEN"}
    expected_auth = "Bearer " + token
for attempt in range(2):
    observed = []
    try:
        with backend._operation("initialize") as api:
            # Real SDK authenticate and lazy HTTP-session auth, but no API call.
            api.authenticate()
            observed.append(dict(api.config_values))
            observed.append([sys.modules[name].get_access_token_from_env() for name in names])
            with api.build_kaggle_client() as client:
                auth = client._http_client._session.auth
                if token:
                    request = auth(type("Request", (), {"headers": {}})())
                    observed.append(request.headers["Authorization"])
                else:
                    observed.append(auth)
            if failure_stage == "operation":
                raise RuntimeError("synthetic operation failure")
    except b.BackendError:
        assert failure_stage != "none"
    else:
        assert failure_stage == "none"
    # Assert outside the adapter's exception-redaction scope so a regression
    # cannot be mistaken for the deliberately injected failure.
    assert initialized_configs and all(config == expected_config for config in initialized_configs)
    if failure_stage == "initialize":
        assert observed == []
        assert backend._api is None
    else:
        assert observed == [expected_config, [(None, None)] * len(names), expected_auth]
    for name in names:
        assert sys.modules[name].get_access_token_from_env is originals.get(name, expected_new_alias)
    assert dict(os.environ) == before_environment
assert token_calls == [], token_calls
assert disk_reads == [], disk_reads
assert network_calls == [], network_calls
'''
    completed = subprocess.run([sys.executable, "-c", script, str(already_imported), failure_stage, auth_mode],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == completed.stderr == ""
