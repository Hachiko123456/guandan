"""Token configuration tests use only synthetic secrets and fake backends."""
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from tools.kaggle_pipeline import core, backend

TOKEN = "KGAT_synthetic_local_test_only"


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("configuration test attempted network")
    monkeypatch.setattr(socket.socket, "connect", denied)


@pytest.fixture
def config(tmp_path):
    values = core.load_config()
    values["account_file"] = str(tmp_path / "account.json")
    values["output_root"] = str(tmp_path / "runs")
    return values


def write(config, value):
    Path(config["account_file"]).write_text(json.dumps(value), encoding="utf-8-sig")


def test_token_only_is_ready_offline_without_leaking_token(config):
    write(config, {"api_token": TOKEN})
    values = core.credentials(config)
    assert values == {"username": "", "key": "", "api_token": TOKEN, "auth_method": "api_token"}
    report = core.doctor(config)
    assert report["status"] == "READY" and report["auth_method"] == "api_token"
    assert report["remote_calls"] is False
    assert TOKEN not in json.dumps(report)
    assert not Path(config["output_root"]).exists()


@pytest.mark.parametrize("value", [
    {"api_token": TOKEN, "key": "legacy-secret"},
    {"username": "sample", "key": TOKEN},
    {"api_token": TOKEN + "\n"}, {"api_token": 123},
    {"api_token": "not-a-kaggle-token"}, {"api_token": "KGAT_"},
    {"api_token": TOKEN, "username": "bad/user"},
])
def test_invalid_or_mixed_modes_are_actionable_without_secret_output(config, value):
    write(config, value)
    with pytest.raises(core.PipelineError) as error:
        core.credentials(config)
    assert config["account_file"] in str(error.value)
    assert TOKEN not in str(error.value)


def confirmed(**changes):
    return {"authenticated": True, "username": "testowner", "dataset_read": True, "kernel_read": True, **changes}


def test_check_caches_only_confirmed_owner_in_same_ignored_account_file(config):
    write(config, {"api_token": TOKEN})
    report = core.check_connection(config, backend=SimpleNamespace(check_connection=lambda: confirmed()))
    assert report["status"] == "CONNECTED" and report["username"] == "testowner"
    assert TOKEN not in json.dumps(report)
    assert json.loads(Path(config["account_file"]).read_text()) == {"username": "testowner", "api_token": TOKEN}
    assert not Path(config["output_root"]).exists()


@pytest.mark.parametrize("changes", [{"username": "bad/user"}, {"authenticated": False}, {"dataset_read": False}, {"kernel_read": False}])
def test_check_does_not_cache_an_unverified_identity(config, changes):
    write(config, {"api_token": TOKEN})
    original = Path(config["account_file"]).read_bytes()
    with pytest.raises(core.PipelineError):
        core.check_connection(config, backend=SimpleNamespace(check_connection=lambda: confirmed(**changes)))
    assert Path(config["account_file"]).read_bytes() == original


def test_check_does_not_overwrite_user_credential_change_during_request(config):
    write(config, {"api_token": TOKEN})
    replacement = "KGAT_synthetic_replacement"
    def check():
        write(config, {"api_token": replacement})
        return confirmed()
    with pytest.raises(core.PipelineError, match="changed"):
        core.check_connection(config, backend=SimpleNamespace(check_connection=check))
    assert core.credentials(config)["api_token"] == replacement


def test_cached_owner_mismatch_is_rejected(config):
    write(config, {"api_token": TOKEN, "username": "otherowner"})
    with pytest.raises(core.PipelineError, match="owner"):
        core.check_connection(config, backend=SimpleNamespace(check_connection=lambda: confirmed()))


def test_backend_uses_explicit_token_and_preserves_owner_guard(config, monkeypatch):
    write(config, {"api_token": TOKEN, "username": "testowner"})
    calls = []
    def factory(username, key, **kwargs):
        calls.append((username, key, kwargs))
        return object()
    monkeypatch.setattr(backend, "KaggleBackend", factory)
    owner, _ = core.backend_for(config, {"owner": "testowner"})
    assert owner == "testowner"
    assert calls == [("testowner", "", {"api_token": TOKEN})]
    with pytest.raises(core.PipelineError, match="account"):
        core.backend_for(config, {"owner": "otherowner"})
    assert len(calls) == 1


def test_legacy_account_still_works(config):
    write(config, {"username": "legacyowner", "key": "synthetic-legacy-key"})
    assert core.account(config) == ("legacyowner", "synthetic-legacy-key")
    assert core.doctor(config)["auth_method"] == "legacy"
