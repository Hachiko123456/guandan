"""Stage-5 前置任务 A: DMC Gate L delivery verifier (offline, fake workspace).

``scripts/verify_dmc_delivery.py`` must fail closed on a tampered or drifted
delivery.  These tests build a tiny but structurally complete delivery inside
a fake workspace (no Kaggle SDK, no network, no subprocess) and then check:

* the happy path passes end-to-end through the real builder;
* a truncated payload, a budget above the platform-timeout margin, an unbound
  notebook, workspace config drift, and a **consistently re-hashed gate
  mutation** (i.e. a dishonest builder) all FAIL;
* the CLI (``scripts/kaggle_pipeline.py``) dispatches the DMC verifier for
  DMC plan schemas and keeps the serial-compare default otherwise.

The fake workspace ships exactly the verifier's required module paths as tiny
placeholder sources, so overlay/driver workspace-identity checks stay cheap.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_dmc_gate_l_delivery import (
    CONFIG_PATH,
    MAGIC,
    NOTEBOOK,
    PAYLOAD,
    build_delivery,
    canonical,
    sha_bytes,
    zip_bytes,
)
from scripts.verify_dmc_delivery import (
    QAFailure,
    REQUIRED_DRIVER,
    REQUIRED_OVERLAY,
    verify,
)

GIT_COMMIT = "f" * 40


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    configs = repo / "configs"
    configs.mkdir(parents=True)
    real_config = Path(__file__).resolve().parents[2] / CONFIG_PATH
    (repo / CONFIG_PATH).write_bytes(real_config.read_bytes())
    for name in REQUIRED_OVERLAY:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# placeholder overlay module {name}\n", encoding="utf-8")
    for name in REQUIRED_DRIVER:
        path = repo / "tools" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# placeholder driver module {name}\n", encoding="utf-8")
    return repo


@pytest.fixture
def delivery(fake_repo: Path, tmp_path: Path) -> Path:
    output = tmp_path / "delivery_v1"
    build_delivery(fake_repo, output, git_commit=GIT_COMMIT)
    return output


def test_verify_passes_on_a_fresh_delivery(delivery: Path, fake_repo: Path) -> None:
    record = verify(delivery, repo=fake_repo, timeout_seconds=15000)
    assert record["status"] == "PASS"
    assert record["budget_total_seconds"] == 14400
    assert record["overlay_files"] == len(REQUIRED_OVERLAY)
    assert record["driver_files"] == len(REQUIRED_DRIVER)
    assert len(record["plan_hash"]) == 64
    assert len(record["payload"]["sha256"]) == 64
    assert len(record["config_sha256"]) == 64


def test_verify_fails_on_a_truncated_payload(delivery: Path, fake_repo: Path) -> None:
    data = (delivery / PAYLOAD).read_bytes()
    (delivery / PAYLOAD).write_bytes(data[:-8])
    with pytest.raises(QAFailure):
        verify(delivery, repo=fake_repo, timeout_seconds=15000)


def test_verify_enforces_the_budget_margin(delivery: Path, fake_repo: Path) -> None:
    # 14400 + 300 == 14700: the exact margin passes, one second less fails.
    assert verify(delivery, repo=fake_repo, timeout_seconds=14700)["status"] == "PASS"
    with pytest.raises(QAFailure, match="exceeds the platform timeout"):
        verify(delivery, repo=fake_repo, timeout_seconds=14699)


def test_verify_fails_when_the_notebook_is_unbound(delivery: Path, fake_repo: Path) -> None:
    notebook = json.loads((delivery / NOTEBOOK).read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    plan = json.loads((delivery / "plan.json").read_text(encoding="utf-8-sig"))
    code = "".join(cell["source"]).replace(repr(plan["plan_hash"]), repr("0" * 64))
    cell["source"] = code.splitlines(keepends=True)
    (delivery / NOTEBOOK).write_text(json.dumps(notebook), encoding="utf-8")
    with pytest.raises(QAFailure, match="not bound to this plan hash"):
        verify(delivery, repo=fake_repo, timeout_seconds=15000)


def test_verify_fails_on_workspace_config_drift(delivery: Path, fake_repo: Path) -> None:
    config = json.loads((fake_repo / CONFIG_PATH).read_text(encoding="utf-8-sig"))
    config["num_envs"] = 64
    (fake_repo / CONFIG_PATH).write_text(json.dumps(config, indent=2), encoding="utf-8")
    with pytest.raises(QAFailure, match="differs from the workspace"):
        verify(delivery, repo=fake_repo, timeout_seconds=15000)


def _rehash_delivery_with_mutated_plan(delivery: Path, mutate) -> None:
    """Rebuild the payload as a dishonest builder would: consistent hashes, bad content."""
    raw = (delivery / PAYLOAD).read_bytes()
    old_payload_sha = sha_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(raw[len(MAGIC):].strip(), validate=True))) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
    manifest = json.loads(members["delivery_manifest.json"])
    plan = json.loads(members["plan.json"])
    old_manifest_hash = manifest["manifest_hash"]
    old_plan_hash = plan["plan_hash"]
    mutate(plan)
    plan["plan_hash"] = canonical({key: value for key, value in plan.items() if key != "plan_hash"})
    members["plan.json"] = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    manifest["plan_hash"] = plan["plan_hash"]
    rows = {row["path"]: row for row in manifest["files"]}
    rows["plan.json"] = {
        "path": "plan.json",
        "bytes": len(members["plan.json"]),
        "sha256": sha_bytes(members["plan.json"]),
    }
    manifest["files"] = [rows[name] for name in sorted(rows)]
    manifest["manifest_hash"] = canonical(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    members["delivery_manifest.json"] = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8") + b"\n"
    payload = MAGIC + base64.b64encode(zip_bytes(members)) + b"\n"
    (delivery / PAYLOAD).write_bytes(payload)
    new_payload_sha = sha_bytes(payload)
    # keep the top-level copies self-consistent too
    (delivery / "plan.json").write_bytes(members["plan.json"])
    (delivery / "DELIVERY_MANIFEST.json").write_bytes(members["delivery_manifest.json"])

    notebook = json.loads((delivery / NOTEBOOK).read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    code = "".join(cell["source"])
    code = code.replace(repr(old_payload_sha), repr(new_payload_sha))
    code = code.replace(repr(old_manifest_hash), repr(manifest["manifest_hash"]))
    code = code.replace(repr(old_plan_hash), repr(plan["plan_hash"]))
    cell["source"] = code.splitlines(keepends=True)
    (delivery / NOTEBOOK).write_text(json.dumps(notebook), encoding="utf-8")
    (delivery / "SHA256SUMS").unlink()


def test_verify_fails_on_a_consistently_rehashed_gate_mutation(delivery: Path, fake_repo: Path) -> None:
    def mutate(plan: dict) -> None:
        plan["gates"]["vs_rule"]["final_min"] = 0.55

    _rehash_delivery_with_mutated_plan(delivery, mutate)
    with pytest.raises(QAFailure, match="gates drifted"):
        verify(delivery, repo=fake_repo, timeout_seconds=15000)


def test_verify_fails_on_a_budget_mutation(delivery: Path, fake_repo: Path) -> None:
    def mutate(plan: dict) -> None:
        plan["budgets"]["total_seconds"] = 16000

    _rehash_delivery_with_mutated_plan(delivery, mutate)
    with pytest.raises(QAFailure, match="drifted from the pre-declared value"):
        verify(delivery, repo=fake_repo, timeout_seconds=15000)


# ---------------------------------------------------------------------------
# CLI dispatch (前置任务 A3: reuse prepare/run without touching pipeline code)
# ---------------------------------------------------------------------------
def test_verifier_dispatch_keeps_serial_default(tmp_path: Path) -> None:
    from scripts.kaggle_pipeline import verifier_for

    assert verifier_for(None) is None
    other = tmp_path / "delivery"
    other.mkdir()
    (other / "plan.json").write_text(json.dumps({"schema": "guandan-serial-compare-plan-v1"}), encoding="utf-8")
    assert verifier_for(other) is None
    (other / "plan.json").unlink()
    assert verifier_for(other) is None


def test_cli_dispatches_the_dmc_verifier(delivery: Path, tmp_path: Path, monkeypatch) -> None:
    import scripts.verify_dmc_delivery as dmc_verify_module
    from scripts.kaggle_pipeline import main as cli_main

    calls: list[str] = []

    def fake_verify(path, **kwargs):
        calls.append(str(Path(path)))
        return {"status": "PASS"}

    monkeypatch.setattr(dmc_verify_module, "verify", fake_verify)
    config = tmp_path / "pipeline.json"
    config.write_text(json.dumps({
        "schema": "guandan-kaggle-pipeline-config-v1",
        "account_file": str(tmp_path / "account.json"),
        "output_root": str(tmp_path / "out"),
        "registry_root": str(tmp_path / "reg"),
        "delivery_parent": str(tmp_path / "parent"),
        "experiment": "dmc-gate-l",
        "accelerator": "NvidiaTeslaT4",
        "timeout_seconds": 15000,
        "poll_seconds": 30,
        "heartbeat_seconds": 300,
        "dataset_wait_seconds": 1800,
        "enable_internet": False,
        "dataset_license": "other",
        "keep_previous_deliveries": 1,
    }), encoding="utf-8")
    code = cli_main(["--config", str(config), "prepare", "--delivery", str(delivery), "--dry-run"])
    assert code == 0
    assert calls == [str(delivery)]
