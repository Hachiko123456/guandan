"""Local-only contracts for delivery publication and cleanup planning."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.kaggle_pipeline import delivery_registry as registry


def canonical_hash(value, omitted="hash"):
    body = {key: item for key, item in value.items() if key != omitted}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def signed(value, field="hash"):
    result = dict(value)
    result[field] = canonical_hash(result, omitted=field)
    return result


def make_delivery(repo: Path, version: int, *, qa_outside_delivery=True):
    delivery = repo / "runs" / "deliveries" / f"delivery_v{version}"
    delivery.mkdir(parents=True)
    payload = delivery / registry.PAYLOAD_NAME
    notebook = delivery / registry.NOTEBOOK_NAME
    payload.write_bytes(f"payload-{version}".encode())
    write_json(notebook, {"cells": [], "nbformat": 4, "version": version})
    plan = signed({"schema": "synthetic-plan-v1", "version": version}, field="plan_hash")
    write_json(delivery / "plan.json", plan)
    manifest = signed({
        "schema": "guandan-serial-compare-delivery-manifest-v1",
        "plan_hash": plan["plan_hash"],
        "version": version,
    }, field="manifest_hash")
    write_json(delivery / "DELIVERY_MANIFEST.json", manifest)
    verification = {
        "schema": registry.VERIFICATION_SCHEMA,
        "status": "BUILT_TARGET_PREFLIGHT_PENDING",
        "payload": registry.PAYLOAD_NAME,
        "payload_bytes": payload.stat().st_size,
        "payload_sha256": file_hash(payload),
        "notebook": registry.NOTEBOOK_NAME,
        "notebook_sha256": file_hash(notebook),
        "plan_hash": plan["plan_hash"],
        "manifest_hash": manifest["manifest_hash"],
    }
    write_json(delivery / "DELIVERY_VERIFICATION.json", verification)
    qa = signed({
        "schema": registry.QA_SCHEMA,
        "delivery": str(delivery.resolve()),
        "payload": {"sha256": file_hash(payload), "bytes": payload.stat().st_size},
        "plan_hash": plan["plan_hash"],
        "notebook": {"sha256": file_hash(notebook), "code_cell_parsed": True},
        "status": "PASS",
    })
    qa_path = (
        repo / "runs" / "qa" / f"ARTIFACT_QA_V{version}.json"
        if qa_outside_delivery else delivery / "ARTIFACT_QA.json"
    )
    write_json(qa_path, qa)
    return delivery, qa_path, qa


def verifier_for(qa):
    return lambda _delivery: json.loads(json.dumps(qa))


@pytest.fixture
def sample(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    delivery, qa_path, qa = make_delivery(repo, 12)
    return repo, delivery, qa_path, qa


def test_inspect_validates_all_identities_and_stores_only_relative_paths(sample):
    repo, delivery, qa_path, qa = sample
    record = registry.inspect_delivery(repo, delivery, qa_path, verifier_for(qa))
    assert record["schema"] == registry.CURRENT_SCHEMA
    assert record["delivery_dir"] == "runs/deliveries/delivery_v12"
    assert record["artifact_qa_path"] == "runs/qa/ARTIFACT_QA_V12.json"
    assert record["payload_sha256"] == file_hash(delivery / registry.PAYLOAD_NAME)
    assert record["notebook_sha256"] == file_hash(delivery / registry.NOTEBOOK_NAME)
    assert record["plan_sha256"] == file_hash(delivery / "plan.json")
    assert record["artifact_qa_hash"] == file_hash(qa_path)
    assert record["delivery_id"].startswith("serial-compare-v12-")
    assert "/" not in record["delivery_id"] and "\\" not in record["delivery_id"]


@pytest.mark.parametrize("change, message", [
    (lambda qa: qa.update(status="FAIL"), "status"),
    (lambda qa: qa["payload"].update(sha256="0" * 64), "payload"),
    (lambda qa: qa["notebook"].update(sha256="1" * 64), "notebook"),
    (lambda qa: qa.update(plan_hash="2" * 64), "plan"),
    (lambda qa: qa.update(delivery=str(Path(qa["delivery"]).parent)), "delivery"),
])
def test_saved_qa_status_and_delivery_hash_bindings_are_mandatory(sample, change, message):
    repo, delivery, qa_path, qa = sample
    change(qa)
    qa["hash"] = canonical_hash(qa)
    write_json(qa_path, qa)
    with pytest.raises(registry.RegistryError, match=message):
        registry.inspect_delivery(repo, delivery, qa_path, verifier_for(qa))


def test_qa_self_hash_and_actual_verifier_result_are_checked_independently(sample):
    repo, delivery, qa_path, qa = sample
    qa["hash"] = "f" * 64
    write_json(qa_path, qa)
    with pytest.raises(registry.RegistryError, match="internally inconsistent"):
        registry.inspect_delivery(repo, delivery, qa_path, verifier_for(qa))

    _, qa_path, qa = make_delivery(repo, 13)
    wrong_actual = json.loads(json.dumps(qa))
    wrong_actual["payload"]["sha256"] = "0" * 64
    wrong_actual["hash"] = canonical_hash(wrong_actual)
    with pytest.raises(registry.RegistryError, match="actual verifier result payload"):
        registry.inspect_delivery(repo, repo / "runs/deliveries/delivery_v13",
                                  qa_path, verifier_for(wrong_actual))


def test_delivery_hash_mismatch_fails_before_publication(sample):
    repo, delivery, qa_path, qa = sample
    (delivery / registry.PAYLOAD_NAME).write_bytes(b"same location, changed payload")
    with pytest.raises(registry.RegistryError, match="payload (size|hash)"):
        registry.inspect_delivery(repo, delivery, qa_path, verifier_for(qa))


def test_paths_cannot_escape_repository(sample, tmp_path):
    repo, delivery, qa_path, qa = sample
    external = tmp_path / "external"
    external.mkdir()
    with pytest.raises(registry.RegistryError, match="escapes"):
        registry.inspect_delivery(repo, external, qa_path, verifier_for(qa))
    external_qa = tmp_path / "external-qa.json"
    write_json(external_qa, qa)
    with pytest.raises(registry.RegistryError, match="escapes"):
        registry.inspect_delivery(repo, delivery, external_qa, verifier_for(qa))


def test_publish_atomically_replaces_current_and_keeps_history_immutable(sample, monkeypatch):
    repo, delivery, qa_path, qa = sample
    registry_root = repo / "state" / "registry"
    replacements = []
    real_replace = registry.os.replace

    def observed_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        assert Path(source).parent == Path(destination).parent
        assert Path(source).name.endswith(".tmp")
        return real_replace(source, destination)

    monkeypatch.setattr(registry.os, "replace", observed_replace)
    first_time = datetime(2026, 10, 24, 12, tzinfo=timezone.utc)
    record = registry.publish(
        repo, delivery, qa_path, registry_root, verifier_for(qa), lambda: first_time,
    )
    current = registry_root / "current.json"
    history = registry_root / "history" / f"{record['delivery_id']}.json"
    assert json.loads(current.read_text(encoding="utf-8")) == record
    assert json.loads(history.read_text(encoding="utf-8")) == record
    assert [destination for _, destination in replacements] == [current]
    assert list(registry_root.rglob("*.tmp")) == []

    # A repeat is idempotent and retains the immutable publication timestamp.
    repeated = registry.publish(
        repo, delivery, qa_path, registry_root, verifier_for(qa),
        lambda: first_time + timedelta(days=1),
    )
    assert repeated == record
    assert json.loads(history.read_text(encoding="utf-8")) == record
    assert [destination for _, destination in replacements] == [current, current]


def test_publish_rejects_differing_existing_history(sample):
    repo, delivery, qa_path, qa = sample
    record = registry.publish(
        repo, delivery, qa_path, verifier=verifier_for(qa),
        clock=lambda: datetime(2026, 10, 24, tzinfo=timezone.utc),
    )
    history = repo / "runs/kaggle_delivery/history" / f"{record['delivery_id']}.json"
    changed = json.loads(history.read_text(encoding="utf-8"))
    changed["payload_sha256"] = "0" * 64
    write_json(history, changed)
    with pytest.raises(registry.RegistryError, match="collision"):
        registry.publish(repo, delivery, qa_path, verifier=verifier_for(qa))


@pytest.mark.parametrize("target", ["payload", "notebook", "plan", "qa"])
def test_resolve_current_rejects_tampering(sample, target):
    repo, delivery, qa_path, qa = sample
    registry.publish(
        repo, delivery, qa_path, verifier=verifier_for(qa),
        clock=lambda: "2026-10-24T12:00:00+08:00",
    )
    path = {
        "payload": delivery / registry.PAYLOAD_NAME,
        "notebook": delivery / registry.NOTEBOOK_NAME,
        "plan": delivery / "plan.json",
        "qa": qa_path,
    }[target]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(registry.RegistryError):
        registry.resolve_current(repo)


def test_resolve_current_returns_verified_delivery(sample):
    repo, delivery, qa_path, qa = sample
    published = registry.publish(
        repo, delivery, qa_path, verifier=verifier_for(qa),
        clock=lambda: datetime(2026, 10, 24, tzinfo=timezone(timedelta(hours=8))),
    )
    resolved, record = registry.resolve_current(repo)
    assert resolved == delivery.resolve()
    assert record == published


def make_cleanup_tree(repo, versions):
    parent = repo / "runs" / "deliveries"
    for version in versions:
        delivery = parent / f"delivery_v{version}"
        build = parent / f"delivery_v{version}_build"
        (delivery / "nested").mkdir(parents=True)
        build.mkdir(parents=True)
        (delivery / "payload.bin").write_bytes(b"d" * version)
        (delivery / "nested" / "more.bin").write_bytes(b"n" * (version + 1))
        (build / "work.bin").write_bytes(b"b" * (version + 2))
    return parent


def current_for(repo, version):
    return {"delivery_dir": f"runs/deliveries/delivery_v{version}"}


def test_cleanup_retention_is_version_based_not_mtime(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = make_cleanup_tree(repo, (1, 2, 3, 4))
    pipeline = repo / "runs/pipeline"
    pipeline.mkdir(parents=True)
    # Deliberately make v1 newest and v3 oldest; retention must still keep v3.
    os.utime(parent / "delivery_v1", (2_000_000_000, 2_000_000_000))
    os.utime(parent / "delivery_v3", (1, 1))
    plan = registry.cleanup_plan(repo, parent, current_for(repo, 4), pipeline)
    assert [row["relative_path"] for row in plan["retained"]] == [
        "runs/deliveries/delivery_v3", "runs/deliveries/delivery_v3_build",
        "runs/deliveries/delivery_v4", "runs/deliveries/delivery_v4_build",
    ]
    assert [row["relative_path"] for row in plan["candidates"]] == [
        "runs/deliveries/delivery_v1", "runs/deliveries/delivery_v1_build",
        "runs/deliveries/delivery_v2", "runs/deliveries/delivery_v2_build",
    ]


def test_cleanup_retains_deliveries_referenced_by_run_json(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = make_cleanup_tree(repo, (1, 2, 3, 4))
    pipeline = repo / "runs/pipeline"
    write_json(pipeline / "active-a/run.json", {"delivery": str(parent / "delivery_v1")})
    write_json(pipeline / "active-b/run.json", {
        "delivery": {"delivery_dir": "runs/deliveries/delivery_v2"},
    })
    plan = registry.cleanup_plan(repo, parent, current_for(repo, 4), pipeline, keep_previous=0)
    retained_versions = {(row["version"], row["kind"]) for row in plan["retained"]}
    assert retained_versions == {
        (1, "delivery"), (1, "build"), (2, "delivery"), (2, "build"),
        (4, "delivery"), (4, "build"),
    }
    assert {row["version"] for row in plan["candidates"]} == {3}
    assert plan["active_run_records"] == [
        "runs/pipeline/active-a/run.json", "runs/pipeline/active-b/run.json",
    ]


def test_cleanup_candidate_bytes_are_exact_and_paths_are_explicit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = make_cleanup_tree(repo, (1, 2))
    pipeline = repo / "runs/pipeline"
    pipeline.mkdir(parents=True)
    plan = registry.cleanup_plan(repo, parent, current_for(repo, 2), pipeline, keep_previous=0)
    rows = plan["candidates"]
    assert [(row["kind"], row["bytes"]) for row in rows] == [
        ("delivery", 1 + 2), ("build", 1 + 2),
    ]
    assert plan["candidate_bytes"] == 6
    assert all(Path(row["absolute_path"]).is_absolute() for row in rows)
    assert all(not Path(row["relative_path"]).is_absolute() for row in rows)


def test_cleanup_rejects_current_outside_explicit_parent(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = make_cleanup_tree(repo, (1, 2))
    other = repo / "runs/other/delivery_v2"
    other.mkdir(parents=True)
    pipeline = repo / "runs/pipeline"
    pipeline.mkdir(parents=True)
    with pytest.raises(registry.RegistryError, match="outside"):
        registry.cleanup_plan(
            repo, parent, {"delivery_dir": "runs/other/delivery_v2"}, pipeline,
        )


def test_cleanup_rejects_symlink_target_and_never_follows_it(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = make_cleanup_tree(repo, (2,))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "valuable.bin").write_bytes(b"do not count or touch")
    link = parent / "delivery_v1"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Windows may deny symlink creation outside Developer Mode.  Simulate
        # the same filesystem observations so the fail-closed branch remains
        # covered on those hosts.
        link.mkdir()
        outside_resolved = outside.resolve()
        original_is_symlink = Path.is_symlink
        original_resolve = Path.resolve

        def synthetic_is_symlink(path):
            return path == link or original_is_symlink(path)

        def synthetic_resolve(path, strict=False):
            if path == link:
                return outside_resolved
            return original_resolve(path, strict=strict)

        monkeypatch.setattr(Path, "is_symlink", synthetic_is_symlink)
        monkeypatch.setattr(Path, "resolve", synthetic_resolve)
    pipeline = repo / "runs/pipeline"
    pipeline.mkdir(parents=True)
    with pytest.raises(registry.RegistryError, match="escapes"):
        registry.cleanup_plan(repo, parent, current_for(repo, 2), pipeline)
    assert (outside / "valuable.bin").read_bytes() == b"do not count or touch"


def test_cleanup_ignores_orphan_build_and_unrelated_names(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = make_cleanup_tree(repo, (1, 2))
    (parent / "delivery_v99_build").mkdir()
    (parent / "delivery_latest").mkdir()
    pipeline = repo / "runs/pipeline"
    pipeline.mkdir(parents=True)
    plan = registry.cleanup_plan(repo, parent, current_for(repo, 2), pipeline, keep_previous=0)
    assert all(row["version"] != 99 for row in plan["candidates"] + plan["retained"])
    assert all("latest" not in row["relative_path"] for row in plan["candidates"])
