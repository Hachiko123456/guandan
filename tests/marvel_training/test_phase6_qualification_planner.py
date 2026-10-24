"""06B P2/P3: qualification planner isolation, Cartesian context and budget.

The old planner cycled one absolute deal index through four lists at once, so
the "4 x 4 x 4" plan only ever produced the four diagonal combinations.  Its
``--execute`` path never loaded a protected manifest and fixed
``max_ranking_points=0``.  These tests pin the repaired planner semantics and
prove that an invalid manifest cannot start a model, engine or collector.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from guandan_marvel.evaluation.checkpoints import EVALUATION_CHECKPOINT_SCHEMA
from guandan_marvel.phase6.design import phase6_design_hash
from guandan_marvel.phase6.value_gate import (
    qualification_producer_source_hash, write_value_gate_registration,
)
from scripts.run_phase6_value_qualification import (
    ISOLATION_MANIFEST_SCHEMA, REQUIRED_PROTECTED_DOMAINS, QualificationPlanError,
    build_cartesian_deals, build_plan, execute_plan, load_isolation_manifest,
    ranking_budget, validate_isolation_manifest,
)

LIVE_PRODUCER = qualification_producer_source_hash()
CONFIG_HASH = "f" * 64
CHECKPOINT_HASH = "e" * 64


def _hands() -> list[list[int]]:
    cards = list(range(108))
    return [cards[index * 27:(index + 1) * 27] for index in range(4)]


def _fake_artifact(config_hash: str = CONFIG_HASH, sha: str = CHECKPOINT_HASH):
    identity = {
        "schema": "guandan-marvel-policy-identity-v1",
        "checkpoint_path": "memory://fake-eval.pt",
        "checkpoint_sha256": sha,
        "checkpoint_schema": EVALUATION_CHECKPOINT_SCHEMA,
        "algorithm": "ippo",
        "architecture": "v1.4-macro-fuse",
        "protocol_version": "GD-ACTION-0.2-MARVEL|GD-ENV-0.2-MARVEL|GD-ENCODING-0.2-MARVEL",
        "backend": "cpp",
        "backend_identity_hash": "9" * 64,
        "resolved_config_hash": config_hash,
        "source_hash": "1" * 64,
        "iteration": 1,
        "training_resume_allowed": False,
    }
    return SimpleNamespace(
        schema=EVALUATION_CHECKPOINT_SCHEMA,
        algorithm="ippo",
        backend="cpp",
        identity=identity,
        metadata={"training_resume_allowed": False, "source_hash": "1" * 64},
    )


def _manifest(**overrides) -> dict[str, object]:
    artifact = _fake_artifact()
    manifest: dict[str, object] = {
        "schema": ISOLATION_MANIFEST_SCHEMA,
        "created_at": "2026-10-20T00:00:00+00:00",
        "producer_source_hash": LIVE_PRODUCER,
        "design_hash": phase6_design_hash(),
        "checkpoint": {
            key: artifact.identity[key] for key in (
                "checkpoint_sha256", "resolved_config_hash", "architecture",
                "source_hash", "backend", "protocol_version",
            )
        },
        "domains": {
            "training": {
                "seed_domains": [[0, 999_999]],
                "canonical_deal_hashes": ["a" * 64],
            },
            "development": {
                "seed_domains": [[1_000_000, 1_999_999]],
                "canonical_deal_hashes": ["b" * 64],
            },
            "selection": {
                "seed_domains": [[2_100_000_000, 2_100_000_100]],
                "canonical_deal_hashes": ["c" * 64],
            },
            "final": {
                "seed_domains": [[3_000_000_000, 3_000_000_100]],
                "canonical_deal_hashes": ["d" * 64],
                "access_policy": "frozen_protection_only",
                # Deliberately nonexistent: qualification must never open it.
                "samples_path": "Z:/never/read/final_samples.json",
            },
        },
    }
    manifest.update(overrides)
    return manifest


def _write_manifest(tmp_path: Path, manifest: dict[str, object]) -> Path:
    path = tmp_path / "isolation_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture()
def registration(tmp_path):
    return write_value_gate_registration(
        tmp_path / "registration.json",
        source_hash=LIVE_PRODUCER, config_hash=CONFIG_HASH,
    )


def test_cartesian_context_plan_covers_all_joint_cells():
    plan_64 = build_plan(deals=64, wall_seconds=1200.0, dry_run=True)
    assert plan_64["context_plan"]["joint_cell_count"] == 64
    assert plan_64["context_plan"]["expected_joint_cell_count"] == 64
    assert len(plan_64["joint_context_cells"]) == 64
    assert set(plan_64["joint_context_cells"].values()) == {1}
    for family in ("level_rank", "leader_seat", "tribute_mode"):
        values = list(plan_64["context_counts"][family].values())
        assert len(values) == 4
        assert min(values) == 16 and max(values) == 16
    assert plan_64["planned_context_coverage_ok"] is True

    plan_128 = build_plan(deals=128, wall_seconds=1200.0, dry_run=True)
    assert len(plan_128["joint_context_cells"]) == 64
    assert set(plan_128["joint_context_cells"].values()) == {2}
    for family in ("level_rank", "leader_seat", "tribute_mode"):
        assert set(plan_128["context_counts"][family].values()) == {32}


def test_execute_plan_has_nonzero_mid_late_ranking_budget():
    budget = ranking_budget(128)
    assert budget["max_ranking_points"] > 0
    assert budget["max_ranking_points"] == 64
    assert budget["phases_required"] == ["mid", "late"]
    assert budget["min_informative_mid_points"] == 16
    assert budget["min_informative_late_points"] == 16
    assert budget["min_ranking_points"] == 64
    assert budget["max_candidates"] > 0 and budget["max_continuation_steps"] > 0
    plan = build_plan(deals=128, wall_seconds=1200.0, dry_run=False)
    assert plan["ranking_budget"]["max_ranking_points"] > 0
    assert plan["would_execute"] is False  # no isolation manifest
    assert plan["isolation"]["status"] == "MISSING_ISOLATION_MANIFEST"


def test_plan_with_valid_manifest_reports_isolation_but_dry_run_does_not_execute(tmp_path):
    manifest = _manifest()
    path = _write_manifest(tmp_path, manifest)
    plan = build_plan(
        deals=128, wall_seconds=1200.0, dry_run=True, manifest_path=path,
    )
    assert plan["isolation"]["status"] == "VALID"
    assert plan["isolation"]["domains"]["final"]["final_samples_read"] is False
    assert plan["would_execute"] is False
    plan_execute = build_plan(
        deals=128, wall_seconds=1200.0, dry_run=False, manifest_path=path,
    )
    assert plan_execute["status"] == "PLANNED"
    assert plan_execute["would_execute"] is True


def _run_with_manifest(
    tmp_path, manifest, *, checkpoint_loader=None, collector=None, registration_path=None,
    producer_hash_fn=None,
):
    path = _write_manifest(tmp_path, manifest)
    calls: list[str] = []

    def loader(*args, **kwargs):
        calls.append("checkpoint")
        return (checkpoint_loader or _fake_artifact)()

    def collect(*args, **kwargs):
        calls.append("collector")
        if collector is not None:
            return collector(*args, **kwargs)
        return SimpleNamespace(metadata={"schema": "fake"}, ranking_points=[])

    kwargs = {}
    if producer_hash_fn is not None:
        kwargs["producer_hash_fn"] = producer_hash_fn
    plan = build_plan(deals=128, wall_seconds=60.0, dry_run=False, manifest_path=path)
    record = execute_plan(
        plan=plan, deals=build_cartesian_deals(128),
        manifest_path=path, registration_path=registration_path or (tmp_path / "registration.json"),
        checkpoint_path="memory://fake-eval.pt", out_dir=tmp_path / "out",
        checkpoint_loader=loader, collector=collect, **kwargs,
    )
    return record, calls


def _assert_blocked(tmp_path, manifest, registration, *, producer_hash_fn=None):
    path = _write_manifest(tmp_path, manifest)
    registration_path = tmp_path / "registration.json"
    calls: list[str] = []
    with pytest.raises(QualificationPlanError):
        plan = build_plan(deals=128, wall_seconds=60.0, dry_run=False, manifest_path=path)
        execute_plan(
            plan=plan, deals=build_cartesian_deals(128),
            manifest_path=path, registration_path=registration_path,
            checkpoint_path="memory://fake-eval.pt", out_dir=tmp_path / "blocked",
            checkpoint_loader=lambda *a, **k: calls.append("checkpoint"),
            collector=lambda *a, **k: calls.append("collector"),
            **(dict(producer_hash_fn=producer_hash_fn) if producer_hash_fn else {}),
        )
    assert calls == [], "an invalid manifest must not start any model or collector"
    return calls


def test_valid_manifest_binds_identity_and_collects_with_budget(tmp_path, registration):
    record, calls = _run_with_manifest(
        tmp_path, _manifest(), registration_path=tmp_path / "registration.json",
    )
    assert registration["source_hash"] == LIVE_PRODUCER
    assert calls == ["checkpoint", "collector"]
    assert record["isolation"]["status"] == "VALID"
    assert record["ranking_budget"]["max_ranking_points"] == 64
    assert record["ranking_actual"]["insufficient_verdict"] == "INCONCLUSIVE"
    assert record["checkpoint"]["resolved_config_hash"] == CONFIG_HASH
    assert record["isolation"]["final_samples_read"] is False


def test_invalid_manifests_never_start_a_model_or_collector(tmp_path, registration):
    # missing required domain
    missing = _manifest()
    missing["domains"].pop("selection")
    _assert_blocked(tmp_path, missing, registration)

    # qualification seed lies inside a protected seed range
    overlap = _manifest()
    overlap["domains"]["training"]["seed_domains"] = [[2_000_000_000, 2_000_000_010]]
    _assert_blocked(tmp_path, overlap, registration)

    # canonical deck overlap across domains
    shared = _manifest()
    shared["domains"]["development"]["canonical_deal_hashes"] = ["a" * 64]
    _assert_blocked(tmp_path, shared, registration)

    # wrong producer source and wrong design identity
    _assert_blocked(
        tmp_path, _manifest(producer_source_hash="0" * 64), registration,
    )
    _assert_blocked(tmp_path, _manifest(design_hash="0" * 64), registration)

    # final holdout must stay frozen protection only
    final_read = _manifest()
    final_read["domains"]["final"]["access_policy"] = "read"
    _assert_blocked(tmp_path, final_read, registration)

    # no canonical evidence at all: disjoint seeds are not enough
    no_evidence = _manifest()
    no_evidence["domains"]["training"].pop("canonical_deal_hashes")
    _assert_blocked(tmp_path, no_evidence, registration)


def test_rotated_and_copy_alias_protected_deals_are_rejected(tmp_path, registration):
    hands = _hands()
    rotated = [hands[(seat + 1) % 4] for seat in range(4)]
    rotation = _manifest()
    rotation["domains"]["training"].pop("canonical_deal_hashes")
    rotation["domains"]["training"]["deals"] = [
        {"deal_id": "train-0", "seed": 10, "hands": hands},
        {"deal_id": "train-1", "seed": 11, "hands": rotated},
    ]
    _assert_blocked(tmp_path, rotation, registration)

    alias = _manifest()
    alias["domains"]["training"].pop("canonical_deal_hashes")
    alias["domains"]["training"]["deals"] = [
        {"deal_id": "train-0", "seed": 10, "hands": hands},
    ]
    alias["domains"]["development"]["deals"] = [
        {"deal_id": " TRAIN-0 ", "seed": 20, "hands": _hands()},
    ]
    alias["domains"]["development"].pop("canonical_deal_hashes")
    _assert_blocked(tmp_path, alias, registration)


def test_registration_source_and_checkpoint_identity_must_bind(tmp_path):
    stale = write_value_gate_registration(
        tmp_path / "stale_registration.json", source_hash="0" * 64, config_hash=CONFIG_HASH,
    )
    path = _write_manifest(tmp_path, _manifest())
    plan = build_plan(deals=128, wall_seconds=60.0, dry_run=False, manifest_path=path)
    calls: list[str] = []
    with pytest.raises(QualificationPlanError, match="different qualification source"):
        execute_plan(
            plan=plan, deals=build_cartesian_deals(128), manifest_path=path,
            registration_path=tmp_path / "stale_registration.json",
            checkpoint_path="memory://fake-eval.pt", out_dir=tmp_path / "stale",
            checkpoint_loader=lambda *a, **k: calls.append("checkpoint"),
            collector=lambda *a, **k: calls.append("collector"),
        )
    assert calls == []
    assert stale["source_hash"] == "0" * 64

    good = write_value_gate_registration(
        tmp_path / "good_registration.json",
        source_hash=LIVE_PRODUCER, config_hash=CONFIG_HASH,
    )
    with pytest.raises(QualificationPlanError, match="resolved config"):
        execute_plan(
            plan=plan, deals=build_cartesian_deals(128), manifest_path=path,
            registration_path=tmp_path / "good_registration.json",
            checkpoint_path="memory://fake-eval.pt", out_dir=tmp_path / "bad-config",
            checkpoint_loader=lambda *a, **k: _fake_artifact(config_hash="0" * 64),
            collector=lambda *a, **k: calls.append("collector"),
        )
    assert good["config_hash"] == CONFIG_HASH
    assert calls == []

    # Manifest checkpoint identity disagreeing with the loaded artifact fails.
    manifest = _manifest()
    manifest["checkpoint"]["checkpoint_sha256"] = "0" * 64
    path = _write_manifest(tmp_path, manifest)
    plan = build_plan(deals=128, wall_seconds=60.0, dry_run=False, manifest_path=path)
    with pytest.raises(QualificationPlanError, match="checkpoint identity"):
        execute_plan(
            plan=plan, deals=build_cartesian_deals(128), manifest_path=path,
            registration_path=tmp_path / "good_registration.json",
            checkpoint_path="memory://fake-eval.pt", out_dir=tmp_path / "bad-identity",
            checkpoint_loader=lambda *a, **k: _fake_artifact(),
            collector=lambda *a, **k: calls.append("collector"),
        )
    assert calls == []


def test_missing_or_malformed_manifest_fails_closed(tmp_path, registration):
    with pytest.raises(QualificationPlanError, match="missing"):
        load_isolation_manifest(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(QualificationPlanError, match="not readable JSON"):
        load_isolation_manifest(broken)
    wrong_schema = _manifest(schema="other")
    _assert_blocked(tmp_path, wrong_schema, registration)


def test_manifest_validation_is_deterministic_and_side_effect_free(tmp_path):
    manifest = _manifest()
    path = _write_manifest(tmp_path, manifest)
    deals = build_cartesian_deals(128)
    first = validate_isolation_manifest(
        load_isolation_manifest(path), deals=deals,
        expected_producer_source_hash=LIVE_PRODUCER,
        expected_design_hash=phase6_design_hash(),
    )
    second = validate_isolation_manifest(
        copy.deepcopy(manifest), deals=deals,
        expected_producer_source_hash=LIVE_PRODUCER,
        expected_design_hash=phase6_design_hash(),
    )
    assert first == second
    assert set(first["domains"]) == set(REQUIRED_PROTECTED_DOMAINS)
    assert first["protected_canonical_hashes"] == 4
    assert first["status"] == "VALID"
