from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from guandan.state import PreviousHandResult
from guandan_marvel.evaluation.checkpoints import load_legacy_v1_opponent, load_v2_policy
from guandan_marvel.evaluation.evaluator import build_evaluation_plan
from guandan_marvel.evaluation.host import seat_assignment
from guandan_marvel.evaluation.manifests import build_deals_manifest, validate_deals_manifest


def _manifest():
    return build_deals_manifest(
        counts={"training": 4, "selection": 3, "final": 2},
        level_ranks=(3, 5, 15),
        leader_seats=(0, 1, 2, 3),
        previous_results=(None, PreviousHandResult((1, 4, 3, 2), 0, "HEAD_THIRD")),
    )


def test_deals_manifest_has_disjoint_training_selection_final_seeds():
    manifest = _manifest()
    validate_deals_manifest(manifest)
    domains = manifest["datasets"]
    seed_sets = {name: {row["seed"] for row in rows} for name, rows in domains.items()}
    assert seed_sets["training"].isdisjoint(seed_sets["selection"])
    assert seed_sets["training"].isdisjoint(seed_sets["final"])
    assert seed_sets["selection"].isdisjoint(seed_sets["final"])


def test_deals_manifest_rejects_overlap_even_with_recomputed_hash():
    from guandan_marvel.evaluation.common import payload_hash
    manifest = _manifest()
    manifest["datasets"]["final"][0]["seed"] = manifest["datasets"]["selection"][0]["seed"]
    manifest["hash"] = payload_hash(manifest)
    with pytest.raises(ValueError, match="overlap"):
        validate_deals_manifest(manifest)


def test_four_seat_rotation_covers_primary_role_and_team_parity():
    primary = []
    teams = []
    for rotation in range(4):
        mapping, policies = seat_assignment(rotation, "champion")
        primary.append(mapping[0])
        candidate_seats = [seat for seat, value in enumerate(policies) if value == "candidate"]
        assert candidate_seats == sorted([rotation, (rotation + 2) % 4])
        assert {seat % 2 for seat in candidate_seats} == {rotation % 2}
        teams.append(rotation % 2)
    assert sorted(primary) == [0, 1, 2, 3]
    assert teams == [0, 1, 0, 1]


def test_checkpoint_identity_and_protocol_algorithm_mismatches_rejected(v2_checkpoint, tmp_path):
    valid = load_v2_policy(v2_checkpoint, expected_algorithm="ippo")
    assert valid.backend == "cpp" and valid.architecture == "v1.3"
    payload = torch.load(v2_checkpoint, map_location="cpu", weights_only=False)

    bad_algorithm = copy.deepcopy(payload)
    bad_algorithm["algorithm"] = "vrpo"
    path = tmp_path / "bad-algorithm.pt"
    torch.save(bad_algorithm, path)
    with pytest.raises(ValueError, match="algorithm mismatch"):
        load_v2_policy(path, expected_algorithm="ippo")

    bad_protocol = copy.deepcopy(payload)
    bad_protocol["protocol_version"] = "GD-ACTION-9"
    path = tmp_path / "bad-protocol.pt"
    torch.save(bad_protocol, path)
    with pytest.raises(ValueError, match="protocol mismatch"):
        load_v2_policy(path)

    bad_identity = copy.deepcopy(payload)
    bad_identity["backend_identity"]["implementation"]["extension_sha256"] = "0" * 64
    path = tmp_path / "bad-identity.pt"
    torch.save(bad_identity, path)
    with pytest.raises(ValueError, match="hash.*inconsistent|identity mismatch"):
        load_v2_policy(path)


def test_legacy_v1_is_opponent_only_and_cannot_enter_v2_loader(legacy_checkpoint):
    legacy = load_legacy_v1_opponent(legacy_checkpoint)
    assert legacy.identity["usage"] == "opponent_only"
    assert legacy.identity["training_resume_allowed"] is False
    with pytest.raises(ValueError, match="opponent-only"):
        load_v2_policy(legacy_checkpoint)


def test_selection_and_final_plans_cannot_swap_deal_domains(v2_checkpoint, legacy_checkpoint):
    manifest = _manifest()
    candidate = load_v2_policy(v2_checkpoint)
    legacy = load_legacy_v1_opponent(legacy_checkpoint)
    pool = {"checkpoint_sha256": candidate.sha256, "checkpoint": str(v2_checkpoint), "id": "pool"}
    with pytest.raises(ValueError, match="selection deals"):
        build_evaluation_plan(
            mode="selection", deals_manifest=manifest, deal_domain="final",
            candidate_identity=candidate.identity, champion_identity=candidate.identity,
            legacy_identity=legacy.identity, pool_sample=pool, temperature=0,
        )
    with pytest.raises(ValueError, match="final deals"):
        build_evaluation_plan(
            mode="final_audit", deals_manifest=manifest, deal_domain="selection",
            candidate_identity=candidate.identity, champion_identity=candidate.identity,
            legacy_identity=legacy.identity, pool_sample=pool, temperature=0,
        )
