"""Phase-6 value qualification: metrics, clustering, pre-registration and gate."""
from __future__ import annotations

import copy
import json

import pytest
import torch

from guandan_marvel.phase6 import value_gate as vg
from guandan_marvel.phase6.design import (
    PHASE6_DESIGN_SCHEMA, SEARCH_IMPLEMENTED, SEARCH_STATUS_IMPLEMENTED,
    SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE, SEARCH_STATUS_SKIPPED_VALUE_REJECTED,
    canonical_hash, search_status_for_verdict,
)
from guandan_marvel.phase6.value_gate import (
    VALUE_RANKING_METRIC_VERSION, QualificationResult, RankingPoint, ValueGateError,
    ValueSample, assert_qualification_isolation, assert_search_allowed,
    build_qualification_deals, canonical_physical_deal, compute_ranking_metrics,
    compute_value_metrics, evaluate_value_gate, load_value_gate,
    load_value_gate_registration, write_value_gate_registration,
)

TARGET_CONTRACT = {
    "perspective": "acting_player",
    "scale": "rank_point_over_3",
    "discount_gamma": 1.0,
    "raw_reward_scale": 1 / 3,
}
CLUSTERING = "canonical_physical_deal_rotations_and_deck_swap"


def _sample(
    index: int,
    *,
    prediction: float,
    target: float,
    canonical: str = "a" * 64,
    deal_id: str = "deal-a",
    phase: str = "early",
    outcome: str = "HEAD_THIRD",
    team: int = 0,
) -> ValueSample:
    return ValueSample(
        sample_id=f"{deal_id}-{index}",
        deal_id=deal_id,
        canonical_deal=canonical,
        token_step=index,
        num_steps=30,
        player=team,
        team=team,
        level_rank=3,
        leader_seat=0,
        tribute_mode="none",
        prediction=prediction,
        target=target,
        game_phase=phase,
        outcome=outcome,
    )


def _result(
    samples, *, ranking=(), metadata=None, registration=None, identity=None,
) -> QualificationResult:
    base = {
        "target": dict(TARGET_CONTRACT),
        "clustering_unit": CLUSTERING,
        "truncated": False,
    }
    if registration is not None:
        base["registration_hash"] = registration["registration_hash"]
        base["design_hash"] = registration["design_hash"]
    if identity is not None:
        base["checkpoint"] = dict(identity)
    if metadata:
        base.update(metadata)
    return QualificationResult(
        samples=list(samples),
        ranking_points=list(ranking),
        deal_records=[],
        metadata=base,
    )


def _point(canonical: str, accuracy: float, *, candidates: int = 4, restricted=False) -> RankingPoint:
    """A point whose raw arrays encode the requested (informative) accuracy."""
    model_scores = [1.0] + [0.0] * (candidates - 1)
    if accuracy:
        realized = [1.0] + [0.0] * (candidates - 1)
    else:
        realized = [0.0, 1.0] + [0.0] * max(0, candidates - 2)
    return RankingPoint(
        deal_id=f"deal-{canonical[:6]}",
        canonical_deal=canonical,
        token_step=0,
        player=0,
        candidate_count=candidates,
        restricted=restricted,
        model_top_index=0,
        realized_top_index=0 if accuracy else 1,
        accuracy=float(accuracy),
        random_baseline=1.0 / candidates,
        delta=float(accuracy) - 1.0 / candidates,
        informative=True,
        all_target_tie=False,
        model_tie=False,
        target_top_count=1,
        model_top_count=1,
        metric_version=VALUE_RANKING_METRIC_VERSION,
        model_scores=tuple(model_scores),
        realized_returns=tuple(realized),
        candidate_token_sequences=tuple((tuple(range(8)),) for _ in range(candidates)),
    )


def test_zero_target_variance_is_undefined_not_epsilon_filled():
    samples = [
        _sample(0, prediction=0.1, target=0.5),
        _sample(1, prediction=0.9, target=0.5),
        _sample(2, prediction=-0.3, target=0.5),
    ]
    metrics = compute_value_metrics(samples)
    assert metrics["overall"]["zero_target_variance_count"] == 1
    assert metrics["overall"]["ev"] is None
    assert metrics["overall"]["pearson"] is None


def test_wrong_scale_or_perspective_is_rejected_before_the_verdict(tmp_path):
    registration = write_value_gate_registration(tmp_path / "registration.json")
    samples = [_sample(0, prediction=0.1, target=0.2)]
    for wrong in (
        {"scale": "raw_rank_point"},
        {"perspective": "team"},
        {"discount_gamma": 0.99},
        {"raw_reward_scale": 1.0},
    ):
        metadata = {"target": {**TARGET_CONTRACT, **wrong}}
        with pytest.raises(ValueGateError, match="scale/perspective/discount"):
            evaluate_value_gate(
                _result(
                    samples, metadata=metadata, registration=registration,
                    identity={"checkpoint_sha256": "a" * 64},
                ),
                registration=registration,
                checkpoint_identities=[{"checkpoint_sha256": "a" * 64}],
            )
    with pytest.raises(ValueGateError, match="clustered by canonical deal"):
        evaluate_value_gate(
            _result(
                samples, metadata={"clustering_unit": "step"}, registration=registration,
                identity={"checkpoint_sha256": "a" * 64},
            ),
            registration=registration,
            checkpoint_identities=[{"checkpoint_sha256": "a" * 64}],
        )


def test_canonical_deal_clustering_identifies_rotations_and_deck_swaps():
    hands = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    # Extend to a real 108-card conservation for the canonical helper.
    remaining = [card for card in range(12, 108)]
    hands = [list(hand) for hand in hands]
    for index, card in enumerate(remaining):
        hands[index % 4].append(card)
    base = canonical_physical_deal(hands)
    rotated = [hands[(seat + 2) % 4] for seat in range(4)]
    swap = [[(card + 54) % 108 for card in hand] for hand in hands]
    assert canonical_physical_deal(rotated) == base
    assert canonical_physical_deal(swap) == base
    changed = [list(hand) for hand in hands]
    changed[0][0], changed[1][0] = changed[1][0], changed[0][0]
    assert canonical_physical_deal(changed) != base

    samples = [
        _sample(0, prediction=0.1, target=0.2, canonical=base, deal_id="deal-a"),
        _sample(1, prediction=0.2, target=0.3, canonical=base, deal_id="deal-b"),
    ]
    metrics = compute_value_metrics(samples)
    assert metrics["overall"]["sample_count"] == 2
    assert metrics["overall"]["independent_deal_count"] == 1


def test_ranking_random_baseline_and_clustered_delta():
    points = [
        _point("a" * 64, 1.0, candidates=4),
        _point("b" * 64, 1.0, candidates=4),
        _point("c" * 64, 0.0, candidates=4),
    ]
    metrics = compute_ranking_metrics(points, resamples=200)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["random_baseline"] == pytest.approx(0.25)
    assert metrics["delta"] == pytest.approx(2 / 3 - 0.25)
    assert metrics["ci"]["low"] is not None
    assert metrics["independent_deals"] == 3


def test_insufficient_evidence_is_inconclusive_and_search_is_not_built(tmp_path):
    registration = write_value_gate_registration(tmp_path / "registration.json")
    samples = [
        _sample(0, prediction=0.1, target=0.3),
        _sample(1, prediction=0.2, target=0.2),
        _sample(2, prediction=0.4, target=-0.1),
    ]
    result = _result(
        samples, registration=registration,
        identity={"checkpoint_sha256": "a" * 64},
    )
    gate = evaluate_value_gate(
        result, registration=registration,
        checkpoint_identities=[{"checkpoint_sha256": "a" * 64}],
    )
    assert gate["verdict"] == "INCONCLUSIVE"
    assert gate["search_status"] == SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE
    assert gate["sufficiency"]["independent_deals"] == 1
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
    loaded = load_value_gate(path, require_pass=False)
    assert loaded["verdict"] == "INCONCLUSIVE"
    with pytest.raises(ValueGateError, match="not implemented"):
        assert_search_allowed(path)


def test_reject_artifact_never_enables_search(tmp_path):
    payload = {
        "schema": vg.VALUE_GATE_SCHEMA,
        "verdict": "REJECT",
        "reasons": ["EV CI upper below threshold"],
        "registration_hash": "a" * 64,
        "design_hash": "b" * 64,
        "design_schema": PHASE6_DESIGN_SCHEMA,
        "checkpoint_identities": [],
        "dataset": {},
        "metrics": {},
        "ev_clustered_ci": {},
        "ranking": {},
        "sufficiency": {},
        "search_status": SEARCH_STATUS_SKIPPED_VALUE_REJECTED,
        "created_at": "2026-10-20T00:00:00+00:00",
    }
    from guandan_marvel.phase6.design import phase6_design_hash

    payload["design_hash"] = phase6_design_hash()
    payload["gate_hash"] = canonical_hash(payload)
    path = tmp_path / "reject.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    loaded = load_value_gate(path, require_pass=False)
    assert loaded["search_status"] == SEARCH_STATUS_SKIPPED_VALUE_REJECTED
    with pytest.raises(ValueGateError, match="not implemented"):
        assert_search_allowed(path)
    assert SEARCH_IMPLEMENTED is False
    assert search_status_for_verdict("INCONCLUSIVE") == SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE
    assert search_status_for_verdict("REJECT") == SEARCH_STATUS_SKIPPED_VALUE_REJECTED
    # A PASS is only VALUE_QUALIFIED while search is not implemented.
    assert search_status_for_verdict("PASS") == (
        "VALUE_QUALIFIED_SEARCH_NOT_IMPLEMENTED"
    )
    assert search_status_for_verdict("PASS") != SEARCH_STATUS_IMPLEMENTED


def test_registration_is_pre_registered_and_tamper_evident(tmp_path):
    first = write_value_gate_registration(
        tmp_path / "registration.json", source_hash="source", config_hash="config",
    )
    second = write_value_gate_registration(
        tmp_path / "registration.json", source_hash="source", config_hash="config",
    )
    assert first == second
    path = tmp_path / "registration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["value_gate_plan"]["metrics"]["ev_min"] = 0.0
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueGateError, match="thresholds were edited"):
        load_value_gate_registration(path)

    other = copy.deepcopy(first)
    other["design_hash"] = "0" * 64
    other["registration_hash"] = canonical_hash(
        {key: value for key, value in other.items() if key != "registration_hash"}
    )
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueGateError, match="design hash"):
        load_value_gate_registration(other_path)


def test_qualification_deals_are_isolated_and_bounded():
    deals = build_qualification_deals(16, base_seed=2_000_000_000)
    assert len(deals) == 16
    assert {deal.seed for deal in deals} == set(range(2_000_000_000, 2_000_000_016))
    protected = [
        {"seed": 100_000, "domain": "training"},
        {"seed": 1_000_000_000, "domain": "selection"},
        {"seed": 1_500_000_000, "domain": "final"},
    ]
    report = assert_qualification_isolation(deals, protected_rows=protected)
    assert report["seed_overlap"] == 0
    with pytest.raises(ValueGateError, match="overlap"):
        assert_qualification_isolation(
            deals, protected_rows=[{"seed": 2_000_000_003, "domain": "training"}],
        )


def test_search_entry_refuses_an_inconclusive_gate(tmp_path):
    from guandan_marvel.phase6.design import phase6_design_hash
    from guandan_marvel.phase6.search import SEARCH_IMPLEMENTED, require_search_allowed

    assert SEARCH_IMPLEMENTED is False
    payload = {
        "schema": vg.VALUE_GATE_SCHEMA,
        "verdict": "INCONCLUSIVE",
        "reasons": ["insufficient independent deals"],
        "registration_hash": "a" * 64,
        "design_hash": phase6_design_hash(),
        "design_schema": PHASE6_DESIGN_SCHEMA,
        "checkpoint_identities": [],
        "dataset": {},
        "metrics": {},
        "ev_clustered_ci": {},
        "ranking": {},
        "sufficiency": {},
        "search_status": SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE,
        "created_at": "2026-10-20T00:00:00+00:00",
    }
    payload["gate_hash"] = canonical_hash(payload)
    path = tmp_path / "inconclusive.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueGateError, match="not implemented"):
        require_search_allowed(path)
