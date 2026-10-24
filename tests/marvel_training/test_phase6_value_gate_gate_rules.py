"""Phase-6 repair: the value gate cannot PASS on biased/incomplete slices."""
from __future__ import annotations

import json

import pytest

from guandan_marvel.phase6.design import (
    PHASE6_DESIGN_SCHEMA, VALUE_GATE_PLAN, canonical_hash, phase6_design_hash,
)
from guandan_marvel.phase6 import value_gate as vg
from guandan_marvel.phase6.value_gate import (
    QualificationResult, RankingPoint, ValueGateError, ValueSample,
    _slice_coverage_problems, assert_search_allowed, evaluate_value_gate,
    load_value_gate, write_value_gate_registration,
)

TARGET = {
    "perspective": "acting_player",
    "scale": "rank_point_over_3",
    "discount_gamma": 1.0,
    "raw_reward_scale": 1 / 3,
}
CLUSTERING = "canonical_physical_deal_rotations_and_deck_swap"


def _canonical(index: int) -> str:
    return f"{index:064x}"


def _sample(index, *, prediction, target, canonical, team, deal_id, phase="early"):
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
        outcome="HEAD_THIRD",
    )


def _point(index, *, accuracy, phase="early"):
    model_scores = [1.0, 0.0, 0.0, 0.0]
    realized = [1.0, 0.0, 0.0, 0.0] if accuracy else [0.0, 1.0, 0.0, 0.0]
    return RankingPoint(
        deal_id=f"deal-{index}",
        canonical_deal=_canonical(index),
        token_step=0,
        num_steps=30,
        game_phase=phase,
        history_consistent=True,
        player=0,
        candidate_count=4,
        restricted=False,
        model_top_index=0,
        realized_top_index=0 if accuracy else 1,
        accuracy=accuracy,
        random_baseline=0.25,
        delta=float(accuracy) - 0.25,
        informative=True,
        all_target_tie=False,
        model_tie=False,
        target_top_count=1,
        model_top_count=1,
        metric_version=vg.VALUE_RANKING_METRIC_VERSION,
        model_scores=tuple(model_scores),
        realized_returns=tuple(realized),
        candidate_token_sequences=tuple((tuple(range(8)),) for _ in range(4)),
    )


def _result(samples, points, *, registration=None, identity=None):
    metadata = {
        "target": dict(TARGET),
        "clustering_unit": CLUSTERING,
        "truncated": False,
    }
    if registration is not None:
        metadata["registration_hash"] = registration["registration_hash"]
        metadata["design_hash"] = registration["design_hash"]
    if identity is not None:
        metadata["checkpoint"] = dict(identity)
    return QualificationResult(
        samples=list(samples), ranking_points=list(points), deal_records=[],
        metadata=metadata,
    )


def _identities():
    return [
        {"checkpoint_sha256": "a" * 64, "iteration": 1},
        {"checkpoint_sha256": "b" * 64, "iteration": 2},
    ]


def _bias_samples():
    """Reviewer 1.1: overall early EV=1/3, team 1 undefined, ranking perfect."""
    import math

    noise = [-0.18, 0.18] * 2
    samples = []
    for deal in range(64):
        canonical = _canonical(deal)
        for index in range(4):
            target = 1.0 if index % 2 == 0 else -1.0
            samples.append(_sample(
                index, prediction=target + noise[index], target=target,
                canonical=canonical, team=0, deal_id=f"deal-{deal}",
            ))
        for index in range(4, 8):
            # team 1: constant target and prediction -> EV undefined, zero bias.
            samples.append(_sample(
                index, prediction=0.5, target=0.5,
                canonical=canonical, team=1, deal_id=f"deal-{deal}",
            ))
    return samples


def test_seat_bias_and_all_early_slices_are_inconclusive(tmp_path):
    registration = write_value_gate_registration(tmp_path / "registration.json")
    points = [_point(index, accuracy=1.0) for index in range(64)]
    gate = evaluate_value_gate(
        _result(
            _bias_samples(), points, registration=registration,
            identity={"checkpoint_sha256": "a" * 64},
        ),
        registration=registration,
        checkpoint_identities=_identities(),
    )
    assert gate["verdict"] == "INCONCLUSIVE"
    assert gate["verdict"] != "PASS"
    reasons = " | ".join(gate["reasons"])
    assert "mid-game ranking point" in reasons or "late-game ranking point" in reasons
    assert "seat_team=team_1" in reasons or "team_1 EV is undefined" in reasons
    assert "no verifiable checkpoint path" in reasons
    assert gate["sufficiency"]["slice_coverage_ok"] is False
    assert gate["sufficiency"]["mid_late_ranking_points"] == 0


def test_zero_target_variance_with_sufficient_counts_is_inconclusive_not_reject(tmp_path):
    registration = write_value_gate_registration(tmp_path / "registration.json")
    samples = []
    for deal in range(64):
        canonical = _canonical(deal)
        for index in range(8):
            samples.append(_sample(
                index, prediction=0.1 * index, target=0.5,
                canonical=canonical, team=index % 2, deal_id=f"deal-{deal}",
            ))
    points = [_point(index, accuracy=0.0) for index in range(64)]
    gate = evaluate_value_gate(
        _result(
            samples, points, registration=registration,
            identity={"checkpoint_sha256": "a" * 64},
        ),
        registration=registration,
        checkpoint_identities=_identities(),
    )
    assert gate["verdict"] == "INCONCLUSIVE"
    assert gate["verdict"] != "REJECT"
    assert any("zero_target_variance" in reason for reason in gate["reasons"])


def test_slice_coverage_helper_enforces_min_deals_gaps_and_mid_late():
    metrics = {
        "slices": {
            "game_phase": {
                "early": {"independent_deal_count": 64, "ev": 0.2},
                "mid": {"independent_deal_count": 16, "ev": 0.2},
                "late": {"independent_deal_count": 16, "ev": 0.2},
            },
            "seat_team": {
                "team_0": {"independent_deal_count": 64, "ev": 0.3},
                "team_1": {"independent_deal_count": 64, "ev": 0.0},
            },
            "outcome": {
                "DOUBLE_DOWN": {"independent_deal_count": 16, "ev": 0.2},
                "HEAD_THIRD": {"independent_deal_count": 16, "ev": 0.2},
                "HEAD_LAST": {"independent_deal_count": 16, "ev": 0.2},
            },
            "level_rank": {"level_3": {"independent_deal_count": 64, "ev": 0.2}},
            "leader_seat": {"leader_0": {"independent_deal_count": 64, "ev": 0.2}},
            "tribute_mode": {"none": {"independent_deal_count": 64, "ev": 0.2}},
        }
    }
    points = [_point(index, accuracy=1.0) for index in range(32)]
    problems, mid_late, gaps = _slice_coverage_problems(metrics, points, VALUE_GATE_PLAN)
    joined = " | ".join(problems)
    assert "seat_team=team_1 EV 0 is not positive" in joined
    assert "seat_team EV gap" in joined
    assert "no informative mid-game ranking point" in joined
    assert mid_late["ranking_points"] == 0
    assert gaps["seat_team_ev_gap"] > float(
        VALUE_GATE_PLAN["slices"]["max_seat_ev_gap_for_pass"]
    )

    metrics["slices"]["game_phase"]["late"]["independent_deal_count"] = 15
    metrics["slices"]["seat_team"]["team_1"]["ev"] = 0.29
    problems, _mid_late, _gaps = _slice_coverage_problems(metrics, points, VALUE_GATE_PLAN)
    joined = " | ".join(problems)
    assert "late has 15 independent deals" in joined
    assert "min_slice_deals=16" in joined


def test_search_remains_rejected_even_for_a_sealed_pass(tmp_path):
    payload = {
        "schema": vg.VALUE_GATE_SCHEMA,
        "verdict": "PASS",
        "reasons": [],
        "registration_hash": "a" * 64,
        "design_hash": phase6_design_hash(),
        "design_schema": PHASE6_DESIGN_SCHEMA,
        "checkpoint_identities": [],
        "dataset": {},
        "metrics": {},
        "ev_clustered_ci": {},
        "ranking": {},
        "sufficiency": {},
        "search_status": "VALUE_QUALIFIED_SEARCH_NOT_IMPLEMENTED",
        "created_at": "2026-10-20T00:00:00+00:00",
    }
    payload["gate_hash"] = canonical_hash(payload)
    path = tmp_path / "forged-pass.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    loaded = load_value_gate(path, require_pass=True)
    assert loaded["verdict"] == "PASS"
    with pytest.raises(ValueGateError, match="not implemented"):
        assert_search_allowed(path)
    from guandan_marvel.phase6.search import require_search_allowed

    with pytest.raises(ValueGateError, match="not implemented"):
        require_search_allowed(path)
