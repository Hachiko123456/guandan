"""06B P1: tie-aware ranking contract and every previously false-PASS path.

The old implementation selected a unique realized winner with a smallest-index
tie-break and a unique model winner the same way, then kept ``1/k`` as the
random baseline.  A deal whose realized returns were all equal therefore scored
accuracy=1 and delta=1-1/k from candidate order alone.  These tests are written
against the raw arrays and hand-computed values, never against a production
helper that could reproduce its own bug.
"""
from __future__ import annotations

import pytest

from guandan_marvel.phase6 import value_gate as vg
from guandan_marvel.phase6.value_gate import (
    MODEL_SCORE_TIE_ATOL, REALIZED_RETURN_TIE_ATOL, VALUE_RANKING_METRIC_VERSION,
    QualificationResult, RankingPoint, ValueGateError, ValueSample,
    compute_ranking_metrics, compute_tie_aware_ranking, evaluate_value_gate,
    revalidate_ranking_point, write_value_gate_registration,
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


def _sample(index, *, prediction, target, canonical, deal_id, phase, team):
    return ValueSample(
        sample_id=f"{deal_id}-{index}", deal_id=deal_id, canonical_deal=canonical,
        token_step=index, num_steps=30, player=team, team=team, level_rank=3,
        leader_seat=0, tribute_mode="none", prediction=prediction, target=target,
        game_phase=phase, outcome="HEAD_THIRD",
    )


def _point(canonical, index, *, model_scores, realized_returns, phase="mid",
           stored=None, version=VALUE_RANKING_METRIC_VERSION, candidate_count=None):
    k = len(model_scores)
    if stored is None:
        record = compute_tie_aware_ranking(model_scores, realized_returns)
        stored = {
            "accuracy": record["accuracy"], "random_baseline": record["random_baseline"],
            "delta": record["delta"], "informative": record["informative"],
            "all_target_tie": record["all_target_tie"], "model_tie": record["model_tie"],
            "target_top_count": record["target_top_count"],
            "model_top_count": record["model_top_count"],
        }
    return RankingPoint(
        deal_id=f"deal-{index}", canonical_deal=canonical, token_step=0, num_steps=30,
        game_phase=phase, history_consistent=True, player=index % 2,
        candidate_count=k if candidate_count is None else candidate_count,
        restricted=False, model_top_index=0, realized_top_index=0,
        model_scores=tuple(float(value) for value in model_scores),
        realized_returns=tuple(float(value) for value in realized_returns),
        candidate_token_sequences=tuple((tuple(range(8)),) for _ in range(k)),
        metric_version=version, **stored,
    )


def _result(points, *, registration, identity=None, samples=None, metadata=None):
    base = {
        "target": dict(TARGET), "clustering_unit": CLUSTERING, "truncated": False,
        "registration_hash": registration["registration_hash"],
        "design_hash": registration["design_hash"],
    }
    if identity is not None:
        base["checkpoint"] = dict(identity)
    if metadata:
        base.update(metadata)
    return QualificationResult(
        samples=list(samples or []), ranking_points=list(points), deal_records=[],
        metadata=base,
    )


def test_tie_tolerances_are_fixed_and_versioned():
    assert REALIZED_RETURN_TIE_ATOL == 1e-9
    assert MODEL_SCORE_TIE_ATOL == 1e-6
    schema = vg.VALUE_RANKING_METRIC_SCHEMA
    assert schema["version"] == VALUE_RANKING_METRIC_VERSION
    assert schema["tie_definition"]["realized_return"]["kind"] == "discrete_exact_tie"
    assert schema["tie_definition"]["model_score"]["kind"] == "continuous_near_tie"


def test_realized_all_tie_has_zero_delta_not_index_win():
    # Real evidence runs/phase6_repair1/value_gate/samples.json,
    # qualification-000000: k=2 and both realized returns are -1.0.  The old
    # code reported accuracy=1.0 against baseline=0.5.
    record = compute_tie_aware_ranking(
        [-0.0020379414781928062, -0.0024632406421005726], [-1.0, -1.0],
    )
    assert record["target_top_count"] == 2
    assert record["accuracy"] == 1.0
    assert record["random_baseline"] == 1.0
    assert record["delta"] == 0.0
    assert record["informative"] is False
    assert record["all_target_tie"] is True

    four = compute_tie_aware_ranking([0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0])
    assert four["accuracy"] == 1.0 and four["random_baseline"] == 1.0
    assert four["delta"] == 0.0 and four["informative"] is False


def test_model_all_tie_gets_baseline_not_index_advantage():
    # M = {0, 1}, T = {0}: uniform tie-breaking succeeds half the time and the
    # uniform candidate strategy also succeeds |T|/k = 1/2, so delta is exactly 0.
    record = compute_tie_aware_ranking([1.0, 1.0], [1.0, 0.0])
    assert record["model_top_count"] == 2
    assert record["accuracy"] == 0.5
    assert record["random_baseline"] == 0.5
    assert record["delta"] == 0.0
    assert record["informative"] is True
    assert record["model_tie"] is True

    # The old smallest-index rule returned accuracy=1, delta=+0.5 here.
    three = compute_tie_aware_ranking([2.0, 2.0, 2.0], [1.0, 0.0, 0.0])
    assert three["accuracy"] == pytest.approx(1 / 3)
    assert three["random_baseline"] == pytest.approx(1 / 3)
    assert three["delta"] == 0.0


def test_partial_ties_match_hand_computation():
    # M = {0,1}, T = {0}, k=3 -> 1/2 - 1/3 = 1/6.
    first = compute_tie_aware_ranking([1.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    assert first["accuracy"] == pytest.approx(0.5)
    assert first["random_baseline"] == pytest.approx(1 / 3)
    assert first["delta"] == pytest.approx(1 / 6)
    # M = {0,1}, T = {0,1}, k=3 -> 1 - 2/3 = 1/3; the old code said 1 - 1/3.
    second = compute_tie_aware_ranking([1.0, 1.0, 0.0], [1.0, 1.0, 0.0])
    assert second["accuracy"] == pytest.approx(1.0)
    assert second["random_baseline"] == pytest.approx(2 / 3)
    assert second["delta"] == pytest.approx(1 / 3)
    # M = {0}, T = {0,1,2}, k=4 -> 1 - 3/4 = 1/4 (not 1 - 1/4).
    third = compute_tie_aware_ranking([4.0, 3.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0])
    assert third["delta"] == 0.0 and third["informative"] is False
    # M = {0}, T = {0,1}, k=4 -> 1 - 2/4 = 1/2.
    fourth = compute_tie_aware_ranking([4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 0.0, 0.0])
    assert fourth["delta"] == pytest.approx(0.5)


def test_candidate_permutation_does_not_change_delta_or_verdict(tmp_path):
    registration = write_value_gate_registration(tmp_path / "registration.json")
    model = [0.9, 0.9, 0.1, 0.5]
    realized = [0.4, 0.4, 0.4, -1.0]
    permutations = [
        list(range(4)), [3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1],
    ]
    deltas = set()
    for order in permutations:
        point = _point(
            _canonical(0), 0,
            model_scores=[model[index] for index in order],
            realized_returns=[realized[index] for index in order],
        )
        metrics = compute_ranking_metrics([point], resamples=50)
        assert metrics["informative_points"] == 1
        deltas.add(round(float(metrics["delta"]), 12))
        gate = evaluate_value_gate(
            _result([point], registration=registration,
                    identity={"checkpoint_sha256": "a" * 64}),
            registration=registration,
            checkpoint_identities=[{"checkpoint_sha256": "a" * 64}],
        )
        assert gate["verdict"] == "INCONCLUSIVE"
    assert deltas == {0.25}


def test_legacy_real_evidence_is_recomputed_not_trusted():
    legacy = RankingPoint(
        deal_id="qualification-000000", canonical_deal=_canonical(1), token_step=193,
        num_steps=285, game_phase="late", history_consistent=True, player=0,
        candidate_count=2, restricted=False, model_top_index=0, realized_top_index=0,
        accuracy=1.0, random_baseline=0.5, delta=0.5,
        model_scores=(-0.0020379414781928062, -0.0024632406421005726),
        realized_returns=(-1.0, -1.0),
        candidate_token_sequences=((tuple(range(8)),), (tuple(range(8)),)),
        metric_version="guandan-marvel-phase6-value-ranking-point-v2",
    )
    fixed = revalidate_ranking_point(legacy)
    assert fixed.accuracy == 1.0
    assert fixed.random_baseline == 1.0
    assert fixed.delta == 0.0
    assert fixed.informative is False
    assert fixed.all_target_tie is True
    assert fixed.legacy_metrics_ignored is True
    metrics = compute_ranking_metrics([legacy], resamples=50)
    assert metrics["delta"] is None
    assert metrics["informative_points"] == 0
    assert metrics["legacy_metric_values_ignored"] == 1


def test_forged_serialized_accuracy_is_rejected():
    forged = _point(
        _canonical(2), 2, model_scores=[0.1, 0.2], realized_returns=[-1.0, -1.0],
        stored={
            "accuracy": 1.0, "random_baseline": 0.5, "delta": 0.5,
            "informative": True, "all_target_tie": False, "model_tie": False,
            "target_top_count": 1, "model_top_count": 1,
        },
    )
    with pytest.raises(ValueGateError, match="disagrees with the recomputed"):
        revalidate_ranking_point(forged)
    with pytest.raises(ValueGateError, match="disagrees with the recomputed"):
        compute_ranking_metrics([forged], resamples=10)


def test_unknown_metric_version_is_rejected():
    unknown = _point(
        _canonical(3), 3, model_scores=[1.0, 0.0], realized_returns=[1.0, 0.0],
        version="guandan-marvel-phase6-value-ranking-point-v9",
    )
    with pytest.raises(ValueGateError, match="unknown ranking metric version"):
        revalidate_ranking_point(unknown)


def test_all_target_tie_points_never_fill_ranking_sufficiency(tmp_path):
    registration = write_value_gate_registration(
        tmp_path / "registration.json", source_hash="s", config_hash="c",
    )
    # 64 mid/late points, every realized return tied.  The old serialized rows
    # claimed accuracy=1.0, baseline=0.5 (k=2) and would have filled the
    # min_ranking_points=64 threshold with a +0.5 delta.
    points = []
    for index in range(64):
        points.append(RankingPoint(
            deal_id=f"deal-{index}", canonical_deal=_canonical(index), token_step=0,
            num_steps=30, game_phase="mid" if index % 2 else "late",
            history_consistent=True, player=index % 2, candidate_count=2,
            restricted=False, model_top_index=0, realized_top_index=0,
            accuracy=1.0, random_baseline=0.5, delta=0.5,
            model_scores=(0.0, 0.0), realized_returns=(-1.0, -1.0),
            candidate_token_sequences=((tuple(range(8)),), (tuple(range(8)),)),
            metric_version="guandan-marvel-phase6-value-ranking-point-v2",
        ))
    samples = []
    for deal in range(64):
        canonical = _canonical(deal)
        for index in range(8):
            target = 0.1 * (index + 1) + (0.05 if index % 2 else -0.05)
            samples.append(_sample(
                index, prediction=target, target=target, canonical=canonical,
                deal_id=f"deal-{deal}", phase=("mid" if index < 6 else "late"),
                team=index % 2,
            ))
    gate = evaluate_value_gate(
        _result(
            points, registration=registration, samples=samples,
            identity={"checkpoint_sha256": "a" * 64},
        ),
        registration=registration,
        checkpoint_identities=[{"checkpoint_sha256": "a" * 64}],
    )
    assert gate["verdict"] == "INCONCLUSIVE"
    assert gate["verdict"] != "PASS"
    assert gate["verdict"] != "REJECT"
    assert gate["sufficiency"]["informative_ranking_points"] == 0
    assert gate["sufficiency"]["all_target_tie_ranking_points"] == 64
    assert gate["sufficiency"]["mid_late_informative_ranking_points"] == 0
    assert gate["ranking"]["delta"] is None
    joined = " | ".join(gate["reasons"])
    assert "all-target-tie ranking points carry zero ranking information" in joined
    assert "informative ranking points 0 < min_ranking_points=64" in joined


def test_model_tie_points_are_informative_with_zero_delta(tmp_path):
    registration = write_value_gate_registration(tmp_path / "registration.json")
    points = [
        _point(_canonical(index), index, model_scores=[1.0, 1.0, 1.0],
               realized_returns=[1.0, 0.0, 0.0])
        for index in range(4)
    ]
    metrics = compute_ranking_metrics(points, resamples=50)
    assert metrics["informative_points"] == 4
    assert metrics["model_tie_points"] == 4
    assert metrics["delta"] == 0.0
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["random_baseline"] == pytest.approx(1 / 3)


def test_single_candidate_and_invalid_inputs_are_explicit():
    single = compute_tie_aware_ranking([1.0], [1.0])
    assert single["informative"] is False
    assert single["delta"] == 0.0
    assert single["not_informative_reason"] == "single_candidate"
    with pytest.raises(ValueGateError, match="same length"):
        compute_tie_aware_ranking([1.0, 0.0], [1.0])
    with pytest.raises(ValueGateError, match="finite"):
        compute_tie_aware_ranking([float("nan"), 0.0], [1.0, 0.0])
    with pytest.raises(ValueGateError, match="finite"):
        compute_tie_aware_ranking([1.0, 0.0], [float("inf"), 0.0])
    with pytest.raises(ValueGateError, match="at least one candidate"):
        compute_tie_aware_ranking([], [])


def test_serialized_v3_payload_round_trips_the_contract():
    point = _point(_canonical(4), 4, model_scores=[1.0, 1.0], realized_returns=[1.0, 0.0])
    payload = point.payload()
    assert payload["schema"] == vg.VALUE_RANKING_POINT_SCHEMA
    assert payload["schema"].endswith("-v3")
    assert payload["metric_version"] == VALUE_RANKING_METRIC_VERSION
    assert payload["delta"] == 0.0
    assert payload["informative"] is True
    assert payload["model_tie"] is True


def test_polluted_result_and_unbound_identities_fail_closed(tmp_path):
    registration = write_value_gate_registration(tmp_path / "registration.json")
    samples = [
        _sample(0, prediction=0.1, target=0.2, canonical=_canonical(1),
                deal_id="deal-1", phase="early", team=0),
        _sample(0, prediction=0.3, target=0.4, canonical=_canonical(1),
                deal_id="deal-1", phase="mid", team=1),
    ]
    point = _point(_canonical(1), 0, model_scores=[1.0, 0.0], realized_returns=[1.0, 0.0])
    result = _result(
        [point, point], registration=registration, samples=samples,
        identity={
            "checkpoint_sha256": "a" * 64,
            "resolved_config_hash": "0" * 64,
            "architecture": "v1.4-macro-fuse",
            "protocol_version": "GD-ACTION-0.2-MARVEL",
            "backend": "cpp",
            "source_hash": "0" * 64,
            "state_dict_sha256": "0" * 64,
        },
        metadata={"sample_count": 999, "ranking_points": 1, "independent_deals": 7},
    )
    gate = evaluate_value_gate(
        result, registration=registration,
        checkpoint_identities=[{"checkpoint_sha256": "a" * 64}],
    )
    assert gate["verdict"] != "PASS"
    assert gate["sufficiency"]["result_pollution_ok"] is False
    assert gate["sufficiency"]["checkpoint_content_fingerprint_bound"] is False
    joined = " | ".join(gate["reasons"])
    assert "polluted result" in joined
    assert "producer is missing" in joined
    assert "pre-registration does not record a qualification producer source hash" in joined
    assert "state_dict_sha256" in joined
