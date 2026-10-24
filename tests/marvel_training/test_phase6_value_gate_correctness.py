"""Repair round 2: context zero-variance, weight identity and result binding."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from guandan_marvel.evaluation.checkpoints import freeze_merged_policy, load_v2_policy
from guandan_marvel.phase6 import value_gate as vg
from guandan_marvel.phase6.value_gate import (
    QualificationResult, RankingPoint, ValueGateError, ValueSample,
    _verify_gate_checkpoint_identities, evaluate_value_gate,
    write_value_gate_registration,
)
from guandan_marvel.training.local import IPPOTrainer
from guandan_marvel.training.smoke import CppSmokeGame

from tests.marvel_training._phase6_pool_fixtures import tiny_config

TARGET = {
    "perspective": "acting_player",
    "scale": "rank_point_over_3",
    "discount_gamma": 1.0,
    "raw_reward_scale": 1 / 3,
}
CLUSTERING = "canonical_physical_deal_rotations_and_deck_swap"
PHASES = ("early", "mid", "late")
OUTCOMES = ("DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST")


def _canonical(index: int) -> str:
    return f"{index:064x}"


def _checkpoint(root: Path, seed: int):
    game = CppSmokeGame(512)
    cfg = tiny_config(game, seed=seed, run_name=f"phase6-repair2-{seed}")
    trainer = IPPOTrainer(cfg, game=game, run_dir=Path(root) / f"trainer-{seed}")
    trainer.train_iteration()
    path = freeze_merged_policy(trainer, Path(root) / f"eval-{seed}.pt")
    artifact = load_v2_policy(path, device="cpu", expected_algorithm="ippo")
    return {"path": path, "artifact": artifact, "identity": artifact.identity, "trainer": trainer}


def _sample(index, *, prediction, target, canonical, team, deal_id, phase, outcome, level):
    return ValueSample(
        sample_id=f"{deal_id}-{index}", deal_id=deal_id, canonical_deal=canonical,
        token_step=index, num_steps=30, player=team, team=team,
        level_rank=level, leader_seat=0, tribute_mode="none",
        prediction=prediction, target=target, game_phase=phase, outcome=outcome,
    )


def _samples(*, level5_zero: bool):
    samples = []
    for deal in range(64):
        canonical = _canonical(deal)
        phase = PHASES[deal % 3]
        outcome = OUTCOMES[deal % 3]
        team = deal % 2
        level = 5 if level5_zero and deal >= 48 else 3
        for index in range(8):
            if level == 5:
                target = prediction = 0.0
            else:
                target = 0.1 * (index + 1) + (0.05 if index % 2 else -0.05)
                prediction = target
            samples.append(_sample(
                index, prediction=prediction, target=target,
                canonical=canonical, team=team, deal_id=f"deal-{deal}",
                phase=phase, outcome=outcome, level=level,
            ))
    return samples


def _points():
    return [
        RankingPoint(
            deal_id=f"deal-{deal}", canonical_deal=_canonical(deal),
            token_step=0, num_steps=30, game_phase=PHASES[deal % 3],
            history_consistent=True, player=deal % 2,
            candidate_count=4, restricted=False,
            model_top_index=0, realized_top_index=0, accuracy=1.0,
            random_baseline=0.25, delta=0.75, informative=True,
            all_target_tie=False, model_tie=False,
            target_top_count=1, model_top_count=1,
            metric_version=vg.VALUE_RANKING_METRIC_VERSION,
            model_scores=(1.0, 0.0, 0.0, 0.0),
            realized_returns=(1.0, 0.0, 0.0, 0.0),
            candidate_token_sequences=tuple((tuple(range(8)),) for _ in range(4)),
        )
        for deal in range(64)
    ]


def _result(samples, points, *, registration, identity):
    return QualificationResult(
        samples=list(samples), ranking_points=list(points), deal_records=[],
        metadata={
            "target": dict(TARGET),
            "clustering_unit": CLUSTERING,
            "truncated": False,
            "registration_hash": registration["registration_hash"],
            "design_hash": registration["design_hash"],
            "checkpoint": dict(identity),
        },
    )


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase6-repair2")
    return {
        "a": _checkpoint(root, 707),
        "b": _checkpoint(root, 708),
        "c": _checkpoint(root, 709),
    }


def test_context_zero_variance_is_reported_not_excluded(checkpoints, tmp_path):
    registration = write_value_gate_registration(tmp_path / "registration.json")
    result = _result(
        _samples(level5_zero=True), _points(),
        registration=registration, identity=checkpoints["a"]["identity"],
    )
    gate = evaluate_value_gate(
        result, registration=registration,
        checkpoint_identities=[
            checkpoints["a"]["identity"], checkpoints["b"]["identity"],
        ],
    )
    assert gate["verdict"] != "PASS"
    reasons = " | ".join(gate["reasons"])
    assert "level_rank=level_5" in reasons
    assert "undefined" in reasons or "zero target variance" in reasons
    assert gate["sufficiency"]["slice_coverage_ok"] is False
    assert gate["ranking"]["slice_gaps"].get("level_rank_ev_gap") in (None, "undefined")


def test_identical_model_weights_do_not_count_as_two_checkpoints(checkpoints, tmp_path):
    registration = write_value_gate_registration(tmp_path / "registration.json")
    duplicated = tmp_path / "eval-a-copy.pt"
    duplicated.write_bytes(checkpoints["a"]["path"].read_bytes())
    duplicate_identity = load_v2_policy(
        duplicated, device="cpu", expected_algorithm="ippo",
    ).identity
    result = _result(
        _samples(level5_zero=False), _points(),
        registration=registration, identity=checkpoints["a"]["identity"],
    )
    gate = evaluate_value_gate(
        result, registration=registration,
        checkpoint_identities=[checkpoints["a"]["identity"], duplicate_identity],
    )
    assert gate["verdict"] != "PASS"
    assert gate["sufficiency"]["distinct_checkpoint_weight_hashes"] == 1
    assert gate["sufficiency"]["duplicate_weight_identity_count"] == 1
    assert any("identical model weights" in reason for reason in gate["reasons"])


def test_result_binding_to_registration_and_identity_is_enforced(checkpoints, tmp_path):
    registration = write_value_gate_registration(tmp_path / "registration.json")
    result = _result(
        _samples(level5_zero=False), _points(),
        registration=registration, identity=checkpoints["a"]["identity"],
    )
    result.metadata["registration_hash"] = "f" * 64
    result.metadata["checkpoint"] = dict(checkpoints["c"]["identity"])
    gate = evaluate_value_gate(
        result, registration=registration,
        checkpoint_identities=[
            checkpoints["a"]["identity"], checkpoints["b"]["identity"],
        ],
    )
    assert gate["verdict"] != "PASS"
    reasons = " | ".join(gate["reasons"])
    assert "registration" in reasons
    assert "checkpoint" in reasons


def test_gate_identity_requires_evaluation_only_cpp_ippo(checkpoints, tmp_path):
    training_checkpoint = checkpoints["a"]["trainer"].save_checkpoint(
        tmp_path / "training.pt"
    )
    training_identity = load_v2_policy(
        training_checkpoint, device="cpu", expected_algorithm="ippo",
    ).identity
    with pytest.raises(ValueGateError, match="evaluation-only|evaluation"):
        _verify_gate_checkpoint_identities([training_identity])
    verified, weight_hashes, all_verified, problems = _verify_gate_checkpoint_identities(
        [checkpoints["a"]["identity"]]
    )
    assert all_verified and not problems
    assert weight_hashes and weight_hashes[0]
    assert verified[0]["checkpoint_schema"] == "guandan-marvel-evaluation-checkpoint-v1"
    assert verified[0]["backend"] == "cpp"
