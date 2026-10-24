# SPDX-License-Identifier: MIT
"""Reproduce the first-reviewer blockers against old vs repaired logic.

This script does not train or qualify anything.  It reconstructs the *old*
formulas/conditions in plain code (never importing a helper for the old side)
and compares them with the repaired production functions.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.phase6.value_gate import (
    _cluster_bootstrap_ev, compute_ranking_metrics, compute_value_metrics,
    evaluate_value_gate, write_value_gate_registration,
)
from guandan_marvel.training.losses import compute_value_loss


def blocker_1_2() -> None:
    rollout = TensorDict({
        "num_steps": torch.tensor([3], dtype=torch.long),
        "player_ids": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "policy_train_mask": torch.tensor([[True, False, True]], dtype=torch.bool),
    }, batch_size=[1])
    prediction = torch.tensor([[1.0, 1.0, 1.0]])
    target = torch.tensor([[0.0, 0.0, 0.0]])
    # Old formula: all acting tokens (history & player_ids>=0), no owner mask.
    old_loss = float(torch.nn.functional.huber_loss(prediction, target, reduction="mean"))
    hostile = prediction.clone()
    hostile[0, 1] = 100.0
    old_hostile = float(torch.nn.functional.huber_loss(hostile, target, reduction="mean"))
    new_loss = float(compute_value_loss(
        rollout, player_state_value_pred=prediction, player_state_value_target=target,
    ).value_loss_mean)
    new_hostile = float(compute_value_loss(
        rollout, player_state_value_pred=hostile, player_state_value_target=target,
    ).value_loss_mean)
    print("[1.2] old loss %.6f -> hostile %.6f (changed=%s)" % (
        old_loss, old_hostile, old_loss != old_hostile,
    ))
    print("[1.2] new loss %.6f -> hostile %.6f (unchanged=%s)" % (
        new_loss, new_hostile, new_loss == new_hostile,
    ))


def blocker_1_4() -> None:
    from tests.marvel_training._phase6_pool_fixtures import make_tiny_eval_checkpoint
    from tests.marvel_training.test_phase6_pool_history import (
        _current_probs, _pool, _router, _step,
    )

    fixture = make_tiny_eval_checkpoint(Path(tempfile.mkdtemp(prefix="phase6-repro-")))
    players = [0, 2, 1, 3]
    # Old behaviour: forward only the session that owns the acting slot.
    old_pool = _pool(fixture)
    old_router = _router(old_pool)
    old_router.prepare(rollout_buffer=None)
    old_session = old_router.sessions["opponent-a"]
    old_lengths = []
    for index, player in enumerate(players, start=1):
        owner = old_pool.assignment(slot=0).owners_by_seat[player]
        if owner != 0:
            old_session.probabilities(_step(player, 100 + index))
        old_lengths.append(old_session.slot_sequence_lengths()[0])
    # New behaviour: every resident session advances every step.
    new_pool = _pool(fixture)
    new_router = _router(new_pool)
    new_router.prepare(rollout_buffer=None)
    new_session = new_router.sessions["opponent-a"]
    new_lengths = []
    for index, player in enumerate(players, start=1):
        new_router.policy_probabilities(
            step_rollout=_step(player, 100 + index),
            parsed_envs_step={"player_ids": torch.tensor([player])},
            current_probabilities=_current_probs(1),
        )
        new_lengths.append(new_session.slot_sequence_lengths()[0])
    print("[1.4] owner sequence [0,0,1,1]")
    print("[1.4] old frozen cache lengths %s" % old_lengths)
    print("[1.4] new frozen cache lengths %s" % new_lengths)


def legacy_verdict(samples, points) -> str:
    """The exact decision structure of the first implementation (no slices)."""
    metrics = compute_value_metrics(samples)
    ranking = compute_ranking_metrics(points)
    ev_ci = _cluster_bootstrap_ev(samples, resamples=2000, seed=20261020, confidence=0.95)
    independent = int(metrics["overall"]["independent_deal_count"])
    sufficient = (
        independent >= 64
        and len(samples) >= 512
        and len({p.canonical_deal for p in points}) >= 64
        and len(points) >= 64
        and not any(p.restricted for p in points)
    )
    ev = metrics["overall"]["ev"]
    if not sufficient:
        return "INCONCLUSIVE"
    pass_conditions = (
        ev is not None
        and ev >= 0.10
        and ev_ci["low"] is not None and ev_ci["low"] >= 0.02
        and metrics["overall"]["pearson"] is not None and metrics["overall"]["pearson"] >= 0.25
        and metrics["overall"]["spearman"] is not None and metrics["overall"]["spearman"] >= 0.20
        and metrics["calibration_ece"] is not None and metrics["calibration_ece"] <= 0.10
        and ranking["delta"] is not None and ranking["delta"] >= 0.05
        and ranking["ci"]["low"] is not None and ranking["ci"]["low"] > 0.0
    )
    if pass_conditions:
        return "PASS"
    ev_upper = ev_ci["high"]
    rank_upper = ranking["ci"]["high"]
    if (ev_upper is not None and ev_upper < 0.10) or (rank_upper is not None and rank_upper <= 0.0):
        return "REJECT"
    return "INCONCLUSIVE"


def blocker_1_1_and_1_5() -> None:
    from tests.marvel_training.test_phase6_value_gate_gate_rules import (
        _bias_samples, _identities, _point, _result,
    )
    import tests.marvel_training.test_phase6_value_gate_gate_rules as rules

    temp = Path(tempfile.mkdtemp(prefix="phase6-gate-repro-"))
    registration = write_value_gate_registration(temp / "registration.json")
    bias_samples = _bias_samples()
    bias_points = [_point(index, accuracy=1.0) for index in range(64)]
    print("[1.1] old verdict (before the SEARCH_IMPLEMENTED raise): %s" % legacy_verdict(
        bias_samples, bias_points,
    ))
    gate = evaluate_value_gate(
        _result(bias_samples, bias_points), registration=registration,
        checkpoint_identities=_identities(),
    )
    print("[1.1] new verdict: %s" % gate["verdict"])
    print("[1.1] new reasons: %s" % " | ".join(gate["reasons"]))

    zero_samples = [
        rules._sample(
            index, prediction=0.1 * (index % 8), target=0.5,
            canonical=rules._canonical(deal), team=index % 2, deal_id=f"deal-{deal}",
        )
        for deal in range(64) for index in range(8)
    ]
    zero_points = [_point(index, accuracy=0.0) for index in range(64)]
    print("[1.5] old verdict: %s" % legacy_verdict(zero_samples, zero_points))
    zero_gate = evaluate_value_gate(
        _result(zero_samples, zero_points), registration=registration,
        checkpoint_identities=_identities(),
    )
    print("[1.5] new verdict: %s" % zero_gate["verdict"])
    print("[1.5] new reasons: %s" % " | ".join(zero_gate["reasons"]))


if __name__ == "__main__":
    blocker_1_2()
    blocker_1_4()
    blocker_1_1_and_1_5()
