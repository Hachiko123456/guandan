"""Phase-2 DDP tests: CPU gloo two-rank smoke and count-weighted global statistics."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from guandan_marvel.config import (
    LEGACY_POLICY_REGULARIZER_MODE, MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE,
)
from guandan_marvel.kaggle.worker import _aggregate_metrics
from guandan_marvel.model import Agent
from guandan_marvel.training.gae import compute_gae_from_value
from guandan_marvel.training.losses import compute_clipped_pg_loss, compute_value_loss
from guandan_marvel.training.macro_actions import segment_rollout_macros
from guandan_marvel.training.macro_ppo import (
    build_macro_batch_view, compute_macro_normalized_kl_mean, compute_macro_pg_loss,
    compute_macro_value_loss, connected_zero_like, model_connected_zero,
)
from guandan_marvel.training.schedules import compute_agent_schedule

from tests.marvel_training._ddp_launcher import launch_two_ranks
from tests.marvel_training._macro_ddp_worker import (
    RANK_CASES, _deterministic_init, _EIGHT_MACRO_CASE, _EMPTY_LOCAL_CASE,
    _InjectedMacroTrainer, _scaled_log_probs, build_macro_ready_rollout,
    empty_rank_game_config, macro_inputs, stack_cases, synthetic_case,
)
from tests.marvel_training._macro_fixtures import make_case

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests" / "marvel_training" / "_macro_ddp_worker.py"


def _single_process_reference():
    cases = [synthetic_case(RANK_CASES[0]), synthetic_case(RANK_CASES[1])]
    rollout, _values, segmentation, probabilities = stack_cases(cases)
    advantage, target, prediction_base = macro_inputs(segmentation)
    view = build_macro_batch_view(segmentation, torch.arange(segmentation.batch_size))
    w = torch.tensor(0.7, requires_grad=True)
    scaled_log_probs = _scaled_log_probs(rollout, probabilities, w)
    pg = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=scaled_log_probs,
        macro_advantage=advantage, clip_coef=0.2, world_size=1,
    )
    value = compute_macro_value_loss(
        player_state_value_pred=prediction_base * w, view=view,
        macro_value_target=target, world_size=1,
    )
    kl = compute_macro_normalized_kl_mean(
        rollout, view=view, avail_probs_new=torch.softmax(scaled_log_probs, dim=-1), world_size=1,
    )
    total = pg.pg_loss_mean + 0.5 * value.macro_value_loss_mean + 0.25 * kl
    total.backward()
    return {
        "grad": float(w.grad.item()),
        "pg_loss": float(pg.pg_loss_mean.detach().item()),
        "value_loss": float(value.macro_value_loss_mean.detach().item()),
        "kl": float(kl.detach().item()),
        "macro_count_global": int(pg.macro_count_global),
        "value_count_global": int(value.macro_count_global),
    }


@pytest.mark.slow
def test_gloo_two_rank_macro_gradient_and_statistics_equal_single_process_reference(tmp_path: Path):
    output = tmp_path / "synthetic.json"
    results = launch_two_ranks(
        WORKER, "--mode", "synthetic", "--out", str(output), cwd=ROOT, timeout=900,
    )
    for result in results:
        assert result.returncode == 0, result.stdout
    rank_payloads = [
        json.loads((tmp_path / f"synthetic.json.rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    reference = _single_process_reference()
    # The two ranks hold a different number of macros (the synthetic counterexample).
    assert rank_payloads[0]["local_macro_count"] != rank_payloads[1]["local_macro_count"]
    assert rank_payloads[0]["local_macro_count"] == 2
    assert rank_payloads[1]["local_macro_count"] == 8
    for payload in rank_payloads:
        # Every rank uses the same count-weighted global statistics.
        assert payload["macro_count_global"] == reference["macro_count_global"] == 10
        assert payload["value_count_global"] == reference["value_count_global"] == 10
        assert payload["world_size"] == 2
    # DDP averages the per-rank gradients: mean(grad_k) == grad of the concatenated batch.
    ddp_averaged_grad = sum(payload["grad"] for payload in rank_payloads) / 2.0
    assert ddp_averaged_grad == pytest.approx(reference["grad"], rel=1e-5, abs=1e-6)
    # L_k = (world_size / N) * sum_{i in k} l_i -> (L_0 + L_1) / world_size == reference.
    for key in ("pg_loss", "value_loss", "kl"):
        combined = (rank_payloads[0][key] + rank_payloads[1][key]) / 2.0
        assert combined == pytest.approx(reference[key], abs=1e-6), key
    # Aggregating the production per-rank records reproduces the single-process reference.
    rows = [payload["local_record"] for payload in rank_payloads]
    aggregated = _aggregate_metrics(_FakeContext(rows), rows[0])
    l0 = rank_payloads[0]["pg_loss"]
    l1 = rank_payloads[1]["pg_loss"]
    # Cross-path float32 rounding keeps this at the same tolerance as the (L0+L1)/2 check.
    assert aggregated["macro_initial/pg_loss"] == pytest.approx(reference["pg_loss"], abs=1e-6)
    # The aggregation identity itself is exact to float64 rounding.
    assert aggregated["macro_initial/pg_loss"] == pytest.approx((l0 + l1) / 2.0, abs=1e-9)
    assert aggregated["macro_initial/pg_loss"] != pytest.approx(
        (2 * l0 + 8 * l1) / 10.0, abs=1e-9,
    )
    assert aggregated["macro_initial/macro_count_local"] == pytest.approx(10.0, abs=0.0)


@pytest.mark.slow
def test_gloo_two_rank_macro_trainer_iteration_profile_c(tmp_path: Path):
    output = tmp_path / "trainer.json"
    results = launch_two_ranks(
        WORKER, "--mode", "trainer", "--out", str(output), cwd=ROOT, timeout=1800,
    )
    for result in results:
        assert result.returncode == 0, result.stdout
    payloads = [
        json.loads((tmp_path / f"trainer.json.rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    for payload in payloads:
        assert payload["profile"] == "macro_decision_v1"
        assert payload["regularizer_mode"] == "legacy_token_kl_uniform_v1"
        assert payload["world_size"] == 2
        assert payload["ddp_enabled"] is True
        assert payload["iteration"] == 1
        assert payload["durable_updates"] == 1
        assert payload["segment_violations"] == 0
        assert payload["incomplete"] == 0
        assert payload["grad_norm"] > 0.0
        assert payload["macro_count_global"] >= 1
    assert payloads[0]["rank"] == 0 and payloads[1]["rank"] == 1
    # Both ranks agree on the count-weighted global macro statistics.
    assert payloads[0]["macro_count_global"] == payloads[1]["macro_count_global"]
    assert payloads[0]["macro_count_local"] + payloads[1]["macro_count_local"] == (
        payloads[0]["macro_count_global"]
    )
    # The all-reduced advantage aggregates are rank-consistent (identical on every rank).
    for key in (
        "adv_raw_mean_global", "adv_raw_std_global", "adv_norm_mean_global",
        "adv_norm_std_global", "adv_count_global",
    ):
        assert payloads[0][key] == payloads[1][key], key
    assert payloads[0]["adv_count_global"] == payloads[0]["macro_count_global"]
    assert payloads[0]["adv_norm_mean_global"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= payloads[0]["adv_norm_std_global"] <= 1.0
    # Aggregating the two real local records reproduces the DDP loss algebra:
    # count-weighted per-sample means == (L_0 + L_1) / world_size and the summed
    # macro count equals the all-reduced global count.
    rows = [payload["local_record"] for payload in payloads]
    aggregated = _aggregate_metrics(_FakeContext(rows), rows[0])
    world = payloads[0]["world_size"]
    global_count = payloads[0]["macro_count_global"]
    assert aggregated["macro_initial/macro_count_local"] == pytest.approx(
        float(global_count), abs=0.0,
    )
    for metric_key in ("macro_initial/pg_loss", "macro_initial/value_loss"):
        counts = [rows[rank]["macro_initial/macro_count_local"] for rank in range(2)]
        scaled_losses = [
            rows[rank][metric_key] * (world * counts[rank] / global_count) for rank in range(2)
        ]
        weighted_sum = sum(
            rows[rank][metric_key] * counts[rank] for rank in range(2)
        ) / global_count
        assert aggregated[metric_key] == pytest.approx(weighted_sum, abs=1e-12), metric_key
        assert aggregated[metric_key] == pytest.approx(
            (scaled_losses[0] + scaled_losses[1]) / 2.0, abs=1e-9,
        ), metric_key
        old_buggy = sum(
            scaled_losses[rank] * counts[rank] for rank in range(2)
        ) / global_count
        if abs(weighted_sum - old_buggy) > 1e-12:
            assert aggregated[metric_key] != pytest.approx(old_buggy, abs=1e-12), metric_key
    assert aggregated["macro_initial/adv_norm_mean_global"] == pytest.approx(
        payloads[0]["adv_norm_mean_global"], abs=1e-12,
    )


class _FakeContext:
    """Minimal two-rank context exposing only ``all_gather_object``."""

    def __init__(self, rows):
        self.rows = rows
        self.world_size = len(rows)

    def all_gather_object(self, value):
        return [dict(row) for row in self.rows]


# The reviewer's reproduction: two ranks with 2 and 8 local macros at world size 2.
# ``L_k`` are the DDP-scaled differentiable losses; the production log records the
# per-sample means ``m_k = L_k * N / (world_size * n_k)`` (local.py).
_WORLD = 2
_COUNTS = (2.0, 8.0)
_SCALED_LOSSES = (-0.0133768385, 0.0039793490)
_SCALED_VALUE_LOSSES = (-0.02, 0.01)
_GLOBAL_COUNT = sum(_COUNTS)
_PER_SAMPLE_MEANS = tuple(
    loss * _GLOBAL_COUNT / (_WORLD * count)
    for loss, count in zip(_SCALED_LOSSES, _COUNTS, strict=True)
)
_VALUE_MEANS = tuple(
    loss * _GLOBAL_COUNT / (_WORLD * count)
    for loss, count in zip(_SCALED_VALUE_LOSSES, _COUNTS, strict=True)
)
_REFERENCE = sum(_SCALED_LOSSES) / _WORLD
_VALUE_REFERENCE = sum(_SCALED_VALUE_LOSSES) / _WORLD
_OLD_BUGGY_AGGREGATE = sum(
    loss * count for loss, count in zip(_SCALED_LOSSES, _COUNTS, strict=True)
) / _GLOBAL_COUNT
_OLD_BUGGY_VALUE_AGGREGATE = sum(
    loss * count for loss, count in zip(_SCALED_VALUE_LOSSES, _COUNTS, strict=True)
) / _GLOBAL_COUNT


def _rank_row(
    *,
    local_macros: float,
    pg_loss_mean: float,
    value_loss_mean: float,
    adv_norm_mean: float,
    token_count: float = 303.0,
    value_unit: str = "macro_start",
    macro_count_global: float = 10.0,
):
    """A per-rank row in the post-fix log representation (per-sample means)."""
    return {
        "macro/macro_count_local": local_macros,
        "macro/macro_count_global": macro_count_global,
        "macro/token_count_local": token_count,
        "macro/pg_loss": pg_loss_mean,
        "macro/value_loss": value_loss_mean,
        "macro/adv_norm_mean": adv_norm_mean,
        "macro_initial/macro_count_local": local_macros,
        "macro_initial/macro_count_global": macro_count_global,
        "macro_initial/token_count_local": token_count,
        "macro_initial/pg_loss": pg_loss_mean,
        "macro_initial/value_loss": value_loss_mean,
        "macro_initial/adv_norm_mean": adv_norm_mean,
        "macro_initial/adv_raw_mean_global": 0.001,
        "macro_initial/adv_norm_mean_global": 0.0,
        "macro_initial/adv_count_global": float(macro_count_global),
        "macro/value_sample_unit": value_unit,
        "agent/pg_loss": pg_loss_mean,  # legacy token path must stay equal-weighted
        "statistics/env_steps_total": 303,
    }


def _repro_rows():
    return [
        _rank_row(
            local_macros=_COUNTS[0],
            pg_loss_mean=_PER_SAMPLE_MEANS[0],
            value_loss_mean=_VALUE_MEANS[0],
            adv_norm_mean=0.018818,
            macro_count_global=_GLOBAL_COUNT,
        ),
        _rank_row(
            local_macros=_COUNTS[1],
            pg_loss_mean=_PER_SAMPLE_MEANS[1],
            value_loss_mean=_VALUE_MEANS[1],
            adv_norm_mean=-0.004704,
            macro_count_global=_GLOBAL_COUNT,
        ),
    ]


def test_aggregate_metrics_loss_metrics_equal_single_process_reference():
    """Aggregated per-sample means reconstruct the concatenated-batch reference.

    Count-weighted aggregation of ``m_k = L_k * N / (W * n_k)`` equals
    ``sum_k L_k / W == (L0 + L1) / 2`` (the single-process reference) and must never
    reproduce the old buggy value ``(2 L0 + 8 L1) / 10``.
    """
    rows = _repro_rows()
    aggregated = _aggregate_metrics(_FakeContext(rows), rows[0])
    for prefix in ("macro/", "macro_initial/"):
        value = aggregated[f"{prefix}pg_loss"]
        assert value == pytest.approx(_REFERENCE, abs=1e-12), prefix
        assert value == pytest.approx(
            (_SCALED_LOSSES[0] + _SCALED_LOSSES[1]) / 2.0, abs=1e-12,
        ), prefix
        assert value != pytest.approx(_OLD_BUGGY_AGGREGATE, abs=1e-9), prefix
        value_loss = aggregated[f"{prefix}value_loss"]
        assert value_loss == pytest.approx(_VALUE_REFERENCE, abs=1e-12), prefix
        assert value_loss == pytest.approx(
            (_SCALED_VALUE_LOSSES[0] + _SCALED_VALUE_LOSSES[1]) / 2.0, abs=1e-12,
        ), prefix
        assert value_loss != pytest.approx(_OLD_BUGGY_VALUE_AGGREGATE, abs=1e-9), prefix
    # Per-sample advantage statistics remain count-weighted (2:8).
    assert aggregated["macro_initial/adv_norm_mean"] == pytest.approx(
        (2 * 0.018818 + 8 * -0.004704) / 10.0, abs=1e-12,
    )
    assert aggregated["macro_initial/adv_norm_mean"] != pytest.approx(
        (0.018818 + -0.004704) / 2.0, abs=1e-9,
    )
    # Non-macro metrics keep the previous equal-weight behavior.
    assert aggregated["agent/pg_loss"] == pytest.approx(
        (_PER_SAMPLE_MEANS[0] + _PER_SAMPLE_MEANS[1]) / 2.0, abs=1e-12,
    )


def test_aggregate_metrics_counts_sum_to_global_count():
    rows = _repro_rows()
    aggregated = _aggregate_metrics(_FakeContext(rows), rows[0])
    for prefix in ("macro/", "macro_initial/"):
        assert aggregated[f"{prefix}macro_count_local"] == pytest.approx(
            _GLOBAL_COUNT, abs=0.0,
        )
        assert aggregated[f"{prefix}macro_count_local"] == pytest.approx(
            aggregated[f"{prefix}macro_count_global"], abs=0.0,
        )
    assert aggregated["statistics/env_steps_total"] == 606


def test_aggregate_metrics_token_unit_metrics_use_token_counts():
    rows = [
        {
            "macro_initial/macro_count_local": 2.0,
            "macro_initial/macro_count_global": 10.0,
            "macro_initial/token_count_local": 100.0,
            "macro_initial/legacy_token_kl": 0.2,
            "macro_initial/token_value_loss": 1.0,
            "macro_initial/value_loss": 1.0,
            "macro/value_sample_unit": "token",
        },
        {
            "macro_initial/macro_count_local": 8.0,
            "macro_initial/macro_count_global": 10.0,
            "macro_initial/token_count_local": 300.0,
            "macro_initial/legacy_token_kl": 0.6,
            "macro_initial/token_value_loss": 3.0,
            "macro_initial/value_loss": 3.0,
            "macro/value_sample_unit": "token",
        },
    ]
    aggregated = _aggregate_metrics(_FakeContext(rows), rows[0])
    token_weighted = {
        "legacy_token_kl": (100 * 0.2 + 300 * 0.6) / 400.0,
        "token_value_loss": (100 * 1.0 + 300 * 3.0) / 400.0,
        "value_loss": (100 * 1.0 + 300 * 3.0) / 400.0,
    }
    macro_weighted = {
        "legacy_token_kl": (2 * 0.2 + 8 * 0.6) / 10.0,
        "token_value_loss": (2 * 1.0 + 8 * 3.0) / 10.0,
        "value_loss": (2 * 1.0 + 8 * 3.0) / 10.0,
    }
    for key, expected in token_weighted.items():
        assert aggregated[f"macro_initial/{key}"] == pytest.approx(expected, abs=1e-12), key
        assert aggregated[f"macro_initial/{key}"] != pytest.approx(
            macro_weighted[key], abs=1e-9,
        ), key


def test_aggregate_metrics_token_metrics_fall_back_to_macro_counts():
    rows = [
        {"macro_initial/macro_count_local": 2.0, "macro_initial/legacy_token_kl": 0.2},
        {"macro_initial/macro_count_local": 8.0, "macro_initial/legacy_token_kl": 0.6},
    ]
    aggregated = _aggregate_metrics(_FakeContext(rows), rows[0])
    assert aggregated["macro_initial/legacy_token_kl"] == pytest.approx(
        (2 * 0.2 + 8 * 0.6) / 10.0, abs=1e-12,
    )


def test_aggregate_metrics_rejects_rank_inconsistent_global_macro_metrics():
    rows = _repro_rows()
    rows[1]["macro_initial/adv_norm_mean_global"] = 0.0
    rows[0]["macro_initial/adv_norm_mean_global"] = 0.0
    rows[0]["macro_initial/adv_raw_mean_global"] = 0.001
    rows[1]["macro_initial/adv_raw_mean_global"] = 0.002
    with pytest.raises(ValueError, match="differs across ranks"):
        _aggregate_metrics(_FakeContext(rows), rows[0])
    # Consistent global keys are taken as-is, never re-averaged or weighted.
    rows[1]["macro_initial/adv_raw_mean_global"] = 0.001
    aggregated = _aggregate_metrics(_FakeContext(rows), rows[0])
    assert aggregated["macro_initial/adv_raw_mean_global"] == pytest.approx(0.001, abs=0.0)
    assert aggregated["macro_initial/adv_norm_mean_global"] == pytest.approx(0.0, abs=0.0)
    assert aggregated["macro_initial/adv_count_global"] == pytest.approx(
        float(rows[0]["macro_initial/macro_count_global"]), abs=0.0,
    )


def test_aggregate_metrics_nan_or_negative_weights_rejected():
    for bad in (float("nan"), -1.0):
        rows = _repro_rows()
        rows[1]["macro_initial/macro_count_local"] = bad
        with pytest.raises(ValueError, match="invalid weight"):
            _aggregate_metrics(_FakeContext(rows), rows[0])


def test_aggregate_metrics_missing_weight_is_not_silently_dropped():
    """A rank present in the value set but missing its weight fails fast."""
    rows = _repro_rows()
    del rows[1]["macro_initial/macro_count_local"]
    with pytest.raises(ValueError, match="no usable"):
        _aggregate_metrics(_FakeContext(rows), rows[0])
    # All-zero weights keep the documented equal-weight fallback.
    zero_rows = [
        {"macro_initial/macro_count_local": 0.0, "macro_initial/pg_loss": 1.0},
        {"macro_initial/macro_count_local": 0.0, "macro_initial/pg_loss": 3.0},
    ]
    aggregated = _aggregate_metrics(_FakeContext(zero_rows), zero_rows[0])
    assert aggregated["macro_initial/pg_loss"] == pytest.approx(2.0, abs=1e-12)


def test_zero_local_macro_regularizer_fallback_is_finite_and_graph_connected():
    """P3: the legacy-KL fallback for an empty local macro batch must be a connected zero."""
    reference = torch.tensor(2.0, requires_grad=True)
    fallback = connected_zero_like(reference)
    assert fallback.item() == 0.0
    fallback.backward()
    assert reference.grad.item() == 0.0
    # The hazard this replaces: an all-inactive rollout turns the legacy KL mean into NaN.
    empty_rollout, _values, _segmentation, _probs = make_case([-1, -1], [True, True])
    avail_probs = torch.zeros((1, 2, 4), dtype=torch.float32)
    avail_probs[..., 0] = 1.0
    legacy = compute_clipped_pg_loss(
        empty_rollout, clip_coef=0.2, avail_probs_new=avail_probs,
        player_advantage=torch.zeros((1, 2)),
    )
    assert math.isnan(legacy.kl_unif_mean.item())


# ---------------------------------------------------------------------------
# Phase 2A: model-connected zero anchor for empty DDP ranks
# ---------------------------------------------------------------------------


def _real_model_and_rollout(profile, spec):
    """A real ``Agent`` (value head on) and a rollout it can consume."""
    cfg, game = empty_rank_game_config(profile, LEGACY_POLICY_REGULARIZER_MODE, 1)
    torch.manual_seed(0)
    model = Agent(cfg, enable_value_head=True, enable_ntp_head=False)
    rollout = build_macro_ready_rollout(spec, game)
    return model, rollout


def test_model_connected_zero_is_finite_zero_connected_and_reaches_every_param():
    """The empty-rank anchor must be exactly 0, finite and reach every parameter."""
    model, rollout = _real_model_and_rollout(SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE, _EIGHT_MACRO_CASE)
    output = model(rollout=rollout)
    assert output.avail_log_probs is not None and output.player_state_value is not None
    anchor = model_connected_zero(output.avail_log_probs, output.player_state_value)
    assert anchor.item() == 0.0
    assert bool(torch.isfinite(anchor).all())
    assert anchor.requires_grad is True
    model.zero_grad(set_to_none=True)
    anchor.backward()
    parameters = dict(model.named_parameters())
    assert parameters
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert float(parameter.grad.abs().sum()) == 0.0, name
    # None entries (e.g. a disabled head) are skipped; an all-None call fails fast.
    assert model_connected_zero(None, output.player_state_value).requires_grad is True
    with pytest.raises(ValueError, match="at least one model tensor"):
        model_connected_zero(None, None)


def test_profile_b_empty_local_batch_value_loss_is_nan_only_on_the_legacy_path():
    """Profile B's legacy value loss is NaN on an empty active set; the anchor is not."""
    _cfg, game = empty_rank_game_config(SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE, LEGACY_POLICY_REGULARIZER_MODE, 1)
    rollout = build_macro_ready_rollout(_EMPTY_LOCAL_CASE, game)
    time = torch.arange(rollout["player_ids"].shape[1]).unsqueeze(0)
    active = (time < rollout["num_steps"].unsqueeze(1)) & (rollout["player_ids"] >= 0)
    assert int(active.sum()) == 0
    prediction = torch.zeros_like(rollout["rewards"][..., 0])
    legacy = compute_value_loss(
        rollout, player_state_value_pred=prediction, player_state_value_target=torch.zeros_like(prediction),
    )
    assert math.isnan(legacy.value_loss_mean.item())
    model, _ = _real_model_and_rollout(SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE, _EMPTY_LOCAL_CASE)
    output = model(rollout=rollout)
    anchor = model_connected_zero(output.player_state_value)
    assert anchor.item() == 0.0
    assert bool(torch.isfinite(anchor).all())
    assert anchor.requires_grad is True


def test_profile_c_constant_macro_loss_is_disconnected_but_the_model_anchor_is_not():
    """``connected_zero_like(constant)`` stays detached; ``model_connected_zero`` does not."""
    model, rollout = _real_model_and_rollout(MACRO_DECISION_TRAINING_SEMANTICS_PROFILE, _EIGHT_MACRO_CASE)
    output = model(rollout=rollout)
    # An empty rank's macro loss is a fresh constant (max_macros == 0).
    constant_macro_loss = output.player_state_value.detach().sum() * 0.0
    assert constant_macro_loss.requires_grad is False
    old_fallback = connected_zero_like(constant_macro_loss)
    assert old_fallback.requires_grad is False
    old_total = constant_macro_loss + old_fallback
    assert old_total.requires_grad is False  # the pre-fix profile-C hazard
    model.zero_grad(set_to_none=True)
    anchor = model_connected_zero(output.avail_log_probs, output.player_state_value)
    assert anchor.requires_grad is True
    new_total = constant_macro_loss + anchor
    assert new_total.requires_grad is True
    new_total.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert float(parameter.grad.abs().sum()) == 0.0, name


def test_nonempty_rank_composition_is_unchanged_for_eight_macros(tmp_path: Path):
    """A non-empty rank keeps the original composition (no empty-rank term inserted)."""
    profile = SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE
    cfg, game = empty_rank_game_config(profile, LEGACY_POLICY_REGULARIZER_MODE, 1)
    trainer = _InjectedMacroTrainer(cfg, game=game, run_dir=tmp_path / "eight", spec=_EIGHT_MACRO_CASE)
    _deterministic_init(trainer)
    rollout = build_macro_ready_rollout(_EIGHT_MACRO_CASE, game)
    segmentation = segment_rollout_macros(rollout)
    view = build_macro_batch_view(segmentation, torch.arange(segmentation.batch_size))
    schedule = compute_agent_schedule(cfg, iteration=1)
    with torch.no_grad():
        state_values = trainer.agent_resume(rollout=rollout).state_values.detach().float()
        token_gae = compute_gae_from_value(
            rollout, state_values,
            discount_gamma=cfg.discount_gamma, gae_lambda=cfg.ppo.gae_lambda,
        )
        macro_advantage = token_gae.player_advantage.detach().gather(
            1, segmentation.macro_start_index.clamp(min=0),
        )
    output = trainer.agent_resume(rollout=rollout)
    pg = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=output.avail_log_probs,
        macro_advantage=macro_advantage, clip_coef=schedule.clip_coef, world_size=1,
    )
    assert pg.macro_count_local == 8
    metrics = trainer.train_iteration()
    trainer.close()
    assert metrics["macro_initial/macro_count_local"] == 8
    assert metrics["macro_initial/empty_local_batch"] == 0
    # world_size == 1 -> the logged per-sample mean is the raw macro loss exactly.
    assert metrics["macro_initial/pg_loss"] == pytest.approx(float(pg.pg_loss_mean.detach()), rel=1e-6, abs=1e-12)


def _run_empty_rank_reference(profile: str, out: Path) -> dict:
    env = dict(os.environ)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable, str(WORKER), "--mode", "empty_rank", "--profile", profile,
            "--emulate-world-size", "2", "--out", str(out),
        ],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(Path(f"{out}.rank0.json").read_text(encoding="utf-8"))


@pytest.mark.slow
@pytest.mark.parametrize(
    "profile",
    [SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE, MACRO_DECISION_TRAINING_SEMANTICS_PROFILE],
)
def test_gloo_empty_rank_is_finite_connected_and_matches_reference(profile: str, tmp_path: Path):
    """Rank 0 empty (n_0 = 0) / rank 1 eight macros (N = 8) under real gloo DDP.

    Pre-fix this fails: Profile B raises ``macro_agent_loss`` NaN and Profile C
    trips ``find_unused_parameters=False``.  The single-process reference
    (``--emulate-world-size 2``) reproduces the DDP gradient average of a world
    where the partner rank holds no samples, so both profiles can be checked
    for exact equality against a genuine one-process run.
    """
    out = tmp_path / "empty.json"
    results = launch_two_ranks(
        WORKER, "--mode", "empty_rank", "--profile", profile, "--out", str(out),
        cwd=ROOT, timeout=1800,
    )
    for result in results:
        assert result.returncode == 0, result.stdout
    rank0 = json.loads((tmp_path / "empty.json.rank0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((tmp_path / "empty.json.rank1.json").read_text(encoding="utf-8"))
    names = rank1["parameter_names"]
    assert names and set(rank0["parameter_names"]) == set(names)
    # The empty rank holds no local macro; the non-empty rank holds all eight.
    assert rank0["spec"] == "empty" and rank0["macro_count_local"] == 0
    assert rank1["spec"] == "eight" and rank1["macro_count_local"] == 8
    assert rank0["macro_count_global"] == rank1["macro_count_global"] == 8
    # Every trainable parameter receives a *local* (pre-all-reduce) gradient; the
    # empty rank's local gradients are exactly zero.
    assert set(rank0["local_grad"]) == set(names)
    assert set(rank1["local_grad"]) == set(names)
    assert all(value == 0.0 for value in rank0["local_grad"].values())
    assert any(value != 0.0 for value in rank1["local_grad"].values())
    # DDP averages the local gradients: identical on both ranks and equal to the mean.
    for name in names:
        assert rank0["post_grad"][name] == pytest.approx(rank1["post_grad"][name], abs=1e-12)
        assert rank0["post_grad"][name] == pytest.approx(
            0.5 * (rank0["local_grad"][name] + rank1["local_grad"][name]), rel=1e-6, abs=1e-12,
        ), name
    # Profile B must not emit the empty-set NaN a metric.
    assert math.isfinite(rank0["grad_norm"]) and math.isfinite(rank1["grad_norm"])
    for record in (rank0["local_record"], rank1["local_record"]):
        assert record["macro_initial/empty_local_batch"] == (1.0 if record is rank0["local_record"] else 0.0)
    if profile == SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE:
        assert rank0["local_record"]["macro_initial/value_loss"] == 0.0
        assert rank0["local_record"]["macro_initial/token_value_loss"] == 0.0

    reference = _run_empty_rank_reference(profile, tmp_path / "ref.json")
    # Post-backward (DDP-averaged) gradients equal the single-process reference bit-for-bit.
    for name in names:
        assert rank1["post_grad"][name] == pytest.approx(reference["post_grad"][name], rel=1e-7, abs=1e-12), name
    assert rank1["post_grad_norm"] == pytest.approx(reference["post_grad_norm"], rel=1e-7, abs=1e-12)
    # The count-weighted aggregation of the two real local records equals the reference.
    rows = [rank0["local_record"], rank1["local_record"]]
    aggregated = _aggregate_metrics(_FakeContext(rows), rows[0])
    assert aggregated["macro_initial/macro_count_local"] == pytest.approx(8.0, abs=0.0)
    for key in ("macro_initial/pg_loss", "macro_initial/value_loss"):
        assert aggregated[key] == pytest.approx(reference["local_record"][key], rel=1e-5, abs=1e-9), key

