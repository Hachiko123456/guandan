"""Phase-2 sequence PPO tests: spec section 10.1 (and normalization parts of 10.4)."""
from __future__ import annotations

import math

import pytest
import torch

from guandan_marvel.training.losses import compute_clipped_pg_loss
from guandan_marvel.training.macro_ppo import (
    MACRO_LOG_RATIO_SAFE_MAX, MacroPPOError, build_macro_batch_view, compute_macro_pg_loss,
    macro_active_token_mask,
)

from tests.marvel_training._macro_fixtures import (
    make_case, scalar_macro_pg_loss, scalar_token_pg_loss,
)

CLIP = 0.2


def _view(segmentation, indices=None, batch=1):
    if indices is None:
        indices = torch.arange(batch)
    return build_macro_batch_view(segmentation, indices)


def test_single_token_macro_sequence_equals_token_ppo_per_sample():
    players = [0, 1, 2, 3]
    commits = [True] * 4
    new_rows = [
        [0.65, 0.35, 0.0, 0.0],
        [0.45, 0.55, 0.0, 0.0],
        [0.58, 0.42, 0.0, 0.0],
        [0.50, 0.50, 0.0, 0.0],
    ]
    rollout, _values, segmentation, new_probs = make_case(
        players, commits, behavior=[0.5] * 4, avail_probs_rows=new_rows,
    )
    assert int(segmentation.num_macros[0]) == 4
    assert all(int(segmentation.macro_length[0, m]) == 1 for m in range(4))
    macro_advantage = torch.tensor([[1.0, -0.5, 0.25, -0.75]], dtype=torch.float32)
    output = compute_macro_pg_loss(
        rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
        macro_advantage=macro_advantage, clip_coef=CLIP,
    )
    log_ratios = [math.log(row[0]) - math.log(0.5) for row in new_rows]
    expected = scalar_macro_pg_loss(log_ratios, macro_advantage[0].tolist(), clip_coef=CLIP)
    assert output.pg_loss_mean.item() == pytest.approx(expected, rel=0, abs=1e-6)
    expected_log_ratios = torch.tensor(log_ratios, dtype=torch.float32)
    assert torch.allclose(output.log_ratio[0], expected_log_ratios, atol=1e-6)
    assert torch.allclose(output.ratio[0], expected_log_ratios.exp(), atol=1e-6)
    assert output.macro_count_global == 4
    assert output.macro_length_max == 1
    # A length-1 macro clips exactly like the token ratio would.
    assert output.clip_frac == pytest.approx(1 / 4)


def test_two_token_macro_ratio_equals_product_of_token_ratios():
    players = [0, 0, 1]
    commits = [False, True, True]
    rows = [
        [0.4, 0.6, 0.0, 0.0],   # selected action 1: 0.6 / 0.5 = 1.2
        [0.425, 0.575, 0.0, 0.0],  # selected action 0: 0.85
        [0.5, 0.5, 0.0, 0.0],
    ]
    rollout, _values, segmentation, new_probs = make_case(
        players, commits, behavior=[0.5] * 3, avail_probs_rows=rows,
    )
    macro_advantage = torch.tensor([[0.4, -0.4]], dtype=torch.float32)
    output = compute_macro_pg_loss(
        rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
        macro_advantage=macro_advantage, clip_coef=CLIP,
    )
    expected_log_ratio = math.log(0.6 / 0.5) + math.log(0.425 / 0.5)
    expected_ratio = (0.6 / 0.5) * (0.425 / 0.5)
    assert output.log_ratio[0, 0].item() == pytest.approx(expected_log_ratio, abs=1e-6)
    assert output.ratio[0, 0].item() == pytest.approx(expected_ratio, abs=1e-6)
    expected = scalar_macro_pg_loss(
        [expected_log_ratio, 0.0], [0.4, -0.4], clip_coef=CLIP,
    )
    assert output.pg_loss_mean.item() == pytest.approx(expected, abs=1e-6)


def test_three_token_macro_logprob_is_sum_of_token_logprobs():
    players = [0, 0, 0, 1]
    commits = [False, False, True, True]
    rows = [
        [0.4, 0.6, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.55, 0.45, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ]
    rollout, _values, segmentation, new_probs = make_case(
        players, commits, behavior=[0.5] * 4, avail_probs_rows=rows,
    )
    macro_advantage = torch.tensor([[0.3, -0.3]], dtype=torch.float32)
    output = compute_macro_pg_loss(
        rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
        macro_advantage=macro_advantage, clip_coef=CLIP,
    )
    assert int(segmentation.macro_length[0, 0]) == 3
    new_sum = math.log(0.6) + math.log(0.5) + math.log(0.55)
    old_sum = 3 * math.log(0.5)
    assert output.log_ratio[0, 0].item() == pytest.approx(new_sum - old_sum, abs=1e-6)


def test_token_clip_versus_macro_clip_counterexamples():
    """Positive/negative advantage cases where per-token and macro clipping differ."""
    players = [0, 0, 1]
    commits = [False, True, True]
    # token ratios: 1.35 (would clip) and 0.85 -> macro ratio 1.1475 (no clip)
    rows = [
        [0.325, 0.675, 0.0, 0.0],
        [0.425, 0.575, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ]
    rollout, _values, segmentation, new_probs = make_case(
        players, commits, behavior=[0.5] * 3, avail_probs_rows=rows,
    )
    macro_ratio = 1.35 * 0.85
    assert not (abs(macro_ratio - 1.0) > CLIP)
    for raw_advantage in (1.5, -1.5):
        advantage = torch.tensor([[raw_advantage, -raw_advantage]], dtype=torch.float32)
        output = compute_macro_pg_loss(
            rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
            macro_advantage=advantage, clip_coef=CLIP,
        )
        assert output.ratio[0, 0].item() == pytest.approx(macro_ratio, abs=1e-6)
        normalized = raw_advantage / (abs(raw_advantage) + 1e-8)  # +/-1 after 2-sample normalization
        macro_clipped = min(max(macro_ratio, 1 - CLIP), 1 + CLIP)
        surrogate_macro = max(-normalized * macro_ratio, -normalized * macro_clipped)
        surrogate_other = -(-normalized)  # ratio 1.0 macro with the opposite advantage
        expected = (surrogate_macro + surrogate_other) / 2
        assert output.pg_loss_mean.item() == pytest.approx(expected, abs=1e-6)
        # The per-token clipping variant is genuinely different.
        per_token = 0.0
        for ratio in (1.35, 0.85):
            clipped = min(max(ratio, 1 - CLIP), 1 + CLIP)
            per_token += max(-normalized * ratio, -normalized * clipped)
        per_token /= 2
        alternative = (per_token + surrogate_other) / 2
        assert abs(alternative - output.pg_loss_mean.item()) > 1e-3


@pytest.mark.parametrize("length", [1, 2, 8, 10, 16, 32])
def test_macro_lengths_cover_one_to_thirty_two(length: int):
    players = [0] * length + [1]
    commits = [False] * (length - 1) + [True, True]
    behavior = [0.5] * (length + 1)
    rows = [[0.5, 0.5, 0.0, 0.0] for _ in range(length + 1)]
    expected_log_ratio = 0.0
    if length >= 3:
        # token 1 (non-final, chosen action 1): 0.6 / 0.5
        rows[1] = [0.4, 0.6, 0.0, 0.0]
        expected_log_ratio += math.log(0.6 / 0.5)
    if length >= 5:
        # token 3 (non-final, chosen action 1): 0.4 / 0.5
        rows[3] = [0.6, 0.4, 0.0, 0.0]
        expected_log_ratio += math.log(0.4 / 0.5)
    rollout, _values, segmentation, new_probs = make_case(
        players, commits, behavior=behavior, avail_probs_rows=rows,
    )
    assert int(segmentation.macro_length[0, 0]) == length
    advantage = torch.tensor([[0.75, -0.75]], dtype=torch.float32)
    output = compute_macro_pg_loss(
        rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
        macro_advantage=advantage, clip_coef=CLIP,
    )
    assert output.log_ratio[0, 0].item() == pytest.approx(expected_log_ratio, abs=1e-5)
    expected = scalar_macro_pg_loss(
        [expected_log_ratio, 0.0], [0.75, -0.75], clip_coef=CLIP,
    )
    assert output.pg_loss_mean.item() == pytest.approx(expected, abs=1e-5)
    assert output.macro_count_global == 2


def test_long_macro_counts_as_one_policy_sample_and_equal_weight():
    """A 12-token macro and a 2-token macro receive the same policy weight."""
    players = [0] * 12 + [1, 1, 2]
    commits = [False] * 11 + [True, False, True, True]
    rows = [[0.5, 0.5, 0.0, 0.0] for _ in range(15)]
    rows[0] = [0.45, 0.55, 0.0, 0.0]   # long macro: 0.55 / 0.5
    rows[1] = [0.55, 0.45, 0.0, 0.0]   # long macro: 0.45 / 0.5
    rows[14] = [0.45, 0.55, 0.0, 0.0]  # player 2 macro ratio 0.9
    rollout, _values, segmentation, new_probs = make_case(
        players, commits, behavior=[0.5] * 15, avail_probs_rows=rows,
    )
    assert [int(segmentation.macro_length[0, m]) for m in range(3)] == [12, 2, 1]
    advantage = torch.tensor([[2.0, -1.0, 0.0]], dtype=torch.float32)
    output = compute_macro_pg_loss(
        rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
        macro_advantage=advantage, clip_coef=CLIP,
    )
    expected_macro = scalar_macro_pg_loss(
        [output.log_ratio[0, 0].item(), output.log_ratio[0, 1].item(), output.log_ratio[0, 2].item()],
        [2.0, -1.0, 0.0], clip_coef=CLIP,
    )
    assert output.pg_loss_mean.item() == pytest.approx(expected_macro, abs=1e-6)
    # The long macro is one of exactly three policy samples.
    assert output.macro_count_global == 3
    token_weighted = (float(output.ratio[0, 0]) * 12 + float(output.ratio[0, 1]) * 2 + 1.0) / 15
    assert abs(token_weighted - output.ratio_mean) > 1e-2


def test_padding_and_chance_tokens_are_excluded():
    players = [0, -1, 0, 1]
    commits = [False, False, True, True]
    base = dict(players=players, commits=commits, behavior=[0.5] * 4)
    clean_rollout, _v, clean_seg, clean_new = make_case(**base, garbage_padding=False)
    dirty_rollout, _v2, dirty_seg, dirty_new = make_case(**base, garbage_padding=True, horizon=12)
    for seg in (clean_seg, dirty_seg):
        assert int(seg.num_macros[0]) == 2
        assert int(seg.macro_length[0, 0]) == 2  # the chance token is not a macro token
        assert seg.incomplete_count == 0
    assert clean_seg.player_switch_violation_count == dirty_seg.player_switch_violation_count == 0
    advantage = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    clean = compute_macro_pg_loss(
        clean_rollout, view=_view(clean_seg), avail_log_probs_new=clean_new.log(),
        macro_advantage=advantage, clip_coef=CLIP,
    )
    dirty = compute_macro_pg_loss(
        dirty_rollout, view=_view(dirty_seg), avail_log_probs_new=dirty_new.log(),
        macro_advantage=advantage, clip_coef=CLIP,
    )
    assert torch.equal(clean.ratio, dirty.ratio)
    assert clean.pg_loss_mean.item() == pytest.approx(dirty.pg_loss_mean.item(), abs=0.0)


def test_inactive_garbage_action_indices_do_not_overflow():
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits, horizon=8)
    rollout["action_taken"][0, 2] = -1
    rollout["action_taken"][0, 3] = 99
    rollout["action_taken"][0, 4] = 999
    rollout["action_taken"][0, 5] = -7
    advantage = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
    output = compute_macro_pg_loss(
        rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
        macro_advantage=advantage, clip_coef=CLIP,
    )
    assert output.macro_count_global == 2


def test_active_action_index_out_of_range_fails_fast():
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits)
    rollout["action_taken"][0, 0] = 4
    with pytest.raises(MacroPPOError, match="outside the available-action range"):
        compute_macro_pg_loss(
            rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
            macro_advantage=torch.zeros((1, 2)), clip_coef=CLIP,
        )


def test_active_action_unavailable_fails_fast():
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits, legal=[[0], [0]])
    rollout["action_taken"][0, 0] = 2  # in range but weight 0
    with pytest.raises(MacroPPOError, match="illegal/unavailable action"):
        compute_macro_pg_loss(
            rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
            macro_advantage=torch.zeros((1, 2)), clip_coef=CLIP,
        )


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_behavior_probability_invalid_fails_fast(bad: float):
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits)
    rollout["behavior_action_prob"][0, 0] = bad
    with pytest.raises(MacroPPOError, match="behavior_action_prob"):
        compute_macro_pg_loss(
            rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
            macro_advantage=torch.zeros((1, 2)), clip_coef=CLIP,
        )


def test_off_policy_macro_token_fails_fast():
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits, on_policy=[False, True])
    with pytest.raises(MacroPPOError, match="on-policy"):
        compute_macro_pg_loss(
            rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
            macro_advantage=torch.zeros((1, 2)), clip_coef=CLIP,
        )


def test_behavior_reference_mismatch_fails_fast():
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(
        players, commits, behavior=[0.5, 0.5], policy_probs_override=[0.3, 0.5],
    )
    with pytest.raises(MacroPPOError, match="reference policy"):
        compute_macro_pg_loss(
            rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
            macro_advantage=torch.zeros((1, 2)), clip_coef=CLIP,
        )


def test_non_finite_new_logprob_fails_fast():
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits)
    log_probs = new_probs.log()
    log_probs[0, 0, 0] = float("nan")
    with pytest.raises(MacroPPOError, match="non-finite"):
        compute_macro_pg_loss(
            rollout, view=_view(segmentation), avail_log_probs_new=log_probs,
            macro_advantage=torch.zeros((1, 2)), clip_coef=CLIP,
        )


def test_extreme_positive_log_ratio_fails_fast_with_details_and_no_clamp():
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits)
    log_probs = new_probs.log().clone()
    log_probs[0, 0, 0] = math.log(0.5) + MACRO_LOG_RATIO_SAFE_MAX + 5.0
    with pytest.raises(MacroPPOError) as info:
        compute_macro_pg_loss(
            rollout, view=_view(segmentation), avail_log_probs_new=log_probs,
            macro_advantage=torch.zeros((1, 2)), clip_coef=CLIP,
        )
    message = str(info.value)
    assert "FP32 exp safety range" in message
    assert "batch=0" in message and "length=1" in message
    assert "no silent clamp" in message
    # Inside the safe band the same sample trains normally (no clamp needed).
    safe_log_probs = new_probs.log().clone()
    safe_log_probs[0, 0, 0] = math.log(0.5) + 5.0
    output = compute_macro_pg_loss(
        rollout, view=_view(segmentation), avail_log_probs_new=safe_log_probs,
        macro_advantage=torch.zeros((1, 2)), clip_coef=CLIP,
    )
    assert bool(torch.isfinite(output.pg_loss_mean).all())


def test_mini_batch_view_equals_full_batch_index_gather():
    players = [0, 0, 1, 2, 2, 3]
    commits = [False, True, True, False, True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits)
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    for name in (
        "token_to_macro_id", "macro_start_index", "macro_end_index", "macro_length",
        "macro_player", "macro_valid", "macro_tau", "macro_is_terminal",
        "macro_is_last_for_player",
    ):
        full = getattr(segmentation, name)
        sliced = getattr(view, name)
        assert torch.equal(sliced, full[torch.tensor([0])]), name
    active_full = macro_active_token_mask(rollout, view)
    assert torch.equal(active_full, torch.ones((1, 6), dtype=torch.bool))


def test_token_normalization_convention_is_population_for_macro_profiles():
    """Single macro / zero-variance normalization is finite zero (no NaN)."""
    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits)
    advantage = torch.tensor([[3.0, 3.0]], dtype=torch.float32)
    output = compute_macro_pg_loss(
        rollout, view=_view(segmentation), avail_log_probs_new=new_probs.log(),
        macro_advantage=advantage, clip_coef=CLIP,
    )
    assert output.zero_variance_count == 1
    # zero-variance normalization -> both advantages are exactly zero.
    assert output.pg_loss_mean.item() == pytest.approx(0.0, abs=0.0)
    assert bool(torch.isfinite(output.pg_loss_mean).all())


def test_empty_macro_batch_fails_fast_without_zero_division():
    import dataclasses

    players = [0, 1]
    commits = [True, True]
    rollout, _values, segmentation, new_probs = make_case(players, commits)
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    empty_view = dataclasses.replace(
        view, macro_valid=torch.zeros_like(view.macro_valid),
    )
    with pytest.raises(MacroPPOError, match="no valid trainable macros"):
        compute_macro_pg_loss(
            rollout, view=empty_view, avail_log_probs_new=new_probs,
            macro_advantage=torch.zeros((1, 2)), clip_coef=CLIP,
        )


def test_legacy_token_pg_loss_matches_independent_scalar_reference():
    """Profile A regression: production token PPO equals a scalar reference."""
    players = [0, 1, 0, 1]
    commits = [False, True, True, True]
    rows = [
        [0.35, 0.65, 0.0, 0.0],
        [0.45, 0.55, 0.0, 0.0],
        [0.60, 0.40, 0.0, 0.0],
        [0.50, 0.50, 0.0, 0.0],
    ]
    chosen = [1, 0, 0, 0]
    rollout, _values, _segmentation, new_probs = make_case(
        players, commits, behavior=[0.5] * 4, avail_probs_rows=rows,
    )
    advantage = torch.tensor([[1.0, -0.5, 0.25, -0.75]], dtype=torch.float32)
    output = compute_clipped_pg_loss(
        rollout, clip_coef=CLIP, avail_probs_new=new_probs, player_advantage=advantage,
    )
    values = [1.0, -0.5, 0.25, -0.75]
    mean = sum(values) / 4
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / 3)  # legacy unbiased std
    total = 0.0
    for index, (row, token_adv) in enumerate(zip(rows, values)):
        ratio = row[chosen[index]] / 0.5
        normalized = (token_adv - mean) / (std + 1e-8)
        clipped = min(max(ratio, 1 - CLIP), 1 + CLIP)
        total += max(-normalized * ratio, -normalized * clipped)
    assert output.pg_loss_mean.item() == pytest.approx(total / 4, abs=1e-5)
    population = scalar_token_pg_loss(
        [math.log(row[chosen[i]] / 0.5) for i, row in enumerate(rows)], values, clip_coef=CLIP,
    )
    assert abs(output.pg_loss_mean.item() - population) < 0.2
