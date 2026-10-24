"""Phase-2 macro-decision GAE tests: spec section 10.3 (16 hand-computation cases)."""
from __future__ import annotations

import math

import pytest
import torch

from guandan_marvel.training.gae import compute_gae_from_value
from guandan_marvel.training.macro_gae import (
    MacroGAEError, compute_macro_decision_gae, require_macro_rollout_ready,
)

from tests.marvel_training._macro_fixtures import (
    P, make_case, scalar_macro_gae, scalar_token_gae,
)

DURATION_LOOKUP = {}


def _reference(rollout, values, *, gamma, lam):
    steps = int(rollout["num_steps"][0])
    players = [int(v) for v in rollout["player_ids"][0, :steps].tolist()]
    rewards = {
        (t, p): float(rollout["rewards"][0, t, p])
        for t in range(steps) for p in range(P)
        if float(rollout["rewards"][0, t, p]) != 0.0
    }
    values_map = {
        (t, p): float(values[0, t, p])
        for t in range(steps) for p in range(P)
        if float(values[0, t, p]) != 0.0
    }
    return players, rewards, values_map


def _assert_macro_matches_scalar(rollout, values, segmentation, *, gamma, lam):
    players, rewards, values_map = _reference(rollout, values, gamma=gamma, lam=lam)
    steps = int(rollout["num_steps"][0])
    macro_starts = [
        (int(segmentation.macro_start_index[0, m]), int(segmentation.macro_player[0, m]))
        for m in range(segmentation.max_macros)
        if bool(segmentation.macro_valid[0, m])
    ]
    expected = scalar_macro_gae(
        players, steps, rewards, values_map, gamma=gamma, lam=lam, macro_starts=macro_starts,
    )
    output = compute_macro_decision_gae(
        rollout, values, segmentation=segmentation, discount_gamma=gamma, macro_lambda=lam,
    )
    checked = 0
    for m in range(segmentation.max_macros):
        if not bool(segmentation.macro_valid[0, m]):
            continue
        start = int(segmentation.macro_start_index[0, m])
        player = int(segmentation.macro_player[0, m])
        assert output.advantage[0, m].item() == pytest.approx(
            expected[(start, player)], abs=1e-5,
        ), (m, start, player)
        checked += 1
    assert checked == int(segmentation.num_macros[0])
    return output


def _rotation_case(steps: int, *, values=None, rewards=None, gamma=0.99, lam=0.95):
    players = [t % 4 for t in range(steps)]
    commits = [True] * steps
    return make_case(players, commits, rewards=rewards, values=values)


def test_delta_4_7_3_hand_computed():
    """Spec 10.3.1: Delta = [4, 7, 3], gamma=0.99, lambda=0.95."""
    # Every token is a length-1 macro; player 0 commits at tokens 0, 4, 11.
    players = [0, 1, 2, 3, 0, 2, 2, 3, 1, 1, 2, 0, 1, 2]
    commits = [True] * 14
    n = 14
    rewards = {(0, 0): 1.0, (5, 0): 2.0, (13, 0): 3.0}
    values = {(0, 0): 0.1, (4, 0): 0.2, (11, 0): -0.3, (5, 0): 0.5}
    rollout, value_tensor, segmentation, _ = make_case(
        players[:n], commits[:n], rewards=rewards, values=values, horizon=n,
    )
    player0 = [
        int(segmentation.macro_start_index[0, m]) for m in range(segmentation.max_macros)
        if int(segmentation.macro_player[0, m]) == 0
    ]
    assert player0 == [0, 4, 11]
    output = _assert_macro_matches_scalar(
        rollout, value_tensor, segmentation, gamma=0.99, lam=0.95,
    )
    slot = {int(segmentation.macro_start_index[0, m]): m for m in range(segmentation.max_macros)}
    assert int(output.duration[0, slot[0]]) == 4
    assert int(output.duration[0, slot[4]]) == 7
    assert int(output.duration[0, slot[11]]) == 3  # num_steps - 11, not tau=0
    assert bool(output.continuation[0, slot[11]]) is False
    assert bool(output.is_last_for_player[0, slot[11]]) is True


def test_four_player_rotation_length_one_macros_delta_four_not_equivalent():
    """Spec 10.3.2: strict rotation, macro length 1 but Delta=4; not equal to legacy."""
    steps = 12
    values = {(0, 0): 0.0, (4, 0): 1.0, (8, 0): 2.0, (0, 1): 0.7, (4, 1): -0.4}
    rollout, value_tensor, segmentation, _ = _rotation_case(12, values=values)
    output = _assert_macro_matches_scalar(
        rollout, value_tensor, segmentation, gamma=0.99, lam=0.95,
    )
    legacy = compute_gae_from_value(
        rollout, value_tensor, discount_gamma=0.99, gae_lambda=0.95,
    ).player_advantage
    slot0 = 0  # macro id 0 starts at token 0 for player 0
    assert int(output.duration[0, slot0]) == 4
    assert abs(output.advantage[0, slot0].item() - legacy[0, 0].item()) > 1e-3
    # Explicit recursion coefficients: macro gamma^Delta*lambda vs legacy (gamma*lambda)^Delta.
    gamma, lam = 0.99, 0.95
    assert (gamma ** 4) * lam != pytest.approx((gamma * lam) ** 4)


def test_delta_one_degenerates_to_legacy():
    """Spec 10.3.3: Delta=1 makes the macro recursion equal the legacy token trace."""
    players = [0, 0, 2, 3]
    commits = [True, True, True, True]
    values = {(0, 0): 0.5, (1, 0): -0.2, (2, 0): 0.3}
    rewards = {(0, 0): 1.0, (3, 0): 2.0}
    rollout, value_tensor, segmentation, _ = make_case(
        players, commits, rewards=rewards, values=values,
    )
    output = _assert_macro_matches_scalar(
        rollout, value_tensor, segmentation, gamma=0.97, lam=0.9,
    )
    legacy = compute_gae_from_value(
        rollout, value_tensor, discount_gamma=0.97, gae_lambda=0.9,
    ).player_advantage
    assert int(output.duration[0, 0]) == 1  # player 0's first macro: same-player Delta = 1
    assert output.advantage[0, 0].item() == pytest.approx(legacy[0, 0].item(), abs=1e-6)
    # The last macro of player 0 has no successor and still matches legacy there.
    last = int(segmentation.num_macros[0]) - 1
    start = int(segmentation.macro_start_index[0, last])
    if int(segmentation.macro_player[0, last]) == 0:
        assert output.advantage[0, last].item() == pytest.approx(
            legacy[0, start].item(), abs=1e-6,
        )


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_lambda_endpoints(lam: float):
    players = [0, 0, 1, 0, 1]
    commits = [True, True, True, True, True]
    rewards = {(0, 0): 1.0, (3, 0): 2.0}
    values = {(0, 0): 0.2, (1, 0): 0.4, (3, 0): -0.1, (4, 0): 0.7}
    rollout, value_tensor, segmentation, _ = make_case(
        players, commits, rewards=rewards, values=values,
    )
    output = _assert_macro_matches_scalar(
        rollout, value_tensor, segmentation, gamma=0.95, lam=lam,
    )
    if lam == 0.0:
        # A_m = delta_m exactly.
        for m in range(segmentation.max_macros):
            if bool(segmentation.macro_valid[0, m]):
                assert output.advantage[0, m].item() == pytest.approx(
                    output.delta[0, m].item(), abs=1e-7,
                )


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.97, 1.0])
def test_gamma_grid(gamma: float):
    players = [0, -1, 0, 2, 0, 1]
    commits = [True, False, True, True, True, True]
    rewards = {(0, 0): 1.0, (3, 0): 4.0, (5, 0): 2.0}
    values = {(0, 0): 0.3, (2, 0): -0.5, (4, 0): 0.9, (5, 0): 1.5}
    rollout, value_tensor, segmentation, _ = make_case(
        players, commits, rewards=rewards, values=values, horizon=8,
    )
    output = _assert_macro_matches_scalar(
        rollout, value_tensor, segmentation, gamma=gamma, lam=0.9,
    )
    if gamma == 0.0:
        # 0**0 == 1: only the reward at the decision token survives.
        first = 0
        assert output.reward_sum[0, first].item() == pytest.approx(1.0, abs=1e-7)


def test_player_finished_early_terminal_reward_later_no_bootstrap():
    """Spec 10.3.8: reward window runs to num_steps, bootstrap is exactly 0."""
    players = [0, 0, 1, 1, 2, 3]
    commits = [True, True, True, True, True, True]
    rewards = {(5, 0): 10.0}
    values = {(0, 0): 0.1, (1, 0): 0.2, (5, 0): 7.0, (2, 1): -0.3}
    rollout, value_tensor, segmentation, _ = make_case(
        players, commits, rewards=rewards, values=values,
    )
    output = _assert_macro_matches_scalar(
        rollout, value_tensor, segmentation, gamma=0.99, lam=0.95,
    )
    slots = {
        int(segmentation.macro_start_index[0, m]): m
        for m in range(segmentation.max_macros)
        if bool(segmentation.macro_valid[0, m])
    }
    last = slots[1]
    assert int(output.duration[0, last]) == 5  # 1 -> num_steps=6
    assert bool(output.continuation[0, last]) is False
    # R = sum_{j=1}^{5} gamma^{j-1} r[j,0] = gamma^4 * 10 (no bootstrap despite V=7).
    assert output.reward_sum[0, last].item() == pytest.approx(0.99 ** 4 * 10.0, abs=1e-4)
    assert output.delta[0, last].item() == pytest.approx(
        0.99 ** 4 * 10.0 - 0.2, abs=1e-4,
    )
    first = slots[0]
    assert output.reward_sum[0, first].item() == pytest.approx(0.0, abs=1e-7)


def test_relay_changes_same_player_successor_distance():
    players = [0, 1, 0, 2, 3, 1, 0, 2]
    commits = [True, True, True, True, True, True, True, True]
    values = {(0, 0): 0.0, (2, 0): 0.1, (6, 0): 0.2}
    rollout, value_tensor, segmentation, _ = make_case(players, commits, values=values)
    output = _assert_macro_matches_scalar(
        rollout, value_tensor, segmentation, gamma=0.99, lam=0.95,
    )
    player0 = [
        int(segmentation.macro_start_index[0, m]) for m in range(segmentation.max_macros)
        if int(segmentation.macro_player[0, m]) == 0
    ]
    assert player0 == [0, 2, 6]
    slots = {
        int(segmentation.macro_start_index[0, m]): m for m in range(segmentation.max_macros)
    }
    assert int(output.duration[0, slots[0]]) == 2
    assert int(output.duration[0, slots[2]]) == 4
    assert int(output.duration[0, slots[6]]) == 2  # num_steps - 6


def test_terminal_reward_only_value_zero_decay_differs():
    """Spec 10.3.10: legacy decays (gamma*lambda)^tau, macro by gamma^Delta per decision."""
    players = [0, 1, 0, 1, 1, 1, 2, 3, 1]
    commits = [True, True, True, True, False, True, True, True, True]
    # player 0 macro starts: 0 (len 1), 2 (len 1); terminal reward at token 8
    rewards = {(8, 0): 1.0}
    n = 9
    rollout, value_tensor, segmentation, _ = make_case(
        players, commits, rewards=rewards, horizon=n,
    )
    gamma, lam = 0.99, 0.95
    output = _assert_macro_matches_scalar(
        rollout, value_tensor, segmentation, gamma=gamma, lam=lam,
    )
    legacy = compute_gae_from_value(
        rollout, value_tensor, discount_gamma=gamma, gae_lambda=lam,
    ).player_advantage
    slots = {int(segmentation.macro_start_index[0, m]): m for m in range(segmentation.max_macros)}
    macro_first = output.advantage[0, slots[0]].item()
    legacy_first = legacy[0, 0].item()
    # macro: gamma^8 * lambda (one lambda per same-player macro decision: A(2)=gamma^6
    # enters A(0) with gamma^2*lambda); legacy: two token hops 0->2 and 2->num_steps,
    # each carrying (gamma*lambda) -> gamma^8 * lambda^2.
    expected_macro = (gamma ** 2) * lam * (gamma ** 6)
    expected_legacy = (gamma * lam) ** 2 * (gamma ** 6)
    assert macro_first == pytest.approx(expected_macro, abs=1e-5)
    assert legacy_first == pytest.approx(expected_legacy, abs=1e-5)
    assert abs(macro_first - legacy_first) == pytest.approx(
        (gamma ** 8) * (lam - lam * lam), abs=1e-6,
    )


def test_multi_batch_different_num_steps():
    first = _rotation_case(8, values={(0, 0): 0.4, (4, 0): -0.2}, gamma=0.99, lam=0.95)
    second = _rotation_case(8, values={(0, 0): 0.9, (4, 0): 0.1}, gamma=0.99, lam=0.95)
    rollout_a, values_a, seg_a, _ = first
    rollout_b, values_b, seg_b, _ = second
    from guandan_marvel.model.tensordict import TensorDict

    batch = 2
    fields = {}
    for name in rollout_a.keys():
        tensor_a = rollout_a[name]
        tensor_b = rollout_b[name]
        if tensor_a.dim() == 1:
            joined = torch.stack([tensor_a, tensor_b])
        else:
            joined = torch.cat([tensor_a, tensor_b], dim=0)
        fields[name] = joined
    fields["num_steps"] = torch.tensor([5, 8], dtype=torch.long)
    fields["is_terminal"] = fields["is_terminal"].clone()
    fields["is_terminal"][0, 4] = True
    fields["is_terminal"][0, 5:] = False
    batch_rollout = TensorDict(fields, batch_size=[batch])
    values = torch.cat([values_a, values_b], dim=0)
    from guandan_marvel.training.macro_actions import segment_macro_actions

    segmentation = segment_macro_actions(
        num_steps=batch_rollout["num_steps"],
        player_ids=batch_rollout["player_ids"],
        action_taken=batch_rollout["action_taken"],
        avail_is_commit=batch_rollout["avail_is_commit"],
        behavior_action_prob=batch_rollout["behavior_action_prob"],
        is_terminal=batch_rollout["is_terminal"],
    )
    output = compute_macro_decision_gae(
        batch_rollout, values, segmentation=segmentation, discount_gamma=0.99, macro_lambda=0.95,
    )
    for b in range(batch):
        steps = int(batch_rollout["num_steps"][b])
        players = [int(v) for v in batch_rollout["player_ids"][b, :steps].tolist()]
        rewards = {
            (t, p): float(batch_rollout["rewards"][b, t, p])
            for t in range(steps) for p in range(P)
            if float(batch_rollout["rewards"][b, t, p]) != 0.0
        }
        values_map = {
            (t, p): float(values[b, t, p])
            for t in range(steps) for p in range(P)
            if float(values[b, t, p]) != 0.0
        }
        expected = scalar_macro_gae(players, steps, rewards, values_map, gamma=0.99, lam=0.95)
        for m in range(segmentation.max_macros):
            if not bool(segmentation.macro_valid[b, m]):
                continue
            start = int(segmentation.macro_start_index[b, m])
            player = int(segmentation.macro_player[b, m])
            assert output.advantage[b, m].item() == pytest.approx(
                expected[(start, player)], abs=1e-5,
            )


def test_hostile_padding_reward_value_player_action_does_not_change_results():
    players = [0, 0, 1, 2]
    commits = [False, True, True, True]
    values = {(0, 0): 0.5, (1, 0): -0.1, (2, 1): 0.4}
    rewards = {(0, 0): 1.0, (3, 0): 3.0}
    clean_rollout, clean_values, clean_seg, _ = make_case(
        players, commits, rewards=rewards, values=values, horizon=4, garbage_padding=False,
    )
    dirty_rollout, dirty_values, dirty_seg, _ = make_case(
        players, commits, rewards=rewards, values=values, horizon=16, garbage_padding=True,
    )
    clean = compute_macro_decision_gae(
        clean_rollout, clean_values, segmentation=clean_seg, discount_gamma=0.99, macro_lambda=0.95,
    )
    dirty = compute_macro_decision_gae(
        dirty_rollout, dirty_values, segmentation=dirty_seg, discount_gamma=0.99, macro_lambda=0.95,
    )
    assert torch.equal(clean.advantage, dirty.advantage)
    assert torch.equal(clean.value_target, dirty.value_target)
    assert torch.equal(clean.duration, dirty.duration)


def test_next_value_only_same_player_next_macro_start():
    players = [0, 1, 0, 2]
    commits = [True, True, True, True]
    values = {(0, 0): 0.0, (2, 0): 0.5}
    rollout, value_tensor, segmentation, _ = make_case(players, commits, values=values, horizon=6)
    baseline = compute_macro_decision_gae(
        rollout, value_tensor, segmentation=segmentation, discount_gamma=0.99, macro_lambda=0.95,
    )
    # Hostile value on another player's macro start must not leak into player 0.
    hostile = value_tensor.clone()
    hostile[0, 1, :] = 123.0
    perturbed = compute_macro_decision_gae(
        rollout, hostile, segmentation=segmentation, discount_gamma=0.99, macro_lambda=0.95,
    )
    player0_slots = [
        m for m in range(segmentation.max_macros)
        if int(segmentation.macro_player[0, m]) == 0
    ]
    for m in player0_slots:
        assert baseline.advantage[0, m].item() == pytest.approx(
            perturbed.advantage[0, m].item(), abs=1e-7,
        )
    # And the explicit scalar reference pins the bootstrap source.
    _assert_macro_matches_scalar(rollout, value_tensor, segmentation, gamma=0.99, lam=0.95)


def test_no_successor_uses_num_steps_start_not_tau_zero():
    players = [0, 1, 2, 3]
    commits = [True, True, True, True]
    rewards = {(3, 0): 5.0}
    values = {(0, 0): 1.0, (3, 0): 2.0}
    rollout, value_tensor, segmentation, _ = make_case(players, commits, rewards=rewards, values=values)
    gamma = 0.99
    output = compute_macro_decision_gae(
        rollout, value_tensor, segmentation=segmentation, discount_gamma=gamma, macro_lambda=0.95,
    )
    slot = 0
    assert int(segmentation.macro_tau[0, slot]) == 0  # phase-1 keeps tau=0 for the last macro
    assert int(output.duration[0, slot]) == 4  # num_steps - start, never tau
    assert output.reward_sum[0, slot].item() == pytest.approx(gamma ** 3 * 5.0, abs=1e-5)
    # The wrong interpretation (duration 0) would drop the whole window.
    wrong = 0.0 - float(value_tensor[0, 0, 0])
    assert abs(output.delta[0, slot].item() - wrong) > 1.0
    _assert_macro_matches_scalar(rollout, value_tensor, segmentation, gamma=gamma, lam=0.95)


def test_truncated_incomplete_is_rejected():
    players = [0, 0, 1]
    commits = [False, False, True]  # player 0 never commits -> incomplete macro
    rollout, value_tensor, segmentation, _ = make_case(players, commits)
    assert segmentation.incomplete_count > 0
    with pytest.raises(MacroGAEError, match="incomplete_count"):
        compute_macro_decision_gae(
            rollout, value_tensor, segmentation=segmentation, discount_gamma=0.99, macro_lambda=0.95,
        )
    rollout_ok, value_ok, segmentation_ok, _ = make_case([0, 1], [True, True])
    rollout_ok["is_terminal"][0, 1] = False
    with pytest.raises(MacroGAEError, match="real terminal"):
        require_macro_rollout_ready(rollout_ok, segmentation_ok)
    with pytest.raises(MacroGAEError, match="real terminal"):
        compute_macro_decision_gae(
            rollout_ok, value_ok, segmentation=segmentation_ok, discount_gamma=0.99, macro_lambda=0.95,
        )


def test_float32_vs_float64_scalar_reference_tolerance():
    players = [0, 0, 1, 0]
    commits = [True, True, True, True]
    rewards = {(0, 0): 1.0, (3, 0): 2.0}
    values = {(0, 0): 0.31415926535, (1, 0): -0.2718281828, (3, 0): 0.5}
    rollout32, values32, seg32, _ = make_case(
        players, commits, rewards=rewards, values=values, dtype=torch.float32,
    )
    rollout64, values64, seg64, _ = make_case(
        players, commits, rewards=rewards, values=values, dtype=torch.float64,
    )
    gamma, lam = 0.97, 0.93
    out32 = compute_macro_decision_gae(
        rollout32, values32, segmentation=seg32, discount_gamma=gamma, macro_lambda=lam,
    )
    out64 = compute_macro_decision_gae(
        rollout64, values64, segmentation=seg64, discount_gamma=gamma, macro_lambda=lam,
    )
    assert out32.advantage.dtype == torch.float32
    reference = _reference(rollout64, values64, gamma=gamma, lam=lam)
    expected = scalar_macro_gae(*[reference[0], int(rollout64["num_steps"][0]), reference[1], reference[2]], gamma=gamma, lam=lam)
    for m in range(seg64.max_macros):
        if not bool(seg64.macro_valid[0, m]):
            continue
        start = int(seg64.macro_start_index[0, m])
        player = int(seg64.macro_player[0, m])
        assert abs(out32.advantage[0, m].item() - expected[(start, player)]) < 1e-5
        assert abs(out64.advantage[0, m].item() - expected[(start, player)]) < 1e-5
    assert torch.allclose(out32.advantage, out64.advantage, atol=1e-6)


def test_chance_environment_and_other_player_tokens_counted_in_delta():
    players = [0, -1, -2, 0, 1, 0, 1]
    commits = [False, False, False, True, True, True, True]
    rollout, value_tensor, segmentation, _ = make_case(players, commits, horizon=7)
    output = compute_macro_decision_gae(
        rollout, value_tensor, segmentation=segmentation, discount_gamma=0.99, macro_lambda=0.95,
    )
    slots = {int(segmentation.macro_start_index[0, m]): m for m in range(segmentation.max_macros)}
    assert int(output.duration[0, slots[0]]) == 5  # chance/env/other tokens counted
    assert int(output.duration[0, slots[5]]) == 2  # num_steps(7) - 5
    assert bool(output.continuation[0, slots[0]]) is True
    assert bool(output.continuation[0, slots[5]]) is False
