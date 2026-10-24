"""Phase-6 macro GAE across frozen-opponent environment intervals."""
from __future__ import annotations

import torch

from guandan_marvel.phase6.policy_owner import segment_pooled_macros
from guandan_marvel.training.macro_gae import compute_macro_decision_gae

from tests.marvel_training._macro_fixtures import make_case

PLAYERS = [0, 1, 0, 1, 0, 1]
COMMITS = [True, True, True, True, True, True]
GAMMA = 0.9
LAMBDA = 0.8


def _rollout():
    rewards = {
        (0, 0): 1.0, (2, 0): 2.0, (4, 0): 3.0,
        (1, 1): 100.0, (3, 1): 100.0, (5, 1): 100.0,
    }
    values = {
        (0, 0): 0.5, (2, 0): 1.0, (4, 0): 1.5, (5, 0): 2.0,
        (0, 1): 50.0, (1, 1): 50.0, (2, 1): 50.0,
        (3, 1): 50.0, (4, 1): 50.0, (5, 1): 50.0,
    }
    rollout, value_tensor, _segmentation, _probs = make_case(
        PLAYERS, COMMITS, horizon=6, rewards=rewards, values=values,
        garbage_padding=False,
    )
    owner = torch.full_like(rollout["player_ids"], -1)
    train = torch.zeros_like(rollout["player_ids"], dtype=torch.bool)
    for step, player in enumerate(PLAYERS):
        seat_owner = 0 if player % 2 == 0 else 1
        owner[0, step] = seat_owner
        train[0, step] = seat_owner == 0
    rollout.set("policy_owner", owner)
    rollout.set("policy_train_mask", train)
    rollout.set("assigned_seats", torch.tensor([[[0, 1, 0, 1]] * 6], dtype=torch.long))
    rollout.set("assignment_id", torch.ones_like(rollout["player_ids"]))
    rollout.set("opponent_checkpoint_id", torch.where(train, -1, 0))
    rollout.set("opponent_policy_version", torch.where(train, -1, 3))
    rollout.set("on_policy_valid", train.clone())
    segmentation = segment_pooled_macros(rollout)
    return rollout, value_tensor, segmentation


def _reference_returns() -> tuple[list[float], list[float], list[float], list[float]]:
    """Independent scalar derivation (plain Python, explicit token windows)."""
    reward = [0.0] * 6
    reward[0], reward[2], reward[4] = 1.0, 2.0, 3.0
    value = {0: 0.5, 2: 1.0, 4: 1.5, 5: 2.0}
    starts = [0, 2, 4]
    ends = [2, 4, 6]
    reward_sum, delta = [], []
    for start, end in zip(starts, ends):
        window = 0.0
        for token in range(start, end):
            window += (GAMMA ** (token - start)) * reward[token]
        reward_sum.append(window)
        bootstrap = 0.0 if end == 6 else (GAMMA ** (end - start)) * value[end]
        delta.append(window + bootstrap - value[start])
    advantage = [0.0, 0.0, 0.0]
    advantage[2] = delta[2]
    advantage[1] = delta[1] + (GAMMA ** 2) * LAMBDA * advantage[2]
    advantage[0] = delta[0] + (GAMMA ** 2) * LAMBDA * advantage[1]
    targets = [value[start] + advantage[index] for index, start in enumerate(starts)]
    return reward_sum, delta, advantage, targets


def test_macro_gae_counts_opponent_tokens_in_the_discount_clock():
    rollout, values, segmentation = _rollout()
    gae = compute_macro_decision_gae(
        rollout, values, segmentation=segmentation,
        discount_gamma=GAMMA, macro_lambda=LAMBDA,
    )
    reward_sum, delta, advantage, targets = _reference_returns()
    slots = [0, 2, 4]  # player 0's macros in start order
    assert gae.duration[0, slots].tolist() == [2, 2, 2]
    assert gae.reward_sum[0, slots].tolist() == pytest_approx(reward_sum)
    assert gae.delta[0, slots].tolist() == pytest_approx(delta)
    assert gae.advantage[0, slots].tolist() == pytest_approx(advantage)
    assert gae.value_target[0, slots].tolist() == pytest_approx(targets)
    assert gae.is_last_for_player.tolist()[0][4] is True
    assert gae.continuation.tolist()[0][4] is False


def test_opponent_reward_and_value_channels_never_enter_the_current_targets():
    rollout, values, segmentation = _rollout()
    baseline = compute_macro_decision_gae(
        rollout, values, segmentation=segmentation,
        discount_gamma=GAMMA, macro_lambda=LAMBDA,
    )
    changed_rewards = rollout["rewards"].clone()
    changed_rewards[:, :, 1] = 999.0
    changed_values = values.clone()
    changed_values[:, :, 1] = -999.0
    changed_rollout = rollout.clone()
    changed_rollout.set("rewards", changed_rewards)
    changed = compute_macro_decision_gae(
        changed_rollout, changed_values, segmentation=segmentation,
        discount_gamma=GAMMA, macro_lambda=LAMBDA,
    )
    slots = [0, 2, 4]
    assert torch.equal(baseline.advantage[0, slots], changed.advantage[0, slots])
    assert torch.equal(baseline.value_target[0, slots], changed.value_target[0, slots])


def test_chance_token_and_hostile_padding_are_counted_and_ignored():
    # t1 is a trainer-chance step; t4/t5 are hostile padding outside num_steps.
    rollout, values, _legacy, _probs = make_case(
        [0, -1, 0, 1], [True, False, True, True], horizon=6,
        rewards={(0, 0): 1.0, (2, 0): 2.0, (3, 1): 100.0},
        values={(0, 0): 0.5, (2, 0): 1.0, (3, 0): 2.0},
        garbage_padding=True,
    )
    owner = torch.full_like(rollout["player_ids"], -1)
    train = torch.zeros_like(rollout["player_ids"], dtype=torch.bool)
    for step, player in enumerate([0, -1, 0, 1]):
        if player < 0:
            continue
        seat_owner = 0 if player == 0 else 1
        owner[0, step] = seat_owner
        train[0, step] = seat_owner == 0
    rollout.set("policy_owner", owner)
    rollout.set("policy_train_mask", train)
    rollout.set("assigned_seats", torch.tensor([[[0, 1, 0, 1]] * 6], dtype=torch.long))
    rollout.set("assignment_id", torch.ones_like(rollout["player_ids"]))
    rollout.set("opponent_checkpoint_id", torch.where(train, -1, 0))
    rollout.set("opponent_policy_version", torch.where(train, -1, 1))
    rollout.set("on_policy_valid", train.clone())
    segmentation = segment_pooled_macros(rollout)
    gae = compute_macro_decision_gae(
        rollout, values, segmentation=segmentation,
        discount_gamma=1.0, macro_lambda=0.5,
    )
    slots = [0, 1]  # player 0 macros at t0 and t2 in start order
    assert gae.duration[0, slots].tolist() == [2, 2]
    assert gae.reward_sum[0, slots].tolist() == pytest_approx([1.0, 2.0])
    assert gae.advantage[0, slots].tolist() == pytest_approx([2.0, 1.0])
    assert gae.value_target[0, slots].tolist() == pytest_approx([2.5, 2.0])
    assert gae.is_last_for_player.tolist()[0][1] is True


def pytest_approx(values):
    import pytest

    return pytest.approx(values, rel=1e-6, abs=1e-6)
