"""Batch rows are independent episodes; attribution stays constant within each.

Only tiny CPU tensors and the real validator are used: no model, sampler, or
training fixture is needed to exercise the production batch-routing contract.
"""
from __future__ import annotations

import pytest
import torch

from guandan_marvel.phase6.policy_owner import (
    PooledTrajectoryStats,
    validate_pooled_trajectory,
)


def _batch(
    batch_size=2, *, players=None, horizon=None, teams=None, owners=None,
    assignments=None,
):
    """Build just the validator's real fields, with one legal team per row."""
    if players is None:
        players = [[0, 1, 2, 3, 0] for _ in range(batch_size)]
    assert len(players) == batch_size
    length = max(map(len, players)) if horizon is None else horizon
    teams = [b % 2 for b in range(batch_size)] if teams is None else teams
    owners = [3, 7, 11, 19][:batch_size] if owners is None else owners
    assignments = [101 + b for b in range(batch_size)] if assignments is None else assignments
    shape = (batch_size, length)
    rollout = {
        "num_steps": torch.tensor([len(row) for row in players], dtype=torch.long),
        "player_ids": torch.full(shape, -1, dtype=torch.long),
        "policy_owner": torch.full(shape, -1, dtype=torch.long),
        "policy_train_mask": torch.zeros(shape, dtype=torch.bool),
        "opponent_checkpoint_id": torch.full(shape, -1, dtype=torch.long),
        "opponent_policy_version": torch.full(shape, -1, dtype=torch.long),
        "assignment_id": torch.full(shape, -1, dtype=torch.long),
        "assigned_seats": torch.full((*shape, 4), -1, dtype=torch.long),
        "behavior_action_prob": torch.full(shape, float("nan")),
        "on_policy_valid": torch.zeros(shape, dtype=torch.bool),
    }
    for b, row in enumerate(players):
        seats = torch.tensor(
            [0 if seat % 2 == teams[b] else owners[b] for seat in range(4)],
            dtype=torch.long,
        )
        for t, player in enumerate(row):
            rollout["player_ids"][b, t] = player
            if player < 0:
                continue
            owner = int(seats[player])
            current = owner == 0
            rollout["policy_owner"][b, t] = owner
            rollout["policy_train_mask"][b, t] = current
            rollout["on_policy_valid"][b, t] = current
            rollout["opponent_checkpoint_id"][b, t] = -1 if current else 20 + b
            rollout["opponent_policy_version"][b, t] = -1 if current else 70 + b
            rollout["assignment_id"][b, t] = assignments[b]
            rollout["assigned_seats"][b, t] = seats
            rollout["behavior_action_prob"][b, t] = 0.25 if current else 0.125
    return rollout


@pytest.mark.parametrize("batch_size", [2, 4])
def test_independent_teams_and_frozen_identities_have_exact_stats(batch_size):
    rollout = _batch(batch_size)
    # The same physical seat is current in one episode, frozen in the next.
    assert rollout["assigned_seats"][0, 0].tolist() == [0, 3, 0, 3]
    assert rollout["assigned_seats"][1, 0].tolist() == [7, 0, 7, 0]
    assert sorted(set(rollout["opponent_checkpoint_id"].flatten().tolist())) == [
        -1, *range(20, 20 + batch_size),
    ]
    assert sorted(set(rollout["opponent_policy_version"].flatten().tolist())) == [
        -1, *range(70, 70 + batch_size),
    ]
    assert validate_pooled_trajectory(rollout) == PooledTrajectoryStats(
        current_token_count=5 * batch_size // 2,
        opponent_token_count=5 * batch_size // 2,
        current_assignment_count=batch_size,
        opponent_owner_ids=tuple([3, 7, 11, 19][:batch_size]),
        current_macro_count=0,
        opponent_macro_count=0,
    )


def test_assignment_and_owner_stats_count_unique_values_across_rows():
    rollout = _batch(4, owners=[7, 7, 3, 3], assignments=[91, 91, 17, 17])
    assert validate_pooled_trajectory(rollout) == PooledTrajectoryStats(
        10, 10, 2, (3, 7), 0, 0,
    )


def test_unequal_episode_lengths_ignore_padding_seats_and_metadata():
    rollout = _batch(
        4, players=[[0, -1, 1, 2, 3, 0], [0, 1, -1, 3], [1, 2], [1]],
        horizon=8,
    )
    expected = PooledTrajectoryStats(7, 4, 4, (3, 7, 11), 0, 0)
    assert validate_pooled_trajectory(rollout) == expected
    for b, count in enumerate(rollout["num_steps"].tolist()):
        for t in range(count, 8):
            # Positive player ids ensure num_steps, rather than player sign,
            # excludes padding. Both ownerless and different valid seats occur.
            rollout["player_ids"][b, t] = t % 4
            rollout["assigned_seats"][b, t] = torch.tensor(
                [-1, -1, -1, -1] if t % 2 else [43, 0, 43, 0],
            )
            rollout["assignment_id"][b, t] = 1000 + t
            rollout["opponent_checkpoint_id"][b, t] = 2000 + t
            rollout["opponent_policy_version"][b, t] = 3000 + t
    assert validate_pooled_trajectory(rollout) == expected


def test_chance_tokens_do_not_enter_constancy_checks_or_stats():
    rollout = _batch(players=[[-1, 0, -1, 1, -1], [-1, 1, -1, 0, -1]])
    expected = PooledTrajectoryStats(2, 2, 2, (3, 7), 0, 0)
    assert validate_pooled_trajectory(rollout) == expected
    chance = rollout["player_ids"] < 0
    rollout["assignment_id"][chance] = torch.arange(6) + 900
    rollout["assigned_seats"][chance] = torch.tensor([0, 29, 0, 29])
    assert validate_pooled_trajectory(rollout) == expected


def test_in_episode_seat_switch_hits_const_guard_with_acting_owner_unchanged():
    # Identical rows let even the old validator accept the control. Switch only
    # the frozen team at the final current-owned token of the SECOND episode.
    rollout = _batch(teams=[0, 0], owners=[3, 3])
    validate_pooled_trajectory(rollout)
    rollout["assigned_seats"][1, 4] = torch.tensor([0, 29, 0, 29])
    observed = rollout["assigned_seats"].gather(
        2, rollout["player_ids"].unsqueeze(-1),
    ).squeeze(-1)
    assert torch.equal(observed, rollout["policy_owner"])
    assert rollout["assignment_id"][1].unique().tolist() == [102]
    assert rollout["policy_owner"][1, 4].item() == 0
    with pytest.raises(ValueError, match="^assigned_seats may not change inside one episode$"):
        validate_pooled_trajectory(rollout)


@pytest.mark.parametrize("row", [0, 1])
def test_assignment_switch_is_rejected_within_either_episode(row):
    rollout = _batch(teams=[0, 0], owners=[3, 3])
    validate_pooled_trajectory(rollout)
    rollout["assignment_id"][row, 3] += 100
    with pytest.raises(ValueError, match="^assignment_id may not change inside one episode;"):
        validate_pooled_trajectory(rollout)


@pytest.mark.parametrize("field,value", [("policy_owner", 0), ("policy_train_mask", True)])
def test_chance_owner_and_train_mask_remain_fail_closed(field, value):
    rollout = _batch(
        players=[[0, -1, 1], [1, -1, 0]], teams=[0, 0], owners=[3, 3],
    )
    validate_pooled_trajectory(rollout)
    rollout[field][1, 1] = value
    with pytest.raises(ValueError, match="^chance/environment steps must carry owner -1"):
        validate_pooled_trajectory(rollout)


@pytest.mark.parametrize(
    "field,step,value,error",
    [
        ("policy_train_mask", 1, False, "policy_train_mask must be True exactly"),
        ("policy_train_mask", 0, True, "policy_train_mask must be True exactly"),
        ("on_policy_valid", 1, False, "on_policy_valid must equal policy_train_mask"),
        ("on_policy_valid", 0, True, "on_policy_valid must equal policy_train_mask"),
    ],
)
def test_current_and_frozen_mask_corruption_is_rejected(field, step, value, error):
    rollout = _batch(teams=[1, 1], owners=[3, 3])
    validate_pooled_trajectory(rollout)
    # Row 1 uses team 1: step 0 is frozen, step 1 is current.
    rollout[field][1, step] = value
    with pytest.raises(ValueError, match=error):
        validate_pooled_trajectory(rollout)


@pytest.mark.parametrize("field", ["num_steps", "assigned_seats", "policy_owner"])
def test_nonrectangular_field_dimensions_are_rejected(field):
    rollout = _batch(teams=[0, 0], owners=[3, 3])
    validate_pooled_trajectory(rollout)
    rollout[field] = (
        rollout[field].unsqueeze(-1) if field == "num_steps"
        else rollout[field][:, :-1]
    )
    with pytest.raises(ValueError, match=rf"field '{field}' must have shape"):
        validate_pooled_trajectory(rollout)


@pytest.mark.parametrize(
    "field,step,value,error",
    [
        ("opponent_checkpoint_id", 0, -1, "frozen-opponent token must record its checkpoint id"),
        ("opponent_policy_version", 0, -1, "frozen-opponent token must record its policy version"),
        ("opponent_checkpoint_id", 1, 5, "current-owned token must not carry an opponent checkpoint id"),
        ("behavior_action_prob", 0, float("nan"), "behavior_action_prob must be finite and positive"),
        ("behavior_action_prob", 0, 0.0, "behavior_action_prob must be finite and positive"),
        ("behavior_action_prob", 1, float("inf"), "behavior_action_prob must be finite and positive"),
    ],
)
def test_checkpoint_version_and_probability_guards_remain_active(field, step, value, error):
    rollout = _batch(teams=[1, 1], owners=[3, 3])
    validate_pooled_trajectory(rollout)
    rollout[field][1, step] = value
    with pytest.raises(ValueError, match=error):
        validate_pooled_trajectory(rollout)
