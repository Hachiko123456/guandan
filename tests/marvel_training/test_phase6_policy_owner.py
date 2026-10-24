"""Phase-6 behaviour attribution: ownership is separated from trainability."""
from __future__ import annotations

import pytest
import torch

from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.phase6.policy_owner import (
    macro_owner_counts, segment_pooled_macros, validate_macro_owners,
    validate_pooled_trajectory,
)
from guandan_marvel.training.macro_ppo import (
    MacroPPOError, build_macro_batch_view, compute_macro_pg_loss,
    macro_trainable_mask,
)

from tests.marvel_training._macro_fixtures import make_case

SEATS = (0, 1, 0, 1)  # team 0 (seats 0,2) current; team 1 (seats 1,3) opponent


def _pool_case(
    players,
    commits,
    *,
    seats=SEATS,
    horizon=None,
    rewards=None,
) -> tuple[TensorDict, object, torch.Tensor]:
    """Build a rollout with a fixed seat assignment and current/opponent owners."""
    rollout, _values, _segmentation, new_probs = make_case(
        players, commits, horizon=horizon, rewards=rewards, garbage_padding=True,
    )
    batch, length = rollout["player_ids"].shape
    owner = torch.full((batch, length), -1, dtype=torch.long)
    train = torch.zeros((batch, length), dtype=torch.bool)
    opponent_id = torch.full((batch, length), -1, dtype=torch.long)
    opponent_version = torch.full((batch, length), -1, dtype=torch.long)
    assignment = torch.full((batch, length), -1, dtype=torch.long)
    assigned = torch.full((batch, length, 4), -1, dtype=torch.long)
    for step, player in enumerate(players):
        if player < 0:
            continue
        seat_owner = int(seats[player])
        owner[:, step] = seat_owner
        train[:, step] = seat_owner == 0
        opponent_id[:, step] = -1 if seat_owner == 0 else 0
        opponent_version[:, step] = -1 if seat_owner == 0 else 7
        assignment[:, step] = 42
        assigned[:, step, :] = torch.tensor(seats, dtype=torch.long)
    rollout.set("policy_owner", owner)
    rollout.set("policy_train_mask", train)
    rollout.set("opponent_checkpoint_id", opponent_id)
    rollout.set("opponent_policy_version", opponent_version)
    rollout.set("assignment_id", assignment)
    rollout.set("assigned_seats", assigned)
    rollout.set("on_policy_valid", train.clone())
    segmentation = segment_pooled_macros(rollout)
    return rollout, segmentation, new_probs


def test_mixed_owner_macro_fails_closed():
    rollout, _seg, _probs = _pool_case([0, 0], [False, True])
    owner = rollout["policy_owner"].clone()
    owner[0, 1] = 1
    rollout.set("policy_owner", owner)
    rollout.set("policy_train_mask", owner == 0)
    with pytest.raises(ValueError, match="mixes policy owners"):
        segment_pooled_macros(rollout)


def test_mixed_train_mask_macro_fails_closed():
    rollout, _seg, _probs = _pool_case([0, 0], [False, True])
    train = rollout["policy_train_mask"].clone()
    train[0, 1] = False  # owner stays current, mask is inconsistent
    rollout.set("policy_train_mask", train)
    # The per-token owner/mask consistency check fires before the macro-level
    # mixed-mask guard; either way the mixed macro cannot reach a loss.
    with pytest.raises(ValueError, match="policy_train_mask must be True exactly"):
        segment_pooled_macros(rollout)


def test_opponent_tokens_are_not_trainable_ppo_samples():
    rollout, segmentation, new_probs = _pool_case(
        [0, 2, 1, 3], [True, True, True, True],
    )
    assert segmentation.max_macros == 4
    assert segmentation.macro_policy_owner.tolist()[0] == [0, 0, 1, 1]
    assert segmentation.macro_train_mask.tolist()[0] == [True, True, False, False]
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    assert view.macro_train_mask is not None
    assert macro_trainable_mask(view).tolist()[0] == [True, True, False, False]
    advantage = torch.tensor([[1.0, -2.0, 123.0, -456.0]])
    baseline = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=new_probs,
        macro_advantage=advantage, clip_coef=0.2, world_size=1,
    )
    assert baseline.macro_count_local == 2
    perturbed = new_probs.clone()
    # Hostile values on frozen-opponent tokens must never reach the current loss.
    perturbed[0, 2, :] = torch.tensor([1e-12, 1.0, 1.0, 1.0], dtype=perturbed.dtype)
    perturbed[0, 3, :] = torch.tensor([1.0, 1.0, 1.0, 1e-12], dtype=perturbed.dtype)
    changed = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=perturbed,
        macro_advantage=advantage, clip_coef=0.2, world_size=1,
    )
    assert torch.equal(baseline.pg_loss_mean, changed.pg_loss_mean)
    assert baseline.macro_count_global == changed.macro_count_global == 2


def test_current_and_opponent_macro_counts():
    _rollout, segmentation, _probs = _pool_case([0, 2, 1, 3], [True, True, True, True])
    current, opponent = macro_owner_counts(segmentation)
    assert (current, opponent) == (2, 2)
    validate_macro_owners(segmentation)


def test_chance_and_padding_are_owned_by_nobody():
    rollout, segmentation, _probs = _pool_case(
        [0, -1, 0], [True, False, True],
    )
    assert segmentation.token_to_macro_id.tolist()[0] == [0, -1, 1]
    assert segmentation.macro_train_mask.tolist()[0] == [True, True]
    assert rollout["policy_owner"][0, 1].item() == -1
    assert not bool(rollout["policy_train_mask"][0, 1])
    assert rollout["assignment_id"][0, 1].item() == -1
    stats = validate_pooled_trajectory(rollout)
    assert stats.current_token_count == 2
    assert stats.opponent_token_count == 0


def test_opponent_behaviour_probability_is_not_compared_to_current_policy():
    rollout, segmentation, new_probs = _pool_case([0, 1], [True, True])
    # The opponent token's reference distribution is the opponent's own; it is
    # deliberately inconsistent with any current-policy expectation.
    policy_probs = rollout["policy_probs"].clone()
    policy_probs[0, 1, :] = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=policy_probs.dtype)
    rollout.set("policy_probs", policy_probs)
    view = build_macro_batch_view(segmentation, torch.tensor([0]))
    advantage = torch.tensor([[1.0, -1.0]])
    loss = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=new_probs,
        macro_advantage=advantage, clip_coef=0.2, world_size=1,
    )
    assert loss.macro_count_local == 1
    # Claiming the opponent token as an on-policy current sample fails closed.
    train = rollout["policy_train_mask"].clone()
    on_policy = rollout["on_policy_valid"].clone()
    train[0, 1] = True
    rollout.set("policy_train_mask", train)
    with pytest.raises(MacroPPOError, match="on_policy_valid must equal policy_train_mask"):
        compute_macro_pg_loss(
            rollout, view=view, avail_log_probs_new=new_probs,
            macro_advantage=advantage, clip_coef=0.2, world_size=1,
        )
    rollout.set("policy_train_mask", train)
    rollout.set("on_policy_valid", on_policy)


def test_pooled_trajectory_rejects_in_episode_switch_and_tampering():
    rollout, _seg, _probs = _pool_case([0, 1, 2, 3], [True, True, True, True])
    validate_pooled_trajectory(rollout)

    changed = rollout.clone()
    assignment = changed["assignment_id"].clone()
    assignment[0, 2] = 43
    changed.set("assignment_id", assignment)
    with pytest.raises(ValueError, match="assignment_id may not change"):
        validate_pooled_trajectory(changed)

    changed = rollout.clone()
    on_policy = changed["on_policy_valid"].clone()
    on_policy[0, 1] = True
    changed.set("on_policy_valid", on_policy)
    with pytest.raises(ValueError, match="on_policy_valid must equal policy_train_mask"):
        validate_pooled_trajectory(changed)

    changed = rollout.clone()
    owner = changed["policy_owner"].clone()
    owner[0, 1] = -1  # an acting token cannot be ownerless
    changed.set("policy_owner", owner)
    with pytest.raises(ValueError, match="non-negative policy owner"):
        validate_pooled_trajectory(changed)

    changed = rollout.clone()
    seats = changed["assigned_seats"].clone()
    seats[0, 1, 1] = 0  # acting seat 1 no longer matches the recorded owner
    changed.set("assigned_seats", seats)
    with pytest.raises(ValueError, match="assigned_seats disagrees"):
        validate_pooled_trajectory(changed)
