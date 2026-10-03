"""Collector tests use full normal deals except explicit terminal-boundary fixtures."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
import torch

from guandan.action_state import StepwiseActionState
from guandan.cards import Rank, full_deck
from guandan.env_batch import GuandanEnvBatch
from guandan.environment_types import GameConfig
from guandan.model import PolicyValueNet
from guandan.state import RoundState
from guandan.training.collector import (
    Deadline, RolloutCollector, critic_state, policy_inputs, truncation_bootstrap_observation,
)
from guandan.training.critic import CentralQCritic
from guandan.training.estimators import compute_gae, masked_expected_values
from guandan.training.runtime import BudgetExceeded, TRAIN_SEED_BASE


@pytest.fixture(autouse=True)
def single_torch_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def normal_batch(size=4, *, max_token_steps=20_000):
    configs = [GameConfig(level_rank=Rank.FIVE, seed=TRAIN_SEED_BASE + i,
                          max_token_steps=max_token_steps) for i in range(size)]
    batch = GuandanEnvBatch(configs, auto_reset=False)
    batch.reset(seeds=[config.seed for config in configs])
    return batch


def actor(seed=7):
    torch.manual_seed(seed)
    return PolicyValueNet(hidden_dim=8)


def boundary_fixture():
    """Reduced hands ONLY to reach a real terminal in a boundary unit test."""
    deck = list(full_deck())
    card = next(card for card in deck if card.rank == Rank.THREE)
    other = [candidate for candidate in deck if candidate != card]
    state = RoundState([[card], other[::2], [], other[1::2]], level_rank=Rank.FIVE,
                       active_seat=0, leader_seat=0, finished_ranks=[2], emptied_seats=[2])
    return StepwiseActionState(state).serialize()


def test_real_full_deal_transitions_have_explicit_clocks_masks_rewards_and_perspective():
    batch = normal_batch()
    for env in batch.envs:
        hands = env.full_state()["round_state"]["hands"]
        assert [len(hand) for hand in hands] == [27] * 4
        assert len({card for hand in hands for card in hand}) == 108
    model = actor()
    collector = RolloutCollector(batch, model, seed=19)
    rollout = collector.collect(32)
    assert rollout.ticks == 8 and rollout.batch_size == 4 and rollout.token_transitions == 32
    assert rollout.values.shape == rollout.next_values.shape == rollout.rewards.shape == (8, 4, 4)
    assert collector.total_token_steps == 32
    np.testing.assert_array_equal(collector.ticks_per_env, [8] * 4)
    for row in rollout.transitions:
        assert row.observation.private_hand_card_ids is None
        torch.testing.assert_close(row.token_step_after, row.token_step_before + 1)
        torch.testing.assert_close(row.committed_step_after, row.committed_step_before + row.is_commit)
        torch.testing.assert_close(row.team_ids, row.player_ids % 2)
        assert row.legal_mask.shape == (4, 256)
        assert torch.all(row.legal_mask.sum(-1) > 0)
        for i, action in enumerate(row.actions):
            assert int(action) in row.observation.legal_next_tokens[i, row.legal_mask[i].numpy()]
        torch.testing.assert_close(row.acting_player_rewards,
                                   row.rewards.gather(1, row.player_ids[:, None]).squeeze(1))
        torch.testing.assert_close(row.values[:, 0], row.values[:, 2])
        torch.testing.assert_close(row.values[:, 1], -row.values[:, 0])
        torch.testing.assert_close(row.values[:, 1], row.values[:, 3])
        assert torch.isfinite(row.old_log_probs).all()
        with torch.no_grad():
            inputs = policy_inputs(row.observation, "cpu")
            logits, values = model(*inputs)
            torch.testing.assert_close(row.old_log_probs, model.log_prob(logits, row.actions, legal_next_tokens=inputs[2]))
            torch.testing.assert_close(row.value, values)
    assert len(rollout.digest()) == 64
    assert rollout.digest() == rollout.digest()
    assert collector.total_committed_steps == int(rollout.stack("is_commit").sum())


def test_true_terminal_is_captured_before_reset_and_reward_does_not_leak():
    initial = boundary_fixture()
    batch = GuandanEnvBatch([GameConfig(level_rank=Rank.FIVE, seed=TRAIN_SEED_BASE)])
    batch.reset(initial_states=[initial])
    collector = RolloutCollector(batch, actor(), seed=11, initial_states=[initial])
    terminal = None
    # Canonical single-card token path is deterministic and genuinely commits.
    for _ in range(32):
        rollout = collector.collect(1)
        if rollout.terminated.any():
            terminal = rollout.transitions[-1]
            break
    assert terminal is not None
    assert terminal.done.item() and not terminal.truncated.item()
    torch.testing.assert_close(terminal.rewards, torch.tensor([[1.0, -1.0, 1.0, -1.0]]))
    assert terminal.acting_player_rewards.item() == 1
    assert not terminal.next_observation.present[0]
    assert terminal.next_observation.phase[0] == "TERMINAL"
    assert not terminal.bootstrap_observation.present[0]
    assert not terminal.next_values.any()
    assert terminal.token_step_after.item() > 0
    assert collector.observation.token_step[0] == 0
    assert collector.observation.committed_step[0] == 0
    assert collector.reset_counts.tolist() == [1]
    record = rollout.terminal_snapshots()[0]
    assert record["episode_index"] == 0 and record["actor"] == 0
    assert record["ranking"][0] == 2 and record["rewards"] == [1, -1, 1, -1]
    snapshot = deepcopy(terminal.next_observation)
    next_row = collector.collect(1).transitions[0]
    assert next_row.episode_indices.tolist() == [1]
    assert not next_row.rewards.any()
    np.testing.assert_array_equal(terminal.next_observation.token_step, snapshot.token_step)
    # Terminal reward cannot bootstrap from the next episode.
    advantages, returns = compute_gae(rollout.rewards, rollout.values, rollout.next_values,
                                       rollout.terminated, rollout.truncated)
    torch.testing.assert_close(returns[-1], terminal.rewards)


def test_truncation_bootstraps_pre_reset_without_mutating_or_replaying_a_step():
    batch = normal_batch(size=1, max_token_steps=1)
    model = actor()
    collector = RolloutCollector(batch, model, seed=11)
    row = collector.collect(1).transitions[0]
    assert row.truncated.item() and not row.done.item()
    assert not row.rewards.any()
    assert not row.next_observation.present[0]
    assert row.next_observation.phase[0] == "TRUNCATED"
    assert row.bootstrap_observation.present[0]
    assert row.bootstrap_observation.token_step.tolist() == [1]
    assert collector.observation.token_step.tolist() == [0]
    assert row.token_step_after.item() == 1 and row.committed_step_after.item() == 0
    with torch.no_grad():
        _, value = model(*policy_inputs(row.bootstrap_observation, "cpu"))
    torch.testing.assert_close(row.next_values[0, row.bootstrap_observation.player_id[0]], value[0])
    assert collector.reset_counts.tolist() == [1]
    assert collector.deal_records[-1]["seed"] == TRAIN_SEED_BASE + 1
    assert collector.deal_records[-1]["hand_sizes"] == [27] * 4
    # Direct snapshot function is a pure clone/encode operation.
    other = normal_batch(size=1, max_token_steps=1)
    obs = other.observe()
    action = obs.legal_next_tokens[0, obs.legal_next_mask[0]][0]
    other.step(np.array([action], dtype=np.int32))
    serialized = other.serialize()
    boot = truncation_bootstrap_observation(other, 0)
    assert boot.token_step == 1
    assert other.serialize() == serialized and other.envs[0].truncated
    assert other.envs[0].token_step == 1
    # Trace cuts at time limit but retains its actual successor value.
    _, returns = compute_gae(row.rewards[None], row.values[None], row.next_values[None],
                             row.terminated[None], row.truncated[None])
    torch.testing.assert_close(returns[0], row.next_values)


def test_critic_privilege_is_separate_from_actor_inputs():
    batch = normal_batch(size=1)
    model, critic = actor(), CentralQCritic(hidden_dim=8)
    seen = []
    handle = model.register_forward_pre_hook(lambda module, args: seen.append(tuple(x.shape for x in args)))
    collector = RolloutCollector(batch, model, seed=11, critic=critic, expected_values=masked_expected_values)
    row = collector.collect(1).transitions[0]
    handle.remove()
    assert row.critic_states.shape == (1, 688)
    assert row.critic_states[0, :432].sum() == 108
    assert row.selected_q.shape == row.values.shape == (1, 4)
    torch.testing.assert_close(row.selected_q[:, 0], -row.selected_q[:, 1])
    assert seen and all(len(shapes) == 4 and shapes[1][-1] == 256 for shapes in seen)
    assert not hasattr(row.observation, "critic_states")
    assert row.observation.private_hand_card_ids is None


def test_collector_checkpoint_restores_rng_env_prefix_and_real_counters_weights_only():
    batch = normal_batch(size=1)
    model = actor()
    first = RolloutCollector(batch, model, seed=91)
    first.collect(2)  # Mid-prefix, not just a reset boundary.
    memory = BytesIO()
    torch.save(first.state_dict(), memory)
    memory.seek(0)
    payload = torch.load(memory, weights_only=True)
    restored_batch = GuandanEnvBatch([batch.envs[0].config])
    restored_batch.load_serialized(payload["envs"])
    second = RolloutCollector(restored_batch, deepcopy(model), seed=0)
    second.load_state_dict(payload)
    assert second.batch.serialize() == first.batch.serialize()
    a, b = first.collect(8), second.collect(8)
    assert a.digest() == b.digest()
    assert first.total_token_steps == second.total_token_steps == 10
    assert first.batch.serialize() == second.batch.serialize()
    torch.testing.assert_close(a.stack("old_log_probs"), b.stack("old_log_probs"))
    assert first.state_dict()["ticks_per_env"] == [10]


def test_deadline_never_scales_or_fabricates_counted_steps():
    batch = normal_batch(size=1)
    collector = RolloutCollector(batch, actor(), seed=11)
    checks = 0

    def deadline():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise BudgetExceeded("test budget exhausted")

    with pytest.raises(BudgetExceeded) as exc:
        collector.collect(8, deadline=deadline)
    assert collector.total_token_steps == 2
    assert collector.ticks_per_env.tolist() == [2]
    assert collector.incomplete
    assert exc.value.diagnostic["status"] == "incomplete"
    assert exc.value.diagnostic["target_token_steps"] == 8
    assert exc.value.diagnostic["actual_token_steps"] == 2
    with pytest.raises(RuntimeError, match="incomplete"):
        collector.collect(8)


def test_deadline_reexport_and_bad_aggregate_validation():
    from guandan.training.runtime import Deadline as RuntimeDeadline
    assert Deadline is RuntimeDeadline
    batch = normal_batch(size=1)
    collector = RolloutCollector(batch, actor())
    for value in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="aggregate"):
            collector.collect(value)
    with pytest.raises(ValueError, match="auto_reset"):
        RolloutCollector(GuandanEnvBatch([GameConfig()], auto_reset=True), actor())
    expired = Deadline(1, clock=lambda: 4000, started_at=0)
    with pytest.raises(BudgetExceeded):
        collector.collect(1, deadline=expired)
    assert collector.total_token_steps == 0
