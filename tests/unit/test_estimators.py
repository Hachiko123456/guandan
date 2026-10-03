"""Independent scalar references and boundary/perspective regression cases."""
from __future__ import annotations

import pytest
import torch

from guandan.training.estimators import (
    acting_player_values,
    compute_gae,
    compute_q_boost,
    expand_team_values,
    masked_expected_values,
)


def f(data):
    return torch.tensor(data, dtype=torch.float64)


def flags(data):
    return torch.tensor(data, dtype=torch.bool)


def scalar_reference(rewards, values, next_values, done, trunc, gamma, lam, selected_q=None):
    """Forward sum, plain Python floats: deliberately not the vector recurrence."""
    steps, batch, seats = rewards.shape
    advantages = torch.empty_like(rewards)
    targets = torch.empty_like(rewards)
    for t in range(steps):
        for b in range(batch):
            for i in range(seats):
                trace = 0.0
                for k in range(t, steps):
                    baseline = float(values[k, b, i] if selected_q is None else selected_q[k, b, i])
                    successor = 0.0 if bool(done[k, b]) else float(next_values[k, b, i])
                    delta = float(rewards[k, b, i]) + gamma * successor - baseline
                    trace += (gamma * lam) ** (k - t) * delta
                    if bool(done[k, b]) or bool(trunc[k, b]):
                        break
                start = float(values[t, b, i] if selected_q is None else selected_q[t, b, i])
                targets[t, b, i] = start + trace
                advantages[t, b, i] = trace if selected_q is None else start - float(values[t, b, i]) + trace
    return advantages, targets


def run_estimator(kind, rewards, values, next_values, done, trunc, selected_q=None, **kwargs):
    if kind == "gae":
        return compute_gae(rewards, values, next_values, done, trunc, **kwargs)
    if selected_q is None:
        selected_q = values + 0.75
    return compute_q_boost(rewards, selected_q, values, next_values, done, trunc, **kwargs)


def test_gae_hand_computed_two_batches_all_four_players() -> None:
    rewards = f([[[1, 2, 3, 4], [10, 20, 30, 40]], [[5, 6, 7, 8], [1, 2, 3, 4]]])
    values = f([[[0, 0, 0, 0], [1, 2, 3, 4]], [[1, 2, 3, 4], [0, 0, 0, 0]]])
    successor = f([[[10, 20, 30, 40], [5, 6, 7, 8]], [[99, 99, 99, 99], [0, 0, 0, 0]]])
    adv, ret = compute_gae(rewards, values, successor, flags([[False, False], [True, True]]), flags([[False, False], [False, False]]), gamma=0.5, gae_lambda=0.25)
    expected = f([[[6.5, 12.5, 18.5, 24.5], [11.625, 21.25, 30.875, 40.5]], [[4, 4, 4, 4], [1, 2, 3, 4]]])
    torch.testing.assert_close(adv, expected, rtol=0, atol=0)
    torch.testing.assert_close(ret, expected + values, rtol=0, atol=0)


def test_q_boost_hand_computed_residuals_and_action_targets() -> None:
    rewards = f([[[1, 2, 3, 4]], [[5, 6, 7, 8]]])
    q = f([[[2, 4, 6, 8]], [[4, 4, 4, 4]]])
    values = f([[[1, 2, 3, 4]], [[1, 1, 1, 1]]])
    successor = f([[[3, 4, 5, 6]], [[0, 0, 0, 0]]])
    adv, target = compute_q_boost(rewards, q, values, successor, flags([[False], [True]]), flags([[False], [False]]), gae_lambda=0.5)
    # delta+ = [2,2,2,2], then [1,2,3,4]; the earlier trace adds half.
    trace = f([[[2.5, 3, 3.5, 4]], [[1, 2, 3, 4]]])
    assert torch.equal(target, q + trace)
    assert torch.equal(adv, q - values + trace)
    # This target is not obtained by applying GAE and relabeling the result.
    gae_target = compute_gae(rewards, values, successor, flags([[False], [True]]), flags([[False], [False]]), gae_lambda=0.5)[1]
    assert not torch.equal(target[0], gae_target[0])


@pytest.mark.parametrize("kind", ["gae", "boost"])
@pytest.mark.parametrize("gamma,lam", [(0.0, 0.95), (1.0, 0.0), (1.0, 1.0), (0.73, 0.41)])
def test_independent_forward_scalar_sum_across_asynchronous_episodes(kind, gamma, lam) -> None:
    generator = torch.Generator().manual_seed(402)
    rewards, values, successor, q = [torch.randn(6, 3, 4, generator=generator, dtype=torch.float64) for _ in range(4)]
    done = flags([[False, False, False], [True, False, False], [False, False, False], [False, False, True], [False, False, False], [False, True, False]])
    trunc = flags([[False, True, False], [False, False, False], [False, False, True], [False, False, True], [True, False, False], [False, False, False]])
    expected = scalar_reference(rewards, values, successor, done, trunc, gamma, lam, None if kind == "gae" else q)
    actual = run_estimator(kind, rewards, values, successor, done, trunc, q, gamma=gamma, gae_lambda=lam)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["gae", "boost"])
@pytest.mark.parametrize("done,trunc,expected", [(True, False, 2.0), (False, True, 7.0), (False, False, 7.0), (True, True, 2.0)])
def test_last_transition_done_truncation_and_rollout_boundary(kind, done, trunc, expected) -> None:
    # End of collection is not automatically a terminal. Done wins if both flags are set.
    reward = torch.full((1, 2, 4), 2.0)
    values = torch.full_like(reward, 3.0)
    successor = torch.full_like(reward, 10.0)
    adv, target = run_estimator(kind, reward, values, successor, torch.full((1, 2), done), torch.full((1, 2), trunc), gamma=0.5)
    assert torch.equal(target, torch.full_like(target, expected))
    assert torch.equal(adv, target - values)


@pytest.mark.parametrize("kind", ["gae", "boost"])
@pytest.mark.parametrize("cut", ["done", "trunc"])
def test_reset_episode_never_leaks_across_a_trace_cut(kind, cut) -> None:
    reward = f([[[0, 0, 0, 0]], [[2, 3, 4, 5]], [[1000, 2000, 3000, 4000]]])
    values = torch.zeros_like(reward)
    successor = f([[[0, 0, 0, 0]], [[10, 20, 30, 40]], [[0, 0, 0, 0]]])
    done = flags([[False], [cut == "done"], [True]])
    trunc = flags([[False], [cut == "trunc"], [False]])
    adv, _ = run_estimator(kind, reward, values, successor, done, trunc, selected_q=values, gae_lambda=0.5)
    middle = f([2, 3, 4, 5]) + (f([10, 20, 30, 40]) if cut == "trunc" else 0)
    assert torch.equal(adv[1, 0], middle)
    assert torch.equal(adv[0, 0], middle * 0.5)
    assert torch.equal(adv[2, 0], reward[2, 0])
    altered = reward.clone()
    altered[2] *= -100
    assert torch.equal(run_estimator(kind, altered, values, successor, done, trunc, selected_q=values, gae_lambda=0.5)[0][:2], adv[:2])


@pytest.mark.parametrize("kind", ["gae", "boost"])
def test_terminal_nan_successor_is_masked_before_arithmetic(kind) -> None:
    reward = torch.ones(1, 1, 4)
    values = torch.zeros_like(reward)
    adv, target = run_estimator(kind, reward, values, torch.full_like(reward, float("nan")), flags([[True]]), flags([[False]]))
    assert torch.equal(adv, reward) and torch.equal(target, reward)
    with pytest.raises(ValueError, match="nonterminal next_values"):
        run_estimator(kind, reward, values, torch.full_like(reward, float("nan")), flags([[False]]), flags([[True]]))


def test_gae_fixed_perspective_survives_all_four_actor_and_team_switches() -> None:
    actors = torch.tensor([[0], [1], [2], [3]])
    scalar_values = f([[1], [-2], [3], [-4]])
    values = expand_team_values(scalar_values, actors)
    successor = torch.cat((values[1:], torch.zeros_like(values[:1])))
    reward = torch.zeros_like(values)
    reward[-1, 0] = f([8, -8, 8, -8])
    adv, ret = compute_gae(reward, values, successor, flags([[False], [False], [False], [True]]), flags([[False]] * 4), gae_lambda=1.0)
    assert torch.equal(ret, f([[[8, -8, 8, -8]]] * 4))
    assert torch.equal(acting_player_values(adv, actors), f([[7], [-6], [5], [-4]]))
    assert torch.equal(acting_player_values(values, actors), scalar_values)


@pytest.mark.parametrize("lam", [0.25, 0.95, 1.0])
def test_exact_q_matching_pennies_has_zero_future_action_sampling_variance(lam) -> None:
    # Exhaust all four outcomes of a two-stage imperfect-information game.
    # Team 0 commits H/T. Team 1 then chooses H/T without seeing that choice.
    # Exact centralized second-stage Q sees the commitment; the actor does not.
    first = torch.tensor([0, 0, 1, 1])
    second = torch.tensor([0, 1, 0, 1])
    teams = f([1, -1, 1, -1])
    payoff = torch.where(first.eq(second), 1.0, -1.0).double()
    second_q0 = torch.where(first[:, None].eq(torch.arange(2)), 1.0, -1.0).double()
    second_q = second_q0.unsqueeze(-1) * teams
    v_second = masked_expected_values(second_q, torch.full((4, 2), 0.5, dtype=torch.float64), torch.ones(4, 2, dtype=torch.bool))
    assert v_second.eq(0).all()
    reward = torch.zeros(2, 4, 4, dtype=torch.float64)
    reward[1] = payoff.unsqueeze(-1) * teams
    q = torch.zeros_like(reward)
    q[1] = second_q[torch.arange(4), second]
    values = torch.zeros_like(reward)
    values[1] = v_second
    successor = torch.zeros_like(reward)
    successor[0] = v_second
    done = flags([[False] * 4, [True] * 4])
    trunc = torch.zeros_like(done)
    gae, _ = compute_gae(reward, values, successor, done, trunc, gae_lambda=lam)
    boost, q_target = compute_q_boost(reward, q, values, successor, done, trunc, gae_lambda=lam)
    torch.testing.assert_close(gae[0], lam * reward[1])
    assert torch.equal(boost[0], torch.zeros(4, 4, dtype=torch.float64))
    assert torch.equal(boost[1], reward[1]) and torch.equal(q_target, q)
    torch.testing.assert_close(gae[0].var(dim=0, unbiased=False), torch.full((4,), lam ** 2, dtype=torch.float64))
    assert boost[0].var(dim=0, unbiased=False).eq(0).all()
    actors = torch.tensor([[0, 2, 0, 2], [1, 3, 1, 3]])
    assert acting_player_values(boost, actors)[0].eq(0).all()
    assert acting_player_values(gae, actors)[0].abs().gt(0).all()


@pytest.mark.parametrize("kind", ["gae", "boost"])
def test_detached_empty_rollouts_preserve_layout_without_mutation(kind) -> None:
    empty = torch.empty(0, 2, 4, dtype=torch.float64, requires_grad=True)
    no_flags = torch.empty(0, 2, dtype=torch.bool)
    adv, target = run_estimator(kind, empty, empty, empty, no_flags, no_flags)
    assert adv.shape == target.shape == (0, 2, 4)
    assert adv.dtype == target.dtype == torch.float64
    assert not adv.requires_grad and not target.requires_grad


@pytest.mark.parametrize("kind", ["gae", "boost"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_device_dtype_detachment_and_input_immutability(kind, device) -> None:
    tensors = [torch.randn(4, 2, 4, dtype=torch.float64, device=device, requires_grad=True) for _ in range(4)]
    before = [x.detach().clone() for x in tensors]
    done = torch.zeros(4, 2, dtype=torch.bool, device=device)
    trunc = done.clone()
    adv, target = run_estimator(kind, *tensors[:3], done, trunc, selected_q=tensors[3])
    assert adv.device == target.device == tensors[0].device
    assert adv.dtype == target.dtype == torch.float64
    assert not adv.requires_grad and not target.requires_grad
    for original, saved in zip(tensors, before):
        assert torch.equal(original, saved)


def test_masked_expectation_hand_reference_arbitrary_four_seat_values() -> None:
    q = f([[[1, 2, 3, 4], [100, 200, 300, 400], [5, 8, 11, 14]], [[3, 6, 9, 12], [7, 8, 9, 10], [50, 60, 70, 80]]])
    probabilities = f([[0.1, 0.6, 0.3], [1, 3, 100]])
    legal = flags([[True, False, True], [True, True, False]])
    actual = masked_expected_values(q, probabilities, legal)
    torch.testing.assert_close(actual, f([[4, 6.5, 9, 11.5], [6, 7.5, 9, 10.5]]))


def test_masked_expectation_nan_padding_has_zero_mass_and_zero_gradient() -> None:
    q = f([[[1, 2, 3, 4], [float("nan")] * 4, [3, 4, 5, 6]]]).requires_grad_()
    probabilities = f([[0.2, float("inf"), 0.5]]).requires_grad_()
    actual = masked_expected_values(q, probabilities, flags([[True, False, True]]))
    torch.testing.assert_close(actual[0], f([17, 24, 31, 38]) / 7)
    actual.sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(probabilities.grad).all()
    assert q.grad[0, 1].eq(0).all() and probabilities.grad[0, 1].item() == 0


@pytest.mark.parametrize("probabilities,legal", [([[0, 0]], [[True, True]]), ([[1, 1]], [[False, False]]), ([[-1, 2]], [[True, True]]), ([[float("nan"), 1]], [[True, True]])])
def test_masked_expectation_rejects_invalid_legal_probability_mass(probabilities, legal) -> None:
    with pytest.raises(ValueError):
        masked_expected_values(torch.zeros(1, 2, 4, dtype=torch.float64), f(probabilities), flags(legal))


def test_masked_expectation_rejects_nonfinite_legal_q_and_non_boolean_mask() -> None:
    with pytest.raises(ValueError, match="q_values"):
        masked_expected_values(torch.full((1, 1, 4), float("inf")), torch.ones(1, 1), flags([[True]]))
    with pytest.raises(TypeError, match="legal_mask"):
        masked_expected_values(torch.ones(1, 1, 4), torch.ones(1, 1), torch.ones(1, 1))


def test_team_expansion_and_gather_every_actor_with_gradients() -> None:
    scalars = f([[2, 2, 2, 2]]).requires_grad_()
    ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    expanded = expand_team_values(scalars, ids)
    assert torch.equal(expanded, f([[[2, -2, 2, -2], [-2, 2, -2, 2], [2, -2, 2, -2], [-2, 2, -2, 2]]]))
    restored = acting_player_values(expanded, ids)
    assert torch.equal(restored, scalars)
    restored.sum().backward()
    assert torch.equal(scalars.grad, torch.ones_like(scalars))
    assert acting_player_values(f([1, 2, 3, 4]), torch.tensor(2)).item() == 3


def test_gather_arbitrary_leading_dimensions_and_only_selected_gradients() -> None:
    values = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).requires_grad_()
    ids = torch.tensor([[0, 1, 2], [3, 0, 1]])
    actual = acting_player_values(values, ids)
    assert torch.equal(actual, f([[0, 5, 10], [15, 16, 21]]))
    actual.sum().backward()
    assert torch.equal(values.grad, torch.nn.functional.one_hot(ids, 4).double())


@pytest.mark.parametrize("ids,error", [(torch.tensor([4]), ValueError), (torch.tensor([-1]), ValueError), (torch.tensor([0.0]), TypeError), (torch.tensor([[0]]), ValueError)])
def test_player_helpers_reject_invalid_id_contracts(ids, error) -> None:
    with pytest.raises(error):
        acting_player_values(torch.ones(1, 4), ids)
    with pytest.raises(error):
        expand_team_values(torch.ones(1), ids)


@pytest.mark.parametrize("gamma,lam,error", [(-0.1, 0.5, ValueError), (1.1, 0.5, ValueError), (1.0, -0.1, ValueError), (1.0, float("nan"), ValueError), (True, 0.5, TypeError)])
def test_estimators_validate_discount_and_lambda(gamma, lam, error) -> None:
    base = torch.zeros(1, 1, 4)
    for kind in ("gae", "boost"):
        with pytest.raises(error):
            run_estimator(kind, base, base, base, flags([[False]]), flags([[False]]), gamma=gamma, gae_lambda=lam)


def test_estimators_reject_scalar_active_player_inputs_and_numeric_flags() -> None:
    base = torch.zeros(2, 1, 4)
    done = flags([[False], [True]])
    with pytest.raises(ValueError, match="rewards"):
        compute_gae(base[..., 0], base, base, done, done)
    with pytest.raises(TypeError, match="terminated"):
        compute_gae(base, base, base, done.float(), done)
    with pytest.raises(ValueError, match="selected_q"):
        compute_q_boost(base, base[..., 0], base, base, done, done)
    with pytest.raises(ValueError, match="values"):
        compute_gae(base, base.double(), base, done, done)
