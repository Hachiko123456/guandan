"""Scalar PPO references, selected-Q fitting, and actor/critic separation."""
from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from guandan.model import PolicyValueNet
from guandan.training.critic import CRITIC_STATE_DIM, OWNERSHIP_CHANNELS, CentralQCritic
from guandan.training.estimators import (
    acting_player_values,
    compute_gae,
    compute_q_boost,
    expand_team_values,
    masked_expected_values,
)
from guandan.training.objectives import ippo_loss, vrpo_loss


def objective_inputs():
    logits = torch.tensor([[0.3, -0.2, -torch.inf], [0.1, 0.7, -torch.inf], [0.2, 0.2, -torch.inf], [0.5, -0.1, -torch.inf]], dtype=torch.float64, requires_grad=True)
    tokens = torch.tensor([[10, 11, 0], [20, 21, 0], [30, 31, 0], [40, 41, 0]], dtype=torch.int64)
    actions = torch.tensor([10, 21, 30, 41])
    indices = torch.tensor([[0], [1], [0], [1]])
    old = F.log_softmax(logits.detach(), dim=-1).gather(1, indices).squeeze(1)
    advantages = torch.tensor([1.0, -1.0, 0.5, 0.25], dtype=torch.float64)
    targets = torch.tensor([2.0, 1.0, -1.0, 0.0], dtype=torch.float64)
    prediction = torch.tensor([0.5, 0.5, -0.5, 0.25], dtype=torch.float64, requires_grad=True)
    return [logits, prediction, actions, old, advantages, targets, tokens]


@pytest.mark.parametrize("objective", [ippo_loss, vrpo_loss])
def test_clipped_policy_and_value_loss_against_plain_python_reference(objective) -> None:
    args = objective_inputs()
    ratios = [1.5, 0.5, 1.5, 0.5]
    args[3] = args[3] - torch.tensor(ratios, dtype=torch.float64).log()
    args[4] = torch.tensor([2.0, 2.0, -2.0, -2.0], dtype=torch.float64)
    args[1] = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    args[5] = torch.tensor([0.0, 2.0, 5.0, 2.0], dtype=torch.float64)
    total, metrics = objective(*args, clip=0.2, entropy_coef=0.07, value_coef=0.5)
    policy = -sum(min(r * a, min(max(r, 0.8), 1.2) * a) for r, a in zip(ratios, [2, 2, -2, -2])) / 4
    mse = sum((v - target) ** 2 for v, target in zip([1, 2, 3, 4], [0, 2, 5, 2])) / 4
    entropy = 0.0
    for row in args[0].detach().tolist():
        exp = [math.exp(x) for x in row[:2]]
        probabilities = [x / sum(exp) for x in exp]
        entropy -= sum(p * math.log(p) for p in probabilities) / 4
    assert policy == pytest.approx(0.3)
    assert metrics["policy_loss"].item() == pytest.approx(policy)
    assert metrics["value_loss"].item() == pytest.approx(mse)
    assert metrics["entropy"].item() == pytest.approx(entropy)
    assert metrics["clip_fraction"].item() == 1
    assert metrics["approx_kl"].item() == pytest.approx(sum(r - 1 - math.log(r) for r in ratios) / 4)
    assert total.item() == pytest.approx(policy + 0.5 * mse - 0.07 * entropy)
    total.backward()
    assert torch.isfinite(args[0].grad).all() and torch.isfinite(args[1].grad).all()


@pytest.mark.parametrize("objective", [ippo_loss, vrpo_loss])
def test_targets_advantages_old_policy_and_metrics_are_detached(objective) -> None:
    args = objective_inputs()
    for index in (3, 4, 5):
        args[index].requires_grad_()
    total, metrics = objective(*args)
    assert total.requires_grad
    assert all(isinstance(value, torch.Tensor) and value.ndim == 0 and torch.isfinite(value) and not value.requires_grad for value in metrics.values())
    total.backward()
    assert all(args[index].grad is None for index in (3, 4, 5))
    assert torch.isfinite(args[0].grad).all() and torch.isfinite(args[1].grad).all()
    assert args[0].grad[:, 2].eq(0).all()
    assert F.softmax(args[0], dim=-1)[:, 2].eq(0).all()


def test_vrpo_fits_selected_action_value_not_policy_expected_value() -> None:
    # Selected a=0 Q=2, but policy-averaged V=0. The regression target is 5.
    q = torch.tensor([[[2.0, -2.0, 2.0, -2.0], [-2.0, 2.0, -2.0, 2.0]]], requires_grad=True)
    logits = torch.zeros(1, 2, requires_grad=True)
    expected = masked_expected_values(q, logits.softmax(-1), torch.ones(1, 2, dtype=torch.bool))
    assert expected.eq(0).all()
    total, metrics = vrpo_loss(logits, q[:, 0, 0], torch.tensor([9]), torch.tensor([-math.log(2)]), torch.zeros(1), torch.tensor([5.0]), torch.tensor([[9, 11]]), entropy_coef=0.0, value_coef=1.0)
    assert metrics["q_loss"].item() == 9.0
    assert metrics["q_loss"].item() != (expected[0, 0] - 5).square().item()
    total.backward()
    assert q.grad[0, 0, 0].item() == -6
    assert q.grad[0, 1].eq(0).all()
    assert q.grad[0, 0, 1:].eq(0).all()
    assert logits.grad.eq(0).all()


@pytest.mark.parametrize("objective", [ippo_loss, vrpo_loss])
def test_token_identifiers_not_candidate_indices_and_duplicate_padding_ignored(objective) -> None:
    logits = torch.tensor([[0.0, -torch.inf, -torch.inf]], requires_grad=True)
    args = [logits, torch.zeros(1, requires_grad=True), torch.tensor([0]), torch.zeros(1), torch.ones(1), torch.zeros(1), torch.tensor([[0, 0, 0]])]
    total, metrics = objective(*args)
    assert total.item() == -1.0 and metrics["entropy"].item() == 0
    total.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.eq(0).all()
    args = objective_inputs()
    args[2] = torch.tensor([0, 1, 0, 1])
    with pytest.raises(ValueError, match="legal token"):
        objective(*args)


@pytest.mark.parametrize("objective", [ippo_loss, vrpo_loss])
@pytest.mark.parametrize("case", ["nan", "positive_inf", "all_invalid", "selected_invalid", "duplicate_legal", "bad_shape", "float_actions", "nan_target", "float_tokens", "vector_predictions"])
def test_objective_contract_validation(objective, case) -> None:
    args = objective_inputs()
    if case == "nan":
        args[0] = args[0].detach().clone()
        args[0][0, 0] = torch.nan
    elif case == "positive_inf":
        args[0] = args[0].detach().clone()
        args[0][0, 0] = torch.inf
    elif case == "all_invalid":
        args[0] = args[0].detach().clone()
        args[0][0] = -torch.inf
    elif case == "selected_invalid":
        args[2][0] = 0
    elif case == "duplicate_legal":
        args[6][0, 1] = 10
    elif case == "bad_shape":
        args[4] = args[4].unsqueeze(-1)
    elif case == "float_actions":
        args[2] = args[2].double()
    elif case == "nan_target":
        args[5][0] = torch.nan
    elif case == "float_tokens":
        args[6] = args[6].double()
    elif case == "vector_predictions":
        args[1] = args[1].unsqueeze(-1).expand(-1, 4)
    with pytest.raises((TypeError, ValueError)):
        objective(*args)


@pytest.mark.parametrize("options", [{"clip": -0.1}, {"clip": 1.1}, {"entropy_coef": -1}, {"value_coef": float("nan")}, {"clip": True}])
def test_invalid_hyperparameters_fail_explicitly(options) -> None:
    with pytest.raises((TypeError, ValueError)):
        ippo_loss(*objective_inputs(), **options)


def test_exponential_ratio_overflow_is_not_silently_accepted() -> None:
    args = objective_inputs()
    args[3].fill_(-10000)
    with pytest.raises(FloatingPointError, match="non-finite"):
        vrpo_loss(*args)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_central_critic_fixed_team_shape_candidate_dependence_and_gradients(device) -> None:
    torch.manual_seed(12)
    critic = CentralQCritic(hidden_dim=32).to(device)
    state = torch.randn(3, CRITIC_STATE_DIM, device=device, requires_grad=True)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device, dtype=torch.int32)
    q = critic(state, tokens)
    assert OWNERSHIP_CHANNELS == 432 and CRITIC_STATE_DIM == 688
    assert q.shape == (3, 3, 4) and torch.isfinite(q).all()
    assert torch.equal(q[..., 0], q[..., 2]) and torch.equal(q[..., 1], q[..., 3])
    assert torch.equal(q[..., 0], -q[..., 1])
    assert not torch.equal(q[:, 0], q[:, 1])
    q.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
    assert torch.isfinite(state.grad).all() and state.grad[:, :432].abs().sum() > 0


def test_critic_candidate_reordering_reorders_q_and_duplicate_tokens_match() -> None:
    torch.manual_seed(31)
    critic = CentralQCritic(hidden_dim=8).double()
    state = torch.randn(2, 688, dtype=torch.float64)
    tokens = torch.tensor([[5, 19, 5, 0], [6, 13, 6, 0]])
    q = critic(state, tokens)
    permutation = torch.tensor([3, 1, 0, 2])
    assert torch.allclose(critic(state, tokens[:, permutation]), q[:, permutation])
    assert torch.equal(q[:, 0], q[:, 2])


@pytest.mark.parametrize("hidden,error", [(0, ValueError), (-1, ValueError), (True, TypeError), (3.5, TypeError)])
def test_critic_rejects_invalid_hidden_dimensions(hidden, error) -> None:
    with pytest.raises(error):
        CentralQCritic(hidden_dim=hidden)


@pytest.mark.parametrize("case", ["state_width", "tokens_shape", "bad_token", "negative_token", "float_tokens", "state_nan", "integer_state", "empty_actions"])
def test_critic_input_contract(case) -> None:
    critic = CentralQCritic()
    state = torch.zeros(2, 688)
    tokens = torch.ones(2, 3, dtype=torch.int64)
    if case == "state_width":
        state = torch.zeros(2, 256)
    elif case == "tokens_shape":
        tokens = torch.ones(3, 3, dtype=torch.int64)
    elif case == "bad_token":
        tokens[0, 0] = 256
    elif case == "negative_token":
        tokens[0, 0] = -1
    elif case == "float_tokens":
        tokens = tokens.float()
    elif case == "state_nan":
        state[0, 0] = torch.nan
    elif case == "integer_state":
        state = state.long()
    elif case == "empty_actions":
        tokens = tokens[:, :0]
    with pytest.raises((TypeError, ValueError)):
        critic(state, tokens)


def policy_inputs(batch=4):
    obs = torch.tensor([[1, 2, 3, 0]] * batch)
    channels = torch.randn(batch, 256)
    tokens = torch.tensor([[9, 13, 27, 0]] * batch)
    legal = torch.tensor([[True, True, True, False]] * batch)
    return obs, channels, tokens, legal


def test_central_full_state_never_enters_actor_or_shares_parameters() -> None:
    torch.manual_seed(48)
    actor = PolicyValueNet(hidden_dim=8)
    critic = CentralQCritic(hidden_dim=8)
    inputs = policy_inputs()
    before_logits, before_values = actor(*inputs)
    state = torch.cat((torch.zeros(4, 432), inputs[1]), dim=-1)
    q_before = critic(state, inputs[2])
    changed = state.clone()
    changed[:, :432] = torch.randn(4, 432)
    q_after = critic(changed, inputs[2])
    after_logits, after_values = actor(*inputs)
    assert not torch.equal(q_before, q_after)
    assert torch.equal(before_logits, after_logits) and torch.equal(before_values, after_values)
    assert {p.data_ptr() for p in actor.parameters()}.isdisjoint({p.data_ptr() for p in critic.parameters()})
    q_after.square().mean().backward()
    assert all(p.grad is None for p in actor.parameters())


@pytest.mark.parametrize("algorithm", ["ippo", "vrpo"])
def test_actor_and_critic_update_with_fixed_seat_estimates_has_finite_gradients(algorithm) -> None:
    torch.manual_seed(72)
    actor = PolicyValueNet(hidden_dim=8)
    critic = CentralQCritic(hidden_dim=8)
    inputs = policy_inputs()
    ids = torch.tensor([[0, 1], [2, 3]])
    done = torch.tensor([[False, False], [True, True]])
    trunc = torch.zeros_like(done)
    reward = torch.zeros(2, 2, 4)
    reward[-1] = torch.tensor([[1.0, -1, 1, -1], [-1.0, 1, -1, 1]])
    indices = torch.tensor([0, 1, 2, 1])
    actions = inputs[2].gather(1, indices[:, None]).squeeze(1)
    logits, scalar_v = actor(*inputs)
    old_log_probs = actor.log_prob(logits.detach(), actions, legal_next_tokens=inputs[2])
    if algorithm == "ippo":
        values = expand_team_values(scalar_v.reshape(2, 2), ids)
        successor = torch.cat((values[1:], torch.zeros_like(values[:1])))
        adv, target = compute_gae(reward, values, successor, done, trunc)
        total, _ = ippo_loss(logits, scalar_v, actions, old_log_probs, acting_player_values(adv, ids).flatten(), acting_player_values(target, ids).flatten(), inputs[2])
    else:
        state = torch.cat((torch.rand(4, 432), inputs[1]), dim=-1)
        qs = critic(state, inputs[2])
        values = masked_expected_values(qs, logits.softmax(-1), inputs[3]).reshape(2, 2, 4)
        selected = qs[torch.arange(4), indices].reshape(2, 2, 4)
        successor = torch.cat((values[1:], torch.zeros_like(values[:1])))
        adv, target = compute_q_boost(reward, selected, values, successor, done, trunc)
        total, _ = vrpo_loss(logits, acting_player_values(selected, ids).flatten(), actions, old_log_probs, acting_player_values(adv, ids).flatten(), acting_player_values(target, ids).flatten(), inputs[2])
    total.backward()
    actor_gradients = [p.grad for p in actor.parameters() if p.grad is not None]
    assert actor_gradients and all(torch.isfinite(g).all() for g in actor_gradients)
    if algorithm == "vrpo":
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
        # Detached critic labels do not train the unused actor value head.
        assert all(p.grad is None for p in actor.value_head.parameters())
    else:
        assert all(p.grad is None for p in critic.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.value_head.parameters())
