from __future__ import annotations

import torch

from guandan.model import (
    PROTOCOL_MAX_LEGAL_NEXT_TOKENS,
    PROTOCOL_MAX_OBSERVATION_TOKENS,
    PROTOCOL_STATE_CHANNELS,
    PROTOCOL_TOKEN_VOCAB_SIZE,
    PolicyValueNet,
)
from guandan.training import PolicyValueNet as TrainingPolicyValueNet


def make_inputs(*, batch: int = 4, history: int = 48, candidates: int = 9):
    generator = torch.Generator().manual_seed(17)
    observation_tokens = torch.randint(
        1,
        PROTOCOL_TOKEN_VOCAB_SIZE,
        (batch, history),
        generator=generator,
        dtype=torch.int32,
    )
    observation_tokens[:, -7:] = 0
    state_channels = torch.randn(batch, PROTOCOL_STATE_CHANNELS, generator=generator)
    legal_next_tokens = torch.zeros(batch, candidates, dtype=torch.int32)
    for row in range(batch):
        legal_next_tokens[row] = torch.arange(1, candidates + 1, dtype=torch.int32) + row * candidates
    legal_mask = torch.zeros(batch, candidates, dtype=torch.bool)
    legal_mask[:, :5] = True
    return observation_tokens, state_channels, legal_next_tokens, legal_mask


def test_protocol_defaults_and_forward_shapes_and_mask() -> None:
    model = PolicyValueNet(hidden_dim=32)
    assert model.token_vocab_size == PROTOCOL_TOKEN_VOCAB_SIZE
    assert model.num_state_channels == PROTOCOL_STATE_CHANNELS
    assert model.max_observation_tokens == PROTOCOL_MAX_OBSERVATION_TOKENS
    assert model.max_legal_next_tokens == PROTOCOL_MAX_LEGAL_NEXT_TOKENS
    assert TrainingPolicyValueNet is PolicyValueNet

    inputs = make_inputs()
    logits, values = model(*inputs)

    assert logits.shape == (4, 9)
    assert values.shape == (4,)
    assert torch.isfinite(logits[inputs[3]]).all()
    assert torch.isneginf(logits[~inputs[3]]).all()


def test_forward_backward_and_distribution_helpers_are_finite() -> None:
    model = PolicyValueNet(hidden_dim=24)
    inputs = make_inputs(batch=3, candidates=7)
    logits, values = model(*inputs)

    entropy = model.entropy(logits)
    sampled = model.sample_actions(*inputs, deterministic=False)
    log_prob = model.log_prob(logits, sampled, legal_next_tokens=inputs[2])
    loss = values.square().mean() + entropy.mean() - log_prob.mean()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(loss)

    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_sampling_returns_only_legal_token_ids() -> None:
    model = PolicyValueNet(hidden_dim=32)
    inputs = make_inputs(batch=8, candidates=11)
    model.train()
    torch.manual_seed(23)

    for _ in range(20):
        actions = model.sample_actions(*inputs, deterministic=False)
        legal = inputs[3] & inputs[2].eq(actions.unsqueeze(1))
        assert legal.any(dim=1).all()


def test_eval_sampling_is_deterministic() -> None:
    model = PolicyValueNet(hidden_dim=32)
    inputs = make_inputs(batch=5, candidates=8)
    model.eval()

    first_logits, first_values = model(*inputs)
    first_actions = model.sample_actions(*inputs)
    second_logits, second_values = model(*inputs)
    second_actions = model.sample_actions(*inputs)

    assert torch.equal(first_logits, second_logits)
    assert torch.equal(first_values, second_values)
    assert torch.equal(first_actions, second_actions)
