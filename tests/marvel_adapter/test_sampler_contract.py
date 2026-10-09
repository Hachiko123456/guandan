from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from guandan.action_state import COMMIT_TOKEN, ProtocolError, TokenCodec
from guandan.combos import CombinationKind
from guandan_marvel import python_game as game
from guandan_marvel.config import GameSpec, ModelConfig, TrainConfig
from guandan_marvel.model import Agent
from guandan_marvel.model.types import AgentOutput
from guandan_marvel.runtime.sampler import Sampler, SamplerConfig, SamplerFailedStateError
from guandan_marvel.training.losses import compute_clipped_pg_loss


class _DecoderConfig:
    num_key_value_heads = 1
    num_attention_heads = 1
    hidden_size = 4


class _Decoder:
    config = _DecoderConfig()


class _NonOneHotAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.decoder = _Decoder()
        self.decoder_n_layers = 1

    def forward(self, *, rollout, **_kwargs) -> AgentOutput:
        weights = rollout["avail_weights"]
        legal = weights > 0
        scores = torch.arange(
            1, weights.shape[-1] + 1, dtype=torch.float32, device=weights.device
        ).view(1, 1, -1).expand_as(weights.float())
        probs = scores * legal
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return AgentOutput(probs)


class _CommitFirstAgent(_NonOneHotAgent):
    def forward(self, *, rollout, **_kwargs) -> AgentOutput:
        weights = rollout["avail_weights"]
        rows = rollout["avail_tokens"]
        legal = weights > 0
        commits = (rows[..., 6] == game.ACTION_COMMIT) & legal
        chosen = torch.where(
            commits.any(-1), commits.float().argmax(-1), legal.float().argmax(-1)
        )
        # Non-one-hot raw policy: the selected action gets the largest mass but
        # the remaining legal actions retain probability.
        scores = legal.float()
        scores.scatter_add_(-1, chosen.unsqueeze(-1), torch.full_like(chosen.unsqueeze(-1), 3.0, dtype=torch.float32))
        probs = scores / scores.sum(dim=-1, keepdim=True)
        return AgentOutput(probs)


def _game_type(*, horizon: int = 16, env_horizon: int | None = None, env_factory=None):
    actual_env_horizon = horizon if env_horizon is None else env_horizon

    class _Game:
        identifier = f"repair-test-{horizon}-{actual_env_horizon}"
        GAME_IDENTIFIER = identifier
        num_players = game.num_players
        num_tokens = game.num_tokens
        pad_token = game.pad_token
        num_channels = game.num_channels
        max_game_horizon = horizon
        max_observation_tokens = game.max_observation_tokens
        max_action_tokens = game.max_action_tokens
        max_available_actions = game.max_available_actions
        protocol_version = game.protocol_version
        actions_from_np = staticmethod(game.actions_from_np)

        @staticmethod
        def EnvBatch(batch_size: int):
            if env_factory is not None:
                return env_factory(batch_size)
            return game.EnvBatch(batch_size, max_horizon=actual_env_horizon)

    return _Game


def _tiny_agent(game_module) -> Agent:
    cfg = TrainConfig(
        game=GameSpec.from_game(game_module),
        device="cpu",
        model=ModelConfig(
            architecture="v1.3", history_dim=8, feature_dim=8,
            decoder_layers=1, decoder_heads=1,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    return Agent(cfg)


def _sampler(*, game_module=None, temperature=1.0, purpose="on_policy", batch=1, collect=0, agent=None):
    game_module = game_module or _game_type()
    sampler = Sampler(config=SamplerConfig(
        game=game_module, device="cpu", rollout_batch_size=batch,
        collect_batch_size=collect, sampling_temperature=temperature,
        purpose=purpose, env_seed=0,
    ))
    model = agent or _NonOneHotAgent()
    sampler.prepare_agent_cache(agent_model=model)
    return sampler, model


def _forced_index_for_token(sampler: Sampler, token: int, slot: int = 0) -> int:
    rows = sampler.parsed_envs_step["avail_tokens"][slot]
    weights = sampler.parsed_envs_step["avail_weights"][slot]
    matches = ((rows[:, 5] == token) & (weights > 0)).nonzero().flatten()
    assert len(matches) == 1
    return int(matches[0])


def test_on_policy_temperature_one_records_matching_policy_and_behavior_probability() -> None:
    sampler, agent = _sampler()
    sampler.rollout_step(agent_model=agent)
    taken = int(sampler.rollout_buffer["action_taken"][0, 0])
    raw = float(sampler.rollout_buffer["policy_probs"][0, 0, taken])
    behavior = float(sampler.rollout_buffer["behavior_action_prob"][0, 0])
    assert 0.0 < raw < 1.0
    assert behavior == pytest.approx(raw)
    assert bool(sampler.rollout_buffer["on_policy_valid"][0, 0])


@pytest.mark.parametrize("temperature", [0.0, 0.5, 2.0])
def test_on_policy_rejects_non_unit_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature=1.0"):
        SamplerConfig(
            game=_game_type(), device="cpu", rollout_batch_size=1,
            collect_batch_size=1, sampling_temperature=temperature,
            purpose="on_policy",
        )


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf"), -0.1])
def test_invalid_temperatures_are_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        SamplerConfig(
            game=_game_type(), device="cpu", rollout_batch_size=1,
            collect_batch_size=0, sampling_temperature=temperature,
            purpose="evaluation",
        )


@pytest.mark.parametrize("temperature", [0.0, 0.5, 2.0])
def test_evaluation_temperature_records_actual_selected_behavior_probability(temperature: float) -> None:
    torch.manual_seed(13)
    sampler, agent = _sampler(temperature=temperature, purpose="evaluation")
    sampler.rollout_step(agent_model=agent)
    taken = int(sampler.rollout_buffer["action_taken"][0, 0])
    policy = sampler.rollout_buffer["policy_probs"][0, 0]
    weights = sampler.rollout_buffer["avail_weights"][0, 0]
    behavior = float(sampler.rollout_buffer["behavior_action_prob"][0, 0])
    if temperature == 0.0:
        expected = 1.0
    else:
        legal = weights > 0
        raw_logits = policy.clamp(min=1e-9).log() - weights.clamp(min=1e-9).log()
        scaled = (raw_logits / temperature + weights.clamp(min=1e-9).log()).masked_fill(~legal, torch.finfo(torch.float32).min)
        expected = float(torch.softmax(scaled, dim=0)[taken])
    assert behavior == pytest.approx(expected, rel=1e-6, abs=1e-7)
    assert not bool(sampler.rollout_buffer["on_policy_valid"][0, 0])


def test_on_policy_forced_action_is_rejected_before_cache_or_counter_mutation() -> None:
    sampler, agent = _sampler()
    before_lens = sampler.agent_kv._seq_lens.clone()
    with pytest.raises(ValueError, match="forbidden"):
        sampler.rollout_step(agent_model=agent, forced_action_indices=torch.tensor([0]))
    assert sampler.total_env_steps == sampler.total_game_steps == sampler.attempted_env_steps == 0
    assert int(sampler.rollout_buffer["num_steps"][0]) == 0
    assert torch.equal(sampler.agent_kv._seq_lens, before_lens)
    assert not sampler.failed


def test_on_policy_rollout_is_accepted_by_ppo_with_behavior_reference_ratio_one() -> None:
    game_module = _game_type()
    sampler, agent = _sampler(game_module=game_module, batch=2, purpose="on_policy")
    sampler.rollout_step(agent_model=agent)
    rollout = sampler.rollout_buffer[:2]
    advantage = torch.zeros_like(rollout["behavior_action_prob"])
    advantage[0, 0], advantage[1, 0] = -1.0, 1.0
    result = compute_clipped_pg_loss(
        rollout,
        clip_coef=0.2,
        avail_probs_new=rollout["policy_probs"].clone(),
        player_advantage=advantage,
    )
    assert torch.isfinite(result.pg_loss_mean)
    assert result.clip_frac.item() == 0.0
    assert result.kl_ref_mean.item() == pytest.approx(0.0, abs=1e-7)


def test_test_mode_forced_action_records_one_behavior_probability_and_is_not_trainable() -> None:
    sampler, agent = _sampler(purpose="test")
    sampler.rollout_step(agent_model=agent, forced_action_indices=torch.tensor([0]))
    assert float(sampler.rollout_buffer["behavior_action_prob"][0, 0]) == 1.0
    assert not bool(sampler.rollout_buffer["on_policy_valid"][0, 0])
    rollout = sampler.rollout_buffer[:1]
    with pytest.raises(ValueError, match="on_policy-valid"):
        compute_clipped_pg_loss(
            rollout,
            clip_coef=0.2,
            avail_probs_new=rollout["policy_probs"].clone(),
            player_advantage=torch.ones_like(rollout["behavior_action_prob"]),
        )


def test_horizon_failure_keeps_completed_counters_and_buffer_stable_then_poisons_real_cache() -> None:
    game_module = _game_type(horizon=16, env_horizon=3)
    agent = _tiny_agent(game_module)
    sampler, _ = _sampler(game_module=game_module, purpose="test", agent=agent)
    family = _forced_index_for_token(sampler, TokenCodec.family(CombinationKind.SINGLE))
    sampler.rollout_step(agent_model=agent, forced_action_indices=torch.tensor([family]))
    rows = sampler.parsed_envs_step["avail_tokens"][0]
    weights = sampler.parsed_envs_step["avail_weights"][0]
    card = int(((((rows[:, 3] - game.ACTION_KIND_BASE) == 6) & (weights > 0)).nonzero().flatten())[0])
    sampler.rollout_step(agent_model=agent, forced_action_indices=torch.tensor([card]))
    before_env = sampler.envs.serialize()
    before_cache = sampler.agent_kv._seq_lens.clone()
    before = (sampler.total_env_steps, sampler.total_game_steps, int(sampler.rollout_buffer["num_steps"][0]))
    commit = _forced_index_for_token(sampler, COMMIT_TOKEN)
    with pytest.raises(game.TrajectoryHorizonError):
        sampler.rollout_step(agent_model=agent, forced_action_indices=torch.tensor([commit]))
    after = (sampler.total_env_steps, sampler.total_game_steps, int(sampler.rollout_buffer["num_steps"][0]))
    assert before == after == (2, 0, 2)
    assert sampler.envs.serialize() == before_env
    assert torch.all(sampler.agent_kv._seq_lens == before_cache + 1)
    assert sampler.failed and sampler.failure_info["stage"] == "environment_step"
    assert sampler.failure_info["exact_resume_allowed"] is False
    with pytest.raises(SamplerFailedStateError):
        sampler.rollout_step(agent_model=agent)
    with pytest.raises(SamplerFailedStateError):
        sampler.collect(agent_model=agent)


class _CorruptSecondSlot:
    def __init__(self, batch_size: int) -> None:
        self.inner = game.EnvBatch(batch_size, max_horizon=16)

    def reset(self, seed: int):
        return self.inner.reset(seed)

    def serialize(self):
        return self.inner.serialize()

    def step(self, actions):
        actions = list(actions)
        victim = actions[1]
        row = list(victim.row)
        row[2] += 1
        actions[1] = game.V2Action(
            victim.token_id, tuple(row), victim.phase_token,
            victim.prefix_stage + 1, victim.kind, victim.is_commit,
        )
        return self.inner.step(actions)


def test_multislot_illegal_action_failure_is_atomic_and_poisons_sampler() -> None:
    game_module = _game_type(horizon=16, env_factory=_CorruptSecondSlot)
    agent = _tiny_agent(game_module)
    sampler, _ = _sampler(game_module=game_module, batch=2, purpose="on_policy", agent=agent)
    before = sampler.envs.serialize()
    with pytest.raises(ProtocolError, match="stale|forged"):
        sampler.rollout_step(agent_model=agent)
    assert sampler.envs.serialize() == before
    assert sampler.total_env_steps == sampler.total_game_steps == 0
    assert sampler.rollout_buffer["num_steps"].tolist() == [0, 0]
    assert sampler.failed and sampler.failed_env_steps == 2


def test_buffer_write_failure_after_environment_step_does_not_publish_counts_or_completed_trajectory(monkeypatch) -> None:
    game_module = _game_type(horizon=16)
    agent = _tiny_agent(game_module)
    sampler, _ = _sampler(game_module=game_module, purpose="on_policy", collect=1, agent=agent)
    before_env = sampler.envs.serialize()

    def fail_add_step(*_args, **_kwargs):
        raise RuntimeError("injected buffer failure")

    monkeypatch.setattr(sampler.rollout_buffer, "add_step", fail_add_step)
    with pytest.raises(RuntimeError, match="injected buffer failure"):
        sampler.rollout_step(agent_model=agent)
    assert sampler.envs.serialize() != before_env
    assert sampler.total_env_steps == sampler.total_game_steps == 0
    assert sampler.commit_buffer_size == 0
    assert sampler.failed and sampler.failure_info["stage"] == "rollout_buffer_write"
    with pytest.raises(SamplerFailedStateError):
        sampler.rollout_step(agent_model=agent)
