from __future__ import annotations

import pytest
import torch
from torch import nn

from guandan.cards import Rank
from guandan.environment_types import GameConfig
from guandan.combos import CombinationKind
from guandan_marvel import python_game as game
from guandan_marvel.config import GameSpec, ModelConfig, TrainConfig
from guandan_marvel.model import Agent
from guandan_marvel.model.types import AgentOutput
from guandan_marvel.runtime.sampler import Sampler, SamplerConfig
from guandan_marvel.testing.complex_actions import DIRECTED_ACTION_CASES, execute_directed_case


@pytest.mark.parametrize("case", DIRECTED_ACTION_CASES, ids=lambda case: case.name)
def test_every_combination_family_executes_full_v2_prefix_commit_against_v1(case) -> None:
    result = execute_directed_case(case)
    assert result["commit_executed"]
    assert result["serialization_resume"]
    assert result["v1_transition_equal"]
    assert result["card_conservation"]
    assert result["public_event_tokens"] <= game.max_observation_tokens


def test_directed_inventory_covers_current_enum_and_wild_modes_and_long_prefix() -> None:
    assert {case.family for case in DIRECTED_ACTION_CASES} == set(CombinationKind)
    modes = {case.wild_mode for case in DIRECTED_ACTION_CASES}
    assert {"none", "natural", "substitute", "substitute_multiple", "mixed_multiple"} <= modes
    results = [execute_directed_case(case) for case in DIRECTED_ACTION_CASES]
    assert max(result["prefix_tokens"] for result in results) >= 15
    assert any(result["wild_assignments"] >= 2 for result in results)


class _DecoderConfig:
    num_key_value_heads = 1
    num_attention_heads = 1
    hidden_size = 4


class _Decoder:
    config = _DecoderConfig()


class _CommitFirstTestAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.decoder = _Decoder()
        self.decoder_n_layers = 1

    def forward(self, *, rollout, **_kwargs) -> AgentOutput:
        weights = rollout["avail_weights"]
        rows = rollout["avail_tokens"]
        legal = weights > 0
        commits = (rows[..., 6] == game.ACTION_COMMIT) & legal
        chosen = torch.where(
            commits.any(-1), commits.float().argmax(-1), legal.float().argmax(-1)
        )
        scores = legal.float()
        scores.scatter_add_(
            -1, chosen.unsqueeze(-1),
            torch.full_like(chosen.unsqueeze(-1), 3.0, dtype=torch.float32),
        )
        return AgentOutput(scores / scores.sum(dim=-1, keepdim=True))


class _ContinuousTwoEnvGame:
    identifier = "repair-continuous-two-env"
    GAME_IDENTIFIER = identifier
    num_players = game.num_players
    num_tokens = game.num_tokens
    pad_token = game.pad_token
    num_channels = game.num_channels
    max_game_horizon = 512
    max_observation_tokens = game.max_observation_tokens
    max_action_tokens = game.max_action_tokens
    max_available_actions = game.max_available_actions
    protocol_version = game.protocol_version
    actions_from_np = staticmethod(game.actions_from_np)

    @staticmethod
    def EnvBatch(batch_size: int):
        assert batch_size == 2
        configs = [
            GameConfig(level_rank=Rank.THREE, max_token_steps=512),
            GameConfig(level_rank=Rank.FOUR, max_token_steps=512),
        ]
        return game.EnvBatch(2, configs=configs, max_horizon=512, auto_reset=True)


def test_multi_env_different_terminal_times_auto_reset_and_two_consecutive_collects() -> None:
    sampler = Sampler(config=SamplerConfig(
        game=_ContinuousTwoEnvGame, device="cpu", rollout_batch_size=2,
        collect_batch_size=2, sampling_temperature=0.0, purpose="test", env_seed=0,
    ))
    agent = _CommitFirstTestAgent()
    first = sampler.collect(agent_model=agent)
    first_lengths = first["num_steps"].tolist()
    assert len(set(first_lengths)) > 1
    for index, length in enumerate(first_lengths):
        assert first["is_terminal"][index, length - 1]
        assert not bool(first["on_policy_valid"][index, :length].any())
    completed_after_first = sampler.total_env_steps
    second = sampler.collect(agent_model=agent)
    assert sampler.total_env_steps > completed_after_first
    for index, length in enumerate(second["num_steps"].tolist()):
        assert second["is_terminal"][index, length - 1]
    assert not sampler.failed


class _TransformerGame:
    identifier = "repair-real-transformer-complete-game"
    GAME_IDENTIFIER = identifier
    num_players = game.num_players
    num_tokens = game.num_tokens
    pad_token = game.pad_token
    num_channels = game.num_channels
    max_game_horizon = 512
    max_observation_tokens = game.max_observation_tokens
    max_action_tokens = game.max_action_tokens
    max_available_actions = game.max_available_actions
    protocol_version = game.protocol_version
    actions_from_np = staticmethod(game.actions_from_np)

    @staticmethod
    def EnvBatch(batch_size: int):
        return game.EnvBatch(batch_size, max_horizon=512, auto_reset=True)


def _forced_commit_first(sampler: Sampler) -> torch.Tensor:
    weights = sampler.parsed_envs_step["avail_weights"]
    commits = sampler.parsed_envs_step["avail_is_commit"] & (weights > 0)
    legal = weights > 0
    return torch.where(commits.any(-1), commits.float().argmax(-1), legal.float().argmax(-1))


@pytest.mark.slow
def test_real_small_transformer_agent_reaches_complete_terminal_only_in_test_mode() -> None:
    cfg = TrainConfig(
        game=GameSpec.from_game(_TransformerGame),
        device="cpu",
        model=ModelConfig(
            architecture="v1.3", history_dim=8, feature_dim=8,
            decoder_layers=1, decoder_heads=1,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    agent = Agent(cfg)
    sampler = Sampler(config=SamplerConfig(
        game=_TransformerGame, device="cpu", rollout_batch_size=1,
        collect_batch_size=1, sampling_temperature=1.0, purpose="test", env_seed=0,
    ))
    sampler.prepare_agent_cache(agent_model=agent)
    while sampler.commit_buffer_size < 1:
        sampler.rollout_step(
            agent_model=agent,
            forced_action_indices=_forced_commit_first(sampler),
        )
    trajectory = sampler.commit_buffer[:1].clone()
    steps = int(trajectory["num_steps"][0])
    assert 0 < steps < _TransformerGame.max_game_horizon
    assert trajectory["is_terminal"][0, steps - 1]
    assert not bool(trajectory["on_policy_valid"][0, :steps].any())
    assert torch.isfinite(trajectory["policy_probs"][0, :steps]).all()
    assert torch.all(trajectory["behavior_action_prob"][0, :steps] == 1.0)
