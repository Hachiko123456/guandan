from __future__ import annotations

import torch

from guandan_marvel import cpp_game
from guandan_marvel.buffer import trajectory_memory_budget
from guandan_marvel.config import GameSpec, ModelConfig, TrainConfig
from guandan_marvel.model import Agent
from guandan_marvel.runtime.sampler import Sampler, SamplerConfig


class _CppSmokeGame:
    identifier = "m04-repair-cpp-sampler"
    GAME_IDENTIFIER = identifier
    num_players = cpp_game.num_players
    num_tokens = cpp_game.num_tokens
    pad_token = cpp_game.pad_token
    num_channels = cpp_game.num_channels
    max_game_horizon = 512
    max_observation_tokens = cpp_game.max_observation_tokens
    max_action_tokens = cpp_game.max_action_tokens
    max_available_actions = cpp_game.max_available_actions
    protocol_version = cpp_game.protocol_version
    actions_from_np = staticmethod(cpp_game.actions_from_np)

    @staticmethod
    def EnvBatch(batch_size: int):
        return cpp_game.EnvBatch(batch_size, max_horizon=512, auto_reset=True)


def _agent() -> Agent:
    config = TrainConfig(
        game=GameSpec.from_game(_CppSmokeGame),
        device="cpu",
        model=ModelConfig(
            architecture="v1.3",
            history_dim=8,
            feature_dim=8,
            decoder_layers=1,
            decoder_heads=1,
            state_layers=1,
            action_layers=1,
            fuse_layers=1,
        ),
    )
    return Agent(config)


def _assert_complete_trajectory(trajectory, index: int) -> int:
    steps = int(trajectory["num_steps"][index])
    assert 0 < steps < _CppSmokeGame.max_game_horizon
    terminal = trajectory["is_terminal"][index, :steps]
    assert int(terminal.sum()) == 1
    assert bool(terminal[-1])
    assert set(trajectory["rewards"][index, steps - 1].tolist()) == {-1.0, 1.0}
    assert torch.isfinite(trajectory["policy_probs"][index, :steps]).all()
    assert torch.isfinite(trajectory["behavior_action_prob"][index, :steps]).all()
    assert bool(trajectory["on_policy_valid"][index, :steps].all())
    assert trajectory["obs_tokens"].shape[2:] == (4, cpp_game.max_observation_tokens)
    assert trajectory["avail_tokens"].shape[2:] == (
        cpp_game.max_available_actions,
        cpp_game.max_action_tokens,
    )
    return steps


def test_real_transformer_on_policy_cpp_sampler_first_step() -> None:
    torch.manual_seed(20261009)
    agent = _agent()
    sampler = Sampler(config=SamplerConfig(
        game=_CppSmokeGame,
        device="cpu",
        rollout_batch_size=1,
        collect_batch_size=0,
        sampling_temperature=1.0,
        purpose="on_policy",
        env_seed=101,
    ))
    sampler.prepare_agent_cache(agent_model=agent)
    done = sampler.rollout_step(agent_model=agent)
    assert done.shape == (1,)
    assert sampler.total_env_steps == sampler.attempted_env_steps == 1
    assert sampler.failed_env_steps == sampler.rejected_rollout_steps == 0
    assert int(sampler.rollout_buffer["num_steps"][0]) == 1
    assert bool(sampler.rollout_buffer["on_policy_valid"][0, 0])
    assert torch.isfinite(sampler.rollout_buffer["policy_probs"][0, 0]).all()
    assert not sampler.failed


def test_real_transformer_cpp_sampler_staggered_terminals_two_collects_and_auto_reset() -> None:
    budget = trajectory_memory_budget(
        GameSpec.from_game(_CppSmokeGame),
        rollout_batch_size=2,
        collect_batch_size=2,
        world_size=1,
    )
    assert budget["sampler_bytes_per_rank"] < 128 * 2**20

    torch.manual_seed(20261010)
    agent = _agent()
    sampler = Sampler(config=SamplerConfig(
        game=_CppSmokeGame,
        device="cpu",
        rollout_batch_size=2,
        collect_batch_size=2,
        sampling_temperature=1.0,
        purpose="on_policy",
        env_seed=202,
    ))

    first = sampler.collect(agent_model=agent)
    first_lengths = [_assert_complete_trajectory(first, index) for index in range(2)]
    assert len(set(first_lengths)) > 1
    first_total_env_steps = sampler.total_env_steps
    first_total_game_steps = sampler.total_game_steps
    assert not sampler.failed
    assert torch.all(sampler.parsed_envs_step["player_ids"] >= 0)

    second = sampler.collect(agent_model=agent)
    second_lengths = [_assert_complete_trajectory(second, index) for index in range(2)]
    assert sampler.total_env_steps > first_total_env_steps
    assert sampler.total_game_steps > first_total_game_steps
    assert torch.all(sampler.parsed_envs_step["player_ids"] >= 0)
    assert not sampler.failed
    # Every returned record is one complete hand. Auto-reset frames remain in
    # rollout state and are not appended to the preceding terminal trajectory.
    assert all(length > 0 for length in first_lengths + second_lengths)
