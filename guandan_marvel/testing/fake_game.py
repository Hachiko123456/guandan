# SPDX-License-Identifier: MIT
"""Test-only fake game contract for exercising the game-agnostic core.

This is not a Guandan implementation and must never be selected as a training backend.
"""
from __future__ import annotations

import torch

from guandan_marvel.config import GameSpec, ModelConfig, TrainConfig
from guandan_marvel.model import TensorDict

FAKE_GAME_SPEC = GameSpec(
    identifier="fake-core-test-only",
    num_players=4,
    num_tokens=32,
    pad_token=0,
    num_channels=6,
    max_game_horizon=5,
    max_observation_tokens=4,
    max_action_tokens=3,
    max_available_actions=4,
    protocol_version="fake-v1",
)


def fake_train_config(*, enable_compile: bool = False) -> TrainConfig:
    cfg = TrainConfig(
        game=FAKE_GAME_SPEC,
        device="cpu",
        model=ModelConfig(
            architecture="v1.3",
            history_dim=16,
            feature_dim=24,
            decoder_layers=1,
            decoder_heads=4,
            state_layers=1,
            action_layers=1,
            fuse_layers=1,
        ),
    )
    cfg.compile.enabled = enable_compile
    return cfg


def fake_rollout(*, batch_size: int = 2, steps: int = 3) -> TensorDict:
    spec = FAKE_GAME_SPEC
    players, actions = spec.num_players, spec.max_available_actions
    obs_tokens = torch.randint(1, spec.num_tokens, (batch_size, steps, players, spec.max_observation_tokens))
    obs_channels = torch.randint(0, 2, (batch_size, steps, players, spec.num_channels), dtype=torch.uint8)
    player_ids = torch.arange(steps).remainder(players).expand(batch_size, -1).clone()
    num_steps = torch.full((batch_size,), steps, dtype=torch.long)
    if batch_size > 1:
        num_steps[-1] = max(1, steps - 1)
        player_ids[-1, num_steps[-1]:] = -1
    avail_tokens = torch.randint(1, spec.num_tokens, (batch_size, steps, actions, spec.max_action_tokens))
    avail_weights = torch.ones((batch_size, steps, actions), dtype=torch.float32)
    return TensorDict(
        {
            "num_steps": num_steps,
            "player_ids": player_ids,
            "obs_tokens": obs_tokens,
            "obs_channels": obs_channels,
            "avail_tokens": avail_tokens,
            "avail_weights": avail_weights,
        },
        batch_size=[batch_size],
    )


__all__ = ["FAKE_GAME_SPEC", "fake_rollout", "fake_train_config"]
