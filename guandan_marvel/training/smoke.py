# SPDX-License-Identifier: MIT
"""Explicit reduced-horizon Python reference game contract for local M03 smoke."""
from __future__ import annotations

from dataclasses import dataclass

from guandan_marvel import python_game as guandan_game
from guandan_marvel.config import GameSpec


@dataclass(frozen=True, slots=True)
class PythonReferenceSmokeGame:
    max_game_horizon: int = 512

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_game_horizon) <= guandan_game.max_game_horizon:
            raise ValueError("smoke horizon must be within the accepted V2 protocol horizon")

    @property
    def identifier(self) -> str:
        return f"{guandan_game.GAME_IDENTIFIER}-local-smoke-T{self.max_game_horizon}"

    @property
    def GAME_IDENTIFIER(self) -> str:
        return self.identifier

    BACKEND = guandan_game.BACKEND
    num_players = guandan_game.num_players
    num_tokens = guandan_game.num_tokens
    num_channels = guandan_game.num_channels
    pad_token = guandan_game.pad_token
    max_available_actions = guandan_game.max_available_actions
    max_observation_tokens = guandan_game.max_observation_tokens
    max_action_tokens = guandan_game.max_action_tokens
    protocol_version = guandan_game.protocol_version
    actions_from_np = staticmethod(guandan_game.actions_from_np)

    def EnvBatch(self, batch_size: int):
        return guandan_game.EnvBatch(
            batch_size,
            backend=guandan_game.BACKEND,
            auto_reset=True,
            max_horizon=int(self.max_game_horizon),
        )

    @property
    def spec(self) -> GameSpec:
        return GameSpec.from_game(self)


@dataclass(frozen=True, slots=True)
class CppSmokeGame:
    """Reduced-horizon wrapper over the actual C++ backend for local DDP tests.

    Identity/build metadata intentionally remain identical to ``cpp_game`` so
    evaluation checkpoints exercise the same strict C++ loader.  The reduced
    horizon is recorded in the resolved GameSpec and is smoke-only.
    """

    max_game_horizon: int = 512

    def __post_init__(self) -> None:
        from guandan_marvel import cpp_game
        if not 1 <= int(self.max_game_horizon) <= cpp_game.max_game_horizon:
            raise ValueError("smoke horizon must be within the accepted V2 protocol horizon")

    @property
    def identifier(self) -> str:
        from guandan_marvel import cpp_game
        return str(cpp_game.identifier)

    @property
    def GAME_IDENTIFIER(self) -> str:
        return self.identifier

    @property
    def BACKEND(self) -> str:
        return "cpp"

    @property
    def __name__(self) -> str:
        from guandan_marvel import cpp_game
        return str(cpp_game.__name__)

    @property
    def num_players(self) -> int:
        from guandan_marvel import cpp_game
        return int(cpp_game.num_players)

    @property
    def num_tokens(self) -> int:
        from guandan_marvel import cpp_game
        return int(cpp_game.num_tokens)

    @property
    def num_channels(self) -> int:
        from guandan_marvel import cpp_game
        return int(cpp_game.num_channels)

    @property
    def pad_token(self) -> int:
        from guandan_marvel import cpp_game
        return int(cpp_game.pad_token)

    @property
    def max_available_actions(self) -> int:
        from guandan_marvel import cpp_game
        return int(cpp_game.max_available_actions)

    @property
    def max_observation_tokens(self) -> int:
        from guandan_marvel import cpp_game
        return int(cpp_game.max_observation_tokens)

    @property
    def max_action_tokens(self) -> int:
        from guandan_marvel import cpp_game
        return int(cpp_game.max_action_tokens)

    @property
    def protocol_version(self) -> str:
        from guandan_marvel import cpp_game
        return str(cpp_game.protocol_version)

    def actions_from_np(self, *args, **kwargs):
        from guandan_marvel import cpp_game
        return cpp_game.actions_from_np(*args, **kwargs)

    def get_backend_build_identity(self):
        from guandan_marvel import cpp_game
        return cpp_game.get_backend_build_identity()

    def EnvBatch(self, batch_size: int):
        from guandan_marvel import cpp_game
        return cpp_game.EnvBatch(
            batch_size, backend="cpp", auto_reset=True,
            max_horizon=int(self.max_game_horizon),
        )

    @property
    def spec(self) -> GameSpec:
        return GameSpec.from_game(self)


__all__ = ["CppSmokeGame", "PythonReferenceSmokeGame"]
