"""Guandan MARVEL V2 C++/pybind runtime.

The extension is explicit: importing this module fails if the C++ artifact was
not built. It never falls back to the Python oracle.
"""

from guandan_marvel.config import GameSpec

from ._guandan_cpp import *  # noqa: F401,F403
from ._guandan_cpp import actions_from_np as _actions_from_np


def actions_from_np(rows):
    """Decode one or more structured rows into immutable C++ action records."""

    return tuple(_actions_from_np(rows))


GAME_SPEC = GameSpec(
    identifier=GAME_IDENTIFIER,
    num_players=num_players,
    num_tokens=num_tokens,
    pad_token=pad_token,
    num_channels=num_channels,
    max_game_horizon=max_game_horizon,
    max_observation_tokens=max_observation_tokens,
    max_action_tokens=max_action_tokens,
    max_available_actions=max_available_actions,
    protocol_version=protocol_version,
)

__all__ = [name for name in globals() if not name.startswith("_")]
