# SPDX-License-Identifier: MIT
"""Game-agnostic MARVEL training core for the Guandan migration.

The independent namespace is deliberate: importing this package must not resolve or
shadow the reference project's top-level ``marvel`` package.
"""
from . import game as python_game
from . import cpp_game
from .backends import DEFAULT_BACKEND, SUPPORTED_BACKENDS, get_game_backend

# M05 cutover: C++ is the official training default.  The Python module remains
# explicitly available as ``python_game`` and through get_game_backend().
game = cpp_game

from .buffer import BufferConfig, BufferFieldSpec, TrajectoryBuffer, trajectory_field_specs, trajectory_memory_budget, require_trajectory_memory_budget
from .config import CompileConfig, DDPConfig, GameSpec, ModelConfig, OptimizerConfig, TrainConfig, deserialize_train_config
from .model import Agent, CentralCritic, TensorDict
from .training import IPPOTrainer, VRPOTrainer, PythonReferenceSmokeGame

__version__ = "0.1.0"

__all__ = ["Agent", "BufferConfig", "BufferFieldSpec", "CentralCritic", "CompileConfig", "DDPConfig", "DEFAULT_BACKEND", "GameSpec", "ModelConfig", "OptimizerConfig", "SUPPORTED_BACKENDS", "TensorDict", "TrainConfig", "TrajectoryBuffer", "cpp_game", "deserialize_train_config", "game", "get_game_backend", "python_game", "trajectory_field_specs", "trajectory_memory_budget", "require_trajectory_memory_budget", "IPPOTrainer", "VRPOTrainer", "PythonReferenceSmokeGame"]
