# SPDX-License-Identifier: MIT
"""Game-agnostic MARVEL training core for the Guandan migration.

The independent namespace is deliberate: importing this package must not resolve or
shadow the reference project's top-level ``marvel`` package.
"""
from .buffer import BufferConfig, BufferFieldSpec, TrajectoryBuffer, trajectory_field_specs
from .config import CompileConfig, DDPConfig, GameSpec, ModelConfig, OptimizerConfig, TrainConfig, deserialize_train_config
from .model import Agent, CentralCritic, TensorDict

__version__ = "0.1.0"

__all__ = ["Agent", "BufferConfig", "BufferFieldSpec", "CentralCritic", "CompileConfig", "DDPConfig", "GameSpec", "ModelConfig", "OptimizerConfig", "TensorDict", "TrainConfig", "TrajectoryBuffer", "deserialize_train_config", "trajectory_field_specs"]
