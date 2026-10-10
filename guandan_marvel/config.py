# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/config.py.
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

CORE_CONFIG_SCHEMA = "guandan-marvel-core-config-v1"
SUPPORTED_MODEL_ARCHITECTURES = frozenset({"v1", "v1.1", "v1.2", "v1.3"})
_MISSING = object()
_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "history_dim": ("history_embd",),
    "feature_dim": ("feature_embd",),
    "decoder_layers": ("decoder_n_layers",),
    "decoder_heads": ("decoder_n_heads",),
    "state_layers": ("n_state_layers",),
    "action_layers": ("n_action_layers",),
    "fuse_layers": ("n_fuse_layers",),
}


@dataclass(frozen=True, slots=True)
class GameSpec:
    """Game-agnostic tensor capacities required by the MARVEL core.

    This is intentionally data-only. A concrete environment adapter is supplied by a
    later migration stage and must not be imported by this package.
    """

    identifier: str
    num_players: int
    num_tokens: int
    pad_token: int
    num_channels: int
    max_game_horizon: int
    max_observation_tokens: int
    max_action_tokens: int
    max_available_actions: int
    protocol_version: str = "unbound"

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("GameSpec.identifier must be non-empty.")
        positive = (
            "num_players",
            "num_tokens",
            "max_game_horizon",
            "max_observation_tokens",
            "max_action_tokens",
            "max_available_actions",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"GameSpec.{name} must be positive.")
        if int(self.num_channels) < 0:
            raise ValueError("GameSpec.num_channels must be non-negative.")
        if not 0 <= int(self.pad_token) < int(self.num_tokens):
            raise ValueError("GameSpec.pad_token must be within the token vocabulary.")

    @classmethod
    def from_game(cls, game: object, *, identifier: str | None = None) -> "GameSpec":
        from guandan_marvel.utils import get_game_attr

        return cls(
            identifier=identifier or str(get_game_attr(game, "identifier")),
            num_players=int(get_game_attr(game, "num_players")),
            num_tokens=int(get_game_attr(game, "num_tokens")),
            pad_token=int(get_game_attr(game, "pad_token")),
            num_channels=int(get_game_attr(game, "num_channels")),
            max_game_horizon=int(get_game_attr(game, "max_game_horizon")),
            max_observation_tokens=int(get_game_attr(game, "max_observation_tokens")),
            max_action_tokens=int(get_game_attr(game, "max_action_tokens")),
            max_available_actions=int(get_game_attr(game, "max_available_actions")),
            protocol_version=str(getattr(game, "protocol_version", "unbound")),
        )


@dataclass
class PPOConfig:
    base_lr: float = 4.0e-4
    agent_epoch: int = 4
    gae_lambda: float = 0.95
    base_clip_coef: float = 0.02
    base_reg_coef: float = 0.1


@dataclass
class VRPOConfig:
    base_lr: float = 4.0e-4
    gae_lambda: float = 0.95
    aux_epoch: int = 4
    replay_buffer_device: str = "cpu"


@dataclass
class ScheduleConfig:
    agent_lr_decay_power: float = 1.0
    critic_lr_decay_power: float = 0.5
    clip_coef_decay_power: float = 1.0
    reg_coef_decay_power: float = 0.5


@dataclass
class OptimizerConfig:
    # Muon is available, but opt-in until a later pilot validates it.
    use_muon: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    muon_beta1: float = 0.95
    weight_decay: float = 1e-2
    grad_max_norm: float = 1.0


@dataclass
class ModelConfig:
    architecture: str = "v1.3"
    history_dim: int = 128
    feature_dim: int = 256
    decoder_layers: int = 4
    decoder_heads: int = 4
    state_layers: int = 3
    action_layers: int = 2
    fuse_layers: int = 2
    ffn_ratio: int = 2

    def __post_init__(self) -> None:
        if self.architecture not in SUPPORTED_MODEL_ARCHITECTURES:
            expected = ", ".join(sorted(SUPPORTED_MODEL_ARCHITECTURES))
            raise ValueError(f"Unsupported model architecture '{self.architecture}'; expected {expected}.")
        if self.history_dim % self.decoder_heads != 0:
            raise ValueError("history_dim must be divisible by decoder_heads.")
        for name in (
            "history_dim", "feature_dim", "decoder_layers", "decoder_heads",
            "state_layers", "action_layers", "fuse_layers", "ffn_ratio",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"ModelConfig.{name} must be positive.")


@dataclass
class CompileConfig:
    enabled: bool = False
    backend: str = "inductor"
    dynamic_shapes: bool = False


@dataclass
class DDPConfig:
    enabled: bool = False
    backend: str = "nccl"
    require_cuda: bool = True
    find_unused_parameters: bool = False


@dataclass
class TrainConfig:
    """Serializable core configuration with an injected, data-only game contract."""

    game: GameSpec | Mapping[str, Any] | None = None
    schema: str = CORE_CONFIG_SCHEMA
    project_name: str = "guandan-marvel"
    run_name: str | None = None
    algo_name: str = "vrpo"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logging_dir: str = ""
    debug: bool = True
    seed: int = 0
    environment_rank_seed_stride: int = 1_000_003
    training_data_domain: str = "training"
    training_data_manifest_hash: str | None = None
    training_data_generator_version: str = "guandan-marvel-env-batch-continuous-prng-v1"
    deterministic: bool = False
    deterministic_disable_tf32: bool = False
    amp_enabled: bool = False
    max_sampler_bytes_per_rank: int | None = None
    checkpoint_replay_buffer: bool = True
    resume_from: str | None = None
    resume_optimizer: bool = True
    total_iterations: int = 2000
    checkpoint_interval: int = 5000
    evaluation_interval: int = 0
    param_stats_interval: int = 1
    rollout_batch_size: int = 512
    collect_batch_size: int = 2048
    replay_buffer_scale: int = 64
    mini_batches: int = 4
    lr_stable_iterations: int = 500
    reg_stable_iterations: int = 500
    discount_gamma: float = 1.0
    agent_merging_beta: float = 0.0
    value_loss_coef: float = 0.5
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    vrpo: VRPOConfig = field(default_factory=VRPOConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)

    def __post_init__(self) -> None:
        if self.schema != CORE_CONFIG_SCHEMA:
            raise ValueError(f"Unsupported core config schema '{self.schema}'.")
        if self.game is None:
            raise ValueError("TrainConfig.game is required; M01 has no implicit game backend.")
        if isinstance(self.game, Mapping):
            self.game = GameSpec(**dict(self.game))
        if not isinstance(self.game, GameSpec):
            self.game = GameSpec.from_game(self.game)
        if self.max_sampler_bytes_per_rank is not None and int(self.max_sampler_bytes_per_rank) <= 0:
            raise ValueError("max_sampler_bytes_per_rank must be positive when configured.")
        if int(self.environment_rank_seed_stride) <= 0:
            raise ValueError("environment_rank_seed_stride must be positive.")
        if not isinstance(self.amp_enabled, bool) or not isinstance(self.checkpoint_replay_buffer, bool):
            raise TypeError("amp_enabled and checkpoint_replay_buffer must be bool.")
        if self.training_data_domain != "training":
            raise ValueError("training_data_domain must be 'training'")
        if not str(self.training_data_generator_version).strip():
            raise ValueError("training_data_generator_version must be non-empty")
        if self.training_data_manifest_hash is None:
            source = {
                "schema": "guandan-marvel-training-source-manifest-v1",
                "domain": self.training_data_domain,
                "generator_version": self.training_data_generator_version,
                "seed_semantics": "EnvBatch initial seed plus slot, then continuous per-slot PRNG reset stream",
                "initial_seed": int(self.seed),
                "environment_rank_seed_stride": int(self.environment_rank_seed_stride),
                "rollout_batch_size": int(self.rollout_batch_size),
            }
            self.training_data_manifest_hash = hashlib.sha256(
                json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        else:
            value = str(self.training_data_manifest_hash).lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError("training_data_manifest_hash must be a 64-character hexadecimal digest")
            self.training_data_manifest_hash = value
        if not self.run_name:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.run_name = f"{self.game_identifier}_{timestamp}"
        if not self.logging_dir:
            repo_root = Path(__file__).resolve().parents[1]
            self.logging_dir = str(repo_root / "runs" / self.run_name)

    @property
    def game_identifier(self) -> str:
        return self.game.identifier

    @property
    def game_name(self) -> str:
        """Compatibility alias; no module resolution is performed."""
        return self.game.identifier

    # Compatibility properties used by the migrated runtime implementation.
    @property
    def use_compile(self) -> bool:
        return self.compile.enabled

    @property
    def compile_backend(self) -> str:
        return self.compile.backend

    @property
    def compile_dynamic_shapes(self) -> bool:
        return self.compile.dynamic_shapes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_key(key: object) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _lookup(data: Mapping[str, Any], field_name: str, aliases: tuple[str, ...] = ()) -> Any:
    names = {_canonical_key(field_name), *(_canonical_key(alias) for alias in aliases)}
    for raw_key, value in data.items():
        if _canonical_key(raw_key) in names:
            return value
    return _MISSING


def _decode_dataclass(cls: type[Any], payload: Any, *, aliases: Mapping[str, tuple[str, ...]] | None = None):
    if isinstance(payload, cls):
        return payload
    if payload is None or payload is _MISSING:
        return cls()
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(payload).__name__}.")
    known = {}
    aliases = aliases or {}
    for descriptor in fields(cls):
        value = _lookup(payload, descriptor.name, aliases.get(descriptor.name, ()))
        if value is not _MISSING:
            known[descriptor.name] = value
    return cls(**known)


def deserialize_train_config(payload: Any) -> TrainConfig | None:
    """Deserialize current or source-compatible checkpoint configuration metadata."""
    if payload is None:
        return None
    if isinstance(payload, TrainConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported train_config payload type: {type(payload).__name__}.")

    game_payload = _lookup(payload, "game")
    if game_payload is _MISSING:
        game_payload = _lookup(payload, "game_spec")
    if game_payload is _MISSING:
        raise KeyError("Checkpoint train_config is missing required 'game'/'game_spec' metadata.")
    if isinstance(game_payload, GameSpec):
        game = game_payload
    elif isinstance(game_payload, Mapping):
        game = GameSpec(**dict(game_payload))
    else:
        raise TypeError("Serialized game metadata must be a GameSpec or mapping.")

    nested = {
        "optimizer": (OptimizerConfig, {}),
        "ppo": (PPOConfig, {}),
        "vrpo": (VRPOConfig, {}),
        "schedule": (ScheduleConfig, {}),
        "model": (ModelConfig, _MODEL_ALIASES),
        "compile": (CompileConfig, {}),
        "ddp": (DDPConfig, {}),
    }
    kwargs: dict[str, Any] = {"game": game}
    train_fields = {item.name for item in fields(TrainConfig)}
    for name in train_fields - set(nested) - {"game"}:
        value = _lookup(payload, name)
        if value is not _MISSING:
            kwargs[name] = value
    for name, (cls, aliases) in nested.items():
        value = _lookup(payload, name)
        kwargs[name] = _decode_dataclass(cls, value, aliases=aliases)
    return TrainConfig(**kwargs)


__all__ = [
    "CORE_CONFIG_SCHEMA", "CompileConfig", "DDPConfig", "GameSpec", "ModelConfig",
    "OptimizerConfig", "PPOConfig", "ScheduleConfig", "TrainConfig", "VRPOConfig",
    "deserialize_train_config",
]
