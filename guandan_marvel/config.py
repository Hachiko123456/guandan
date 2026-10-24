# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/config.py.
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

from guandan_marvel.context_schedule import (
    validate_context_schedule, validate_context_curriculum,
)
from guandan_marvel.dmc import (
    DMC_ACTION_FEATURE_VERSION,
    DMC_CHECKPOINT_SCHEMA,
    DMC_TRAINING_SEMANTICS_PROFILE,
    DMCRuntimeConfig,
)
from guandan_marvel.phase6.config_types import OpponentPoolConfig
from guandan_marvel.reward import REWARD_SEMANTICS, TRAINING_REWARD_SCALE

CORE_CONFIG_SCHEMA = "guandan-marvel-core-config-v1"
# MARVEL v1.4 actor-residual-fuse architecture.  The fuse branch is a genuine
# architectural addition (new ``actor_fuse.*`` parameters), so it gets its own
# explicit version string instead of being smuggled into ``v1.3``.
MACRO_FUSE_MODEL_ARCHITECTURE = "v1.4-macro-fuse"
SUPPORTED_MODEL_ARCHITECTURES = frozenset({
    "v1", "v1.1", "v1.2", "v1.3", MACRO_FUSE_MODEL_ARCHITECTURE,
})

# MARVEL v1.4 phase-2 training semantics.  The profile is a versioned enum
# (never a combination of ad-hoc booleans) and the regularizer mode is an
# independent, explicitly opt-in ablation switch.
LEGACY_TRAINING_SEMANTICS_PROFILE = "legacy_token_v1"
SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE = "macro_sequence_ablation_v1"
MACRO_DECISION_TRAINING_SEMANTICS_PROFILE = "macro_decision_v1"
TRAINING_SEMANTICS_PROFILES = (
    LEGACY_TRAINING_SEMANTICS_PROFILE,
    SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE,
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
    # DMC is a candidate-level Q-learning route, not a policy-gradient profile.
    # It is registered explicitly so ``algo_name="dmc"`` can never silently
    # borrow PPO/macro semantics.
    DMC_TRAINING_SEMANTICS_PROFILE,
)
LEGACY_POLICY_REGULARIZER_MODE = "legacy_token_kl_uniform_v1"
MACRO_NORMALIZED_POLICY_REGULARIZER_MODE = "macro_normalized_kl_uniform_v1"
POLICY_REGULARIZER_MODES = (
    LEGACY_POLICY_REGULARIZER_MODE,
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
)
# Fixed for phase 2: macro semantics never re-clock the environment discount.
ENVIRONMENT_DISCOUNT_CLOCK = "base_token_v1"
MACRO_TRAINING_SEMANTICS_PROFILES = frozenset({
    SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE,
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
})

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
    training_semantics_profile: str = LEGACY_TRAINING_SEMANTICS_PROFILE
    policy_regularizer_mode: str = LEGACY_POLICY_REGULARIZER_MODE
    macro_gae_lambda: float | None = None
    # Optional macro-sample approximate-KL early stop.  ``None`` (the default)
    # is bitwise inert: it is dropped from the serialized resolved config, so
    # every pre-existing checkpoint stays an exact resume.  A positive value
    # stops the remaining updates of the current iteration when the *global*
    # (cross-rank sum / cross-rank count) macro sample KL exceeds it.
    target_kl: float | None = None

    def __post_init__(self) -> None:
        validate_training_semantics(self, algorithm=None)


def validate_training_semantics(
    ppo: "PPOConfig", *, algorithm: str | None, opponent_pool_enabled: bool = False,
) -> None:
    """Strictly validate the phase-2/DMC training-semantics matrix.

    The matrix is deliberately explicit so that no ad-hoc boolean combination
    can silently select an undefined estimator:

    ``(legacy_token_v1, legacy_token_kl_uniform_v1)``
        byte-compatible legacy token PPO/GAE/value/regularizer.
    ``(macro_sequence_ablation_v1 | macro_decision_v1, *)``
        full macro-sequence PPO with either the legacy token regularizer or the
        independent per-macro normalized uniform-KL regularizer.
    ``(legacy_token_v1, macro_normalized_kl_uniform_v1)``
        undefined: token PPO with a macro regularizer is rejected.
    ``(dmc_v1, legacy_token_kl_uniform_v1)``
        DMC candidate-level Q learning (``algo_name="dmc"``).  DMC does not
        read ``PPOConfig`` at all: there is no policy-gradient estimator,
        regularizer or macro-KL early stop, so the profile is only accepted
        with the structurally inert default regularizer and
        ``target_kl=None``, and it is rejected for every other algorithm.
        ``opponent_pool`` is likewise undefined for DMC.

    VRPO keeps its legacy Q-critic semantics; requesting any macro profile for
    ``algo_name=vrpo`` must fail fast instead of silently degrading to token or
    grafting IPPO macro semantics onto the Q-critic.
    """
    profile = ppo.training_semantics_profile
    if profile not in TRAINING_SEMANTICS_PROFILES:
        raise ValueError(
            f"unsupported training_semantics_profile {profile!r}; expected one of "
            f"{', '.join(TRAINING_SEMANTICS_PROFILES)}"
        )
    regularizer = ppo.policy_regularizer_mode
    if regularizer not in POLICY_REGULARIZER_MODES:
        raise ValueError(
            f"unsupported policy_regularizer_mode {regularizer!r}; expected one of "
            f"{', '.join(POLICY_REGULARIZER_MODES)}"
        )
    if profile == DMC_TRAINING_SEMANTICS_PROFILE:
        # DMC never reads PPOConfig.  ``legacy_token_kl_uniform_v1`` is only the
        # inert placeholder every PPOConfig carries by default; any other mode
        # means the caller expects PPO regularizer semantics DMC does not have.
        if regularizer != LEGACY_POLICY_REGULARIZER_MODE:
            raise ValueError(
                "training_semantics_profile='dmc_v1' does not use policy_regularizer_mode; "
                f"expected the inert default {LEGACY_POLICY_REGULARIZER_MODE!r}, "
                f"got {regularizer!r}"
            )
        if getattr(ppo, "target_kl", None) is not None:
            raise ValueError(
                "ppo.target_kl is a macro PPO-KL stop rule and is undefined for "
                "training_semantics_profile='dmc_v1'"
            )
        if opponent_pool_enabled:
            raise ValueError(
                "opponent_pool requires algo_name=ippo; DMC has no owner-masked "
                "current-sample unit"
            )
    if profile == LEGACY_TRAINING_SEMANTICS_PROFILE and regularizer != LEGACY_POLICY_REGULARIZER_MODE:
        raise ValueError(
            "legacy_token_v1 only supports the legacy_token_kl_uniform_v1 regularizer; "
            f"got {regularizer!r}"
        )
    macro_lambda = ppo.macro_gae_lambda
    if macro_lambda is not None:
        value = float(macro_lambda)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("macro_gae_lambda must be None or lie within [0, 1]")
    target_kl = getattr(ppo, "target_kl", None)
    if target_kl is not None:
        value = float(target_kl)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("ppo.target_kl must be None or a finite positive number")
        # The stop rule is defined on the *macro* sample KL sum/count.  A token
        # or legacy path has no macro sample unit, so requesting it there is an
        # undefined combination instead of a silently ignored option.
        if profile not in MACRO_TRAINING_SEMANTICS_PROFILES:
            raise ValueError(
                "ppo.target_kl requires a macro training_semantics_profile "
                f"({sorted(MACRO_TRAINING_SEMANTICS_PROFILES)}); got {profile!r}"
            )
    if algorithm == "dmc" and profile != DMC_TRAINING_SEMANTICS_PROFILE:
        raise ValueError(
            "algo_name=dmc requires training_semantics_profile="
            f"{DMC_TRAINING_SEMANTICS_PROFILE!r}; got {profile!r}"
        )
    if algorithm is not None and algorithm != "dmc" and profile == DMC_TRAINING_SEMANTICS_PROFILE:
        raise ValueError(
            "training_semantics_profile='dmc_v1' is only defined for algo_name='dmc'; "
            f"got algo_name={algorithm!r}"
        )
    if algorithm == "vrpo" and profile != LEGACY_TRAINING_SEMANTICS_PROFILE:
        raise ValueError(
            "algo_name=vrpo keeps legacy token semantics; macro training profiles are "
            f"supported only for algo_name=ippo (requested {profile!r})"
        )
    if algorithm == "vrpo" and target_kl is not None:
        raise ValueError("ppo.target_kl is a macro-sequence option and is undefined for algo_name=vrpo")


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
class SymmetryConfig:
    """Phase-5 conditional symmetry controls.  Defaults are OFF.

    ``symmetry_weight == 0`` must not read any file, add a forward pass, a
    parameter, an RNG draw or any latency: the phase-4 code path is taken
    verbatim.  A positive ``symmetry_weight`` enables the aligned
    ``pi(P_T(a)|T(s))`` vs ``pi(a|s)`` consistency term and therefore requires

    * ``diagnostic_artifact``: a sealed, SIGNIFICANT phase-5 diagnostic whose
      bytes, design hash and pre-registered thresholds are verified;
    * ``compatibility_checkpoint``: the exact initial policy checkpoint the
      diagnostic was measured on (SHA256 must match the seal).

    ``augmentation`` (feeding transformed trajectories into PPO) is *not*
    implemented: its behaviour distribution and importance ratio would have to
    be proved independently, so any non-False value is rejected before a
    trainer is constructed instead of being silently ignored.
    """

    symmetry_weight: float = 0.0
    augmentation: bool = False
    diagnostic_artifact: str | None = None
    compatibility_checkpoint: str | None = None

    def __post_init__(self) -> None:
        weight = float(self.symmetry_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("symmetry.symmetry_weight must be finite and non-negative")
        object.__setattr__(self, "symmetry_weight", weight)
        if not isinstance(self.augmentation, bool):
            raise TypeError("symmetry.augmentation must be bool")
        if self.augmentation:
            raise ValueError(
                "symmetry.augmentation is not implemented: an augmented trajectory "
                "entering PPO would need an independently proved behaviour "
                "distribution and importance ratio; only the extra consistency "
                "loss branch is supported"
            )
        for name in ("diagnostic_artifact", "compatibility_checkpoint"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"symmetry.{name} must be a path string or None")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"symmetry.{name} must be a non-empty path when provided")
        if weight > 0.0:
            # Fail closed against the recorded artifact, never against a
            # free-standing flag.  The default (weight == 0) reads no file.
            from guandan_marvel.phase5.symmetry_training import validate_config_artifact

            validate_config_artifact(
                artifact_path=self.diagnostic_artifact,
                checkpoint_path=self.compatibility_checkpoint,
            )


@dataclass
class BeliefConfig:
    """Phase-5 conditional belief-head controls.  Both defaults are OFF.

    ``belief_weight == 0`` must not construct the head at all, so the inference
    path has no head, no extra parameter and no extra forward/latency.
    ``build_head`` may only be True after the B0 information gate PASSED; the
    trainer enforces this against the recorded gate artifact, not against a
    free-standing flag.
    """

    belief_weight: float = 0.0
    build_head: bool = False
    gate_artifact: str | None = None

    def __post_init__(self) -> None:
        weight = float(self.belief_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("belief.belief_weight must be finite and non-negative")
        object.__setattr__(self, "belief_weight", weight)
        if not isinstance(self.build_head, bool):
            raise TypeError("belief.build_head must be bool")
        if self.build_head != (weight > 0.0):
            raise ValueError(
                "belief.build_head and belief.belief_weight must be enabled together "
                "(a head without a loss or a loss without a head is undefined)"
            )
        if self.build_head and not (self.gate_artifact and str(self.gate_artifact).strip()):
            raise ValueError(
                "belief.build_head requires belief.gate_artifact pointing at a PASSED B0 gate"
            )
        if self.gate_artifact is not None and not str(self.gate_artifact).strip():
            raise ValueError("belief.gate_artifact must be a non-empty path when provided")
        if self.build_head:
            # Fail closed against the recorded B0 artifact, never against a
            # free-standing flag.  The default (head off) does not touch phase-5
            # code or read any file.
            from guandan_marvel.phase5.gate import assert_belief_head_allowed

            assert_belief_head_allowed(self)


@dataclass
class DDPConfig:
    enabled: bool = False
    backend: str = "nccl"
    require_cuda: bool = True
    find_unused_parameters: bool = False


def validate_opponent_pool_combinations(
    *,
    enabled: bool,
    training_semantics_profile: str,
    policy_regularizer_mode: str,
    symmetry_weight: float,
    ntp_weight: float,
) -> None:
    """Phase-6 pool combinations that would mix owner masks into other losses.

    Profile C (macro decision) is the pre-registered value sample unit.  The
    sequence-ablation token value loss, the symmetry consistency branch and the
    NTP auxiliary head have no independently proved owner mask, so enabling the
    pool with any of them fails closed instead of silently training on opponent
    tokens.
    """
    if not enabled:
        return
    if training_semantics_profile != MACRO_DECISION_TRAINING_SEMANTICS_PROFILE:
        raise ValueError(
            "opponent_pool requires training_semantics_profile="
            f"{MACRO_DECISION_TRAINING_SEMANTICS_PROFILE!r}; other profiles have no "
            "owner-masked value/loss sample unit"
        )
    if policy_regularizer_mode != MACRO_NORMALIZED_POLICY_REGULARIZER_MODE:
        raise ValueError(
            "opponent_pool requires policy_regularizer_mode="
            f"{MACRO_NORMALIZED_POLICY_REGULARIZER_MODE!r}; the legacy token KL is "
            "defined over opponent behaviour and cannot be a current-sample regularizer"
        )
    if float(symmetry_weight) > 0.0:
        raise ValueError(
            "opponent_pool cannot be combined with symmetry.symmetry_weight>0 until the "
            "symmetry active/candidate rows are owner-masked and independently proved"
        )
    if float(ntp_weight) > 0.0:
        raise ValueError(
            "opponent_pool cannot be combined with ntp_weight>0; the NTP auxiliary head "
            "has no owner-masked current-sample contract"
        )


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
    training_context_schedule: dict[str, Any] | None = None
    training_context_curriculum: dict[str, Any] | None = None
    reward_semantics: str = REWARD_SEMANTICS
    reward_scale: float = TRAINING_REWARD_SCALE
    deterministic: bool = False
    deterministic_disable_tf32: bool = False
    amp_enabled: bool = False
    # Explicit, checkpoint-bound performance layout: remove only completed-trajectory tail padding.
    trim_completed_padding: bool = False
    # Optional local AMP adaptation.  ``None`` keeps PyTorch's production
    # default initial scale (65536); a small local GPU may explicitly choose a
    # lower scale (for example 1024) to avoid FP16 overflow in a smoke test.
    # It is recorded in the config and the semantic contract when set.
    amp_init_scale: float | None = None
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
    ntp_weight: float = 0.0
    ntp_pos_weight: float = 10.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    vrpo: VRPOConfig = field(default_factory=VRPOConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)
    # Phase-5 conditional features.  Both blocks default to fully disabled and
    # are recorded in the resolved config so a checkpoint can never silently
    # change feature semantics.
    symmetry: SymmetryConfig = field(default_factory=SymmetryConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    # Phase-6 conditional opponent pool.  Default off; when off it is omitted
    # from the serialized resolved config so a phase-5 checkpoint remains an
    # exact resume of the unchanged default training semantics.
    opponent_pool: OpponentPoolConfig = field(default_factory=OpponentPoolConfig)
    # DMC runtime block (stage 1: configuration/validation only; no DMC trainer
    # exists yet).  It is used only by ``algo_name="dmc"``; for every other
    # algorithm the default block is omitted from the serialized config so
    # pre-existing unsaved/resolved configs stay bitwise identical.
    dmc: DMCRuntimeConfig = field(default_factory=DMCRuntimeConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.trim_completed_padding, bool):
            raise TypeError("trim_completed_padding must be bool")
        if self.schema != CORE_CONFIG_SCHEMA:
            raise ValueError(f"Unsupported core config schema '{self.schema}'.")
        if self.game is None:
            raise ValueError("TrainConfig.game is required; M01 has no implicit game backend.")
        if isinstance(self.game, Mapping):
            self.game = GameSpec(**dict(self.game))
        if not isinstance(self.game, GameSpec):
            self.game = GameSpec.from_game(self.game)
        validate_training_semantics(
            self.ppo,
            algorithm=str(self.algo_name),
            opponent_pool_enabled=bool(getattr(self.opponent_pool, "enabled", False)),
        )
        if not isinstance(self.opponent_pool, OpponentPoolConfig):
            raise TypeError("TrainConfig.opponent_pool must be an OpponentPoolConfig")
        if not isinstance(self.dmc, DMCRuntimeConfig):
            raise TypeError("TrainConfig.dmc must be a DMCRuntimeConfig")
        if self.opponent_pool.enabled:
            if str(self.algo_name) != "ippo":
                raise ValueError(
                    "opponent_pool requires algo_name=ippo; VRPO has no macro ownership contract"
                )
            validate_opponent_pool_combinations(
                enabled=True,
                training_semantics_profile=self.ppo.training_semantics_profile,
                policy_regularizer_mode=self.ppo.policy_regularizer_mode,
                symmetry_weight=float(self.symmetry.symmetry_weight),
                ntp_weight=float(self.ntp_weight),
            )
        if self.max_sampler_bytes_per_rank is not None and int(self.max_sampler_bytes_per_rank) <= 0:
            raise ValueError("max_sampler_bytes_per_rank must be positive when configured.")
        if int(self.environment_rank_seed_stride) <= 0:
            raise ValueError("environment_rank_seed_stride must be positive.")
        if not isinstance(self.amp_enabled, bool) or not isinstance(self.checkpoint_replay_buffer, bool):
            raise TypeError("amp_enabled and checkpoint_replay_buffer must be bool.")
        if self.amp_init_scale is not None:
            scale = float(self.amp_init_scale)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("amp_init_scale must be a finite positive number or None")
            self.amp_init_scale = scale
        if getattr(self.ppo, "target_kl", None) is not None:
            # Normalised so the serialized resolved config/hash is deterministic
            # for any caller that passes an int or a numeric string.
            self.ppo.target_kl = float(self.ppo.target_kl)
        for name in ("ntp_weight", "ntp_pos_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"TrainConfig.{name} must be finite and non-negative.")
        if self.training_data_domain != "training":
            raise ValueError("training_data_domain must be 'training'")
        if str(self.reward_semantics) != REWARD_SEMANTICS:
            raise ValueError(
                f"unsupported reward_semantics {self.reward_semantics!r}; "
                f"the phase-4 environment emits {REWARD_SEMANTICS!r} raw rank points"
            )
        if float(self.reward_scale) != TRAINING_REWARD_SCALE:
            raise ValueError(
                "reward_scale must be exactly 1/3; the training reward is "
                "defined as R_train = R_raw / 3"
            )
        if not str(self.training_data_generator_version).strip():
            raise ValueError("training_data_generator_version must be non-empty")
        if self.training_context_schedule is not None:
            if not isinstance(self.training_context_schedule, Mapping):
                raise TypeError("training_context_schedule must be a mapping or None")
            self.training_context_schedule = validate_context_schedule(
                dict(self.training_context_schedule),
            )
        if self.training_context_curriculum is not None:
            if not isinstance(self.training_context_curriculum, Mapping):
                raise TypeError("training_context_curriculum must be a mapping or None")
            self.training_context_curriculum = validate_context_curriculum(
                dict(self.training_context_curriculum),
            )
            if self.training_context_schedule is None:
                raise ValueError(
                    "training_context_curriculum requires a target training_context_schedule"
                )
        if self.training_data_manifest_hash is None:
            source = {
                "schema": "guandan-marvel-training-source-manifest-v1",
                "domain": self.training_data_domain,
                "generator_version": self.training_data_generator_version,
                "seed_semantics": "EnvBatch initial seed plus slot, then continuous per-slot PRNG reset stream",
                "initial_seed": int(self.seed),
                "environment_rank_seed_stride": int(self.environment_rank_seed_stride),
                "rollout_batch_size": int(self.rollout_batch_size),
                "reward_semantics": self.reward_semantics,
                "reward_scale": self.reward_scale,
            }
            if self.training_context_schedule is not None:
                source["training_context_schedule"] = self.training_context_schedule
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
        payload = asdict(self)
        if not payload.get("trim_completed_padding", False):
            payload.pop("trim_completed_padding", None)
        pool = payload.get("opponent_pool") or {}
        if not pool.get("enabled", False):
            # The default pool is inactive and must not perturb the serialized
            # semantic identity (phase-5 checkpoints stay exact resumes).
            payload.pop("opponent_pool", None)
        if payload.get("amp_init_scale") is None:
            payload.pop("amp_init_scale", None)
        if str(self.algo_name) != "dmc" and payload.get("dmc") == asdict(DMCRuntimeConfig()):
            # The DMC block is inert for ippo/vrpo: dropping the default keeps
            # every pre-existing resolved config/hash unchanged.
            payload.pop("dmc", None)
        ppo_payload = payload.get("ppo")
        if isinstance(ppo_payload, dict) and ppo_payload.get("target_kl") is None:
            # ``target_kl=None`` is the production default: it must not change
            # any pre-existing resolved config/hash or checkpoint identity.
            ppo_payload.pop("target_kl", None)
        return payload


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
        "symmetry": (SymmetryConfig, {}),
        "belief": (BeliefConfig, {}),
        "opponent_pool": (OpponentPoolConfig, {}),
        "dmc": (DMCRuntimeConfig, {}),
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
    "CORE_CONFIG_SCHEMA", "BeliefConfig", "CompileConfig", "DDPConfig",
    "DMC_ACTION_FEATURE_VERSION", "DMC_CHECKPOINT_SCHEMA",
    "DMC_TRAINING_SEMANTICS_PROFILE", "DMCRuntimeConfig",
    "ENVIRONMENT_DISCOUNT_CLOCK", "GameSpec", "LEGACY_POLICY_REGULARIZER_MODE",
    "LEGACY_TRAINING_SEMANTICS_PROFILE", "MACRO_DECISION_TRAINING_SEMANTICS_PROFILE",
    "MACRO_NORMALIZED_POLICY_REGULARIZER_MODE", "MACRO_TRAINING_SEMANTICS_PROFILES",
    "ModelConfig", "OpponentPoolConfig", "OptimizerConfig", "POLICY_REGULARIZER_MODES", "PPOConfig",
    "SEQUENCE_ABLATION_TRAINING_SEMANTICS_PROFILE", "ScheduleConfig", "SymmetryConfig",
    "TRAINING_SEMANTICS_PROFILES", "TrainConfig", "VRPOConfig",
    "deserialize_train_config", "validate_opponent_pool_combinations",
    "validate_training_semantics",
]
