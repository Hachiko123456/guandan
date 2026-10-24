# SPDX-License-Identifier: MIT
"""Small, fail-closed local IPPO/VRPO trainers for M03 semantic smoke."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import torch
import torch.nn as nn

from guandan_marvel.buffer import (
    BUFFER_STATE_SCHEMA, BufferConfig, TrajectoryBuffer,
    require_trajectory_memory_budget, trajectory_memory_budget,
)
from guandan_marvel.config import (
    LEGACY_POLICY_REGULARIZER_MODE, LEGACY_TRAINING_SEMANTICS_PROFILE,
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE, MACRO_FUSE_MODEL_ARCHITECTURE,
    MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
    MACRO_TRAINING_SEMANTICS_PROFILES, TrainConfig, validate_training_semantics,
)
from guandan_marvel.model import Agent, CentralCritic
from guandan_marvel.optim.common import (
    ExponentialMovingAverageUpdater, MultiOptimizer, apply_learning_rate,
)
from guandan_marvel.runtime.compilation import ModuleCompiler
from guandan_marvel.runtime.distributed import DistributedRuntime
from guandan_marvel.runtime.logger import MetricSinks
from guandan_marvel.runtime.sampler import (
    SAMPLER_CONTRACT_SCHEMA, SAMPLER_STATE_SCHEMA, Sampler, SamplerConfig,
)
from guandan_marvel.training.checkpoints import (
    TRAINING_CHECKPOINT_SCHEMA, CheckpointCompatibilityError, TrainingCheckpointStore,
    build_backend_identity, build_execution_fingerprint, build_semantic_configuration,
    build_semantic_contract, canonical_json_hash, capture_rng_state, classify_resume,
    compute_source_fingerprint, restore_rng_state, serialize_config,
)
from guandan_marvel.training.gae import compute_gae_from_qvalue, compute_gae_from_value
from guandan_marvel.training.losses import (
    compute_clipped_pg_loss, compute_next_observation_token_loss, compute_qvalue_loss,
    compute_value_loss,
)
from guandan_marvel.training.macro_actions import macro_diagnostics, segment_rollout_macros
from guandan_marvel.training.macro_gae import compute_macro_decision_gae
from guandan_marvel.training.macro_ppo import (
    build_macro_batch_view, compute_macro_normalized_kl_mean, compute_macro_pg_loss,
    compute_macro_value_loss, explain_macro_value, macro_sample_kl_control,
    macro_trainable_mask, macro_regularizer_diagnostics, model_connected_zero,
    parameter_group_gradient_norms,
)
from guandan_marvel.training.metrics import rollout_metrics
from guandan_marvel.training.schedules import compute_agent_schedule
from guandan_marvel.training.validation import (
    ensure_finite_gradients, ensure_finite_tensor, validate_complete_trajectories,
)
from guandan_marvel.utils import set_global_seed

M03_RUNTIME_SCHEMA = "guandan-marvel-m03-local-runtime-v2"
OPPONENT_POOL_CHECKPOINT_SCHEMA = "guandan-marvel-phase6-opponent-pool-checkpoint-v1"
VRPO_Q_TARGET_CONTRACT = "guandan-marvel-vrpo-q-lambda-td1-v1"
# Explicit, closed set of policy-warm-start architecture transitions.  Anything
# not listed here (including the reverse v1.4 -> v1.3 direction) is rejected.
WARM_START_ALLOWED_ARCHITECTURE_TRANSITIONS = frozenset({
    ("v1.3", MACRO_FUSE_MODEL_ARCHITECTURE),
})
COMMON_RUNTIME_SOURCE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "guandan_rules": (
        "guandan/cards.py", "guandan/combos.py", "guandan/round.py",
        "guandan/state.py", "guandan/action_state.py", "guandan/encoding.py",
        "guandan/environment.py", "guandan/environment_types.py", "guandan/env_batch.py",
    ),
    "game_adapter_common": (
        "guandan_marvel/__init__.py", "guandan_marvel/backends.py",
        "guandan_marvel/env_batch.py", "guandan_marvel/buffer.py",
    ),
    "model": (
        "guandan_marvel/model/__init__.py", "guandan_marvel/model/agent.py",
        "guandan_marvel/model/base.py", "guandan_marvel/model/cache.py",
        "guandan_marvel/model/checkpointable.py", "guandan_marvel/model/critic.py",
        "guandan_marvel/model/tensordict.py", "guandan_marvel/model/trainable.py",
        "guandan_marvel/model/transformer.py", "guandan_marvel/model/types.py",
    ),
    "optimization": (
        "guandan_marvel/optim/__init__.py", "guandan_marvel/optim/common.py",
        "guandan_marvel/optim/muon.py",
    ),
    "runtime": (
        "guandan_marvel/config.py", "guandan_marvel/runtime/__init__.py",
        "guandan_marvel/runtime/compilation.py", "guandan_marvel/runtime/distributed.py",
        "guandan_marvel/runtime/logger.py", "guandan_marvel/runtime/sampler.py",
        "guandan_marvel/runtime/trajectory_layout.py",
        "guandan_marvel/utils.py",
    ),
    "training": (
        "guandan_marvel/training/__init__.py", "guandan_marvel/training/checkpoints.py",
        "guandan_marvel/training/gae.py", "guandan_marvel/training/ippo.py",
        "guandan_marvel/training/local.py", "guandan_marvel/training/losses.py",
        "guandan_marvel/training/macro_actions.py", "guandan_marvel/training/macro_gae.py",
        "guandan_marvel/training/macro_ppo.py",
        "guandan_marvel/training/metrics.py", "guandan_marvel/training/schedules.py",
        "guandan_marvel/training/smoke.py", "guandan_marvel/training/validation.py",
        "guandan_marvel/training/vrpo.py",
    ),
    # Phase-6 conditional features are part of the training source identity: a
    # changed pool/value/search/gate implementation must never be an "exact
    # resume" of a checkpoint produced by different phase-6 code.
    "phase6_conditionals": (
        "guandan_marvel/phase6/__init__.py", "guandan_marvel/phase6/config_types.py",
        "guandan_marvel/phase6/design.py", "guandan_marvel/phase6/opponent_pool.py",
        "guandan_marvel/phase6/policy_owner.py", "guandan_marvel/phase6/search.py",
        "guandan_marvel/phase6/training_gates.py", "guandan_marvel/phase6/value_gate.py",
    ),
    "distributed_orchestration": (
        "guandan_marvel/kaggle/bootstrap.py", "guandan_marvel/kaggle/checkpoints.py",
        "guandan_marvel/kaggle/common.py", "guandan_marvel/kaggle/discovery.py",
        "guandan_marvel/kaggle/evaluation.py", "guandan_marvel/kaggle/profiles.py",
        "guandan_marvel/kaggle/progress.py", "guandan_marvel/kaggle/runner.py",
        "guandan_marvel/kaggle/worker.py", "guandan_marvel/kaggle/pilot_observability.py",
    ),
}
PYTHON_BACKEND_SOURCE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "python_backend": (
        "guandan_marvel/game.py", "guandan_marvel/python_game.py",
    ),
}
CPP_BACKEND_SOURCE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "cpp_backend_wrapper": (
        "guandan_marvel/cpp_game.py", "nanopokerlib/__init__.py",
        "nanopokerlib/guandan.py", "nanopokerlib/build_identity.py",
    ),
    "cpp_native_source": (
        "nanopokerlib/cpp/bindings.cpp", "nanopokerlib/cpp/core/types.hpp",
        "nanopokerlib/cpp/guandan/guandan.cpp", "nanopokerlib/cpp/guandan/guandan.hpp",
        "nanopokerlib/cpp/guandan_build_identity.hpp", "setup.py", "MANIFEST.in",
    ),
}
# Union retained for source-inventory tooling and tests. Trainers use only the
# component set for the backend that is actually executing.
RUNTIME_SOURCE_COMPONENTS: dict[str, tuple[str, ...]] = {
    **COMMON_RUNTIME_SOURCE_COMPONENTS,
    **PYTHON_BACKEND_SOURCE_COMPONENTS,
    **CPP_BACKEND_SOURCE_COMPONENTS,
}


def runtime_source_components(backend: str, root: Path) -> dict[str, tuple[str, ...]]:
    components = dict(COMMON_RUNTIME_SOURCE_COMPONENTS)
    if backend == "python_reference":
        components.update(PYTHON_BACKEND_SOURCE_COMPONENTS)
    elif backend == "cpp":
        components.update(CPP_BACKEND_SOURCE_COMPONENTS)
    else:
        raise CheckpointCompatibilityError(f"unsupported runtime backend {backend!r}")
    # A clean wheel may intentionally omit native source files. The loaded
    # extension still carries its embedded source digest and binary identity.
    return {
        name: tuple(path for path in paths if (root / path).is_file())
        for name, paths in components.items()
        if any((root / path).is_file() for path in paths)
    }




def _git_head(root: Path) -> str:
    exported = os.environ.get("GUANDAN_MARVEL_SOURCE_COMMIT", "").strip().lower()
    if len(exported) == 40 and all(ch in "0123456789abcdef" for ch in exported):
        return exported
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unavailable"


def _optimizer_state(optimizer) -> Mapping[str, object]:
    return optimizer.state_dict()


def _load_optimizer_state(optimizer, state: Mapping[str, object]) -> None:
    optimizer.load_state_dict(state)


def _scaler_optimizers(optimizer) -> list:
    """The concrete torch optimizers a single GradScaler must step/unscale."""
    if isinstance(optimizer, MultiOptimizer):
        return list(optimizer.optimizers.values())
    return [optimizer]


def _finite_metrics(metrics: Mapping[str, object]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"metric {key!r} is non-finite")


def _per_sample_macro_metric(
    loss_value: float,
    *,
    global_count: int,
    local_count: int,
    world_size: int,
) -> float:
    """Convert a DDP-scaled local macro loss into this rank's per-sample mean.

    The differentiable local loss is ``L_k = (world_size / N) * sum_{i in k} l_i``.
    Logging is *not* allowed to keep that scale: count-weighted metric aggregation
    over per-sample means must reproduce the single-process concatenated-batch
    reference ``(1/N) * sum_i l_i``.  With ``world_size == 1`` the factor is exactly
    ``1.0`` (``global_count == local_count``), so single-process logs are bitwise
    unchanged.
    """
    if local_count <= 0:
        return 0.0
    return float(loss_value) * (float(global_count) / (float(world_size) * float(local_count)))


def _explained_variance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    degenerate_threshold: float = 1e-12,
) -> tuple[float | None, int]:
    """EV on the masked sample unit; degenerate target variance is undefined."""
    selected = mask.to(torch.bool)
    if int(selected.sum().item()) == 0:
        return None, 0
    y = target[selected].detach().to(torch.float64)
    y_hat = prediction[selected].detach().to(torch.float64)
    variance = float(y.var(unbiased=False).item())
    if not math.isfinite(variance) or variance < float(degenerate_threshold):
        return None, 1
    return 1.0 - float((y - y_hat).var(unbiased=False).item()) / variance, 0


class LocalTrainerBase:
    algorithm: str
    q_target_contract: str | None = None

    def __init__(self, cfg: TrainConfig, *, game: object, run_dir: str | Path) -> None:
        self.cfg = cfg
        self.training_semantics_profile = str(cfg.ppo.training_semantics_profile)
        self.policy_regularizer_mode = str(cfg.ppo.policy_regularizer_mode)
        self.macro_gae_lambda = cfg.ppo.macro_gae_lambda
        self.macro_semantics = self.training_semantics_profile in MACRO_TRAINING_SEMANTICS_PROFILES
        validate_training_semantics(cfg.ppo, algorithm=self.algorithm)
        self.game = game
        self.distributed_runtime = DistributedRuntime.from_config(cfg.ddp, device=cfg.device)
        self.distributed = self.distributed_runtime.context
        self.device = self.distributed.device
        self.rank = int(self.distributed.rank)
        self.local_rank = int(self.distributed.local_rank)
        self.world_size = int(self.distributed.world_size)
        self.is_primary = bool(self.distributed.is_primary)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("MARVEL runtime supports cpu/cuda devices only")
        # Mixed precision.  CUDA uses fp16 autocast driven by a real GradScaler;
        # CPU (this project's torch build has no CUDA) uses bfloat16 autocast,
        # whose wide exponent range needs no loss scaling, so the scaler stays
        # disabled there.  The scaler object exists in both cases so its state is
        # always checkpointed and restored.
        self.amp_enabled = bool(cfg.amp_enabled)
        self.amp_device_type = self.device.type
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        self.amp_scaler_enabled = bool(self.amp_enabled and self.device.type == "cuda")
        scaler_kwargs = {}
        if getattr(cfg, "amp_init_scale", None) is not None:
            scaler_kwargs["init_scale"] = float(cfg.amp_init_scale)
        self.grad_scaler = torch.amp.GradScaler(
            self.amp_device_type, enabled = self.amp_scaler_enabled, **scaler_kwargs,
        )
        self.amp_skipped_steps = 0
        if cfg.max_sampler_bytes_per_rank is None:
            raise ValueError("max_sampler_bytes_per_rank is required before sampler allocation")
        if cfg.game.identifier != getattr(game, "identifier"):
            raise ValueError("TrainConfig.game and runtime game identifier differ")
        if cfg.rollout_batch_size % self.world_size or cfg.collect_batch_size % self.world_size:
            raise ValueError("global rollout/collect batch sizes must be divisible by DDP world size")
        self.local_rollout_batch_size = cfg.rollout_batch_size // self.world_size
        self.local_collect_batch_size = cfg.collect_batch_size // self.world_size
        if self.local_rollout_batch_size <= 0 or self.local_collect_batch_size <= 0:
            raise ValueError("each DDP rank must receive positive rollout and collect batches")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / f"{self.algorithm}_metrics.jsonl"
        self.evaluation_results_path = self.run_dir / "evaluation_results.jsonl"
        self.metrics = MetricSinks(jsonl_path=self.metrics_path)
        self.memory_budget = require_trajectory_memory_budget(
            game,
            rollout_batch_size=cfg.rollout_batch_size,
            collect_batch_size=cfg.collect_batch_size,
            world_size=self.world_size,
            max_sampler_bytes_per_rank=cfg.max_sampler_bytes_per_rank,
        )
        self.replay_memory_budget: dict[str, object] | None = None
        # All ranks create identical model parameters.  Per-rank stochastic streams
        # are activated after DDP has synchronized the trainable modules.
        set_global_seed(cfg.seed)
        self.iteration = 0
        self.durable_updates = 0
        self.learner_updates = 0
        self.durable_learner_updates = 0
        self.completed_games_total = 0
        # Explicit logical curriculum origin for a fresh-initialization import.
        # ``0`` is the production default and reproduces the historical
        # ``set_curriculum_iteration(self.iteration)`` behaviour exactly.
        self.curriculum_offset = 0
        self.failed = False
        self.failure: dict[str, object] | None = None
        self.iteration_status = "idle"
        self.active_iteration: int | None = None
        self._iteration_start_learner_updates = 0
        self.last_metrics: dict[str, object] = {}
        self.last_trajectory_keys: tuple[str, ...] = ()
        self.last_trajectory_num_steps: tuple[int, ...] = ()
        self.last_policy_epochs = 0
        self.last_critic_batch_sources: tuple[str, ...] = ()
        # Phase-5 consistency branch.  ``None`` on the phase-4 path: zero extra
        # forward passes, parameters, file reads or RNG draws.
        self.symmetry = None
        self.evaluation_hook = None
        self.phase_callback = None
        self.evaluation_result_history: list[dict[str, object]] = []
        self.repo_root = Path(__file__).resolve().parents[2]
        self.backend_identity = build_backend_identity(game)
        self.backend_name = str(self.backend_identity["backend"])
        self.runtime_source_components = runtime_source_components(
            self.backend_name, self.repo_root,
        )
        self.source_fingerprint = compute_source_fingerprint(
            self.repo_root, self.runtime_source_components,
        )
        self.source_hash = str(self.source_fingerprint["hash"])
        self.execution_fingerprint = build_execution_fingerprint(self.device)
        self.source_parent_commit = _git_head(self.repo_root)
        self.resolved_config = serialize_config(cfg)
        self.resolved_config_hash = canonical_json_hash(self.resolved_config)
        self.semantic_configuration = build_semantic_configuration(
            cfg, algorithm=self.algorithm,
        )
        self.semantic_contract = build_semantic_contract(
            semantic_config=self.semantic_configuration,
            sampler_contract=SAMPLER_CONTRACT_SCHEMA,
            sampler_state_schema=SAMPLER_STATE_SCHEMA,
            buffer_schema=BUFFER_STATE_SCHEMA,
            q_target_contract=self.q_target_contract,
        )
        # Phase-6 opponent pool (default off).  Loading pool checkpoints must
        # not consume the global RNG stream that initializes the current model,
        # so the complete RNG state is captured and restored around loading.
        self.opponent_pool: OpponentPool | None = None
        self.policy_router: PoolPolicyRouter | None = None
        self.opponent_pool_manifest_path: Path | None = None
        self.last_pool_stats = None
        self._frozen_hashes: dict[str, str] | None = None
        pool_field_specs_value: dict[str, object] = {}
        if cfg.opponent_pool.enabled:
            from guandan_marvel.phase6.opponent_pool import (
                OpponentPool, PoolPolicyRouter, write_manifest,
            )
            from guandan_marvel.phase6.policy_owner import pool_field_specs

            rng_snapshot = capture_rng_state()
            try:
                self.opponent_pool = OpponentPool.from_config(
                    cfg.opponent_pool,
                    device=self.device,
                    expected_architecture=str(cfg.model.architecture),
                    expected_protocol_version=str(cfg.game.protocol_version),
                    expected_backend=self.backend_name,
                    expected_backend_identity_hash=str(self.backend_identity["hash"]),
                    model_feature_identity={
                        "history_dim": int(cfg.model.history_dim),
                        "feature_dim": int(cfg.model.feature_dim),
                        "decoder_layers": int(cfg.model.decoder_layers),
                        "decoder_heads": int(cfg.model.decoder_heads),
                        "state_layers": int(cfg.model.state_layers),
                        "action_layers": int(cfg.model.action_layers),
                        "fuse_layers": int(cfg.model.fuse_layers),
                    },
                    batch_slots=self.local_rollout_batch_size,
                )
            finally:
                restore_rng_state(rng_snapshot)
            self.opponent_pool_manifest_path = write_manifest(
                self.opponent_pool.manifest, self.run_dir / "opponent_pool_manifest.json",
            )
            self.policy_router = PoolPolicyRouter(
                pool=self.opponent_pool,
                batch_size=self.local_rollout_batch_size,
                num_players=int(cfg.game.num_players),
                max_ctx_len=int(cfg.game.max_game_horizon),
                device=self.device,
                max_resident_policies=int(cfg.opponent_pool.max_resident_policies),
            )
            pool_field_specs_value = pool_field_specs(
                max_game_horizon=int(cfg.game.max_game_horizon),
                num_players=int(cfg.game.num_players),
            )
        self.sampler = Sampler(
            config=SamplerConfig(
                game=game,
                device=self.device,
                rollout_batch_size=self.local_rollout_batch_size,
                collect_batch_size=self.local_collect_batch_size,
                sampling_temperature=1.0,
                trim_completed_padding=bool(cfg.trim_completed_padding),
                purpose="on_policy",
                env_seed=cfg.seed + self.rank * int(cfg.environment_rank_seed_stride),
                data_domain=cfg.training_data_domain,
                data_manifest_hash=cfg.training_data_manifest_hash,
                data_generator_version=cfg.training_data_generator_version,
                context_schedule=cfg.training_context_schedule,
                context_curriculum=cfg.training_context_curriculum,
                reward_scale=cfg.reward_scale,
                field_specs=pool_field_specs_value or None,
            ),
            policy_router=self.policy_router,
        )

    def load_warm_start_policy(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Load policy weights only; optimizer, sampler, counters, and schedule remain fresh.

        Auxiliary heads introduced by the current configuration (for example the
        NTP head) may be absent from the source checkpoint; they are then left
        at their fresh initialization.  All other parameters must match exactly.

        The only permitted *architecture* transition is the explicit
        ``v1.3 -> v1.4-macro-fuse`` policy-warm start: the new ``actor_fuse.*``
        branch is left at its fresh (zero-residual) initialization while every
        legacy backbone/state/action/logit/value key must still be present and no
        unexpected key of any kind is tolerated.
        """
        self._require_healthy()
        if getattr(self, "policy_router", None) is not None:
            raise ValueError(
                "policy-only warm start cannot reconstruct opponent-pool assignments; "
                "use an exact checkpoint resume with the same pool manifest"
            )
        if payload.get("schema") != "guandan-marvel-evaluation-checkpoint-v1":
            raise ValueError("warm start requires an evaluation-only checkpoint")
        if payload.get("algorithm") != self.algorithm:
            raise ValueError("warm-start algorithm mismatch")
        source_architecture = str(payload.get("architecture"))
        target_architecture = str(self.cfg.model.architecture)
        migrating_to_fuse = (
            source_architecture, target_architecture,
        ) in WARM_START_ALLOWED_ARCHITECTURE_TRANSITIONS
        if source_architecture != target_architecture and not migrating_to_fuse:
            raise ValueError(
                "warm-start architecture mismatch: "
                f"{source_architecture!r} -> {target_architecture!r} is not a permitted transition"
            )
        if payload.get("protocol_version") != self.cfg.game.protocol_version:
            raise ValueError("warm-start protocol mismatch")
        state = payload.get("agent_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("warm-start checkpoint lacks agent_state_dict")

        # The fuse branch may only be *absent* when the source is a pre-fuse
        # policy; a same-architecture warm start that drops ``actor_fuse.*`` is a
        # genuine incompatibility and must fail.
        allowed_missing_prefixes = ("ntp_head.",) + (
            ("actor_fuse.",) if migrating_to_fuse else ()
        )

        def _load_compatible(module, name: str) -> None:
            missing, unexpected = module.load_state_dict(state, strict=False)
            allowed_missing = {key for key in missing if key.startswith(allowed_missing_prefixes)}
            disallowed = sorted(set(missing) - allowed_missing)
            if disallowed or unexpected:
                raise ValueError(
                    f"warm-start state incompatible with {name}: "
                    f"missing={disallowed}, unexpected={sorted(unexpected)}"
                )

        _load_compatible(self.agent_resume, "agent_resume")
        _load_compatible(self.agent_merged, "agent_merged")
        if migrating_to_fuse:
            # The migrated residual gate must still be *exactly* zero: nothing in
            # this load is allowed to have disturbed the fresh fuse branch.
            for module in (self.agent_resume, self.agent_merged):
                fuse = getattr(module, "actor_fuse", None)
                if fuse is None:
                    raise ValueError("warm-start migration requires a fuse-enabled target")
                if not bool(torch.equal(fuse.output_proj.weight.detach(), torch.zeros_like(fuse.output_proj.weight))):
                    raise ValueError("migrated actor_fuse.output_proj is not bitwise zero")
        self.merge_updater.set_reference_iteration(0)
        return {
            "schema": "guandan-marvel-warm-start-load-v1",
            "source_iteration": int(payload.get("iteration", -1)),
            "source_hash": canonical_json_hash({key: payload.get(key) for key in ("algorithm", "architecture", "protocol_version", "iteration", "resolved_config_hash", "source_hash")}),
            "optimizer_reset": True,
            "sampler_reset": True,
            "counters_reset": True,
            "migration": {
                "schema": "guandan-marvel-policy-warm-start-migration-v1",
                "source_architecture": source_architecture,
                "target_architecture": target_architecture,
                "permitted_missing_prefixes": sorted(allowed_missing_prefixes),
                "fuse_migrated": bool(migrating_to_fuse),
                "output_proj_zero_verified": bool(migrating_to_fuse),
            },
        }

    @property
    def merged_policy_identity(self) -> str:
        return "current_policy_copy" if self.cfg.agent_merging_beta == 0.0 else "exponential_moving_average"

    def set_phase_callback(self, callback) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("phase callback must be callable")
        self.phase_callback = callback

    def _emit_phase(self, phase: str) -> None:
        callback = getattr(self, "phase_callback", None)
        if callback is not None:
            callback(str(phase))

    def set_evaluation_hook(self, hook: object | None) -> None:
        """Attach a process-isolated M06 evaluator hook at durable boundaries."""
        if hook is not None:
            for method in ("on_iteration_completed", "collect_ready", "close"):
                if not callable(getattr(hook, method, None)):
                    raise TypeError(f"evaluation hook must define {method}()")
        self.evaluation_hook = hook

    def _record_evaluation_results(self, results) -> tuple[dict[str, object], ...]:
        values = tuple(dict(item) for item in results)
        if not values:
            return values
        self.evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.evaluation_results_path.open("a", encoding="utf-8", newline="\n") as stream:
            for item in values:
                stream.write(json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.evaluation_result_history.extend(values)
        return values

    def _collect_evaluation_results(self) -> tuple[dict[str, object], ...]:
        if self.evaluation_hook is None:
            return tuple()
        return self._record_evaluation_results(self.evaluation_hook.collect_ready())

    def close_evaluation_hook(self, *, wait: bool = True) -> tuple[dict[str, object], ...]:
        if self.evaluation_hook is None:
            return tuple()
        return self._record_evaluation_results(self.evaluation_hook.close(wait=wait))

    def _compile(self, module: nn.Module) -> nn.Module:
        if not self.cfg.compile.enabled:
            return module
        return ModuleCompiler.from_cfg(self.cfg).compile(module)

    def _prepare_trainable(self, module: nn.Module) -> nn.Module:
        compiled = self._compile(module)
        return self.distributed.wrap_model(
            compiled, find_unused_parameters=self.cfg.ddp.find_unused_parameters,
        )

    def _activate_rank_rng(self) -> None:
        set_global_seed(self.cfg.seed, offset=self.rank * 1_000_003)

    def _build_symmetry(self, agent: nn.Module) -> None:
        """Attach the sealed phase-5 consistency branch (disabled by default).

        The artifact and initial checkpoint are re-verified inside
        :class:`SymmetryConsistency`; a mismatch fails trainer construction
        before any rollout or optimisation step.
        """
        if float(self.cfg.symmetry.symmetry_weight) <= 0.0:
            self.symmetry = None
            return
        from guandan_marvel.phase5.symmetry_training import SymmetryConsistency

        self.symmetry = SymmetryConsistency.from_config(
            self.cfg.symmetry,
            agent=agent,
            architecture=str(self.cfg.model.architecture),
            algorithm=self.algorithm,
            history_dim=int(self.cfg.model.history_dim),
            feature_dim=int(self.cfg.model.feature_dim),
            device=self.device,
        )

    def _symmetry_loss(self, mb, output):
        """Return the weighted consistency tensor, or ``None`` when disabled."""
        if self.symmetry is None:
            return None
        return self.symmetry.loss(
            mb,
            base_log_probs=output.avail_log_probs,
            forward=self._forward_agent,
            world_size=self.world_size,
        )

    def close(self) -> None:
        self.distributed_runtime.close()

    def _require_healthy(self) -> None:
        if self.failed or self.iteration_status == "failed":
            raise RuntimeError(f"trainer is unusable after failed iteration/load: {self.failure}")
        if self.sampler.failed:
            raise RuntimeError(f"trainer sampler is unusable: {self.sampler.failure_info}")

    def _assert_frozen_opponents_unchanged(self) -> None:
        """A frozen opponent must be bitwise identical before/after training."""
        if self.policy_router is None:
            return
        hashes = self.policy_router.frozen_parameter_hashes()
        if self._frozen_hashes is None:
            self._frozen_hashes = dict(hashes)
            return
        if hashes != self._frozen_hashes:
            raise RuntimeError(
                "a frozen opponent's parameters changed during training; the pool contract "
                "requires eval-mode, requires_grad=False and optimizer exclusion"
            )

    def _verify_frozen_excluded_from_optimizer(self, optimizer) -> None:
        if self.opponent_pool is None:
            return
        frozen_ids = {
            id(parameter)
            for entry in self.opponent_pool.entries
            for parameter in entry.artifact.model.parameters()
        }
        for target in _scaler_optimizers(optimizer):
            for group in target.param_groups:
                leaked = sorted(
                    str(index) for index, parameter in enumerate(group["params"])
                    if id(parameter) in frozen_ids
                )
                if leaked:
                    raise RuntimeError(
                        "a frozen opponent parameter is registered in the current optimizer"
                    )

    def _begin_iteration(self) -> None:
        self._require_healthy()
        if self.iteration_status in {"running", "loading"} or self.active_iteration is not None:
            raise RuntimeError("trainer already has an active operation")
        self.iteration_status = "running"
        self.active_iteration = self.iteration + 1
        self._iteration_start_learner_updates = self.learner_updates
        # The context curriculum is a pure function of the iteration index, so it
        # is pinned before the rollout is collected and is exactly reproduced on
        # resume (the sampler's own state carries the same iteration number).
        # ``curriculum_offset`` is 0 on every historical path; a fresh-import run
        # may set it so the curriculum continues from its logical origin instead
        # of silently replaying the early schedule.
        self.sampler.set_curriculum_iteration(self.iteration + int(self.curriculum_offset))

    def _begin_load(self) -> None:
        self._require_healthy()
        if self.iteration_status in {"running", "loading"} or self.active_iteration is not None:
            raise RuntimeError("cannot load while an iteration or load is active")
        self.iteration_status = "loading"

    def _poison(self, *, stage: str, error: BaseException) -> None:
        partial_updates = max(0, self.learner_updates - self._iteration_start_learner_updates)
        self.failed = True
        self.iteration_status = "failed"
        self.failure = {
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "active_iteration": self.active_iteration,
            "iteration": self.iteration,
            "durable_updates": self.durable_updates,
            "durable_learner_updates": self.durable_learner_updates,
            "learner_updates_observed": self.learner_updates,
            "partial_learner_updates": partial_updates,
            "exact_resume_allowed": False,
        }
        if not self.sampler.failed:
            self.sampler.invalidate(stage=f"trainer:{stage}", error=error)

    def _require_checkpointable(self) -> None:
        self._require_healthy()
        if self.iteration_status in {"running", "loading"} or self.active_iteration is not None:
            raise RuntimeError("checkpoint is forbidden while an iteration/load is active")
        if self.iteration != self.durable_updates:
            raise RuntimeError("checkpoint counters are not at a durable iteration boundary")
        if self.learner_updates != self.durable_learner_updates:
            raise RuntimeError("checkpoint contains partial learner updates outside a durable boundary")
        if self.durable_updates <= 0:
            raise RuntimeError("refusing to checkpoint before a durable learner update")

    def _collect(self, agent_model: nn.Module):
        self._emit_phase("rollout")
        started = time.perf_counter()
        rollout = self.sampler.collect(agent_model=agent_model)
        seconds = time.perf_counter() - started
        validation = validate_complete_trajectories(
            rollout, require_on_policy=self.policy_router is None,
        )
        if self.policy_router is not None:
            from guandan_marvel.phase6.policy_owner import validate_pooled_trajectory

            # Frozen-opponent steps are intentionally not on-policy current
            # samples; the pool contract is validated independently and fails
            # closed on any mixed owner/mask.
            self.last_pool_stats = validate_pooled_trajectory(
                rollout, num_players=int(self.cfg.game.num_players),
            )
        terminal_infos = self.sampler.pop_terminal_infos(validation.completed_games)
        self.last_trajectory_keys = tuple(sorted(rollout.keys()))
        self.last_trajectory_num_steps = tuple(int(value) for value in rollout["num_steps"].tolist())
        return rollout, validation, terminal_infos, seconds

    def _amp_autocast(self):
        """Autocast context deliberately scoped to *model forward passes only*.

        ``torch.autocast`` only redirects the eligible ops (Linear / matmul / the
        transformer and residual-MLP graph) to the reduced precision.  Running the
        forwards inside this context and every loss-side computation outside it
        keeps softmax, selected log-probability, macro log-ratio, ratio, KL,
        entropy, GAE, value targets and loss accumulation in FP32.

        The norms upcast explicitly and therefore emit FP32 activations:
        ``guandan_marvel.model.base.RMSNorm`` (used by every ``ResGatedMLP`` -
        ``state_mlp``, ``action_mlp``, ``critic_mixer`` and ``actor_fuse.trunk``)
        casts its input to FP32 and computes the mean of squares there, and
        ``TransformerRMSNorm`` does the same, so no normalisation statistic is
        ever evaluated in the reduced precision.
        """
        return torch.autocast(
            device_type = self.amp_device_type,
            dtype = self.amp_dtype,
            enabled = self.amp_enabled,
        )

    def _forward_agent(self, rollout):
        with self._amp_autocast():
            return self.agent_model(rollout = rollout)

    def _forward_critic(self, rollout):
        with self._amp_autocast():
            return self.critic_model(rollout = rollout)

    def _optimizer_step(
        self, *, loss: torch.Tensor, module: nn.Module, optimizer, name: str,
        group_norms_out: dict[str, float] | None = None,
    ) -> float:
        self._emit_phase("learner")
        ensure_finite_tensor(f"{name}_loss", loss)
        optimizer.zero_grad(set_to_none=True)
        scaler = self.grad_scaler
        targets = _scaler_optimizers(optimizer)
        if scaler.is_enabled():
            # Strict AMP ordering: scale(loss).backward() -> unscale_(optimizer)
            # -> gradient clip -> step/update.
            scaler.scale(loss).backward()
            for target in targets:
                scaler.unscale_(target)
            # A genuine fp16 overflow is a *skippable* step, not a fatal error:
            # the update is dropped, counted and the scaler shrinks the scale.
            non_finite = any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in module.parameters()
            )
            if non_finite:
                self.amp_skipped_steps += 1
                for target in targets:
                    scaler.step(target)
                scaler.update()
                return 0.0
            if group_norms_out is not None:
                group_norms_out.update(parameter_group_gradient_norms(module))
            norm = nn.utils.clip_grad_norm_(module.parameters(), self.cfg.optimizer.grad_max_norm)
            ensure_finite_tensor(f"{name}_grad_norm", torch.as_tensor(norm))
            for target in targets:
                scaler.step(target)
            scaler.update()
        else:
            loss.backward()
            ensure_finite_gradients(module, name=name)
            if group_norms_out is not None:
                group_norms_out.update(parameter_group_gradient_norms(module))
            norm = nn.utils.clip_grad_norm_(module.parameters(), self.cfg.optimizer.grad_max_norm)
            ensure_finite_tensor(f"{name}_grad_norm", torch.as_tensor(norm))
            optimizer.step()
        self.learner_updates += 1
        return float(torch.as_tensor(norm).detach().cpu())

    def _common_metrics(self, rollout, validation, terminal_infos, rollout_seconds: float) -> dict[str, object]:
        metrics = rollout_metrics(
            rollout,
            terminal_infos=terminal_infos,
            env_steps_total=self.sampler.total_env_steps,
            game_steps_total=self.sampler.total_game_steps,
            completed_games_total=self.completed_games_total + validation.completed_games,
            rollout_seconds=rollout_seconds,
        )
        metrics.update({
            "statistics/learner_updates": self.learner_updates,
            "statistics/iterations": self.iteration + 1,
            "statistics/durable_updates": self.durable_updates + 1,
            "runtime/schema": M03_RUNTIME_SCHEMA,
            "runtime/backend": str(getattr(self.game, "BACKEND", "unknown")),
            "runtime/algorithm": self.algorithm,
            "runtime/smoke_profile": "local-smoke" in self.cfg.game.identifier,
            "runtime/reduced_horizon": int(self.cfg.game.max_game_horizon),
            "runtime/ddp_enabled": self.distributed.enabled,
            "runtime/world_size": self.world_size,
            "runtime/rank": self.rank,
            "runtime/local_rollout_batch_size": self.local_rollout_batch_size,
            "runtime/local_collect_batch_size": self.local_collect_batch_size,
            "runtime/merged_policy_identity": self.merged_policy_identity,
            "runtime/optimizer": "muon+adamw" if self.cfg.optimizer.use_muon else "adamw",
            "runtime/compile_enabled": self.cfg.compile.enabled,
            "runtime/amp_enabled": self.cfg.amp_enabled,
            "runtime/amp_autocast_dtype": str(self.amp_dtype) if self.amp_enabled else "none",
            "runtime/amp_scaler_enabled": self.amp_scaler_enabled,
            "runtime/amp_scaler_scale": float(self.grad_scaler.get_scale()),
            "runtime/amp_skipped_steps": int(self.amp_skipped_steps),
            "memory/sampler_bytes_per_rank_estimate": int(self.memory_budget["sampler_bytes_per_rank"]),
        })
        if self.cfg.trim_completed_padding:
            time_capacity = int(rollout["player_ids"].shape[1])
            metrics.update({
                "runtime/trim_completed_padding": True,
                "perf/learner_time_capacity": time_capacity,
                "perf/original_time_capacity": int(self.cfg.game.max_game_horizon),
                "perf/removed_tail_padding_fraction": 1.0 - time_capacity / int(self.cfg.game.max_game_horizon),
            })
        return metrics

    def _finish_iteration(self, *, metrics: dict[str, object], completed_games: int) -> dict[str, object]:
        if self.iteration_status != "running" or self.active_iteration is None:
            raise RuntimeError("cannot finish without an active iteration")
        target_iteration = self.active_iteration
        target_durable = self.durable_updates + 1
        target_completed = self.completed_games_total + completed_games
        metrics["statistics/iterations"] = target_iteration
        metrics["statistics/durable_updates"] = target_durable
        metrics["statistics/completed_games_total"] = target_completed
        if self.evaluation_hook is not None:
            try:
                completed = self._collect_evaluation_results()
                metrics["evaluation/results_collected"] = len(completed)
                metrics["evaluation/results_failed"] = sum(
                    item.get("status") in {"failed", "cancelled", "interrupted"}
                    for item in completed
                )
                if completed:
                    latest = completed[-1]
                    metrics["evaluation/last_result_status"] = str(latest.get("status", "unknown"))
                    metrics["evaluation/last_result_iteration"] = int(latest.get("iteration", -1))
                    if latest.get("error") is not None:
                        metrics["evaluation/last_result_error"] = str(latest["error"])
                hook_result = self.evaluation_hook.on_iteration_completed(
                    self, iteration=target_iteration,
                )
                metrics["evaluation/hook_status"] = str(hook_result.get("status", "unknown"))
                if hook_result.get("snapshot") is not None:
                    metrics["evaluation/snapshot"] = str(hook_result["snapshot"])
            except Exception as error:  # Evaluation scheduling/collection must not poison training.
                metrics["evaluation/hook_status"] = "schedule_failed"
                metrics["evaluation/hook_error"] = f"{type(error).__name__}: {error}"
        _finite_metrics(metrics)
        self._emit_phase("metrics")
        self.metrics.log(metrics, step=target_iteration)
        self.iteration = target_iteration
        self.durable_updates = target_durable
        self.durable_learner_updates = self.learner_updates
        self.completed_games_total = target_completed
        self.iteration_status = "completed"
        self.active_iteration = None
        self.last_metrics = dict(metrics)
        return dict(metrics)

    def _checkpoint_common(self) -> dict[str, object]:
        self._require_checkpointable()
        sampler_state = self.sampler.state_dict()
        payload: dict[str, object] = {
            "schema": TRAINING_CHECKPOINT_SCHEMA,
            "algorithm": self.algorithm,
            "architecture": self.cfg.model.architecture,
            "protocol_version": self.cfg.game.protocol_version,
            "backend": self.backend_name,
            "backend_identity": self.backend_identity,
            "semantic_contract": self.semantic_contract,
            "training_semantics_profile": self.training_semantics_profile,
            "policy_regularizer_mode": self.policy_regularizer_mode,
            "reward_semantics": self.cfg.reward_semantics,
            "reward_scale": self.cfg.reward_scale,
            "source_fingerprint": self.source_fingerprint,
            "source_hash": self.source_hash,
            "execution_fingerprint": self.execution_fingerprint,
            "source_parent_commit": self.source_parent_commit,
            "resolved_config": self.resolved_config,
            "resolved_config_hash": self.resolved_config_hash,
            "iteration": self.iteration,
            "durable_updates": self.durable_updates,
            "learner_updates": self.learner_updates,
            "durable_learner_updates": self.durable_learner_updates,
            "iteration_status": self.iteration_status,
            "counters": {
                "total_env_steps": self.sampler.total_env_steps,
                "total_game_steps": self.sampler.total_game_steps,
                "completed_games_total": self.completed_games_total,
            },            # Only present for a fresh-import run whose curriculum origin is
            # explicit; the historical payload stays byte-identical (offset 0).
            **(
                {"curriculum_offset": int(self.curriculum_offset)}
                if int(getattr(self, "curriculum_offset", 0)) else {}
            ),
            "rng_state": capture_rng_state(),
            "sampler_state": sampler_state,
            "grad_scaler_state": self.grad_scaler.state_dict(),
            "amp": {
                "enabled": self.amp_enabled,
                "device_type": self.amp_device_type,
                "autocast_dtype": str(self.amp_dtype),
                "scaler_enabled": self.amp_scaler_enabled,
                "skipped_steps": self.amp_skipped_steps,
            },
            "schedule_state": asdict(compute_agent_schedule(self.cfg, iteration=max(self.iteration, 1))),
            "merged_policy": {
                "identity": self.merged_policy_identity,
                "beta": self.cfg.agent_merging_beta,
                "not_smoothed_when_beta_zero": self.cfg.agent_merging_beta == 0.0,
            },
            "selection_identity": None,
            "champion_identity": None,
            "training_data_provenance": self.sampler.training_data_provenance(),
            "resume_classification": "exact_resume",
        }
        payload["opponent_pool"] = self._opponent_pool_checkpoint_block(sampler_state)
        return payload

    def _opponent_pool_checkpoint_block(self, sampler_state: Mapping[str, object]) -> dict[str, object] | None:
        if self.policy_router is None or self.opponent_pool is None:
            return None
        router_state = sampler_state.get("policy_router")
        if not isinstance(router_state, Mapping):
            raise RuntimeError("pool-enabled sampler state is missing the router state")
        block: dict[str, object] = {
            "schema": OPPONENT_POOL_CHECKPOINT_SCHEMA,
            "enabled": True,
            "manifest_hash": self.opponent_pool.manifest_hash,
            "manifest_path": (
                None if self.opponent_pool_manifest_path is None
                else str(self.opponent_pool_manifest_path)
            ),
            "entries": [entry.identity() for entry in self.opponent_pool.entries],
            "sampling_seed": int(self.opponent_pool.sampling_seed),
            "sampling_scheme": str(self.opponent_pool.sampling_scheme),
            "assignment_scheme": str(self.opponent_pool.assignment_scheme),
            "max_resident_policies": int(self.cfg.opponent_pool.max_resident_policies),
            "router_state_hash": str(router_state.get("state_hash")),
        }
        block["hash"] = canonical_json_hash(block)
        return block

    def _verify_opponent_pool_checkpoint(self, payload: Mapping[str, object]) -> None:
        """Reject a pool mismatch before any trainer state is modified."""
        block = payload.get("opponent_pool")
        if self.policy_router is None:
            if block is not None:
                raise CheckpointCompatibilityError(
                    "checkpoint contains opponent-pool state but the runtime pool is disabled"
                )
            return
        if self.opponent_pool is None:
            raise RuntimeError("pool router exists without an opponent pool")
        if not isinstance(block, Mapping):
            raise CheckpointCompatibilityError(
                "pool-enabled runtime requires an opponent_pool checkpoint block"
            )
        declared = block.get("hash")
        if declared != canonical_json_hash(
            {key: value for key, value in block.items() if key != "hash"}
        ):
            raise CheckpointCompatibilityError("opponent_pool checkpoint block hash is inconsistent")
        if block.get("schema") != OPPONENT_POOL_CHECKPOINT_SCHEMA:
            raise CheckpointCompatibilityError("opponent_pool checkpoint block schema mismatch")
        if not bool(block.get("enabled", False)):
            raise CheckpointCompatibilityError("opponent_pool checkpoint block is not enabled")
        if str(block.get("manifest_hash")) != self.opponent_pool.manifest_hash:
            raise CheckpointCompatibilityError(
                "checkpoint opponent pool manifest differs from the running pool; exact resume "
                "cannot ignore a pool change"
            )
        expected_entries = [entry.identity() for entry in self.opponent_pool.entries]
        if block.get("entries") != expected_entries:
            raise CheckpointCompatibilityError(
                "checkpoint opponent pool entries differ from the running pool"
            )
        for key, expected in (
            ("sampling_seed", self.opponent_pool.sampling_seed),
            ("sampling_scheme", self.opponent_pool.sampling_scheme),
            ("assignment_scheme", self.opponent_pool.assignment_scheme),
            ("max_resident_policies", int(self.cfg.opponent_pool.max_resident_policies)),
        ):
            if block.get(key) != expected:
                raise CheckpointCompatibilityError(f"opponent_pool checkpoint {key} mismatch")
        sampler_state = payload.get("sampler_state")
        if not isinstance(sampler_state, Mapping):
            raise CheckpointCompatibilityError("pool-enabled checkpoint lacks sampler state")
        router_state = sampler_state.get("policy_router")
        if not isinstance(router_state, Mapping):
            raise CheckpointCompatibilityError("pool-enabled checkpoint lacks router state")
        if str(router_state.get("state_hash")) != str(block.get("router_state_hash")):
            raise CheckpointCompatibilityError(
                "opponent pool router state hash disagrees with the checkpoint block"
            )

    def _classify_checkpoint(self, payload: Mapping[str, object], *, require_replay: bool):
        decision = classify_resume(
            payload,
            expected_contract=self.semantic_contract,
            expected_source_fingerprint=self.source_fingerprint,
            expected_execution_fingerprint=self.execution_fingerprint,
            expected_backend_identity=self.backend_identity,
            expected_config=self.resolved_config,
            require_replay=require_replay,
        )
        if decision.kind not in {"exact_resume", "compatible_resume"}:
            raise CheckpointCompatibilityError(
                f"checkpoint cannot continue training: kind={decision.kind}; "
                + "; ".join(decision.reasons)
            )
        return decision

    def _verify_checkpoint_training_semantics(self, payload: Mapping[str, object]) -> None:
        """Fail closed when a checkpoint carries a different training semantics.

        Missing keys are treated as the legacy defaults so that pre-phase-2
        checkpoints can still be classified honestly instead of being
        mislabelled as macro checkpoints.
        """
        profile = str(payload.get("training_semantics_profile", LEGACY_TRAINING_SEMANTICS_PROFILE))
        regularizer = str(payload.get("policy_regularizer_mode", LEGACY_POLICY_REGULARIZER_MODE))
        if profile != self.training_semantics_profile or regularizer != self.policy_regularizer_mode:
            raise CheckpointCompatibilityError(
                "checkpoint training semantics do not match the runtime configuration: "
                f"checkpoint=({profile}, {regularizer}) "
                f"runtime=({self.training_semantics_profile}, {self.policy_regularizer_mode})"
            )
        # A v1.3 training recovery is never a valid v1.4 exact resume, and vice
        # versa: the residual-fuse architecture changes the parameter set, so the
        # checkpoint architecture must equal the runtime architecture exactly.
        checkpoint_architecture = payload.get("architecture")
        if checkpoint_architecture is not None and str(checkpoint_architecture) != str(self.cfg.model.architecture):
            raise CheckpointCompatibilityError(
                "checkpoint model architecture does not match the runtime configuration: "
                f"checkpoint={checkpoint_architecture!r} runtime={self.cfg.model.architecture!r}"
            )

    def _load_common_state(self, payload: Mapping[str, object]) -> None:
        self.iteration = int(payload["iteration"])
        self.durable_updates = int(payload["durable_updates"])
        self.learner_updates = int(payload.get("learner_updates", self.durable_updates))
        self.durable_learner_updates = int(payload.get("durable_learner_updates", self.learner_updates))
        counters = payload["counters"]
        self.completed_games_total = int(counters["completed_games_total"])
        self.curriculum_offset = int(payload.get("curriculum_offset", 0))
        self.sampler.load_state_dict(payload["sampler_state"])
        self._frozen_hashes = None
        scaler_state = payload.get("grad_scaler_state")
        if isinstance(scaler_state, Mapping):
            self.grad_scaler.load_state_dict(scaler_state)
        amp_state = payload.get("amp")
        if isinstance(amp_state, Mapping):
            self.amp_skipped_steps = int(amp_state.get("skipped_steps", 0) or 0)
        restore_rng_state(payload["rng_state"])
        self.iteration_status = "completed"
        self.active_iteration = None
        self.failed = False
        self.failure = None



class IPPOTrainer(LocalTrainerBase):
    algorithm = "ippo"

    def __init__(self, cfg: TrainConfig, *, game: object, run_dir: str | Path) -> None:
        super().__init__(cfg, game=game, run_dir=run_dir)
        enable_ntp = Agent.ntp_enabled_for_config(cfg)
        self.pad_token = int(getattr(game, "pad_token", 0))
        self.agent_resume = Agent(cfg, enable_value_head=True, enable_ntp_head=enable_ntp).to(self.device)
        self.agent_merged = Agent(cfg, enable_value_head=True, enable_ntp_head=enable_ntp).to(self.device)
        self._build_symmetry(self.agent_resume)
        self.merge_updater = ExponentialMovingAverageUpdater(
            model=self.agent_resume,
            merged_model=self.agent_merged,
            beta=cfg.agent_merging_beta,
        )
        self.agent_model = self._prepare_trainable(self.agent_resume)
        self.agent_optimizer = self.agent_resume.configure_optimizer(
            lr=cfg.ppo.base_lr, optimizer=cfg.optimizer,
        )
        self._verify_frozen_excluded_from_optimizer(self.agent_optimizer)
        self._activate_rank_rng()

    def train_iteration(self) -> dict[str, object]:
        if self.macro_semantics:
            return self._macro_train_iteration()
        self._begin_iteration()
        stage = "collect"
        try:
            schedule = compute_agent_schedule(self.cfg, iteration=self.iteration + 1)
            apply_learning_rate(self.agent_optimizer, schedule.agent_lr)
            rollout, validation, infos, rollout_seconds = self._collect(self.agent_model)
            learner_started = time.perf_counter()
            stage = "gae"
            with torch.no_grad():
                initial = self._forward_agent(rollout)
                state_values = initial.state_values.detach().float()
                ensure_finite_tensor("ippo_state_values", state_values)
                gae = compute_gae_from_value(
                    rollout,
                    state_values,
                    discount_gamma=self.cfg.discount_gamma,
                    gae_lambda=self.cfg.ppo.gae_lambda,
                )
                ensure_finite_tensor("ippo_advantage", gae.player_advantage)
                ensure_finite_tensor("ippo_value_target", gae.player_state_value)
            batch_size = int(rollout.batch_size[0])
            mini_batch_size = max(1, math.ceil(batch_size / self.cfg.mini_batches))
            first_metrics: dict[str, float] | None = None
            last_metrics: dict[str, float] | None = None
            grad_norm = 0.0
            degenerate_target_variance_count = 0
            stage = "policy_update"
            for epoch in range(self.cfg.ppo.agent_epoch):
                self.last_policy_epochs = epoch + 1
                indices = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, mini_batch_size):
                    mb_indices = indices[start:min(start + mini_batch_size, batch_size)]
                    mb = rollout[mb_indices]
                    output = self._forward_agent(mb)
                    pg = compute_clipped_pg_loss(
                        mb,
                        clip_coef=schedule.clip_coef,
                        avail_probs_new=output.avail_probs.float(),
                        player_advantage=gae.player_advantage[mb_indices],
                    )
                    value = compute_value_loss(
                        mb,
                        player_state_value_pred=output.player_state_value,
                        player_state_value_target=gae.player_state_value[mb_indices],
                    )
                    total = pg.pg_loss_mean + schedule.reg_coef * pg.kl_unif_mean + self.cfg.value_loss_coef * value.value_loss_mean
                    symmetry_record: dict[str, object] = {}
                    symmetry_out = self._symmetry_loss(mb, output)
                    if symmetry_out is not None:
                        total = total + self.cfg.symmetry.symmetry_weight * symmetry_out.loss
                        symmetry_record = {
                            "symmetry_loss": float(symmetry_out.loss.detach()),
                            "symmetry_mean_kl": float(symmetry_out.mean_kl_global),
                            "symmetry_token_count_global": int(
                                symmetry_out.token_count_global
                            ),
                            "symmetry_candidate_count_local": int(
                                symmetry_out.candidate_count_local
                            ),
                        }
                    ntp_loss_value: float | None = None
                    ntp_pos_rate: float | None = None
                    if self.cfg.ntp_weight > 0.0 and output.ntp_logits is not None:
                        ntp = compute_next_observation_token_loss(
                            mb,
                            ntp_logits=output.ntp_logits,
                            pad_token=self.pad_token,
                            pos_weight=self.cfg.ntp_pos_weight,
                        )
                        total = total + self.cfg.ntp_weight * ntp.ntp_loss_mean
                        ntp_loss_value = float(ntp.ntp_loss_mean.detach())
                        ntp_pos_rate = float(ntp.ntp_pos_rate)
                    mb_time = torch.arange(mb["player_ids"].shape[1], device=self.device).unsqueeze(0)
                    mb_active = (mb_time < mb["num_steps"].unsqueeze(1)) & (mb["player_ids"] >= 0)
                    explained_variance, degenerate = _explained_variance(
                        output.player_state_value, gae.player_state_value[mb_indices], mb_active,
                    )
                    degenerate_target_variance_count += degenerate
                    group_norms: dict[str, float] = {}
                    grad_norm = self._optimizer_step(
                        loss=total, module=self.agent_resume,
                        optimizer=self.agent_optimizer, name="ippo_agent",
                        group_norms_out=group_norms,
                    )
                    record = {
                        "pg_loss": float(pg.pg_loss_mean.detach()),
                        "value_loss": float(value.value_loss_mean.detach()),
                        "kl_unif": float(pg.kl_unif_mean.detach()),
                        "kl_ref": float(pg.kl_ref_mean.detach()),
                        "clip_frac": float(pg.clip_frac.detach()),
                        "adv_mean": float(pg.adv_mean.detach()),
                        "adv_std": float(pg.adv_std.detach()),
                        "value_ev": explained_variance,
                        "value_ev_undefined": 1 if explained_variance is None else 0,
                        **symmetry_record,
                    }
                    for group, value_norm in group_norms.items():
                        record[f"grad_norm/{group}"] = value_norm
                    if ntp_loss_value is not None:
                        record["ntp_loss"] = ntp_loss_value
                        record["ntp_pos_rate"] = float(ntp_pos_rate)
                    if first_metrics is None:
                        first_metrics = record
                    last_metrics = record
            self.merge_updater.step(iteration=self.iteration + 1)
            learner_seconds = time.perf_counter() - learner_started
            metrics = self._common_metrics(rollout, validation, infos, rollout_seconds)
            for prefix, values in (("agent_initial", first_metrics), ("agent", last_metrics)):
                for key, value in (values or {}).items():
                    metrics[f"{prefix}/{key}"] = value
            metrics.update({
                "agent/learning_rate": schedule.agent_lr,
                "agent/clip_coef": schedule.clip_coef,
                "agent/reg_coef": schedule.reg_coef,
                "agent/grad_norm": grad_norm,
                "value/degenerate_target_variance_count": degenerate_target_variance_count,
                "perf/learner_seconds": learner_seconds,
            })
            return self._finish_iteration(metrics=metrics, completed_games=validation.completed_games)
        except BaseException as error:
            self._poison(stage=stage, error=error)
            raise

    def _macro_train_iteration(self) -> dict[str, object]:
        """Profile B (sequence ablation) / Profile C (macro decision) learner."""
        self._begin_iteration()
        stage = "collect"
        try:
            schedule = compute_agent_schedule(self.cfg, iteration=self.iteration + 1)
            apply_learning_rate(self.agent_optimizer, schedule.agent_lr)
            rollout, validation, infos, rollout_seconds = self._collect(self.agent_model)
            learner_started = time.perf_counter()
            stage = "macro_targets"
            if self.policy_router is not None:
                from guandan_marvel.phase6.policy_owner import segment_pooled_macros

                segmentation = segment_pooled_macros(rollout)
            else:
                segmentation = segment_rollout_macros(rollout)
            profile = self.training_semantics_profile
            regularizer_mode = self.policy_regularizer_mode
            macro_lambda = (
                float(self.cfg.ppo.gae_lambda)
                if self.macro_gae_lambda is None
                else float(self.macro_gae_lambda)
            )
            with torch.no_grad():
                initial = self._forward_agent(rollout)
                state_values = initial.state_values.detach().float()
                ensure_finite_tensor("macro_state_values", state_values)
                token_gae = compute_gae_from_value(
                    rollout,
                    state_values,
                    discount_gamma=self.cfg.discount_gamma,
                    gae_lambda=self.cfg.ppo.gae_lambda,
                )
                token_advantage = token_gae.player_advantage.detach()
                token_value_target = token_gae.player_state_value.detach()
                ensure_finite_tensor("macro_token_advantage", token_advantage)
                ensure_finite_tensor("macro_token_value_target", token_value_target)
                if profile == MACRO_DECISION_TRAINING_SEMANTICS_PROFILE:
                    macro_gae = compute_macro_decision_gae(
                        rollout,
                        state_values,
                        segmentation=segmentation,
                        discount_gamma=self.cfg.discount_gamma,
                        macro_lambda=macro_lambda,
                    )
                    macro_advantage = macro_gae.advantage.detach()
                    macro_value_target = macro_gae.value_target.detach()
                    macro_ready = macro_gae.valid
                    ensure_finite_tensor("macro_decision_advantage", macro_advantage[macro_ready])
                    ensure_finite_tensor("macro_decision_value_target", macro_value_target[macro_ready])
                else:
                    macro_gae = None
                    macro_advantage = token_advantage.gather(
                        1, segmentation.macro_start_index.clamp(min=0),
                    ).detach()
                    macro_value_target = token_value_target
                    macro_ready = segmentation.macro_valid
                macro_advantage = macro_advantage.contiguous()
                ensure_finite_tensor("macro_advantage", macro_advantage[macro_ready])
            batch_size = int(rollout.batch_size[0])
            mini_batch_size = max(1, math.ceil(batch_size / self.cfg.mini_batches))
            first_metrics: dict[str, object] | None = None
            last_metrics: dict[str, object] | None = None
            grad_norm = 0.0
            degenerate_target_variance_count = 0
            macro_value_loss_record: float | None = None
            # Optional macro sample-KL early stop (``ppo.target_kl``).  ``None``
            # runs the historical loop unchanged: no extra collective, no
            # extra metric key and no extra arithmetic on the loss path.
            target_kl = getattr(self.cfg.ppo, "target_kl", None)
            kl_attempted = 0
            kl_applied = 0
            kl_skipped = 0
            kl_amp_skipped = 0
            kl_total_sum = 0.0
            kl_total_count = 0
            kl_stop_position: dict[str, object] | None = None
            kl_last_decision = None
            kl_stop_requested = False
            stage = "policy_update"
            for epoch in range(self.cfg.ppo.agent_epoch):
                self.last_policy_epochs = epoch + 1
                indices = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, mini_batch_size):
                    mb_indices = indices[start:min(start + mini_batch_size, batch_size)]
                    mb = rollout[mb_indices]
                    view = build_macro_batch_view(segmentation, mb_indices)
                    output = self._forward_agent(mb)
                    if output.avail_log_probs is None:
                        raise RuntimeError("macro training requires FP32 avail_log_probs from the agent")
                    pg = compute_macro_pg_loss(
                        mb,
                        view=view,
                        avail_log_probs_new=output.avail_log_probs,
                        macro_advantage=macro_advantage[mb_indices],
                        clip_coef=schedule.clip_coef,
                        world_size=self.world_size,
                    )
                    if target_kl is not None:
                        # Documented monitoring point: the macro KL of the
                        # *pending* minibatch, measured before its update.  The
                        # statistic is the global sum/count (never a mean of
                        # rank means) and every rank evaluates the same
                        # all-reduced value, so the stop decision is identical.
                        kl_decision = macro_sample_kl_control(
                            pg.log_ratio, pg.valid, view.macro_length,
                            world_size=self.world_size, target_kl=float(target_kl),
                        )
                        kl_attempted += 1
                        kl_last_decision = kl_decision
                        kl_total_sum += float(kl_decision.global_sum)
                        kl_total_count += int(kl_decision.global_count)
                        if kl_decision.stop:
                            kl_skipped += 1
                            kl_stop_position = {
                                "epoch": int(epoch),
                                "minibatch": int(start // mini_batch_size),
                                "epochs_total": int(self.cfg.ppo.agent_epoch),
                                "global_sum": float(kl_decision.global_sum),
                                "global_count": int(kl_decision.global_count),
                                "global_mean": kl_decision.global_mean,
                                "target_kl": float(target_kl),
                            }
                            kl_stop_requested = True
                            break
                    # Global N > 0 is guaranteed by compute_macro_pg_loss (it fails fast
                    # when the all-reduced macro count is zero).  A rank may still hold
                    # no local macros; spec 9.3 requires such a rank to emit finite,
                    # model-connected zero losses instead of an empty-set NaN or a
                    # disconnected constant that would trip find_unused_parameters=False.
                    local_empty = pg.macro_count_local == 0
                    regularizer = macro_regularizer_diagnostics(
                        mb, view=view, avail_probs_new=output.avail_probs,
                    )
                    if regularizer_mode == MACRO_NORMALIZED_POLICY_REGULARIZER_MODE:
                        macro_reg_term = compute_macro_normalized_kl_mean(
                            mb,
                            view=view,
                            avail_probs_new=output.avail_probs,
                            world_size=self.world_size,
                        )
                        legacy_kl = float(regularizer["legacy_token_kl_mean"])
                    elif local_empty:
                        # A rank without local macro samples must still contribute a
                        # graph-connected zero regularizer term (spec 9.3); averaging an
                        # empty active set through compute_clipped_pg_loss would be NaN,
                        # and connected_zero_like(pg.pg_loss_mean) is disconnected because
                        # the local macro loss is a fresh constant here.
                        macro_reg_term = model_connected_zero(
                            output.avail_log_probs, output.player_state_value,
                        )
                        legacy_kl = 0.0
                    else:
                        legacy_pg = compute_clipped_pg_loss(
                            mb,
                            clip_coef=schedule.clip_coef,
                            avail_probs_new=output.avail_probs.float(),
                            player_advantage=token_advantage[mb_indices],
                        )
                        macro_reg_term = legacy_pg.kl_unif_mean
                        legacy_kl = float(legacy_pg.kl_unif_mean.detach())
                    total = pg.pg_loss_mean + schedule.reg_coef * macro_reg_term
                    value_record: dict[str, object] = {}
                    if profile == MACRO_DECISION_TRAINING_SEMANTICS_PROFILE:
                        macro_value = compute_macro_value_loss(
                            player_state_value_pred=output.player_state_value,
                            view=view,
                            macro_value_target=macro_value_target[mb_indices],
                            world_size=self.world_size,
                        )
                        total = total + self.cfg.value_loss_coef * macro_value.macro_value_loss_mean
                        macro_value_loss_record = _per_sample_macro_metric(
                            float(macro_value.macro_value_loss_mean.detach()),
                            global_count=macro_value.macro_count_global,
                            local_count=macro_value.macro_count_local,
                            world_size=self.world_size,
                        )
                        macro_trainable = macro_trainable_mask(view)
                        prediction = output.player_state_value.detach().gather(
                            1, view.macro_start_index.clamp(min=0),
                        )
                        ev, degenerate = explain_macro_value(
                            prediction, macro_value_target[mb_indices], macro_trainable,
                        )
                        degenerate_target_variance_count += degenerate
                        value_record = {
                            "value_loss": macro_value_loss_record,
                            "value_ev": ev,
                            "value_ev_undefined": 1 if ev is None else 0,
                        }
                    else:
                        if local_empty:
                            # No local acting token: F.huber_loss(empty, empty, mean) is
                            # NaN.  Use a model-connected zero so the value path stays
                            # graph-connected (spec 4.3); the recorded loss is exactly 0.0.
                            token_value_term = model_connected_zero(output.player_state_value)
                            value_record = {
                                "value_loss": 0.0,
                                "token_value_loss": 0.0,
                                "value_ev": None,
                                "value_ev_undefined": 1,
                            }
                        else:
                            token_value = compute_value_loss(
                                mb,
                                player_state_value_pred=output.player_state_value,
                                player_state_value_target=token_value_target[mb_indices],
                            )
                            token_value_term = token_value.value_loss_mean
                            token_time = torch.arange(mb["player_ids"].shape[1], device=self.device).unsqueeze(0)
                            token_active = (
                                (token_time < mb["num_steps"].unsqueeze(1)) & (mb["player_ids"] >= 0)
                            )
                            if "policy_train_mask" in mb:
                                token_active = token_active & mb["policy_train_mask"].to(torch.bool)
                            token_ev, token_degenerate = _explained_variance(
                                output.player_state_value, token_value_target[mb_indices], token_active,
                            )
                            degenerate_target_variance_count += token_degenerate
                            value_record = {
                                "value_loss": float(token_value.value_loss_mean.detach()),
                                "token_value_loss": float(token_value.value_loss_mean.detach()),
                                "value_ev": token_ev,
                                "value_ev_undefined": 1 if token_ev is None else 0,
                            }
                        total = total + self.cfg.value_loss_coef * token_value_term
                    symmetry_record: dict[str, object] = {}
                    symmetry_out = self._symmetry_loss(mb, output)
                    if symmetry_out is not None:
                        total = total + self.cfg.symmetry.symmetry_weight * symmetry_out.loss
                        symmetry_record = {
                            "symmetry_loss": float(symmetry_out.loss.detach()),
                            "symmetry_mean_kl": float(symmetry_out.mean_kl_global),
                            "symmetry_token_count_global": int(
                                symmetry_out.token_count_global
                            ),
                            "symmetry_candidate_count_local": int(
                                symmetry_out.candidate_count_local
                            ),
                        }
                    record: dict[str, object] = {
                        # Per-rank per-sample mean (DDP loss scaling removed): the
                        # aggregator reconstructs the global per-sample mean by
                        # weighting with macro_count_local.  w == 1 is bitwise unchanged.
                        "pg_loss": _per_sample_macro_metric(
                            float(pg.pg_loss_mean.detach()),
                            global_count=pg.macro_count_global,
                            local_count=pg.macro_count_local,
                            world_size=self.world_size,
                        ),
                        "clip_frac": pg.clip_frac,
                        "macro_count_local": pg.macro_count_local,
                        "macro_count_global": pg.macro_count_global,
                        "token_count_local": int(regularizer["token_count"]),
                        # Rank-consistent DDP aggregates (identical on every rank; the
                        # metrics aggregator must not average them across ranks).
                        "adv_raw_mean_global": pg.adv_raw_mean_global,
                        "adv_raw_std_global": pg.adv_raw_std_global,
                        "adv_norm_mean_global": pg.adv_norm_mean_global,
                        "adv_norm_std_global": pg.adv_norm_std_global,
                        "adv_count_global": pg.adv_count_global,
                        "adv_raw_mean": pg.adv_raw_mean,
                        "adv_raw_std": pg.adv_raw_std,
                        "adv_raw_min": pg.adv_raw_min,
                        "adv_raw_max": pg.adv_raw_max,
                        "adv_norm_mean": pg.adv_norm_mean,
                        "adv_norm_std": pg.adv_norm_std,
                        "adv_norm_min": pg.adv_norm_min,
                        "adv_norm_max": pg.adv_norm_max,
                        "log_ratio_mean": pg.log_ratio_mean,
                        "log_ratio_std": pg.log_ratio_std,
                        "log_ratio_min": pg.log_ratio_min,
                        "log_ratio_max": pg.log_ratio_max,
                        "ratio_mean": pg.ratio_mean,
                        "ratio_min": pg.ratio_min,
                        "ratio_max": pg.ratio_max,
                        "old_logprob_mean": pg.old_logprob_mean,
                        "new_logprob_mean": pg.new_logprob_mean,
                        "macro_length_mean": pg.macro_length_mean,
                        "macro_length_max": pg.macro_length_max,
                        "zero_variance_count": pg.zero_variance_count,
                        "empty_local_batch": 1 if pg.empty_local_batch else 0,
                        "old_logprob_crosscheck_max_abs_diff": pg.old_logprob_crosscheck_max_abs_diff,
                        "legacy_token_kl": legacy_kl,
                        "macro_kl": regularizer["macro_kl_mean"],
                        # Rollout-policy sample estimate KL(old || new), not KL(pi || U).
                        "approx_kl_old_new": (
                            float((torch.expm1(pg.log_ratio[pg.valid].detach()) - pg.log_ratio[pg.valid].detach()).mean().item())
                            if pg.macro_count_local else 0.0
                        ),
                        "raw_entropy": regularizer["raw_entropy_mean"],
                        "normalized_entropy": regularizer["normalized_entropy_mean"],
                        # Forced single-legal-action decisions count as zero in the
                        # overall entropy; the non-forced mean is kept separately.
                        "non_forced_normalized_entropy": regularizer["non_forced_normalized_entropy_mean"],
                        "non_forced_token_count": regularizer["non_forced_token_count"],
                        "regularizer_mode": regularizer_mode,
                        "single_legal_token_count": regularizer["single_legal_token_count"],
                    }
                    record.update(value_record)
                    record.update(symmetry_record)
                    for bucket, value in pg.clip_frac_by_length.items():
                        record[f"clip_frac_by_length/{bucket}"] = value
                    if self.cfg.ntp_weight > 0.0 and output.ntp_logits is not None:
                        ntp = compute_next_observation_token_loss(
                            mb,
                            ntp_logits=output.ntp_logits,
                            pad_token=self.pad_token,
                            pos_weight=self.cfg.ntp_pos_weight,
                        )
                        total = total + self.cfg.ntp_weight * ntp.ntp_loss_mean
                        record["ntp_loss"] = float(ntp.ntp_loss_mean.detach())
                        record["ntp_pos_rate"] = float(ntp.ntp_pos_rate)
                    if local_empty:
                        # Spec 4.4.2: the final total on an empty rank must additionally
                        # carry a zero anchor built from real model outputs so that every
                        # policy/value (and NTP, when enabled) parameter is touched by
                        # autograd on this rank under find_unused_parameters=False.
                        total = total + model_connected_zero(
                            output.avail_log_probs, output.player_state_value, output.ntp_logits,
                        )
                    group_norms: dict[str, float] = {}
                    updates_before = self.learner_updates
                    grad_norm = self._optimizer_step(
                        loss=total, module=self.agent_resume,
                        optimizer=self.agent_optimizer, name="macro_agent",
                        group_norms_out=group_norms,
                    )
                    if target_kl is not None:
                        if self.learner_updates > updates_before:
                            kl_applied += 1
                        else:
                            # A genuine fp16 overflow skip is neither an applied
                            # update nor a KL stop; keep the two counters apart.
                            kl_amp_skipped += 1
                    for group, value in group_norms.items():
                        record[f"grad_norm/{group}"] = value
                    if first_metrics is None:
                        first_metrics = record
                    last_metrics = record
                if kl_stop_requested:
                    break
            self.merge_updater.step(iteration=self.iteration + 1)
            learner_seconds = time.perf_counter() - learner_started
            metrics = self._common_metrics(rollout, validation, infos, rollout_seconds)
            for prefix, values in (("macro_initial", first_metrics), ("macro", last_metrics)):
                for key, value in (values or {}).items():
                    metrics[f"{prefix}/{key}"] = value
            metrics.update({
                "macro/profile": profile,
                "macro/regularizer_mode": regularizer_mode,
                "macro/lambda": macro_lambda,
                "macro/discount_gamma": self.cfg.discount_gamma,
                "macro/segmentation_schema": segmentation.schema,
                "macro/diagnostic_incomplete_count": segmentation.incomplete_count,
                "macro/diagnostic_player_switch_violation_count": segmentation.player_switch_violation_count,
                "macro/legacy_token_advantage": profile != MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
                "macro/value_sample_unit": "macro_start" if profile == MACRO_DECISION_TRAINING_SEMANTICS_PROFILE else "token",
                "macro/degenerate_target_variance_count": degenerate_target_variance_count,
                "macro/value_loss_last": macro_value_loss_record,
                "macro/clamp_count": 0,
                "macro/truncated_count": 0,
                "macro/illegal_active_action_count": 0,
                "health/legal_single_action_tokens_last": int(
                    (last_metrics or {}).get("single_legal_token_count", 0)
                ),
                "agent/learning_rate": schedule.agent_lr,
                "agent/clip_coef": schedule.clip_coef,
                "agent/reg_coef": schedule.reg_coef,
                "agent/grad_norm": grad_norm,
                "perf/learner_seconds": learner_seconds,
            })
            if target_kl is not None:
                # Explicit, separately named global KL-control block.  The
                # legacy ``macro/approx_kl_old_new`` field keeps its historical
                # meaning (the *last minibatch*, per-rank mean) and is never
                # relabelled as the whole-iteration final KL.
                metrics.update({
                    "macro/kl_control/schema": kl_last_decision.schema if kl_last_decision else None,
                    "macro/kl_control/target_kl": float(target_kl),
                    "macro/kl_control/stopped": bool(kl_stop_requested),
                    "macro/kl_control/stop_position": kl_stop_position,
                    "macro/kl_control/attempted_steps": int(kl_attempted),
                    "macro/kl_control/applied_steps": int(kl_applied),
                    "macro/kl_control/skipped_steps": int(kl_skipped),
                    "macro/kl_control/amp_skipped_steps": int(kl_amp_skipped),
                    "macro/kl_control/iteration_global_sum": float(kl_total_sum),
                    "macro/kl_control/iteration_global_count": int(kl_total_count),
                    "macro/kl_control/iteration_global_mean": (
                        float(kl_total_sum / kl_total_count) if kl_total_count else None
                    ),
                    "macro/kl_control/last_minibatch_global_sum": (
                        None if kl_last_decision is None else float(kl_last_decision.global_sum)
                    ),
                    "macro/kl_control/last_minibatch_global_count": (
                        None if kl_last_decision is None else int(kl_last_decision.global_count)
                    ),
                    "macro/kl_control/last_minibatch_global_mean": (
                        None if kl_last_decision is None else kl_last_decision.global_mean
                    ),
                    "macro/kl_control/last_minibatch_global_max": (
                        None if kl_last_decision is None else float(kl_last_decision.global_max)
                    ),
                    "macro/kl_control/length_bucket_sums": (
                        {} if kl_last_decision is None else dict(kl_last_decision.length_bucket_sums)
                    ),
                    "macro/kl_control/length_bucket_counts": (
                        {} if kl_last_decision is None else dict(kl_last_decision.length_bucket_counts)
                    ),
                    "macro/kl_control/quantiles": (
                        {} if kl_last_decision is None else dict(kl_last_decision.quantiles)
                    ),
                    "macro/kl_control/quantiles_approximate": (
                        None if kl_last_decision is None else bool(kl_last_decision.quantiles_approximate)
                    ),
                    "macro/kl_control/formula": "expm1(log_ratio) - log_ratio; global sum / global count",
                    "macro/kl_control/note": (
                        "monitored before the pending minibatch update; stopping does not roll back "
                        "an update that already happened"
                    ),
                })
            metrics.update(macro_diagnostics(segmentation, prefix="macro/seg_"))
            if self.policy_router is not None:
                self._assert_frozen_opponents_unchanged()
                full_view = build_macro_batch_view(
                    segmentation, torch.arange(batch_size, device=self.device),
                )
                trainable = macro_trainable_mask(full_view)
                complete = full_view.macro_valid & (full_view.macro_end_index >= 0)
                metrics.update(self.policy_router.metrics())
                if self.last_pool_stats is not None:
                    metrics.update({
                        "pool/current_token_count": int(self.last_pool_stats.current_token_count),
                        "pool/opponent_token_count": int(self.last_pool_stats.opponent_token_count),
                        "pool/current_assignment_count": int(
                            self.last_pool_stats.current_assignment_count
                        ),
                        "pool/opponent_owner_ids": list(self.last_pool_stats.opponent_owner_ids),
                        "pool/current_macro_count_local": int(trainable.sum().item()),
                        "pool/opponent_macro_count_local": int(
                            (complete & ~trainable).sum().item()
                        ),
                    })
                metrics["pool/frozen_parameters_unchanged"] = True
            return self._finish_iteration(metrics=metrics, completed_games=validation.completed_games)
        except BaseException as error:
            self._poison(stage=stage, error=error)
            raise

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            **self._checkpoint_common(),
            "agent_resume_state_dict": self.agent_resume.state_dict(),
            "agent_state_dict": self.agent_merged.state_dict(),
            "agent_optimizer_state_dict": _optimizer_state(self.agent_optimizer),
            "merge_state": {
                "last_iteration": self.merge_updater.last_iteration,
                "update_count": self.merge_updater.update_count,
                "total_weight": self.merge_updater.total_weight,
            },
        }

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        destination = Path(path) if path is not None else self.checkpoint_dir / f"guandan-marvel-v2-ippo-iter{self.iteration:06d}.pt"
        return TrainingCheckpointStore.save(self.checkpoint_payload(), destination)

    def load_checkpoint_payload(self, payload: Mapping[str, object]):
        self._begin_load()
        try:
            decision = self._classify_checkpoint(payload, require_replay=False)
            self._verify_checkpoint_training_semantics(payload)
            self._verify_opponent_pool_checkpoint(payload)
            self.agent_resume.load_state_dict(payload["agent_resume_state_dict"], strict=True)
            self.agent_merged.load_state_dict(payload["agent_state_dict"], strict=True)
            _load_optimizer_state(self.agent_optimizer, payload["agent_optimizer_state_dict"])
            self._load_common_state(payload)
            self.merge_updater.set_reference_iteration(self.iteration)
            return decision
        except BaseException as error:
            self._poison(stage="checkpoint_load", error=error)
            raise

    def load_checkpoint(self, path: str | Path):
        payload = TrainingCheckpointStore.load(path, device=self.device)
        return self.load_checkpoint_payload(payload)


class VRPOTrainer(LocalTrainerBase):
    algorithm = "vrpo"
    q_target_contract = VRPO_Q_TARGET_CONTRACT

    def __init__(self, cfg: TrainConfig, *, game: object, run_dir: str | Path) -> None:
        super().__init__(cfg, game=game, run_dir=run_dir)
        self.agent_resume = Agent(cfg, enable_value_head=False).to(self.device)
        self.agent_merged = Agent(cfg, enable_value_head=False).to(self.device)
        self._build_symmetry(self.agent_resume)
        self.critic = CentralCritic(cfg, output_mode="qvalue").to(self.device)
        self.merge_updater = ExponentialMovingAverageUpdater(
            model=self.agent_resume, merged_model=self.agent_merged,
            beta=cfg.agent_merging_beta,
        )
        self.agent_model = self._prepare_trainable(self.agent_resume)
        self.critic_model = self._prepare_trainable(self.critic)
        self.agent_optimizer = self.agent_resume.configure_optimizer(
            lr=cfg.ppo.base_lr, optimizer=cfg.optimizer,
        )
        self.critic_optimizer = self.critic.configure_optimizer(
            lr=cfg.vrpo.base_lr, optimizer=cfg.optimizer,
        )
        replay_capacity = max(int(cfg.replay_buffer_scale * self.local_collect_batch_size), 1)
        per_trajectory = int(self.memory_budget["per_trajectory_bytes"])
        self.replay_memory_budget = {
            "capacity": replay_capacity,
            "bytes": replay_capacity * per_trajectory,
        }
        self.replay_buffer = TrajectoryBuffer(
            replay_capacity,
            BufferConfig(game=cfg.game, device=cfg.vrpo.replay_buffer_device),
        )
        self._activate_rank_rng()

    def _validate_replay_storage(self) -> None:
        mask = self.replay_buffer.storage["num_steps"] > 0
        if int(mask.sum().item()) != len(self.replay_buffer):
            raise ValueError("replay size does not match complete stored trajectory count")
        if bool(mask.any().item()):
            validate_complete_trajectories(
                self.replay_buffer.storage[mask], require_on_policy=True,
            )

    def train_iteration(self) -> dict[str, object]:
        self._begin_iteration()
        stage = "collect"
        try:
            schedule = compute_agent_schedule(self.cfg, iteration=self.iteration + 1)
            critic_lr = self.cfg.vrpo.base_lr * schedule.critic_lr_scale
            apply_learning_rate(self.agent_optimizer, schedule.agent_lr)
            apply_learning_rate(self.critic_optimizer, critic_lr)
            rollout, validation, infos, rollout_seconds = self._collect(self.agent_model)
            learner_started = time.perf_counter()
            stage = "fresh_critic_forward"
            with torch.no_grad():
                fresh_qvalues = self._forward_critic(rollout).qvalues.detach().float()
                ensure_finite_tensor("vrpo_fresh_qvalues", fresh_qvalues)
            self.replay_buffer.add(rollout.to(self.replay_buffer.device))
            self._validate_replay_storage()
            batch_size = int(rollout.batch_size[0])
            mini_batch_size = max(1, math.ceil(batch_size / self.cfg.mini_batches))
            actor_first = actor_last = None
            critic_first = critic_last = None
            actor_grad_norm = critic_grad_norm = 0.0
            stage = "policy_update"
            for epoch in range(self.cfg.ppo.agent_epoch):
                self.last_policy_epochs = epoch + 1
                indices = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, mini_batch_size):
                    mb_indices = indices[start:min(start + mini_batch_size, batch_size)]
                    mb = rollout[mb_indices]
                    output = self._forward_agent(mb)
                    with torch.no_grad():
                        gae = compute_gae_from_qvalue(
                            mb, fresh_qvalues[mb_indices],
                            avail_probs_new=output.avail_probs.detach().float(),
                            discount_gamma=self.cfg.discount_gamma,
                            gae_lambda=self.cfg.vrpo.gae_lambda,
                        )
                        ensure_finite_tensor("vrpo_actor_advantage", gae.player_advantage)
                    pg = compute_clipped_pg_loss(
                        mb,
                        clip_coef=schedule.clip_coef,
                        avail_probs_new=output.avail_probs.float(),
                        player_advantage=gae.player_advantage,
                    )
                    total = pg.pg_loss_mean + schedule.reg_coef * pg.kl_unif_mean
                    symmetry_record: dict[str, object] = {}
                    symmetry_out = self._symmetry_loss(mb, output)
                    if symmetry_out is not None:
                        total = total + self.cfg.symmetry.symmetry_weight * symmetry_out.loss
                        symmetry_record = {
                            "symmetry_loss": float(symmetry_out.loss.detach()),
                            "symmetry_mean_kl": float(symmetry_out.mean_kl_global),
                            "symmetry_token_count_global": int(
                                symmetry_out.token_count_global
                            ),
                            "symmetry_candidate_count_local": int(
                                symmetry_out.candidate_count_local
                            ),
                        }
                    actor_grad_norm = self._optimizer_step(
                        loss=total, module=self.agent_resume,
                        optimizer=self.agent_optimizer, name="vrpo_agent",
                    )
                    record = {
                        "pg_loss": float(pg.pg_loss_mean.detach()),
                        "kl_unif": float(pg.kl_unif_mean.detach()),
                        "kl_ref": float(pg.kl_ref_mean.detach()),
                        "clip_frac": float(pg.clip_frac.detach()),
                        "adv_mean": float(pg.adv_mean.detach()),
                        "adv_std": float(pg.adv_std.detach()),
                        **symmetry_record,
                    }
                    actor_first = record if actor_first is None else actor_first
                    actor_last = record

            stage = "critic_update"
            current = rollout
            critic_sources: list[str] = []
            for aux_epoch in range(self.cfg.vrpo.aux_epoch):
                critic_sources.append("fresh" if aux_epoch == 0 else "replay")
                validate_complete_trajectories(current, require_on_policy=True)
                output = self._forward_critic(current)
                qvalues = output.qvalues.float()
                with torch.no_grad():
                    policy = self._forward_agent(current).avail_probs.detach().float()
                    targets = compute_gae_from_qvalue(
                        current, qvalues.detach(),
                        avail_probs_new=policy,
                        discount_gamma=self.cfg.discount_gamma,
                        gae_lambda=self.cfg.vrpo.gae_lambda,
                    )
                    ensure_finite_tensor("vrpo_q_lambda_target", targets.qvalue_lbd_true)
                    ensure_finite_tensor("vrpo_q_td1_target", targets.qvalue_td1_true)
                qloss = compute_qvalue_loss(
                    current, qvalues,
                    qvalue_lbd_true=targets.qvalue_lbd_true,
                    qvalue_td1_true=targets.qvalue_td1_true,
                )
                critic_grad_norm = self._optimizer_step(
                    loss=qloss.qv_lbd_loss_mean, module=self.critic,
                    optimizer=self.critic_optimizer, name="vrpo_critic",
                )
                record = {
                    "qv_lbd_loss": float(qloss.qv_lbd_loss_mean.detach()),
                    "qv_td1_loss": float(qloss.qv_td1_loss_mean.detach()),
                }
                critic_first = record if critic_first is None else critic_first
                critic_last = record
                if aux_epoch + 1 < self.cfg.vrpo.aux_epoch:
                    current = self.replay_buffer.sample(batch_size).to(self.device)
            self.last_critic_batch_sources = tuple(critic_sources)
            self.merge_updater.step(iteration=self.iteration + 1)
            learner_seconds = time.perf_counter() - learner_started
            metrics = self._common_metrics(rollout, validation, infos, rollout_seconds)
            for prefix, values in (("agent_initial", actor_first), ("agent", actor_last), ("critic_initial", critic_first), ("critic", critic_last)):
                for key, value in (values or {}).items():
                    metrics[f"{prefix}/{key}"] = value
            metrics.update({
                "agent/learning_rate": schedule.agent_lr,
                "agent/clip_coef": schedule.clip_coef,
                "agent/reg_coef": schedule.reg_coef,
                "agent/grad_norm": actor_grad_norm,
                "critic/learning_rate": critic_lr,
                "critic/grad_norm": critic_grad_norm,
                "buffer/replay_buffer_size": len(self.replay_buffer),
                "memory/replay_buffer_bytes_estimate": int(self.replay_memory_budget["bytes"]),
                "perf/learner_seconds": learner_seconds,
            })
            return self._finish_iteration(metrics=metrics, completed_games=validation.completed_games)
        except BaseException as error:
            self._poison(stage=stage, error=error)
            raise

    def checkpoint_payload(self) -> dict[str, object]:
        self._require_checkpointable()
        self._validate_replay_storage()
        payload = {
            **self._checkpoint_common(),
            "agent_resume_state_dict": self.agent_resume.state_dict(),
            "agent_state_dict": self.agent_merged.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "agent_optimizer_state_dict": _optimizer_state(self.agent_optimizer),
            "critic_optimizer_state_dict": _optimizer_state(self.critic_optimizer),
            "merge_state": {
                "last_iteration": self.merge_updater.last_iteration,
                "update_count": self.merge_updater.update_count,
                "total_weight": self.merge_updater.total_weight,
            },
            "q_target_contract": VRPO_Q_TARGET_CONTRACT,
            "replay_persistence": "persisted" if self.cfg.checkpoint_replay_buffer else "omitted_warm_restart",
            "resume_classification": "exact_resume" if self.cfg.checkpoint_replay_buffer else "warm_restart",
        }
        if self.cfg.checkpoint_replay_buffer:
            payload["replay_buffer_state"] = self.replay_buffer.state_dict()
        return payload

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        destination = Path(path) if path is not None else self.checkpoint_dir / f"guandan-marvel-v2-vrpo-iter{self.iteration:06d}.pt"
        return TrainingCheckpointStore.save(self.checkpoint_payload(), destination)

    def load_checkpoint_payload(self, payload: Mapping[str, object]):
        self._begin_load()
        try:
            decision = self._classify_checkpoint(payload, require_replay=True)
            self._verify_checkpoint_training_semantics(payload)
            self._verify_opponent_pool_checkpoint(payload)
            self.agent_resume.load_state_dict(payload["agent_resume_state_dict"], strict=True)
            self.agent_merged.load_state_dict(payload["agent_state_dict"], strict=True)
            self.critic.load_state_dict(payload["critic_state_dict"], strict=True)
            _load_optimizer_state(self.agent_optimizer, payload["agent_optimizer_state_dict"])
            _load_optimizer_state(self.critic_optimizer, payload["critic_optimizer_state_dict"])
            self.replay_buffer.load_state_dict(payload["replay_buffer_state"])
            self._validate_replay_storage()
            self._load_common_state(payload)
            self.merge_updater.set_reference_iteration(self.iteration)
            return decision
        except BaseException as error:
            self._poison(stage="checkpoint_load", error=error)
            raise

    def load_checkpoint(self, path: str | Path):
        payload = TrainingCheckpointStore.load(path, device=self.device)
        return self.load_checkpoint_payload(payload)


__all__ = [
    "COMMON_RUNTIME_SOURCE_COMPONENTS", "CPP_BACKEND_SOURCE_COMPONENTS",
    "IPPOTrainer", "LocalTrainerBase", "M03_RUNTIME_SCHEMA",
    "PYTHON_BACKEND_SOURCE_COMPONENTS", "RUNTIME_SOURCE_COMPONENTS", "VRPOTrainer",
    "WARM_START_ALLOWED_ARCHITECTURE_TRANSITIONS",
    "runtime_source_components",
    "VRPO_Q_TARGET_CONTRACT",
]
