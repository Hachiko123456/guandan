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
from guandan_marvel.config import TrainConfig
from guandan_marvel.model import Agent, CentralCritic
from guandan_marvel.optim.common import ExponentialMovingAverageUpdater, apply_learning_rate
from guandan_marvel.runtime.compilation import ModuleCompiler
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
    compute_clipped_pg_loss, compute_qvalue_loss, compute_value_loss,
)
from guandan_marvel.training.metrics import rollout_metrics
from guandan_marvel.training.schedules import compute_agent_schedule
from guandan_marvel.training.validation import (
    ensure_finite_gradients, ensure_finite_tensor, validate_complete_trajectories,
)
from guandan_marvel.utils import set_global_seed

M03_RUNTIME_SCHEMA = "guandan-marvel-m03-local-runtime-v2"
VRPO_Q_TARGET_CONTRACT = "guandan-marvel-vrpo-q-lambda-td1-v1"
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
        "guandan_marvel/runtime/compilation.py", "guandan_marvel/runtime/logger.py",
        "guandan_marvel/runtime/sampler.py", "guandan_marvel/utils.py",
    ),
    "training": (
        "guandan_marvel/training/__init__.py", "guandan_marvel/training/checkpoints.py",
        "guandan_marvel/training/gae.py", "guandan_marvel/training/ippo.py",
        "guandan_marvel/training/local.py", "guandan_marvel/training/losses.py",
        "guandan_marvel/training/metrics.py", "guandan_marvel/training/schedules.py",
        "guandan_marvel/training/smoke.py", "guandan_marvel/training/validation.py",
        "guandan_marvel/training/vrpo.py",
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
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unavailable"


def _optimizer_state(optimizer) -> Mapping[str, object]:
    return optimizer.state_dict()


def _load_optimizer_state(optimizer, state: Mapping[str, object]) -> None:
    optimizer.load_state_dict(state)


def _finite_metrics(metrics: Mapping[str, object]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"metric {key!r} is non-finite")


class LocalTrainerBase:
    algorithm: str
    q_target_contract: str | None = None

    def __init__(self, cfg: TrainConfig, *, game: object, run_dir: str | Path) -> None:
        self.cfg = cfg
        self.game = game
        self.device = torch.device(cfg.device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("M03 local runtime supports cpu/cuda devices only")
        if cfg.ddp.enabled:
            raise ValueError("M03 local smoke does not enable DDP")
        if cfg.amp_enabled:
            raise ValueError("AMP is explicitly disabled for the first M03 smoke profile")
        if cfg.max_sampler_bytes_per_rank is None:
            raise ValueError("max_sampler_bytes_per_rank is required before sampler allocation")
        if cfg.game.identifier != getattr(game, "identifier"):
            raise ValueError("TrainConfig.game and runtime game identifier differ")
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
            world_size=1,
            max_sampler_bytes_per_rank=cfg.max_sampler_bytes_per_rank,
        )
        self.replay_memory_budget: dict[str, object] | None = None
        set_global_seed(cfg.seed)
        self.iteration = 0
        self.durable_updates = 0
        self.learner_updates = 0
        self.durable_learner_updates = 0
        self.completed_games_total = 0
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
        self.evaluation_hook = None
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
        self.sampler = Sampler(config=SamplerConfig(
            game=game,
            device=self.device,
            rollout_batch_size=cfg.rollout_batch_size,
            collect_batch_size=cfg.collect_batch_size,
            sampling_temperature=1.0,
            purpose="on_policy",
            env_seed=cfg.seed,
            data_domain=cfg.training_data_domain,
            data_manifest_hash=cfg.training_data_manifest_hash,
            data_generator_version=cfg.training_data_generator_version,
        ))

    @property
    def merged_policy_identity(self) -> str:
        return "current_policy_copy" if self.cfg.agent_merging_beta == 0.0 else "exponential_moving_average"

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

    def _require_healthy(self) -> None:
        if self.failed or self.iteration_status == "failed":
            raise RuntimeError(f"trainer is unusable after failed iteration/load: {self.failure}")
        if self.sampler.failed:
            raise RuntimeError(f"trainer sampler is unusable: {self.sampler.failure_info}")

    def _begin_iteration(self) -> None:
        self._require_healthy()
        if self.iteration_status in {"running", "loading"} or self.active_iteration is not None:
            raise RuntimeError("trainer already has an active operation")
        self.iteration_status = "running"
        self.active_iteration = self.iteration + 1
        self._iteration_start_learner_updates = self.learner_updates

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
        started = time.perf_counter()
        rollout = self.sampler.collect(agent_model=agent_model)
        seconds = time.perf_counter() - started
        validation = validate_complete_trajectories(rollout, require_on_policy=True)
        terminal_infos = self.sampler.pop_terminal_infos(validation.completed_games)
        self.last_trajectory_keys = tuple(sorted(rollout.keys()))
        self.last_trajectory_num_steps = tuple(int(value) for value in rollout["num_steps"].tolist())
        return rollout, validation, terminal_infos, seconds

    def _optimizer_step(self, *, loss: torch.Tensor, module: nn.Module, optimizer, name: str) -> float:
        ensure_finite_tensor(f"{name}_loss", loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        ensure_finite_gradients(module, name=name)
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
            "runtime/smoke_profile": True,
            "runtime/reduced_horizon": int(self.cfg.game.max_game_horizon),
            "runtime/merged_policy_identity": self.merged_policy_identity,
            "runtime/optimizer": "muon+adamw" if self.cfg.optimizer.use_muon else "adamw",
            "runtime/compile_enabled": self.cfg.compile.enabled,
            "runtime/amp_enabled": self.cfg.amp_enabled,
            "memory/sampler_bytes_per_rank_estimate": int(self.memory_budget["sampler_bytes_per_rank"]),
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
        return {
            "schema": TRAINING_CHECKPOINT_SCHEMA,
            "algorithm": self.algorithm,
            "architecture": self.cfg.model.architecture,
            "protocol_version": self.cfg.game.protocol_version,
            "backend": self.backend_name,
            "backend_identity": self.backend_identity,
            "semantic_contract": self.semantic_contract,
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
            },
            "rng_state": capture_rng_state(),
            "sampler_state": self.sampler.state_dict(),
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

    def _load_common_state(self, payload: Mapping[str, object]) -> None:
        self.iteration = int(payload["iteration"])
        self.durable_updates = int(payload["durable_updates"])
        self.learner_updates = int(payload.get("learner_updates", self.durable_updates))
        self.durable_learner_updates = int(payload.get("durable_learner_updates", self.learner_updates))
        counters = payload["counters"]
        self.completed_games_total = int(counters["completed_games_total"])
        self.sampler.load_state_dict(payload["sampler_state"])
        restore_rng_state(payload["rng_state"])
        self.iteration_status = "completed"
        self.active_iteration = None
        self.failed = False
        self.failure = None



class IPPOTrainer(LocalTrainerBase):
    algorithm = "ippo"

    def __init__(self, cfg: TrainConfig, *, game: object, run_dir: str | Path) -> None:
        super().__init__(cfg, game=game, run_dir=run_dir)
        self.agent_resume = Agent(cfg, enable_value_head=True).to(self.device)
        self.agent_merged = Agent(cfg, enable_value_head=True).to(self.device)
        self.merge_updater = ExponentialMovingAverageUpdater(
            model=self.agent_resume,
            merged_model=self.agent_merged,
            beta=cfg.agent_merging_beta,
        )
        self.agent_model = self._compile(self.agent_resume)
        self.agent_optimizer = self.agent_resume.configure_optimizer(
            lr=cfg.ppo.base_lr, optimizer=cfg.optimizer,
        )

    def train_iteration(self) -> dict[str, object]:
        self._begin_iteration()
        stage = "collect"
        try:
            schedule = compute_agent_schedule(self.cfg, iteration=self.iteration + 1)
            apply_learning_rate(self.agent_optimizer, schedule.agent_lr)
            rollout, validation, infos, rollout_seconds = self._collect(self.agent_model)
            learner_started = time.perf_counter()
            stage = "gae"
            with torch.no_grad():
                initial = self.agent_model(rollout=rollout)
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
            stage = "policy_update"
            for epoch in range(self.cfg.ppo.agent_epoch):
                self.last_policy_epochs = epoch + 1
                indices = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, mini_batch_size):
                    mb_indices = indices[start:min(start + mini_batch_size, batch_size)]
                    mb = rollout[mb_indices]
                    output = self.agent_model(rollout=mb)
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
                    grad_norm = self._optimizer_step(
                        loss=total, module=self.agent_resume,
                        optimizer=self.agent_optimizer, name="ippo_agent",
                    )
                    record = {
                        "pg_loss": float(pg.pg_loss_mean.detach()),
                        "value_loss": float(value.value_loss_mean.detach()),
                        "kl_unif": float(pg.kl_unif_mean.detach()),
                        "kl_ref": float(pg.kl_ref_mean.detach()),
                        "clip_frac": float(pg.clip_frac.detach()),
                        "adv_mean": float(pg.adv_mean.detach()),
                        "adv_std": float(pg.adv_std.detach()),
                    }
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
                "perf/learner_seconds": learner_seconds,
            })
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

    def load_checkpoint(self, path: str | Path):
        self._begin_load()
        try:
            payload = TrainingCheckpointStore.load(path, device=self.device)
            decision = self._classify_checkpoint(payload, require_replay=False)
            self.agent_resume.load_state_dict(payload["agent_resume_state_dict"], strict=True)
            self.agent_merged.load_state_dict(payload["agent_state_dict"], strict=True)
            _load_optimizer_state(self.agent_optimizer, payload["agent_optimizer_state_dict"])
            self._load_common_state(payload)
            self.merge_updater.set_reference_iteration(self.iteration)
            return decision
        except BaseException as error:
            self._poison(stage="checkpoint_load", error=error)
            raise


class VRPOTrainer(LocalTrainerBase):
    algorithm = "vrpo"
    q_target_contract = VRPO_Q_TARGET_CONTRACT

    def __init__(self, cfg: TrainConfig, *, game: object, run_dir: str | Path) -> None:
        super().__init__(cfg, game=game, run_dir=run_dir)
        self.agent_resume = Agent(cfg, enable_value_head=False).to(self.device)
        self.agent_merged = Agent(cfg, enable_value_head=False).to(self.device)
        self.critic = CentralCritic(cfg, output_mode="qvalue").to(self.device)
        self.merge_updater = ExponentialMovingAverageUpdater(
            model=self.agent_resume, merged_model=self.agent_merged,
            beta=cfg.agent_merging_beta,
        )
        self.agent_model = self._compile(self.agent_resume)
        self.critic_model = self._compile(self.critic)
        self.agent_optimizer = self.agent_resume.configure_optimizer(
            lr=cfg.ppo.base_lr, optimizer=cfg.optimizer,
        )
        self.critic_optimizer = self.critic.configure_optimizer(
            lr=cfg.vrpo.base_lr, optimizer=cfg.optimizer,
        )
        replay_capacity = max(int(cfg.replay_buffer_scale * cfg.collect_batch_size), 1)
        per_trajectory = int(self.memory_budget["per_trajectory_bytes"])
        self.replay_memory_budget = {
            "capacity": replay_capacity,
            "bytes": replay_capacity * per_trajectory,
        }
        self.replay_buffer = TrajectoryBuffer(
            replay_capacity,
            BufferConfig(game=cfg.game, device=cfg.vrpo.replay_buffer_device),
        )

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
                fresh_qvalues = self.critic_model(rollout=rollout).qvalues.detach().float()
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
                    output = self.agent_model(rollout=mb)
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
                    }
                    actor_first = record if actor_first is None else actor_first
                    actor_last = record

            stage = "critic_update"
            current = rollout
            critic_sources: list[str] = []
            for aux_epoch in range(self.cfg.vrpo.aux_epoch):
                critic_sources.append("fresh" if aux_epoch == 0 else "replay")
                validate_complete_trajectories(current, require_on_policy=True)
                output = self.critic_model(rollout=current)
                qvalues = output.qvalues.float()
                with torch.no_grad():
                    policy = self.agent_model(rollout=current).avail_probs.detach().float()
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

    def load_checkpoint(self, path: str | Path):
        self._begin_load()
        try:
            payload = TrainingCheckpointStore.load(path, device=self.device)
            decision = self._classify_checkpoint(payload, require_replay=True)
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


__all__ = [
    "COMMON_RUNTIME_SOURCE_COMPONENTS", "CPP_BACKEND_SOURCE_COMPONENTS",
    "IPPOTrainer", "LocalTrainerBase", "M03_RUNTIME_SCHEMA",
    "PYTHON_BACKEND_SOURCE_COMPONENTS", "RUNTIME_SOURCE_COMPONENTS", "VRPOTrainer",
    "runtime_source_components",
    "VRPO_Q_TARGET_CONTRACT",
]
