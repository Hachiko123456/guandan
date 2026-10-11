# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from guandan_marvel.config import (
    DDPConfig, GameSpec, ModelConfig, TrainConfig, deserialize_train_config,
)
from guandan_marvel.training.checkpoints import NON_SEMANTIC_CONFIG_PATHS, ResumeDecision

RESUME_CONTROL_FIELDS = frozenset({
    "additional_iterations", "target_iterations", "checkpoint_interval", "evaluation_interval",
    "max_wall_seconds", "checkpoint_keep_last", "evaluation_keep_last",
})


def _training_profile(profile: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    training = profile.get("training")
    ddp = profile.get("ddp")
    if not isinstance(training, Mapping) or not isinstance(ddp, Mapping):
        raise ValueError("training and ddp profile objects are required")
    return training, ddp


def build_train_config(
    profile: Mapping[str, object],
    *,
    game: object,
    run_root: str | Path,
    local_cpu_test: bool = False,
) -> TrainConfig:
    """Build a new pilot/full trainer; resume uses build_resume_train_config."""
    if profile.get("mode") == "resume":
        raise ValueError("resume configuration must be inherited from a validated checkpoint")
    training, ddp = _training_profile(profile)
    model_payload = training.get("model")
    if not isinstance(model_payload, Mapping):
        raise ValueError("training.model is required")
    algorithm = str(profile.get("algorithm", ""))
    if algorithm not in {"ippo", "vrpo"}:
        raise ValueError(f"unsupported training algorithm {algorithm!r}")
    device = "cpu" if local_cpu_test else "cuda"
    backend = "gloo" if local_cpu_test else str(ddp.get("backend", "nccl"))
    require_cuda = False if local_cpu_test else bool(ddp.get("require_cuda", True))
    config = TrainConfig(
        game=GameSpec.from_game(game),
        device=device,
        algo_name=algorithm,
        run_name=f"m07-{profile['mode']}-{algorithm}",
        logging_dir=str(Path(run_root) / "rank-runtime"),
        seed=int(training["seed"]),
        environment_rank_seed_stride=int(training.get("environment_rank_seed_stride", 1_000_003)),
        training_data_generator_version=str(training.get(
            "training_data_generator_version", "guandan-marvel-env-batch-continuous-prng-v1",
        )),
        rollout_batch_size=int(training["rollout_batch_size"]),
        collect_batch_size=int(training["collect_batch_size"]),
        replay_buffer_scale=int(training["replay_buffer_scale"]),
        mini_batches=int(training["mini_batches"]),
        total_iterations=int(training.get("target_iterations", 1)),
        checkpoint_interval=int(training["checkpoint_interval"]),
        evaluation_interval=int(training["evaluation_interval"]),
        max_sampler_bytes_per_rank=int(training["max_sampler_bytes_per_rank"]),
        amp_enabled=bool(training.get("amp_enabled", False)),
        checkpoint_replay_buffer=True,
        model=ModelConfig(**dict(model_payload)),
        ddp=DDPConfig(
            enabled=int(ddp.get("world_size", 0)) > 1,
            backend=backend,
            require_cuda=require_cuda,
            find_unused_parameters=bool(ddp.get("find_unused_parameters", False)),
        ),
    )
    config.ppo.agent_epoch = int(training.get("agent_epoch", 1))
    config.vrpo.aux_epoch = int(training.get("critic_aux_epoch", 1))
    config.agent_merging_beta = float(training.get("agent_merging_beta", 0.0))
    config.optimizer.use_muon = bool(training.get("use_muon", False))
    config.compile.enabled = bool(training.get("compile_enabled", False))
    return config


def build_resume_train_config(
    profile: Mapping[str, object],
    *,
    checkpoint_payload: Mapping[str, object],
    game: object,
    run_root: str | Path,
) -> TrainConfig:
    """Inherit every training-semantic field from the validated rank shard."""
    if profile.get("mode") != "resume":
        raise ValueError("build_resume_train_config requires mode=resume")
    controls = profile.get("resume")
    if not isinstance(controls, Mapping):
        raise ValueError("resume profile requires a resume control object")
    unknown = sorted(set(controls) - RESUME_CONTROL_FIELDS)
    if unknown:
        raise ValueError(f"resume profile contains forbidden semantic overrides: {unknown}")
    additional = int(controls.get("additional_iterations", 0) or 0)
    target_iteration = int(controls.get("target_iterations", 0) or 0)
    if (additional > 0) == (target_iteration > 0):
        raise ValueError("resume must set exactly one of additional_iterations or target_iterations")
    cfg = deserialize_train_config(checkpoint_payload.get("resolved_config"))
    if cfg is None:
        raise ValueError("rank shard resolved_config cannot be decoded")
    algorithm = str(checkpoint_payload.get("algorithm", ""))
    if algorithm not in {"ippo", "vrpo"} or cfg.algo_name != algorithm:
        raise ValueError("rank shard algorithm/config identity mismatch")
    expected_algorithm = str(profile.get("expected_algorithm", ""))
    if expected_algorithm and algorithm != expected_algorithm:
        raise ValueError(
            f"recovery algorithm {algorithm!r} does not match expected_algorithm={expected_algorithm!r}"
        )
    runtime_spec = GameSpec.from_game(game)
    if cfg.game.identifier != runtime_spec.identifier or cfg.game.protocol_version != runtime_spec.protocol_version:
        raise ValueError(
            "recovery game identity is incompatible with this runtime: "
            f"checkpoint={cfg.game.identifier}/{cfg.game.protocol_version}, "
            f"runtime={runtime_spec.identifier}/{runtime_spec.protocol_version}"
        )
    if cfg.game != runtime_spec:
        raise ValueError("recovery game tensor/horizon contract differs from this runtime")
    checkpoint_iteration = int(checkpoint_payload.get("iteration", -1))
    if checkpoint_iteration < 0:
        raise ValueError("rank shard iteration is invalid")
    cfg.run_name = f"m07-resume-{algorithm}"
    cfg.logging_dir = str(Path(run_root) / "rank-runtime")
    cfg.resume_from = "validated-ddp-recovery"
    cfg.total_iterations = checkpoint_iteration + additional if additional > 0 else target_iteration
    if "checkpoint_interval" in controls:
        cfg.checkpoint_interval = int(controls["checkpoint_interval"])
    if "evaluation_interval" in controls:
        cfg.evaluation_interval = int(controls["evaluation_interval"])
    return cfg


def require_strict_resume_decision(decision: ResumeDecision) -> ResumeDecision:
    if decision.kind == "exact_resume":
        return decision
    if decision.kind != "compatible_resume":
        raise RuntimeError(
            f"recovery is not trainable: kind={decision.kind}; " + "; ".join(decision.reasons)
        )
    allowed = set(NON_SEMANTIC_CONFIG_PATHS)
    bad = []
    for reason in decision.reasons:
        path = str(reason).split(":", 1)[0].split(".", 1)[0]
        if path not in allowed:
            bad.append(str(reason))
    if bad:
        raise RuntimeError("compatible resume contained non-run-control differences: " + "; ".join(bad))
    return decision


__all__ = [
    "RESUME_CONTROL_FIELDS", "build_resume_train_config", "build_train_config",
    "require_strict_resume_decision",
]
