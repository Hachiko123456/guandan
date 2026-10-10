# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from guandan_marvel.config import DDPConfig, GameSpec, ModelConfig, TrainConfig


def build_train_config(
    profile: Mapping[str, object],
    *,
    game: object,
    run_root: str | Path,
    local_cpu_test: bool = False,
) -> TrainConfig:
    training = profile.get("training")
    ddp = profile.get("ddp")
    if not isinstance(training, Mapping) or not isinstance(ddp, Mapping):
        raise ValueError("training and ddp profile objects are required")
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
        rollout_batch_size=int(training["rollout_batch_size"]),
        collect_batch_size=int(training["collect_batch_size"]),
        replay_buffer_scale=int(training["replay_buffer_scale"]),
        mini_batches=int(training["mini_batches"]),
        total_iterations=int(training.get("target_iterations", training.get("additional_iterations", 1))),
        checkpoint_interval=int(training["checkpoint_interval"]),
        evaluation_interval=int(training["evaluation_interval"]),
        max_sampler_bytes_per_rank=int(training["max_sampler_bytes_per_rank"]),
        amp_enabled=bool(training.get("amp_enabled", False)),
        checkpoint_replay_buffer=True,
        model=ModelConfig(**dict(model_payload)),
        ddp=DDPConfig(
            enabled=True,
            backend=backend,
            require_cuda=require_cuda,
            find_unused_parameters=bool(ddp.get("find_unused_parameters", False)),
        ),
    )
    config.ppo.agent_epoch = int(training.get("agent_epoch", 1))
    config.vrpo.aux_epoch = int(training.get("critic_aux_epoch", 1))
    config.optimizer.use_muon = bool(training.get("use_muon", False))
    config.compile.enabled = bool(training.get("compile_enabled", False))
    return config


__all__ = ["build_train_config"]
