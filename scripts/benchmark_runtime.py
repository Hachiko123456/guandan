"""Bounded local runtime benchmark; never reports remote or GPU acceptance."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan.cards import Rank
from guandan.env_batch import GuandanEnvBatch
from guandan.environment_types import GameConfig
from guandan.training.trainer import train


def bench_batch(*, transactional: bool, envs: int, ticks: int) -> dict:
    batch = GuandanEnvBatch(
        [GameConfig(seed=70_000 + index, level_rank=Rank.FIVE) for index in range(envs)],
        auto_reset=False, transactional=transactional,
    )
    observation = batch.reset(seeds=[70_000 + index for index in range(envs)])
    started = time.perf_counter()
    completed = 0
    for _ in range(ticks):
        actions = observation.legal_next_tokens[:, 0].astype("int32", copy=True)
        observation = batch.step(actions).observation
        completed += envs
        # A terminal slot is not expected in this short microbenchmark; if it is,
        # reset it explicitly so the benchmark measures the batch boundary.
        for index, env in enumerate(batch.envs):
            if env.done or env.truncated:
                row = batch.reset_at(index, seed=71_000 + index + completed)
                observation.player_id[index] = row.player_id
                observation.team_id[index] = row.team_id
                observation.token_step[index] = row.token_step
                observation.committed_step[index] = row.committed_step
                observation.observation_tokens[index] = row.observation_tokens
                observation.state_channels[index] = row.state_channels
                observation.legal_next_tokens[index] = row.legal_next_tokens
                observation.legal_next_mask[index] = row.legal_next_mask
                observation.legal_next_is_commit[index] = row.legal_next_is_commit
                observation.legal_next_kinds[index] = row.legal_next_kinds
                observation.legal_next_values[index] = row.legal_next_values
                observation.present[index] = True
    elapsed = time.perf_counter() - started
    return {"transactional": transactional, "envs": envs, "ticks": ticks,
            "token_steps": completed, "elapsed_seconds": elapsed,
            "token_steps_per_second": completed / elapsed if elapsed else 0.0}


def bench_train(*, device: str, updates: int, rollout_envs: int, rollout_steps: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="guandan-benchmark-") as directory:
        started = time.perf_counter()
        result = train(
            algorithm="ippo", profile="local_fast", device=device, updates=updates,
            rollout_envs=rollout_envs, rollout_steps=rollout_steps, hidden_dim=16,
            checkpoint_dir=directory, seed=10_017,
        )
        elapsed = time.perf_counter() - started
        return {
            "status": result.status, "device": device, "updates": result.updates,
            "token_steps": result.total_token_steps, "wall_seconds": elapsed,
            "token_steps_per_second": result.total_token_steps / elapsed if elapsed else 0.0,
            "last_loss": result.last_loss,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=8)
    args = parser.parse_args(argv)
    result = {"format": "guandan-runtime-benchmark-v1", "generated_at": time.time(),
              "batch": [bench_batch(transactional=value, envs=args.envs, ticks=args.ticks) for value in (True, False)],
              "training": bench_train(device=args.device, updates=args.updates,
                                        rollout_envs=args.envs, rollout_steps=args.rollout_steps)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
