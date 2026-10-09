"""M05 repair benchmark with a real frozen Transformer Agent rollout."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil
import torch

from guandan_marvel import cpp_game, python_game
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.env_batch import EnvStepBatch
from guandan_marvel.model import Agent
from guandan_marvel.runtime.sampler import Sampler, SamplerConfig
from guandan_marvel.testing.differential import (
    DifferentialCase, build_case, cpp_batch_from_case, python_batch_from_case,
)
from scripts.benchmark_marvel_m05 import benchmark_full_iteration

SCHEMA = "guandan-marvel-m05-repair-benchmark-v1"
REAL_AGENT_GATE = 2.0


@dataclass(frozen=True, slots=True)
class GameView:
    module: Any
    max_game_horizon: int = 512

    def __getattr__(self, name: str):
        return getattr(self.module, name)

    @property
    def BACKEND(self) -> str:
        return str(self.module.BACKEND)

    @property
    def identifier(self) -> str:
        return f"{self.module.identifier}-m05-repair-T{self.max_game_horizon}"

    def actions_from_np(self, rows):
        return self.module.actions_from_np(rows)


class RecordingEnv:
    def __init__(self, inner: Any):
        self.inner = inner
        self.digest = hashlib.sha256()
        self.action_count = 0

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def step(self, actions):
        for action in actions:
            row = np.asarray(action.row, dtype=np.int64)
            self.digest.update(row.tobytes())
            self.action_count += 1
        result = self.inner.step(actions)
        self.digest.update(np.asarray(result.rewards, dtype=np.float32).tobytes())
        self.digest.update(np.asarray(result.is_terminal, dtype=np.bool_).tobytes())
        return result


def build_config() -> TrainConfig:
    cfg = TrainConfig(
        game=python_game.GAME_SPEC, device="cpu", algo_name="ippo",
        logging_dir="runs/M05_differential_performance_repair/benchmark",
        run_name="frozen-transformer", seed=91_001,
        rollout_batch_size=1, collect_batch_size=0, replay_buffer_scale=1,
        mini_batches=1, total_iterations=1,
        max_sampler_bytes_per_rank=256 * 2**20,
        checkpoint_replay_buffer=False, amp_enabled=False,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg


def _batch(backend: str, case: DifferentialCase):
    if backend == "python_reference":
        return python_batch_from_case(case, auto_reset=False)
    return cpp_batch_from_case(case, auto_reset=False)


def run_agent_lane(
    backend: str,
    *,
    cases: list[DifferentialCase],
    agent: Agent,
) -> dict[str, Any]:
    module = python_game if backend == "python_reference" else cpp_game
    game = GameView(module)
    timings: dict[str, float] = {}
    timing_counts: dict[str, int] = {}

    def record(name: str, seconds: float) -> None:
        timings[name] = timings.get(name, 0.0) + float(seconds)
        timing_counts[name] = timing_counts.get(name, 0) + 1

    trajectory_digest = hashlib.sha256()
    total_actions = 0
    completed_games = 0
    setup_seconds = 0.0
    cache_prepare_seconds = 0.0
    rollout_started = time.perf_counter()
    for case in cases:
        setup_started = time.perf_counter()
        env, step = _batch(backend, case)
        recording = RecordingEnv(env)
        config = SamplerConfig(
            game=game, device="cpu", rollout_batch_size=1,
            collect_batch_size=0, sampling_temperature=0.0,
            purpose="evaluation", env_seed=case.seed,
            timing_callback=record,
        )
        sampler = Sampler(
            config=config, envs=recording,
            parsed_envs_step=EnvStepBatch.from_env_step(step, config),
        )
        setup_seconds += time.perf_counter() - setup_started
        cache_started = time.perf_counter()
        sampler.prepare_agent_cache(agent_model=agent)
        cache_prepare_seconds += time.perf_counter() - cache_started
        with torch.inference_mode():
            for _ in range(game.max_game_horizon):
                done = sampler.rollout_step(agent_model=agent)
                if bool(done[0]):
                    break
            else:
                raise RuntimeError(f"case {case.game_index} exceeded repair benchmark horizon")
        trajectory_digest.update(recording.digest.digest())
        total_actions += recording.action_count
        completed_games += 1
        del sampler, recording, env, step
    rollout_seconds = time.perf_counter() - rollout_started
    return {
        "backend": backend,
        "completed_games": completed_games,
        "actions": total_actions,
        "trajectory_digest": trajectory_digest.hexdigest(),
        "rollout_seconds": rollout_seconds,
        "actions_per_second": total_actions / rollout_seconds,
        "segments_seconds": {
            "environment_setup": setup_seconds,
            "cache_prepare": cache_prepare_seconds,
            **timings,
        },
        "segment_call_counts": timing_counts,
    }


def measured_lane(backend: str, *, cases: list[DifferentialCase], agent: Agent) -> dict[str, Any]:
    gc.collect()
    process = psutil.Process()
    cpu_before = process.cpu_times()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    result = run_agent_lane(backend, cases=cases, agent=agent)
    wall = time.perf_counter() - started
    cpu_after = process.cpu_times()
    result.update({
        "measured_wall_seconds": wall,
        "process_cpu_seconds": (
            cpu_after.user - cpu_before.user + cpu_after.system - cpu_before.system
        ),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": process.memory_info().rss,
    })
    return result


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    median = statistics.median(sample["measured_wall_seconds"] for sample in samples)
    representative = min(samples, key=lambda sample: abs(sample["measured_wall_seconds"] - median))
    return {
        "samples": samples,
        "median_wall_seconds": median,
        "median_actions_per_second": statistics.median(
            sample["actions"] / sample["measured_wall_seconds"] for sample in samples
        ),
        "representative_segments_seconds": representative["segments_seconds"],
        "completed_games": representative["completed_games"],
        "actions": representative["actions"],
        "trajectory_digest": representative["trajectory_digest"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.games <= 0 or args.repetitions < 2:
        parser.error("games must be positive and repetitions must be at least 2")

    torch.set_num_threads(1)
    random.seed(91_001)
    np.random.seed(91_001)
    torch.manual_seed(91_001)
    cfg = build_config()
    agent = Agent(cfg, enable_value_head=False).eval()
    cases = [build_case(50_000 + index) for index in range(args.games)]

    # Warm both backends outside measured samples.
    run_agent_lane("python_reference", cases=cases[:1], agent=agent)
    run_agent_lane("cpp", cases=cases[:1], agent=agent)

    collected = {"python_reference": [], "cpp": []}
    orders = []
    for repetition in range(args.repetitions):
        order = ["python_reference", "cpp"] if repetition % 2 == 0 else ["cpp", "python_reference"]
        orders.append(order)
        for backend in order:
            collected[backend].append(measured_lane(backend, cases=cases, agent=agent))
    python_summary = summarize(collected["python_reference"])
    cpp_summary = summarize(collected["cpp"])
    digests = {
        sample["trajectory_digest"]
        for samples in collected.values() for sample in samples
    }
    action_counts = {
        sample["actions"] for samples in collected.values() for sample in samples
    }
    if len(digests) != 1 or len(action_counts) != 1:
        raise AssertionError("real Agent benchmark trajectories are not matched")
    speedup = python_summary["median_wall_seconds"] / cpp_summary["median_wall_seconds"]

    original = json.loads((
        REPO_ROOT / "runs/M05_differential_performance/20261009_acceptance/benchmark.json"
    ).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    full_iteration = benchmark_full_iteration(run_dir / "full_iteration")
    result = {
        "schema": SCHEMA,
        "stage": "M05_differential_performance_repair",
        "timestamp_unix": time.time(),
        "workload": {
            "model": {
                "architecture": cfg.model.architecture,
                "history_dim": cfg.model.history_dim,
                "feature_dim": cfg.model.feature_dim,
                "decoder_layers": cfg.model.decoder_layers,
                "decoder_heads": cfg.model.decoder_heads,
                "state_layers": cfg.model.state_layers,
                "action_layers": cfg.model.action_layers,
                "fuse_layers": cfg.model.fuse_layers,
            },
            "mode": "eval+torch.inference_mode",
            "sampling": "greedy temperature=0",
            "same_frozen_weights": True,
            "same_materialized_deals": True,
            "same_device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
            "games": args.games,
            "repetitions": args.repetitions,
            "alternating_order": orders,
        },
        "real_frozen_transformer_rollout": {
            "python_reference": python_summary,
            "cpp": cpp_summary,
            "speedup": speedup,
            "threshold": REAL_AGENT_GATE,
            "passes_2x": speedup >= REAL_AGENT_GATE,
            "matched_trajectory_digest": next(iter(digests)),
            "matched_actions": next(iter(action_counts)),
        },
        "complete_ippo_iteration": full_iteration,
        "historical_deterministic_selector": {
            "source": "runs/M05_differential_performance/20261009_acceptance/benchmark.json",
            "historical_lane_name": "end_to_end_frozen_policy_rollout",
            "corrected_description": "deterministic selector / fixed-action-rule rollout; no Transformer Agent forward pass",
            "speedup": original["lanes"]["end_to_end_rollout"]["speedup"],
            "retained": True,
        },
        "acceptance_decision_point": {
            "interpretation": "The master M05 plan lists end-to-end rollout within the >=2x performance comparison. The repair therefore treats the real frozen Transformer rollout as required for conservative re-acceptance; learner computation itself is not required to accelerate 2x.",
            "real_agent_2x_required": True,
            "passes": speedup >= REAL_AGENT_GATE,
        },
        "valid": speedup >= REAL_AGENT_GATE,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "speedup": speedup,
        "passes_2x": speedup >= REAL_AGENT_GATE,
        "matched_actions": next(iter(action_counts)),
        "full_iteration_speedup": full_iteration["speedup"],
    }, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
