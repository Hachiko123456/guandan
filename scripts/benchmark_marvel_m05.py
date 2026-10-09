"""Matched-workload M05 performance benchmark for Python and C++ backends."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil
import torch

from guandan.action_state import StepwiseActionState
from guandan.environment import GuandanEnv
from guandan.environment_types import GameConfig
from guandan_marvel import python_game
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.env_batch import EnvStepBatch
from guandan_marvel.runtime.sampler import Sampler, SamplerConfig
from guandan_marvel.testing.differential import (
    DifferentialCase,
    build_case,
    cpp_batch_from_case,
    previous_result_from_dict,
    python_batch_from_case,
    round_state_from_case,
)
from guandan_marvel.training.local import IPPOTrainer
from nanopokerlib import guandan as cpp_game

BENCHMARK_SCHEMA = "guandan-marvel-m05-benchmark-v1"
THRESHOLD = 2.0


@dataclass(frozen=True, slots=True)
class PreparedTrajectory:
    case: DifferentialCase
    rows: tuple[np.ndarray, ...]


class PeakRssSampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRssSampler":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()


def nvidia_inventory() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    records = []
    for line in output.splitlines():
        if not line.strip():
            continue
        name, memory, driver = [value.strip() for value in line.split(",", 2)]
        records.append({"name": name, "memory_mib": memory, "driver": driver})
    return records


def measure(function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    gc.collect()
    process = psutil.Process()
    cpu_before = process.cpu_times()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    with PeakRssSampler() as memory:
        payload = function()
    wall = time.perf_counter() - started
    cpu_after = process.cpu_times()
    return {
        "wall_seconds": wall,
        "process_cpu_seconds": (
            cpu_after.user - cpu_before.user + cpu_after.system - cpu_before.system
        ),
        "rss_before_bytes": rss_before,
        "peak_rss_bytes": memory.peak,
        "rss_delta_peak_bytes": max(0, memory.peak - rss_before),
        "payload": payload,
    }


def summarize_samples(samples: list[dict[str, Any]], *, work: int, unit: str) -> dict[str, Any]:
    walls = [sample["wall_seconds"] for sample in samples]
    median_wall = statistics.median(walls)
    chosen = min(samples, key=lambda sample: abs(sample["wall_seconds"] - median_wall))
    return {
        "repetitions": len(samples),
        "work": work,
        "unit": unit,
        "wall_seconds_median": median_wall,
        "wall_seconds_samples": walls,
        "throughput_per_second": work / median_wall,
        "process_cpu_seconds_median": statistics.median(
            sample["process_cpu_seconds"] for sample in samples
        ),
        "peak_rss_bytes_max": max(sample["peak_rss_bytes"] for sample in samples),
        "rss_delta_peak_bytes_max": max(sample["rss_delta_peak_bytes"] for sample in samples),
        "segments_seconds": chosen["payload"].get("segments_seconds", {}),
        "result": chosen["payload"],
    }


def prepare_trajectories(count: int, *, start_index: int = 30_000) -> list[PreparedTrajectory]:
    prepared = []
    for offset in range(count):
        case = build_case(start_index + offset)
        batch, step = python_batch_from_case(case, auto_reset=False)
        rng = np.random.default_rng(31_000 + offset)
        rows = []
        for _ in range(python_game.max_game_horizon):
            if bool(step.is_terminal[0]):
                break
            legal = np.flatnonzero(step.available_actions.weights[0] > 0)
            row = step.available_actions.tokens[0, int(rng.choice(legal))].copy()
            rows.append(row)
            step = batch.step(row[None, :])
        else:
            raise RuntimeError("benchmark trajectory did not reach terminal")
        prepared.append(PreparedTrajectory(case, tuple(rows)))
    return prepared


def replay_single(
    backend: str,
    trajectories: list[PreparedTrajectory],
) -> dict[str, Any]:
    setup_seconds = 0.0
    environment_seconds = 0.0
    digest = hashlib.sha256()
    transitions = 0
    for trajectory in trajectories:
        started = time.perf_counter()
        if backend == "python_reference":
            batch, step = python_batch_from_case(trajectory.case, auto_reset=False)
        else:
            batch, step = cpp_batch_from_case(trajectory.case, auto_reset=False)
        setup_seconds += time.perf_counter() - started
        for row in trajectory.rows:
            started = time.perf_counter()
            step = batch.step(row[None, :])
            environment_seconds += time.perf_counter() - started
            digest.update(np.asarray(step.player_ids).tobytes())
            digest.update(np.asarray(step.rewards).tobytes())
            transitions += 1
        if not bool(step.is_terminal[0]):
            raise AssertionError("matched replay did not terminate")
    return {
        "backend": backend,
        "completed_games": len(trajectories),
        "transitions": transitions,
        "trajectory_digest": digest.hexdigest(),
        "segments_seconds": {
            "setup": setup_seconds,
            "environment": environment_seconds,
        },
    }


def _python_batch_many(cases: list[DifferentialCase]):
    envs = []
    for case in cases:
        state = round_state_from_case(case)
        env = GuandanEnv(GameConfig(
            level_rank=case.level_rank,
            leader_seat=case.leader_seat,
            previous_result=previous_result_from_dict(case.previous_result),
            max_token_steps=python_game.max_game_horizon,
        ))
        env.reset(initial_state=StepwiseActionState(state).serialize())
        envs.append(env)
    batch = python_game.EnvBatch.from_envs(envs, auto_reset=False)
    return batch, batch.observe()


def _cpp_batch_many(cases: list[DifferentialCase]):
    batch = cpp_game.EnvBatch(len(cases), auto_reset=False)
    step = batch.reset_from_deals(
        [[list(hand) for hand in case.deal] for case in cases],
        level_ranks=[case.level_rank for case in cases],
        leader_seats=[case.leader_seat for case in cases],
        previous_results=[case.previous_result for case in cases],
    )
    return batch, step


def replay_batch(
    backend: str,
    trajectories: list[PreparedTrajectory],
    *,
    prefix_steps: int,
    cycles: int,
) -> dict[str, Any]:
    selected = trajectories
    cases = [trajectory.case for trajectory in selected]
    if any(len(trajectory.rows) < prefix_steps for trajectory in selected):
        raise ValueError("batch prefix exceeds a prepared trajectory")
    setup_seconds = 0.0
    environment_seconds = 0.0
    digest = hashlib.sha256()
    for _ in range(cycles):
        started = time.perf_counter()
        if backend == "python_reference":
            batch, _step = _python_batch_many(cases)
        else:
            batch, _step = _cpp_batch_many(cases)
        setup_seconds += time.perf_counter() - started
        for step_index in range(prefix_steps):
            rows = np.asarray(
                [trajectory.rows[step_index] for trajectory in selected], dtype=np.int64,
            )
            started = time.perf_counter()
            step = batch.step(rows)
            environment_seconds += time.perf_counter() - started
            digest.update(np.asarray(step.player_ids).tobytes())
    transitions = cycles * len(selected) * prefix_steps
    return {
        "backend": backend,
        "batch_size": len(selected),
        "cycles": cycles,
        "prefix_steps": prefix_steps,
        "transitions": transitions,
        "trajectory_digest": digest.hexdigest(),
        "segments_seconds": {
            "setup": setup_seconds,
            "environment": environment_seconds,
        },
    }


def frozen_policy_rollout(backend: str, cases: list[DifferentialCase]) -> dict[str, Any]:
    conversion_seconds = 0.0
    policy_seconds = 0.0
    environment_seconds = 0.0
    transitions = 0
    digest = hashlib.sha256()
    cfg = SimpleNamespace(device="cpu")
    for case in cases:
        if backend == "python_reference":
            batch, step = python_batch_from_case(case, auto_reset=False)
        else:
            batch, step = cpp_batch_from_case(case, auto_reset=False)
        for step_index in range(python_game.max_game_horizon):
            if bool(step.is_terminal[0]):
                break
            started = time.perf_counter()
            parsed = EnvStepBatch.from_env_step(step, cfg)
            conversion_seconds += time.perf_counter() - started
            started = time.perf_counter()
            legal = torch.nonzero(parsed["avail_weights"][0] > 0, as_tuple=False).flatten()
            position = (case.seed + 17 * step_index) % int(legal.numel())
            row = parsed["avail_tokens"][0, legal[position]].cpu().numpy().copy()
            policy_seconds += time.perf_counter() - started
            digest.update(np.ascontiguousarray(row).tobytes())
            started = time.perf_counter()
            step = batch.step(row[None, :])
            environment_seconds += time.perf_counter() - started
            transitions += 1
        else:
            raise RuntimeError("frozen-policy rollout did not terminate")
    return {
        "backend": backend,
        "completed_games": len(cases),
        "transitions": transitions,
        "trajectory_digest": digest.hexdigest(),
        "segments_seconds": {
            "observation_to_torch": conversion_seconds,
            "frozen_policy": policy_seconds,
            "environment": environment_seconds,
        },
    }


def _training_config(game: Any, root: Path) -> TrainConfig:
    cfg = TrainConfig(
        game=game.GAME_SPEC,
        device="cpu",
        algo_name="ippo",
        logging_dir=str(root),
        run_name="m05-benchmark",
        seed=71_001,
        rollout_batch_size=1,
        collect_batch_size=1,
        replay_buffer_scale=1,
        mini_batches=1,
        total_iterations=1,
        max_sampler_bytes_per_rank=512 * 2**20,
        amp_enabled=False,
        checkpoint_replay_buffer=False,
        model=ModelConfig(
            architecture="v1.3",
            history_dim=8,
            feature_dim=8,
            decoder_layers=1,
            decoder_heads=1,
            state_layers=1,
            action_layers=1,
            fuse_layers=1,
        ),
    )
    cfg.ppo.agent_epoch = 1
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg


def _matched_sampler(game: Any, cfg: TrainConfig, case: DifferentialCase) -> Sampler:
    if game is python_game:
        envs, step = python_batch_from_case(case, auto_reset=True)
    else:
        envs, step = cpp_batch_from_case(case, auto_reset=True)
    return Sampler(
        config=SamplerConfig(
            game=game,
            device="cpu",
            rollout_batch_size=1,
            collect_batch_size=1,
            sampling_temperature=1.0,
            purpose="on_policy",
            env_seed=cfg.seed,
        ),
        envs=envs,
        parsed_envs_step=EnvStepBatch.from_env_step(step, cfg),
    )


def benchmark_full_iteration(root: Path) -> dict[str, Any]:
    case = build_case(40_000)  # no previous result; exact same deal for both lanes
    python_cfg = _training_config(python_game, root / "python")
    cpp_cfg = _training_config(cpp_game, root / "cpp")
    python_trainer = IPPOTrainer(python_cfg, game=python_game, run_dir=root / "python")
    cpp_trainer = IPPOTrainer(cpp_cfg, game=cpp_game, run_dir=root / "cpp")
    cpp_trainer.agent_resume.load_state_dict(python_trainer.agent_resume.state_dict())
    cpp_trainer.agent_merged.load_state_dict(python_trainer.agent_merged.state_dict())
    python_trainer.sampler = _matched_sampler(python_game, python_cfg, case)
    cpp_trainer.sampler = _matched_sampler(cpp_game, cpp_cfg, case)

    records = {}
    for name, trainer in (("python_reference", python_trainer), ("cpp", cpp_trainer)):
        random.seed(99_001)
        np.random.seed(99_001)
        torch.manual_seed(99_001)
        sample = measure(lambda trainer=trainer: {"metrics": trainer.train_iteration()})
        metrics = sample["payload"]["metrics"]
        records[name] = {
            "wall_seconds": sample["wall_seconds"],
            "process_cpu_seconds": sample["process_cpu_seconds"],
            "peak_rss_bytes": sample["peak_rss_bytes"],
            "rss_delta_peak_bytes": sample["rss_delta_peak_bytes"],
            "completed_games": int(metrics["statistics/completed_games_total"]),
            "env_steps": int(metrics["statistics/env_steps_total"]),
            "game_steps": int(metrics["statistics/game_steps_total"]),
            "trajectory_length": float(metrics["statistics/trajectory_length_mean"]),
            "segments_seconds": {
                "rollout": float(metrics["perf/rollout_seconds"]),
                "learner": float(metrics["perf/learner_seconds"]),
            },
            "runtime_backend": metrics["runtime/backend"],
        }
    if records["python_reference"]["trajectory_length"] != records["cpp"]["trajectory_length"]:
        raise AssertionError("full-iteration trajectories differ")
    return {
        "workload": {
            "same_deal": case.to_json(),
            "same_model_state": True,
            "same_seed": 99_001,
            "algorithm": "IPPO",
            "agent_epoch": 1,
            "learner_unchanged": True,
        },
        "lanes": records,
        "speedup": records["python_reference"]["wall_seconds"] / records["cpp"]["wall_seconds"],
        "rollout_speedup": (
            records["python_reference"]["segments_seconds"]["rollout"]
            / records["cpp"]["segments_seconds"]["rollout"]
        ),
        "learner_ratio": (
            records["python_reference"]["segments_seconds"]["learner"]
            / records["cpp"]["segments_seconds"]["learner"]
        ),
        "gating": False,
        "note": "Diagnostic complete iteration: the identical neural policy and learner dominate wall time, so the hard backend gate is evaluated on matched environment and frozen-policy rollout workloads.",
    }


def lane_pair(
    name: str,
    python_fn: Callable[[], dict[str, Any]],
    cpp_fn: Callable[[], dict[str, Any]],
    *,
    work: int,
    unit: str,
    repetitions: int,
) -> dict[str, Any]:
    python_samples = [measure(python_fn) for _ in range(repetitions)]
    cpp_samples = [measure(cpp_fn) for _ in range(repetitions)]
    python_summary = summarize_samples(python_samples, work=work, unit=unit)
    cpp_summary = summarize_samples(cpp_samples, work=work, unit=unit)
    if python_summary["result"]["trajectory_digest"] != cpp_summary["result"]["trajectory_digest"]:
        raise AssertionError(f"{name} benchmark lanes did not replay the same trajectory")
    speedup = python_summary["wall_seconds_median"] / cpp_summary["wall_seconds_median"]
    return {
        "name": name,
        "work": work,
        "unit": unit,
        "python_reference": python_summary,
        "cpp": cpp_summary,
        "speedup": speedup,
        "threshold": THRESHOLD,
        "passes_threshold": speedup >= THRESHOLD,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-prefix", type=int, default=64)
    parser.add_argument("--batch-cycles", type=int, default=4)
    args = parser.parse_args()
    if min(args.games, args.repetitions, args.batch_size, args.batch_prefix, args.batch_cycles) <= 0:
        parser.error("all numeric arguments must be positive")
    prepared_count = max(args.games, args.batch_size)
    prepared = prepare_trajectories(prepared_count)
    singles = prepared[:args.games]
    batches = prepared[:args.batch_size]
    single_work = sum(len(item.rows) for item in singles)
    batch_work = args.batch_size * args.batch_prefix * args.batch_cycles
    end_cases = [item.case for item in singles]

    # One untimed transition per backend warms imports/allocator paths without
    # entering any measured sample.
    replay_single("python_reference", singles[:1])
    replay_single("cpp", singles[:1])

    environment_single = lane_pair(
        "environment_single_complete_games",
        lambda: replay_single("python_reference", singles),
        lambda: replay_single("cpp", singles),
        work=single_work,
        unit="token_transitions",
        repetitions=args.repetitions,
    )
    environment_batch = lane_pair(
        "environment_batch_fixed_prefix",
        lambda: replay_batch(
            "python_reference", batches,
            prefix_steps=args.batch_prefix, cycles=args.batch_cycles,
        ),
        lambda: replay_batch(
            "cpp", batches,
            prefix_steps=args.batch_prefix, cycles=args.batch_cycles,
        ),
        work=batch_work,
        unit="token_transitions",
        repetitions=args.repetitions,
    )
    # Obtain the exact work count before timing the pair.  Outputs are
    # deterministic for the fixed cases and policy.
    end_probe = frozen_policy_rollout("python_reference", end_cases)
    end_to_end = lane_pair(
        "end_to_end_frozen_policy_rollout",
        lambda: frozen_policy_rollout("python_reference", end_cases),
        lambda: frozen_policy_rollout("cpp", end_cases),
        work=int(end_probe["transitions"]),
        unit="token_transitions",
        repetitions=args.repetitions,
    )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    iteration_root = output.parent / "full_iteration"
    full_iteration = benchmark_full_iteration(iteration_root)
    gated = [environment_single, environment_batch, end_to_end]
    result = {
        "schema": BENCHMARK_SCHEMA,
        "stage": "M05_differential_performance",
        "timestamp_unix": time.time(),
        "source_commit": git_head(),
        "protocol_version": python_game.protocol_version,
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_bytes": psutil.virtual_memory().total,
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "gpus": nvidia_inventory(),
        },
        "workload": {
            "same_machine": True,
            "same_deals": True,
            "same_action_rows_or_frozen_policy": True,
            "single_games": args.games,
            "single_transitions": single_work,
            "batch_size": args.batch_size,
            "batch_prefix": args.batch_prefix,
            "batch_cycles": args.batch_cycles,
            "batch_transitions": batch_work,
            "repetitions": args.repetitions,
        },
        "lanes": {
            "environment_single": environment_single,
            "environment_batch": environment_batch,
            "end_to_end_rollout": end_to_end,
            "full_iteration": full_iteration,
        },
        "gate": {
            "threshold": THRESHOLD,
            "definition": "All matched environment-only and frozen-policy end-to-end rollout comparisons must reach >=2x. The unchanged full learner iteration is segmented and reported diagnostically because identical model/learner work is backend-independent.",
            "gated_lanes": [lane["name"] for lane in gated],
            "minimum_speedup": min(lane["speedup"] for lane in gated),
            "passes": all(lane["passes_threshold"] for lane in gated),
            "target_5x_reached": all(lane["speedup"] >= 5.0 for lane in gated),
        },
        "valid": all(lane["passes_threshold"] for lane in gated),
        "errors": [],
        "notes": [
            "Torch is CPU-only in this environment; the installed NVIDIA GPU is inventoried but not used.",
            "No GPU busy loop or synthetic utilization workload is used.",
            "Model/learner bottlenecks are not mixed into the primary environment gate; the complete iteration provides explicit rollout and learner segments.",
        ],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid": result["valid"],
        "minimum_speedup": result["gate"]["minimum_speedup"],
        "single_speedup": environment_single["speedup"],
        "batch_speedup": environment_batch["speedup"],
        "end_to_end_speedup": end_to_end["speedup"],
        "full_iteration_speedup": full_iteration["speedup"],
        "output": str(output),
    }, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
