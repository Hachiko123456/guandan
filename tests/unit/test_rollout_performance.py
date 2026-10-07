"""Bounded collection regression and optional real on-policy microbenchmark.

CLI: python tests/unit/test_rollout_performance.py --baseline-root /path/to/current
For an integrated checkout use --baseline-ref <pre-change-commit> instead.
Timings are host wall times, not a claim of CUDA utilization or T4 throughput.
No timing threshold in CI: exact call counts are the deterministic regression.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter as monotonic
import types

import pytest
import torch

from guandan.cards import Rank
from guandan.env_batch import GuandanEnvBatch
from guandan.environment_types import GameConfig
from guandan.model import PolicyValueNet
from guandan.training.collector import RolloutCollector, Rollout
from guandan.training.critic import CentralQCritic
from guandan.training.estimators import compute_gae, compute_q_boost, masked_expected_values
from guandan.training.runtime import TRAIN_SEED_BASE


def make_collector(algorithm, device="cpu", collector_cls=RolloutCollector, *, limits=(20000, 20000)):
    torch.manual_seed(123)
    configs = [GameConfig(level_rank=Rank.FIVE, seed=TRAIN_SEED_BASE+i, max_token_steps=limit)
               for i, limit in enumerate(limits)]
    batch = GuandanEnvBatch(configs, auto_reset=False, transactional=False)
    batch.reset(seeds=[c.seed for c in configs])
    actor = PolicyValueNet(hidden_dim=16).to(device)
    critic = CentralQCritic(hidden_dim=16).to(device) if algorithm == "vrpo" else None
    return collector_cls(batch, actor, seed=77, critic=critic,
        expected_values=masked_expected_values if critic is not None else None)


def compare_rollouts(left, right, algorithm):
    assert left.digest() == right.digest()
    errors = {}
    # FP32 evaluation shapes differ at subset bootstraps; do not claim bitwise.
    for field in ("old_log_probs", "values", "next_values", "selected_q"):
        if field == "selected_q" and algorithm == "ippo":
            continue
        a, b = left.stack(field), right.stack(field)
        errors[field] = (a-b).abs().max().item()
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    def targets(r):
        args = (r.rewards, r.values, r.next_values, r.terminated, r.truncated)
        if algorithm == "ippo":
            return compute_gae(*args, gamma=1, gae_lambda=.95)
        return compute_q_boost(r.rewards, r.stack("selected_q"), r.values, r.next_values,
                               r.terminated, r.truncated, gamma=1, gae_lambda=.95)
    for index, (a, b) in enumerate(zip(targets(left), targets(right))):
        errors[f"target_{index}"] = (a-b).abs().max().item()
        torch.testing.assert_close(a, b, atol=5e-6, rtol=3e-5)
    return errors


@pytest.mark.parametrize("algorithm", ["ippo", "vrpo"])
@pytest.mark.parametrize("limits", [(20000, 20000), (2, 3)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bounded_matches_immediate_onpolicy_reference(algorithm, limits, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; CPU cannot certify GPU equivalence")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bounded = make_collector(algorithm, device, limits=limits)
        eager = make_collector(algorithm, device, limits=limits)
        # Each one-tick reference call samples immediately, steps the real env,
        # and bootstraps before reset. This is NOT offline action replay.
        reference = Rollout(tuple(row for _ in range(8) for row in eager.collect(2).transitions))
        actual = bounded.collect(16)
        errors = compare_rollouts(reference, actual, algorithm)
        assert eager.batch.serialize() == bounded.batch.serialize()
        assert torch.equal(eager.generator.get_state(), bounded.generator.get_state())
        print(f"{device} {algorithm} limits={limits} max_abs_errors={errors}")
    finally:
        torch.set_num_threads(threads)


def load_baseline(root, ref=None):
    relative = "guandan/training/collector.py"
    source = (subprocess.check_output(["git", "-C", str(root), "show", f"{ref}:{relative}"], text=True)
              if ref else (Path(root)/relative).read_text(encoding="utf-8"))
    module = types.ModuleType("guandan.training._collector_benchmark_baseline")
    module.__package__ = "guandan.training"
    sys.modules[module.__name__] = module
    exec(compile(source, str(Path(root)/relative), "exec"), module.__dict__)
    return module.RolloutCollector


def benchmark(root, ref, device, repeats, ticks, envs=2):
    baseline = load_baseline(root, ref)
    torch.set_num_threads(1)
    results = []
    for algorithm in ("ippo", "vrpo"):
        measurements = {"baseline": [], "bounded": []}
        for repeat in range(repeats):
            outputs = {}
            order = [("baseline", baseline), ("bounded", RolloutCollector)]
            if repeat % 2:
                order.reverse()
            for name, cls in order:
                collector = make_collector(algorithm, device, cls, limits=(20000,)*envs)
                # Initialize kernels outside the timed section; never steps an env.
                from guandan.training.collector import critic_state
                state = critic_state(collector.batch, collector.observation) if algorithm == "vrpo" else None
                collector._evaluate(collector.observation, state)
                sizes = []
                handle = collector.model.register_forward_pre_hook(lambda m, args: sizes.append(len(args[0])))
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                started = monotonic()
                rollout = collector.collect(ticks * envs)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = monotonic()-started
                handle.remove()
                record = {"seconds": elapsed, "forward_calls": len(sizes), "max_forward_batch": max(sizes),
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                          "timings": getattr(collector, "last_timings", {})}
                measurements[name].append(record)
                outputs[name] = (rollout, collector.generator.get_state(), collector.batch.serialize())
                del collector
            errors = compare_rollouts(outputs["baseline"][0], outputs["bounded"][0], algorithm)
            assert torch.equal(outputs["baseline"][1], outputs["bounded"][1])
            assert outputs["baseline"][2] == outputs["bounded"][2]
        results.append({"algorithm": algorithm, "device": device, "ticks": ticks,
                        "token_steps": ticks*envs, "measurements": measurements,
                        "median_speedup": statistics.median(r["seconds"] for r in measurements["baseline"])/
                                          statistics.median(r["seconds"] for r in measurements["bounded"]),
                        "last_repeat_max_abs_error": errors})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=64)
    parser.add_argument("--envs", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.ticks < 1 or args.envs < 1:
        parser.error("repeats, ticks and envs must be positive")
    benchmark(args.baseline_root, args.baseline_ref, args.device, args.repeats, args.ticks, args.envs)
