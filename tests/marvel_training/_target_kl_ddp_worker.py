"""Two-rank Gloo worker: unequal macro counts must agree on the global KL stop.

Launched by ``test_policy_import_target_kl.py`` via ``torch.distributed.run``.
Rank 0 holds 1 macro, rank 1 holds 9 macros (or the reverse), so a naive
mean-of-rank-means would disagree with the global sum/count statistic and could
stop one rank only (deadlock or divergence).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def _log_ratio_for_kl(target_kl: float) -> float:
    """Bisection on ``expm1(x) - x`` (independent of the production helper)."""
    low, high = 0.0, 50.0
    for _ in range(200):
        middle = (low + high) / 2.0
        value = math.expm1(middle) - middle
        if value < target_kl:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def main() -> int:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError("this worker requires exactly two ranks")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from guandan_marvel.training.macro_ppo import macro_sample_kl_control

    scenario = os.environ.get("KL_SCENARIO", "one_vs_nine")
    target = float(os.environ["KL_TARGET"])
    if scenario == "one_vs_nine":
        count = 1 if rank == 0 else 9
        ratio = torch.tensor([_log_ratio_for_kl(0.04)] + [0.0] * (count - 1)) if rank == 0 else torch.zeros(9)
    elif scenario == "nine_vs_one":
        ratio = torch.tensor([_log_ratio_for_kl(0.02)] * 9) if rank == 0 else torch.zeros(1)
    elif scenario == "empty_rank":
        ratio = torch.zeros(4) if rank == 0 else torch.zeros(0)
    else:
        raise RuntimeError(f"unknown scenario {scenario!r}")
    valid = torch.ones(ratio.numel(), dtype=torch.bool)
    lengths = torch.ones(ratio.numel(), dtype=torch.long)
    decision = macro_sample_kl_control(ratio.view(1, -1), valid.view(1, -1), lengths.view(1, -1),
                                       world_size=2, target_kl=target)
    payload = {
        "rank": rank,
        "global_count": int(decision.global_count),
        "global_sum": float(decision.global_sum),
        "global_mean": decision.global_mean,
        "stop": bool(decision.stop),
        "local_count": int(decision.local_count),
        "quantile_sample_size": int(decision.quantile_sample_size),
    }
    gathered = [None, None]
    dist.all_gather_object(gathered, payload)
    dist.barrier()
    if rank == 0:
        out = Path(os.environ["KL_OUTPUT"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(gathered, indent=2), encoding="utf-8")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
