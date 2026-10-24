"""Real two-process launch/communication semantics, no model or optimizer."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from serial_compare.arm_worker import read_launch_context
import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--rank", type=int, required=True)
args = parser.parse_args()
launch = read_launch_context()
assert not dist.is_initialized()
dist.init_process_group("gloo", timeout=timedelta(seconds=12))
try:
    assert dist.get_rank() == launch.rank and dist.get_world_size() == launch.world_size
    value = torch.tensor([launch.rank + 1], dtype=torch.int64)
    dist.all_reduce(value)
    (args.output_dir / f"rank{launch.rank}.json").write_text(json.dumps({
        "rank": launch.rank, "local_rank": launch.local_rank,
        "world_size": launch.world_size, "reduced": value.item(),
    }), encoding="utf8")
finally:
    dist.destroy_process_group()
