"""Two-rank CPU gloo worker for the phase-6 empty-current-macro boundary.

Rank 0 holds one current-owned macro and one opponent macro; rank 1 holds two
current macros.  The production macro PPO loss must exclude the opponent
macro, all-reduce a global current count of 2, and give rank 0 a
model-connected exact zero anchor.  This is a CPU gloo test and does not claim
NCCL behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from guandan_marvel.phase6.policy_owner import segment_pooled_macros  # noqa: E402
from guandan_marvel.training.macro_ppo import (  # noqa: E402
    build_macro_batch_view, compute_macro_pg_loss,
)

from tests.marvel_training._macro_fixtures import make_case  # noqa: E402

SEATS = (0, 1, 0, 1)
CASES = {
    # rank 0 holds only frozen-opponent macros: zero current samples.
    0: {"players": [1, 3], "commits": [True, True]},
    # rank 1 holds two current-owned macros.
    1: {"players": [0, 2, 1, 3], "commits": [True, True, True, True]},
}
MIXED_CASES = {
    0: {"players": [0, 1], "commits": [True, True]},
    1: {"players": [0, 2, 1, 3], "commits": [True, True, True, True]},
}


def build_case(rank: int, mode: str = "empty"):
    spec = (MIXED_CASES if mode == "mixed" else CASES)[int(rank)]
    rollout, values, _segmentation, new_probs = make_case(
        spec["players"], spec["commits"], garbage_padding=False,
    )
    length = rollout["player_ids"].shape[1]
    owner = torch.full((1, length), -1, dtype=torch.long)
    train = torch.zeros((1, length), dtype=torch.bool)
    opponent_id = torch.full((1, length), -1, dtype=torch.long)
    opponent_version = torch.full((1, length), -1, dtype=torch.long)
    assignment = torch.full((1, length), -1, dtype=torch.long)
    assigned = torch.full((1, length, 4), -1, dtype=torch.long)
    for step, player in enumerate(spec["players"]):
        seat_owner = SEATS[player]
        owner[0, step] = seat_owner
        train[0, step] = seat_owner == 0
        opponent_id[0, step] = -1 if seat_owner == 0 else 0
        opponent_version[0, step] = -1 if seat_owner == 0 else 1
        assignment[0, step] = 1
        assigned[0, step, :] = torch.tensor(SEATS, dtype=torch.long)
    rollout.set("policy_owner", owner)
    rollout.set("policy_train_mask", train)
    rollout.set("opponent_checkpoint_id", opponent_id)
    rollout.set("opponent_policy_version", opponent_version)
    rollout.set("assignment_id", assignment)
    rollout.set("assigned_seats", assigned)
    rollout.set("on_policy_valid", train.clone())
    segmentation = segment_pooled_macros(rollout)
    return rollout, segmentation, new_probs, train


def scaled_log_probs(rollout, probabilities, w):
    steps = rollout["num_steps"].unsqueeze(1)
    time = torch.arange(probabilities.shape[1]).unsqueeze(0)
    active = time < steps
    base = torch.where(
        active.unsqueeze(-1),
        probabilities.clamp_min(1e-6).log(),
        torch.zeros_like(probabilities),
    )
    return base * w


def run(rank: int, out_path: Path, mode: str = "empty") -> None:
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    world_size = int(dist.get_world_size())
    rollout, segmentation, probabilities, train = build_case(rank, mode)
    view = build_macro_batch_view(segmentation, torch.arange(segmentation.batch_size))
    trainable = segmentation.macro_train_mask
    assert trainable is not None
    advantage = torch.zeros((segmentation.batch_size, segmentation.max_macros))
    sign = 1.0
    for index in range(segmentation.max_macros):
        if bool(trainable[0, index]):
            advantage[0, index] = sign * (index + 1.0)
            sign = -sign
    w = torch.tensor(0.7, requires_grad=True)
    log_probs = scaled_log_probs(rollout, probabilities, w)
    pg = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=log_probs,
        macro_advantage=advantage, clip_coef=0.2, world_size=world_size,
    )
    pg.pg_loss_mean.backward()
    payload = {
        "rank": int(rank),
        "world_size": world_size,
        "local_trainable_macros": int(pg.macro_count_local),
        "global_trainable_macros": int(pg.macro_count_global),
        "local_current_tokens": int(train.sum().item()),
        "local_active_macros": int(
            (segmentation.macro_valid & (segmentation.macro_end_index >= 0)).sum().item()
        ),
        "loss": float(pg.pg_loss_mean.detach().item()),
        "grad": float(w.grad.item()),
        "finite": bool(torch.isfinite(pg.pg_loss_mean).all().item()),
        "empty_local_batch": bool(pg.empty_local_batch),
        "record": {
            "pg_loss": pg.pg_loss_mean.detach().item(),
            "macro_count_local": pg.macro_count_local,
            "macro_count_global": pg.macro_count_global,
        },
    }
    target = Path(f"{out_path}.rank{rank}.json")
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--mode", default="empty", choices=["empty", "mixed"])
    args = parser.parse_args()
    run(args.rank, Path(args.out), args.mode)


if __name__ == "__main__":
    main()
