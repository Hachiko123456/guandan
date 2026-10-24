# SPDX-License-Identifier: MIT
"""Two-rank arm worker (launched by ``torch.distributed.run``).

Usage (inside the Kaggle runtime, with the original native restored)::

    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        -m serial_compare.arm_worker --plan plan.json --arm candidate ...

Only rank 0 writes the result JSON; every rank writes its own trainer state
through ``save_distributed_checkpoint``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class LaunchContext:
    world_size: int
    rank: int
    local_rank: int

    @property
    def device(self) -> str:
        return f"cuda:{self.local_rank}"


def read_launch_context(environment: Mapping[str, str] | None = None) -> LaunchContext:
    """Read torchrun's environment before the trainer initializes its PG.

    torchrun launches processes; it does not initialize their process groups.
    Treating ``dist.is_initialized() == False`` as a single-rank launch silently
    disables DDP and maps both workers to GPU 0.
    """
    env = os.environ if environment is None else environment
    required = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if not env.get(name)]
    if missing:
        raise ValueError(f"incomplete torchrun environment: {', '.join(missing)}")
    try:
        world, rank, local = (int(env[name]) for name in required[:3])
        port = int(env["MASTER_PORT"])
    except (TypeError, ValueError) as error:
        raise ValueError("torchrun ranks, WORLD_SIZE and MASTER_PORT must be integers") from error
    if world <= 1 or not 0 <= rank < world or local < 0 or not 1 <= port <= 65535:
        raise ValueError("invalid torchrun world/rank/local-rank or rendezvous port; no single-rank fallback")
    if env.get("LOCAL_WORLD_SIZE"):
        try:
            local_world = int(env["LOCAL_WORLD_SIZE"])
        except (TypeError, ValueError) as error:
            raise ValueError("LOCAL_WORLD_SIZE must be an integer") from error
        if not 1 <= local_world <= world or local >= local_world:
            raise ValueError("LOCAL_RANK is outside LOCAL_WORLD_SIZE")
    return LaunchContext(world, rank, local)


def validate_launch_device(torch, launch: LaunchContext) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("serial comparison requires CUDA; CPU fallback is forbidden")
    if launch.local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={launch.local_rank} has no visible CUDA device")
    dist = torch.distributed
    if dist.is_initialized():
        if dist.get_world_size() != launch.world_size or dist.get_rank() != launch.rank:
            raise RuntimeError("initialized process group disagrees with torchrun world/rank")
        if str(dist.get_backend()).lower() != "nccl":
            raise RuntimeError("serial comparison requires the NCCL process group; no backend fallback")
    torch.cuda.set_device(launch.local_rank)


def _record_failure(args, launch: LaunchContext | None, stage: str, error: BaseException) -> None:
    trace = traceback.format_exc()
    payload = {
        "rank": None if launch is None else launch.rank,
        "local_rank": None if launch is None else launch.local_rank,
        "world_size": None if launch is None else launch.world_size,
        "stage": stage, "type": type(error).__name__, "message": str(error),
        "traceback": trace,
    }
    # Preserve the original exception even if diagnostics cannot be written.
    try:
        directory = args.run_dir / "worker_errors"
        directory.mkdir(parents=True, exist_ok=True)
        rank_label = "unknown" if launch is None else f"{launch.rank:03d}"
        destination = directory / f"rank{rank_label}.json"
        temporary = directory / f".rank{rank_label}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        print(f"[ARM-ERROR] rank={payload['rank']} stage={stage} {type(error).__name__}: {error}; details={destination}", file=sys.stderr, flush=True)
    except Exception as diagnostic_error:
        print(f"[ARM-ERROR] could not write diagnostics: {diagnostic_error}", file=sys.stderr, flush=True)
    print(trace, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=("control", "candidate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--save-interval-seconds", type=float, default=600.0)
    parser.add_argument("--iteration-limit", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--hard-reserve-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)

    launch = None
    torch = None
    stage = "torchrun_environment"
    try:
        launch = read_launch_context()
        sys.path.insert(0, str(Path(args.source).resolve()))
        stage = "cuda_launch"
        import torch

        validate_launch_device(torch, launch)
        print(
            f"[ARM-START] arm={args.arm} rank={launch.rank} local_rank={launch.local_rank} "
            f"world_size={launch.world_size} device={launch.device} ddp_enabled=True "
            f"pg_initialized={torch.distributed.is_initialized()}", flush=True,
        )
        # The trainer's DistributedRuntime.from_config is the one initialization
        # path; pass the torchrun world even while its PG is not initialized.
        stage = "backend_identity"
        from guandan_marvel import cpp_game
        from guandan_marvel.training.checkpoints import build_backend_identity
        from guandan_marvel.training.smoke import CppSmokeGame

        plan = _read(args.plan)
        backend = build_backend_identity(cpp_game)
        if backend["hash"] != str(plan["source"]["backend_hash"]):
            raise RuntimeError("loaded backend identity differs from the frozen plan; refusing to train")
        game = CppSmokeGame(int(plan["evaluation"]["horizon"]))
        stage = "policy_load"
        source_payload = torch.load(args.dataset / "policy_375.pt", map_location="cpu", weights_only=False)
        from serial_compare.training import run_training_arm

        stage = "train_arm"
        result = run_training_arm(
            plan=plan,
            arm=args.arm,
            run_dir=args.run_dir,
            source_payload=source_payload,
            source_model_path=args.dataset / "policy_375.pt",
            model_paths={
                "p375": args.dataset / "policy_375.pt",
                "p250": args.dataset / "anchor_250.pt",
                "p500": args.dataset / "policy_500.pt",
            },
            game=game,
            device=launch.device,
            ddp_enabled=True,
            world_size=launch.world_size,
            max_seconds=float(args.max_seconds),
            save_interval_seconds=float(args.save_interval_seconds),
            resume_from=args.resume_from,
            iteration_limit=args.iteration_limit,
            seed_offset=int(args.seed_offset),
            hard_reserve_seconds=float(args.hard_reserve_seconds),
            rank=launch.rank,
        )
        stage = "result_publish"
        if launch.rank == 0:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    except BaseException as error:
        _record_failure(args, launch, stage, error)
        raise
    finally:
        if torch is not None and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as cleanup_error:
                print(f"[ARM-CLEANUP] {cleanup_error}", file=sys.stderr, flush=True)


if __name__ == "__main__":  # pragma: no cover - target-only entry point
    from torch.distributed.elastic.multiprocessing.errors import record

    raise SystemExit(record(main)())
