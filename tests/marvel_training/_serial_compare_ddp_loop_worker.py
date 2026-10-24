"""Two-rank Gloo worker exercising the production ``run_training_arm`` loop.

The trainer is fake but every ``train_iteration`` performs a real ``all_reduce``
so the DDP collective order is genuine.  ``_save_recovery`` is replaced by a
real ``dist.barrier()`` (the first collective of the production save path), so a
rank-local save/stop decision immediately desynchronises the group.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


class FakeConfig:
    seed = 570_700
    training_data_generator_version = "ddp-loop-worker"
    training_context_schedule = {"schedule": "fake"}
    training_context_curriculum = {"curriculum": "fake"}

    def to_dict(self) -> dict[str, object]:
        return {
            "training_context_schedule": self.training_context_schedule,
            "training_context_curriculum": self.training_context_curriculum,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--seconds-rank1", type=float, default=None)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--save-interval", type=float, required=True)
    parser.add_argument("--isolation-run-dirs", action="store_true")
    parser.add_argument("--slack", type=float, default=0.0)
    args = parser.parse_args(argv)
    rank = int(args.rank)
    seconds = float(args.seconds) if rank == 0 or args.seconds_rank1 is None else float(args.seconds_rank1)

    from serial_compare import training
    from serial_compare.common import build_plan, sha

    dist.init_process_group("gloo", timeout=timedelta(seconds=float(args.slack) + 6.0))
    events: list[dict[str, object]] = []
    clock = [0.0]

    def fake_save(trainer, run_dir):
        events.append({
            "kind": "save", "rank": rank, "iteration": int(trainer.iteration),
            "t": round(clock[0], 3),
        })
        # Production save begins with context.barrier(); a mismatch here hangs.
        dist.barrier()
        return Path(run_dir) / "fake-checkpoint"

    training._save_recovery = fake_save

    class FakeTrainer:
        world_size = 2
        is_primary = rank == 0
        distributed = None
        opponent_pool = None

        def __init__(self):
            self.iteration = 0
            self.learner_updates = 0
            self.cfg = FakeConfig()

        def train_iteration(self):
            tensor = torch.tensor([1], dtype=torch.int64)
            dist.all_reduce(tensor)
            time.sleep(seconds)
            clock[0] += seconds
            self.iteration += 1
            self.learner_updates += 1
            events.append({"kind": "iteration", "rank": rank,
                           "iteration": self.iteration, "t": round(clock[0], 3)})
            return {
                "statistics/learner_updates": 1,
                "macro/value_ev": 0.5,
            }

    workdir = args.workdir
    source = workdir / "policy_375.pt"
    plan = build_plan(
        source={key: "x" * 64 for key in (
            "archive_sha256", "content_manifest_hash", "core_source_hash",
            "backend_hash", "native_sha256")}
        | {"git_commit": "f" * 40, "runtime_path": "/tmp", "python": "3.13.15",
           "torch": "fake", "native_name": "n"},
        models={name: {"sha256": "1" * 64, "iteration": 1, "training_resume_allowed": False}
                for name in ("p375", "p250", "p500")},
    )
    plan["models"]["p375"]["sha256"] = sha(source)
    run_dir = workdir / (f"run_rank{rank}" if args.isolation_run_dirs else "run")

    payload: dict[str, object] = {"ok": False}
    try:
        result = training.run_training_arm(
            plan=plan, arm="control", run_dir=run_dir, source_payload={},
            source_model_path=source, model_paths={}, game=object(), device="cpu",
            ddp_enabled=True, world_size=2, clock=lambda: clock[0],
            max_seconds=float(args.max_seconds),
            save_interval_seconds=float(args.save_interval),
            trainer_factory=lambda cfg: FakeTrainer(),
            import_policy=lambda *a, **k: {"hash": "import"},
            initial_state_fingerprint=lambda trainer: {"hash": "fp"},
            config_override=FakeConfig(), pool_entries_override=(),
            rank=rank,
            policy_exporter=lambda trainer, path, iteration: Path(path).write_bytes(b"policy"),
        )
        payload = {"ok": True, "result": result}
    except BaseException as error:  # noqa: BLE001 - record the failure honestly
        payload = {"ok": False, "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
    events_path = args.events.with_name(args.events.stem + f".rank{rank}" + args.events.suffix)
    events_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    result_path = args.result.with_name(args.result.stem + f".rank{rank}" + args.result.suffix)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
