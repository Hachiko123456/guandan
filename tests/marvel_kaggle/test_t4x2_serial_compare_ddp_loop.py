"""Two-rank Gloo counterexamples for the production serial-compare loop.

The trainer is fake but every iteration performs a real ``all_reduce`` and the
production ``_save_recovery`` is replaced by a real ``dist.barrier()``.  A
rank-local save/stop decision therefore deadlocks the process group; a
collective decision keeps both ranks aligned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests" / "marvel_training"))

from _ddp_launcher import launch_two_ranks  # noqa: E402

WORKER = REPO / "tests/marvel_training/_serial_compare_ddp_loop_worker.py"


def _run(tmp_path: Path, *, max_seconds: float, save_interval: float,
         isolation: bool = False, timeout: float = 90.0):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "policy_375.pt").write_bytes(b"the actual 375 artifact")
    args = [
        "--workdir", str(workdir),
        "--events", str(workdir / "events.jsonl"),
        "--result", str(workdir / "result.json"),
        "--seconds", "0.7",
        "--seconds-rank1", "0.45",
        "--max-seconds", str(max_seconds),
        "--save-interval", str(save_interval),
    ]
    if isolation:
        args.append("--isolation-run-dirs")
    completed = launch_two_ranks(WORKER, *args, cwd=REPO, timeout=timeout)
    return workdir, [int(process.returncode) for process in completed]


def _result(workdir: Path, rank: int):
    path = workdir / f"result.rank{rank}.json"
    assert path.is_file(), f"rank {rank} produced no result"
    return json.loads(path.read_text(encoding="utf-8"))


def _events(workdir: Path, rank: int):
    path = workdir / f"events.rank{rank}.jsonl"
    assert path.is_file(), f"rank {rank} produced no events"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _saves(events):
    return sorted(event["iteration"] for event in events if event["kind"] == "save")


def test_skewed_clocks_save_at_the_same_iterations(tmp_path):
    workdir, codes = _run(tmp_path, max_seconds=10.0, save_interval=1.0)
    assert codes == [0, 0], [_result(workdir, rank) for rank in (0, 1)]
    results = [_result(workdir, rank) for rank in (0, 1)]
    saves = [_saves(_events(workdir, rank)) for rank in (0, 1)]
    assert saves[0] == saves[1] and saves[0], saves
    assert results[0]["result"]["final_iteration"] == results[1]["result"]["final_iteration"]
    # Multi-rank shared JSON is never written to the same path by both ranks.
    run_dir = workdir / "run"
    for name in ("resolved_config", "policy_import_manifest", "initial_state_fingerprint"):
        assert (run_dir / f"{name}.rank000.json").is_file()
        assert (run_dir / f"{name}.rank001.json").is_file()
    assert (run_dir / "resolved_config.json").is_file()


def test_one_rank_reaching_the_budget_stops_both_with_updates(tmp_path):
    workdir, codes = _run(tmp_path, max_seconds=2.0, save_interval=100.0)
    assert codes == [0, 0], [_result(workdir, rank) for rank in (0, 1)]
    results = [_result(workdir, rank)["result"] for rank in (0, 1)]
    assert results[0]["final_iteration"] == results[1]["final_iteration"]
    assert results[0]["status"] == results[1]["status"] == "budget_stopped"
    assert results[0]["applied_updates"] > 0 and results[1]["applied_updates"] > 0
    # The final (collective) save ran on both ranks at the same iteration.
    final_saves = [_saves(_events(workdir, rank)) for rank in (0, 1)]
    assert final_saves[0] == final_saves[1] == [results[0]["final_iteration"]]


class _NoCollectiveTrainer:
    world_size = 1
    is_primary = True
    distributed = None
    opponent_pool = None

    def __init__(self, clock=None):
        self.iteration = 0
        self.learner_updates = 0
        self.clock = clock if clock is not None else [0.0]

    def train_iteration(self):
        self.clock[0] += 1.0
        self.iteration += 1
        self.learner_updates += 1
        return {"statistics/learner_updates": 1}

    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"checkpoint")


def test_world_size_one_never_calls_a_distributed_collective(tmp_path, monkeypatch):
    import torch.distributed as dist

    from serial_compare import training

    def forbidden(*_args, **_kwargs):
        raise AssertionError("world_size=1 must not call a distributed collective")

    monkeypatch.setattr(dist, "all_reduce", forbidden)
    monkeypatch.setattr(dist, "all_gather_object", forbidden)
    monkeypatch.setattr(dist, "barrier", forbidden)

    class FakeConfig:
        seed = 570_700
        training_data_generator_version = "world-one"
        training_context_schedule = {"s": 1}
        training_context_curriculum = {"c": 1}

        def to_dict(self):
            return {"training_context_schedule": self.training_context_schedule,
                    "training_context_curriculum": self.training_context_curriculum}

    source = tmp_path / "policy_375.pt"
    source.write_bytes(b"the actual 375 artifact")
    from serial_compare.common import build_plan, sha

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
    clock = [0.0]
    result = training.run_training_arm(
        plan=plan, arm="control", run_dir=tmp_path / "run", source_payload={},
        source_model_path=source, model_paths={}, game=object(), device="cpu",
        ddp_enabled=False, world_size=1, clock=lambda: clock[0], max_seconds=2.0,
        trainer_factory=lambda cfg: _NoCollectiveTrainer(clock),
        import_policy=lambda *a, **k: {"hash": "import"},
        initial_state_fingerprint=lambda trainer: {"hash": "fp"},
        config_override=FakeConfig(), pool_entries_override=(), rank=0,
        policy_exporter=lambda trainer, path, iteration: Path(path).write_bytes(b"policy"),
    )
    assert result["status"] == "budget_stopped"
    assert result["applied_updates"] > 0
    assert result["final_iteration"] == result["applied_updates"]
