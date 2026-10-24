"""CLI launch contracts, with model execution replaced at its boundary only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from serial_compare import arm_worker


def environment(rank=0, world=2, local=None):
    return {"WORLD_SIZE": str(world), "RANK": str(rank),
            "LOCAL_RANK": str(rank if local is None else local),
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "12345"}


def configure(monkeypatch, env):
    for key in (*environment(), "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def fake_runtime(monkeypatch, *, initialized=False, pg_rank=0, pg_world=2, backend="nccl", fail=None):
    state = {"initialized": initialized, "devices": [], "destroyed": False, "calls": []}
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                 set_device=lambda index: state["devices"].append(index))
    def destroy():
        state.update(initialized=False, destroyed=True)
    torch.distributed = SimpleNamespace(
        is_initialized=lambda: state["initialized"], get_rank=lambda: pg_rank,
        get_world_size=lambda: pg_world, get_backend=lambda: backend,
        destroy_process_group=destroy,
    )
    torch.load = lambda *a, **kw: {}
    game = ModuleType("guandan_marvel")
    game.cpp_game = object()
    checks = ModuleType("guandan_marvel.training.checkpoints")
    checks.build_backend_identity = lambda _: {"hash": "test-backend"}
    smoke = ModuleType("guandan_marvel.training.smoke")
    smoke.CppSmokeGame = lambda horizon: SimpleNamespace(horizon=horizon)
    training = ModuleType("serial_compare.training")
    def train(**kwargs):
        state["calls"].append(kwargs)
        state["initialized"] = True  # trainer owns initialization in production
        if fail:
            raise RuntimeError(fail)
        return {"status": "test-only", "rank": kwargs["rank"]}
    training.run_training_arm = train
    for name, module in {"torch": torch, "guandan_marvel": game,
                         "guandan_marvel.training.checkpoints": checks,
                         "guandan_marvel.training.smoke": smoke,
                         "serial_compare.training": training}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return torch, state


def args(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"source": {"backend_hash": "test-backend"},
                                "evaluation": {"horizon": 1024}}), encoding="utf8")
    return ["--plan", str(plan), "--arm", "control", "--dataset", str(tmp_path),
            "--source", str(tmp_path), "--run-dir", str(tmp_path / "run"),
            "--result", str(tmp_path / "result.json"), "--max-seconds", "60"]


@pytest.mark.parametrize("rank", [0, 1])
def test_real_main_reads_torchrun_before_process_group_init(tmp_path, monkeypatch, rank):
    configure(monkeypatch, environment(rank))
    _, state = fake_runtime(monkeypatch)
    assert arm_worker.main(args(tmp_path)) == 0
    captured = state["calls"][0]
    assert (captured["world_size"], captured["rank"], captured["device"], captured["ddp_enabled"]) == (
        2, rank, f"cuda:{rank}", True)
    assert state["devices"] == [rank]
    assert state["destroyed"]
    assert (tmp_path / "result.json").exists() == (rank == 0)


def test_global_rank_is_not_the_local_cuda_index(tmp_path, monkeypatch):
    configure(monkeypatch, environment(rank=3, world=4, local=1))
    _, state = fake_runtime(monkeypatch)
    arm_worker.main(args(tmp_path))
    call = state["calls"][0]
    assert call["rank"] == 3 and call["world_size"] == 4 and call["device"] == "cuda:1"
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("key,value", [
    ("WORLD_SIZE", "1"), ("WORLD_SIZE", "bad"), ("WORLD_SIZE", "0"),
    ("RANK", "2"), ("RANK", "-1"), ("LOCAL_RANK", "-1"),
    ("MASTER_PORT", "0"), ("MASTER_PORT", "65536"),
    ("LOCAL_WORLD_SIZE", "1"), ("LOCAL_WORLD_SIZE", "bad"),
])
def test_invalid_environment_has_no_single_rank_fallback(key, value):
    env = environment(rank=1)
    env[key] = value
    with pytest.raises(ValueError):
        arm_worker.read_launch_context(env)


@pytest.mark.parametrize("key", list(environment()))
def test_missing_environment_is_rejected(key):
    env = environment()
    env.pop(key)
    with pytest.raises(ValueError, match="incomplete torchrun"):
        arm_worker.read_launch_context(env)


@pytest.mark.parametrize("pg_rank,pg_world,backend", [(1, 2, "nccl"), (0, 4, "nccl"), (0, 2, "gloo")])
def test_initialized_group_must_match_environment(monkeypatch, pg_rank, pg_world, backend):
    torch, _ = fake_runtime(monkeypatch, initialized=True, pg_rank=pg_rank, pg_world=pg_world, backend=backend)
    with pytest.raises(RuntimeError):
        arm_worker.validate_launch_device(torch, arm_worker.read_launch_context(environment()))


def test_cuda_unavailable_does_not_fallback_to_cpu(monkeypatch):
    torch, _ = fake_runtime(monkeypatch)
    torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="CPU fallback"):
        arm_worker.validate_launch_device(torch, arm_worker.read_launch_context(environment()))


def test_local_rank_outside_visible_devices_is_rejected(monkeypatch):
    torch, _ = fake_runtime(monkeypatch)
    with pytest.raises(RuntimeError, match="visible CUDA"):
        arm_worker.validate_launch_device(torch, arm_worker.read_launch_context(environment(local=2)))


def test_real_main_records_original_rank_exception_and_cleans_group(tmp_path, monkeypatch, capsys):
    configure(monkeypatch, environment(rank=1))
    _, state = fake_runtime(monkeypatch, fail="independent injected learner exception")
    with pytest.raises(RuntimeError, match="independent injected"):
        arm_worker.main(args(tmp_path))
    record = json.loads((tmp_path / "run/worker_errors/rank001.json").read_text(encoding="utf8"))
    assert record["rank"] == 1 and record["stage"] == "train_arm"
    assert record["type"] == "RuntimeError" and "independent injected" in record["traceback"]
    assert "independent injected" in capsys.readouterr().err
    assert state["destroyed"] and not (tmp_path / "result.json").exists()


def test_cli_early_environment_failure_has_original_traceback(tmp_path):
    env = dict(os.environ)
    for key in (*environment(), "LOCAL_WORLD_SIZE"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO / "tools")
    env["TORCHELASTIC_ERROR_FILE"] = str(tmp_path / "elastic_error.json")
    completed = subprocess.run([sys.executable, "-m", "serial_compare.arm_worker", *args(tmp_path)],
                               cwd=REPO, env=env, capture_output=True, text=True, timeout=45)
    assert completed.returncode != 0
    assert "incomplete torchrun environment" in completed.stderr
    record = json.loads((tmp_path / "run/worker_errors/rankunknown.json").read_text(encoding="utf8"))
    assert record["stage"] == "torchrun_environment"
    assert (tmp_path / "elastic_error.json").is_file()


def test_two_rank_gloo_launch_uses_environment_without_cpu_model(tmp_path):
    from tests.marvel_training._ddp_launcher import launch_two_ranks
    outputs = launch_two_ranks(REPO / "tests/marvel_training/_serial_compare_launch_worker.py",
                               "--output-dir", str(tmp_path), cwd=REPO, timeout=60)
    assert [row.returncode for row in outputs] == [0, 0], [row.stdout for row in outputs]
    for rank in (0, 1):
        data = json.loads((tmp_path / f"rank{rank}.json").read_text(encoding="utf8"))
        assert data == {"rank": rank, "local_rank": rank, "world_size": 2, "reduced": 3}
