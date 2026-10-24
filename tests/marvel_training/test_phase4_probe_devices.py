"""Device selection is explicit and independent of packaged Kaggle semantics."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch

from guandan_marvel.kaggle.phase4_delivery import (
    _build_local_probe_config, _training_block,
    resolve_phase4_probe_device, run_phase4_runtime_probe,
)
from guandan_marvel.training.smoke import CppSmokeGame


@pytest.mark.parametrize("available,expected", [(False, "cpu"), (True, "cuda:0")])
def test_auto_probe_device_selects_available_backend(monkeypatch, available, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1 if available else 0)
    assert str(resolve_phase4_probe_device("auto")) == expected


def test_explicit_cpu_stays_cpu_on_cuda_host(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(resolve_phase4_probe_device("cpu")) == "cpu"


def test_explicit_cuda_unavailable_fails_without_fallback_or_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    work = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        run_phase4_runtime_probe({}, warm_start_path=tmp_path / "absent.pt",
                                 game=None, work_dir=work, device="cuda")
    assert not work.exists()


@pytest.mark.parametrize("bad", ["mps", "cpu:1", "not-a-device", "cuda:7"])
def test_invalid_or_missing_probe_device_is_rejected(monkeypatch, bad):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError):
        resolve_phase4_probe_device(bad)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_local_probe_config_does_not_mutate_packaged_two_t4_profile(tmp_path, device):
    training = _training_block()
    profile = {"mode": "full", "algorithm": "ippo", "training": training,
               "ddp": {"world_size": 1, "backend": "gloo", "require_cuda": device != "cpu"}}
    original = deepcopy(profile)
    cfg = _build_local_probe_config(profile, game=CppSmokeGame(512), work_dir=tmp_path,
                                    device=torch.device(device))
    assert cfg.device == device
    assert not cfg.ddp.enabled
    assert cfg.amp_enabled and cfg.model.architecture == "v1.4-macro-fuse"
    assert profile == original
    assert _training_block() == training


def test_local_probe_rejects_multi_rank_launch_before_creating_output(tmp_path):
    work = tmp_path / "unused"
    with pytest.raises(ValueError, match="one rank only"):
        run_phase4_runtime_probe({}, warm_start_path=tmp_path / "absent.pt", game=None,
                                 work_dir=work, ddp_world_size=2, device="cpu")
    assert not work.exists()


def test_local_probe_rejects_zero_iterations_before_creating_output(tmp_path):
    work = tmp_path / "unused"
    with pytest.raises(ValueError, match="iterations must be positive"):
        run_phase4_runtime_probe({}, warm_start_path=tmp_path / "absent.pt", game=None,
                                 work_dir=work, iterations=0, device="cpu")
    assert not work.exists()


@pytest.mark.parametrize("kind", ["build", "verify"])
def test_delivery_cli_exposes_explicit_runtime_device(kind, capsys):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{kind}_marvel_v14_phase4_delivery.py"
    spec = importlib.util.spec_from_file_location(f"phase4_device_{kind}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as stopped:
        module.main(["--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--device {auto,cpu,cuda}" in help_text
    assert "--runtime-max-game-horizon" in help_text


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_local_probe_rejects_invalid_scaler_state_before_writes(tmp_path, bad):
    work = tmp_path / "unused"
    with pytest.raises(ValueError, match="initial scale"):
        run_phase4_runtime_probe({}, warm_start_path=tmp_path / "absent.pt", game=None,
                                 work_dir=work, device="cpu", cuda_init_scale=bad)
    assert not work.exists()


def test_exact_probe_state_check_detects_optimizer_and_dtype_corruption():
    from guandan_marvel.kaggle.phase4_delivery import _probe_state_equal

    good = {"optimizer": [{"step": torch.tensor(1.), "moment": torch.ones(2)}]}
    assert _probe_state_equal(good, deepcopy(good))
    bad = deepcopy(good)
    bad["optimizer"][0]["moment"][0] += 1
    assert not _probe_state_equal(good, bad)
    assert not _probe_state_equal(torch.ones(2), torch.ones(2, dtype=torch.bfloat16))
    assert not _probe_state_equal({"a": 1}, {"b": 1})
    assert not _probe_state_equal([1], [1, 2])


def test_fixed_migration_probe_is_fp32_even_inside_cpu_autocast():
    from types import SimpleNamespace
    from guandan_marvel.kaggle.phase4_delivery import _capture_policy_outputs

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_head = torch.nn.Linear(2, 1)

        def forward(self, *, rollout):
            values = self.logit_head(rollout)
            return SimpleNamespace(avail_probs=values, avail_log_probs=values,
                                   player_state_value=values)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = _capture_policy_outputs(Probe(), torch.ones(1, 2))
    assert all(value.dtype is torch.float32 for value in result.values())


def test_builder_explicit_cuda_fails_before_replacing_existing_delivery(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_marvel_v14_phase4_delivery.py"
    spec = importlib.util.spec_from_file_location("phase4_builder_fail_fast", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "delivery"
    output.mkdir()
    marker = output / "preserved.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        module.main(["--output", str(output), "--runtime-probe", "--device", "cuda"])
    assert marker.read_text(encoding="utf-8") == "original"
    assert not (tmp_path / "phase4_notebook_template.ipynb").exists()
