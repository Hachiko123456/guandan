"""Phase-6 repair: hard caps, verified counters and recovery reload."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from guandan_marvel.phase6.design import FORMAL_CONFIRMATION_TOKEN
from guandan_marvel.phase6.training_gates import GateBudget, GateRunner


class FakeTrainer:
    def __init__(self, *, start_iteration=0, games_per_iter=1, sleep=0.0, resume_from=None):
        self.iteration = int(start_iteration)
        self.completed_games_total = 0
        self.games_per_iter = int(games_per_iter)
        self.sleep = float(sleep)
        self.last_metrics = {
            "macro/value_ev": 0.2,
            "runtime/amp_skipped_steps": 0,
            "statistics/learner_updates": 1,
            "macro/illegal_active_action_count": 0,
        }

    def train_iteration(self):
        if self.sleep:
            time.sleep(self.sleep)
        self.iteration += 1
        self.completed_games_total += self.games_per_iter

    def save_checkpoint(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"iteration": self.iteration}, destination)
        return destination


def _budget(**kwargs):
    base = dict(gate=1, completed_iterations=0, dry_run=False,
                formal_confirmation=FORMAL_CONFIRMATION_TOKEN, smoke=True)
    base.update(kwargs)
    return GateBudget(**base)


def _runner(tmp_path, trainer, budget, **kwargs):
    return GateRunner(
        trainer_factory=lambda **_: trainer, budget=budget,
        checkpoint_path=tmp_path / "gate.pt", output_dir=tmp_path / "out",
        **kwargs,
    )


def test_max_games_stops_before_a_known_over_budget_iteration(tmp_path):
    trainer = FakeTrainer(games_per_iter=3)
    record = _runner(
        tmp_path, trainer, _budget(max_games=4, max_iterations=2),
    ).run()
    assert record["stop_reason"] == "max_games"
    assert record["trained_games"] == 3
    assert record["budget_exceeded"] is False


def test_unknown_first_iteration_over_budget_is_marked_exceeded(tmp_path):
    trainer = FakeTrainer(games_per_iter=10)
    record = _runner(
        tmp_path, trainer, _budget(max_games=4, max_iterations=2),
    ).run()
    assert record["stop_reason"] == "max_games_exceeded"
    assert record["budget_exceeded"] is True
    assert record["trained_games"] == 10


def test_wall_budget_exceeded_is_marked(tmp_path):
    trainer = FakeTrainer(games_per_iter=1, sleep=0.15)
    record = _runner(
        tmp_path, trainer, _budget(max_wall_seconds=0.05, max_iterations=5, smoke=False),
    ).run()
    assert record["stop_reason"] == "max_wall_seconds_exceeded"
    assert record["budget_exceeded"] is True


def test_already_satisfied_uses_the_actual_checkpoint_counter(tmp_path):
    checkpoint = tmp_path / "gate.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": 50}, checkpoint)
    budget = _budget(completed_iterations=50, max_iterations=2)
    record = _runner(tmp_path, FakeTrainer(), budget).run()
    assert record["status"] == "already_satisfied"
    assert record["counter_verified"] is True
    assert record["actual_completed_iterations"] == 50
    assert record["formal_gate_reached"] is False

    missing = _runner(
        tmp_path / "none", FakeTrainer(), _budget(completed_iterations=50, max_iterations=2),
    ).run()
    assert missing["status"] == "already_satisfied_unverified"
    assert missing["counter_verified"] is False
    assert missing["formal_gate_reached"] is False


def test_declared_counter_mismatch_is_reported(tmp_path):
    checkpoint = tmp_path / "gate.pt"
    torch.save({"iteration": 40}, checkpoint)
    record = _runner(
        tmp_path, FakeTrainer(), _budget(completed_iterations=50, max_iterations=2),
    ).run()
    assert record["status"] == "counter_mismatch"
    assert record["actual_completed_iterations"] == 40
    assert record["remaining_iterations"] == 10
    assert record["formal_gate_reached"] is False


def test_near_target_recovery_reload_and_health_are_recorded(tmp_path):
    trainer = FakeTrainer(start_iteration=49, games_per_iter=1)
    reloaded = []

    def recovery_loader(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["iteration"] == 50
        reloaded.append(path)

    budget = _budget(
        completed_iterations=49, max_iterations=1, smoke=False,
    )
    record = _runner(
        tmp_path, trainer, budget, recovery_loader=recovery_loader,
    ).run()
    assert record["completed_iterations"] == 50
    assert record["remaining_iterations"] == 0
    assert record["health"]["passed"] is True
    assert record["recovery"]["verified"] is True
    assert record["formal_gate_reached"] is True
    assert reloaded
