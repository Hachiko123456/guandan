"""Phase-6 formal training-gate tools: dry-run, absolute targets and hard caps."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from guandan_marvel.phase6.design import (
    FORMAL_CONFIRMATION_TOKEN, FORMAL_GATE_TARGETS, GATE_EVIDENCE_SCHEMA,
)
from guandan_marvel.phase6.training_gates import (
    FORMAL_GATES_NOT_RUN, GateBudget, GateRunner, TrainingGateError,
    absolute_target_iterations, build_gate_evidence, decide_strength,
    evaluate_health, plan_gate_run,
)


class FakeTrainer:
    def __init__(self, *, start_iteration=0, games_per_iter=1, resume_from=None):
        self.iteration = int(start_iteration)
        self.completed_games_total = 0
        self.games_per_iter = int(games_per_iter)
        self.checkpoints: list[int] = []

    def train_iteration(self):
        assert not self.iteration >= 10**6
        self.iteration += 1
        self.completed_games_total += self.games_per_iter
        return {"iteration": self.iteration}

    def save_checkpoint(self, path):
        self.checkpoints.append(self.iteration)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"iteration={self.iteration}", encoding="utf-8")
        return destination


def _factory(**kwargs):
    return FakeTrainer(**kwargs)


def test_gate_budget_validation_and_absolute_targets():
    assert [absolute_target_iterations(gate) for gate in (1, 2, 3, 4, 5)] == [
        50, 250, 500, 1000, 1500
    ]
    assert FORMAL_GATE_TARGETS[1] == 50
    with pytest.raises(TrainingGateError, match="gate must be"):
        GateBudget(gate=9, max_iterations=1)
    with pytest.raises(TrainingGateError, match="at least one hard limit"):
        GateBudget(gate=1)
    with pytest.raises(TrainingGateError, match="1-2 iterations"):
        GateBudget(gate=1, max_iterations=3, smoke=True)
    with pytest.raises(TrainingGateError, match="max_iterations"):
        GateBudget(gate=1, max_iterations=0)
    budget = GateBudget(gate=2, completed_iterations=100, max_iterations=10)
    assert budget.target_iterations == 250
    assert budget.remaining_iterations == 150
    assert GateBudget(gate=2, completed_iterations=250, max_iterations=10).remaining_iterations == 0
    assert GateBudget(
        gate=2, completed_iterations=400, max_iterations=10,
    ).remaining_iterations == 0
    assert GateBudget(
        gate=2, completed_games=4000, target_games=5000, max_iterations=10,
    ).remaining_games == 1000


def test_dry_run_constructs_no_trainer_and_trains_zero(tmp_path):
    def factory(**kwargs):
        raise AssertionError("dry-run must not construct a trainer")

    budget = GateBudget(gate=1, completed_iterations=10, max_iterations=2)
    plan = plan_gate_run(budget)
    assert plan["status"] == "dry_run"
    assert plan["will_train"] is False
    runner = GateRunner(
        trainer_factory=factory, budget=budget,
        checkpoint_path=tmp_path / "never.pt", output_dir=tmp_path / "out",
    )
    record = runner.run()
    assert record["trainer_constructed"] is False
    assert record["trained_iterations"] == 0
    assert record["checkpoint"] is None
    assert record["dry_run"] is True
    assert record["formal_gates_not_run"] is True


def test_formal_target_requires_explicit_confirmation(tmp_path):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return FakeTrainer(**kwargs)

    budget = GateBudget(
        gate=1, completed_iterations=0, max_iterations=2, dry_run=False,
    )
    record = GateRunner(
        trainer_factory=factory, budget=budget,
        checkpoint_path=tmp_path / "gate.pt", output_dir=tmp_path / "out",
    ).run()
    assert record["status"] == "confirmation_required"
    assert record["trainer_constructed"] is False
    assert calls == []


def test_already_satisfied_target_trains_nothing(tmp_path):
    def factory(**kwargs):
        raise AssertionError("a satisfied target must not construct a trainer")

    budget = GateBudget(
        gate=1, completed_iterations=60, max_iterations=2, dry_run=False,
        formal_confirmation=FORMAL_CONFIRMATION_TOKEN,
    )
    record = GateRunner(
        trainer_factory=factory, budget=budget,
        checkpoint_path=tmp_path / "gate.pt", output_dir=tmp_path / "out",
    ).run()
    # The declared counter has no checkpoint/counter_reader to verify it, so
    # the tool must not claim a verified or formal completion.
    assert record["status"] == "already_satisfied_unverified"
    assert record["counter_verified"] is False
    assert record["trainer_constructed"] is False
    assert record["remaining_iterations"] == 0
    assert record["formal_gate_reached"] is False


def test_iteration_and_game_caps_stop_at_a_checkpoint_boundary(tmp_path):
    output = tmp_path / "out"
    budget = GateBudget(
        gate=1, completed_iterations=0, max_iterations=2, dry_run=False,
        formal_confirmation=FORMAL_CONFIRMATION_TOKEN, smoke=True,
    )
    trainer = FakeTrainer(games_per_iter=3)
    record = GateRunner(
        trainer_factory=lambda **kwargs: trainer, budget=budget,
        checkpoint_path=tmp_path / "gate.pt", output_dir=output,
    ).run()
    assert record["status"] == "budget_stopped"
    assert record["stop_reason"] == "max_iterations"
    assert record["trained_iterations"] == 2
    assert record["completed_iterations"] == 2
    assert record["remaining_iterations"] == 48
    assert record["checkpoint_written_at_iteration_boundary"] is True
    assert record["formal_gate_reached"] is False
    assert record["smoke"] is True
    assert (output / "gate1_run.json").is_file()

    games_budget = GateBudget(
        gate=1, completed_iterations=0, max_games=4, max_iterations=2,
        dry_run=False, formal_confirmation=FORMAL_CONFIRMATION_TOKEN, smoke=True,
    )
    trainer = FakeTrainer(games_per_iter=3)
    record = GateRunner(
        trainer_factory=lambda **kwargs: trainer, budget=games_budget,
        checkpoint_path=tmp_path / "games.pt", output_dir=output,
    ).run()
    # The first iteration played 3 games; the next known 3 would cross the hard
    # cap of 4, so the runner stops before it without exceeding the budget.
    assert record["stop_reason"] == "max_games"
    assert record["trained_iterations"] == 1
    assert record["trained_games"] == 3
    assert record["budget_exceeded"] is False

    target_games = GateBudget(
        gate=1, completed_iterations=0, target_games=3, max_iterations=2,
        dry_run=False, formal_confirmation=FORMAL_CONFIRMATION_TOKEN, smoke=True,
    )
    trainer = FakeTrainer(games_per_iter=3)
    record = GateRunner(
        trainer_factory=lambda **kwargs: trainer, budget=target_games,
        checkpoint_path=tmp_path / "target-games.pt", output_dir=output,
    ).run()
    assert record["stop_reason"] == "target_games_reached"
    assert record["trained_games"] == 3


def test_resume_near_target_uses_absolute_remaining_and_labels_smoke(tmp_path):
    trainer = FakeTrainer(start_iteration=49, games_per_iter=1)
    budget = GateBudget(
        gate=1, completed_iterations=49, max_iterations=2, dry_run=False,
        formal_confirmation=FORMAL_CONFIRMATION_TOKEN, smoke=True,
    )
    record = GateRunner(
        trainer_factory=lambda **kwargs: trainer, budget=budget,
        checkpoint_path=tmp_path / "near.pt", output_dir=tmp_path / "out",
    ).run()
    assert record["trained_iterations"] == 1
    assert record["completed_iterations"] == 50
    assert record["remaining_iterations"] == 0
    assert record["formal_gate_reached"] is False  # smoke never claims a formal gate


def test_health_gate_and_strength_decision_are_pre_registered():
    healthy = {
        "macro/value_ev": 0.2,
        "runtime/amp_skipped_steps": 0,
        "statistics/learner_updates": 10,
        "macro/illegal_active_action_count": 0,
    }
    assert evaluate_health(healthy, learner_updates=10)["passed"] is True
    assert evaluate_health({**healthy, "loss": float("nan")}, learner_updates=10)["passed"] is False
    assert evaluate_health(
        {**healthy, "runtime/amp_skipped_steps": 2}, learner_updates=10,
    )["passed"] is False
    assert evaluate_health(
        {**healthy, "macro/value_ev": None}, learner_updates=10,
    )["passed"] is False
    assert evaluate_health(
        {**healthy, "macro/illegal_active_action_count": 1}, learner_updates=10,
    )["passed"] is False

    continue_decision = decide_strength(
        ci_low=0.01, ci_high=0.2, health_passed=True, recovery_passed=True,
    )
    assert continue_decision["recommendation"] == "CONTINUE"
    assert continue_decision["auto_expand"] is False
    assert decide_strength(
        ci_low=-0.2, ci_high=-0.01, health_passed=True, recovery_passed=True,
    )["recommendation"] == "STOP"
    inconclusive = decide_strength(
        ci_low=-0.01, ci_high=0.01, health_passed=True, recovery_passed=True,
    )
    assert inconclusive["recommendation"] == "INCONCLUSIVE"
    assert inconclusive["auto_expand"] is False
    assert decide_strength(
        ci_low=0.01, ci_high=0.2, health_passed=False, recovery_passed=True,
    )["recommendation"] == "INCONCLUSIVE"


def test_gate_evidence_records_tools_only_and_formal_not_run():
    run_record = {"hash": "a" * 64, "formal_gate_reached": False}
    evidence = build_gate_evidence(
        gate=1,
        run_record=run_record,
        health={"passed": True},
        strength={"recommendation": "INCONCLUSIVE"},
        recovery={"exact_resume": True},
        evaluation=None,
    )
    assert evidence["schema"] == GATE_EVIDENCE_SCHEMA
    assert evidence["absolute_target_iterations"] == 50
    assert evidence["formal_gates_not_run"] is True
    assert evidence["formal_gate_reached"] is False
    assert FORMAL_GATES_NOT_RUN is True
