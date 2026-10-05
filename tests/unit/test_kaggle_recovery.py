"""A07 recovery fixtures: local CPU only, never remote_full acceptance counts.

The trainer is real. Only time/stop delivery is injected; weights, optimizer,
collector/environment bytes, RNG and actual resumed updates are not mocked.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import shutil
import signal

import pytest
import torch

from guandan.deployment.control import CheckpointController, CombinedDeadline, SessionStop
from guandan.env_batch import GuandanEnvBatch
from guandan.training.checkpoint import load_checkpoint
from guandan.training.runtime import BudgetExceeded, Deadline, file_sha256
from guandan.training.trainer import train


class FakeClock:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def assert_tree_equal(actual, expected, path="checkpoint"):
    """Exact equality, including tensor dtype/shape; no tolerance hides drift."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert actual.dtype == expected.dtype, path
        assert actual.shape == expected.shape, path
        assert torch.equal(actual.cpu(), expected.cpu()), path
    elif isinstance(expected, Mapping):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_tree_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), path
        assert len(actual) == len(expected), path
        for index, item in enumerate(expected):
            assert_tree_equal(actual[index], item, f"{path}[{index}]")
    else:
        assert type(actual) is type(expected), path
        assert actual == expected, path


def assert_checkpoint_equal(actual_path, expected_path):
    actual, expected = load_checkpoint(actual_path), load_checkpoint(expected_path)
    assert actual.metadata == expected.metadata
    assert_tree_equal(actual.model_state, expected.model_state, "weights")
    assert_tree_equal(actual.optimizer_state, expected.optimizer_state, "optimizer")
    assert_tree_equal(actual.model_config, expected.model_config, "architecture")
    assert_tree_equal(actual.training_state, expected.training_state, "resume_state")


@pytest.fixture(autouse=True)
def local_fixture_profile(monkeypatch):
    # These explicitly scoped lifecycle fixtures must not inherit a remote gate.
    monkeypatch.delenv("GUANDAN_PROFILE", raising=False)
    monkeypatch.delenv("GUANDAN_RESOLVED_PROFILE_JSON", raising=False)


@pytest.mark.parametrize("field", ["session_hours", "save_margin_seconds", "checkpoint_seconds"])
@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_checkpoint_policy_rejects_invalid_budgets(field, value):
    settings = {"session_hours": 1, "save_margin_seconds": 30, "checkpoint_seconds": 60}
    settings[field] = value
    with pytest.raises(ValueError):
        CheckpointController(**settings)


@pytest.mark.parametrize("margin", [3600, 3601])
def test_save_margin_must_leave_positive_work_budget(margin):
    with pytest.raises(ValueError):
        CheckpointController(session_hours=1, save_margin_seconds=margin)


def test_default_session_policy_and_exact_soft_boundary():
    clock = FakeClock(73)
    controller = CheckpointController(clock=clock)
    assert controller.hard_seconds == 10 * 3600
    assert controller.soft_seconds == 10 * 3600 - 300
    assert controller.interval == 600
    clock.advance(controller.soft_seconds - 0.001)
    controller.check()
    clock.advance(0.001)
    with pytest.raises(SessionStop, match="session_save_margin"):
        controller.check()
    assert controller.reason == "session_save_margin"


def test_periodic_save_is_measured_from_last_successful_save():
    clock = FakeClock(99)
    controller = CheckpointController(clock=clock)
    clock.advance(599)
    assert not controller.checkpoint_due()
    clock.advance(1)
    assert controller.checkpoint_due()
    # Asking whether a save is due must not mark it durable.
    assert controller.checkpoint_due()
    controller.checkpoint_saved()
    assert not controller.checkpoint_due()
    clock.advance(600)
    assert controller.checkpoint_due()


def test_predictive_stop_uses_longest_completed_update_and_save_margin():
    clock = FakeClock()
    controller = CheckpointController(session_hours=1, save_margin_seconds=300, clock=clock)
    controller.after_update(200)
    controller.after_update(10)
    clock.value = 2999
    controller.before_update()
    clock.value = 3000
    with pytest.raises(SessionStop, match="insufficient_time_for_next_update"):
        controller.before_update()
    assert controller.completed_update_seconds == [200.0, 10.0]


def test_requested_stop_is_cooperative_and_sticky():
    controller = CheckpointController(clock=FakeClock())
    assert controller.request_stop("operator_stop") is None
    for check in (controller.check, controller.before_update):
        with pytest.raises(SessionStop, match="operator_stop"):
            check()


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_signal_handler_requests_stop_and_restores_original_even_on_error(number):
    controller = CheckpointController(clock=FakeClock())
    originals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(RuntimeError, match="fixture-body"):
        with controller.signal_handlers():
            handler = signal.getsignal(number)
            assert callable(handler)
            # Deliver to the installed handler, not the OS/pytest process.
            handler(number, None)
            with pytest.raises(SessionStop, match=f"signal_{int(number)}"):
                controller.check()
            raise RuntimeError("fixture-body")
    assert {sig: signal.getsignal(sig) for sig in originals} == originals


@pytest.mark.parametrize("expired", ["profile", "session"])
def test_combined_deadline_honors_both_independent_budgets(expired):
    profile_clock, session_clock = FakeClock(), FakeClock()
    profile = Deadline(1, clock=profile_clock)
    controller = CheckpointController(session_hours=1, save_margin_seconds=60, clock=session_clock)
    combined = CombinedDeadline(profile, controller)
    combined.check(stage="healthy")
    if expired == "profile":
        profile_clock.value = 3600
    else:
        session_clock.value = 3540
    with pytest.raises(BudgetExceeded):
        combined.check(stage="fixture")


def test_cooperative_policy_is_optional_and_does_not_change_train_defaults():
    parameters = inspect.signature(train).parameters
    assert parameters["checkpoint_controller"].default is None
    assert parameters["profile"].default == "local_fast"
    assert parameters["updates"].default is None
    assert parameters["rollout_envs"].default is None
    assert parameters["rollout_steps"].default is None
    assert parameters["epochs"].default == 2


def tiny_train(directory, *, algorithm="ippo", **kwargs):
    return train(
        algorithm=algorithm, profile="local_fast", device="cpu", rollout_envs=1,
        rollout_steps=1, hidden_dim=8, seed=10_017, checkpoint_dir=directory, **kwargs,
    )


class EveryUpdateController(CheckpointController):
    """Fake time makes every completed update due, before profile interval 5/10."""
    def __init__(self):
        self.fake_clock = FakeClock()
        super().__init__(session_hours=1, save_margin_seconds=60,
                         checkpoint_seconds=1, clock=self.fake_clock)
        self.saves = 0

    def after_update(self, elapsed):
        super().after_update(elapsed)
        self.fake_clock.advance(1)

    def checkpoint_saved(self):
        super().checkpoint_saved()
        self.saves += 1


@pytest.fixture(scope="module", params=["ippo", "vrpo"])
def uninterrupted(request, tmp_path_factory):
    # Explicit context because this module fixture precedes function autouse fixtures.
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("GUANDAN_PROFILE", raising=False)
        patch.delenv("GUANDAN_RESOLVED_PROFILE_JSON", raising=False)
        controller = EveryUpdateController()
        directory = tmp_path_factory.mktemp(f"a07_uninterrupted_{request.param}")
        result = tiny_train(directory, algorithm=request.param, updates=2,
                            checkpoint_controller=controller)
    assert result.status == "complete", asdict(result)
    assert result.updates == result.durable_updates == 2
    assert result.total_token_steps == result.durable_token_steps == 2
    assert not result.profile_counts_match
    assert result.runtime["device"] == "cpu"
    assert [row["update"] for row in result.checkpoints] == [1, 2]
    assert controller.saves == 2
    return request.param, result


def test_time_trigger_saves_first_update_before_tenth(uninterrupted):
    algorithm, result = uninterrupted
    first = load_checkpoint(result.checkpoints[0]["path"], expected_algorithm=algorithm)
    assert first.metadata.update_count == 1 < 10
    assert first.training_state["engine"]["collector"]["total_token_steps"] == 1
    assert first.training_state["engine"]["optimizer_steps"] == 2
    manifest = json.loads(Path(result.recovery_manifest).read_text(encoding="utf-8"))
    assert manifest["format"] == "guandan-recovery-v1"
    assert manifest["checkpoint_filename"] == Path(result.checkpoint).name
    assert manifest["checkpoint_sha256"] == file_sha256(result.checkpoint)
    assert manifest["durable_updates"] == manifest["durable_token_steps"] == 2
    assert manifest["accepted"] is False and manifest["remote_verified"] is False


class InjectedStopController(CheckpointController):
    def __init__(self, phase):
        self.phase = phase
        self.fake_clock = FakeClock()
        super().__init__(session_hours=1, save_margin_seconds=60,
                         checkpoint_seconds=600, clock=self.fake_clock)

    def before_update(self):
        completed = len(self.completed_update_seconds)
        if self.phase == "before_first" or (self.phase == "before_second" and completed == 1):
            self.request_stop(f"fixture_{self.phase}")
        super().before_update()

    def after_update(self, elapsed):
        super().after_update(elapsed)
        if self.phase == "after_first" and len(self.completed_update_seconds) == 1:
            self.request_stop("fixture_after_first")


@pytest.mark.parametrize("phase", [
    "before_first", "before_second", "after_first", "soft_mid_rollout",
    "sigterm_mid_rollout", "soft_mid_optimizer",
])
def test_interrupted_real_training_restores_durable_state_then_really_continues(
    phase, uninterrupted, tmp_path, monkeypatch,
):
    algorithm, baseline = uninterrupted
    controller = InjectedStopController(phase)
    with monkeypatch.context() as injection:
        original_step = GuandanEnvBatch.step
        sampled = 0

        def step_and_deliver_stop(batch, actions):
            nonlocal sampled
            receipt = original_step(batch, actions)
            sampled += 1
            if sampled == 2 and phase == "soft_mid_rollout":
                controller.fake_clock.value = controller.started + controller.soft_seconds
            if sampled == 2 and phase == "sigterm_mid_rollout":
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return receipt

        injection.setattr(GuandanEnvBatch, "step", step_and_deliver_stop)
        original_optimizer_step = torch.optim.Adam.step
        optimizer_calls = 0

        def optimizer_step_and_stop(optimizer, *args, **kwargs):
            nonlocal optimizer_calls
            value = original_optimizer_step(optimizer, *args, **kwargs)
            optimizer_calls += 1
            if optimizer_calls == 3 and phase == "soft_mid_optimizer":
                controller.fake_clock.value = controller.started + controller.soft_seconds
            return value

        injection.setattr(torch.optim.Adam, "step", optimizer_step_and_stop)
        with controller.signal_handlers():
            stopped = tiny_train(tmp_path / "old_working", algorithm=algorithm, updates=2,
                                 checkpoint_controller=controller)

    durable = 0 if phase == "before_first" else 1
    discarded = int(phase in {"soft_mid_rollout", "sigterm_mid_rollout", "soft_mid_optimizer"})
    assert stopped.status == "incomplete", asdict(stopped)
    assert stopped.target_updates == 2
    assert stopped.updates == stopped.durable_updates == stopped.lifetime_updates == durable
    assert stopped.durable_token_steps == stopped.lifetime_token_steps == durable
    assert stopped.discarded_partial_steps == discarded
    assert stopped.discarded_optimizer_steps == int(phase == "soft_mid_optimizer")
    assert stopped.total_token_steps == durable + discarded == sampled
    assert len(stopped.metrics) == durable
    assert stopped.profile == "local_fast" and not stopped.profile_counts_match
    assert stopped.runtime["device"] == "cpu"
    assert stopped.reason
    assert stopped.checkpoint is not None
    assert len(stopped.checkpoints) == 1  # Interrupt must save even before periodic threshold.
    checkpoint = load_checkpoint(stopped.checkpoint, expected_algorithm=algorithm)
    state = checkpoint.training_state["engine"]
    assert checkpoint.metadata.update_count == state["update_count"] == durable
    assert state["optimizer_steps"] == durable * 2
    assert state["collector"]["total_token_steps"] == durable
    assert state["collector"]["incomplete"] is False
    assert len(state["collector"]["envs"]) == 1
    assert all(isinstance(raw, bytes) for raw in state["collector"]["envs"])
    assert state["torch_rng_state"].dtype == torch.uint8
    assert state["collector"]["rng_state"].dtype == torch.uint8
    if durable:
        assert_checkpoint_equal(stopped.checkpoint, baseline.checkpoints[0]["path"])

    evidence = json.loads(Path(stopped.evidence).read_text(encoding="utf-8"))
    for key in ("status", "durable_updates", "durable_token_steps", "discarded_partial_steps",
                "discarded_optimizer_steps", "total_token_steps", "updates"):
        assert evidence[key] == getattr(stopped, key)
    assert evidence["status"] == "incomplete"

    # Simulate a new input mount by copying the pair, then REMOVE the original
    # checkpoint/manifest. Temp paths only; success cannot rely on old absolute paths.
    input_mount = tmp_path / "new_input" / "restored_dataset"
    input_mount.mkdir(parents=True)
    copied_checkpoint = input_mount / Path(stopped.checkpoint).name
    copied_manifest = input_mount / "manifest.json"
    shutil.copyfile(stopped.checkpoint, copied_checkpoint)
    shutil.copyfile(stopped.recovery_manifest, copied_manifest)
    original_digest = file_sha256(copied_checkpoint)
    Path(stopped.checkpoint).unlink()
    Path(stopped.recovery_manifest).unlink()
    manifest = json.loads(copied_manifest.read_text(encoding="utf-8"))
    assert manifest["checkpoint_filename"] == copied_checkpoint.name
    assert manifest["checkpoint_sha256"] == original_digest
    assert str(tmp_path / "old_working") not in copied_manifest.read_text(encoding="utf-8")

    continued = tiny_train(tmp_path / "new_working", algorithm=algorithm, updates=2-durable,
                           resume=copied_checkpoint, checkpoint_controller=CheckpointController())
    assert continued.status == "complete", asdict(continued)
    assert continued.start_update == durable
    assert continued.updates == continued.total_token_steps == 2-durable
    assert continued.durable_updates == continued.lifetime_updates == 2
    assert continued.durable_token_steps == continued.lifetime_token_steps == 2
    assert continued.discarded_partial_steps == continued.discarded_optimizer_steps == 0
    assert continued.resume_sha256 == original_digest
    assert continued.resumed_from == str(copied_checkpoint.resolve())
    assert file_sha256(copied_checkpoint) == original_digest  # Input is not overwritten.
    assert_checkpoint_equal(continued.checkpoint, baseline.checkpoint)
    assert [row["rollout_sha256"] for row in continued.metrics] == [
        row["rollout_sha256"] for row in baseline.metrics[durable:]
    ]
    if durable:
        assert any(not torch.equal(value, checkpoint.model_state[name])
                   for name, value in load_checkpoint(continued.checkpoint).model_state.items())


def test_training_progress_callback_reports_real_completed_updates(tmp_path):
    events = []
    result = tiny_train(tmp_path, updates=2, progress_callback=events.append)
    assert result.status == "complete"
    assert [event["update"] for event in events] == [1, 2]
    assert all(event["algorithm"] == "ippo" for event in events)
    assert events[-1]["lifetime_token_steps"] == 2


def test_existing_noncooperative_train_still_runs_real_updates(tmp_path):
    result = tiny_train(tmp_path, updates=2)
    assert result.status == "complete", asdict(result)
    assert result.updates == result.total_token_steps == 2
    assert result.profile_counts_match is False
    assert [row["update"] for row in result.checkpoints] == [2]
    assert load_checkpoint(result.checkpoint).metadata.update_count == 2


@pytest.fixture
def scoped_remote_metadata(uninterrupted, tmp_path, monkeypatch):
    """Synthetic remote envelope, NOT a remote run: actual CPU state has ONE step.

    The resolver's profile reader is scoped to 1env/1token only for these tests.
    The on-disk canonical profiles and trainer execution are never altered.
    """
    from copy import deepcopy
    from guandan.deployment import session
    from guandan.training.checkpoint import CURRENT_PROTOCOL_VERSIONS, save_checkpoint

    algorithm, baseline = uninterrupted
    actual = load_checkpoint(baseline.checkpoints[0]["path"])
    profile = deepcopy(session._profile())
    profile["config"]["training"]["rollout_envs"] = 1
    profile["config"]["training"]["token_steps_per_update"] = 1
    monkeypatch.setattr(session, "_profile", lambda: deepcopy(profile))
    state = deepcopy(actual.training_state)
    state["compatibility"]["profile_sha256"] = profile["sha256"]
    old_mount = tmp_path / "fixture_old_working"
    old_mount.mkdir()
    checkpoint = old_mount / "fixture_one_actual_cpu_step.pt"
    save_checkpoint(
        checkpoint, profile="remote_full", algorithm=algorithm, update=1,
        seed=actual.metadata.seed_count, git_commit=actual.metadata.git_commit,
        protocol_versions=CURRENT_PROTOCOL_VERSIONS, model_state=actual.model_state,
        optimizer_state=actual.optimizer_state, model_config=actual.model_config,
        training_state=state,
    )
    data = {
        "format": "guandan-recovery-v1", "profile": "remote_full", "algorithm": algorithm,
        "checkpoint_filename": checkpoint.name, "checkpoint_sha256": file_sha256(checkpoint),
        "model_config": actual.model_config, "compatibility": state["compatibility"],
        "protocol_versions": CURRENT_PROTOCOL_VERSIONS,
        "source_git_commit": actual.metadata.git_commit,
        "durable_updates": 1, "durable_token_steps": 1,
        "accepted": False, "remote_verified": False,
        "fixture_scope": "SYNTHETIC remote envelope; one real local CPU token; no remote run",
    }
    manifest = old_mount / "manifest.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    return algorithm, checkpoint, manifest, data


def test_manifest_resolver_uses_copied_sibling_and_never_original_mount(scoped_remote_metadata, tmp_path):
    from guandan.deployment.session import resolve_recovery

    algorithm, checkpoint, manifest, data = scoped_remote_metadata
    new_mount = tmp_path / "different_input_mount"
    new_mount.mkdir()
    copied_checkpoint = new_mount / checkpoint.name
    copied_manifest = new_mount / manifest.name
    shutil.copyfile(checkpoint, copied_checkpoint)
    shutil.copyfile(manifest, copied_manifest)
    checkpoint.unlink()
    manifest.unlink()
    resolved = resolve_recovery(copied_manifest, algorithm=algorithm)
    assert resolved["checkpoint_path"] == str(copied_checkpoint.resolve())
    assert resolved["manifest_path"] == str(copied_manifest.resolve())
    assert resolved["durable_updates"] == resolved["durable_token_steps"] == 1
    assert resolved["checkpoint_sha256"] == file_sha256(copied_checkpoint)
    assert resolved["accepted"] is False and resolved["remote_verified"] is False
    assert resolved["compatibility"] == data["compatibility"]


@pytest.mark.parametrize("filename", ["../outside.pt", "/kaggle/input/x.pt", "C:\\old\\x.pt",
                                      "subdir/x.pt", "subdir\\x.pt", ".", "..", ""])
def test_manifest_refuses_absolute_or_traversing_checkpoint_names(scoped_remote_metadata, filename):
    from guandan.deployment.session import resolve_recovery

    algorithm, _, manifest, data = scoped_remote_metadata
    data["checkpoint_filename"] = filename
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="sibling checkpoint basename"):
        resolve_recovery(manifest, algorithm=algorithm)


@pytest.mark.parametrize("field,value,error", [
    ("format", "unknown-format", "format/profile/algorithm"),
    ("profile", "local_fast", "format/profile/algorithm"),
    ("algorithm", "unsupported", "format/profile/algorithm"),
    ("checkpoint_sha256", "0" * 64, "digest mismatch"),
    ("source_git_commit", "0" * 40, "source commit"),
    ("durable_updates", 100, "update counters"),
    ("durable_token_steps", 102400, "token counters"),
    ("protocol_versions", {}, "protocol versions"),
    ("compatibility", {}, "compatibility"),
])
def test_manifest_rejects_forged_metadata_without_loading_a_training_run(
    scoped_remote_metadata, field, value, error,
):
    from guandan.deployment.session import resolve_recovery

    algorithm, _, manifest, data = scoped_remote_metadata
    data[field] = value
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        resolve_recovery(manifest, algorithm=algorithm)


def test_manifest_rejects_missing_and_tampered_checkpoint(scoped_remote_metadata):
    from guandan.deployment.session import resolve_recovery

    algorithm, checkpoint, manifest, _ = scoped_remote_metadata
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="digest mismatch"):
        resolve_recovery(manifest, algorithm=algorithm)
    checkpoint.unlink()
    with pytest.raises(ValueError, match="missing"):
        resolve_recovery(manifest, algorithm=algorithm)


def test_local_fixture_checkpoint_is_not_accepted_as_remote_recovery(uninterrupted):
    from guandan.deployment.session import resolve_recovery

    algorithm, result = uninterrupted
    with pytest.raises(ValueError, match="format/profile/algorithm"):
        resolve_recovery(result.recovery_manifest, algorithm=algorithm)


def test_resolver_rejects_profile_drift_after_fixture_checkpoint(scoped_remote_metadata, monkeypatch):
    from copy import deepcopy
    from guandan.deployment import session

    algorithm, _, manifest, _ = scoped_remote_metadata
    changed = deepcopy(session._profile())
    changed["config"]["training"]["token_steps_per_update"] = 2
    monkeypatch.setattr(session, "_profile", lambda: changed)
    with pytest.raises(ValueError, match="profile counts/hash changed"):
        session.resolve_recovery(manifest, algorithm=algorithm)
