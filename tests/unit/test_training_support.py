from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from guandan import execution_profiles
from guandan.model import PolicyValueNet
from guandan.training import checkpoint as checkpoint_module
from guandan.training.checkpoint import (
    CHECKPOINT_VERSION,
    CURRENT_PROTOCOL_VERSIONS,
    MODEL_VERSION,
    TRAINING_VERSION,
    CheckpointError,
    load_and_resume_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from guandan.training.config import ProfileConfigError, UnknownProfileError, load_profile, load_profiles
from guandan.training.critic import CentralQCritic


def test_profiles_are_strict_and_have_expected_local_remote_budgets() -> None:
    local = load_profile("local_fast")
    remote = load_profile("remote_full")
    assert local.updates_per_algorithm == 5
    assert local.rollout_envs == 4
    assert remote.updates_per_algorithm == 100
    assert remote.rollout_envs == 8


def test_checkpoint_round_trip_and_resume_metadata(tmp_path: Path) -> None:
    model = PolicyValueNet(hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    metadata = save_checkpoint(path, model=model, optimizer=optimizer, profile="local_fast", algorithm="ippo", update=3, seed=7, protocol_versions=CURRENT_PROTOCOL_VERSIONS, git_commit="test-commit")
    assert metadata.update_count == 3
    restored = PolicyValueNet(hidden_dim=16)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    loaded = load_checkpoint(path, model=restored, optimizer=restored_optimizer, expected_profile="local_fast", expected_algorithm="ippo")
    assert loaded.metadata.update_count == 3
    assert loaded.metadata.seed_count == 7
    assert loaded.metadata.git_commit == "test-commit"


def test_checkpoint_rejects_protocol_mismatch(tmp_path: Path) -> None:
    model = PolicyValueNet(hidden_dim=8)
    path = tmp_path / "bad.pt"
    versions = dict(CURRENT_PROTOCOL_VERSIONS)
    versions["rules_version"] = "GD-RULES-bad"
    save_checkpoint(path, model=model, profile="local_fast", algorithm="ippo", protocol_versions=versions)
    try:
        load_checkpoint(path)
    except Exception as exc:
        assert "rules_version" in str(exc)
    else:
        raise AssertionError("incompatible protocol checkpoint was accepted")

# No remote launches or collector API assumptions: only persistence/config tests.


def _equal_tree(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.device == right.device
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal_tree(a, b)
    else:
        assert type(left) is type(right)
        assert left == right


def _step(model, optimizer, gradient: float = 0.25) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, gradient)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _pair(*, combined=False, critic=True):
    actor = PolicyValueNet(hidden_dim=8)
    model = actor
    if combined:
        children = {"actor": actor}
        if critic:
            children["critic"] = CentralQCritic(hidden_dim=8)
        model = torch.nn.ModuleDict(children)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
    _step(model, optimizer)
    return model, optimizer


def _saved_pair(tmp_path, **kwargs):
    model, optimizer = _pair()
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, git_commit=None, **kwargs)
    return path, model, optimizer


def _payload(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _reject_without_mutation(path, model, optimizer, *, match=None, **kwargs):
    before_model = deepcopy(model.state_dict())
    before_optimizer = deepcopy(optimizer.state_dict())
    parameters = list(model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(CheckpointError, match=match):
        load_checkpoint(path, model, optimizer, **kwargs)
    _equal_tree(before_model, model.state_dict())
    _equal_tree(before_optimizer, optimizer.state_dict())
    assert all(old is new for old, new in zip(parameters, model.parameters()))
    assert torch.equal(rng, torch.get_rng_state())


def test_v2_infers_policy_architecture_and_preserves_mapping_api(tmp_path):
    path, model, _ = _saved_pair(tmp_path)
    loaded = load_checkpoint(path)
    assert loaded["checkpoint_version"] == CHECKPOINT_VERSION == 2
    assert loaded["model_version"] == MODEL_VERSION == "GD-MODEL-MLP-0.2"
    assert loaded["training_version"] == TRAINING_VERSION == "GD-TRAIN-0.2"
    assert loaded.model_config == {
        "token_vocab_size": model.token_vocab_size,
        "num_state_channels": model.num_state_channels,
        "hidden_dim": 8,
        "pad_token_id": model.pad_token_id,
        "max_observation_tokens": model.max_observation_tokens,
        "max_legal_next_tokens": model.max_legal_next_tokens,
    }
    assert loaded["model_config"] is loaded.model_config
    assert loaded["training_state"] == {}
    assert loaded.as_dict()["training_state"] == {}
    assert loaded.get("unknown", "fallback") == "fallback"
    assert loaded.metadata["protocol_versions"] == CURRENT_PROTOCOL_VERSIONS
    with pytest.raises(KeyError):
        _ = loaded["not_a_field"]


@pytest.mark.parametrize("algorithm,include_critic", [("ippo", False), ("vrpo", True)])
def test_combined_actor_critic_adam_rng_and_env_bytes_round_trip(tmp_path, algorithm, include_critic):
    model, optimizer = _pair(combined=True, critic=include_critic)
    generator = np.random.default_rng(4096)
    generator.random(7)
    training = {
        "env_payloads": [b"opaque-env-0\x00\xff", b"opaque-env-1"],
        "torch_rng_cpu": torch.get_rng_state().clone(),
        "torch_rng_cuda": [],
        "numpy_rng_state": generator.bit_generator.state,
        "counters": {"tokens": 256, "updates": 3, "seed": 4096},
        "collector": {"pending": None, "completed": (3, 4), "active": True},
        "hyperparameters": {"gamma": 0.99, "clip": 0.2},
        "profile_sha": "caller-owned-digest",
    }
    path = tmp_path / "combined.pt"
    save_checkpoint(path, model, optimizer, algorithm=algorithm, update=3, seed=4096,
                    training_state=training, git_commit=None)
    restored, restored_optimizer = _pair(combined=True, critic=include_critic)
    # Distinct initialized parameters/optimizer must really be replaced.
    _step(restored, restored_optimizer, 0.75)
    rng_before_load = torch.get_rng_state().clone()
    loaded = load_and_resume_checkpoint(path, restored, restored_optimizer, expected_algorithm=algorithm)
    assert torch.equal(rng_before_load, torch.get_rng_state())
    _equal_tree(model.state_dict(), restored.state_dict())
    _equal_tree(training, loaded.training_state)
    assert set(loaded.model_config) == ({"actor", "critic"} if include_critic else {"actor"})
    assert any(key.startswith("critic.") for key in loaded.model_state) == include_critic
    assert loaded.optimizer_state["param_groups"][0]["param_names"] == list(dict(model.named_parameters()))
    for source, target in zip(optimizer.state.values(), restored_optimizer.state.values()):
        _equal_tree(source, target)
    # Actual next Adam step agrees, not just tensor serialization.
    _step(model, optimizer, 0.5)
    _step(restored, restored_optimizer, 0.5)
    _equal_tree(model.state_dict(), restored.state_dict())
    resumed_generator = np.random.default_rng()
    resumed_generator.bit_generator.state = loaded.training_state["numpy_rng_state"]
    assert np.array_equal(generator.random(10), resumed_generator.random(10))
    resumed_torch_generator = torch.Generator().set_state(loaded.training_state["torch_rng_cpu"])
    expected_torch_generator = torch.Generator().set_state(training["torch_rng_cpu"])
    assert torch.equal(torch.rand(8, generator=resumed_torch_generator),
                       torch.rand(8, generator=expected_torch_generator))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_rng_and_combined_optimizer_can_resume_on_cuda(tmp_path):
    model, _ = _pair(combined=True)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    _step(model, optimizer)
    path = tmp_path / "cuda.pt"
    rng_state = torch.cuda.get_rng_state_all()
    save_checkpoint(path, model, optimizer, training_state={"torch_rng_cuda": rng_state}, git_commit=None)
    restored, _ = _pair(combined=True)
    restored.cuda()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
    # Default CPU map_location keeps RNG bytes on CPU for set_rng_state_all.
    loaded = load_checkpoint(path, restored, restored_optimizer)
    _equal_tree(rng_state, loaded.training_state["torch_rng_cuda"])
    _step(model, optimizer)
    _step(restored, restored_optimizer)
    _equal_tree(model.state_dict(), restored.state_dict())


@pytest.mark.parametrize("key,bad", [
    ("checkpoint_version", 1), ("checkpoint_version", True), ("checkpoint_version", 2.0),
    ("model_version", "GD-MODEL-bad"), ("training_version", "GD-TRAIN-bad"),
    *[(key, "incompatible") for key in CURRENT_PROTOCOL_VERSIONS],
])
def test_all_versions_rejected_before_any_live_mutation(tmp_path, key, bad):
    path, _, _ = _saved_pair(tmp_path)
    payload = _payload(path)
    payload[key] = bad
    if key in CURRENT_PROTOCOL_VERSIONS:
        payload["protocol_versions"][key] = bad
    torch.save(payload, path)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer, match=key)


@pytest.mark.parametrize("key", ["model_version", "training_version", *CURRENT_PROTOCOL_VERSIONS])
def test_missing_version_metadata_is_rejected(tmp_path, key):
    path, _, _ = _saved_pair(tmp_path)
    payload = _payload(path)
    del payload[key]
    if key in CURRENT_PROTOCOL_VERSIONS:
        del payload["protocol_versions"][key]
    torch.save(payload, path)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer)


@pytest.mark.parametrize("expectations", [
    {"expected_algorithm": "vrpo"}, {"expected_profile": "remote_full"},
    {"expected_training_version": "GD-TRAIN-different"},
    {"expected_model_version": "GD-MODEL-different"},
    {"expected_model_config": {"hidden_dim": 8}},
    {"expected_protocol_versions": {**CURRENT_PROTOCOL_VERSIONS, "action_version": "different"}},
])
def test_expected_context_mismatch_preserves_state(tmp_path, expectations):
    path, _, _ = _saved_pair(tmp_path)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer, **expectations)


@pytest.mark.parametrize("key,value", [
    ("algorithm", "unknown"), ("algorithm", ""), ("profile", "missing"),
    ("seed", True), ("seed_count", -1), ("update", True), ("update_count", 1.0),
    ("git_commit", 17), ("model_config", []), ("training_state", None),
])
def test_invalid_metadata_is_rejected(tmp_path, key, value):
    path, _, _ = _saved_pair(tmp_path)
    payload = _payload(path)
    payload[key] = value
    torch.save(payload, path)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer)


@pytest.mark.parametrize("key", ["model_state", "optimizer_state", "model_config", "training_state", "git_commit"])
def test_missing_v2_fields_fail_without_mutation(tmp_path, key):
    path, _, _ = _saved_pair(tmp_path)
    payload = _payload(path)
    del payload[key]
    torch.save(payload, path)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer, match=key)


def test_missing_optimizer_is_detected_before_loading_model(tmp_path):
    source, _ = _pair()
    path = tmp_path / "model-only.pt"
    save_checkpoint(path, source, git_commit=None)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer, match="optimizer state")
    loaded = load_checkpoint(path, model=target)
    assert loaded.optimizer_state is None
    _equal_tree(source.state_dict(), target.state_dict())


@pytest.mark.parametrize("corruption", ["shape", "dtype", "missing", "extra", "nan", "inf", "not-tensor"])
def test_corrupted_model_state_rejected_atomically(tmp_path, corruption):
    path, _, _ = _saved_pair(tmp_path)
    payload = _payload(path)
    state = payload["model_state"]
    key = next(iter(state))
    if corruption == "shape":
        state[key] = state[key][:-1].clone()
    elif corruption == "dtype":
        state[key] = state[key].double()
    elif corruption == "missing":
        del state[key]
    elif corruption == "extra":
        state["unexpected.weight"] = torch.ones(1)
    elif corruption in {"nan", "inf"}:
        state[key].view(-1)[0] = float(corruption)
    else:
        state[key] = "not a tensor"
    torch.save(payload, path)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer)


@pytest.mark.parametrize("corruption", [
    "shape", "nan", "step-shape", "step-negative", "step-fractional", "missing-moment",
    "missing-step", "missing-group-field", "missing-group", "missing-param", "unknown-param",
    "duplicate-param", "name-order", "missing-state-field",
])
def test_corrupted_adam_state_rejected_atomically(tmp_path, corruption):
    path, _, _ = _saved_pair(tmp_path)
    payload = _payload(path)
    saved = payload["optimizer_state"]
    state = next(iter(saved["state"].values()))
    group = saved["param_groups"][0]
    if corruption == "shape":
        state["exp_avg"] = torch.zeros(1)
    elif corruption == "nan":
        state["exp_avg_sq"].view(-1)[0] = float("nan")
    elif corruption == "step-shape":
        state["step"] = torch.ones(1)
    elif corruption == "step-negative":
        state["step"] = torch.tensor(-1.0)
    elif corruption == "step-fractional":
        state["step"] = torch.tensor(0.5)
    elif corruption == "missing-moment":
        del state["exp_avg"]
    elif corruption == "missing-step":
        del state["step"]
    elif corruption == "missing-group-field":
        del group["lr"]
    elif corruption == "missing-group":
        saved["param_groups"] = []
    elif corruption == "missing-param":
        group["params"].pop()
    elif corruption == "unknown-param":
        saved["state"][100000] = deepcopy(state)
    elif corruption == "duplicate-param":
        group["params"][1] = group["params"][0]
    elif corruption == "name-order":
        group["param_names"].reverse()
    else:
        del saved["state"]
    torch.save(payload, path)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer)


def test_policy_config_detects_same_tensor_shapes_different_semantics(tmp_path):
    path, _, _ = _saved_pair(tmp_path)
    target = PolicyValueNet(hidden_dim=8, pad_token_id=1, max_observation_tokens=2048)
    optimizer = torch.optim.Adam(target.parameters(), lr=0.002, amsgrad=True)
    _reject_without_mutation(path, target, optimizer, match="model_config")


def test_module_dict_actor_critic_missing_target_rejected(tmp_path):
    source, optimizer = _pair(combined=True)
    path = tmp_path / "combined.pt"
    save_checkpoint(path, source, optimizer, algorithm="vrpo", git_commit=None)
    target, target_optimizer = _pair(combined=True, critic=False)
    _reject_without_mutation(path, target, target_optimizer, match="model_state")


def test_custom_config_and_state_dict_apis_remain_usable(tmp_path):
    source, optimizer = _pair()
    path = tmp_path / "explicit.pt"
    config = {"architecture": "parent-owned", "dimensions": [8, 256]}
    save_checkpoint(path, model_state=source.state_dict(), optimizer_state=optimizer.state_dict(),
                    model_config=config, update_count=3, seed_count=2, git_commit=None)
    target, target_optimizer = _pair()
    loaded = load_checkpoint(path, target, target_optimizer, expected_model_config=config)
    assert (loaded["update"], loaded["seed"]) == (3, 2)
    assert loaded.model_config == config
    _equal_tree(source.state_dict(), target.state_dict())
    # Original raw-state entry point also works without explicit architecture.
    save_checkpoint(path, model_state=source.state_dict(), optimizer_state=optimizer.state_dict(), git_commit=None)
    load_checkpoint(path, target, target_optimizer)
    _equal_tree(source.state_dict(), target.state_dict())


@pytest.mark.parametrize("unsafe", [np.zeros(2), np.int64(7), np.random.default_rng(3), object(), {1, 2}])
def test_training_state_rejects_non_safe_types_on_save(tmp_path, unsafe):
    path = tmp_path / "unsafe.pt"
    with pytest.raises(CheckpointError, match="unsupported object type"):
        save_checkpoint(path, PolicyValueNet(hidden_dim=8), training_state={"bad": unsafe}, git_commit=None)
    assert not path.exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), torch.tensor([float("inf")])])
def test_training_state_rejects_nonfinite_values_on_load(tmp_path, bad):
    path, _, _ = _saved_pair(tmp_path)
    payload = _payload(path)
    payload["training_state"] = {"bad": bad}
    torch.save(payload, path)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer, match="nonfinite")


def test_training_state_rejects_cycles_and_accepts_flexible_empty_map(tmp_path):
    path = tmp_path / "cycle.pt"
    cycle = {}
    cycle["self"] = cycle
    with pytest.raises(CheckpointError, match="cyclic"):
        save_checkpoint(path, PolicyValueNet(hidden_dim=8), training_state=cycle, git_commit=None)
    save_checkpoint(path, PolicyValueNet(hidden_dim=8), training_state={}, git_commit=None)
    assert load_checkpoint(path).training_state == {}


@pytest.mark.parametrize("contents", [b"not a checkpoint", b"", b"PK\x03\x04truncated"])
def test_corrupted_checkpoint_bytes_fail_closed(tmp_path, contents):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(contents)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer, match="safely load")


def test_unsafe_numpy_pickle_is_not_loaded_with_fallback(tmp_path, monkeypatch):
    path, _, _ = _saved_pair(tmp_path)
    payload = _payload(path)
    payload["training_state"] = {"unsafe": np.zeros(2)}
    torch.save(payload, path)
    calls = []
    original = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    target, optimizer = _pair()
    _reject_without_mutation(path, target, optimizer, match="safely load")
    assert calls == [True]


def test_no_unsafe_fallback_on_unsupported_weights_only(tmp_path, monkeypatch):
    path, _, _ = _saved_pair(tmp_path)
    calls = []

    def old_torch(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("weights_only unsupported")

    monkeypatch.setattr(torch, "load", old_torch)
    with pytest.raises(CheckpointError, match="safely load"):
        load_checkpoint(path)
    assert len(calls) == 1 and calls[0]["weights_only"] is True


def test_live_model_failure_rolls_back_already_copied_parameters(tmp_path, monkeypatch):
    path, _, _ = _saved_pair(tmp_path)
    target, optimizer = _pair()
    original = PolicyValueNet.load_state_dict

    def fail_live(self, state_dict, *args, **kwargs):
        result = original(self, state_dict, *args, **kwargs)
        if self is target:
            raise RuntimeError("after model mutation")
        return result

    monkeypatch.setattr(PolicyValueNet, "load_state_dict", fail_live)
    _reject_without_mutation(path, target, optimizer, match="rolled back")


def test_live_optimizer_failure_rolls_back_model_and_optimizer(tmp_path, monkeypatch):
    path, _, _ = _saved_pair(tmp_path)
    target, optimizer = _pair()
    original = torch.optim.Adam.load_state_dict

    def fail_live(self, state_dict):
        result = original(self, state_dict)
        if self is optimizer:
            self.param_groups[0]["lr"] = 999.0
            for state in self.state.values():
                state["exp_avg"].zero_()
            raise RuntimeError("after optimizer mutation")
        return result

    monkeypatch.setattr(torch.optim.Adam, "load_state_dict", fail_live)
    _reject_without_mutation(path, target, optimizer, match="rolled back")


def test_preflight_failure_never_calls_live_loaders(tmp_path, monkeypatch):
    path, _, _ = _saved_pair(tmp_path)
    target, optimizer = _pair()
    called = []

    def reject(self, *args, **kwargs):
        called.append(self is optimizer)
        raise RuntimeError("staged rejection")

    monkeypatch.setattr(torch.optim.Adam, "load_state_dict", reject)
    _reject_without_mutation(path, target, optimizer, match="preflight")
    assert called == [False]


def test_atomic_save_failure_preserves_existing_file_and_cleans_temporary(tmp_path, monkeypatch):
    path, model, optimizer = _saved_pair(tmp_path)
    original_bytes = path.read_bytes()

    def reject_replace(*args):
        raise OSError("replacement blocked")

    monkeypatch.setattr(checkpoint_module.os, "replace", reject_replace)
    with pytest.raises(OSError, match="replacement blocked"):
        save_checkpoint(path, model, optimizer, update=17, git_commit=None)
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_save_preserves_existing_file(tmp_path):
    path, model, optimizer = _saved_pair(tmp_path)
    before = path.read_bytes()
    with pytest.raises(CheckpointError):
        save_checkpoint(path, model, optimizer, training_state={"bad": np.arange(3)}, git_commit=None)
    assert path.read_bytes() == before


def _catalog(tmp_path, change=None):
    source = Path(__file__).resolve().parents[2] / "configs" / "acceptance_profiles.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if change is not None:
        change(payload)
    destination = tmp_path / "configs" / "acceptance_profiles.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def test_profiles_delegate_to_canonical_loader_and_keep_all_properties(tmp_path, monkeypatch):
    path = _catalog(tmp_path)
    original = execution_profiles.load_execution_profile
    calls = []

    def spy(root, name):
        calls.append((Path(root), name))
        return original(root, name)

    monkeypatch.setattr(execution_profiles, "load_execution_profile", spy)
    profiles = load_profiles(path)
    assert calls == [(tmp_path, "local_fast"), (tmp_path, "remote_full")]
    for name, profile in profiles.items():
        canonical = original(tmp_path, name)
        train = canonical["config"]["training"]
        assert profile.version == canonical["version"]
        assert profile.algorithms == ("ippo", "vrpo")
        assert profile.mode == canonical["config"]["mode"]
        assert profile.device == canonical["config"]["device"]
        for key, value in train.items():
            assert getattr(profile, key) == value
        assert profile.rollout_steps == train["token_steps_per_update"]
        assert profile.games_per_pairing == canonical["derived"]["games_per_pairing"]
    assert load_profile("local_fast", tmp_path) == profiles["local_fast"]


@pytest.mark.parametrize("corruption", [
    "extra-catalog-key", "wrong-version", "missing-profile", "extra-profile-key", "wrong-mode",
    "wrong-device", "one-algorithm", "extra-training-key", "bool-integer", "nondivisible-tokens",
    "checkpoint-after-end", "nan-budget", "infinite-budget", "boolean-seat", "missing-opponent",
    "extra-evaluation-key", "malformed-remote",
])
def test_profile_wrapper_rejects_every_canonical_catalog_violation(tmp_path, corruption):
    def corrupt(payload):
        local = payload["profiles"]["local_fast"]
        training = local["training"]
        evaluation = local["evaluation"]
        if corruption == "extra-catalog-key":
            payload["surprise"] = True
        elif corruption == "wrong-version":
            payload["version"] = "profiles-old"
        elif corruption == "missing-profile":
            del payload["profiles"]["remote_full"]
        elif corruption == "extra-profile-key":
            local["surprise"] = True
        elif corruption == "wrong-mode":
            local["mode"] = "remote"
        elif corruption == "wrong-device":
            local["device"] = "cuda"
        elif corruption == "one-algorithm":
            local["algorithms"] = ["ippo"]
        elif corruption == "extra-training-key":
            training["surprise"] = 1
        elif corruption == "bool-integer":
            training["rollout_envs"] = True
        elif corruption == "nondivisible-tokens":
            training["token_steps_per_update"] = 257
        elif corruption == "checkpoint-after-end":
            training["checkpoint_interval"] = 6
        elif corruption == "nan-budget":
            training["max_hours"] = float("nan")
        elif corruption == "infinite-budget":
            evaluation["max_hours"] = float("inf")
        elif corruption == "boolean-seat":
            evaluation["seat_rotations"] = [False, 1, 2, 3]
        elif corruption == "missing-opponent":
            evaluation["opponents"] = ["random", "rule"]
        elif corruption == "extra-evaluation-key":
            evaluation["surprise"] = 1
        else:
            payload["profiles"]["remote_full"]["device"] = "auto"

    path = _catalog(tmp_path, corrupt)
    with pytest.raises(ValueError):
        execution_profiles.load_execution_profile(tmp_path, "local_fast")
    with pytest.raises(ProfileConfigError):
        load_profile("local_fast", path)


def test_profiles_fail_closed_for_unknown_missing_and_noncanonical_files(tmp_path):
    with pytest.raises(UnknownProfileError):
        load_profile("missing")
    with pytest.raises(ProfileConfigError):
        load_profile("local_fast", tmp_path / "configs" / "acceptance_profiles.json")
    with pytest.raises(ProfileConfigError, match="configs/acceptance_profiles.json"):
        load_profiles(tmp_path / "arbitrary.json")
