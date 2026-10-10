from __future__ import annotations

import copy
from pathlib import Path
import shutil

import pytest
import torch

from guandan_marvel import cpp_game
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.training.checkpoints import (
    CheckpointCompatibilityError, PRE_BACKEND_IDENTITY_CHECKPOINT_SCHEMA,
    TRAINING_CHECKPOINT_SCHEMA, TrainingCheckpointStore, build_backend_identity,
    build_execution_fingerprint, build_semantic_configuration,
    build_semantic_contract, canonical_json_hash, classify_resume,
    compute_source_fingerprint, serialize_config,
)
from guandan_marvel.training.local import (
    CPP_BACKEND_SOURCE_COMPONENTS, IPPOTrainer, runtime_source_components,
)
from guandan_marvel.training.smoke import PythonReferenceSmokeGame
from nanopokerlib.build_identity import compute_native_source_manifest

ROOT = Path(__file__).resolve().parents[2]


def _config(root: Path) -> tuple[TrainConfig, PythonReferenceSmokeGame]:
    game = PythonReferenceSmokeGame(512)
    cfg = TrainConfig(
        game=game.spec, device="cpu", algo_name="ippo",
        logging_dir=str(root), run_name="backend-identity", seed=8701,
        rollout_batch_size=1, collect_batch_size=1, replay_buffer_scale=1,
        mini_batches=1, total_iterations=2,
        max_sampler_bytes_per_rank=64 * 2**20,
        checkpoint_replay_buffer=True, amp_enabled=False,
        model=ModelConfig(
            architecture="v1.3", history_dim=8, feature_dim=8,
            decoder_layers=1, decoder_heads=1,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.ppo.agent_epoch = 1
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg, game


def _identity_hash(identity: dict) -> None:
    identity["hash"] = canonical_json_hash({
        key: value for key, value in identity.items() if key != "hash"
    })


def _classification_fixture(tmp_path: Path):
    cfg, game = _config(tmp_path)
    semantic = build_semantic_configuration(cfg, algorithm="ippo")
    contract = build_semantic_contract(
        semantic_config=semantic, sampler_contract="sampler-v2",
        sampler_state_schema="sampler-state-v1", buffer_schema="buffer-v2",
        q_target_contract=None,
    )
    source_file = tmp_path / "source.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    source = compute_source_fingerprint(tmp_path, {"test": ("source.py",)})
    execution = build_execution_fingerprint("cpu")
    backend = build_backend_identity(game)
    resolved = serialize_config(cfg)
    provenance_base = {
        "schema": "guandan-marvel-training-data-provenance-v1",
        "domain": "training",
        "manifest_hash": resolved["training_data_manifest_hash"],
        "generator_version": resolved["training_data_generator_version"],
        "seed_semantics": "fixture",
        "initial_env_seed": resolved["seed"],
        "rollout_batch_size": resolved["rollout_batch_size"],
        "deal_count": 1,
        "deal_hashes": [],
        "records_hash": canonical_json_hash([]),
        "complete": True,
    }
    provenance = {**provenance_base, "hash": canonical_json_hash(provenance_base)}
    payload = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "backend": backend["backend"], "backend_identity": backend,
        "semantic_contract": contract,
        "source_fingerprint": source, "source_hash": source["hash"],
        "execution_fingerprint": execution,
        "resolved_config": resolved,
        "resolved_config_hash": canonical_json_hash(resolved),
        "agent_resume_state_dict": {}, "agent_state_dict": {},
        "agent_optimizer_state_dict": {}, "iteration": 1,
        "durable_updates": 1, "counters": {}, "rng_state": {},
        "sampler_state": {}, "schedule_state": {}, "merged_policy": {},
    }
    expected = dict(
        expected_contract=contract, expected_source_fingerprint=source,
        expected_execution_fingerprint=execution,
        expected_backend_identity=backend, expected_config=resolved,
        require_replay=False,
    )
    return payload, expected


def test_cpp_build_identity_matches_loaded_binary_and_native_sources():
    identity = build_backend_identity(cpp_game)
    implementation = identity["implementation"]
    assert identity["backend"] == "cpp"
    assert implementation["source_matches_loaded_binary"] is True
    assert implementation["extension_sha256"]
    assert implementation["embedded_source_sha256"] == implementation["current_source_manifest"]["hash"]
    assert Path(implementation["extension_path"]).is_file()


def test_native_source_and_cpp_wrapper_changes_are_fingerprinted(tmp_path: Path):
    for relative in {path for paths in CPP_BACKEND_SOURCE_COMPONENTS.values() for path in paths}:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    native_before = compute_native_source_manifest(tmp_path)
    native = tmp_path / "nanopokerlib/cpp/guandan/guandan.cpp"
    native.write_text(native.read_text(encoding="utf-8") + "\n// fixture mutation\n", encoding="utf-8")
    native_after = compute_native_source_manifest(tmp_path)
    assert native_before["hash"] != native_after["hash"]

    components = runtime_source_components("cpp", tmp_path)
    before = compute_source_fingerprint(tmp_path, components)
    wrapper = tmp_path / "guandan_marvel/cpp_game.py"
    wrapper.write_text(wrapper.read_text(encoding="utf-8") + "\n# fixture mutation\n", encoding="utf-8")
    after = compute_source_fingerprint(tmp_path, components)
    assert before["hash"] != after["hash"]
    assert before["components"]["cpp_native_source"]["hash"] == after["components"]["cpp_native_source"]["hash"]
    assert before["components"]["cpp_backend_wrapper"]["hash"] != after["components"]["cpp_backend_wrapper"]["hash"]


def test_backend_or_extension_identity_change_is_migration(tmp_path: Path):
    payload, expected = _classification_fixture(tmp_path)
    cpp_identity = build_backend_identity(cpp_game)
    changed_backend = dict(expected)
    changed_backend["expected_backend_identity"] = cpp_identity
    decision = classify_resume(payload, **changed_backend)
    assert decision.kind == "migration"
    assert any("backend_identity" in reason for reason in decision.reasons)

    cpp_payload = copy.deepcopy(payload)
    cpp_payload["backend"] = "cpp"
    cpp_payload["backend_identity"] = cpp_identity
    changed_build = copy.deepcopy(cpp_identity)
    changed_build["implementation"]["extension_sha256"] = "0" * 64
    _identity_hash(changed_build)
    cpp_expected = dict(expected)
    cpp_expected["expected_backend_identity"] = changed_build
    build_decision = classify_resume(cpp_payload, **cpp_expected)
    assert build_decision.kind == "migration"
    assert any("extension_sha256" in reason for reason in build_decision.reasons)


def test_pre_backend_identity_v3_checkpoint_is_not_exact(tmp_path: Path):
    payload, expected = _classification_fixture(tmp_path)
    payload["schema"] = PRE_BACKEND_IDENTITY_CHECKPOINT_SCHEMA
    payload.pop("backend")
    payload.pop("backend_identity")
    decision = classify_resume(payload, **expected)
    assert decision.kind == "migration"
    assert "backend" in decision.reasons[0]


def test_checkpoint_records_backend_and_failed_backend_load_poisoned(tmp_path: Path):
    cfg, game = _config(tmp_path)
    source = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "source")
    source.train_iteration()
    checkpoint = source.save_checkpoint(tmp_path / "python-backend-v4.pt")
    payload = dict(TrainingCheckpointStore.load(checkpoint))
    assert payload["schema"] == TRAINING_CHECKPOINT_SCHEMA
    assert payload["backend"] == "python_reference"
    assert payload["backend_identity"]["backend"] == "python_reference"
    assert payload["source_hash"] == payload["source_fingerprint"]["hash"]

    payload["backend"] = "cpp"
    payload["backend_identity"] = build_backend_identity(cpp_game)
    tampered = tmp_path / "wrong-backend.pt"
    torch.save(payload, tampered)
    target = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "target")
    with pytest.raises(CheckpointCompatibilityError, match="checkpoint cannot continue training"):
        target.load_checkpoint(tampered)
    assert target.failed and target.sampler.failed
    with pytest.raises(RuntimeError, match="unusable"):
        target.train_iteration()
    with pytest.raises(RuntimeError):
        target.save_checkpoint(tmp_path / "must-not-save.pt")
