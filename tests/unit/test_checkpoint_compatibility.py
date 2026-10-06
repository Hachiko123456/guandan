from pathlib import Path

import pytest
import torch

from guandan.model import PolicyValueNet
from guandan.training.checkpoint import (
    IncompatibleCheckpointError,
    ResumeMode,
    checkpoint_resume_decision,
    load_checkpoint,
    save_checkpoint,
)
from guandan.training.compatibility import SemanticContract, classify_resume


def _model():
    return PolicyValueNet(hidden_dim=8)


def _contract(model):
    return {
        "algorithm": "ippo",
        "profile": "local_fast",
        "model_version": "GD-MODEL-MLP-0.2",
        "training_version": "GD-TRAIN-0.2",
        "protocol_versions": {
            "rules_version": "rules-v1",
            "action_version": "action-v1",
            "environment_version": "env-v1",
            "encoding_version": "encoding-v1",
        },
        "model_config": {"hidden_dim": model.hidden_dim},
        "optimizer": {"name": "Adam", "lr": 0.001},
    }


def test_resume_classification_separates_semantics_and_fingerprints():
    contract = SemanticContract({"algorithm": "ippo", "model": {"hidden_dim": 8}})
    assert classify_resume(contract, contract, checkpoint_source_fingerprint="a", expected_source_fingerprint="a", checkpoint_implementation_fingerprint="i", expected_implementation_fingerprint="i").mode is ResumeMode.EXACT
    decision = classify_resume(contract, contract, checkpoint_source_fingerprint="a", expected_source_fingerprint="b", checkpoint_implementation_fingerprint="i", expected_implementation_fingerprint="j")
    assert decision.mode is ResumeMode.COMPATIBLE
    assert "source fingerprint differs" in decision.reasons
    assert classify_resume(contract, {"algorithm": "vrpo"}).mode is ResumeMode.MIGRATION


def test_checkpoint_stores_contract_and_distinct_fingerprints(tmp_path):
    model = _model()
    path = tmp_path / "checkpoint.pt"
    contract = _contract(model)
    save_checkpoint(path, model=model, git_commit="legacy-commit", source_fingerprint="source-a", implementation_fingerprint="impl-a", semantic_contract=contract)
    loaded = load_checkpoint(path)
    assert loaded.metadata.git_commit == "legacy-commit"
    assert loaded.metadata.source_fingerprint == "source-a"
    assert loaded.metadata.implementation_fingerprint == "impl-a"
    assert loaded.metadata.semantic_contract == contract
    assert checkpoint_resume_decision(loaded, contract, expected_source_fingerprint="source-a", expected_implementation_fingerprint="impl-a").mode is ResumeMode.EXACT
    assert checkpoint_resume_decision(loaded, contract, expected_source_fingerprint="source-b", expected_implementation_fingerprint="impl-b").mode is ResumeMode.COMPATIBLE


def test_load_resume_rejects_semantic_change_without_mutating_model(tmp_path):
    model = _model()
    path = tmp_path / "checkpoint.pt"
    contract = _contract(model)
    save_checkpoint(path, model=model, git_commit=None, semantic_contract=contract)
    target = _model()
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(IncompatibleCheckpointError, match="semantic contract differs"):
        load_checkpoint(path, model=target, expected_semantic_contract={**contract, "optimizer": {"lr": 0.002}}, resume_mode=ResumeMode.COMPATIBLE)
    assert all(torch.equal(before[key], value) for key, value in target.state_dict().items())


def test_old_checkpoint_requires_explicit_migration(tmp_path):
    model = _model()
    path = tmp_path / "old.pt"
    save_checkpoint(path, model=model, git_commit=None)
    payload = torch.load(path, weights_only=True)
    payload.pop("compatibility_schema_version")
    payload.pop("semantic_contract")
    payload.pop("source_fingerprint")
    payload.pop("implementation_fingerprint")
    torch.save(payload, path)
    with pytest.raises(IncompatibleCheckpointError, match="explicit offline migration"):
        load_checkpoint(path)
