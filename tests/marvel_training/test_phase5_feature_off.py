"""Phase-5 defaults must be bitwise inert and phase-5 flags semantic."""
from __future__ import annotations

import json

import pytest
import torch

from guandan_marvel import python_game
from guandan_marvel.config import (
    BeliefConfig, ModelConfig, SymmetryConfig, TrainConfig,
)
from guandan_marvel.model import Agent, TensorDict
from guandan_marvel.phase5.design import phase5_design_hash
from guandan_marvel.phase5.gate import BeliefGateError, seal_gate_artifact
from guandan_marvel.training.checkpoints import build_semantic_configuration


def _config(*, symmetry=None, belief=None) -> TrainConfig:
    cfg = TrainConfig(
        game=python_game.GAME_SPEC, device="cpu",
        rollout_batch_size=2, collect_batch_size=1,
        max_sampler_bytes_per_rank=512 * 2**20,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
        symmetry=SymmetryConfig() if symmetry is None else symmetry,
        belief=BeliefConfig() if belief is None else belief,
    )
    cfg.optimizer.use_muon = False
    return cfg


def _rollout() -> TensorDict:
    torch.manual_seed(321)
    B, T, P, H, C, A, L = 2, 3, 4, 48, 512, 8, 8
    num_steps = torch.tensor([3, 2], dtype=torch.long)
    player_ids = torch.tensor([[0, 1, 2], [3, 0, -1]], dtype=torch.long)
    obs_tokens = torch.randint(1, 309, (B, T, P, H), dtype=torch.long)
    obs_channels = torch.randint(0, 2, (B, T, P, C), dtype=torch.uint8)
    avail_tokens = torch.randint(1, 309, (B, T, A, L), dtype=torch.long)
    avail_weights = torch.zeros(B, T, A)
    avail_weights[..., :7] = torch.rand(B, T, 7) + 0.1
    return TensorDict({
        "num_steps": num_steps, "player_ids": player_ids,
        "obs_tokens": obs_tokens, "obs_channels": obs_channels,
        "avail_tokens": avail_tokens, "avail_weights": avail_weights,
    }, batch_size=[B])


def test_feature_defaults_are_off_and_invalid_combinations_fail_closed():
    assert SymmetryConfig().symmetry_weight == 0.0
    assert SymmetryConfig().augmentation is False
    assert BeliefConfig().belief_weight == 0.0
    assert BeliefConfig().build_head is False
    with pytest.raises(ValueError):
        SymmetryConfig(symmetry_weight=-1.0)
    with pytest.raises(ValueError):
        SymmetryConfig(symmetry_weight=float("nan"))
    with pytest.raises(ValueError):
        SymmetryConfig(symmetry_weight=0.0, augmentation=True)
    with pytest.raises(ValueError):
        BeliefConfig(belief_weight=0.1, build_head=False)
    with pytest.raises(ValueError):
        BeliefConfig(belief_weight=0.0, build_head=True)
    with pytest.raises(ValueError):
        BeliefConfig(belief_weight=0.1, build_head=True, gate_artifact=None)


def _write_gate(tmp_path, verdict):
    payload = seal_gate_artifact({
        "schema": "guandan-marvel-phase5-belief-gate-v1",
        "verdict": verdict,
        "design_hash": phase5_design_hash(),
        "source_hash": "source",
        "config_hash": "config",
        "dataset": {"records_hash": "r", "split_hash": "s"},
        "epsilon": 0.05,
        "primary": {"improvement": 1.0},
        "metrics": {},
        "sufficiency": {},
        "reasons": [],
    })
    path = tmp_path / f"{verdict}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_belief_head_config_requires_a_pass_artifact(tmp_path):
    pass_gate = _write_gate(tmp_path, "PASS")
    config = BeliefConfig(
        belief_weight=0.1, build_head=True, gate_artifact=str(pass_gate),
    )
    assert config.build_head is True
    reject_gate = _write_gate(tmp_path, "REJECT")
    with pytest.raises(BeliefGateError):
        BeliefConfig(belief_weight=0.1, build_head=True, gate_artifact=str(reject_gate))
    with pytest.raises(BeliefGateError):
        BeliefConfig(
            belief_weight=0.1, build_head=True,
            gate_artifact=str(tmp_path / "missing.json"),
        )


def test_default_agents_have_identical_parameters_and_outputs():
    torch.manual_seed(1234)
    base = Agent(_config(), enable_value_head=True, enable_ntp_head=False)
    torch.manual_seed(1234)
    explicit = Agent(
        _config(
            symmetry=SymmetryConfig(symmetry_weight=0.0, augmentation=False),
            belief=BeliefConfig(belief_weight=0.0, build_head=False),
        ),
        enable_value_head=True, enable_ntp_head=False,
    )
    assert list(base.state_dict()) == list(explicit.state_dict())
    for name, value in base.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
    assert not any("belief" in name or "symmetry" in name for name in base.state_dict())
    rollout = _rollout()
    base.eval()
    explicit.eval()
    rng_before = torch.get_rng_state().clone()
    with torch.no_grad():
        first = base(rollout)
        rng_middle = torch.get_rng_state().clone()
        second = explicit(rollout)
        rng_after = torch.get_rng_state().clone()
    assert torch.equal(rng_before, rng_middle)
    assert torch.equal(rng_middle, rng_after)
    assert torch.equal(first.avail_probs, second.avail_probs)
    assert torch.equal(first.avail_log_probs, second.avail_log_probs)
    assert first.state_values is not None and second.state_values is not None
    assert torch.equal(first.state_values, second.state_values)


def _write_symmetry_seal(tmp_path):
    """Minimal valid seal + checkpoint for the enabled semantic-contract test."""
    import hashlib

    from guandan_marvel.phase5.design import phase5_design_hash
    from guandan_marvel.phase5.symmetry_training import canonical_hash

    checkpoint = tmp_path / "symmetry_init.pt"
    checkpoint.write_bytes(b"phase5-symmetry-initial-checkpoint")
    artifact = {
        "schema": "guandan-phase5-symmetry-diagnostic-run-v1",
        "status": "COMPLETED",
        "training": False,
        "truncated": False,
        "deals_requested": 16,
        "design_hash": phase5_design_hash(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_policy_identity": {
            "architecture": "v1.3", "algorithm": "vrpo",
            "history_dim": 16, "feature_dim": 24,
            "parameter_count": 1, "state_dict_keys": 1,
        },
        "decision": {
            "status": "SIGNIFICANT", "augmentation_recommended": True,
            "sample_count": 1536, "deal_count": 16, "non_finite_kl_count": 0,
            "mean_kl": 0.02857, "argmax_disagreement_rate": 0.11263,
            "thresholds": {
                "min_macro_samples": 256, "min_deals": 8,
                "mean_restricted_kl_threshold": 0.02,
                "argmax_disagreement_threshold": 0.10,
            },
        },
        "diagnostic": {
            "transforms": ["copy_swap"],
            "aggregate": {
                "mean_kl": 0.02857, "median_kl": 0.02, "max_kl": 0.2,
                "non_finite_kl_count": 0,
            },
        },
    }
    artifact_path = tmp_path / "symmetry_artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    seal = {
        "schema": "guandan-marvel-phase5-symmetry-artifact-seal-v1",
        "artifact_path": str(artifact_path),
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "content_hash": canonical_hash(artifact),
        "design_hash": artifact["design_hash"],
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "source_policy_identity": artifact["source_policy_identity"],
        "transforms": ["copy_swap"],
    }
    seal["seal_hash"] = canonical_hash(seal)
    seal_path = tmp_path / "symmetry_seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    return seal_path, checkpoint


def test_phase5_flags_are_part_of_the_semantic_checkpoint_contract(tmp_path):
    base = build_semantic_configuration(_config(), algorithm="vrpo")
    seal_path, checkpoint = _write_symmetry_seal(tmp_path)
    flipped = build_semantic_configuration(
        _config(symmetry=SymmetryConfig(
            symmetry_weight=0.5,
            diagnostic_artifact=str(seal_path),
            compatibility_checkpoint=str(checkpoint),
        )), algorithm="vrpo",
    )
    assert base["phase5_conditionals"] == {
        "symmetry_weight": 0.0, "augmentation": False,
        "belief_weight": 0.0, "belief_head": False,
    }
    assert flipped["phase5_conditionals"]["symmetry_weight"] == 0.5
    assert base["hash"] != flipped["hash"]
    pass_gate = _write_gate(tmp_path, "PASS")
    belief = build_semantic_configuration(
        _config(belief=BeliefConfig(
            belief_weight=0.1, build_head=True, gate_artifact=str(pass_gate),
        )),
        algorithm="vrpo",
    )
    assert belief["phase5_conditionals"]["belief_weight"] == 0.1
    assert belief["phase5_conditionals"]["belief_head"] is True
    assert belief["hash"] != base["hash"]
