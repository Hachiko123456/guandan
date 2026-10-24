"""05E closeout counterexamples: global ledger caps, canonical records hash, symmetry.

Every test here fails on the pre-05E implementation:

* Baseline B only checked each seat's row against the physical per-rank maximum,
  so the same eight physical cards could be counted in two seats;
* ``dataset_hash`` serialised in-memory int keys numerically but JSONL string
  keys lexicographically, so a delivered file could not reproduce its hash;
* ``SymmetryConfig`` accepted a positive weight without any training consumer
  or artifact gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from guandan_marvel import python_game
from guandan_marvel.config import ModelConfig, SymmetryConfig, TrainConfig
from guandan_marvel.model import Agent, TensorDict
from guandan_marvel.phase5.belief_data import (
    canonical_records_json, canonical_records_value, dataset_hash,
)
from guandan_marvel.phase5.belief_probe import BaselineB
from guandan_marvel.phase5.design import phase5_design_hash
from guandan_marvel.phase5.symmetry_training import (
    SymmetryArtifactError, SymmetryConsistency, canonical_hash,
    load_sealed_symmetry_artifact, validate_config_artifact,
)

PUBLIC_FEATURE_WIDTH = 64


# ---------------------------------------------------------------------------
# Baseline B: global physical-entity caps
# ---------------------------------------------------------------------------


def _features(
    remaining, known, unknown, escrow=0,
) -> np.ndarray:
    known = np.asarray(known, dtype=np.float64)
    assert known.shape == (3, 15)
    vector = np.concatenate([
        np.asarray(remaining, dtype=np.float64),
        known.reshape(-1),
        np.asarray(unknown, dtype=np.float64),
        np.asarray([escrow], dtype=np.float64),
    ])
    assert vector.shape == (PUBLIC_FEATURE_WIDTH,)
    return vector[None, :]


def test_baseline_b_rejects_cross_seat_duplicate_known_cards():
    known = np.zeros((3, 15))
    known[0, 0] = 8
    known[1, 0] = 8
    features = _features([8, 8, 0], known, np.zeros(15), escrow=0)
    with pytest.raises(ValueError, match="global physical per-rank maximum"):
        BaselineB().predict(features)


def test_baseline_b_rejects_cross_seat_joker_overflow():
    known = np.zeros((3, 15))
    known[0, 13] = 2
    known[1, 13] = 2
    features = _features([2, 2, 0], known, np.zeros(15), escrow=0)
    with pytest.raises(ValueError, match="global physical per-rank maximum"):
        BaselineB().predict(features)


def test_baseline_b_rejects_known_plus_unknown_overflow():
    known = np.zeros((3, 15))
    known[0, 0] = 4
    known[1, 0] = 4
    unknown = np.zeros(15)
    unknown[0] = 1  # K=8 + u=1 > C_0=8
    features = _features([8, 8, 0], known, unknown, escrow=0)
    with pytest.raises(ValueError, match="residual unknown"):
        BaselineB().predict(features)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 1.5])
def test_baseline_b_rejects_non_finite_fractional_and_negative_counts(bad):
    features = _features([8, 8, 8], np.zeros((3, 15)), np.zeros(15))
    features[0, 0] = bad
    with pytest.raises(ValueError):
        BaselineB().predict(features)


def test_baseline_b_conservation_and_lower_bounds_still_enforced():
    # sum(u) != H + X
    features = _features([8, 8, 8], np.zeros((3, 15)), np.zeros(15))
    features[0, 48] = 1
    with pytest.raises(ValueError, match="conserve"):
        BaselineB().predict(features)
    # known lower bound above a published hand
    known = np.zeros((3, 15))
    known[0, 0] = 3
    with pytest.raises(ValueError, match="published hand"):
        BaselineB().predict(_features([2, 8, 8], known, np.zeros(15)))


def test_baseline_b_valid_allocation_is_bitwise_unchanged():
    known = np.zeros((3, 15))
    known[0, 0] = 2
    unknown = np.zeros(15)
    unknown[0] = 4  # H = 6 - 2 = 4
    features = _features([2, 2, 2], known, unknown, escrow=0)
    prediction = BaselineB().predict(features).reshape(3, 15)
    expected = known + (np.asarray([0.0, 2.0, 2.0])[:, None] * unknown[None, :] / 4.0)
    np.testing.assert_array_equal(prediction, expected)
    # empty unknown pool returns the legal known counts unchanged
    zeros = BaselineB().predict(
        _features([0, 0, 0], np.zeros((3, 15)), np.zeros(15))
    ).reshape(3, 15)
    assert zeros.tolist() == np.zeros((3, 15)).tolist()


# ---------------------------------------------------------------------------
# records_hash canonicalisation
# ---------------------------------------------------------------------------


def test_canonical_hash_survives_json_roundtrip_and_delivered_jsonl(tmp_path):
    record = {
        "group_id": "g",
        "public": list(range(64)),
        "ledger": {"known_owner": {2: 1, 10: 3}, "remaining": [1, 2, 3]},
        "label_counts": [[0] * 15] * 3,
    }
    delivered = tmp_path / "samples.jsonl"
    delivered.write_text(json.dumps([record])[1:-1] + "\n", encoding="utf-8")
    loaded = [json.loads(delivered.read_text(encoding="utf-8"))]
    in_memory = dataset_hash([record])
    roundtrip = dataset_hash([json.loads(json.dumps(record))])
    from_file = dataset_hash(loaded)
    assert in_memory == roundtrip == from_file


def test_canonical_hash_rejects_int_str_key_collision_and_non_finite():
    with pytest.raises(ValueError, match="collision"):
        canonical_records_json([{"ledger": {1: "a", "1": "b"}}])
    with pytest.raises(ValueError, match="non-finite"):
        canonical_records_json([{"value": float("nan")}])
    with pytest.raises(ValueError, match="non-finite"):
        canonical_records_json([{"value": float("inf")}])
    with pytest.raises(TypeError):
        canonical_records_json([{"value": object()}])


def test_canonical_list_order_is_preserved():
    first = [{"x": [1, 2, 3]}]
    second = [{"x": [3, 2, 1]}]
    assert dataset_hash(first) != dataset_hash(second)
    assert canonical_records_value({"b": 1, "a": 2}) == {"a": 2, "b": 1}


def test_delivered_b0_samples_are_recomputable():
    root = Path(__file__).resolve().parents[2]
    samples = root / "runs" / "phase5" / "b0" / "b0_r4.samples.jsonl"
    if not samples.is_file():  # evidence not shipped with a clean wheel
        pytest.skip("delivered B0 samples are absent")
    records = [
        json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(records) == 4096
    canonical = dataset_hash(records)
    again = dataset_hash([json.loads(json.dumps(record)) for record in records])
    assert canonical == again
    # The old algorithm sorted the in-memory *integer* keys numerically.  Its
    # hash differed from the delivered string-key form for records that carry a
    # nested int-key mapping (root cause of the delivery mismatch).
    ledger_keys = [
        key for record in records
        if isinstance(record.get("ledger"), dict)
        for key in (record["ledger"].get("known_owner") or {})
    ]
    assert all(isinstance(key, str) for key in ledger_keys)


# ---------------------------------------------------------------------------
# Symmetry artifact gate / disabled parity / enabled KL
# ---------------------------------------------------------------------------


def _tiny_agent() -> Agent:
    cfg = TrainConfig(
        game=python_game.GAME_SPEC, device="cpu",
        rollout_batch_size=2, collect_batch_size=1,
        max_sampler_bytes_per_rank=512 * 2**20,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.optimizer.use_muon = False
    torch.manual_seed(7)
    return Agent(cfg, enable_value_head=True, enable_ntp_head=False)


def _synthetic_artifact(tmp_path: Path, agent: Agent, **overrides) -> tuple[Path, Path]:
    checkpoint = tmp_path / "init.pt"
    checkpoint.write_bytes(b"phase5-initial-checkpoint")
    document = {
        "schema": "guandan-phase5-symmetry-diagnostic-run-v1",
        "status": "COMPLETED",
        "training": False,
        "truncated": False,
        "deals_requested": 16,
        "design_hash": phase5_design_hash(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_policy_identity": {
            "architecture": "v1.3",
            "algorithm": "ippo",
            "history_dim": 16,
            "feature_dim": 24,
            "parameter_count": sum(value.numel() for value in agent.state_dict().values()),
            "state_dict_keys": len(agent.state_dict()),
        },
        "decision": {
            "status": "SIGNIFICANT",
            "augmentation_recommended": True,
            "sample_count": 1536,
            "deal_count": 16,
            "non_finite_kl_count": 0,
            "mean_kl": 0.02857,
            "argmax_disagreement_rate": 0.11263,
            "thresholds": {
                "min_macro_samples": 256,
                "min_deals": 8,
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
    document.update(overrides)
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(document), encoding="utf-8")
    seal = {
        "schema": "guandan-marvel-phase5-symmetry-artifact-seal-v1",
        "artifact_path": str(artifact),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "content_hash": canonical_hash(document),
        "design_hash": document["design_hash"],
        "checkpoint_sha256": document["checkpoint_sha256"],
        "source_policy_identity": document["source_policy_identity"],
        "transforms": ["copy_swap"],
    }
    seal["seal_hash"] = canonical_hash(seal)
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    return seal_path, checkpoint


def test_symmetry_default_is_off_and_reads_nothing(monkeypatch):
    config = SymmetryConfig()
    assert config.symmetry_weight == 0.0
    assert config.diagnostic_artifact is None
    with pytest.raises(ValueError, match="not implemented"):
        SymmetryConfig(symmetry_weight=0.5, augmentation=True)
    with pytest.raises(SymmetryArtifactError):
        SymmetryConfig(symmetry_weight=0.5)


def test_symmetry_gate_rejects_tampering_and_wrong_identity(tmp_path):
    agent = _tiny_agent()
    seal_path, checkpoint = _synthetic_artifact(tmp_path, agent)
    contract = load_sealed_symmetry_artifact(seal_path)
    assert contract.decision["status"] == "SIGNIFICANT"
    with pytest.raises(SymmetryArtifactError, match="checkpoint"):
        validate_config_artifact(
            artifact_path=str(seal_path),
            checkpoint_path=str(tmp_path / "wrong.pt"),
        )
    # wrong checkpoint bytes
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"different")
    with pytest.raises(SymmetryArtifactError, match="SHA256"):
        validate_config_artifact(artifact_path=str(seal_path), checkpoint_path=str(wrong))
    # tampered seal hash
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["transforms"] = ["seat_rotation_1"]
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(SymmetryArtifactError, match="seal_hash"):
        load_sealed_symmetry_artifact(seal_path)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"decision": {"status": "NOT_SIGNIFICANT"}}, "SIGNIFICANT"),
        ({"schema": "guandan-marvel-phase5-symmetry-diagnostic-v1"}, "schema"),
        ({"design_hash": "0" * 64}, "design hash"),
        ({"truncated": True}, "truncated"),
    ],
)
def test_symmetry_gate_rejects_stale_or_non_significant_artifacts(
    tmp_path, override, message,
):
    agent = _tiny_agent()
    seal_path, checkpoint = _synthetic_artifact(tmp_path, agent, **override)
    if "decision" in override:
        # A minimal decision block must still trigger the status check first.
        pass
    with pytest.raises(SymmetryArtifactError, match=message):
        validate_config_artifact(artifact_path=str(seal_path), checkpoint_path=str(checkpoint))


def _symmetry_rollout() -> TensorDict:
    torch.manual_seed(11)
    b, t, p, h, c, a, l = 1, 2, 4, 48, 512, 3, 8
    obs_tokens = torch.zeros((b, t, p, h), dtype=torch.long)
    obs_tokens[..., 0] = 1
    obs_tokens[..., 1] = 256
    obs_tokens[..., 2] = 2
    obs_tokens[..., 3] = 26
    obs_tokens[..., 4] = 38
    obs_tokens[..., 5] = 308
    obs_channels = torch.zeros((b, t, p, c), dtype=torch.uint8)
    for observer in range(p):
        obs_channels[:, :, observer, 364 + observer] = 1
        obs_channels[:, :, observer, 384] = 1
        obs_channels[:, :, observer, 388 + 4] = 1
        obs_channels[:, :, observer, 405 + (observer % 2)] = 1
        obs_channels[:, :, observer, 0 + observer] = 1
    rows = torch.zeros((b, t, a, l), dtype=torch.long)
    for index, token in enumerate((6, 64 + 3, 64 + 57)):
        rows[:, :, index] = torch.tensor([262, 2, 263, 296, 304, token, 306, 307])
    weights = torch.zeros((b, t, a))
    weights[..., :3] = 1.0
    return TensorDict({
        "num_steps": torch.tensor([2], dtype=torch.long),
        "player_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "obs_tokens": obs_tokens,
        "obs_channels": obs_channels,
        "avail_tokens": rows,
        "avail_weights": weights,
    }, batch_size=[b])


def test_symmetry_enabled_kl_is_finite_and_backpropagates(tmp_path):
    agent = _tiny_agent()
    seal_path, checkpoint = _synthetic_artifact(tmp_path, agent)
    contract = load_sealed_symmetry_artifact(seal_path)
    from guandan_marvel.phase5.symmetry_training import validate_contract_against_model

    validate_contract_against_model(
        contract, agent=agent, architecture="v1.3", algorithm="ippo",
        history_dim=16, feature_dim=24,
    )
    consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
    rollout = _symmetry_rollout()
    output = agent(rollout)
    before = {name: value.detach().clone() for name, value in agent.state_dict().items()}
    result = consistency.loss(
        rollout, base_log_probs=output.avail_log_probs,
        forward=lambda batch: agent(batch), world_size=1,
    )
    assert result.token_count_global == 2
    assert np.isfinite(result.mean_kl_global)
    assert result.mean_kl_global >= 0.0
    assert torch.isfinite(result.loss)
    assert result.loss.requires_grad
    result.loss.backward()
    agent.zero_grad(set_to_none=True)
    output = agent(rollout)
    result = consistency.loss(
        rollout, base_log_probs=output.avail_log_probs,
        forward=lambda batch: agent(batch), world_size=1,
    )
    result.loss.backward()
    gradients = [
        parameter.grad for parameter in agent.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in gradients)
    # The original forward is detached: every weight-update gradient comes from
    # the transformed branch only, and no parameter was mutated in place.
    for name, value in agent.state_dict().items():
        assert torch.equal(value, before[name])


def test_disabled_consistency_never_forwards_or_reads(monkeypatch, tmp_path):
    agent = _tiny_agent()
    calls = {"forward": 0}

    def forward(batch):
        calls["forward"] += 1
        return agent(batch)

    rollout = _symmetry_rollout()
    output = forward(rollout)
    assert calls["forward"] == 1
    trainer_like = type("T", (), {"symmetry": None, "cfg": type("C", (), {"symmetry": SymmetryConfig()})()})()
    # _symmetry_loss is a LocalTrainerBase helper; its disabled branch must not
    # touch the model at all.
    from guandan_marvel.training.local import LocalTrainerBase

    result = LocalTrainerBase._symmetry_loss(trainer_like, rollout, output)
    assert result is None
    assert calls["forward"] == 1
