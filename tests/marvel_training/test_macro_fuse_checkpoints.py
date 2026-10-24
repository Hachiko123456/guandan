"""Phase-3 versioned checkpoint + policy-only warm-start migration tests.

Covers: v1.3 and v1.4 strict evaluation loads, a pseudo-v1.4 artifact that omits
``actor_fuse.*`` (must fail), the explicit ``v1.3 -> v1.4-macro-fuse`` policy-only
warm start with its prefix whitelist, and the refusal to treat a v1.3 training
recovery as a v1.4 exact resume.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from guandan_marvel.config import MACRO_FUSE_MODEL_ARCHITECTURE
from guandan_marvel.evaluation.checkpoints import freeze_merged_policy, load_v2_policy
from guandan_marvel.training.checkpoints import (
    CheckpointCompatibilityError, TrainingCheckpointStore, canonical_json_hash, classify_resume,
)
from guandan_marvel.training.local import IPPOTrainer
from tests.marvel_training._fuse_fixtures import (
    FUSE_ARCH, LEGACY_ARCH, capture_logits, game_rollout, smoke_trainer_config,
)

FUSE_PREFIX = "actor_fuse."
LEGACY_ARCHITECTURES = ("v1.3",)


@pytest.fixture(scope="module")
def smoke_trainers(tmp_path_factory):
    """A v1.3 and a v1.4 trainer on the same CppSmokeGame config (no iterations)."""
    base = tmp_path_factory.mktemp("phase3-checkpoints")
    legacy_cfg, legacy_game = smoke_trainer_config(LEGACY_ARCH, run_name = "ckpt-v13")
    fused_cfg, fused_game = smoke_trainer_config(FUSE_ARCH, run_name = "ckpt-v14")
    legacy = IPPOTrainer(legacy_cfg, game = legacy_game, run_dir = base / "legacy")
    fused = IPPOTrainer(fused_cfg, game = fused_game, run_dir = base / "fused")
    try:
        yield legacy, fused, base
    finally:
        legacy.close()
        fused.close()


def _freeze(trainer, path: Path) -> Path:
    return freeze_merged_policy(trainer, path)


# ---------------------------------------------------------------------------
# Strict evaluation loads
# ---------------------------------------------------------------------------


def test_v13_evaluation_checkpoint_loads_strict_without_a_fuse_branch(smoke_trainers, tmp_path):
    legacy, _fused, _base = smoke_trainers
    path = _freeze(legacy, tmp_path / "v13.pt")
    artifact = load_v2_policy(path, device = "cpu", expected_algorithm = "ippo")
    assert artifact.architecture == "v1.3"
    assert artifact.model.actor_fuse is None
    assert not any(n.startswith(FUSE_PREFIX) for n, _ in artifact.model.named_parameters())


def test_v14_evaluation_checkpoint_loads_strict_with_the_fuse_branch(smoke_trainers, tmp_path):
    _legacy, fused, _base = smoke_trainers
    path = _freeze(fused, tmp_path / "v14.pt")
    artifact = load_v2_policy(path, device = "cpu", expected_algorithm = "ippo")
    assert artifact.architecture == MACRO_FUSE_MODEL_ARCHITECTURE
    assert artifact.model.actor_fuse is not None
    # strict=True means the reloaded parameter set is exactly the saved one.
    assert set(artifact.model.state_dict()) == set(fused.agent_merged.state_dict())
    fuse_keys = {n for n in artifact.model.state_dict() if n.startswith(FUSE_PREFIX)}
    assert fuse_keys and fuse_keys <= set(fused.agent_merged.state_dict())


def test_relaxed_comparison_loader_does_not_whitelist_the_fuse_architecture(smoke_trainers, tmp_path):
    """The comparison pipeline's relaxed fallback stays fail-closed for v1.4."""
    _legacy, fused, _base = smoke_trainers
    from guandan_marvel.comparison.loader import _load_v2_policy_relaxed

    path = _freeze(fused, tmp_path / "v14-relaxed.pt")
    with pytest.raises(ValueError, match="relaxed loader requires architecture v1.3"):
        _load_v2_policy_relaxed(path, device = "cpu", expected_algorithm = "ippo")


def test_pseudo_v14_checkpoint_without_actor_fuse_keys_is_rejected(tmp_path):
    cfg, game = smoke_trainer_config(LEGACY_ARCH, run_name = "pseudo")
    trainer = IPPOTrainer(cfg, game = game, run_dir = tmp_path / "run")
    try:
        v13_path = _freeze(trainer, tmp_path / "v13.pt")
        payload = TrainingCheckpointStore.load(v13_path, device = "cpu")
        assert not any(key.startswith(FUSE_PREFIX) for key in payload["agent_state_dict"])

        forged = copy.deepcopy(dict(payload))
        forged["architecture"] = MACRO_FUSE_MODEL_ARCHITECTURE
        forged["resolved_config"]["model"]["architecture"] = MACRO_FUSE_MODEL_ARCHITECTURE
        forged["resolved_config_hash"] = canonical_json_hash(forged["resolved_config"])
        forged_path = tmp_path / "forged-v14.pt"
        torch.save(forged, forged_path)

        # The generic loader reconstructs a v1.4 model from resolved_config and
        # loads with strict=True, so the missing actor_fuse keys fail closed.
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            load_v2_policy(forged_path, device = "cpu", expected_algorithm = "ippo")
        assert "actor_fuse" in str(excinfo.value)
    finally:
        trainer.close()


def test_evaluation_loader_rejects_legacy_architectures(tmp_path):
    cfg, game = smoke_trainer_config(LEGACY_ARCH, run_name = "legacy-arch")
    trainer = IPPOTrainer(cfg, game = game, run_dir = tmp_path / "run")
    try:
        path = _freeze(trainer, tmp_path / "v13.pt")
        payload = dict(TrainingCheckpointStore.load(path, device = "cpu"))
        payload["architecture"] = "v1.2"
        payload["resolved_config"]["model"]["architecture"] = "v1.2"
        payload["resolved_config_hash"] = canonical_json_hash(payload["resolved_config"])
        target = tmp_path / "v12.pt"
        torch.save(payload, target)
        with pytest.raises(ValueError, match = "MARVEL V2 architecture"):
            load_v2_policy(target, device = "cpu", expected_algorithm = "ippo")
    finally:
        trainer.close()


# ---------------------------------------------------------------------------
# Policy-only warm start migration
# ---------------------------------------------------------------------------


def test_v13_policy_warm_start_into_v14_records_migration_and_is_probe_identical(tmp_path):
    legacy_cfg, legacy_game = smoke_trainer_config(LEGACY_ARCH, run_name = "warm-v13")
    fused_cfg, fused_game = smoke_trainer_config(FUSE_ARCH, run_name = "warm-v14")
    legacy_trainer = IPPOTrainer(legacy_cfg, game = legacy_game, run_dir = tmp_path / "legacy")
    fused_trainer = IPPOTrainer(fused_cfg, game = fused_game, run_dir = tmp_path / "fused")
    try:
        path = _freeze(legacy_trainer, tmp_path / "v13.pt")
        payload = TrainingCheckpointStore.load(path, device = "cpu")
        assert torch.count_nonzero(legacy_trainer.agent_merged.logit_head.weight) > 0

        report = fused_trainer.load_warm_start_policy(payload)
        migration = report["migration"]
        assert migration["schema"] == "guandan-marvel-policy-warm-start-migration-v1"
        assert migration["source_architecture"] == "v1.3"
        assert migration["target_architecture"] == MACRO_FUSE_MODEL_ARCHITECTURE
        assert migration["fuse_migrated"] is True
        assert migration["output_proj_zero_verified"] is True
        assert migration["permitted_missing_prefixes"] == ["actor_fuse.", "ntp_head."]

        # The migrated residual gate is exactly zero on both live policies.
        for module in (fused_trainer.agent_resume, fused_trainer.agent_merged):
            assert torch.count_nonzero(module.actor_fuse.output_proj.weight) == 0

        # Fixed probe: the pre-migration v1.3 policy and the post-migration v1.4
        # policy must agree bitwise on identical inputs.
        rollout = game_rollout(fused_game, batch = 2, steps = 3, seed = 91)
        source = load_v2_policy(path, device = "cpu", expected_algorithm = "ippo").model
        source.eval()
        fused_trainer.agent_merged.eval()
        with torch.no_grad():
            old_out, _ = capture_logits(source, rollout)
            new_out, _ = capture_logits(fused_trainer.agent_merged, rollout)
        assert torch.equal(old_out.avail_probs, new_out.avail_probs)
        assert torch.equal(old_out.avail_log_probs, new_out.avail_log_probs)
        assert torch.equal(old_out.player_state_value, new_out.player_state_value)
        assert torch.equal(
            torch.argmax(old_out.avail_probs, dim = -1),
            torch.argmax(new_out.avail_probs, dim = -1),
        )
        # Warm start resets optimizer / sampler / counters.
        assert report["optimizer_reset"] and report["sampler_reset"] and report["counters_reset"]
        assert fused_trainer.iteration == 0
    finally:
        legacy_trainer.close()
        fused_trainer.close()


def test_warm_start_prefix_whitelist_rejects_missing_legacy_and_unexpected_keys(tmp_path):
    legacy_cfg, legacy_game = smoke_trainer_config(LEGACY_ARCH, run_name = "wl-v13")
    fused_cfg, fused_game = smoke_trainer_config(FUSE_ARCH, run_name = "wl-v14")
    legacy_trainer = IPPOTrainer(legacy_cfg, game = legacy_game, run_dir = tmp_path / "legacy")
    fused_trainer = IPPOTrainer(fused_cfg, game = fused_game, run_dir = tmp_path / "fused")
    try:
        payload = dict(TrainingCheckpointStore.load(
            _freeze(legacy_trainer, tmp_path / "v13.pt"), device = "cpu",
        ))

        # (a) a missing *legacy* key is not covered by the actor_fuse./ntp_head. prefixes.
        dropped = copy.deepcopy(payload)
        dropped["agent_state_dict"].pop("logit_head.weight")
        with pytest.raises(ValueError, match = "missing=\\['logit_head.weight'\\]"):
            fused_trainer.load_warm_start_policy(dropped)

        # (a2) another legacy key, deeper in the backbone.
        dropped_backbone = copy.deepcopy(payload)
        dropped_backbone["agent_state_dict"].pop("decoder.norm.weight")
        with pytest.raises(ValueError, match = "decoder.norm.weight"):
            fused_trainer.load_warm_start_policy(dropped_backbone)

        # (b) an unexpected key is never tolerated.
        extra = copy.deepcopy(payload)
        extra["agent_state_dict"]["ghost.weight"] = torch.zeros(2, 2)
        with pytest.raises(ValueError, match = "unexpected"):
            fused_trainer.load_warm_start_policy(extra)

        # (c) a same-architecture v1.4 warm start must not lose actor_fuse keys.
        fused_payload = TrainingCheckpointStore.load(
            _freeze(fused_trainer, tmp_path / "v14.pt"), device = "cpu",
        )
        fused_dropped = copy.deepcopy(dict(fused_payload))
        removed = fused_dropped["agent_state_dict"].pop("actor_fuse.output_proj.weight")
        assert removed.shape == (24, 24)
        with pytest.raises(ValueError, match = "actor_fuse.output_proj.weight"):
            fused_trainer.load_warm_start_policy(fused_dropped)

        # (d) the pristine checkpoint still migrates fine (control).
        assert fused_trainer.load_warm_start_policy(payload)["migration"]["fuse_migrated"] is True
    finally:
        legacy_trainer.close()
        fused_trainer.close()


def test_warm_start_rejects_unlisted_architecture_transitions(tmp_path):
    fused_cfg, fused_game = smoke_trainer_config(FUSE_ARCH, run_name = "rev-v14")
    legacy_cfg, legacy_game = smoke_trainer_config(LEGACY_ARCH, run_name = "rev-v13")
    fused_trainer = IPPOTrainer(fused_cfg, game = fused_game, run_dir = tmp_path / "fused")
    legacy_trainer = IPPOTrainer(legacy_cfg, game = legacy_game, run_dir = tmp_path / "legacy")
    try:
        v14_payload = TrainingCheckpointStore.load(
            _freeze(fused_trainer, tmp_path / "v14.pt"), device = "cpu",
        )
        # v1.4 -> v1.3 is not an allowed transition.
        with pytest.raises(ValueError, match = "not a permitted transition"):
            legacy_trainer.load_warm_start_policy(v14_payload)

        # A pre-v1.3 source is rejected outright.
        forged = copy.deepcopy(dict(v14_payload))
        forged["architecture"] = "v1.2"
        with pytest.raises(ValueError, match = "not a permitted transition"):
            fused_trainer.load_warm_start_policy(forged)
    finally:
        fused_trainer.close()
        legacy_trainer.close()


# ---------------------------------------------------------------------------
# Exact resume refusal
# ---------------------------------------------------------------------------


def test_v13_training_recovery_is_not_accepted_as_v14_exact_resume(tmp_path):
    legacy_cfg, legacy_game = smoke_trainer_config(LEGACY_ARCH, run_name = "res-v13")
    fused_cfg, fused_game = smoke_trainer_config(FUSE_ARCH, run_name = "res-v14")
    legacy_trainer = IPPOTrainer(legacy_cfg, game = legacy_game, run_dir = tmp_path / "legacy")
    fused_trainer = IPPOTrainer(fused_cfg, game = fused_game, run_dir = tmp_path / "fused")
    try:
        legacy_trainer.train_iteration()
        path = legacy_trainer.save_checkpoint(tmp_path / "v13-training.pt")
        payload = TrainingCheckpointStore.load(path, device = "cpu")
        assert payload["architecture"] == "v1.3"

        # Classification-level evidence: the architecture differs, so this is a
        # migration, never an exact/compatible resume.
        decision = classify_resume(
            payload,
            expected_contract = fused_trainer.semantic_contract,
            expected_source_fingerprint = fused_trainer.source_fingerprint,
            expected_execution_fingerprint = fused_trainer.execution_fingerprint,
            expected_backend_identity = fused_trainer.backend_identity,
            expected_config = fused_trainer.resolved_config,
            require_replay = False,
        )
        assert decision.kind == "migration"
        assert any("architecture" in reason for reason in decision.reasons)

        # End-to-end: the v1.4 trainer refuses to adopt it.
        with pytest.raises(CheckpointCompatibilityError):
            fused_trainer.load_checkpoint_payload(payload)
        assert fused_trainer.failed is True
        assert fused_trainer.failure["exact_resume_allowed"] is False

        # The same checkpoint is a *valid* v1.3 recovery for the v1.3 trainer.
        control_cfg, control_game = smoke_trainer_config(LEGACY_ARCH, run_name = "res-control")
        control = IPPOTrainer(control_cfg, game = control_game, run_dir = tmp_path / "control")
        try:
            decision = control.load_checkpoint_payload(payload)
            assert decision.kind in {"exact_resume", "compatible_resume"}
        finally:
            control.close()
    finally:
        fused_trainer.close()
        legacy_trainer.close()


def test_v14_training_checkpoint_records_amp_and_scaler_state(tmp_path):
    cfg, game = smoke_trainer_config(FUSE_ARCH, run_name = "amp-ckpt", amp_enabled = True)
    trainer = IPPOTrainer(cfg, game = game, run_dir = tmp_path / "run")
    try:
        trainer.train_iteration()
        path = trainer.save_checkpoint(tmp_path / "v14-amp.pt")
        payload = TrainingCheckpointStore.load(path, device = "cpu")
        assert payload["architecture"] == MACRO_FUSE_MODEL_ARCHITECTURE
        assert "grad_scaler_state" in payload
        assert payload["amp"]["enabled"] is True
        assert payload["amp"]["device_type"] == "cpu"
        assert payload["amp"]["autocast_dtype"] == "torch.bfloat16"
        assert payload["amp"]["scaler_enabled"] is False
        # The saved configuration carries the AMP switch as semantic state.
        assert payload["resolved_config"]["amp_enabled"] is True
        assert payload["semantic_contract"]["semantic_config"]["execution_semantics"]["amp_enabled"] is True
        # JSON round-trip sanity on the AMP block.
        json.loads(json.dumps(payload["amp"], sort_keys = True))
    finally:
        trainer.close()
