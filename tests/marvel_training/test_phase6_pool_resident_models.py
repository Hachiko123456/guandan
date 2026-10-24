"""Phase-6 repair: max_resident_policies bounds actual resident models."""
from __future__ import annotations

from pathlib import Path

import pytest

from guandan_marvel.phase6.opponent_pool import (
    OpponentPool, OpponentPoolError, PoolPolicyRouter,
)
from guandan_marvel.config import OpponentPoolConfig

from tests.marvel_training._phase6_pool_fixtures import make_tiny_eval_checkpoint


def _common(fixture):
    return dict(
        device="cpu",
        expected_architecture="v1.4-macro-fuse",
        expected_protocol_version=str(fixture["game"].protocol_version),
        expected_backend="cpp",
        expected_backend_identity_hash=None,
        model_feature_identity={
            "history_dim": 16, "feature_dim": 24, "decoder_layers": 1,
            "decoder_heads": 4, "state_layers": 1, "action_layers": 1,
            "fuse_layers": 1,
        },
        batch_slots=1,
    )


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    return make_tiny_eval_checkpoint(tmp_path_factory.mktemp("phase6-resident"))


def test_two_entries_with_one_resident_slot_fail_closed(fixture):
    entry = dict(fixture["entry"])
    second = dict(entry)
    second["id"] = "opponent-b"
    config = OpponentPoolConfig(
        enabled=True, entries=(entry, second), sampling_seed=3,
        max_resident_policies=1,
    )
    with pytest.raises(OpponentPoolError, match="max_resident_policies"):
        OpponentPool.from_config(config, **_common(fixture))


def test_single_entry_is_frozen_at_load_and_metrics_report_actual_models(fixture):
    config = OpponentPoolConfig(
        enabled=True, entries=(fixture["entry"],), sampling_seed=3,
        max_resident_policies=1,
    )
    pool = OpponentPool.from_config(config, **_common(fixture))
    assert pool.loaded_model_count == 1
    for parameter in pool.entries[0].artifact.model.parameters():
        assert parameter.requires_grad is False
        assert parameter.grad is None
    router = PoolPolicyRouter(
        pool=pool, batch_size=1, num_players=4, max_ctx_len=64,
        device="cpu", max_resident_policies=1,
    )
    router.prepare(rollout_buffer=None)
    metrics = router.metrics()
    assert metrics["pool/resident_model_count"] == 1
    assert metrics["pool/resident_session_count"] == 1
    assert metrics["pool/estimated_parameter_bytes"] > 0
    assert metrics["pool/estimated_cache_bytes"] > 0
