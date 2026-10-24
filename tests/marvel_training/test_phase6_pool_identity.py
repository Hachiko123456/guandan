"""Phase-6 repair: manifest identity must agree with the loaded artifact."""
from __future__ import annotations

import pytest

from guandan_marvel.config import OpponentPoolConfig
from guandan_marvel.phase6.opponent_pool import OpponentPool, OpponentPoolError

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
    return make_tiny_eval_checkpoint(tmp_path_factory.mktemp("phase6-identity"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture", "v1.3-COMPLETELY-WRONG"),
        ("protocol_version", "BOGUS"),
        ("backend", "python"),
    ],
)
def test_false_manifest_identity_is_rejected(fixture, field, value):
    entry = dict(fixture["entry"])
    entry[field] = value
    config = OpponentPoolConfig(enabled=True, entries=(entry,), sampling_seed=5)
    with pytest.raises(OpponentPoolError, match="manifest"):
        OpponentPool.from_config(config, **_common(fixture))


def test_empty_identity_field_is_rejected_before_loading(fixture):
    entry = dict(fixture["entry"])
    entry["protocol_version"] = ""
    with pytest.raises(OpponentPoolError, match="protocol_version"):
        OpponentPool.from_config(
            OpponentPoolConfig(enabled=True, entries=(entry,), sampling_seed=5),
            **_common(fixture),
        )
