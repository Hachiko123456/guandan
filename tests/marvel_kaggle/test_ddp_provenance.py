from __future__ import annotations

from copy import deepcopy

import pytest

from guandan_marvel.evaluation.provenance import verify_data_isolation
from guandan_marvel.kaggle.checkpoints import aggregate_training_provenance
from guandan_marvel.kaggle.common import canonical_hash
from guandan_marvel.evaluation.manifests import build_deals_manifest


def _provenance(seed: int, deal_hash: str):
    body = {
        "schema": "guandan-marvel-training-data-provenance-v1",
        "domain": "training",
        "manifest_hash": "1" * 64,
        "generator_version": "generator-v1",
        "seed_semantics": "rank-local",
        "initial_env_seed": seed,
        "deal_hashes": [deal_hash],
        "complete": True,
    }
    return {**body, "hash": canonical_hash(body)}


def test_aggregate_ddp_provenance_tracks_every_rank_seed_and_hash():
    rows = [
        {"training_data_provenance": _provenance(10, "a" * 64)},
        {"training_data_provenance": _provenance(20, "b" * 64)},
    ]
    result = aggregate_training_provenance(rows)
    assert result["initial_env_seeds"] == [10, 20]
    assert result["deal_hashes"] == ["a" * 64, "b" * 64]
    assert result["world_size"] == 2
    assert result["complete"] is True


def test_aggregate_ddp_provenance_rejects_manifest_mismatch():
    rows = [
        {"training_data_provenance": _provenance(10, "a" * 64)},
        {"training_data_provenance": _provenance(20, "b" * 64)},
    ]
    rows[1]["training_data_provenance"]["manifest_hash"] = "2" * 64
    with pytest.raises(ValueError, match="manifest_hash"):
        aggregate_training_provenance(rows)
