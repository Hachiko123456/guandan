# SPDX-License-Identifier: MIT
"""Versioned M06 evaluation, champion, and opponent-pool system."""
from .checkpoints import (
    EVALUATION_CHECKPOINT_SCHEMA,
    LegacyV1OpponentArtifact,
    V2PolicyArtifact,
    freeze_merged_policy,
    load_legacy_v1_opponent,
    load_v2_policy,
    verify_policy_identity,
)
from .evaluator import EvaluationRunner, build_evaluation_plan
from .hooks import TrainingEvaluatorHook
from .manifests import (
    DEALS_MANIFEST_SCHEMA,
    FinalHoldoutAudit,
    build_deals_manifest,
    dataset_identity,
    validate_deals_manifest,
)
from .promotion import ChampionStore, PromotionPolicy, decide_promotion
from .provenance import materialize_deal_identity, verify_data_isolation
from .pool import OpponentPoolStore
from .stats import clustered_mean_ci95, summarize_records

__all__ = [
    "ChampionStore", "DEALS_MANIFEST_SCHEMA", "EVALUATION_CHECKPOINT_SCHEMA",
    "EvaluationRunner", "FinalHoldoutAudit", "LegacyV1OpponentArtifact",
    "OpponentPoolStore", "PromotionPolicy", "TrainingEvaluatorHook", "V2PolicyArtifact",
    "build_deals_manifest", "build_evaluation_plan", "clustered_mean_ci95",
    "dataset_identity", "decide_promotion", "freeze_merged_policy",
    "load_legacy_v1_opponent", "load_v2_policy", "materialize_deal_identity",
    "summarize_records", "validate_deals_manifest", "verify_data_isolation",
    "verify_policy_identity",
]
