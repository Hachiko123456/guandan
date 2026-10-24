# SPDX-License-Identifier: MIT
"""MARVEL v1.4 phase-5 conditional enhancements.

Phase 5 adds two *conditional* feature families on top of the accepted
phase-1..4 stack:

``symmetry``
    seat rotation ``k = 0..3`` and same-rank/same-suit double-deck copy-ID
    swap, with the matching macro-action permutation and the aligned
    ``KL(pi(a|s) || pi(P_T(a)|T(s)))`` comparison.  Diagnostics only unless a
    pre-registered threshold is exceeded.

``belief``
    B0 is an offline information gate (do visible histories carry extra
    information about hidden hands beyond public information?).  B1 is a
    training-time belief head that may only be built when B0 PASSES.

Every default is *off*: ``TrainConfig.symmetry.symmetry_weight == 0`` and
``TrainConfig.symmetry.augmentation is False`` and
``TrainConfig.belief.belief_weight == 0`` must leave the accepted phase-4
forward/loss/RNG trajectory bitwise unchanged.
"""

from __future__ import annotations

from .design import (
    PHASE5_DESIGN_SCHEMA, PHASE5_SYMMETRY_DEFAULTS, PHASE5_BELIEF_DEFAULTS,
    PHASE5_DESIGN, phase5_design_payload, phase5_design_hash,
    PHASE5_B0_PLAN, PHASE5_B0_RESULT_SCHEMA, PHASE5_BUDGET, PHASE5_BUCKETS,
    PHASE5_BELIEF_GATE_SCHEMA, PHASE5_BELIEF_PROBE_SCHEMA,
    PHASE5_PUBLIC_LEDGER_SCHEMA, PUBLIC_FEATURE_WIDTH,
    PRINTED_RANK_LABELS, COUNT_CLASSES, RELATIVE_SEATS,
    SYMMETRY_SIGNIFICANCE, SYMMETRY_TRANSFORMS,
)
from .symmetry import (
    SYMMETRY_SCHEMA, SeatRotation, CopySwap, ComposedTransform,
    compose_transforms, identity_transform,
    transform_seat, inverse_seat, transform_observation, transform_candidate_rows,
    inverse_candidate_rows, macro_action_permutation, check_transform_involution,
)
from .diagnostics import (
    SYMMETRY_DIAGNOSTIC_SCHEMA, aligned_policy_comparison, restricted_candidate_kl,
    bucket_key, aggregate_diagnostic, decide_augmentation,
)
from .belief_data import (
    PublicLedger, PublicLedgerError, build_label, canonical_deal_identity,
    canonical_physical_deal, channel_phase, conservation_audit, dataset_hash,
    decode_public_ledger, group_split, label_class_mask, leakage_probe_identity,
    parse_public_event_row, printed_rank_index, public_balance_features,
    public_event_from_observations, public_history_bits, public_history_counts,
    public_ledger_features, public_ledger_snapshot, public_unseen_by_rank,
    rank_counts, split_hash, validate_public_features,
)
from .belief_probe import (
    BaselineA, BaselineB, ProbeFit,
    b0_gate, build_probe_features, count_metrics, paired_bootstrap_ci,
    require_cuda_device, run_b0_probes, severe_buckets, train_linear_probe,
)
from .gate import (
    BeliefGateError, assert_belief_head_allowed, load_b0_gate, seal_gate_artifact,
)

__all__ = [
    "PHASE5_DESIGN_SCHEMA", "PHASE5_SYMMETRY_DEFAULTS", "PHASE5_BELIEF_DEFAULTS",
    "PHASE5_DESIGN", "phase5_design_payload", "phase5_design_hash",
    "PHASE5_B0_PLAN", "PHASE5_B0_RESULT_SCHEMA", "PHASE5_BUDGET", "PHASE5_BUCKETS",
    "PHASE5_BELIEF_GATE_SCHEMA", "PHASE5_BELIEF_PROBE_SCHEMA",
    "PHASE5_PUBLIC_LEDGER_SCHEMA", "PUBLIC_FEATURE_WIDTH",
    "PRINTED_RANK_LABELS", "COUNT_CLASSES", "RELATIVE_SEATS",
    "SYMMETRY_SIGNIFICANCE", "SYMMETRY_TRANSFORMS",
    "SYMMETRY_SCHEMA", "SeatRotation", "CopySwap", "ComposedTransform",
    "compose_transforms",
    "identity_transform", "transform_seat", "inverse_seat", "transform_observation",
    "transform_candidate_rows", "inverse_candidate_rows", "macro_action_permutation",
    "check_transform_involution",
    "SYMMETRY_DIAGNOSTIC_SCHEMA", "aligned_policy_comparison",
    "restricted_candidate_kl", "bucket_key", "aggregate_diagnostic",
    "decide_augmentation",
    "PublicLedger", "PublicLedgerError", "build_label", "canonical_deal_identity",
    "canonical_physical_deal", "channel_phase", "conservation_audit",
    "dataset_hash", "decode_public_ledger", "group_split", "label_class_mask",
    "leakage_probe_identity", "parse_public_event_row", "printed_rank_index",
    "public_balance_features", "public_event_from_observations",
    "public_history_bits", "public_history_counts", "public_ledger_features",
    "public_ledger_snapshot", "public_unseen_by_rank", "rank_counts",
    "split_hash", "validate_public_features",
    "BaselineA", "BaselineB", "ProbeFit", "b0_gate", "build_probe_features",
    "count_metrics", "paired_bootstrap_ci", "require_cuda_device", "run_b0_probes",
    "severe_buckets", "train_linear_probe",
    "BeliefGateError", "assert_belief_head_allowed", "load_b0_gate",
    "seal_gate_artifact",
]
