# SPDX-License-Identifier: MIT
"""Frozen phase-5 experimental design.

Everything in this module is *pre-registered*: it must be written down before
any test/final result is inspected, and it must not be changed afterwards.
The hash :func:`phase5_design_hash` is recorded in every phase-5 artifact so a
later reader can prove the design was not adapted to the observed outcome.

Nothing here is a training default.  ``PHASE5_SYMMETRY_DEFAULTS`` and
``PHASE5_BELIEF_DEFAULTS`` are both "off"; a feature may only be enabled by an
explicit config value and (for belief) by a PASS on the B0 gate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PHASE5_DESIGN_SCHEMA = "guandan-marvel-phase5-design-v2"
PHASE5_SYMMETRY_SCHEMA = "guandan-marvel-phase5-symmetry-v1"
PHASE5_DIAGNOSTIC_SCHEMA = "guandan-marvel-phase5-symmetry-diagnostic-v1"
PHASE5_BELIEF_DATASET_SCHEMA = "guandan-marvel-phase5-belief-dataset-v2"
PHASE5_BELIEF_PROBE_SCHEMA = "guandan-marvel-phase5-belief-probe-v2"
PHASE5_BELIEF_GATE_SCHEMA = "guandan-marvel-phase5-belief-gate-v1"
PHASE5_BELIEF_HEAD_SCHEMA = "guandan-marvel-phase5-belief-head-v1"
PHASE5_B0_RESULT_SCHEMA = "guandan-marvel-phase5-b0-result-v1"
# Public-information ledger carried by every v2 sample.  The *gate* schema is
# deliberately unchanged: a v1 gate is rejected because the design hash below
# changed, never because its document schema was silently reinterpreted.
PHASE5_PUBLIC_LEDGER_SCHEMA = "guandan-marvel-phase5-public-ledger-v2"

# ---------------------------------------------------------------------------
# Feature flags.  Default OFF, always.
# ---------------------------------------------------------------------------
PHASE5_SYMMETRY_DEFAULTS = {
    "symmetry_weight": 0.0,
    "augmentation": False,
}
PHASE5_BELIEF_DEFAULTS = {
    "belief_weight": 0.0,
    "build_head": False,
}

# ---------------------------------------------------------------------------
# Observation/label vocabulary (derived from the accepted V2 protocol).
# ---------------------------------------------------------------------------
# 13 printed ranks in printed order plus the two jokers: the label axis.
PRINTED_RANK_LABELS: tuple[str, ...] = (
    "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
    "TEN", "JACK", "QUEEN", "KING", "ACE", "TWO",
    "SMALL_JOKER", "BIG_JOKER",
)
NUM_PRINTED_RANKS = len(PRINTED_RANK_LABELS)
# Count classes 0..8 for the 13 normal ranks (two decks x four suits).
COUNT_CLASSES = 9
# The two jokers have exactly two physical copies each (one per deck).
JOKER_COUNT_MAX = 2
# Per-rank maximum physical copies over the whole 108-card shoe.
MAX_COPIES_BY_RANK: tuple[int, ...] = (8,) * 13 + (JOKER_COUNT_MAX, JOKER_COUNT_MAX)
# Relative seat axis: next (actor+1), partner (actor+2), previous (actor+3).
RELATIVE_SEATS: tuple[str, ...] = ("next", "partner", "previous")
RELATIVE_SEAT_OFFSETS: tuple[int, ...] = (1, 2, 3)
NUM_RELATIVE_SEATS = len(RELATIVE_SEATS)

# Public-ledger feature layout (v2, width 64):
#   [0:3]    published remaining counts of next / partner / previous
#   [3:48]   K[3 relative seats, 15 printed ranks]: publicly known cards that
#            the acting seat can prove are *currently held* by that seat
#   [48:63]  u[15]: residual physical inventory not attributable to the acting
#            seat's hand, publicly played cards, K, or known escrow.  It is the
#            disjoint union of unknown opponent hand capacity H and unknown
#            escrow capacity X, so sum(u) == H + X.
#   [63]     X: escrow cards whose identity is still private
PUBLIC_OWN_SLICE = slice(0, 0)  # documented sentinel; own hand comes from channels
PUBLIC_REMAINING_SLICE = slice(0, NUM_RELATIVE_SEATS)
PUBLIC_KNOWN_SLICE = slice(NUM_RELATIVE_SEATS, NUM_RELATIVE_SEATS + 45)
PUBLIC_UNKNOWN_SLICE = slice(
    NUM_RELATIVE_SEATS + 45, NUM_RELATIVE_SEATS + 45 + NUM_PRINTED_RANKS,
)
PUBLIC_ESCROW_UNKNOWN_INDEX = NUM_RELATIVE_SEATS + 45 + NUM_PRINTED_RANKS
PUBLIC_FEATURE_WIDTH = PUBLIC_ESCROW_UNKNOWN_INDEX + 1

# Physical card layout of the accepted engine: 2 decks x 54 cards.
CARDS_PER_DECK = 54
TOTAL_PHYSICAL_CARDS = 108
# ``card_id // 54`` is the deck index; the twin of a card is the same printed
# rank/suit in the other deck.
COPY_SWAP_OFFSET = CARDS_PER_DECK

# ---------------------------------------------------------------------------
# Symmetry transforms.
# ---------------------------------------------------------------------------
SYMMETRY_TRANSFORMS: tuple[str, ...] = (
    "seat_rotation_0", "seat_rotation_1", "seat_rotation_2", "seat_rotation_3",
    "copy_swap",
)
# A rotation by k maps seat s to (s + k) % 4.  Teams are TEAM_OF = (0, 1, 0, 1),
# so an odd rotation swaps the two teams; the reward vector is compared after
# the same seat permutation (equivariance, not equality).
SEAT_ROTATIONS: tuple[int, ...] = (0, 1, 2, 3)

# Pre-registered "significant inconsistency" threshold, decided *before* the
# diagnostic was run.  A feature may only be trained when at least one of the
# two conditions holds on at least ``min_macro_samples`` macro starts drawn from
# at least ``min_deals`` independent deals.
SYMMETRY_SIGNIFICANCE = {
    "min_macro_samples": 256,
    "min_deals": 8,
    "mean_restricted_kl_threshold": 0.02,   # nats, next-token aligned KL
    "argmax_disagreement_threshold": 0.10,  # fraction of macro starts
    "tie_rule": "lowest candidate index (engine canonical order)",
    # Full-macro KL is only claimed where the complete legal macro-action set
    # is enumerable under these caps; everything else is reported as a
    # *restricted-candidate* next-token KL and never called full-action KL.
    "full_macro_max_actions": 8,
    "full_macro_max_prefix_nodes": 24,
    "max_macro_starts_per_deal": 24,
    "deals": 16,
}

# Diagnostic buckets (spec 3.2: phase / action kind / level / leader / tribute).
PHASE5_BUCKETS: tuple[str, ...] = (
    "phase", "action_kind", "level_rank", "leader_seat", "tribute_mode",
)

# ---------------------------------------------------------------------------
# Belief B0 plan.
# ---------------------------------------------------------------------------
PHASE5_B0_PLAN: dict[str, Any] = {
    "schema": PHASE5_BELIEF_DATASET_SCHEMA,
    "engine": "cpp",
    "max_deals": 512,
    "pilot_deals": 16,
    "collection_wall_seconds_cap": 1800,
    "probe_wall_seconds_cap": 1200,
    # At most this many policy macro starts are recorded per deal, so a fixed
    # deal budget maps to a bounded sample count (512 x 8 = 4096 samples).
    "max_macro_starts_per_deal": 8,
    "collection_batch_slots": 16,
    "sample_unit": "policy_macro_start",
    "label_unit": "remaining physical cards by printed rank",
    "label_seats": list(RELATIVE_SEATS),
    "label_ranks": list(PRINTED_RANK_LABELS),
    "count_classes": COUNT_CLASSES,
    "joker_count_max": JOKER_COUNT_MAX,
    # The split unit is the *physical deal*, canonicalised over the symmetry
    # group (four seat rotations x two deck swaps) and deliberately excludes the
    # context.  A context/name change or a symmetry augmentation therefore
    # cannot move one physical deal into another split.
    "group_id": (
        "sha256 of the lexicographically minimal canonical physical deal over "
        "seat rotations k=0..3 and deck swap; level/leader/tribute context is excluded"
    ),
    "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
    "split_seed": 20261019,
    "split_unit": "group_id (physical deal), never step, context or augmentation",
    "contexts": {
        "level_ranks": list(range(3, 16)),
        "leader_seats": [0, 1, 2, 3],
        "tribute_modes": [
            "none", "DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST", "cancelled",
        ],
        "assignment": "deal_index mod len(cell) inside each stratified cell",
    },
    "causal_history": {
        "definition": (
            "every observation the acting seat actually received from deal start "
            "through the sampled macro start; public commit events contribute the "
            "seat and cards they played, resolved-exchange events contribute their "
            "complete (kind, source, target, card) transfer records, private "
            "prefix/exchange-selection events never enter another seat's history"
        ),
        "features": (
            "108 public physical-card bits + 64 public-ledger features: 3 relative "
            "remaining counts, 45 known-held K counts, 15 residual unknown-by-rank "
            "counts, 1 unknown escrow capacity"
        ),
        "ledger_schema": PHASE5_PUBLIC_LEDGER_SCHEMA,
        "ledger_invariants": (
            "buckets {own hand O, publicly played P, known-held K, known escrow E, "
            "residual unknown u} are disjoint and partition the 108 physical cards; "
            "h_s = R_s - sum_r K[s,r] >= 0; u_r >= 0; sum_r u_r == sum_s h_s + X"
        ),
    },
    "baselines": {
        "A": (
            "unconditional empirical prior: per (relative seat, printed rank) "
            "training-set mean count; no per-sample information"
        ),
        "B": (
            "public-information lower-bounded proportional allocation (rule based, "
            "CPU): pred[s,r] = K[s,r] + h_s * u_r / sum(u) with h_s = R_s - "
            "sum_r K[s,r]; publicly transferred cards that are still held are "
            "counted in K for their current owner, played cards only in P, and "
            "unknown escrow capacity is retained in the denominator.  No hidden "
            "hand, oracle label or future event is read."
        ),
        "C": "linear probe on the current 512-dim observation channels (CUDA, FP32)",
        "D": (
            "linear probe on the full causal public history + current channels: "
            "108 public physical-card bits + 64 public-ledger features + current "
            "512 observation channels (CUDA, FP32)"
        ),
    },
    "probe": {
        "seeds": [0, 1, 2],
        "epochs": 40,
        "batch_size": 256,
        "optimizer": "AdamW",
        "learning_rate": 3.0e-3,
        "weight_decay": 1.0e-2,
        "early_stop_patience": 6,
        "loss": "masked L1 on counts; predictions clamped to the physical rank maximum",
        "standardization": "train-split mean/std stored with the probe",
        "capacity": {
            "C": "Linear(512, 45)",
            "D": "Linear(108 + 64 + 512 = 684, 45)",
        },
        "device": "cuda",
        "dtype": "float32",
        "primary_metric": "count_mae",
        "primary_metric_definition": (
            "mean over samples and over the 45 (relative seat, printed rank) entries "
            "of |predicted count - true count|; lower is better"
        ),
        "secondary_metrics": [
            "count_l1", "rank_mae", "exact_count_accuracy", "conservation_error",
            "calibration_ece",
        ],
        "validation_use": "capacity/early-stop selection only",
        "test_use": "single confirmatory evaluation after the design is locked",
    },
    "gate": {
        "primary": "improvement = B_error - D_error (positive means D better)",
        "epsilon": 0.05,
        "epsilon_note": (
            "0.05 counts of average absolute count error per (seat, rank) entry; "
            "a deliberate minimum practical effect, not a statistical-only epsilon"
        ),
        "bootstrap": {
            "cluster": "group_id (canonical deal)",
            "resamples": 2000,
            "seed": 12345,
            "confidence": 0.95,
            "weights": "each deal contributes one paired improvement value; seeds averaged inside the deal",
        },
        "sufficiency": {
            "min_completed_deals": 512,
            "min_test_deals": 32,
            "min_test_samples": 64,
            "min_test_level_ranks": 2,
            "min_test_leader_seats": 2,
            "min_test_tribute_modes": 2,
            "require_disjoint_splits": True,
            "require_leakage_check": True,
        },
        "pass": (
            "sufficient sample AND CI lower bound > epsilon AND every pre-registered "
            "seed has a positive improvement AND no leakage/group contamination AND "
            "no severe bucket degradation"
        ),
        "reject": "sufficient sample AND CI upper bound <= epsilon",
        "inconclusive": (
            "budget/sample/coverage insufficient, CI straddles epsilon, leakage or "
            "group contamination, or a numerical degeneracy"
        ),
        "severe_bucket_degradation": (
            "a pre-registered bucket with >= 30 samples whose improvement CI upper bound < 0"
        ),
    },
}

PHASE5_BUDGET: dict[str, Any] = {
    "pilot_deals": 16,
    "b0_max_deals": 512,
    "collection_wall_seconds_cap": 1800,
    "probe_wall_seconds_cap": 1200,
    "micro_rl_iterations_per_arm": 10,
    "micro_rl_max_arms": 4,
    "forbidden": [
        "50+ iteration formal gate", "Kaggle", "dual T4 / NCCL",
        "final holdout", "FableDan formal evaluation", "stage 6 pool/search",
    ],
}

# Which stage-6 pieces are explicitly *not* implemented here.
UNIMPLEMENTED_STAGE6: tuple[str, ...] = (
    "opponent pool", "value qualification", "search", "formal training gate",
)

PHASE5_DESIGN: dict[str, Any] = {
    "schema": PHASE5_DESIGN_SCHEMA,
    "stage": 5,
    "symmetry": {
        "schema": PHASE5_SYMMETRY_SCHEMA,
        "defaults": dict(PHASE5_SYMMETRY_DEFAULTS),
        "transforms": list(SYMMETRY_TRANSFORMS),
        "seat_rotation_rule": "seat -> (seat + k) mod 4",
        "copy_swap_rule": "card_id -> card_id + 54 (mod 108)",
        "action_permutation": "canonical action token sequence, card ids permuted, rank/suit/length/wild tokens unchanged",
        "comparison": (
            "KL(pi(.|s) || q_T(.|s)) with q_T(a|s) = pi(P_T(a)|T(s)), complete legal "
            "macro actions where enumerable, restricted-candidate next-token KL otherwise"
        ),
        "significance": dict(SYMMETRY_SIGNIFICANCE),
        "buckets": list(PHASE5_BUCKETS),
    },
    "belief": {
        "b0": json.loads(json.dumps(PHASE5_B0_PLAN)),
        "defaults": dict(PHASE5_BELIEF_DEFAULTS),
        "b1_condition": "only when B0 PASS",
        "head_schema": PHASE5_BELIEF_HEAD_SCHEMA,
        "probe_schema": PHASE5_BELIEF_PROBE_SCHEMA,
        "gate_schema": PHASE5_BELIEF_GATE_SCHEMA,
        "b0_result_schema": PHASE5_B0_RESULT_SCHEMA,
    },
    "budget": json.loads(json.dumps(PHASE5_BUDGET)),
    "unimplemented_stage6": list(UNIMPLEMENTED_STAGE6),
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def phase5_design_payload() -> dict[str, Any]:
    """Return a deep copy of the frozen design payload."""
    return json.loads(json.dumps(PHASE5_DESIGN))


def phase5_design_hash() -> str:
    return hashlib.sha256(_canonical(PHASE5_DESIGN)).hexdigest()


def phase5_design_identity() -> dict[str, Any]:
    return {
        "schema": PHASE5_DESIGN_SCHEMA,
        "hash": phase5_design_hash(),
        "design": phase5_design_payload(),
    }


__all__ = [
    "COUNT_CLASSES", "JOKER_COUNT_MAX", "MAX_COPIES_BY_RANK", "NUM_PRINTED_RANKS",
    "NUM_RELATIVE_SEATS", "PHASE5_B0_PLAN", "PHASE5_B0_RESULT_SCHEMA",
    "PHASE5_BELIEF_DATASET_SCHEMA", "PHASE5_BELIEF_DEFAULTS",
    "PHASE5_BELIEF_GATE_SCHEMA", "PHASE5_BELIEF_HEAD_SCHEMA",
    "PHASE5_BELIEF_PROBE_SCHEMA", "PHASE5_BUCKETS", "PHASE5_BUDGET",
    "PHASE5_DESIGN", "PHASE5_DESIGN_SCHEMA", "PHASE5_DIAGNOSTIC_SCHEMA",
    "PHASE5_PUBLIC_LEDGER_SCHEMA", "PHASE5_SYMMETRY_DEFAULTS",
    "PHASE5_SYMMETRY_SCHEMA", "PRINTED_RANK_LABELS", "PUBLIC_ESCROW_UNKNOWN_INDEX",
    "PUBLIC_FEATURE_WIDTH", "PUBLIC_KNOWN_SLICE", "PUBLIC_OWN_SLICE",
    "PUBLIC_REMAINING_SLICE", "PUBLIC_UNKNOWN_SLICE", "RELATIVE_SEAT_OFFSETS",
    "RELATIVE_SEATS", "SEAT_ROTATIONS", "SYMMETRY_SIGNIFICANCE",
    "SYMMETRY_TRANSFORMS", "TOTAL_PHYSICAL_CARDS", "UNIMPLEMENTED_STAGE6",
    "phase5_design_hash", "phase5_design_identity", "phase5_design_payload",
]
