# SPDX-License-Identifier: MIT
"""Frozen phase-6 contracts: opponent pool, value gate and training gates.

Everything in this module is *pre-registered*.  It was written down before any
phase-6 pool smoke, value-qualification pilot or gate run produced a result and
it must not be edited in response to an observed outcome.  The hash
:func:`phase6_design_hash` is recorded in every phase-6 artifact so a later
reader can prove the thresholds were not adapted to the data.

Nothing here enables a feature.  ``PHASE6_DEFAULTS`` is fully off: an opponent
pool, deployment search and the formal training gates all require an explicit
configuration value (and, for search, a PASS value gate).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PHASE6_DESIGN_SCHEMA = "guandan-marvel-phase6-design-v1"

# ---------------------------------------------------------------------------
# Opponent pool contract (06A section 2).
# ---------------------------------------------------------------------------
POOL_SCHEMA = "guandan-marvel-phase6-opponent-pool-v1"
POOL_STATE_SCHEMA = "guandan-marvel-phase6-opponent-pool-state-v1"
POOL_MANIFEST_SCHEMA = "guandan-marvel-phase6-opponent-manifest-v1"
POOL_ASSIGNMENT_SCHEMA = "guandan-marvel-phase6-seat-assignment-v1"
POOL_ROUTER_STATE_SCHEMA = "guandan-marvel-phase6-pool-router-state-v1"
POOL_SAMPLING_SCHEMA = "guandan-marvel-phase6-weighted-active-sampling-v1"
POOL_ENTRY_SCHEMA = "guandan-marvel-phase6-opponent-entry-v1"

# The only executable pool entry kind accepted in phase 6.  A frozen opponent
# must be a strict evaluation-only MARVEL V2 policy checkpoint; rule/random
# baselines live in the evaluation matrix, not in the training behaviour pool.
POOL_ENTRY_KINDS: tuple[str, ...] = ("v2_evaluation",)
POOL_ENTRY_TAGS: tuple[str, ...] = ("champion", "recent", "spaced")

CURRENT_POLICY_OWNER_ID = 0
NO_POLICY_OWNER_ID = -1

# Explicit field contract recorded per acting token.  ``policy_owner`` is the
# integer owner id (0 = trainable current policy, 1..N = frozen pool entry,
# -1 = chance/padding).  ``policy_train_mask`` is True only for current-owned
# acting tokens.  The actual behaviour probability remains the sampler's
# ``behavior_action_prob`` field (one per token, emitted by the policy that
# actually sampled the action).
POOL_FIELD_NAMES: tuple[str, ...] = (
    "policy_owner",
    "policy_train_mask",
    "opponent_checkpoint_id",
    "opponent_policy_version",
    "assignment_id",
    "assigned_seats",
)

POOL_ASSIGNMENT_PLAN: dict[str, Any] = {
    "schema": POOL_ASSIGNMENT_SCHEMA,
    "scheme": "team_rotation_v1",
    "unit": "episode",
    "current_seats": "exactly one team (two seats) is controlled by the current policy",
    "rotation": (
        "episode e of slot b uses physical current team (e+b) mod 2 and seat_rotation "
        "b mod 4 (a recorded identity field); the current seats are {team, team+2} so "
        "both teams and every physical seat alternate as current over consecutive episodes"
    ),
    "opponent_seats": "the two seats of the other team, all controlled by one sampled pool entry",
    "in_episode_switch_allowed": False,
    "all_opponent_episodes": False,
    "zero_current_episodes": False,
    "sampling_unit": "episode (per slot), before the episode's first token",
    "rng": "isolated torch.Generator(device='cpu') seeded by the pool sampling seed",
    "balance": (
        "over consecutive episodes every seat is current exactly half of the time and "
        "both teams appear as the current team equally often"
    ),
    "single_legal_action": "a one-action macro is a legal policy decision and keeps its owner mask",
}

POOL_PLAN: dict[str, Any] = {
    "schema": POOL_SCHEMA,
    "manifest_schema": POOL_MANIFEST_SCHEMA,
    "entry_schema": POOL_ENTRY_SCHEMA,
    "entry_kinds": list(POOL_ENTRY_KINDS),
    "entry_tags": list(POOL_ENTRY_TAGS),
    "field_names": list(POOL_FIELD_NAMES),
    "assignment": json.loads(json.dumps(POOL_ASSIGNMENT_PLAN)),
    "checkpoint_requirements": {
        "schema": "guandan-marvel-evaluation-checkpoint-v1",
        "strict_loader": "load_v2_policy + verify_policy_identity",
        "requires": [
            "unchanged checkpoint SHA256",
            "same architecture as the current policy",
            "same protocol version and C++ backend/build identity",
            "same model feature configuration (history/feature/head widths)",
            "finite, non-empty state dict loaded with strict=True",
        ],
        "rejected_before_trainer_state_change": True,
    },
    "frozen_semantics": {
        "model_mode": "eval",
        "requires_grad": False,
        "optimizer_membership": "excluded",
        "parameter_hash_checked_per_iteration": True,
        "resident_policy_limit": "configurable max_resident_policies; an opponent that is "
        "active in the current episode can never be evicted mid-episode",
    },
    "loss_semantics": {
        "policy_sample_unit": "current-owned complete macro decision",
        "value_sample_unit": "current-owned macro start (profile C)",
        "opponent_tokens": "timeline steps that advance the real engine, kept in the "
        "discount clock and reward windows, excluded from every current loss",
        "macro_owner": "one macro must have exactly one owner; mixed ownership fails closed",
        "ddp_weighting": "global current-macro count (count/sum/sumsq all-reduce), never a "
        "rank mean; a rank with no current macro emits a model-connected zero anchor",
    },
}

# Pre-registered, configurable sampling weights.  The default is uniform over
# active positive-weight entries; the manifest may carry explicit weights and
# every smoke configuration is listed before it is run.
POOL_SAMPLING_DEFAULTS: dict[str, Any] = {
    "scheme": POOL_SAMPLING_SCHEMA,
    "default_weight": 1.0,
    "zero_weight": "inactive for sampling",
    "no_entries": "fail closed: a pool without an active positive-weight entry cannot train",
}

# ---------------------------------------------------------------------------
# Value gate contract (06A section 3).  These thresholds are the
# pre-registration; the qualification runner may not change them.
# ---------------------------------------------------------------------------
VALUE_REGISTRATION_SCHEMA = "guandan-marvel-phase6-value-gate-registration-v1"
VALUE_GATE_SCHEMA = "guandan-marvel-phase6-value-gate-v1"
VALUE_DATASET_SCHEMA = "guandan-marvel-phase6-value-dataset-v1"
VALUE_SAMPLE_SCHEMA = "guandan-marvel-phase6-value-sample-v1"

VALUE_GATE_PLAN: dict[str, Any] = {
    "schema": VALUE_GATE_SCHEMA,
    "engine": "cpp",
    "device": "cuda",
    "checkpoint": {
        "schema": "guandan-marvel-evaluation-checkpoint-v1",
        "training_resume_allowed": False,
        "identity": "immutable SHA256 + architecture + protocol + backend identity + config hash",
        "distinct_checkpoints_required_for_pass": 2,
    },
    "target": {
        "perspective": "acting_player",
        "scale": "rank_point_over_3",
        "raw_reward_scale": 1.0 / 3.0,
        "discount": "gamma=1.0 from the macro start to the real terminal",
        "excludes": "no raw-reward mixing, no other-seat perspective, no filtered opponent clock",
    },
    "dataset": {
        "schema": VALUE_DATASET_SCHEMA,
        "domain": "qualification",
        "seed_base": 2_000_000_000,
        "pilot_deals": 16,
        "max_pilot_deals": 32,
        "max_qualification_attempt_deals": 128,
        "min_independent_deals": 64,
        "min_samples": 512,
        "min_slice_deals": 16,
        "wall_seconds_cap": 1200.0,
        "clustering": (
            "canonical physical deal: lexicographically minimal hand partition over the "
            "four seat rotations and the two deck swaps; level/leader/tribute context is "
            "excluded and repeated contexts of one physical deal form ONE cluster"
        ),
        "split_unit": "canonical deal; states from one deal are never treated as independent",
        "isolation": "qualification deals must not overlap training/selection/final seeds",
    },
    "metrics": {
        "primary": "EV = 1 - Var(target - prediction) / Var(target)",
        "require_target_variance": True,
        "zero_target_variance": "EV is undefined and reported explicitly; never epsilon-filled",
        "secondary": [
            "pearson", "spearman", "calibration_ece", "calibration_bins",
            "sample_count", "independent_deal_count", "zero_target_variance_count",
        ],
        "ev_min": 0.10,
        "ev_ci95_low_min": 0.02,
        "pearson_min": 0.25,
        "spearman_min": 0.20,
        "calibration_bins": 10,
        "max_calibration_ece": 0.10,
        "bootstrap": {
            "cluster": "canonical_deal_sha256",
            "resamples": 2000,
            "seed": 20261020,
            "confidence": 0.95,
            "undefined_resample_policy": "report undefined_resample_count; never impute",
        },
    },
    "slices": {
        "game_phase": [
            "early", "mid", "late",
        ],
        "game_phase_definition": (
            "early: step <= num_steps/3; mid: <= 2*num_steps/3; late: otherwise "
            "(num_steps is the real terminal length of the deal)"
        ),
        "outcome": ["DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST"],
        "seat_team": ["team_0", "team_1"],
        "context": ["level_rank", "leader_seat", "tribute_mode"],
        "max_seat_ev_gap_for_pass": 0.15,
        "max_seat_ranking_gap_for_pass": 0.15,
    },
    "counterfactual": {
        "unit": "complete legal macro action",
        "candidate_construction": (
            "DFS over the real C++ engine's legal next-token rows until COMMIT/PASS; "
            "only complete macros are candidates, never next-token top-k"
        ),
        "model_score": (
            "immediate current-player reward window (rank-point/3) + gamma^L * V_post for "
            "the same player, evaluated by one real model forward on the post-macro state"
        ),
        "realized_return": (
            "branch the real engine through the candidate and continue with the fixed "
            "evaluation policy under a pre-registered common continuation seed; the model's "
            "own value is never used as ground truth"
        ),
        "max_candidates": 8,
        "max_enumeration_nodes": 64,
        "max_continuation_steps": 512,
        "min_ranking_points": 64,
        "min_ranking_deals": 16,
        "random_baseline": "uniform over the enumerated complete macro candidates (1/k)",
        "min_effect": 0.05,
        "ranking_ci95_low_min": 0.0,
        "max_points_per_deal": 2,
    },
    "verdict_rules": {
        "PASS": (
            ">= min_independent_deals, >= min_samples, >= 2 distinct evaluation checkpoints, "
            "EV point estimate >= ev_min AND EV clustered CI lower >= ev_ci95_low_min, "
            "Pearson/Spearman thresholds met, calibration ECE <= max, >= min_ranking_points "
            "ranking points with paired delta CI lower > 0, and no severe seat/context bias"
        ),
        "REJECT": (
            "sufficient evidence AND (EV clustered CI upper < ev_min OR ranking delta CI "
            "upper <= 0), i.e. the checkpoint is demonstrably not useful for search"
        ),
        "INCONCLUSIVE": (
            "insufficient independent deals/samples/checkpoints/ranking points, undefined "
            "target variance, straddling CI, budget exhaustion or coverage gaps; the search "
            "must not be built from this evidence"
        ),
        "no_threshold_editing": True,
        "no_automatic_budget_increase": True,
    },
}

SEARCH_STATUS_IMPLEMENTED = "IMPLEMENTED"
SEARCH_STATUS_SKIPPED_VALUE_REJECTED = "SKIPPED_VALUE_REJECTED"
SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE = "NOT_RUN_VALUE_INCONCLUSIVE"
# A value-gate PASS only establishes VALUE_QUALIFIED.  As long as this phase
# does not implement deployment search, no artifact may ever report
# ``search_status=IMPLEMENTED``; the qualification result is recorded
# separately from the (still closed) search entry point.
SEARCH_STATUS_VALUE_QUALIFIED_SEARCH_NOT_IMPLEMENTED = (
    "VALUE_QUALIFIED_SEARCH_NOT_IMPLEMENTED"
)
SEARCH_IMPLEMENTED = False  # phase 6 does not build a search path without a PASS gate


def search_status_for_verdict(verdict: str) -> str:
    """Map a value-gate verdict to the only allowed search status.

    ``SEARCH_IMPLEMENTED`` is returned for PASS only when a search path actually
    exists in this source tree.  Otherwise a PASS artifact records
    ``VALUE_QUALIFIED_SEARCH_NOT_IMPLEMENTED`` and the search entry point keeps
    refusing.
    """
    if verdict == "PASS":
        return (
            SEARCH_STATUS_IMPLEMENTED
            if SEARCH_IMPLEMENTED
            else SEARCH_STATUS_VALUE_QUALIFIED_SEARCH_NOT_IMPLEMENTED
        )
    if verdict == "REJECT":
        return SEARCH_STATUS_SKIPPED_VALUE_REJECTED
    if verdict == "INCONCLUSIVE":
        return SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE
    raise ValueError(f"unknown value gate verdict {verdict!r}")


# ---------------------------------------------------------------------------
# Formal training gate tools (06A section 5).  Implemented, never executed by
# phase 6: 50/250/500/1000/1500 are absolute iteration targets.
# ---------------------------------------------------------------------------
FORMAL_GATE_TARGETS: dict[int, int] = {
    1: 50,
    2: 250,
    3: 500,
    4: 1000,
    5: 1500,
}
FORMAL_GATE_GAMES_AT_20_PER_ITER: dict[int, int] = {
    1: 1000,
    2: 5000,
    3: 10000,
    4: 20000,
    5: 30000,
}
TRAINING_GATE_TOOLS_SCHEMA = "guandan-marvel-phase6-training-gate-tools-v1"
GATE_PLAN_SCHEMA = "guandan-marvel-phase6-training-gate-plan-v1"
GATE_RUN_SCHEMA = "guandan-marvel-phase6-training-gate-run-v1"
GATE_EVIDENCE_SCHEMA = "guandan-marvel-phase6-training-gate-evidence-v1"
FORMAL_CONFIRMATION_TOKEN = "PHASE6_FORMAL_GATE_CONFIRMED"

TRAINING_GATE_PLAN: dict[str, Any] = {
    "schema": TRAINING_GATE_TOOLS_SCHEMA,
    "absolute_iteration_targets": dict(FORMAL_GATE_TARGETS),
    "games_at_20_per_iteration_is_an_estimate_only": dict(FORMAL_GATE_GAMES_AT_20_PER_ITER),
    "budget_unit": "completed_games (authoritative) and absolute iterations",
    "remaining": "remaining = max(0, absolute_target - actual_counter); resume never adds a new target",
    "dry_run_default": True,
    "requires_explicit_formal_confirmation": FORMAL_CONFIRMATION_TOKEN,
    "hard_limits": ["max_iterations", "max_games", "max_wall_seconds"],
    "iteration_boundary_checkpoint": (
        "after every completed iteration (never mid-iteration unless the trainer itself "
        "reports a durable boundary), with an output record that a resume can consume"
    ),
    "health": {
        "require_finite_loss": True,
        "require_finite_gradient": True,
        "max_nonfinite_count": 0,
        "max_amp_skipped_fraction": 0.05,
        "require_value_ev_defined": True,
        "min_value_ev": 0.0,
        "requires_legal_action_check": True,
        "requires_rule_random_smoke": True,
    },
    "strength": {
        "primary": "paired new-vs-M09 delta on fixed development deals",
        "cluster": "canonical deal",
        "ci": 0.95,
        "decision": {
            "continue": "clustered CI lower > 0 AND health and recovery pass",
            "stop": "clustered CI upper < 0",
            "inconclusive": "CI straddles 0: report a recommendation, never auto-add seeds/deals",
        },
        "auto_expand": False,
        "fixed_deals_required": True,
    },
    "smoke_only": "1-2 iterations may be used to test the tool boundaries and must be labelled smoke",
}

PHASE6_BUDGET: dict[str, Any] = {
    "pool_smoke": {
        "batch_size": 1,
        "iterations_per_configuration": 2,
        "max_pre_registered_configurations": 3,
        "device": "cuda",
    },
    "value_qualification": {
        "max_pilot_deals": 32,
        "max_attempt_deals": 128,
        "wall_seconds_cap": 1200.0,
    },
    "search_smoke_if_pass": {
        "max_deals": 16,
        "note": "only if a PASS value gate already exists; not run in phase 6",
    },
    "forbidden_without_new_authorization": [
        "formal gate 1-5 (50/250/500/1000/1500 iterations)",
        "Kaggle",
        "dual T4 / NCCL",
        "FableDan formal evaluation",
        "final holdout",
    ],
}

PHASE6_DEFAULTS: dict[str, Any] = {
    "opponent_pool_enabled": False,
    "search_enabled": False,
    "training_gate_dry_run": True,
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_hash(value: object) -> str:
    """SHA256 of a canonical JSON payload (local, dependency-free)."""
    return hashlib.sha256(_canonical(value)).hexdigest()


PHASE6_DESIGN: dict[str, Any] = {
    "schema": PHASE6_DESIGN_SCHEMA,
    "stage": 6,
    "defaults": dict(PHASE6_DEFAULTS),
    "pool": json.loads(json.dumps(POOL_PLAN)),
    "pool_sampling": json.loads(json.dumps(POOL_SAMPLING_DEFAULTS)),
    "value_gate": json.loads(json.dumps(VALUE_GATE_PLAN)),
    "search": {
        "implemented": SEARCH_IMPLEMENTED,
        "requires": "VALUE_GATE PASS",
        "statuses": [
            SEARCH_STATUS_IMPLEMENTED,
            SEARCH_STATUS_SKIPPED_VALUE_REJECTED,
            SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE,
        ],
        "when_disabled": "search_enabled=false adds zero forward passes, no RNG draws, no latency",
    },
    "training_gates": json.loads(json.dumps(TRAINING_GATE_PLAN)),
    "budget": json.loads(json.dumps(PHASE6_BUDGET)),
    "isolation": {
        "domains": ["training", "development", "selection", "final", "qualification"],
        "qualification_seed_base": 2_000_000_000,
        "fabledan": "external evaluation opponent only; never distillation/labels/training data",
        "final_holdout": "not run in phase 6; frozen candidate/threshold/CI first",
    },
}


def phase6_design_payload() -> dict[str, Any]:
    """Return a deep copy of the frozen design payload."""
    return json.loads(json.dumps(PHASE6_DESIGN))


def phase6_design_hash() -> str:
    return canonical_hash(PHASE6_DESIGN)


def phase6_design_identity() -> dict[str, Any]:
    return {
        "schema": PHASE6_DESIGN_SCHEMA,
        "hash": phase6_design_hash(),
        "design": phase6_design_payload(),
    }


__all__ = [
    "CURRENT_POLICY_OWNER_ID", "FORMAL_CONFIRMATION_TOKEN",
    "FORMAL_GATE_GAMES_AT_20_PER_ITER", "FORMAL_GATE_TARGETS",
    "GATE_EVIDENCE_SCHEMA", "GATE_PLAN_SCHEMA", "GATE_RUN_SCHEMA",
    "NO_POLICY_OWNER_ID", "PHASE6_BUDGET", "PHASE6_DEFAULTS", "PHASE6_DESIGN",
    "PHASE6_DESIGN_SCHEMA", "POOL_ASSIGNMENT_PLAN", "POOL_ASSIGNMENT_SCHEMA",
    "POOL_ENTRY_KINDS", "POOL_ENTRY_SCHEMA", "POOL_ENTRY_TAGS", "POOL_FIELD_NAMES",
    "POOL_MANIFEST_SCHEMA", "POOL_PLAN", "POOL_ROUTER_STATE_SCHEMA",
    "POOL_SAMPLING_DEFAULTS", "POOL_SAMPLING_SCHEMA", "POOL_SCHEMA",
    "POOL_STATE_SCHEMA", "SEARCH_IMPLEMENTED", "SEARCH_STATUS_IMPLEMENTED",
    "SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE", "SEARCH_STATUS_SKIPPED_VALUE_REJECTED",
    "SEARCH_STATUS_VALUE_QUALIFIED_SEARCH_NOT_IMPLEMENTED",
    "TRAINING_GATE_PLAN", "TRAINING_GATE_TOOLS_SCHEMA", "VALUE_DATASET_SCHEMA",
    "VALUE_GATE_PLAN", "VALUE_GATE_SCHEMA", "VALUE_REGISTRATION_SCHEMA",
    "VALUE_SAMPLE_SCHEMA", "canonical_hash", "phase6_design_hash",
    "phase6_design_identity", "phase6_design_payload", "search_status_for_verdict",
]
