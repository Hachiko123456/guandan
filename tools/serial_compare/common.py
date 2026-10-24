# SPDX-License-Identifier: MIT
"""Shared contracts for the single-upload serial recipe comparison.

This module is dependency-light on purpose: it can be imported by the build
script, the offline verifier and the Kaggle driver before any model or CUDA
import happens.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "guandan-serial-compare-plan-v1"
DEAL_MANIFEST_SCHEMA = "guandan-serial-compare-deal-manifest-v1"
DEAL_GENERATOR_VERSION = "guandan-serial-compare-deal-seed-generator-v2"
RESULT_SCHEMA = "guandan-serial-compare-result-v1"
STATE_SCHEMA = "guandan-serial-compare-state-v1"
MARKER_SCHEMA = "guandan-serial-compare-phase-marker-v1"
BUDGET_SCHEMA = "guandan-serial-compare-budget-v1"

# Domain seed bases.  Deliberately disjoint from the project's reserved bases
# (training 100_000+, selection 1_000_000_000+, final holdout 1_500_000_000+).
DOMAIN_BASES = {
    # The *actual* environment seed used by both formal arms (see
    # ``_arm_common``: each DDP rank adds ``environment_rank_seed_stride``).
    "training": 570_700,
    "preflight": 1_600_000_000,
    "development": 1_610_000_000,
    "confirmation": 1_620_000_000,
}
DOMAIN_COUNTS = {"preflight": 8, "development": 32, "confirmation": 128}
RESERVED_BASES = {"selection": 1_000_000_000, "final": 1_500_000_000}

FABLEDAN = "NOT_RUN"
FINAL_HOLDOUT = "NOT_TOUCHED"
FORMAL_PRODUCTION_PROMOTION = False

CONFIRMATION_ARMS = ("candidate_vs_control",)
DEVELOPMENT_ARMS = (
    "candidate_vs_control",
    "control_vs_375",
    "candidate_vs_375",
    "control_vs_rule",
    "candidate_vs_rule",
    "control_vs_500",
    "candidate_vs_500",
)
UNIQUE_ASSIGNMENTS = (0, 1)

BUDGET_PLAN = {
    "total_seconds": 14400,
    "phases": {
        "dependencies_and_preflight": 1200,
        "train_control": 2400,
        "train_candidate": 2400,
        "evaluation_reserve": 6000,
        "finalize_reserve": 600,
    },
    "shared_slack": 1800,
    "max_new_iterations_per_arm": 256,
    "started_at": "bootstrap",
    "symmetric_reduction_rule": (
        "only before either arm has started formal training, by a published deterministic "
        "scale applied to both per-arm quotas; the resolved budget and subplan hash are frozen"
    ),
}

PHASE_ORDER: tuple[str, ...] = (
    "bootstrap/input_check",
    "preflight/control",
    "preflight/candidate",
    "freeze_execution_budget",
    "train/control",
    "train/candidate",
    "freeze_final_checkpoints",
    "evaluate/confirmation_primary",
    "evaluate/development_diagnostics",
    "summarize_and_package",
)

TRAINING_PHASES = ("train/control", "train/candidate")
EVALUATION_PHASES = ("evaluate/confirmation_primary", "evaluate/development_diagnostics")


class PlanError(ValueError):
    """Raised when a plan/manifest violates the frozen experiment contract."""


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path: str | Path, value: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    return target


def append_jsonl(path: str | Path, value: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_hash_record(value: Mapping[str, object], key: str = "hash") -> None:
    if value.get(key) != canonical({k: v for k, v in value.items() if k != key}):
        raise PlanError(f"invalid {key}: record is not self-consistent")


def safe_path(root: str | Path, name: str) -> Path:
    root_path = Path(root).resolve()
    if not name or any(character in name for character in ("\\", ":", "\0")):
        raise PlanError(f"unsafe member name {name!r}")
    target = (root_path / name).resolve()
    if target == root_path or not target.is_relative_to(root_path):
        raise PlanError(f"member {name!r} escapes its owned root")
    return target


# ---------------------------------------------------------------------------
# Deals
# ---------------------------------------------------------------------------
# Engine Rank values (THREE through ACE, then TWO), not face labels 2..14.
# Keep this contract independent of workspace/native/torch imports for offline
# verification. Tests compare it with guandan.cards and the real GameConfig.
ENGINE_LEVEL_RANKS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def _previous(index: int) -> dict[str, object] | None:
    """Cycle the four tribute contexts deterministically.

    ``None`` is "no previous hand"; the other three are the real terminal
    outcomes the engine accepts for a following deal.
    """
    mode = index % 4
    if mode == 0:
        return None
    if mode == 1:
        return {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN", "tribute_cancelled": True}
    if mode == 2:
        return {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN", "tribute_cancelled": False}
    return {"ranking": [1, 2, 3, 4], "winner_team": 0, "outcome_class": "HEAD_THIRD", "tribute_cancelled": False}


def _validate_previous_result(previous: object) -> None:
    """Check PreviousHandResult semantics without importing the runtime.

    Ranking maps seat -> finishing rank. Partners sit two seats apart; the
    rank-1 seat identifies the winning team and its partner's rank the outcome.
    Exhaustive tests reconcile this check with the real rules constructor.
    """
    if previous is None:
        return
    if not isinstance(previous, Mapping):
        raise PlanError("deal previous_result must be an object or null")
    ranking = previous.get("ranking")
    if (
        not isinstance(ranking, (list, tuple)) or len(ranking) != 4
        or any(type(rank) is not int for rank in ranking)
        or set(ranking) != {1, 2, 3, 4}
    ):
        raise PlanError("deal previous_result ranking must contain ranks 1..4 once per seat")
    winner = previous.get("winner_team")
    head_seat = ranking.index(1)
    if type(winner) is not int or winner not in (0, 1) or winner != head_seat % 2:
        raise PlanError("deal previous_result winner_team does not match the rank-1 seat")
    expected_outcome = {2: "DOUBLE_DOWN", 3: "HEAD_THIRD", 4: "HEAD_LAST"}[
        ranking[(head_seat + 2) % 4]
    ]
    if previous.get("outcome_class") != expected_outcome:
        raise PlanError("deal previous_result outcome_class does not match the ranking")
    if type(previous.get("tribute_cancelled", False)) is not bool:
        raise PlanError("deal previous_result tribute_cancelled must be boolean")


def build_deal_manifest(domain: str, *, count: int | None = None, seed_base: int | None = None) -> dict[str, object]:
    if domain not in ("preflight", "development", "confirmation"):
        raise PlanError(f"unsupported evaluation domain {domain!r}")
    base = int(DOMAIN_BASES[domain] if seed_base is None else seed_base)
    size = int(DOMAIN_COUNTS[domain] if count is None else count)
    if size <= 0:
        raise PlanError("deal count must be positive")
    if any(base <= reserved < base + size for reserved in RESERVED_BASES.values()):
        raise PlanError("evaluation seed range collides with a reserved project domain")
    rows = []
    for index in range(size):
        rows.append({
            "deal_id": f"{domain}-{index:04d}",
            "domain": domain,
            "seed": base + index,
            "level_rank": ENGINE_LEVEL_RANKS[index % len(ENGINE_LEVEL_RANKS)],
            "leader_seat": index % 4,
            "previous_result": _previous(index),
        })
    payload: dict[str, object] = {
        "schema": DEAL_MANIFEST_SCHEMA,
        "generator_version": DEAL_GENERATOR_VERSION,
        "domain": domain,
        "seed_base": base,
        "deal_count": size,
        "seed_semantics": "each row seed independently initializes one fixed evaluation deal",
        "deals": rows,
        "deal_ids": [row["deal_id"] for row in rows],
        "seeds": [row["seed"] for row in rows],
    }
    payload["hash"] = canonical(payload)
    return payload


def validate_deal_manifest(manifest: Mapping[str, object], *, expected_domain: str | None = None) -> list[dict[str, object]]:
    if not isinstance(manifest, Mapping) or manifest.get("schema") != DEAL_MANIFEST_SCHEMA:
        raise PlanError("unsupported deal manifest schema")
    verify_hash_record(manifest)
    if manifest.get("generator_version") != DEAL_GENERATOR_VERSION:
        raise PlanError("unsupported deal generator version; regenerate legacy deals")
    domain = str(manifest.get("domain"))
    if expected_domain is not None and domain != expected_domain:
        raise PlanError(f"deal manifest domain {domain!r} != expected {expected_domain!r}")
    rows = manifest.get("deals")
    if not isinstance(rows, list) or not rows:
        raise PlanError("deal manifest has no deals")
    if len(rows) != int(manifest.get("deal_count", -1)):
        raise PlanError("deal manifest count disagrees with its rows")
    if any(not isinstance(row, Mapping) for row in rows):
        raise PlanError("deal rows must be objects")
    ids = [str(row.get("deal_id")) for row in rows]
    seeds = [int(row.get("seed")) for row in rows]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise PlanError("deal identities/seeds must be unique")
    if ids != [str(value) for value in manifest.get("deal_ids", [])]:
        raise PlanError("deal id list disagrees with the rows")
    if seeds != [int(value) for value in manifest.get("seeds", [])]:
        raise PlanError("deal seed list disagrees with the rows")
    for row in rows:
        if str(row.get("domain")) != domain:
            raise PlanError("deal row domain mismatch")
        if type(row.get("level_rank")) is not int or row["level_rank"] not in ENGINE_LEVEL_RANKS:
            raise PlanError("deal level_rank is outside the engine printed rank range 3..15")
        if type(row.get("leader_seat")) is not int or row["leader_seat"] not in (0, 1, 2, 3):
            raise PlanError("deal leader_seat is invalid")
        _validate_previous_result(row.get("previous_result"))
    return [dict(row) for row in rows]


def assert_domains_disjoint(manifests: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Fail closed on any deal identity/seed overlap between domains."""
    seen_seeds: dict[int, str] = {}
    seen_ids: dict[str, str] = {}
    for domain, manifest in manifests.items():
        for row in validate_deal_manifest(manifest, expected_domain=domain):
            seed = int(row["seed"])
            deal_id = str(row["deal_id"])
            if seed in seen_seeds:
                raise PlanError(f"seed {seed} is shared by {seen_seeds[seed]!r} and {domain!r}")
            if deal_id in seen_ids:
                raise PlanError(f"deal id {deal_id!r} is shared by {seen_ids[deal_id]!r} and {domain!r}")
            for reserved, base in RESERVED_BASES.items():
                if base <= seed < base + 1_000_000:
                    raise PlanError(f"{domain!r} deal seed {seed} lies in the reserved {reserved!r} range")
            seen_seeds[seed] = domain
            seen_ids[deal_id] = domain
    return {
        "domains": sorted(manifests),
        "distinct_seeds": len(seen_seeds),
        "distinct_deal_ids": len(seen_ids),
        "disjoint": True,
    }


# ---------------------------------------------------------------------------
# Training/evaluation data isolation
# ---------------------------------------------------------------------------
def training_env_seeds(plan: Mapping[str, object], *, ranks: int = 2) -> set[int]:
    """The exact per-rank environment seeds the formal arms will sample with."""
    arms = dict(plan.get("arms") or {})
    seeds: set[int] = set()
    for arm in ("control", "candidate"):
        spec = dict(arms.get(arm) or {})
        base = int(spec.get("seed", DOMAIN_BASES["training"]))
        stride = int(spec.get("environment_rank_seed_stride", 1_000_003))
        for rank in range(int(ranks)):
            seeds.add(base + rank * stride)
    return seeds


def physical_deal_sha256(seed: int) -> str:
    """Physical 108-card deal identity produced by the engine for ``seed``.

    Deal identity depends only on the seed (level/leader/previous-hand do not
    change the hands), so this is the same hash the evaluation host records as
    ``base_deal_sha256``.  Imported lazily so the dependency-light driver can
    still be imported before native/CUDA.
    """
    from guandan_marvel import cpp_game
    from guandan_marvel.evaluation.manifests import deal_game_config

    config = deal_game_config({
        "deal_id": "isolation-probe",
        "seed": int(seed),
        "level_rank": 3,
        "leader_seat": 0,
        "previous_result": None,
    })
    env = cpp_game.EnvBatch(1, configs=[config], auto_reset=False, max_horizon=4)
    env.reset(int(seed))
    state = env.debug_state(0)["state"]
    hands = [sorted(int(card) for card in hand) for hand in state["hands"]]
    if len(hands) != 4 or any(len(hand) != 27 for hand in hands):
        raise PlanError("isolation probe did not produce a complete 27x4 deal")
    if sorted(card for hand in hands for card in hand) != list(range(108)):
        raise PlanError("isolation probe deal violates 108-card conservation")
    return canonical(hands)


def assert_training_seeds_disjoint(
    plan: Mapping[str, object],
    manifests: Mapping[str, Mapping[str, object]],
    *,
    ranks: int = 2,
    physical: bool = True,
) -> dict[str, object]:
    """Fail closed if a formal training namespace touches an evaluation namespace.

    Compares both seed *names* and, when ``physical`` is true, the actual
    engine-generated 108-card deal hashes (never only the seed label).
    """
    training_seeds = training_env_seeds(plan, ranks=ranks)
    eval_rows: list[tuple[str, int]] = []
    for domain, manifest in manifests.items():
        for row in validate_deal_manifest(manifest, expected_domain=domain):
            eval_rows.append((domain, int(row["seed"])))
    eval_seeds = {seed for _, seed in eval_rows}
    overlap = sorted(training_seeds & eval_seeds)
    if overlap:
        raise PlanError(f"training seeds overlap an evaluation namespace: {overlap}")

    payload: dict[str, object] = {
        "training_seeds": sorted(training_seeds),
        "preflight_seed_base": int(manifests["preflight"]["seed_base"]) if "preflight" in manifests else None,
        "evaluation_seed_bases": {
            domain: int(manifest["seed_base"]) for domain, manifest in manifests.items()
        },
        "seed_names_disjoint": True,
        "physical_deal_probe": "NOT_RUN",
    }
    if not physical:
        return payload

    training_hashes = {seed: physical_deal_sha256(seed) for seed in sorted(training_seeds)}
    eval_hashes: dict[str, str] = {}
    for domain, seed in eval_rows:
        eval_hashes[f"{domain}:{seed}"] = physical_deal_sha256(seed)
    collision = sorted(set(training_hashes.values()) & set(eval_hashes.values()))
    if collision:
        raise PlanError(
            "a formal training deal is physically identical to an evaluation deal: "
            f"{collision[:3]}"
        )
    if len(set(training_hashes.values())) != len(training_hashes):
        raise PlanError("the two ranks/training namespaces generated the same first deal")
    payload.update({
        "physical_deal_probe": "RUN",
        "training_deal_sha256": {str(seed): digest for seed, digest in training_hashes.items()},
        "evaluation_deal_sha256_count": len(eval_hashes),
        "physical_deal_hashes_disjoint": True,
    })
    return payload


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def _arm_common() -> dict[str, object]:
    return {
        "start_policy": "p375",
        "model": {
            "architecture": "v1.4-macro-fuse",
            "history_dim": 128, "feature_dim": 256, "decoder_layers": 4, "decoder_heads": 4,
            "state_layers": 3, "action_layers": 2, "fuse_layers": 2, "ffn_ratio": 2,
        },
        "ppo": {
            "training_semantics_profile": "macro_decision_v1",
            "policy_regularizer_mode": "macro_normalized_kl_uniform_v1",
            "base_lr": 1e-4, "base_clip_coef": 0.2, "base_reg_coef": 0.01,
            "agent_epoch": 4, "gae_lambda": 0.95, "macro_gae_lambda": None,
        },
        "rollout_batch_size": 48,
        "collect_batch_size": 40,
        "mini_batches": 8,
        "value_loss_coef": 0.5,
        "discount_gamma": 1.0,
        "seed": DOMAIN_BASES["training"],
        "environment_rank_seed_stride": 1_000_003,
        "amp_enabled": True,
        "trim_completed_padding": True,
        "max_new_iterations": 256,
        "curriculum_offset": 375,
        "reward_semantics_must_match_source": True,
        "context_schedule_must_match_source": True,
    }


def build_arms() -> dict[str, object]:
    control = {
        **_arm_common(),
        "name": "control",
        "ntp_weight": 0.02, "ntp_pos_weight": 10.0,
        "opponent_pool": {"enabled": False},
        "target_kl": None,
    }
    candidate = {
        **_arm_common(),
        "name": "candidate",
        "ntp_weight": 0.0, "ntp_pos_weight": 10.0,
        "opponent_pool": {
            "enabled": True,
            "entries": ["p250", "p375"],
            "weights": {"p250": 1.0, "p375": 1.0},
            "max_resident_policies": 2,
            "sampling_seed": 20261023,
            "sampling_scheme": "guandan-marvel-phase6-weighted-active-sampling-v1",
            "assignment_scheme": "team_rotation_v1",
            "weight_note": "engineering default 1:1; not claimed to be a theoretical optimum",
        },
        "target_kl": 0.015,
        "target_kl_stop": "global macro sample KL (sum/count across ranks) > target_kl, monitored before the pending minibatch update",
    }
    return {"control": control, "candidate": candidate}


def build_plan(*, source: Mapping[str, object], models: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    for name in ("p375", "p250", "p500"):
        if name not in models:
            raise PlanError(f"plan requires the {name} model identity")
    plan: dict[str, object] = {
        "schema": SCHEMA,
        "id": "single-upload-serial-recipe-comparison-20261023-v1",
        "test_only": False,
        "source": dict(source),
        "models": {name: dict(value) for name, value in models.items()},
        "arms": build_arms(),
        "budget": json.loads(json.dumps(BUDGET_PLAN)),
        "evaluation": {
            "deal_generator_version": DEAL_GENERATOR_VERSION,
            "protocol": "two_unique_team_assignments_v1",
            "assignments": list(UNIQUE_ASSIGNMENTS),
            "temperature": 0.0,
            "horizon": 1024,
            "confirmation": {"domain": "confirmation", "deals": 128, "games": 256, "arms": list(CONFIRMATION_ARMS)},
            "development": {"domain": "development", "deals": 32, "games": 448, "arms": list(DEVELOPMENT_ARMS)},
            "total_formal_games": 704,
            "statistics": {
                "method": "bounded-empirical-Bernstein-95",
                "cluster": "independent_deal",
                "primary": "candidate_final vs control_final team win rate",
                "verdict": {"supported": "low > 0.5", "worse": "high < 0.5", "otherwise": "INCONCLUSIVE"},
            },
            "secondary_500": "not in the training pool; unfiltered old-opponent probe",
            "secondary_375": "in the training pool; targeted benefit only, never the sole generalisation metric",
            "rule": "fixed heuristic baseline, not a FableDan substitute",
        },
        "isolation": {
            "domains": ["training", "preflight", "development", "confirmation"],
            "domain_seed_bases": dict(DOMAIN_BASES),
            "reserved_bases_not_touched": dict(RESERVED_BASES),
            "confirmation_never_feeds_training_or_selection": True,
        },
        "boundaries": {
            "fabledan": FABLEDAN,
            "final_holdout": FINAL_HOLDOUT,
            "formal_production_promotion": FORMAL_PRODUCTION_PROMOTION,
            "kaggle_execution": "NOT_RUN",
            "target_dual_t4_validation": "NOT_RUN",
        },
        "phases": list(PHASE_ORDER),
    }
    plan["plan_hash"] = canonical(plan)
    return plan


def validate_plan(plan: Mapping[str, object], *, testing: bool = False) -> Mapping[str, object]:
    verify_hash_record(plan, "plan_hash")
    if plan.get("schema") != SCHEMA:
        raise PlanError("unsupported serial-compare plan schema")
    if plan.get("test_only"):
        if not testing:
            raise PlanError("a test-only plan is forbidden in production")
        return plan
    arms = plan.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"control", "candidate"}:
        raise PlanError("plan must define exactly the control and candidate arms")
    control, candidate = arms["control"], arms["candidate"]
    for arm in (control, candidate):
        if arm.get("model") != build_arms()["control"]["model"]:
            raise PlanError("arm model architecture changed")
        if arm.get("start_policy") != "p375":
            raise PlanError("both arms must start from the same 375 policy")
        if int(arm.get("curriculum_offset", -1)) != 375:
            raise PlanError("curriculum logical origin changed")
        if int(arm.get("max_new_iterations", 0)) > 256:
            raise PlanError("per-arm new-iteration cap changed")
        if int(arm.get("seed", -1)) != DOMAIN_BASES["training"]:
            raise PlanError("the formal training seed must match the declared training namespace")
    if float(control.get("ppo", {}).get("base_lr")) != 1e-4 or float(control["ppo"]["base_clip_coef"]) != 0.2 or float(control["ppo"]["base_reg_coef"]) != 0.01:
        raise PlanError("learning recipe changed")
    if control.get("target_kl") is not None:
        raise PlanError("the control arm must keep target_kl=None")
    if float(candidate.get("target_kl")) != 0.015:
        raise PlanError("candidate target_kl must be the pre-registered 0.015")
    if bool(control.get("opponent_pool", {}).get("enabled")) or not bool(candidate.get("opponent_pool", {}).get("enabled")):
        raise PlanError("pool must be off for control and on for candidate")
    if float(candidate["ntp_weight"]) != 0.0 or float(control["ntp_weight"]) <= 0.0:
        raise PlanError("NTP must stay on for control and off for candidate")
    pool = candidate["opponent_pool"]
    if sorted(pool.get("entries", [])) != ["p250", "p375"]:
        raise PlanError("candidate pool entries changed")
    if int(pool.get("max_resident_policies", 0)) < len(pool.get("entries", [])):
        raise PlanError("candidate pool resident capacity is smaller than its entries")
    budget = plan.get("budget") or {}
    if int(budget.get("total_seconds", 0)) != 14400:
        raise PlanError("total budget changed")
    if int(budget.get("max_new_iterations_per_arm", 0)) > 256:
        raise PlanError("per-arm new-iteration cap changed")
    if sum(int(value) for value in budget.get("phases", {}).values()) + int(budget.get("shared_slack", 0)) != 14400:
        raise PlanError("budget phase allowance does not sum to the total budget")
    evaluation = plan.get("evaluation") or {}
    if evaluation.get("deal_generator_version") != DEAL_GENERATOR_VERSION:
        raise PlanError("plan evaluation generator version changed; rebuild the signed plan")
    if int(evaluation.get("confirmation", {}).get("games", 0)) != 256:
        raise PlanError("confirmation game count changed")
    if int(evaluation.get("development", {}).get("games", 0)) != 448:
        raise PlanError("development game count changed")
    if int(evaluation.get("total_formal_games", 0)) != 704:
        raise PlanError("total formal game count changed")
    boundaries = plan.get("boundaries") or {}
    if boundaries.get("fabledan") != FABLEDAN or boundaries.get("final_holdout") != FINAL_HOLDOUT:
        raise PlanError("evaluation boundary changed")
    if boundaries.get("formal_production_promotion") is not False:
        raise PlanError("formal_production_promotion must always be false")
    if tuple(plan.get("phases", ())) != PHASE_ORDER:
        raise PlanError("serial phase order changed")
    models = plan.get("models") or {}
    for name in ("p375", "p250", "p500"):
        entry = models.get(name)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
            raise PlanError(f"missing model identity {name!r}")
        if entry.get("training_resume_allowed") is not False:
            raise PlanError(f"{name!r} must be an evaluation-only policy")
    return plan


def bounded_cluster_ci(values: Sequence[float]) -> dict[str, object]:
    """Deal-clustered bounded empirical-Bernstein 95% interval.

    Kept local so the driver has no dependency on the training package, and
    written from the formula (not by calling the production helper) so tests can
    compare both against a hand computation.
    """
    sample = [float(value) for value in values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in sample):
        raise PlanError("cluster values must be finite and within [0,1]")
    if not sample:
        return {"method": "bounded-empirical-Bernstein-95", "clusters": 0, "mean": None,
                "low": None, "high": None, "interval_available": False}
    estimate = sum(sample) / len(sample)
    if len(sample) < 2:
        return {"method": "bounded-empirical-Bernstein-95", "clusters": 1, "mean": estimate,
                "low": None, "high": None, "interval_available": False}
    variance = sum((value - estimate) ** 2 for value in sample) / (len(sample) - 1)
    log_term = math.log(3.0 / 0.05)
    margin = math.sqrt(2.0 * variance * log_term / len(sample)) + 3.0 * log_term / len(sample)
    return {
        "method": "bounded-empirical-Bernstein-95",
        "confidence": 0.95,
        "clusters": len(sample),
        "mean": estimate,
        "sample_variance": variance,
        "low": max(0.0, estimate - margin),
        "high": min(1.0, estimate + margin),
        "interval_available": True,
        "note": "interval_available is NOT a claim of improvement",
    }


def formal_verdict(ci: Mapping[str, object], *, complete: bool) -> dict[str, object]:
    if not complete:
        return {"verdict": "EVALUATION_INCOMPLETE", "formal_production_promotion": False,
                "reason": "the pre-registered formal games are incomplete"}
    if not ci.get("interval_available"):
        return {"verdict": "INCONCLUSIVE", "formal_production_promotion": False,
                "reason": "fewer than two independent deal clusters"}
    low, high = float(ci["low"]), float(ci["high"])
    if low > 0.5:
        verdict, reason = "SUPPORTED", "cluster CI lower bound is above 0.5"
    elif high < 0.5:
        verdict, reason = "WORSE", "cluster CI upper bound is below 0.5"
    else:
        verdict, reason = "INCONCLUSIVE", "the cluster CI straddles 0.5"
    return {
        "verdict": verdict,
        "formal_production_promotion": False,
        "reason": reason,
        "note": "'continue investing in this recipe' is a recommendation, not production promotion",
    }


__all__ = [
    "BUDGET_PLAN", "BUDGET_SCHEMA", "CONFIRMATION_ARMS", "DEAL_GENERATOR_VERSION",
    "DEAL_MANIFEST_SCHEMA", "DEVELOPMENT_ARMS", "DOMAIN_BASES", "DOMAIN_COUNTS",
    "EVALUATION_PHASES", "FABLEDAN", "FINAL_HOLDOUT", "FORMAL_PRODUCTION_PROMOTION",
    "MARKER_SCHEMA", "PHASE_ORDER", "PlanError", "RESERVED_BASES", "RESULT_SCHEMA",
    "SCHEMA", "STATE_SCHEMA", "TRAINING_PHASES", "UNIQUE_ASSIGNMENTS",
    "append_jsonl", "assert_domains_disjoint", "assert_training_seeds_disjoint", "atomic_json",
    "bounded_cluster_ci", "physical_deal_sha256", "training_env_seeds",
    "build_arms", "build_deal_manifest", "build_plan", "canonical", "formal_verdict",
    "read_json", "safe_path", "sha", "validate_deal_manifest", "validate_plan",
    "verify_hash_record",
]
