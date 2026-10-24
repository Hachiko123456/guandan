# SPDX-License-Identifier: MIT
"""Bounded phase-6 value-qualification planner/runner (default: plan only).

The plan is a true Cartesian allocation of the four pre-registered level ranks,
four leader seats and four tribute modes (4 x 4 x 4 = 64 joint cells, repeated
per replicate), so every marginal group reaches ``min_slice_deals`` within the
128-deal attempt cap.  ``--execute`` refuses to start any model, engine or
collection unless a real isolation manifest protects the training/development/
selection/final domains and the live producer/checkpoint/config identities bind
to the frozen pre-registration.

    D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase6_value_qualification.py --deals 128
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.evaluation.checkpoints import (  # noqa: E402
    EVALUATION_CHECKPOINT_SCHEMA, load_v2_policy,
)
from guandan_marvel.phase6.design import (  # noqa: E402
    VALUE_GATE_PLAN, phase6_design_hash,
)
from guandan_marvel.phase6.value_gate import (  # noqa: E402
    VALUE_RANKING_METRIC_VERSION, QualificationDeal, _tribute_mode,
    assert_qualification_isolation, build_qualification_deals,
    canonical_physical_deal, load_value_gate_registration,
    qualification_producer_source_hash, run_value_qualification,
)

QUALIFICATION_CONFIRMATION = "PHASE6_QUALIFICATION_ATTEMPT_CONFIRMED"
ISOLATION_MANIFEST_SCHEMA = "guandan-marvel-phase6-qualification-isolation-manifest-v1"
RANKING_BUDGET_SCHEMA = "guandan-marvel-phase6-ranking-budget-v1"
REQUIRED_PROTECTED_DOMAINS = ("training", "development", "selection", "final")
LEVEL_RANKS = (3, 7, 11, 15)
LEADER_SEATS = (0, 1, 2, 3)
PREVIOUS_RESULTS = (
    None,
    {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN", "tribute_cancelled": False},
    {"ranking": [2, 1, 4, 3], "winner_team": 1, "outcome_class": "HEAD_THIRD", "tribute_cancelled": False},
    {"ranking": [1, 2, 4, 3], "winner_team": 0, "outcome_class": "HEAD_LAST", "tribute_cancelled": True},
)
_TOP_LEVEL_MANIFEST_KEYS = frozenset({
    "schema", "created_at", "producer_source_hash", "design_hash", "domains",
    "checkpoint", "notes",
})
_DOMAIN_KEYS = frozenset({
    "seed_domains", "deals", "canonical_deal_hashes", "access_policy", "samples_path",
})


class QualificationPlanError(RuntimeError):
    """The qualification plan or its isolation evidence is not trustworthy."""


def _tribute_label(previous: Mapping[str, object] | None) -> str:
    return _tribute_mode(previous)


def build_cartesian_deals(
    count: int, *, base_seed: int = 2_000_000_000,
) -> tuple[QualificationDeal, ...]:
    """Cartesian 4 level x 4 leader x 4 tribute allocation (not a diagonal)."""
    if int(count) < 0:
        raise ValueError("count must be non-negative")
    cells = [
        (level, leader, tribute)
        for level in LEVEL_RANKS
        for leader in LEADER_SEATS
        for tribute in range(len(PREVIOUS_RESULTS))
    ]
    deals = []
    for index in range(int(count)):
        level, leader, tribute = cells[index % len(cells)]
        deals.append(QualificationDeal(
            deal_id=f"qualification-{index:06d}",
            seed=int(base_seed) + index,
            level_rank=int(level),
            leader_seat=int(leader),
            previous_result=PREVIOUS_RESULTS[tribute],
        ))
    for deal in deals:
        deal.game_config()
    return tuple(deals)


def _context_counts(deals: Sequence[QualificationDeal]) -> dict[str, dict[str, int]]:
    def counts(values):
        result: dict[str, int] = {}
        for value in values:
            result[str(value)] = result.get(str(value), 0) + 1
        return result

    levels = counts(deal.level_rank for deal in deals)
    leaders = counts(deal.leader_seat for deal in deals)
    tributes = counts(_tribute_label(deal.previous_result) for deal in deals)
    return {"level_rank": levels, "leader_seat": leaders, "tribute_mode": tributes}


def _joint_cells(deals: Sequence[QualificationDeal]) -> dict[str, int]:
    cells: dict[str, int] = {}
    for deal in deals:
        key = (
            f"level={deal.level_rank}|leader={deal.leader_seat}|"
            f"tribute={_tribute_label(deal.previous_result)}"
        )
        cells[key] = cells.get(key, 0) + 1
    return dict(sorted(cells.items()))


def ranking_budget(deals: int) -> dict[str, object]:
    """A non-zero, bounded ranking plan that covers mid and late."""
    counterfactual = VALUE_GATE_PLAN["counterfactual"]
    dataset = VALUE_GATE_PLAN["dataset"]
    minimum_slice = int(dataset["min_slice_deals"])
    max_points = min(
        int(counterfactual["max_points_per_deal"]) * int(deals),
        max(int(counterfactual["min_ranking_points"]), 2 * minimum_slice),
    )
    return {
        "schema": RANKING_BUDGET_SCHEMA,
        "metric_version": VALUE_RANKING_METRIC_VERSION,
        "planned_deals": int(deals),
        "max_ranking_points": int(max_points),
        "min_ranking_points": int(counterfactual["min_ranking_points"]),
        "min_ranking_deals": int(counterfactual["min_ranking_deals"]),
        "min_informative_mid_points": minimum_slice,
        "min_informative_late_points": minimum_slice,
        "phases_required": ["mid", "late"],
        "selection_order": ["mid", "late", "early"],
        "max_points_per_deal": int(counterfactual["max_points_per_deal"]),
        "max_candidates": int(counterfactual["max_candidates"]),
        "max_enumeration_nodes": int(counterfactual["max_enumeration_nodes"]),
        "max_continuation_steps": int(counterfactual["max_continuation_steps"]),
        "wall_budget": "the qualification wall_seconds cap is shared with collection; "
                       "ranking points never extend it",
        "deal_budget": "ranking starts only inside the bounded deal loop",
        "expected_informative_points": "observed per collected point; all-target-tie "
                                       "points are reported but never counted",
        "sufficiency_if_actual_below": "INCONCLUSIVE",
    }


# ---------------------------------------------------------------------------
# Isolation manifest
# ---------------------------------------------------------------------------
def load_isolation_manifest(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise QualificationPlanError(f"isolation manifest is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationPlanError(
            f"isolation manifest is not readable JSON: {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise QualificationPlanError("isolation manifest must be a JSON object")
    if payload.get("schema") != ISOLATION_MANIFEST_SCHEMA:
        raise QualificationPlanError("isolation manifest schema mismatch")
    unknown = sorted(set(payload) - _TOP_LEVEL_MANIFEST_KEYS)
    if unknown:
        raise QualificationPlanError(f"isolation manifest has unknown keys: {unknown}")
    return dict(payload)


def _normalized_deal_id(value: object) -> str:
    return " ".join(str(value).split()).lower()


def _parse_seed_domains(value: object, domain: str) -> list[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise QualificationPlanError(f"domain {domain!r} has no seed domains")
    ranges: list[tuple[int, int]] = []
    for row in value:
        if (
            not isinstance(row, (list, tuple)) or len(row) != 2
            or isinstance(row[0], bool) or isinstance(row[1], bool)
        ):
            raise QualificationPlanError(f"domain {domain!r} seed domain row is malformed")
        low, high = int(row[0]), int(row[1])
        if low > high:
            raise QualificationPlanError(f"domain {domain!r} seed domain is inverted")
        ranges.append((low, high))
    return ranges


def _canonical_hashes_for_domain(
    domain: str, payload: Mapping[str, object],
) -> tuple[set[str], list[dict[str, object]], list[int], list[str]]:
    """Return (canonical hashes, deal rows, explicit seeds, deal ids)."""
    hashes: set[str] = set()
    rows: list[dict[str, object]] = []
    seeds: list[int] = []
    deal_ids: list[str] = []
    provided = payload.get("canonical_deal_hashes")
    if provided is not None:
        if not isinstance(provided, list) or not provided:
            raise QualificationPlanError(
                f"domain {domain!r} canonical_deal_hashes must be a non-empty list"
            )
        for value in provided:
            text = str(value)
            if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text.lower()):
                raise QualificationPlanError(
                    f"domain {domain!r} has a malformed canonical deal hash"
                )
            if text.lower() in hashes:
                raise QualificationPlanError(
                    f"domain {domain!r} repeats a canonical deck (rotation/copy alias)"
                )
            hashes.add(text.lower())
    deals = payload.get("deals")
    if deals is not None:
        if not isinstance(deals, list) or not deals:
            raise QualificationPlanError(f"domain {domain!r} deals must be a non-empty list")
        computed: set[str] = set()
        for row in deals:
            if not isinstance(row, Mapping):
                raise QualificationPlanError(f"domain {domain!r} deal row is malformed")
            deal_id = _normalized_deal_id(row.get("deal_id", ""))
            if not deal_id:
                raise QualificationPlanError(f"domain {domain!r} deal row has no ID")
            if "seed" not in row:
                raise QualificationPlanError(f"domain {domain!r} deal row has no seed")
            seed = int(row["seed"])
            hands = row.get("hands")
            if hands is None:
                raise QualificationPlanError(
                    f"domain {domain!r} deal {deal_id!r} has no physical hands; a seed "
                    "alone cannot prove the entity deck is distinct"
                )
            canonical = canonical_physical_deal(hands)
            if canonical in computed:
                raise QualificationPlanError(
                    f"domain {domain!r} repeats canonical physical deal {canonical} "
                    "(rotation/swap/copy alias)"
                )
            computed.add(canonical)
            if hashes and canonical not in hashes:
                raise QualificationPlanError(
                    f"domain {domain!r} declared canonical hashes disagree with its hands"
                )
            rows.append({"deal_id": deal_id, "seed": seed, "domain": domain})
            seeds.append(seed)
            deal_ids.append(deal_id)
        if hashes and computed != hashes:
            raise QualificationPlanError(
                f"domain {domain!r} declared canonical hashes disagree with its hands"
            )
        hashes = computed
    if not hashes:
        raise QualificationPlanError(
            f"domain {domain!r} provides no canonical deal evidence; disjoint seeds alone "
            "do not prove the physical deals are distinct"
        )
    return hashes, rows, seeds, deal_ids


def validate_isolation_manifest(
    manifest: Mapping[str, object],
    *,
    deals: Sequence[QualificationDeal],
    expected_producer_source_hash: str,
    expected_design_hash: str,
) -> dict[str, object]:
    """Fail-closed validation of every protected domain before any model/engine."""
    if manifest.get("schema") != ISOLATION_MANIFEST_SCHEMA:
        raise QualificationPlanError("isolation manifest schema mismatch")
    if str(manifest.get("producer_source_hash")) != str(expected_producer_source_hash):
        raise QualificationPlanError(
            "isolation manifest producer source hash differs from the live qualification "
            "producer"
        )
    if str(manifest.get("design_hash")) != str(expected_design_hash):
        raise QualificationPlanError(
            "isolation manifest design hash differs from the frozen phase-6 design"
        )
    domains = manifest.get("domains")
    if not isinstance(domains, Mapping):
        raise QualificationPlanError("isolation manifest has no domains mapping")
    unknown = sorted(set(domains) - set(REQUIRED_PROTECTED_DOMAINS))
    missing = sorted(set(REQUIRED_PROTECTED_DOMAINS) - set(domains))
    if unknown:
        raise QualificationPlanError(f"isolation manifest has unknown domains: {unknown}")
    if missing:
        raise QualificationPlanError(f"isolation manifest is missing domains: {missing}")

    qualification_seeds = [int(deal.seed) for deal in deals]
    seed_ranges: dict[str, list[tuple[int, int]]] = {}
    all_rows: list[dict[str, object]] = []
    all_hashes: set[str] = set()
    all_ids: dict[str, str] = {}
    explicit_seed_domains: dict[int, str] = {}
    report_domains: dict[str, object] = {}
    for domain in REQUIRED_PROTECTED_DOMAINS:
        payload = domains[domain]
        if not isinstance(payload, Mapping):
            raise QualificationPlanError(f"domain {domain!r} must be a mapping")
        extra = sorted(set(payload) - _DOMAIN_KEYS)
        if extra:
            raise QualificationPlanError(
                f"domain {domain!r} has unknown keys: {extra}"
            )
        ranges = _parse_seed_domains(payload.get("seed_domains"), domain)
        seed_ranges[domain] = ranges
        if domain == "final":
            if payload.get("access_policy") != "frozen_protection_only":
                raise QualificationPlanError(
                    "final domain must be protected by the frozen seed-domain/hash policy "
                    "only; it may never be read or consumed"
                )
            # ``samples_path`` is intentionally never opened: final holdout data is
            # not read, generated or consumed by qualification.
        elif "access_policy" in payload:
            if payload.get("access_policy") != "protected":
                raise QualificationPlanError(
                    f"domain {domain!r} access policy is not 'protected'"
                )
        hashes, rows, seeds, deal_ids = _canonical_hashes_for_domain(domain, payload)
        for low, high in ranges:
            for seed in qualification_seeds:
                if low <= seed <= high:
                    raise QualificationPlanError(
                        f"qualification seed {seed} lies inside protected domain "
                        f"{domain!r} seed range [{low}, {high}]"
                    )
        for seed in seeds:
            if seed in explicit_seed_domains:
                raise QualificationPlanError(
                    f"protected seed {seed} is reused by {explicit_seed_domains[seed]!r} "
                    f"and {domain!r}"
                )
            explicit_seed_domains[seed] = domain
            if seed in qualification_seeds:
                raise QualificationPlanError(
                    f"qualification seed {seed} is protected by domain {domain!r}"
                )
        for deal_id in deal_ids:
            if deal_id in all_ids:
                raise QualificationPlanError(
                    f"deal ID {deal_id!r} is aliased across {all_ids[deal_id]!r} and "
                    f"{domain!r}"
                )
            all_ids[deal_id] = domain
        overlap = sorted(hashes & all_hashes)
        if overlap:
            raise QualificationPlanError(
                f"domain {domain!r} canonical deals overlap protected data: {overlap[:3]}"
            )
        all_hashes |= hashes
        all_rows.extend(rows)
        report_domains[domain] = {
            "seed_domains": [list(row) for row in ranges],
            "explicit_deal_seeds": len(seeds),
            "canonical_deal_hashes": len(hashes),
            "access_policy": (
                "frozen_protection_only" if domain == "final"
                else str(payload.get("access_policy", "protected"))
            ),
            "final_samples_read": False if domain == "final" else None,
        }
    # Cross-domain seed ranges must not overlap either.
    names = list(REQUIRED_PROTECTED_DOMAINS)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1:]:
            for low_a, high_a in seed_ranges[first]:
                for low_b, high_b in seed_ranges[second]:
                    if max(low_a, low_b) <= min(high_a, high_b):
                        raise QualificationPlanError(
                            f"protected seed domains {first!r} and {second!r} overlap; "
                            "training/development/selection/final must be disjoint"
                        )
    # Call the actual production isolation gate as well.
    gate_report = assert_qualification_isolation(
        deals,
        protected_rows=all_rows,
        protected_canonical_hashes=sorted(all_hashes),
        protected_seed_domains=explicit_seed_domains,
    )
    return {
        "schema": "guandan-marvel-phase6-qualification-isolation-report-v1",
        "status": "VALID",
        "domains": report_domains,
        "qualification_deals": len(deals),
        "protected_canonical_hashes": len(all_hashes),
        "protected_explicit_seeds": len(explicit_seed_domains),
        "seed_range_overlap": 0,
        "canonical_deal_overlap": 0,
        "production_gate": dict(gate_report),
        "final_samples_read": False,
    }


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
def build_plan(
    *, deals: int, wall_seconds: float, dry_run: bool,
    manifest_path: str | Path | None = None,
    producer_source_hash: str | None = None,
) -> dict[str, object]:
    if not 1 <= int(deals) <= 128:
        raise SystemExit("--deals must be within 1..128")
    if not 0 < float(wall_seconds) <= 1200:
        raise SystemExit("--wall-seconds must be >0 and <=1200")
    qualification_deals = build_cartesian_deals(int(deals))
    counts = _context_counts(qualification_deals)
    joint = _joint_cells(qualification_deals)
    min_slice = int(VALUE_GATE_PLAN["dataset"]["min_slice_deals"])
    coverage_ok = all(
        all(value >= min_slice for value in family.values())
        for family in counts.values()
    )
    isolation: dict[str, object] = {"status": "MISSING_ISOLATION_MANIFEST"}
    if manifest_path is not None:
        live = producer_source_hash or qualification_producer_source_hash()
        manifest = load_isolation_manifest(manifest_path)
        isolation = validate_isolation_manifest(
            manifest, deals=qualification_deals,
            expected_producer_source_hash=live,
            expected_design_hash=phase6_design_hash(),
        )
    status = "QUALIFICATION_ATTEMPT_NOT_RUN" if dry_run else (
        "PLANNED" if coverage_ok else "INSUFFICIENT_PLANNED_CONTEXT"
    )
    return {
        "schema": "guandan-marvel-phase6-qualification-plan-v2",
        "status": status,
        "dry_run": bool(dry_run),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planned_deals": int(deals),
        "max_deals": 128,
        "wall_seconds_cap": float(wall_seconds),
        "pilot_tool_limit": 32,
        "pilot_tool_gap": (
            "the pilot is capped at 32 deals and cycles 13 level ranks, so it cannot meet "
            "min_slice_deals=16 for its context groups"
        ),
        "context_plan": {
            "level_ranks": list(LEVEL_RANKS),
            "leader_seats": list(LEADER_SEATS),
            "previous_result_modes": len(PREVIOUS_RESULTS),
            "assignment": (
                "true Cartesian product 4 level ranks x 4 leader seats x 4 tribute modes "
                "(64 joint cells), repeated per replicate; not a single index mod 4"
            ),
            "joint_cell_count": len(joint),
            "expected_joint_cell_count": (
                len(LEVEL_RANKS) * len(LEADER_SEATS) * len(PREVIOUS_RESULTS)
            ),
        },
        "joint_context_cells": joint,
        "context_counts": counts,
        "min_slice_deals": min_slice,
        "planned_context_coverage_ok": coverage_ok,
        "ranking_budget": ranking_budget(int(deals)),
        "isolation": isolation,
        "would_execute": bool(
            coverage_ok and isolation.get("status") == "VALID" and not dry_run
        ),
    }


def execute_plan(
    *,
    plan: Mapping[str, object],
    deals: Sequence[QualificationDeal],
    manifest_path: str | Path,
    registration_path: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    device: str = "cuda",
    checkpoint_loader: Callable[..., object] | None = None,
    collector: Callable[..., object] | None = None,
    producer_hash_fn: Callable[[], str] = qualification_producer_source_hash,
) -> dict[str, object]:
    """Execute only after manifest, identity and budget validation.

    The ordering is contractual: registration -> manifest/isolation -> strict
    checkpoint load -> recognised checkpoint identity/config -> collection.  An
    invalid manifest therefore cannot start a model, engine or collector.
    """
    if not bool(plan.get("planned_context_coverage_ok")):
        raise QualificationPlanError(
            "planned context coverage cannot satisfy min_slice_deals"
        )
    budget = dict(plan.get("ranking_budget") or {})
    if int(budget.get("max_ranking_points", 0)) <= 0:
        raise QualificationPlanError("an execute plan requires a non-zero ranking budget")
    if "mid" not in list(budget.get("phases_required") or ()) or "late" not in list(
        budget.get("phases_required") or ()
    ):
        raise QualificationPlanError("the ranking budget must cover mid and late")
    registration = load_value_gate_registration(registration_path)
    live_producer = producer_hash_fn()
    manifest = load_isolation_manifest(manifest_path)
    isolation = validate_isolation_manifest(
        manifest, deals=deals,
        expected_producer_source_hash=live_producer,
        expected_design_hash=phase6_design_hash(),
    )
    if str(registration.get("source_hash") or "") != live_producer:
        raise QualificationPlanError(
            "the pre-registration was produced by a different qualification source; "
            "it may not be mixed with evidence produced by this source"
        )
    if not str(registration.get("config_hash") or ""):
        raise QualificationPlanError("the pre-registration has no resolved config hash")
    loader = checkpoint_loader or load_v2_policy
    artifact = loader(str(checkpoint_path), device=device, expected_algorithm="ippo")
    identity = dict(artifact.identity)
    if artifact.schema != EVALUATION_CHECKPOINT_SCHEMA:
        raise QualificationPlanError(
            "qualification requires an evaluation-only checkpoint "
            f"({EVALUATION_CHECKPOINT_SCHEMA}); got {artifact.schema!r}"
        )
    if artifact.backend != "cpp" or artifact.algorithm != "ippo":
        raise QualificationPlanError(
            "qualification requires a strict evaluation-only C++ ippo checkpoint"
        )
    if artifact.metadata.get("training_resume_allowed") is True:
        raise QualificationPlanError("qualification checkpoint advertises training resume")
    if str(identity.get("resolved_config_hash")) != str(registration.get("config_hash")):
        raise QualificationPlanError(
            "checkpoint resolved config does not match the pre-registered config"
        )
    recorded = manifest.get("checkpoint")
    if not isinstance(recorded, Mapping):
        raise QualificationPlanError(
            "isolation manifest has no checkpoint identity; it cannot be cross-verified"
        )
    for key in (
        "checkpoint_sha256", "resolved_config_hash", "architecture",
        "source_hash", "backend", "protocol_version",
    ):
        if str(recorded.get(key)) != str(identity.get(key)):
            raise QualificationPlanError(
                f"isolation manifest checkpoint identity disagrees on {key}"
            )
    collect = collector or run_value_qualification
    result = collect(
        checkpoint_path=str(checkpoint_path),
        registration_path=str(registration_path),
        deals=list(deals),
        device=device,
        max_deals=len(deals),
        wall_seconds=float(plan["wall_seconds_cap"]),
        max_ranking_points=int(budget["max_ranking_points"]),
        protected_canonical_hashes=_manifest_canonical_hashes(manifest),
    )
    informative = [
        point for point in result.ranking_points if bool(point.informative)
    ]
    actual = {
        "ranking_points": len(result.ranking_points),
        "informative_points": len(informative),
        "all_target_tie_points": sum(
            1 for point in result.ranking_points if point.all_target_tie
        ),
        "model_tie_points": sum(1 for point in result.ranking_points if point.model_tie),
        "informative_mid_points": sum(
            1 for point in informative if point.game_phase == "mid"
        ),
        "informative_late_points": sum(
            1 for point in informative if point.game_phase == "late"
        ),
        "sufficient_for_gate": (
            len(informative) >= int(budget["min_ranking_points"])
            and len({point.canonical_deal for point in informative})
            >= int(budget["min_ranking_deals"])
            and sum(1 for point in informative if point.game_phase == "mid")
            >= int(budget["min_informative_mid_points"])
            and sum(1 for point in informative if point.game_phase == "late")
            >= int(budget["min_informative_late_points"])
        ),
        "insufficient_verdict": "INCONCLUSIVE",
    }
    record = {
        "schema": "guandan-marvel-phase6-qualification-execution-v1",
        "status": "EXECUTED",
        "device": device,
        "isolation": isolation,
        "ranking_budget": budget,
        "ranking_actual": actual,
        "checkpoint": identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "execution.json").write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (destination / "result_metadata.json").write_text(
        json.dumps(result.metadata, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


def _manifest_canonical_hashes(manifest: Mapping[str, object]) -> list[str]:
    hashes: set[str] = set()
    domains = dict(manifest.get("domains") or {})
    for payload in domains.values():
        if not isinstance(payload, Mapping):
            continue
        for value in payload.get("canonical_deal_hashes") or ():
            hashes.add(str(value).lower())
        for row in payload.get("deals") or ():
            if isinstance(row, Mapping) and row.get("hands") is not None:
                hashes.add(canonical_physical_deal(row["hands"]))
    return sorted(hashes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=128)
    parser.add_argument("--wall-seconds", type=float, default=1200.0)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-qualification", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--registration", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out-dir", default="runs/phase6_closeout/qualification_plan")
    args = parser.parse_args()
    dry_run = not args.execute
    plan = build_plan(
        deals=args.deals, wall_seconds=args.wall_seconds, dry_run=dry_run,
        manifest_path=args.manifest,
    )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(
        json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0
    if args.confirm_qualification != QUALIFICATION_CONFIRMATION:
        raise SystemExit(
            "refusing a qualification attempt without --confirm-qualification "
            f"{QUALIFICATION_CONFIRMATION}"
        )
    if not plan["planned_context_coverage_ok"]:
        raise SystemExit("planned context coverage cannot satisfy min_slice_deals=16")
    if not args.manifest or not args.checkpoint or not args.registration:
        raise SystemExit(
            "--execute requires --manifest, --checkpoint and --registration"
        )
    record = execute_plan(
        plan=plan,
        deals=build_cartesian_deals(args.deals),
        manifest_path=args.manifest,
        registration_path=args.registration,
        checkpoint_path=args.checkpoint,
        out_dir=output,
        device="cuda",
    )
    print(json.dumps({
        "status": record["status"],
        "informative_ranking_points": record["ranking_actual"]["informative_points"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
