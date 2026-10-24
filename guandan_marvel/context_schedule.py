# SPDX-License-Identifier: MIT
"""Versioned training-context schedules for MARVEL v14 phase 4.

A *context* is the tuple ``(level_rank, leader_seat, previous_result)`` that
parameterises one dealt round:

* 7A (no tribute) — thirteen levels ``3..15`` crossed with four leader seats,
  giving 52 contexts and a ``previous_result`` of ``None``.
* 7B (tribute) — the same grid with an explicit ``previous_result`` payload
  ``{ranking, winner_team, outcome_class, tribute_cancelled}``.

The context schedule is part of the *training-data manifest*: its ordered rows
and integer weights are hashed into ``training_data_manifest_hash``.  A
``previous_result`` payload is always validated through the rules object
``guandan.state.PreviousHandResult`` so an out-of-domain ranking/outcome is
rejected instead of silently dealt.

7B is currently **deferred**: the phase-4 tribute-observation audit found a
semantic collision in the actor observation representation (see
``TRIBUTE_OBSERVATION_AUDIT``).  A schedule that carries any tribute payload is
therefore rejected unless the audit is explicitly marked passed.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from guandan.cards import PRINTED_RANKS
from guandan.state import PreviousHandResult

CONTEXT_SCHEDULE_SCHEMA_V1 = "guandan-marvel-training-context-schedule-v1"
CONTEXT_SCHEDULE_SCHEMA_V2 = "guandan-marvel-training-context-schedule-v2"
CONTEXT_SCHEDULE_SCHEMAS = (CONTEXT_SCHEDULE_SCHEMA_V1, CONTEXT_SCHEDULE_SCHEMA_V2)
TRIBUTE_AUDIT_SCHEMA = "guandan-marvel-tribute-observation-audit-v1"

#: Level ranks accepted by a training context (PRINTED_RANKS are 3..Ace, plus 15==TWO).
DEFAULT_LEVEL_RANKS: tuple[int, ...] = tuple(int(rank) for rank in PRINTED_RANKS)
DEFAULT_LEADER_SEATS: tuple[int, ...] = (0, 1, 2, 3)


def _previous_payload(previous: PreviousHandResult | Mapping[str, object] | None) -> dict[str, object] | None:
    if previous is None:
        return None
    if isinstance(previous, PreviousHandResult):
        return {
            "ranking": list(previous.ranking),
            "winner_team": int(previous.winner_team),
            "outcome_class": str(previous.outcome_class),
            "tribute_cancelled": bool(previous.tribute_cancelled),
        }
    return {
        "ranking": [int(value) for value in previous["ranking"]],
        "winner_team": int(previous["winner_team"]),
        "outcome_class": str(previous["outcome_class"]),
        "tribute_cancelled": bool(previous.get("tribute_cancelled", False)),
    }


def previous_from_payload(payload: Mapping[str, object]) -> PreviousHandResult:
    """Validate an untrusted ``previous_result`` payload through the rules object."""
    if not isinstance(payload, Mapping):
        raise TypeError("previous_result payload must be a mapping")
    ranking = payload.get("ranking")
    if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
        raise ValueError("previous_result.ranking must be a sequence")
    return PreviousHandResult(
        tuple(int(value) for value in ranking),
        int(payload["winner_team"]),
        str(payload["outcome_class"]),
        bool(payload.get("tribute_cancelled", False)),
    )


def context_hash(context: Mapping[str, object]) -> str:
    """Canonical hash of a single context row (level/leader/previous_result)."""
    normalized = {
        "level_rank": int(context["level_rank"]),
        "leader_seat": int(context["leader_seat"]),
        "previous_result": None if context.get("previous_result") is None
        else _previous_payload(context["previous_result"]),
    }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_context_schedule(
    *,
    level_ranks: Sequence[int] = DEFAULT_LEVEL_RANKS,
    leader_seats: Sequence[int] = DEFAULT_LEADER_SEATS,
    previous_results: Sequence[Mapping[str, object] | None] | None = None,
    weight: int = 1,
) -> dict[str, object]:
    """Build an ordered, uniformly weighted ``v2`` context schedule.

    The traversal order is ``for leader in leader_seats: for level in level_ranks``
    so the row order is a deterministic function of the inputs (and therefore
    of ``training_data_manifest_hash``).  ``previous_results`` cycles together
    with the level/leader grid to produce honest coverage of each tribute mode.
    """
    levels = tuple(int(value) for value in level_ranks)
    leaders = tuple(int(value) for value in leader_seats)
    if not levels or not leaders:
        raise ValueError("level_ranks and leader_seats must be non-empty")
    payloads: tuple[Mapping[str, object] | None, ...]
    if previous_results is None:
        payloads = (None,)
    else:
        payloads = tuple(previous_results)
        if not payloads:
            raise ValueError("previous_results must be non-empty when provided")
    integer_weight = int(weight)
    if integer_weight <= 0:
        raise ValueError("context weight must be positive")
    contexts: list[dict[str, object]] = []
    index = 0
    for leader in leaders:
        for level in levels:
            contexts.append({
                "level_rank": level,
                "leader_seat": leader,
                "previous_result": None if payloads[index % len(payloads)] is None
                else _previous_payload(payloads[index % len(payloads)]),
                "weight": integer_weight,
            })
            index += 1
    schedule = {"schema": CONTEXT_SCHEDULE_SCHEMA_V2, "contexts": contexts}
    # A schedule may be *built* with tribute payloads (the schema is versioned to
    # support 7B); the payloads are still validated by the rules object here.
    # The 7B observation gate is enforced when the schedule is enabled
    # (config/Sampler call ``validate_context_schedule`` with the default gate).
    validate_context_schedule(schedule, allow_tribute=True)
    return schedule


def build_rank_leader_schedule(*, weight: int = 1) -> dict[str, object]:
    """The canonical 7A schedule: 13 levels x 4 leaders, no tribute, uniform."""
    return build_context_schedule(previous_results=None, weight=weight)


def _is_tribute_context(context: Mapping[str, object]) -> bool:
    return context.get("previous_result") is not None


def validate_context_schedule(
    schedule: Mapping[str, object], *, allow_tribute: bool | None = None,
) -> dict[str, object]:
    """Fail-closed structural *and* rules validation of a context schedule.

    ``allow_tribute`` defaults to the audit verdict: tribute payloads are only
    accepted when the tribute-observation audit has passed.
    """
    if not isinstance(schedule, Mapping):
        raise TypeError("training_context_schedule must be a mapping")
    schema = schedule.get("schema")
    if schema not in CONTEXT_SCHEDULE_SCHEMAS:
        raise ValueError(f"unsupported training_context_schedule schema {schema!r}")
    contexts = schedule.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("training_context_schedule.contexts must be a non-empty list")
    tribute_allowed = bool(TRIBUTE_OBSERVATION_AUDIT["passed"]) if allow_tribute is None else bool(allow_tribute)
    seen: set[str] = set()
    for index, context in enumerate(contexts):
        if not isinstance(context, Mapping):
            raise TypeError(f"training context {index} must be a mapping")
        level = context.get("level_rank")
        if not isinstance(level, int) or isinstance(level, bool) or int(level) not in {int(rank) for rank in PRINTED_RANKS}:
            raise ValueError(f"training context {index} has an invalid level_rank")
        leader = context.get("leader_seat")
        if not isinstance(leader, int) or isinstance(leader, bool) or int(leader) not in (0, 1, 2, 3):
            raise ValueError(f"training context {index} has an invalid leader_seat")
        payload = context.get("previous_result")
        if payload is not None:
            if not tribute_allowed:
                raise ValueError(
                    "tribute (7B) contexts are blocked: the tribute-observation audit "
                    f"has not passed ({TRIBUTE_OBSERVATION_AUDIT['reason']})"
                )
            previous_from_payload(payload)  # rules-object validation
        weight = context.get("weight", 1)
        if not isinstance(weight, int) or isinstance(weight, bool) or int(weight) <= 0:
            raise ValueError(f"training context {index} weight must be a positive integer")
        if schema == CONTEXT_SCHEDULE_SCHEMA_V2:
            digest = context_hash(context)
            if digest in seen:
                raise ValueError("v2 context schedule must not repeat a context row")
            seen.add(digest)
    return dict(schedule)


def expand_context_schedule(schedule: Mapping[str, object]) -> list[dict[str, object]]:
    """Expand weights into the ordered deterministic list the env consumes."""
    validate_context_schedule(schedule)
    expanded: list[dict[str, object]] = []
    for context in schedule["contexts"]:
        row = {
            "level_rank": int(context["level_rank"]),
            "leader_seat": int(context["leader_seat"]),
            "previous_result": context.get("previous_result"),
        }
        expanded.extend(dict(row) for _ in range(int(context.get("weight", 1))))
    return expanded


def schedule_coverage(schedule: Mapping[str, object]) -> dict[str, object]:
    """Coverage derived from the schedule rows themselves (never from a claim)."""
    validate_context_schedule(schedule)
    rows: list[Mapping[str, object]] = list(schedule["contexts"])
    modes = set()
    for row in rows:
        payload = row.get("previous_result")
        if payload is None:
            modes.add("none")
        elif bool(payload.get("tribute_cancelled", False)):
            modes.add("cancelled")
        else:
            modes.add(str(payload["outcome_class"]).lower())
    return {
        "level_ranks": sorted({int(row["level_rank"]) for row in rows}),
        "leader_seats": sorted({int(row["leader_seat"]) for row in rows}),
        "tribute_modes": sorted(modes),
        "context_count": len(rows),
        "weight_total": sum(int(row.get("weight", 1)) for row in rows),
    }


#: Phase-4 tribute-observation audit verdict (repair round).
#: The complete observation (event tokens + ``_encode_channels`` planes) is
#: produced by ``guandan_marvel.testing.tribute_audit.audit_tribute_observation``
#: using the real encoders and legal ``PreviousHandResult`` values.  No legal
#: exchange relation collides and no hidden card leaks, so 7B is enabled.  The
#: earlier pre-repair counterexample compared only the transfer token *bag* and
#: used an illegal seat structure; the bag coincidence it found is broken by
#: ``TRIBUTE_ORDER_SLICE``/``SELF_HAND_SLICE``/``REMAINING_COUNTS_SLICE``.
TRIBUTE_OBSERVATION_AUDIT: dict[str, Any] = {
    "schema": TRIBUTE_AUDIT_SCHEMA,
    "passed": True,
    "reason": "complete observation (tokens + channels) distinguishes every legal exchange relation; no hidden-card leak",
    "observation_public_info_complete": True,
    "hidden_card_leak": False,
    "legal_relations_checked": 7,
    "full_observation_collisions": [],
    "verified_by": "guandan_marvel.testing.tribute_audit.audit_tribute_observation",
    "verdict": "7B enabled; delivery still trains on 7A (no tribute contexts)",
}


CURRICULUM_SCHEMA = "guandan-marvel-context-curriculum-v1"


def validate_context_curriculum(spec: Mapping[str, object]) -> dict[str, object]:
    """Validate a versioned warm-start context curriculum.

    ``p_t = (1 - alpha_t) * p_old + alpha_t * p_target`` with
    ``alpha_t = clamp((t + iteration_offset - warmup_iterations) / blend_iterations, 0, 1)`` where
    ``t`` is the zero-based index of the iteration about to be trained.  The
    curriculum is therefore a pure function of ``t`` (never of wall-clock time)
    and is exactly reproducible across a save/resume cycle.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("training_context_curriculum must be a mapping")
    if spec.get("schema") != CURRICULUM_SCHEMA:
        raise ValueError(f"unsupported training_context_curriculum schema {spec.get('schema')!r}")
    old = spec.get("old_context")
    if not isinstance(old, Mapping):
        raise TypeError("curriculum.old_context must be a mapping")
    level = old.get("level_rank")
    if not isinstance(level, int) or isinstance(level, bool) or int(level) not in {int(rank) for rank in PRINTED_RANKS}:
        raise ValueError("curriculum.old_context.level_rank is invalid")
    leader = old.get("leader_seat")
    if not isinstance(leader, int) or isinstance(leader, bool) or int(leader) not in (0, 1, 2, 3):
        raise ValueError("curriculum.old_context.leader_seat is invalid")
    if old.get("previous_result") is not None:
        if not bool(TRIBUTE_OBSERVATION_AUDIT["passed"]):
            raise ValueError("a tribute old_context is blocked while 7B is deferred")
        previous_from_payload(old["previous_result"])
    warmup = spec.get("warmup_iterations", 20)
    blend = spec.get("blend_iterations", 80)
    resolution = spec.get("resolution", 1000)
    offset = spec.get("iteration_offset", 0)
    for name, value in (("warmup_iterations", warmup), ("blend_iterations", blend), ("resolution", resolution), ("iteration_offset", offset)):
        if not isinstance(value, int) or isinstance(value, bool) or int(value) < 0:
            raise ValueError(f"curriculum.{name} must be a nonnegative integer")
    if int(blend) <= 0 or int(resolution) <= 0:
        raise ValueError("curriculum blend_iterations and resolution must be positive")
    return dict(spec)


def curriculum_alpha(spec: Mapping[str, object], iteration: int) -> float:
    warmup = int(spec.get("warmup_iterations", 20))
    blend = int(spec.get("blend_iterations", 80))
    t = int(iteration) + int(spec.get("iteration_offset", 0))
    if t <= warmup:
        return 0.0
    if t >= warmup + blend:
        return 1.0
    return (t - warmup) / blend


def _smooth_weighted_round_robin(types: list[dict[str, object]], total: int) -> list[dict[str, object]]:
    """Deterministic smooth weighted round-robin over ``(context, weight)`` types.

    Every prefix of the emitted sequence tracks the target mixture to within one
    item, so a contiguous window of length ~collect_batch_size already realises
    ``p_t``.  Ties break on declaration order, making the sequence a pure
    function of the input.
    """
    weights = [int(item["weight"]) for item in types]
    if total <= 0 or sum(weights) != total:
        raise AssertionError("smooth round-robin requires weights summing to total")
    current = [0] * len(types)
    sequence: list[dict[str, object]] = []
    for _ in range(total):
        for index, weight in enumerate(weights):
            current[index] += weight
        best = 0
        for index in range(1, len(types)):
            if current[index] > current[best]:
                best = index
        sequence.append(dict(types[best]["context"]))
        current[best] -= total
    return sequence


def curriculum_expanded_contexts(
    spec: Mapping[str, object], schedule: Mapping[str, object], iteration: int,
) -> list[dict[str, object]]:
    """Deterministic length-``resolution`` expansion of the iteration's mixture.

    Integer allocation uses the largest-remainder rule and the ordering is a
    deterministic smooth weighted round-robin, so (a) the empirical mixture the
    environment consumes matches ``p_t`` to within one resolution unit and (b)
    any contiguous window of the sequence also approximates ``p_t``.  The result
    is a pure function of ``(curriculum, schedule, iteration)``.
    """
    validate_context_curriculum(spec)
    targets = schedule["contexts"]
    resolution = int(spec.get("resolution", 1000))
    alpha = curriculum_alpha(spec, iteration)
    old = {
        "level_rank": int(spec["old_context"]["level_rank"]),
        "leader_seat": int(spec["old_context"]["leader_seat"]),
        "previous_result": spec["old_context"].get("previous_result"),
    }
    old_slots = int(round((1.0 - alpha) * resolution))
    if alpha >= 1.0:
        old_slots = 0
    remaining = resolution - old_slots
    n = len(targets)
    base = remaining // n
    remainder = remaining - base * n
    types: list[dict[str, object]] = []
    if old_slots > 0:
        types.append({"context": dict(old), "weight": old_slots})
    for index, context in enumerate(targets):
        count = base + (1 if index < remainder else 0)
        if count <= 0:
            continue
        row = {
            "level_rank": int(context["level_rank"]),
            "leader_seat": int(context["leader_seat"]),
            "previous_result": context.get("previous_result"),
        }
        types.append({"context": row, "weight": count})
    expanded = _smooth_weighted_round_robin(types, resolution) if types else []
    if len(expanded) != resolution:
        raise AssertionError("curriculum expansion produced an unexpected length")
    return expanded


def curriculum_provenance(
    spec: Mapping[str, object] | None, schedule: Mapping[str, object] | None, iteration: int,
) -> dict[str, object]:
    if spec is None:
        return {"schema": None, "active": False}
    context_rows = None if schedule is None else [
        {"level_rank": int(row["level_rank"]), "leader_seat": int(row["leader_seat"]),
         "previous_result": row.get("previous_result")}
        for row in schedule["contexts"]
    ]
    body = {
        "schema": CURRICULUM_SCHEMA,
        "active": True,
        "iteration": int(iteration),
        **({"iteration_offset": int(spec["iteration_offset"])} if "iteration_offset" in spec else {}),
        "alpha": curriculum_alpha(spec, iteration),
        "warmup_iterations": int(spec.get("warmup_iterations", 20)),
        "blend_iterations": int(spec.get("blend_iterations", 80)),
        "resolution": int(spec.get("resolution", 1000)),
        "old_context": dict(spec["old_context"]),
        "target_contexts": context_rows,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**body, "hash": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "CONTEXT_SCHEDULE_SCHEMAS", "CONTEXT_SCHEDULE_SCHEMA_V1", "CONTEXT_SCHEDULE_SCHEMA_V2",
    "CURRICULUM_SCHEMA", "DEFAULT_LEADER_SEATS", "DEFAULT_LEVEL_RANKS", "TRIBUTE_AUDIT_SCHEMA",
    "TRIBUTE_OBSERVATION_AUDIT", "build_context_schedule", "build_rank_leader_schedule",
    "context_hash", "curriculum_alpha", "curriculum_expanded_contexts", "curriculum_provenance",
    "expand_context_schedule", "previous_from_payload", "schedule_coverage",
    "validate_context_curriculum", "validate_context_schedule",
]
