# SPDX-License-Identifier: MIT
"""Phase-6 value qualification and pre-registered value gate.

The tool measures, on real C++ engine trajectories and an evaluation-only
checkpoint, whether the value head is a useful ranking signal.  It can only
produce a PASS from evidence that satisfies the thresholds frozen in
:mod:`guandan_marvel.phase6.design`; a small local pilot is expected to be
INCONCLUSIVE, and that verdict forbids building the conditional search.

Dataset contract:

- every sample comes from a real C++ game played by the frozen checkpoint;
- samples are clustered by *canonical physical deal* (seat rotations and deck
  swaps are identified), never by step;
- the target is the same player's realized future reward in the training scale
  (rank point / 3) with ``gamma = 1`` to the real terminal;
- zero target variance is reported as undefined EV, never epsilon-filled;
- counterfactual macro rankings use real engine branches and realized
  continuation returns with a common continuation seed, never the model's own
  value as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from guandan.environment_types import GameConfig
from guandan.state import PreviousHandResult
from guandan_marvel import cpp_game
from guandan_marvel.evaluation.agents import V2PolicySession
from guandan_marvel.evaluation.checkpoints import (
    EVALUATION_CHECKPOINT_SCHEMA, V2PolicyArtifact, load_v2_policy,
    verify_policy_identity,
)
from guandan_marvel.reward import TRAINING_REWARD_SCALE

from .design import (
    PHASE6_DESIGN_SCHEMA, SEARCH_IMPLEMENTED,
    SEARCH_STATUS_IMPLEMENTED, SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE,
    SEARCH_STATUS_SKIPPED_VALUE_REJECTED, VALUE_DATASET_SCHEMA, VALUE_GATE_PLAN,
    VALUE_GATE_SCHEMA, VALUE_REGISTRATION_SCHEMA, VALUE_SAMPLE_SCHEMA,
    canonical_hash, phase6_design_hash, phase6_design_payload,
    search_status_for_verdict,
)

VALUE_QUALIFICATION_DATA_SCHEMA = "guandan-marvel-phase6-value-qualification-data-v1"
VALUE_RANKING_POINT_SCHEMA = "guandan-marvel-phase6-value-ranking-point-v3"
VALUE_RANKING_LEGACY_POINT_SCHEMAS = ("guandan-marvel-phase6-value-ranking-point-v2",)
ACTION_STAGE_BASE = 263

# ---------------------------------------------------------------------------
# Versioned tie-aware ranking contract (06B section 3).
#
# The first implementation used ``max((value, -index))`` to pick a single
# realized "winner" and compared it with the model's argmax while keeping a
# ``1/k`` random baseline.  A deal whose realized returns were all equal (for
# example two candidates both returning -1.0) therefore produced accuracy=1
# and delta=1-1/k from nothing but the candidate order.  The contract below is
# fixed *before* looking at any new result and is shared by collection,
# serialization, offline re-evaluation and :func:`evaluate_value_gate`.
# ---------------------------------------------------------------------------
VALUE_RANKING_METRIC_VERSION = "guandan-marvel-phase6-tie-aware-ranking-v1"

# Realized returns are sums of the discrete terminal rank-point/3 rewards, so a
# tie is an exact-value tie; 1e-9 only absorbs float64 summation noise.  Model
# scores are continuous one-step-reward + value-head outputs, so a tie is a
# numerical near-tie on the float32 value head; 1e-6 is fixed here and may not
# be widened after seeing a verdict.
REALIZED_RETURN_TIE_ATOL = 1e-9
MODEL_SCORE_TIE_ATOL = 1e-6

VALUE_RANKING_METRIC_SCHEMA: dict[str, object] = {
    "schema": "guandan-marvel-phase6-ranking-metric-v1",
    "version": VALUE_RANKING_METRIC_VERSION,
    "tie_definition": {
        "realized_return": {
            "kind": "discrete_exact_tie",
            "atol": REALIZED_RETURN_TIE_ATOL,
            "rtol": 0.0,
            "dtype": "float64",
            "rationale": (
                "realized returns are gamma=1 sums of the discrete terminal rank-point/3 "
                "reward; equal logical returns compare equal up to float64 summation noise"
            ),
        },
        "model_score": {
            "kind": "continuous_near_tie",
            "atol": MODEL_SCORE_TIE_ATOL,
            "rtol": 0.0,
            "dtype": "float64",
            "rationale": (
                "model scores are immediate rank-point/3 reward + a float32 value-head output; "
                "a near-tie is a tie in the ranking the model can actually express"
            ),
        },
    },
    "candidate_count": "k = finite, complete, legal macro candidates (k >= 2 to rank)",
    "target_top_set": "T = candidates whose realized return reaches the maximum (tie-aware)",
    "model_top_set": "M = candidates whose model score reaches the maximum (tie-aware)",
    "accuracy": "|M intersect T| / |M|",
    "random_baseline": "|T| / k (uniform candidate strategy)",
    "delta": "accuracy - random_baseline",
    "informative": "k >= 2 and not all-target-tie; all-target-tie points never fill a threshold",
    "model_all_tie": "accuracy == random_baseline and delta == 0; no index-preference advantage",
    "all_target_tie": "accuracy = 1, random_baseline = 1, delta = 0, informative = false",
    "single_candidate": "k = 1 is not comparable: informative = false, delta = 0",
    "ordering": "every quantity is invariant to candidate permutation",
}

VALUE_QUALIFICATION_PRODUCER_SCHEMA = "guandan-marvel-phase6-qualification-producer-v1"
QUALIFICATION_PRODUCER_SOURCE_FILES: tuple[str, ...] = (
    "guandan_marvel/phase6/design.py",
    "guandan_marvel/phase6/value_gate.py",
    "scripts/run_phase6_value_qualification.py",
    "scripts/run_phase6_value_pilot.py",
    "scripts/run_phase6_reevaluate_gate.py",
)
VALUE_TARGET_CONTRACT: dict[str, object] = {
    "perspective": "acting_player",
    "scale": "rank_point_over_3",
    "discount_gamma": 1.0,
    "raw_reward_scale": TRAINING_REWARD_SCALE,
}
_TARGET_CONTRACT_KEYS = frozenset(VALUE_TARGET_CONTRACT)


class ValueGateError(RuntimeError):
    """The value gate contract was violated; the search must not be enabled."""


def qualification_producer_source_hash(root: str | Path | None = None) -> str:
    """SHA256 over the phase-6 qualification producer files only.

    This is deliberately *not* the training source fingerprint: it identifies
    the code that produced qualification evidence (gate/metrics/planner), while
    a checkpoint's ``source_hash`` identifies the code that trained it.  The two
    have different semantics and are never forced to be equal.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in QUALIFICATION_PRODUCER_SOURCE_FILES:
        path = base / relative
        if not path.is_file():
            raise ValueGateError(f"qualification producer source file is missing: {relative}")
        data = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------
_REGISTRATION_KEYS = frozenset({
    "schema", "design_schema", "design_hash", "value_gate_plan", "source_hash",
    "config_hash", "created_at", "state", "registration_hash",
})


def write_value_gate_registration(
    path: str | Path,
    *,
    source_hash: str = "",
    config_hash: str = "",
    overwrite: bool = False,
) -> dict[str, object]:
    """Write the pre-registration before any qualification sample exists."""
    destination = Path(path)
    payload: dict[str, object] = {
        "schema": VALUE_REGISTRATION_SCHEMA,
        "design_schema": PHASE6_DESIGN_SCHEMA,
        "design_hash": phase6_design_hash(),
        "value_gate_plan": json.loads(json.dumps(phase6_design_payload()["value_gate"])),
        "source_hash": str(source_hash),
        "config_hash": str(config_hash),
        "created_at": _timestamp(),
        "state": "registered_before_any_qualification_result",
    }
    payload["registration_hash"] = canonical_hash(payload)
    if destination.exists() and not overwrite:
        existing = load_value_gate_registration(destination)
        for key in ("design_hash", "source_hash", "config_hash", "value_gate_plan"):
            if existing.get(key) != payload.get(key):
                raise FileExistsError(
                    "a different value-gate registration already exists; thresholds may not be "
                    "edited after seeing results"
                )
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{id(payload)}")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return payload


def load_value_gate_registration(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise ValueGateError(f"value-gate registration is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueGateError(f"value-gate registration is not readable JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueGateError("value-gate registration must be a JSON object")
    unknown = sorted(set(payload) - _REGISTRATION_KEYS)
    if unknown:
        raise ValueGateError(f"value-gate registration has unknown keys: {unknown}")
    missing = sorted(_REGISTRATION_KEYS - set(payload))
    if missing:
        raise ValueGateError(f"value-gate registration is missing keys: {missing}")
    if payload.get("schema") != VALUE_REGISTRATION_SCHEMA:
        raise ValueGateError("value-gate registration schema mismatch")
    if payload.get("design_schema") != PHASE6_DESIGN_SCHEMA:
        raise ValueGateError("value-gate registration design schema mismatch")
    if str(payload.get("design_hash")) != phase6_design_hash():
        raise ValueGateError(
            "value-gate registration design hash differs from the running frozen design"
        )
    if dict(payload.get("value_gate_plan") or {}) != json.loads(
        json.dumps(phase6_design_payload()["value_gate"])
    ):
        raise ValueGateError("value-gate registration thresholds were edited after registration")
    declared = payload.get("registration_hash")
    if declared != canonical_hash(
        {key: value for key, value in payload.items() if key != "registration_hash"}
    ):
        raise ValueGateError("value-gate registration hash is inconsistent")
    if payload.get("state") != "registered_before_any_qualification_result":
        raise ValueGateError("value-gate registration state is invalid")
    return dict(payload)


# ---------------------------------------------------------------------------
# Qualification deals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QualificationDeal:
    deal_id: str
    seed: int
    level_rank: int
    leader_seat: int
    previous_result: dict[str, object] | None = None

    def game_config(self) -> GameConfig:
        return GameConfig(
            level_rank=int(self.level_rank),
            seed=int(self.seed),
            leader_seat=int(self.leader_seat),
            previous_result=_previous_from_payload(self.previous_result),
        )

    def payload(self) -> dict[str, object]:
        return {
            "deal_id": self.deal_id,
            "domain": "qualification",
            "seed": int(self.seed),
            "level_rank": int(self.level_rank),
            "leader_seat": int(self.leader_seat),
            "previous_result": None if self.previous_result is None else dict(self.previous_result),
        }


def _previous_from_payload(payload: Mapping[str, object] | None) -> PreviousHandResult | None:
    if payload is None:
        return None
    return PreviousHandResult(
        tuple(int(value) for value in payload["ranking"]),
        int(payload["winner_team"]),
        str(payload["outcome_class"]),
        bool(payload.get("tribute_cancelled", False)),
    )


def build_qualification_deals(
    count: int,
    *,
    base_seed: int = 2_000_000_000,
    start_index: int = 0,
    level_ranks: Sequence[int] = tuple(range(3, 16)),
    leader_seats: Sequence[int] = (0, 1, 2, 3),
    previous_results: Sequence[dict[str, object] | None] = (None,),
) -> tuple[QualificationDeal, ...]:
    if isinstance(count, bool) or int(count) < 0:
        raise ValueError("count must be a non-negative integer")
    if int(start_index) < 0:
        raise ValueError("start_index must be non-negative")
    if not level_ranks or not leader_seats or not previous_results:
        raise ValueError("deal context lists must be non-empty")
    deals = []
    for index in range(int(count)):
        absolute = int(start_index) + index
        deals.append(QualificationDeal(
            deal_id=f"qualification-{absolute:06d}",
            seed=int(base_seed) + absolute,
            level_rank=int(level_ranks[absolute % len(level_ranks)]),
            leader_seat=int(leader_seats[absolute % len(leader_seats)]),
            previous_result=previous_results[absolute % len(previous_results)],
        ))
    for deal in deals:
        deal.game_config()
    return tuple(deals)


def assert_qualification_isolation(
    deals: Sequence[QualificationDeal],
    *,
    protected_rows: Sequence[Mapping[str, object]] = (),
    protected_canonical_hashes: Sequence[str] = (),
    protected_seed_domains: Mapping[int, str] | None = None,
) -> dict[str, object]:
    """Qualification deals must not overlap training/selection/final data."""
    seeds = [int(deal.seed) for deal in deals]
    if len(set(seeds)) != len(seeds):
        raise ValueGateError("qualification deal seeds must be unique")
    ids = [deal.deal_id for deal in deals]
    if len(set(ids)) != len(ids):
        raise ValueGateError("qualification deal IDs must be unique")
    protected_seeds = {
        int(row["seed"]) for row in protected_rows if "seed" in row
    }
    overlap = sorted(set(seeds) & protected_seeds)
    if overlap:
        raise ValueGateError(f"qualification seeds overlap protected data: {overlap}")
    if protected_seed_domains is not None:
        for seed in seeds:
            if seed in protected_seed_domains:
                raise ValueGateError(
                    f"qualification seed {seed} belongs to {protected_seed_domains[seed]}"
                )
    protected_hashes = {str(value) for value in protected_canonical_hashes}
    return {
        "schema": "guandan-marvel-phase6-qualification-isolation-v1",
        "domain": "qualification",
        "deal_count": len(deals),
        "seed_base": min(seeds) if seeds else None,
        "protected_deal_count": len(protected_rows),
        "protected_hash_count": len(protected_hashes),
        "seed_overlap": 0,
        "protected_canonical_hashes_provided": len(protected_hashes),
    }


def canonical_physical_deal(hands: Sequence[Sequence[int]]) -> str:
    """Canonical deal identity over four seat rotations and the deck swap."""
    if len(hands) != 4:
        raise ValueError("a Guandan deal has four hands")
    normalized = [tuple(sorted(int(card) for card in hand)) for hand in hands]
    if sorted(card for hand in normalized for card in hand) != list(range(108)):
        raise ValueGateError("a canonical deal must conserve the 108 physical cards")
    variants = []
    for rotation in range(4):
        rotated = [normalized[(seat + rotation) % 4] for seat in range(4)]
        variants.append(tuple(rotated))
        variants.append(tuple(
            tuple(sorted((card + 54) % 108 for card in hand)) for hand in rotated
        ))
    return canonical_hash(min(variants))


def _deal_hands(env) -> list[list[int]]:
    state = env.debug_state(0)["state"]
    hands = [[int(card) for card in hand] for hand in state["hands"]]
    if len(hands) != 4 or any(len(hand) != 27 for hand in hands):
        raise ValueGateError("the engine did not produce a complete 27x4 deal")
    return hands


def _tribute_mode(previous: Mapping[str, object] | None) -> str:
    if previous is None:
        return "none"
    if bool(previous.get("tribute_cancelled", False)):
        return "cancelled"
    return str(previous.get("outcome_class", "unknown")).lower()


def _new_env(deal: QualificationDeal, *, max_steps: int):
    return cpp_game.EnvBatch(
        1, configs=[deal.game_config()], auto_reset=False, max_horizon=int(max_steps),
    )


def _clone_env(env):
    return cpp_game.EnvBatch.deserialize(env.serialize())


# ---------------------------------------------------------------------------
# Samples and collection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValueSample:
    sample_id: str
    deal_id: str
    canonical_deal: str
    token_step: int
    num_steps: int
    player: int
    team: int
    level_rank: int
    leader_seat: int
    tribute_mode: str
    prediction: float
    target: float
    game_phase: str
    outcome: str

    def payload(self) -> dict[str, object]:
        return {
            "schema": VALUE_SAMPLE_SCHEMA,
            "sample_id": self.sample_id,
            "deal_id": self.deal_id,
            "canonical_deal": self.canonical_deal,
            "token_step": int(self.token_step),
            "num_steps": int(self.num_steps),
            "player": int(self.player),
            "team": int(self.team),
            "level_rank": int(self.level_rank),
            "leader_seat": int(self.leader_seat),
            "tribute_mode": self.tribute_mode,
            "prediction": float(self.prediction),
            "target": float(self.target),
            "game_phase": self.game_phase,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class RankingPoint:
    deal_id: str
    canonical_deal: str
    token_step: int
    player: int
    candidate_count: int
    restricted: bool
    model_top_index: int
    realized_top_index: int
    accuracy: float
    random_baseline: float
    model_scores: tuple[float, ...]
    realized_returns: tuple[float, ...]
    candidate_token_sequences: tuple[tuple[tuple[int, ...], ...], ...]
    num_steps: int = 0
    game_phase: str = "early"
    history_consistent: bool = True
    metric_version: str = ""
    delta: float = 0.0
    informative: bool = False
    all_target_tie: bool = False
    model_tie: bool = False
    target_top_count: int = 0
    model_top_count: int = 0
    legacy_metrics_ignored: bool = False

    def payload(self) -> dict[str, object]:
        return {
            "schema": VALUE_RANKING_POINT_SCHEMA,
            "metric_version": self.metric_version or VALUE_RANKING_METRIC_VERSION,
            "deal_id": self.deal_id,
            "canonical_deal": self.canonical_deal,
            "token_step": int(self.token_step),
            "num_steps": int(self.num_steps),
            "game_phase": self.game_phase,
            "history_consistent": bool(self.history_consistent),
            "player": int(self.player),
            "candidate_count": int(self.candidate_count),
            "restricted": bool(self.restricted),
            "model_top_index": int(self.model_top_index),
            "realized_top_index": int(self.realized_top_index),
            "accuracy": float(self.accuracy),
            "random_baseline": float(self.random_baseline),
            "delta": float(self.delta),
            "informative": bool(self.informative),
            "all_target_tie": bool(self.all_target_tie),
            "model_tie": bool(self.model_tie),
            "target_top_count": int(self.target_top_count),
            "model_top_count": int(self.model_top_count),
            "legacy_metrics_ignored": bool(self.legacy_metrics_ignored),
            "model_scores": [float(value) for value in self.model_scores],
            "realized_returns": [float(value) for value in self.realized_returns],
            "candidate_token_sequences": [
                [list(row) for row in candidate] for candidate in self.candidate_token_sequences
            ],
        }


def _top_indices(values: Sequence[float], *, atol: float) -> tuple[int, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueGateError("a ranking point must have at least one candidate")
    if not np.all(np.isfinite(array)):
        raise ValueGateError("ranking inputs must be finite")
    top = float(np.max(array))
    return tuple(int(index) for index in np.flatnonzero(array >= top - float(atol)))


def compute_tie_aware_ranking(
    model_scores: Sequence[float], realized_returns: Sequence[float],
) -> dict[str, object]:
    """Apply the frozen tie-aware contract to one ranking point.

    ``accuracy = |M intersect T| / |M|`` is the expected success of breaking a
    model tie uniformly at random; ``random_baseline = |T| / k`` is the same
    quantity for a uniform candidate strategy.  All-target-tie points are
    reported (accuracy=1, baseline=1, delta=0) but never counted as
    informative ranking evidence.
    """
    if len(model_scores) != len(realized_returns):
        raise ValueGateError(
            "model_scores and realized_returns must have the same length: "
            f"{len(model_scores)} != {len(realized_returns)}"
        )
    k = len(model_scores)
    if k == 0:
        raise ValueGateError("a ranking point must have at least one candidate")
    model_top = _top_indices(model_scores, atol=MODEL_SCORE_TIE_ATOL)
    target_top = _top_indices(realized_returns, atol=REALIZED_RETURN_TIE_ATOL)
    if k == 1:
        accuracy = 1.0
        baseline = 1.0
        informative = False
        reason = "single_candidate"
    else:
        accuracy = len(set(model_top) & set(target_top)) / float(len(model_top))
        baseline = len(target_top) / float(k)
        informative = len(target_top) < k
        reason = None if informative else "all_target_tie"
    return {
        "metric_version": VALUE_RANKING_METRIC_VERSION,
        "candidate_count": int(k),
        "model_top_indices": model_top,
        "realized_top_indices": target_top,
        "model_top_index": int(model_top[0]),
        "realized_top_index": int(target_top[0]),
        "model_top_count": len(model_top),
        "target_top_count": len(target_top),
        "accuracy": float(accuracy),
        "random_baseline": float(baseline),
        "delta": float(accuracy - baseline),
        "informative": bool(informative),
        "all_target_tie": bool(len(target_top) == k),
        "model_tie": bool(len(model_top) == k),
        "not_informative_reason": reason,
    }


def revalidate_ranking_point(point: RankingPoint) -> RankingPoint:
    """Recompute a point from its raw arrays and cross-check serialized values.

    The gate never trusts caller-supplied ``accuracy``/``random_baseline``/
    ``top_index``.  A point carrying an unknown metric version fails closed; a
    legacy v2 point (or a point built without a version) is recomputed and its
    stored metrics are explicitly marked as ignored.
    """
    record = compute_tie_aware_ranking(point.model_scores, point.realized_returns)
    if int(point.candidate_count) != int(record["candidate_count"]):
        raise ValueGateError(
            "ranking point candidate_count disagrees with its raw score arrays"
        )
    if len(point.candidate_token_sequences) not in (0, int(record["candidate_count"])):
        raise ValueGateError(
            "ranking point candidate_token_sequences disagrees with its candidate count"
        )
    version = str(point.metric_version or "")
    if version == VALUE_RANKING_METRIC_VERSION:
        legacy = False
        checks = {
            "accuracy": float(point.accuracy),
            "random_baseline": float(point.random_baseline),
            "delta": float(point.delta),
        }
        for key, stored in checks.items():
            if not math.isclose(stored, float(record[key]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueGateError(
                    f"serialized ranking {key}={stored!r} disagrees with the recomputed "
                    f"tie-aware value {record[key]!r}; unknown or forged metrics are "
                    "never trusted"
                )
        if int(point.model_top_count) and int(point.model_top_count) != int(
            record["model_top_count"]
        ):
            raise ValueGateError("serialized ranking model_top_count is inconsistent")
        if int(point.target_top_count) and int(point.target_top_count) != int(
            record["target_top_count"]
        ):
            raise ValueGateError("serialized ranking target_top_count is inconsistent")
        if bool(point.informative) != bool(record["informative"]):
            raise ValueGateError("serialized ranking informative flag is inconsistent")
        if bool(point.all_target_tie) != bool(record["all_target_tie"]):
            raise ValueGateError("serialized ranking all-target-tie flag is inconsistent")
        if bool(point.model_tie) != bool(record["model_tie"]):
            raise ValueGateError("serialized ranking model-tie flag is inconsistent")
    elif version and version not in VALUE_RANKING_LEGACY_POINT_SCHEMAS:
        raise ValueGateError(
            f"unknown ranking metric version {version!r}; refusing to trust its metrics"
        )
    else:
        legacy = True
    return RankingPoint(
        deal_id=point.deal_id,
        canonical_deal=point.canonical_deal,
        token_step=int(point.token_step),
        num_steps=int(point.num_steps),
        game_phase=str(point.game_phase),
        history_consistent=bool(point.history_consistent),
        player=int(point.player),
        candidate_count=int(record["candidate_count"]),
        restricted=bool(point.restricted),
        model_top_index=int(record["model_top_index"]),
        realized_top_index=int(record["realized_top_index"]),
        accuracy=float(record["accuracy"]),
        random_baseline=float(record["random_baseline"]),
        delta=float(record["delta"]),
        informative=bool(record["informative"]),
        all_target_tie=bool(record["all_target_tie"]),
        model_tie=bool(record["model_tie"]),
        target_top_count=int(record["target_top_count"]),
        model_top_count=int(record["model_top_count"]),
        metric_version=VALUE_RANKING_METRIC_VERSION,
        legacy_metrics_ignored=legacy,
        model_scores=tuple(float(value) for value in point.model_scores),
        realized_returns=tuple(float(value) for value in point.realized_returns),
        candidate_token_sequences=point.candidate_token_sequences,
    )


def _phase_bucket(token_step: int, num_steps: int) -> str:
    if num_steps <= 0:
        return "late"
    fraction = token_step / float(num_steps)
    if fraction <= 1.0 / 3.0:
        return "early"
    if fraction <= 2.0 / 3.0:
        return "mid"
    return "late"


def _legal_rows(step) -> list[np.ndarray]:
    weights = np.asarray(step.available_actions.weights[0], dtype=np.float32)
    rows = np.asarray(step.available_actions.tokens[0], dtype=np.int64)
    return [np.array(rows[index], dtype=np.int64) for index in np.flatnonzero(weights > 0)]


def _is_commit_row(row: Sequence[int]) -> bool:
    return int(row[6]) == int(cpp_game.ACTION_COMMIT)


def _row_is_macro_start(step) -> bool:
    rows = _legal_rows(step)
    if not rows:
        return False
    stages = {int(row[2]) - ACTION_STAGE_BASE for row in rows}
    if len(stages) != 1:
        raise ValueGateError("legal rows disagree on the action prefix stage")
    return stages.pop() == 0


def enumerate_complete_macros(
    env,
    *,
    max_candidates: int = VALUE_GATE_PLAN["counterfactual"]["max_candidates"],
    max_nodes: int = VALUE_GATE_PLAN["counterfactual"]["max_enumeration_nodes"],
) -> tuple[list[dict[str, object]], int, bool]:
    """Enumerate complete legal macros by DFS over real engine branches."""
    root = _clone_env(env)
    stack: list[tuple[Any, list[np.ndarray], float]] = [(root, [], 0.0)]
    candidates: list[dict[str, object]] = []
    nodes = 0
    restricted = False
    seen: set[tuple[tuple[int, ...], ...]] = set()
    while stack:
        if nodes >= int(max_nodes):
            restricted = True
            break
        branch, sequence, immediate = stack.pop()
        step = branch.observe()
        if int(step.player_ids[0]) < 0:
            continue
        for row in _legal_rows(step):
            if nodes >= int(max_nodes):
                restricted = True
                break
            nodes += 1
            next_env = _clone_env(branch)
            try:
                next_step = next_env.step(row.reshape(1, -1))
            except cpp_game.TrajectoryHorizonError:
                restricted = True
                continue
            player = int(step.player_ids[0])
            reward = float(next_step.rewards[0, player]) * TRAINING_REWARD_SCALE
            new_sequence = sequence + [row]
            key = tuple(tuple(int(value) for value in item) for item in new_sequence)
            terminal = bool(next_step.is_terminal[0])
            if _is_commit_row(row):
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "rows": new_sequence,
                    "env": next_env,
                    "terminal": terminal,
                    "immediate_reward": immediate + reward,
                })
            elif not terminal:
                stack.append((next_env, new_sequence, immediate + reward))
    if len(candidates) > int(max_candidates):
        restricted = True
        candidates = candidates[: int(max_candidates)]
    return candidates, nodes, restricted


def _value_at_state(
    artifact: V2PolicyArtifact,
    *,
    deal: QualificationDeal,
    rows_to_state: Sequence[np.ndarray],
    target_player: int,
    device,
    max_steps: int,
) -> float:
    """Full-history value for ``target_player`` at the state after ``rows``.

    The review of the first implementation found that a one-step forward only
    sees the current observation and is therefore not the value the policy
    actually assigns to the state.  This helper replays the real engine from the
    deal start, records every seat's observation for every step, and runs one
    stateless full-history forward with ``player_ids`` fixed to the target
    player.  The last step's state value is the full-history value.
    """
    from guandan_marvel.model.tensordict import TensorDict

    env = _new_env(deal, max_steps=int(max_steps) + len(rows_to_state) + 1)
    step = env.reset(int(deal.seed))

    def _record(current) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(current.observations.tokens, dtype=torch.long, device=device),
            torch.as_tensor(current.observations.channels, dtype=torch.uint8, device=device),
            torch.as_tensor(current.available_actions.tokens, dtype=torch.long, device=device),
            torch.as_tensor(current.available_actions.weights, dtype=torch.float32, device=device),
        )

    observations = [_record(step)]
    for row in rows_to_state:
        try:
            step = env.step(np.asarray(row, dtype=np.int64).reshape(1, -1))
        except cpp_game.TrajectoryHorizonError as error:
            raise ValueGateError("full-history value replay exceeded the engine horizon") from error
        observations.append(_record(step))
    tokens = torch.stack([item[0] for item in observations], dim=1)
    channels = torch.stack([item[1] for item in observations], dim=1)
    avail_tokens = torch.stack([item[2] for item in observations], dim=1)
    avail_weights = torch.stack([item[3] for item in observations], dim=1)
    length = tokens.shape[1]
    rollout = TensorDict({
        "num_steps": torch.full((1,), length, dtype=torch.long, device=device),
        "player_ids": torch.full((1, length), int(target_player), dtype=torch.long, device=device),
        "obs_tokens": tokens,
        "obs_channels": channels,
        "avail_tokens": avail_tokens,
        "avail_weights": avail_weights,
    }, batch_size=[1], device=device)
    with torch.inference_mode():
        output = artifact.model(rollout=rollout)
    if output.player_state_value is None:
        raise ValueGateError("the qualification checkpoint has no value head")
    value = float(output.player_state_value[0, -1].float().item())
    if not math.isfinite(value):
        raise ValueGateError("the value head produced a non-finite prediction")
    return value


def _continue_from_rows(
    deal: QualificationDeal,
    rows: Sequence[np.ndarray],
    *,
    history_length: int,
    player: int,
    artifact: V2PolicyArtifact,
    device,
    common_seed: int,
    max_steps: int,
) -> float:
    """Replay ``rows`` then continue with the fixed policy; realized return.

    ``rows[:history_length]`` merely rebuilds the policy's causal context;
    only the candidate macro and its continuation contribute to the returned
    value.
    """
    env = _new_env(deal, max_steps=max_steps)
    step = env.reset(int(deal.seed))
    session = V2PolicySession(artifact, seed=int(common_seed), temperature=1.0, device=device)
    rewards: list[float] = []
    for row in rows:
        probabilities = session.observe_and_score(step)
        del probabilities
        try:
            step = env.step(np.asarray(row, dtype=np.int64).reshape(1, -1))
        except cpp_game.TrajectoryHorizonError as error:
            raise ValueGateError("counterfactual replay exceeded the engine horizon") from error
        rewards.append(float(step.rewards[0, int(player)]) * TRAINING_REWARD_SCALE)
        if bool(step.is_terminal[0]):
            return float(sum(rewards[int(history_length):]))
    for _ in range(int(max_steps)):
        if bool(step.is_terminal[0]):
            break
        probabilities = session.observe_and_score(step)
        index = session.select_index(probabilities, step)
        row = np.asarray(step.available_actions.tokens[0, index], dtype=np.int64)
        try:
            step = env.step(row.reshape(1, -1))
        except cpp_game.TrajectoryHorizonError as error:
            raise ValueGateError("counterfactual continuation exceeded the engine horizon") from error
        rewards.append(float(step.rewards[0, int(player)]) * TRAINING_REWARD_SCALE)
    if not bool(step.is_terminal[0]):
        raise ValueGateError("counterfactual continuation did not reach a terminal")
    return float(sum(rewards[int(history_length):]))


def evaluate_ranking_point(
    env,
    *,
    deal: QualificationDeal,
    artifact: V2PolicyArtifact,
    history_rows: Sequence[np.ndarray],
    player: int,
    canonical_deal: str,
    token_step: int,
    num_steps: int,
    game_phase: str,
    device,
    common_seed: int,
    max_steps: int,
) -> RankingPoint | None:
    candidates, _nodes, restricted = enumerate_complete_macros(env)
    if len(candidates) < 2:
        return None
    model_scores: list[float] = []
    realized_returns: list[float] = []
    sequences: list[tuple[tuple[int, ...], ...]] = []
    for candidate in candidates:
        immediate = float(candidate["immediate_reward"])
        if bool(candidate["terminal"]):
            post_value = 0.0
        else:
            post_value = _value_at_state(
                artifact,
                deal=deal,
                rows_to_state=list(history_rows) + list(candidate["rows"]),
                target_player=int(player),
                device=device,
                max_steps=int(max_steps),
            )
        model_scores.append(immediate + post_value)
        realized_returns.append(_continue_from_rows(
            deal,
            list(history_rows) + list(candidate["rows"]),
            history_length=len(history_rows),
            player=int(player), artifact=artifact, device=device,
            common_seed=int(common_seed), max_steps=int(max_steps),
        ))
        sequences.append(tuple(
            tuple(int(value) for value in row) for row in candidate["rows"]
        ))
    if not all(math.isfinite(value) for value in model_scores):
        raise ValueGateError("a counterfactual model score is non-finite")
    if not all(math.isfinite(value) for value in realized_returns):
        raise ValueGateError("a counterfactual realized return is non-finite")
    ranking = compute_tie_aware_ranking(model_scores, realized_returns)
    return RankingPoint(
        deal_id=deal.deal_id,
        canonical_deal=canonical_deal,
        token_step=int(token_step),
        num_steps=int(num_steps),
        game_phase=str(game_phase),
        history_consistent=True,
        player=int(player),
        candidate_count=len(candidates),
        restricted=bool(restricted),
        model_top_index=int(ranking["model_top_index"]),
        realized_top_index=int(ranking["realized_top_index"]),
        accuracy=float(ranking["accuracy"]),
        random_baseline=float(ranking["random_baseline"]),
        delta=float(ranking["delta"]),
        informative=bool(ranking["informative"]),
        all_target_tie=bool(ranking["all_target_tie"]),
        model_tie=bool(ranking["model_tie"]),
        target_top_count=int(ranking["target_top_count"]),
        model_top_count=int(ranking["model_top_count"]),
        metric_version=VALUE_RANKING_METRIC_VERSION,
        model_scores=tuple(float(value) for value in model_scores),
        realized_returns=tuple(float(value) for value in realized_returns),
        candidate_token_sequences=tuple(sequences),
    )


@dataclass
class QualificationResult:
    samples: list[ValueSample] = field(default_factory=list)
    ranking_points: list[RankingPoint] = field(default_factory=list)
    deal_records: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def _stratified_starts(
    starts: Sequence[Mapping[str, object]], num_steps: int, limit: int,
    *, order: Sequence[str],
) -> list[Mapping[str, object]]:
    buckets: dict[str, list[Mapping[str, object]]] = {"early": [], "mid": [], "late": []}
    for start in starts:
        buckets[_phase_bucket(int(start["token_step"]), int(num_steps))].append(start)
    selected: list[Mapping[str, object]] = []
    index = 0
    while len(selected) < int(limit):
        progressed = False
        for name in order:
            if index < len(buckets[name]):
                selected.append(buckets[name][index])
                progressed = True
                if len(selected) >= int(limit):
                    break
        if not progressed:
            break
        index += 1
    return selected


def _replay_to_step(
    deal: QualificationDeal,
    rows: Sequence[np.ndarray],
    *,
    max_steps: int,
):
    env = _new_env(deal, max_steps=int(max_steps) + len(rows) + 1)
    env.reset(int(deal.seed))
    applied: list[np.ndarray] = []
    for row in rows:
        try:
            env.step(np.asarray(row, dtype=np.int64).reshape(1, -1))
        except cpp_game.TrajectoryHorizonError as error:
            raise ValueGateError("qualification replay exceeded the engine horizon") from error
        applied.append(np.asarray(row, dtype=np.int64))
    return env, applied


def run_value_qualification(
    *,
    checkpoint_path: str | Path,
    registration_path: str | Path,
    deals: Sequence[QualificationDeal],
    device: str = "cuda",
    max_deals: int | None = None,
    max_macro_starts_per_deal: int = 32,
    max_steps_per_deal: int = 1024,
    wall_seconds: float = 300.0,
    max_ranking_points: int = 2,
    protected_canonical_hashes: Sequence[str] = (),
) -> QualificationResult:
    """Collect bounded qualification evidence from real C++ games.

    Exceeding the wall budget or deal cap stops collection and reports partial
    evidence; it never increases the budget or falls back to CPU.  Macro-start
    samples are stratified over early/mid/late after the real terminal length
    is known, so a cap can no longer leave the late phase empty.  Ranking
    points are selected per deal with a mid/late preference and replayed from
    the deal start, so their value uses the full observation history.
    """
    registration = load_value_gate_registration(registration_path)
    artifact = load_v2_policy(checkpoint_path, device=device, expected_algorithm="ippo")
    if artifact.schema != EVALUATION_CHECKPOINT_SCHEMA:
        raise ValueGateError(
            "value qualification requires an evaluation-only checkpoint "
            f"({EVALUATION_CHECKPOINT_SCHEMA}); got {artifact.schema!r}"
        )
    if artifact.metadata.get("training_resume_allowed") is True:
        raise ValueGateError("qualification checkpoint advertises training resume")
    if float(wall_seconds) <= 0:
        raise ValueError("wall_seconds must be positive")
    limit = len(deals) if max_deals is None else min(int(max_deals), len(deals))
    if limit <= 0:
        raise ValueError("at least one qualification deal is required")
    protected_hashes = {str(value) for value in protected_canonical_hashes}
    started = time.perf_counter()
    deadline = started + float(wall_seconds)
    result = QualificationResult()
    truncated = False
    ranking_skipped = 0
    seen_canonical: dict[str, str] = {}
    for deal in deals[:limit]:
        if time.perf_counter() >= deadline:
            truncated = True
            break
        env = _new_env(deal, max_steps=int(max_steps_per_deal))
        step = env.reset(int(deal.seed))
        hands = _deal_hands(env)
        canonical = canonical_physical_deal(hands)
        if canonical in protected_hashes:
            raise ValueGateError(
                "qualification deal overlaps protected training/selection/final data "
                f"(canonical {canonical})"
            )
        if canonical in seen_canonical:
            result.deal_records.append({
                "deal_id": deal.deal_id,
                "canonical_deal": canonical,
                "status": "duplicate_canonical_deal",
                "duplicate_of": seen_canonical[canonical],
            })
            continue
        seen_canonical[canonical] = deal.deal_id
        session = V2PolicySession(
            artifact, seed=int(deal.seed) + 999_983, temperature=1.0, device=device,
        )
        history_rows: list[np.ndarray] = []
        reward_steps: list[list[float]] = []
        starts: list[dict[str, object]] = []
        token_step = 0
        terminal_info: dict[str, object] | None = None
        while token_step < int(max_steps_per_deal):
            if time.perf_counter() >= deadline:
                truncated = True
                break
            player = int(step.player_ids[0])
            if player < 0:
                raise ValueGateError("a nonterminal qualification frame has no active player")
            probabilities, value = session.observe_score_and_value(step)
            if _row_is_macro_start(step) and value is not None and len(starts) < 256:
                starts.append({
                    "token_step": token_step,
                    "player": player,
                    "prediction": float(value.item()),
                })
            index = session.select_index(probabilities, step)
            row = np.asarray(step.available_actions.tokens[0, index], dtype=np.int64)
            try:
                step = env.step(row.reshape(1, -1))
            except cpp_game.TrajectoryHorizonError:
                truncated = True
                break
            history_rows.append(row)
            reward_steps.append([
                float(value) * TRAINING_REWARD_SCALE
                for value in np.asarray(step.rewards[0]).tolist()
            ])
            token_step += 1
            if bool(step.is_terminal[0]):
                terminal_info = dict(step.infos[0])
                break
        if terminal_info is None:
            result.deal_records.append({
                "deal_id": deal.deal_id,
                "canonical_deal": canonical,
                "status": "truncated",
                "token_steps": token_step,
            })
            continue
        num_steps = token_step
        outcome = str(terminal_info.get("outcome_class", "UNKNOWN"))
        tribute_mode = _tribute_mode(deal.previous_result)
        selected_starts = _stratified_starts(
            starts, num_steps, int(max_macro_starts_per_deal),
            order=("late", "mid", "early"),
        )
        for start in selected_starts:
            index = int(start["token_step"])
            player = int(start["player"])
            target = float(sum(row[player] for row in reward_steps[index:]))
            result.samples.append(ValueSample(
                sample_id=f"{deal.deal_id}-{index}",
                deal_id=deal.deal_id,
                canonical_deal=canonical,
                token_step=index,
                num_steps=num_steps,
                player=player,
                team=player % 2,
                level_rank=int(deal.level_rank),
                leader_seat=int(deal.leader_seat),
                tribute_mode=tribute_mode,
                prediction=float(start["prediction"]),
                target=target,
                game_phase=_phase_bucket(index, num_steps),
                outcome=outcome,
            ))
        if len(result.ranking_points) < int(max_ranking_points):
            ranking_candidates = _stratified_starts(
                starts, num_steps,
                min(
                    int(VALUE_GATE_PLAN["counterfactual"]["max_points_per_deal"]),
                    int(max_ranking_points) - len(result.ranking_points),
                ),
                order=("mid", "late", "early"),
            )
            for start in ranking_candidates:
                if len(result.ranking_points) >= int(max_ranking_points):
                    break
                decision_step = int(start["token_step"])
                decision_player = int(start["player"])
                try:
                    replay_env, replay_rows = _replay_to_step(
                        deal, history_rows[:decision_step],
                        max_steps=int(max_steps_per_deal),
                    )
                    point = evaluate_ranking_point(
                        replay_env, deal=deal, artifact=artifact,
                        history_rows=replay_rows, player=decision_player,
                        canonical_deal=canonical, token_step=decision_step,
                        num_steps=num_steps,
                        game_phase=_phase_bucket(decision_step, num_steps),
                        device=device,
                        common_seed=(
                            int(deal.seed) + 10_000 * (decision_step + 1) + decision_player
                        ),
                        max_steps=int(
                            VALUE_GATE_PLAN["counterfactual"]["max_continuation_steps"]
                        ),
                    )
                except ValueGateError:
                    ranking_skipped += 1
                    point = None
                if point is not None:
                    result.ranking_points.append(point)
        result.deal_records.append({
            "deal_id": deal.deal_id,
            "canonical_deal": canonical,
            "status": "complete",
            "token_steps": num_steps,
            "outcome": outcome,
            "ranking": list(terminal_info.get("ranking", ())),
        })
    result.metadata = {
        "schema": VALUE_QUALIFICATION_DATA_SCHEMA,
        "checkpoint": {
            **dict(artifact.identity),
            "state_dict_sha256": _state_dict_weight_hash(artifact.model.state_dict()),
        },
        "registration_hash": registration["registration_hash"],
        "design_hash": registration["design_hash"],
        "device": str(device),
        "deals_requested": len(deals),
        "deals_attempted": limit,
        "deals_completed": sum(
            1 for record in result.deal_records if record.get("status") == "complete"
        ),
        "independent_deals": len(seen_canonical),
        "duplicate_canonical_deals": sum(
            1 for record in result.deal_records
            if record.get("status") == "duplicate_canonical_deal"
        ),
        "sample_count": len(result.samples),
        "sample_selection": "stratified_late_mid_early",
        "ranking_points": len(result.ranking_points),
        "ranking_points_skipped": int(ranking_skipped),
        "ranking_selection": "mid_late_early_replayed_full_history",
        "ranking_metric_version": VALUE_RANKING_METRIC_VERSION,
        "ranking_informative_points": sum(
            1 for point in result.ranking_points if point.informative
        ),
        "ranking_all_target_tie_points": sum(
            1 for point in result.ranking_points if point.all_target_tie
        ),
        "ranking_model_tie_points": sum(
            1 for point in result.ranking_points if point.model_tie
        ),
        "ranking_mid_late_informative_points": sum(
            1 for point in result.ranking_points
            if point.informative and point.game_phase in ("mid", "late")
        ),
        "producer": {
            "schema": VALUE_QUALIFICATION_PRODUCER_SCHEMA,
            "source_hash": qualification_producer_source_hash(),
            "source_scope": list(QUALIFICATION_PRODUCER_SOURCE_FILES),
            "source_semantics": "phase-6 qualification producer source; not checkpoint origin",
            "design_hash": registration["design_hash"],
        },
        "sample_target_contract_hash": canonical_hash(VALUE_TARGET_CONTRACT),
        "protected_canonical_hash_count": len(protected_hashes),
        "truncated": bool(truncated),
        "target": {
            "perspective": "acting_player",
            "scale": "rank_point_over_3",
            "discount_gamma": 1.0,
            "raw_reward_scale": TRAINING_REWARD_SCALE,
        },
        "clustering_unit": "canonical_physical_deal_rotations_and_deck_swap",
        "wall_seconds": time.perf_counter() - started,
        "wall_seconds_cap": float(wall_seconds),
    }
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _ev(prediction: Sequence[float], target: Sequence[float]) -> tuple[float | None, int]:
    if len(target) == 0:
        return None, 0
    y = np.asarray(target, dtype=np.float64)
    y_hat = np.asarray(prediction, dtype=np.float64)
    variance = float(np.var(y))
    if not math.isfinite(variance) or variance < 1e-12:
        return None, 1
    return float(1.0 - np.var(y - y_hat) / variance), 0


def _pearson(prediction: Sequence[float], target: Sequence[float]) -> float | None:
    if len(target) < 2:
        return None
    y = np.asarray(target, dtype=np.float64)
    y_hat = np.asarray(prediction, dtype=np.float64)
    if float(np.var(y)) < 1e-12 or float(np.var(y_hat)) < 1e-12:
        return None
    value = float(np.corrcoef(y_hat, y)[0, 1])
    return value if math.isfinite(value) else None


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks for ties (independent of scipy)."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    sorted_values = array[order]
    index = 0
    while index < len(array):
        end = index + 1
        while end < len(array) and sorted_values[end] == sorted_values[index]:
            end += 1
        average = (index + end - 1) / 2.0 + 1.0
        ranks[order[index:end]] = average
        index = end
    return ranks


def _spearman(prediction: Sequence[float], target: Sequence[float]) -> float | None:
    if len(target) < 2:
        return None
    ranks_pred = _rankdata(prediction)
    ranks_target = _rankdata(target)
    return _pearson(ranks_pred.tolist(), ranks_target.tolist())


def _calibration(
    prediction: Sequence[float], target: Sequence[float], *, bins: int,
) -> tuple[dict[str, object], float | None]:
    if len(target) == 0 or bins <= 0:
        return {"bins": [], "ece": None}, None
    y = np.asarray(target, dtype=np.float64)
    y_hat = np.asarray(prediction, dtype=np.float64)
    order = np.argsort(y_hat, kind="mergesort")
    chunks = np.array_split(order, min(int(bins), len(order)))
    rows = []
    ece = 0.0
    for chunk in chunks:
        if chunk.size == 0:
            continue
        mean_pred = float(np.mean(y_hat[chunk]))
        mean_target = float(np.mean(y[chunk]))
        row = {
            "count": int(chunk.size),
            "mean_prediction": mean_pred,
            "mean_target": mean_target,
            "abs_gap": abs(mean_pred - mean_target),
        }
        rows.append(row)
        ece += chunk.size / len(order) * row["abs_gap"]
    return {"bins": rows, "ece": float(ece)}, float(ece)


def _cluster_bootstrap_ev(
    samples: Sequence[ValueSample], *, resamples: int, seed: int, confidence: float,
) -> dict[str, object]:
    clusters: dict[str, list[ValueSample]] = {}
    for sample in samples:
        clusters.setdefault(sample.canonical_deal, []).append(sample)
    if not clusters:
        return {
            "method": "cluster_bootstrap_ev", "resamples": 0, "low": None, "high": None,
            "undefined_resample_count": 0, "conclusive": False,
        }
    keys = sorted(clusters)
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    undefined = 0
    for _ in range(int(resamples)):
        chosen = rng.integers(0, len(keys), size=len(keys))
        pooled: list[ValueSample] = []
        for index in chosen:
            pooled.extend(clusters[keys[int(index)]])
        ev, degenerate = _ev(
            [sample.prediction for sample in pooled],
            [sample.target for sample in pooled],
        )
        if ev is None:
            undefined += int(degenerate)
            continue
        values.append(ev)
    if len(values) < 2:
        return {
            "method": "cluster_bootstrap_ev", "resamples": int(resamples),
            "low": None, "high": None, "undefined_resample_count": undefined,
            "conclusive": False,
        }
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "method": "cluster_bootstrap_ev",
        "confidence": float(confidence),
        "resamples": int(resamples),
        "low": float(np.quantile(values, alpha)),
        "high": float(np.quantile(values, 1.0 - alpha)),
        "undefined_resample_count": undefined,
        "conclusive": undefined == 0,
    }


def _subset_metrics(samples: Sequence[ValueSample]) -> dict[str, object]:
    if not samples:
        return {
            "sample_count": 0, "independent_deal_count": 0,
            "zero_target_variance_count": 0, "ev": None, "pearson": None,
            "spearman": None,
        }
    ev, degenerate = _ev(
        [sample.prediction for sample in samples],
        [sample.target for sample in samples],
    )
    return {
        "sample_count": len(samples),
        "independent_deal_count": len({sample.canonical_deal for sample in samples}),
        "zero_target_variance_count": int(degenerate),
        "ev": ev,
        "pearson": _pearson(
            [sample.prediction for sample in samples],
            [sample.target for sample in samples],
        ),
        "spearman": _spearman(
            [sample.prediction for sample in samples],
            [sample.target for sample in samples],
        ),
    }


def compute_value_metrics(samples: Sequence[ValueSample]) -> dict[str, object]:
    slots: dict[str, dict[str, list[ValueSample]]] = {
        "game_phase": {}, "outcome": {}, "seat_team": {},
        "level_rank": {}, "leader_seat": {}, "tribute_mode": {},
    }
    for sample in samples:
        slots["game_phase"].setdefault(sample.game_phase, []).append(sample)
        slots["outcome"].setdefault(sample.outcome, []).append(sample)
        slots["seat_team"].setdefault(f"team_{sample.team}", []).append(sample)
        slots["level_rank"].setdefault(f"level_{sample.level_rank}", []).append(sample)
        slots["leader_seat"].setdefault(f"leader_{sample.leader_seat}", []).append(sample)
        slots["tribute_mode"].setdefault(sample.tribute_mode, []).append(sample)
    overall = _subset_metrics(samples)
    calibration, ece = _calibration(
        [sample.prediction for sample in samples],
        [sample.target for sample in samples],
        bins=int(VALUE_GATE_PLAN["metrics"]["calibration_bins"]),
    )
    return {
        "overall": overall,
        "calibration": calibration,
        "calibration_ece": ece,
        "slices": {
            name: {key: _subset_metrics(value) for key, value in sorted(groups.items())}
            for name, groups in slots.items()
        },
    }


def compute_ranking_metrics(
    points: Sequence[RankingPoint], *, resamples: int = 2000, seed: int = 20261020,
) -> dict[str, object]:
    """Aggregate tie-aware ranking evidence over *informative* points only.

    All-target-tie points are reported (they carry zero advantage by the
    contract) but never fill the informative threshold.  Cluster bootstrap is
    over canonical deals, exactly like the EV interval.
    """
    points = [revalidate_ranking_point(point) for point in points]
    informative = [point for point in points if point.informative]
    legacy_ignored = sum(1 for point in points if point.legacy_metrics_ignored)
    base: dict[str, object] = {
        "metric_version": VALUE_RANKING_METRIC_VERSION,
        "ranking_points": len(points),
        "informative_points": len(informative),
        "all_target_tie_points": sum(1 for point in points if point.all_target_tie),
        "model_tie_points": sum(1 for point in points if point.model_tie),
        "legacy_metric_values_ignored": legacy_ignored,
        "independent_deals": len({point.canonical_deal for point in points}),
        "informative_deals": len({point.canonical_deal for point in informative}),
        "restricted_points": sum(1 for point in points if point.restricted),
        "restricted_informative_points": sum(
            1 for point in informative if point.restricted
        ),
    }
    if not informative:
        return {
            **base,
            "accuracy": None,
            "random_baseline": None,
            "delta": None,
            "ci": {"low": None, "high": None, "conclusive": False, "method": "cluster_bootstrap_accuracy_delta", "resamples": int(resamples)},
            "by_team": {
                f"team_{team}": {"points": 0, "informative_points": 0, "accuracy": None}
                for team in (0, 1)
            },
        }
    accuracy = float(np.mean([point.accuracy for point in informative]))
    baseline = float(np.mean([point.random_baseline for point in informative]))
    clusters: dict[str, list[RankingPoint]] = {}
    for point in informative:
        clusters.setdefault(point.canonical_deal, []).append(point)
    keys = sorted(clusters)
    rng = np.random.default_rng(int(seed))
    deltas: list[float] = []
    for _ in range(int(resamples)):
        chosen = rng.integers(0, len(keys), size=len(keys))
        pooled = [
            point for index in chosen for point in clusters[keys[int(index)]]
        ]
        deltas.append(
            float(np.mean([point.accuracy for point in pooled]))
            - float(np.mean([point.random_baseline for point in pooled]))
        )
    low = float(np.quantile(deltas, 0.025)) if deltas else None
    high = float(np.quantile(deltas, 0.975)) if deltas else None
    by_team = {}
    for team in (0, 1):
        total_subset = [point for point in points if int(point.player) % 2 == team]
        subset = [point for point in informative if int(point.player) % 2 == team]
        by_team[f"team_{team}"] = {
            "points": len(total_subset),
            "informative_points": len(subset),
            "accuracy": float(np.mean([point.accuracy for point in subset])) if subset else None,
        }
    return {
        **base,
        "accuracy": accuracy,
        "random_baseline": baseline,
        "delta": accuracy - baseline,
        "by_team": by_team,
        "ci": {
            "method": "cluster_bootstrap_accuracy_delta",
            "resamples": int(resamples),
            "low": low,
            "high": high,
            "conclusive": bool(deltas),
        },
    }


def _state_dict_weight_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Deterministic model-weight fingerprint (name/shape/dtype/bytes)."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _verify_gate_checkpoint_identities(
    identities: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str | None], bool, list[str]]:
    verified: list[dict[str, object]] = []
    weight_hashes: list[str | None] = []
    problems: list[str] = []
    all_verified = True
    for identity in identities:
        if not isinstance(identity, Mapping):
            raise ValueGateError("checkpoint identity must be a mapping")
        sha = str(identity.get("checkpoint_sha256", ""))
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha.lower()):
            raise ValueGateError("checkpoint identity SHA256 is missing or malformed")
        path = identity.get("checkpoint_path")
        if path is not None and Path(str(path)).is_file():
            try:
                artifact = verify_policy_identity(identity, device="cpu")
            except Exception as error:  # noqa: BLE001 - fail closed on any verifier error
                raise ValueGateError(
                    f"checkpoint identity verification failed: {type(error).__name__}: {error}"
                ) from error
            actual = artifact.identity
            if str(actual.get("checkpoint_sha256")) != sha:
                raise ValueGateError("checkpoint identity SHA256 differs from the artifact")
            if artifact.schema != EVALUATION_CHECKPOINT_SCHEMA:
                raise ValueGateError(
                    "value-gate identity requires an evaluation-only checkpoint "
                    f"({EVALUATION_CHECKPOINT_SCHEMA}); got {artifact.schema!r}"
                )
            if artifact.metadata.get("training_resume_allowed") is True:
                raise ValueGateError(
                    "value-gate identity checkpoint advertises training resume"
                )
            if artifact.backend != "cpp":
                raise ValueGateError(
                    f"value-gate identity requires the C++ backend; got {artifact.backend!r}"
                )
            if artifact.algorithm != "ippo":
                raise ValueGateError(
                    f"value-gate identity requires algorithm ippo; got {artifact.algorithm!r}"
                )
            if artifact.metadata.get("source_hash") in (None, ""):
                raise ValueGateError(
                    "value-gate identity checkpoint has no origin training source hash"
                )
            content_sha256 = _state_dict_weight_hash(artifact.model.state_dict())
            verified_identity = dict(actual)
            # The gate computes this fingerprint itself from the strictly
            # loaded state dict; a caller cannot invent it.
            verified_identity["state_dict_sha256"] = content_sha256
            verified_identity["origin_source_semantics"] = "checkpoint training source"
            verified.append(verified_identity)
            weight_hashes.append(content_sha256)
        else:
            all_verified = False
            problems.append(
                "checkpoint identity has no verifiable checkpoint path; the caller's "
                "identity cannot be trusted"
            )
            verified.append(dict(identity))
            weight_hashes.append(None)
    return verified, weight_hashes, all_verified, problems


def _slice_coverage_problems(
    metrics: Mapping[str, object],
    points: Sequence[RankingPoint],
    design: Mapping[str, object],
) -> tuple[list[str], dict[str, object], dict[str, object]]:
    problems: list[str] = []
    slices = dict(metrics.get("slices") or {})
    min_deals = int(dict(design["dataset"])["min_slice_deals"])
    required_groups = {
        "game_phase": ("early", "mid", "late"),
        "seat_team": ("team_0", "team_1"),
        "outcome": ("DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST"),
    }
    for family, groups in required_groups.items():
        family_rows = dict(slices.get(family) or {})
        for group in groups:
            row = family_rows.get(group)
            if row is None:
                problems.append(f"required slice {family}={group} is missing")
                continue
            deals = int(row.get("independent_deal_count", 0))
            if deals < min_deals:
                problems.append(
                    f"required slice {family}={group} has {deals} independent deals "
                    f"< min_slice_deals={min_deals}"
                )
            if row.get("ev") is None:
                problems.append(f"required slice {family}={group} EV is undefined")
            elif float(row["ev"]) <= 0.0:
                problems.append(
                    f"required slice {family}={group} EV {float(row['ev']):.6g} is not positive"
                )
    max_gap = float(dict(design["slices"])["max_seat_ev_gap_for_pass"])
    gaps: dict[str, object] = {}
    for family in ("seat_team", "level_rank", "leader_seat", "tribute_mode"):
        family_rows = dict(slices.get(family) or {})
        rows = [
            row for row in family_rows.values()
            if int(row.get("independent_deal_count", 0)) >= min_deals
            and row.get("ev") is not None
        ]
        context_undefined = False
        if family not in ("seat_team", "game_phase"):
            for name, row in sorted(family_rows.items()):
                deals = int(row.get("independent_deal_count", 0))
                if deals < min_deals:
                    continue
                if row.get("ev") is None:
                    context_undefined = True
                    problems.append(
                        f"context slice {family}={name} EV is undefined with {deals} "
                        "independent deals (zero target variance); no bias conclusion"
                    )
                elif float(row["ev"]) <= 0.0:
                    problems.append(f"context slice {family}={name} EV is not positive")
            if not rows and not context_undefined:
                problems.append(
                    f"context family {family} has no group with >= {min_deals} independent "
                    "deals and a defined EV"
                )
        values = [float(row["ev"]) for row in rows]
        gap = (
            "undefined" if context_undefined
            else (max(values) - min(values) if len(values) >= 2 else None)
        )
        gaps[f"{family}_ev_gap"] = gap
        if isinstance(gap, (int, float)) and float(gap) > max_gap:
            problems.append(
                f"{family} EV gap {float(gap):.6g} exceeds pre-registered {max_gap}"
            )
    mid = [
        point for point in points
        if point.game_phase == "mid" and point.informative
    ]
    late = [
        point for point in points
        if point.game_phase == "late" and point.informative
    ]
    all_target_tie = sum(1 for point in points if point.all_target_tie)
    if all_target_tie:
        problems.append(
            f"{all_target_tie} all-target-tie ranking points carry zero ranking information "
            "and cannot fill the informative threshold"
        )
    if not mid:
        problems.append("no informative mid-game ranking point")
    if not late:
        problems.append("no informative late-game ranking point")
    mid_late = mid + late
    mid_late_metrics = compute_ranking_metrics(
        mid_late,
        resamples=int(dict(design["metrics"]["bootstrap"])["resamples"]),
        seed=int(dict(design["metrics"]["bootstrap"])["seed"]),
    )
    if len(mid_late) < min_deals:
        problems.append(
            f"informative mid+late ranking points {len(mid_late)} < min_slice_deals={min_deals}"
        )
    if int(mid_late_metrics["informative_deals"]) < min_deals:
        problems.append(
            "informative mid+late ranking independent deals "
            f"{mid_late_metrics['informative_deals']} < min_slice_deals={min_deals}"
        )
    if any(not point.history_consistent for point in points):
        problems.append("a ranking point is not full-history consistent")
    ranking_gap_limit = float(dict(design["slices"])["max_seat_ranking_gap_for_pass"])
    by_team = dict(mid_late_metrics.get("by_team") or {})
    team_rows = [
        row for row in by_team.values() if int(row.get("informative_points", 0)) > 0
    ]
    if len(team_rows) < 2:
        problems.append("ranking points do not cover both seat teams")
    else:
        accuracies = [float(row["accuracy"]) for row in team_rows]
        ranking_gap = max(accuracies) - min(accuracies)
        gaps["seat_ranking_gap"] = ranking_gap
        if ranking_gap > ranking_gap_limit:
            problems.append(
                f"seat ranking gap {ranking_gap:.6g} exceeds pre-registered "
                f"{ranking_gap_limit}"
            )
    return problems, mid_late_metrics, gaps


def _result_pollution_problems(result: QualificationResult) -> list[str]:
    """Structural checks so duplicated/edited evidence cannot inflate a verdict."""
    problems: list[str] = []
    metadata = dict(result.metadata or {})
    samples = list(result.samples)
    points = list(result.ranking_points)
    if "sample_count" in metadata and int(metadata["sample_count"]) != len(samples):
        problems.append(
            "result.metadata.sample_count disagrees with the sample rows (polluted result)"
        )
    if "ranking_points" in metadata and int(metadata["ranking_points"]) != len(points):
        problems.append(
            "result.metadata.ranking_points disagrees with the ranking rows (polluted result)"
        )
    canonical = {sample.canonical_deal for sample in samples}
    if "independent_deals" in metadata and int(metadata["independent_deals"]) != len(canonical):
        problems.append(
            "result.metadata.independent_deals disagrees with the sample clusters "
            "(polluted result)"
        )
    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        problems.append("qualification sample IDs are duplicated (polluted result)")
    deal_to_canonical: dict[str, str] = {}
    for sample in samples:
        previous = deal_to_canonical.setdefault(sample.deal_id, sample.canonical_deal)
        if previous != sample.canonical_deal:
            problems.append(
                f"sample deal {sample.deal_id!r} maps to two canonical deals "
                "(polluted result)"
            )
            break
    for record in result.deal_records:
        if not isinstance(record, Mapping):
            problems.append("a deal record is malformed (polluted result)")
            continue
        deal_id = str(record.get("deal_id", ""))
        record_canonical = record.get("canonical_deal")
        if deal_id in deal_to_canonical and record_canonical is not None:
            if str(record_canonical) != deal_to_canonical[deal_id]:
                problems.append(
                    f"deal record {deal_id!r} canonical deal disagrees with its samples "
                    "(polluted result)"
                )
    point_keys: set[tuple[str, int, int]] = set()
    for point in points:
        key = (point.deal_id, int(point.token_step), int(point.player))
        if key in point_keys:
            problems.append(
                f"ranking point {key!r} is duplicated (polluted result)"
            )
            continue
        point_keys.add(key)
        if point.deal_id in deal_to_canonical and (
            point.canonical_deal != deal_to_canonical[point.deal_id]
        ):
            problems.append(
                f"ranking point deal {point.deal_id!r} canonical deal disagrees with "
                "its samples (polluted result)"
            )
        if len(point.candidate_token_sequences) not in (0, int(point.candidate_count)):
            problems.append(
                f"ranking point {key!r} candidate sequences disagree with candidate_count"
            )
        if int(point.candidate_count) != len(point.model_scores):
            problems.append(
                f"ranking point {key!r} candidate_count disagrees with model_scores"
            )
    return problems


def _qualification_binding_problems(
    result: QualificationResult,
    registration: Mapping[str, object],
    verified_identities: Sequence[Mapping[str, object]],
    *,
    identities_verified: bool,
) -> tuple[list[str], dict[str, object]]:
    """Bind evidence to distinct, separately meaningful hashes.

    The four identities are not interchangeable:

    * ``producer.source_hash`` identifies the qualification producer code and
      must equal the live producer fingerprint *and* the pre-registration.
    * ``registration.config_hash`` is the resolved qualification config frozen
      before results and must equal the verified evaluation config.
    * ``checkpoint.source_hash`` is the checkpoint's *training* origin and must
      equal the strictly loaded artifact, never the producer hash.
    * ``sample_target_contract_hash`` is the pre-registered target contract.
    """
    problems: list[str] = []
    evidence: dict[str, object] = {"identity_semantics": {
        "producer_source_hash": "phase-6 qualification producer code",
        "registration_config_hash": "resolved qualification config at pre-registration",
        "checkpoint_source_hash": "checkpoint origin training code",
        "sample_target_contract_hash": "pre-registered target perspective/scale/gamma",
    }}
    live_producer = qualification_producer_source_hash()
    evidence["live_producer_source_hash"] = live_producer
    producer = result.metadata.get("producer")
    if not isinstance(producer, Mapping):
        problems.append(
            "result.metadata.producer is missing; producer identity cannot be verified"
        )
        producer = {}
    else:
        if producer.get("schema") != VALUE_QUALIFICATION_PRODUCER_SCHEMA:
            problems.append("result.metadata.producer schema mismatch")
        if str(producer.get("source_hash")) != live_producer:
            problems.append(
                "result.metadata.producer.source_hash does not match the live qualification "
                "producer source"
            )
        if list(producer.get("source_scope") or ()) != list(QUALIFICATION_PRODUCER_SOURCE_FILES):
            problems.append("result.metadata.producer source scope is not the phase-6 scope")
        if str(producer.get("design_hash")) != phase6_design_hash():
            problems.append("result.metadata.producer design hash is not the frozen design")
    registration_source = str(registration.get("source_hash") or "")
    if not registration_source:
        problems.append(
            "pre-registration does not record a qualification producer source hash"
        )
    elif registration_source != live_producer:
        problems.append(
            "pre-registration producer source hash differs from the live producer; evidence "
            "produced by a different source cannot be bound to this registration"
        )
    registration_config = str(registration.get("config_hash") or "")
    if not registration_config:
        problems.append("pre-registration does not record a resolved config hash")
    expected_target = dict(VALUE_TARGET_CONTRACT)
    recorded_hash = result.metadata.get("sample_target_contract_hash")
    if not recorded_hash:
        problems.append(
            "result.metadata.sample_target_contract_hash is missing; the target contract "
            "is not bound to the pre-registration"
        )
    elif str(recorded_hash) != canonical_hash(expected_target):
        problems.append("result.metadata.sample_target_contract_hash is not the frozen contract")
    recorded_checkpoint = result.metadata.get("checkpoint")
    content_bound = False
    if isinstance(recorded_checkpoint, Mapping):
        for identity in verified_identities:
            if str(identity.get("checkpoint_sha256")) != str(
                recorded_checkpoint.get("checkpoint_sha256")
            ):
                continue
            origin_source = str(identity.get("source_hash") or "")
            recorded_origin = recorded_checkpoint.get("source_hash")
            if recorded_origin is None or str(recorded_origin) != origin_source:
                problems.append(
                    "result.metadata.checkpoint.source_hash does not match the checkpoint "
                    "origin training source (producer and checkpoint source are distinct)"
                )
            if registration_config and str(
                identity.get("resolved_config_hash")
            ) != registration_config:
                problems.append(
                    "pre-registered config hash does not match the verified checkpoint "
                    "resolved config"
                )
            recorded_content = recorded_checkpoint.get("state_dict_sha256")
            computed_content = identity.get("state_dict_sha256")
            if recorded_content is None:
                problems.append(
                    "result.metadata.checkpoint.state_dict_sha256 is missing; the state-dict "
                    "content fingerprint is not bound"
                )
            elif str(recorded_content) != str(computed_content):
                problems.append(
                    "result.metadata.checkpoint.state_dict_sha256 disagrees with the strictly "
                    "loaded state dict"
                )
            else:
                content_bound = True
            break
    for identity in verified_identities:
        if registration_config and str(identity.get("resolved_config_hash")) != registration_config:
            problems.append(
                "a verified checkpoint resolved config differs from the pre-registration"
            )
            break
    architectures = {str(identity.get("architecture")) for identity in verified_identities}
    protocols = {str(identity.get("protocol_version")) for identity in verified_identities}
    backends = {str(identity.get("backend")) for identity in verified_identities}
    if len(architectures) > 1:
        problems.append("verified checkpoints disagree on architecture")
    if len(protocols) > 1:
        problems.append("verified checkpoints disagree on protocol version")
    if backends - {"cpp"}:
        problems.append("verified checkpoints are not all on the C++ backend")
    evidence["registration_config_hash"] = registration_config
    evidence["content_fingerprint_bound"] = content_bound
    return problems, evidence


def evaluate_value_gate(
    result: QualificationResult,
    *,
    registration: Mapping[str, object],
    checkpoint_identities: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the pre-registered verdict rules; never edits a threshold.

    The first phase-6 reviewer found a false PASS built from 64 all-early
    deals where one team's target was constant.  The gate now enforces the
    frozen slice coverage, seat/context gaps, mid+late ranking coverage and
    checkpoint identity verification before any PASS can be considered.
    """
    design = VALUE_GATE_PLAN
    if registration.get("schema") != VALUE_REGISTRATION_SCHEMA:
        raise ValueGateError("value gate requires a valid pre-registration document")
    declared_registration = registration.get("registration_hash")
    if declared_registration != canonical_hash(
        {key: value for key, value in registration.items() if key != "registration_hash"}
    ):
        raise ValueGateError("value-gate registration hash is inconsistent with its contents")
    if str(registration.get("design_hash")) != phase6_design_hash():
        raise ValueGateError("value-gate registration belongs to a different frozen design")
    if dict(registration.get("value_gate_plan") or {}) != json.loads(json.dumps(design)):
        raise ValueGateError("value-gate registration thresholds were edited")
    target_contract = dict(result.metadata.get("target") or {})
    if set(target_contract) != _TARGET_CONTRACT_KEYS or any(
        not (
            target_contract.get(key) == expected if isinstance(expected, str)
            else math.isclose(
                float(target_contract.get(key, float("nan"))), float(expected),
                rel_tol=0.0, abs_tol=1e-12,
            )
        )
        for key, expected in VALUE_TARGET_CONTRACT.items()
    ):
        raise ValueGateError(
            "qualification targets use a scale/perspective/discount that is not the "
            "pre-registered acting-player rank-point/3 gamma=1 contract"
        )
    if result.metadata.get("clustering_unit") != (
        "canonical_physical_deal_rotations_and_deck_swap"
    ):
        raise ValueGateError("qualification samples are not clustered by canonical deal")
    for sample in result.samples:
        if not math.isfinite(sample.prediction) or not math.isfinite(sample.target):
            raise ValueGateError("a qualification sample has a non-finite prediction/target")
    (
        verified_identities,
        weight_hashes,
        identities_verified,
        identity_problems,
    ) = _verify_gate_checkpoint_identities(checkpoint_identities)
    verified_weights = [value for value in weight_hashes if value is not None]
    duplicate_weight_count = len(verified_weights) - len(set(verified_weights))
    if duplicate_weight_count > 0:
        identity_problems.append(
            "checkpoint identities share identical model weights; cross-checkpoint "
            "stability is not established"
        )
    distinct_checkpoints = len(set(verified_weights))
    # Never trust caller-supplied ranking metrics: recompute every point from
    # its raw finite arrays with the frozen tie-aware contract.
    points = [revalidate_ranking_point(point) for point in result.ranking_points]
    pollution_problems = _result_pollution_problems(result)
    # Bind the result to the pre-registration and to a verified checkpoint.
    binding_problems: list[str] = []
    recorded_registration = result.metadata.get("registration_hash")
    if recorded_registration is None:
        binding_problems.append(
            "result.metadata.registration_hash is missing; the evidence is not bound to "
            "the pre-registration"
        )
    elif str(recorded_registration) != str(registration["registration_hash"]):
        binding_problems.append(
            "result.metadata.registration_hash does not match the pre-registration"
        )
    recorded_design = result.metadata.get("design_hash")
    if recorded_design is None:
        binding_problems.append(
            "result.metadata.design_hash is missing; the evidence is not bound to the "
            "pre-registered design"
        )
    elif str(recorded_design) != str(registration["design_hash"]):
        binding_problems.append(
            "result.metadata.design_hash does not match the pre-registration design"
        )
    recorded_checkpoint = result.metadata.get("checkpoint")
    binding_keys = (
        "checkpoint_sha256", "resolved_config_hash", "architecture",
        "protocol_version", "backend",
    )
    if not isinstance(recorded_checkpoint, Mapping):
        binding_problems.append(
            "result.metadata.checkpoint is missing; evidence is not bound to a checkpoint"
        )
    else:
        matched = any(
            all(
                str(recorded_checkpoint.get(key)) == str(identity.get(key))
                for key in binding_keys
            )
            for identity in verified_identities
        )
        if not matched:
            binding_problems.append(
                "result.metadata.checkpoint does not match any verified checkpoint identity"
            )
    identity_binding_problems, identity_binding = _qualification_binding_problems(
        result, registration, verified_identities,
        identities_verified=identities_verified,
    )
    binding_problems.extend(identity_binding_problems)
    binding_problems.extend(pollution_problems)
    metrics = compute_value_metrics(result.samples)
    ranking = compute_ranking_metrics(
        points,
        resamples=int(design["metrics"]["bootstrap"]["resamples"]),
        seed=int(design["metrics"]["bootstrap"]["seed"]),
    )
    ev_ci = _cluster_bootstrap_ev(
        result.samples,
        resamples=int(design["metrics"]["bootstrap"]["resamples"]),
        seed=int(design["metrics"]["bootstrap"]["seed"]),
        confidence=float(design["metrics"]["bootstrap"]["confidence"]),
    )
    slice_problems, mid_late_ranking, slice_gaps = _slice_coverage_problems(
        metrics, points, design,
    )
    independent_deals = int(metrics["overall"]["independent_deal_count"])
    sufficiency = {
        "independent_deals": independent_deals,
        "min_independent_deals": int(design["dataset"]["min_independent_deals"]),
        "sample_count": len(result.samples),
        "min_samples": int(design["dataset"]["min_samples"]),
        "distinct_checkpoints": distinct_checkpoints,
        "distinct_checkpoint_weight_hashes": distinct_checkpoints,
        "duplicate_weight_identity_count": int(duplicate_weight_count),
        "min_distinct_checkpoints": int(
            design["checkpoint"]["distinct_checkpoints_required_for_pass"]
        ),
        "checkpoint_identities_verified": bool(identities_verified),
        "checkpoint_content_fingerprint_bound": bool(
            identity_binding.get("content_fingerprint_bound")
        ),
        "result_binding_ok": not binding_problems,
        "result_pollution_ok": not pollution_problems,
        "slice_coverage_ok": not slice_problems,
        "ranking_points": int(ranking["ranking_points"]),
        "informative_ranking_points": int(ranking["informative_points"]),
        "all_target_tie_ranking_points": int(ranking["all_target_tie_points"]),
        "model_tie_ranking_points": int(ranking["model_tie_points"]),
        "min_ranking_points": int(design["counterfactual"]["min_ranking_points"]),
        "ranking_deals": int(ranking["independent_deals"]),
        "informative_ranking_deals": int(ranking["informative_deals"]),
        "min_ranking_deals": int(design["counterfactual"]["min_ranking_deals"]),
        "mid_late_ranking_points": int(mid_late_ranking["ranking_points"]),
        "mid_late_informative_ranking_points": int(
            mid_late_ranking["informative_points"]
        ),
        "mid_late_ranking_deals": int(mid_late_ranking["independent_deals"]),
        "mid_late_informative_ranking_deals": int(
            mid_late_ranking["informative_deals"]
        ),
        "min_slice_deals": int(design["dataset"]["min_slice_deals"]),
        "restricted_ranking_points": int(ranking["restricted_points"]),
        "restricted_informative_ranking_points": int(
            ranking["restricted_informative_points"]
        ),
        "ranking_metric_version": VALUE_RANKING_METRIC_VERSION,
        "legacy_ranking_metric_values_ignored": int(
            ranking["legacy_metric_values_ignored"]
        ),
        "truncated": bool(result.metadata.get("truncated")),
        "identity_binding": dict(identity_binding),
    }
    significant = (
        independent_deals >= sufficiency["min_independent_deals"]
        and sufficiency["sample_count"] >= sufficiency["min_samples"]
        and distinct_checkpoints >= sufficiency["min_distinct_checkpoints"]
        and duplicate_weight_count == 0
        and identities_verified
        and sufficiency["checkpoint_content_fingerprint_bound"]
        and not binding_problems
        and not pollution_problems
        and sufficiency["slice_coverage_ok"]
        and sufficiency["informative_ranking_points"] >= sufficiency["min_ranking_points"]
        and sufficiency["informative_ranking_deals"] >= sufficiency["min_ranking_deals"]
        and sufficiency["restricted_informative_ranking_points"] == 0
        and not sufficiency["truncated"]
    )
    reasons: list[str] = (
        list(identity_problems) + list(binding_problems) + list(slice_problems)
    )
    ev = metrics["overall"]["ev"]
    divergence = int(metrics["overall"]["zero_target_variance_count"])
    target_undefined = ev is None or divergence > 0
    if divergence:
        reasons.append(
            f"zero_target_variance_count={divergence}: EV is undefined on those samples"
        )
    if ev is None:
        reasons.append("overall EV is undefined (zero target variance)")
    elif ev < float(design["metrics"]["ev_min"]):
        reasons.append(f"EV {ev:.6g} < pre-registered minimum {design['metrics']['ev_min']}")
    if ev_ci["low"] is None:
        reasons.append("EV clustered CI is undefined")
    elif float(ev_ci["low"]) < float(design["metrics"]["ev_ci95_low_min"]):
        reasons.append(
            f"EV clustered CI lower {ev_ci['low']:.6g} < "
            f"{design['metrics']['ev_ci95_low_min']}"
        )
    if ranking["delta"] is None:
        reasons.append("ranking delta is undefined")
    elif ranking["ci"]["low"] is None or float(ranking["ci"]["low"]) <= float(
        design["counterfactual"]["ranking_ci95_low_min"]
    ):
        reasons.append("ranking delta clustered CI does not establish a positive effect")
    if sufficiency["informative_ranking_points"] < sufficiency["min_ranking_points"]:
        reasons.append(
            f"informative ranking points {sufficiency['informative_ranking_points']} < "
            f"min_ranking_points={sufficiency['min_ranking_points']} "
            f"(all-target-tie points {sufficiency['all_target_tie_ranking_points']} are "
            "not informative)"
        )
    if sufficiency["informative_ranking_deals"] < sufficiency["min_ranking_deals"]:
        reasons.append(
            f"informative ranking deals {sufficiency['informative_ranking_deals']} < "
            f"min_ranking_deals={sufficiency['min_ranking_deals']}"
        )
    if target_undefined:
        # The first implementation fell through to REJECT here; undefined target
        # variance is missing evidence, never a scientific rejection.
        verdict = "INCONCLUSIVE"
    elif not significant:
        reasons.append("sufficiency/coverage thresholds were not met")
        verdict = "INCONCLUSIVE"
    else:
        pass_conditions = (
            ev is not None
            and ev >= float(design["metrics"]["ev_min"])
            and ev_ci["low"] is not None
            and float(ev_ci["low"]) >= float(design["metrics"]["ev_ci95_low_min"])
            and metrics["overall"]["pearson"] is not None
            and metrics["overall"]["pearson"] >= float(design["metrics"]["pearson_min"])
            and metrics["overall"]["spearman"] is not None
            and metrics["overall"]["spearman"] >= float(design["metrics"]["spearman_min"])
            and metrics["calibration_ece"] is not None
            and metrics["calibration_ece"] <= float(design["metrics"]["max_calibration_ece"])
            and mid_late_ranking["delta"] is not None
            and float(mid_late_ranking["delta"]) >= float(
                design["counterfactual"]["min_effect"]
            )
            and mid_late_ranking["ci"]["low"] is not None
            and float(mid_late_ranking["ci"]["low"]) > float(
                design["counterfactual"]["ranking_ci95_low_min"]
            )
        )
        if pass_conditions:
            verdict = "PASS"
        else:
            ev_upper = ev_ci["high"]
            rank_upper = mid_late_ranking["ci"]["high"]
            reject_conditions = (
                (ev_upper is not None and float(ev_upper) < float(design["metrics"]["ev_min"]))
                or (rank_upper is not None and float(rank_upper) <= 0.0)
            )
            verdict = "REJECT" if reject_conditions else "INCONCLUSIVE"
    search_status = search_status_for_verdict(verdict)
    payload: dict[str, object] = {
        "schema": VALUE_GATE_SCHEMA,
        "verdict": verdict,
        "reasons": reasons,
        "registration_hash": registration["registration_hash"],
        "design_hash": registration["design_hash"],
        "design_schema": PHASE6_DESIGN_SCHEMA,
        "checkpoint_identities": verified_identities,
        "dataset": {
            "schema": VALUE_DATASET_SCHEMA,
            "domain": "qualification",
            "deal_records": list(result.deal_records),
            "metadata": dict(result.metadata),
        },
        "metrics": metrics,
        "ev_clustered_ci": ev_ci,
        "ranking": {**ranking, "mid_late": mid_late_ranking, "slice_gaps": slice_gaps},
        "sufficiency": sufficiency,
        "search_status": search_status,
        "created_at": _timestamp(),
    }
    payload["gate_hash"] = canonical_hash(payload)
    return payload

# ---------------------------------------------------------------------------
# Search gate
# ---------------------------------------------------------------------------
_GATE_KEYS = frozenset({
    "schema", "verdict", "reasons", "registration_hash", "design_hash",
    "design_schema", "checkpoint_identities", "dataset", "metrics",
    "ev_clustered_ci", "ranking", "sufficiency", "search_status", "created_at",
    "gate_hash",
})


def load_value_gate(path: str | Path, *, require_pass: bool = True) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise ValueGateError(f"value gate artifact is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueGateError(f"value gate artifact is not readable JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueGateError("value gate artifact must be a JSON object")
    unknown = sorted(set(payload) - _GATE_KEYS)
    if unknown:
        raise ValueGateError(f"value gate artifact has unknown keys: {unknown}")
    missing = sorted(_GATE_KEYS - set(payload))
    if missing:
        raise ValueGateError(f"value gate artifact is missing keys: {missing}")
    if payload.get("schema") != VALUE_GATE_SCHEMA:
        raise ValueGateError("value gate schema mismatch")
    declared = payload.get("gate_hash")
    if declared != canonical_hash(
        {key: value for key, value in payload.items() if key != "gate_hash"}
    ):
        raise ValueGateError("value gate hash is inconsistent with its contents")
    if payload.get("verdict") not in ("PASS", "REJECT", "INCONCLUSIVE"):
        raise ValueGateError("value gate verdict is invalid")
    if str(payload.get("design_hash")) != phase6_design_hash():
        raise ValueGateError("value gate was produced under a different frozen design")
    if payload.get("design_schema") != PHASE6_DESIGN_SCHEMA:
        raise ValueGateError("value gate design schema mismatch")
    if not isinstance(payload.get("sufficiency"), Mapping):
        raise ValueGateError("value gate sufficiency block is malformed")
    expected_status = search_status_for_verdict(str(payload["verdict"]))
    if payload.get("search_status") != expected_status:
        raise ValueGateError("value gate search status disagrees with its verdict")
    if require_pass and payload["verdict"] != "PASS":
        raise ValueGateError(
            f"search requires a PASS value gate, got {payload['verdict']!r} "
            f"(search_status={payload.get('search_status')!r}, reasons={payload.get('reasons')!r})"
        )
    return dict(payload)


def assert_search_allowed(path: str | Path) -> dict[str, object]:
    """Fail closed unless search is implemented and a sealed PASS gate exists."""
    if not SEARCH_IMPLEMENTED:
        raise ValueGateError(
            "deployment search is not implemented in phase 6; a PASS artifact cannot enable it"
        )
    gate = load_value_gate(path, require_pass=True)
    if gate.get("search_status") != SEARCH_STATUS_IMPLEMENTED:
        raise ValueGateError("search_status is not IMPLEMENTED")
    return gate


__all__ = [
    "MODEL_SCORE_TIE_ATOL", "QUALIFICATION_PRODUCER_SOURCE_FILES",
    "REALIZED_RETURN_TIE_ATOL", "QualificationDeal", "QualificationResult",
    "RankingPoint", "VALUE_QUALIFICATION_DATA_SCHEMA", "VALUE_QUALIFICATION_PRODUCER_SCHEMA",
    "VALUE_RANKING_LEGACY_POINT_SCHEMAS", "VALUE_RANKING_METRIC_SCHEMA",
    "VALUE_RANKING_METRIC_VERSION", "VALUE_RANKING_POINT_SCHEMA", "VALUE_TARGET_CONTRACT",
    "ValueGateError", "ValueSample", "assert_qualification_isolation",
    "assert_search_allowed", "build_qualification_deals", "canonical_physical_deal",
    "compute_ranking_metrics", "compute_tie_aware_ranking", "compute_value_metrics",
    "enumerate_complete_macros", "evaluate_ranking_point", "evaluate_value_gate",
    "load_value_gate", "load_value_gate_registration",
    "qualification_producer_source_hash", "revalidate_ranking_point",
    "run_value_qualification", "write_value_gate_registration",
]
