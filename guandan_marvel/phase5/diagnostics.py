# SPDX-License-Identifier: MIT
"""Aligned policy-symmetry diagnostics (spec 3.2).

The comparison is always

.. code-block:: text

    pi(P_T(a) | T(s))  vs  pi(a | s)
    q_T(a | s) = pi(P_T(a) | T(s))        (mapped back to the original order)
    L_sym(s)   = sum_{a in A(s)} pi(a|s) * [log pi(a|s) - log q_T(a|s)]

Action unit is declared explicitly:

* ``full_macro``: the complete legal macro actions of the state were
  enumerated, so the per-token KL is the *complete-action* KL under the
  independent-per-token policy model (the exact sequence log-probability of a
  legal path is the sum of its token log-probabilities, and every legal macro
  path is enumerated, so no conditioning is involved).
* ``restricted_candidate``: only the next-token candidate distribution is
  compared; this is *not* a full-action KL and is never reported as one.

Only legal, active candidates participate.  Padding, inactive and illegal
candidates are masked out.  Log-probabilities are taken from the FP32
``log_softmax`` of masked logits (numerically stable); a zero probability on a
legal candidate is treated as ``log p = -inf`` and is reported through an
explicit ``zero_probability_count`` instead of being clamped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SYMMETRY_DIAGNOSTIC_SCHEMA = "guandan-marvel-phase5-symmetry-diagnostic-v1"

ACTION_UNIT_FULL_MACRO = "full_macro"
ACTION_UNIT_RESTRICTED = "restricted_candidate"


@dataclass(frozen=True, slots=True)
class AlignedComparison:
    """One aligned ``pi(.|s)`` vs ``q_T(.|s)`` comparison."""

    action_unit: str
    kl: float
    max_prob_diff: float
    mean_prob_diff: float
    argmax_agreement: int
    candidate_count: int
    zero_probability_count: int
    max_prob_diff_action: int | None = None
    # Distribution diagnostics are reported but never used for the gate.
    probabilities: np.ndarray | None = field(default=None, repr=False)
    transformed: np.ndarray | None = field(default=None, repr=False)


def _as_probabilities(log_probs: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_probs.to(torch.float64))


def aligned_policy_comparison(
    log_probs: torch.Tensor,
    transformed_log_probs: torch.Tensor,
    permutation: Sequence[int],
    *,
    legal: torch.Tensor | None = None,
    action_unit: str = ACTION_UNIT_RESTRICTED,
    tie_rule_index: Sequence[int] | None = None,
) -> AlignedComparison:
    """Compare ``pi(.|s)`` with ``q_T(.|s) = pi(P_T(.)|T(s))``.

    ``log_probs`` / ``transformed_log_probs`` are ``[A]`` FP32 log-softmax
    vectors in their own candidate order; ``permutation[i]`` is the index of
    ``P_T(a_i)`` inside the transformed list.  ``legal`` selects the candidates
    that participate (default: all).
    """
    if action_unit not in (ACTION_UNIT_FULL_MACRO, ACTION_UNIT_RESTRICTED):
        raise ValueError(f"unknown action unit {action_unit!r}")
    base = torch.as_tensor(log_probs, dtype=torch.float64).reshape(-1)
    other = torch.as_tensor(transformed_log_probs, dtype=torch.float64).reshape(-1)
    perm = [int(index) for index in permutation]
    if len(perm) != base.numel():
        raise ValueError("permutation length must equal the candidate count")
    if any(index < 0 or index >= other.numel() for index in perm):
        raise ValueError("permutation index is outside the transformed candidate list")
    if len(set(perm)) != len(perm):
        raise ValueError("permutation must be injective")
    if legal is None:
        mask = torch.ones(base.numel(), dtype=torch.bool)
    else:
        mask = torch.as_tensor(legal, dtype=torch.bool).reshape(-1)
        if mask.numel() != base.numel():
            raise ValueError("legal mask must match the candidate count")
    indices = torch.nonzero(mask).reshape(-1).tolist()
    if not indices:
        raise ValueError("no legal candidate participates in the comparison")
    selected = base[torch.as_tensor(indices)]
    mapped = other[torch.as_tensor([perm[index] for index in indices])]
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("legal candidate carries a non-finite log-probability")
    probabilities = _as_probabilities(selected)
    mapped_probabilities = _as_probabilities(mapped)
    if not bool(torch.isfinite(probabilities).all()) or not bool(torch.isfinite(mapped_probabilities).all()):
        raise ValueError("candidate probability is non-finite")
    zero_probability_count = int((probabilities <= 0).sum().item())
    difference = (probabilities - mapped_probabilities).abs()
    if zero_probability_count:
        # A legal candidate with p == 0 contributes log 0 = -inf only where it
        # also carries zero *weight* in the expectation.  If the transformed
        # distribution gives it zero probability instead, the KL is +inf and is
        # reported as such rather than clamped.
        weighted = probabilities * (selected - mapped)
        finite = torch.isfinite(weighted)
        kl = float(weighted[finite].sum().item())
        infinite = bool((~finite & (probabilities > 0)).any().item())
        kl = float("inf") if infinite else kl
    else:
        kl = float((probabilities * (selected - mapped)).sum().item())
    order = indices
    if tie_rule_index is not None:
        rank_order = [int(index) for index in tie_rule_index]
        if len(rank_order) != base.numel():
            raise ValueError("tie-rule order must cover every candidate")
        positions = {index: position for position, index in enumerate(rank_order)}
        order = sorted(indices, key=lambda index: positions[index])
    base_argmax = _argmax_with_tie_rule(order, lambda index: float(base[index].item()))
    mapped_scores = {index: float(other[perm[index]].item()) for index in order}
    mapped_argmax = _argmax_with_tie_rule(order, lambda index: mapped_scores[index])
    argmax_agreement = int(mapped_argmax == base_argmax)
    max_position = int(torch.argmax(difference).item())
    return AlignedComparison(
        action_unit=action_unit,
        kl=kl,
        max_prob_diff=float(difference.max().item()),
        mean_prob_diff=float(difference.mean().item()),
        argmax_agreement=argmax_agreement,
        candidate_count=len(indices),
        zero_probability_count=zero_probability_count,
        max_prob_diff_action=int(indices[max_position]),
        probabilities=probabilities.cpu().numpy().copy(),
        transformed=mapped_probabilities.cpu().numpy().copy(),
    )


def _argmax_with_tie_rule(order: Sequence[int], score) -> int:
    """Lowest-position argmax under the declared tie rule (never NaN-driven)."""
    best = order[0]
    best_score = score(best)
    for index in order[1:]:
        value = score(index)
        if value > best_score:
            best, best_score = index, value
    return best


def positions_rank(order: Sequence[int], index: int) -> int:
    """Position of ``index`` inside ``order`` (deterministic tie resolution)."""
    for position, candidate in enumerate(order):
        if candidate == index:
            return position
    raise ValueError("index is absent from the tie-rule order")


def restricted_candidate_kl(
    log_probs: torch.Tensor,
    transformed_log_probs: torch.Tensor,
    permutation: Sequence[int],
    *,
    legal: torch.Tensor | None = None,
) -> float:
    """KL of the *restricted* next-token candidate distribution (never full-action)."""
    return aligned_policy_comparison(
        log_probs, transformed_log_probs, permutation, legal=legal,
        action_unit=ACTION_UNIT_RESTRICTED,
    ).kl


def bucket_key(
    *,
    phase: str,
    action_kind: str,
    level_rank: int,
    leader_seat: int,
    tribute_mode: str,
) -> dict[str, str]:
    """Stable bucket label set used for every diagnostic table."""
    return {
        "phase": str(phase),
        "action_kind": str(action_kind),
        "level_rank": f"level_{int(level_rank)}",
        "leader_seat": f"leader_{int(leader_seat)}",
        "tribute_mode": str(tribute_mode),
    }


def aggregate_diagnostic(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-macro comparisons into the pre-registered decision inputs."""
    if not records:
        raise ValueError("diagnostic aggregation requires at least one record")
    by_bucket: dict[str, dict[str, list[float]]] = {}
    kls, finite_kls, agreements, diffs, maxima = [], [], [], [], []
    full_macro_count = 0
    non_finite_count = 0
    deals: set[str] = set()
    for record in records:
        kl = float(record["kl"])
        kls.append(kl)
        if np.isfinite(kl):
            finite_kls.append(kl)
        else:
            non_finite_count += 1
        agreements.append(int(record["argmax_agreement"]))
        diffs.append(float(record["mean_prob_diff"]))
        maxima.append(float(record["max_prob_diff"]))
        if str(record["action_unit"]) == ACTION_UNIT_FULL_MACRO:
            full_macro_count += 1
        deals.add(str(record["deal_id"]))
        for name, value in dict(record["bucket"]).items():
            bucket = by_bucket.setdefault(name, {})
            # count, finite KL sum, finite KL count, argmax agreement sum,
            # non-finite KL count
            entry = bucket.setdefault(str(value), [0, 0.0, 0, 0, 0])
            entry[0] += 1
            if np.isfinite(kl):
                entry[1] += kl
                entry[2] += 1
            else:
                entry[4] += 1
            entry[3] += int(record["argmax_agreement"])
    buckets = {
        name: {
            key: {
                "count": entry[0],
                "mean_kl": (entry[1] / entry[2]) if entry[2] else float("inf"),
                "argmax_agreement_rate": entry[3] / entry[0],
                "non_finite_kl_count": entry[4],
            }
            for key, entry in sorted(values.items())
        }
        for name, values in sorted(by_bucket.items())
    }
    finite_agreements = [agreements[i] for i, kl in enumerate(kls) if np.isfinite(kl)]
    finite_diffs = [diffs[i] for i, kl in enumerate(kls) if np.isfinite(kl)]
    finite_maxima = [maxima[i] for i, kl in enumerate(kls) if np.isfinite(kl)]
    return {
        "schema": SYMMETRY_DIAGNOSTIC_SCHEMA,
        "sample_count": len(records),
        "deal_count": len(deals),
        "deals": sorted(deals),
        "finite_sample_count": len(finite_kls),
        "non_finite_kl_count": non_finite_count,
        "full_macro_sample_count": full_macro_count,
        "restricted_candidate_sample_count": len(records) - full_macro_count,
        "mean_kl": float(np.mean(finite_kls)) if finite_kls else float("inf"),
        "median_kl": float(np.median(finite_kls)) if finite_kls else float("inf"),
        "max_kl": float(np.max(finite_kls)) if finite_kls else float("inf"),
        "mean_max_prob_diff": float(np.mean(finite_maxima)) if finite_maxima else float("inf"),
        "mean_prob_diff": float(np.mean(finite_diffs)) if finite_diffs else float("inf"),
        "argmax_agreement_rate": (
            float(np.mean(finite_agreements)) if finite_agreements else float("inf")
        ),
        "argmax_disagreement_rate": (
            1.0 - float(np.mean(finite_agreements)) if finite_agreements else float("inf")
        ),
        "buckets": buckets,
    }


def decide_augmentation(
    aggregate: Mapping[str, Any],
    *,
    significance: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered significance rule; never train on a whim."""
    samples = int(aggregate["sample_count"])
    deals = int(aggregate["deal_count"])
    non_finite = int(aggregate.get("non_finite_kl_count", 0))
    mean_kl = float(aggregate["mean_kl"])
    disagreement = float(aggregate["argmax_disagreement_rate"])
    enough_samples = samples >= int(significance["min_macro_samples"])
    enough_deals = deals >= int(significance["min_deals"])
    kl_trigger = bool(np.isfinite(mean_kl)) and mean_kl > float(
        significance["mean_restricted_kl_threshold"]
    )
    argmax_trigger = bool(np.isfinite(disagreement)) and disagreement > float(
        significance["argmax_disagreement_threshold"]
    )
    significant = bool(enough_samples and enough_deals and (kl_trigger or argmax_trigger))
    reasons: list[str] = []
    if non_finite:
        reasons.append("non_finite_kl")
    if not enough_samples:
        reasons.append("insufficient_macro_samples")
    if not enough_deals:
        reasons.append("insufficient_deals")
    if not (kl_trigger or argmax_trigger):
        reasons.append("below_pre_registered_threshold")
    if non_finite:
        status = "INCONCLUSIVE"
    elif not enough_samples or not enough_deals:
        status = "INCONCLUSIVE"
    elif significant:
        status = "SIGNIFICANT"
    else:
        status = "NOT_SIGNIFICANT"
    return {
        "schema": SYMMETRY_DIAGNOSTIC_SCHEMA,
        "status": status,
        "augmentation_recommended": status == "SIGNIFICANT",
        "reasons": sorted(set(reasons)),
        "sample_count": samples,
        "deal_count": deals,
        "non_finite_kl_count": non_finite,
        "mean_kl": mean_kl,
        "argmax_disagreement_rate": disagreement,
        "thresholds": {
            "min_macro_samples": int(significance["min_macro_samples"]),
            "min_deals": int(significance["min_deals"]),
            "mean_restricted_kl_threshold": float(significance["mean_restricted_kl_threshold"]),
            "argmax_disagreement_threshold": float(significance["argmax_disagreement_threshold"]),
        },
        "triggered_by": {
            "sample_count": enough_samples,
            "deal_count": enough_deals,
            "mean_kl": kl_trigger,
            "argmax_disagreement": argmax_trigger,
        },
    }


__all__ = [
    "ACTION_UNIT_FULL_MACRO", "ACTION_UNIT_RESTRICTED", "AlignedComparison",
    "SYMMETRY_DIAGNOSTIC_SCHEMA", "aggregate_diagnostic", "aligned_policy_comparison",
    "bucket_key", "decide_augmentation", "positions_rank", "restricted_candidate_kl",
]
