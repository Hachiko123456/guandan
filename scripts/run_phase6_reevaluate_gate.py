# SPDX-License-Identifier: MIT
"""Re-evaluate saved phase-6 qualification samples without recollecting.

    D:/project/guandan/.venv-gpu/Scripts/python.exe scripts/run_phase6_reevaluate_gate.py \
      --samples runs/phase6_repair1/value_gate/samples.json \
      --registration runs/phase6_repair1/value_gate/registration.json \
      --checkpoint runs/phase6_repair1/value_gate/pilot-eval.pt \
      --out runs/phase6_closeout/value_gate/qualification_reevaluated.json

The saved evidence is never modified.  Ranking points are *recomputed* from
their raw ``model_scores``/``realized_returns`` with the frozen tie-aware
contract; the legacy serialized ``accuracy``/``random_baseline``/``delta`` are
never trusted and their difference is written to the evidence file.  The
producer source of the historical collection is not invented: the gate will
report it as unverifiable and keep the verdict INCONCLUSIVE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.evaluation.checkpoints import load_v2_policy  # noqa: E402
from guandan_marvel.phase6.value_gate import (  # noqa: E402
    VALUE_RANKING_METRIC_VERSION, VALUE_TARGET_CONTRACT, QualificationResult,
    RankingPoint, ValueSample, _state_dict_weight_hash, canonical_hash,
    evaluate_value_gate, load_value_gate_registration, revalidate_ranking_point,
)

SAMPLE_FIELDS = {
    "sample_id", "deal_id", "canonical_deal", "token_step", "num_steps",
    "player", "team", "level_rank", "leader_seat", "tribute_mode",
    "prediction", "target", "game_phase", "outcome",
}
POINT_FIELDS = {
    "deal_id", "canonical_deal", "token_step", "num_steps", "game_phase",
    "history_consistent", "player", "candidate_count", "restricted",
    "model_top_index", "realized_top_index", "accuracy", "random_baseline",
    "delta", "informative", "all_target_tie", "model_tie", "target_top_count",
    "model_top_count", "model_scores", "realized_returns",
    "candidate_token_sequences",
}


def _legacy_point(row: dict[str, object]) -> RankingPoint:
    fields = {key: value for key, value in row.items() if key in POINT_FIELDS}
    fields["model_scores"] = tuple(float(value) for value in row.get("model_scores", ()))
    fields["realized_returns"] = tuple(
        float(value) for value in row.get("realized_returns", ())
    )
    fields["candidate_token_sequences"] = tuple(
        tuple(tuple(int(value) for value in token_row) for token_row in candidate)
        for candidate in row.get("candidate_token_sequences", ())
    )
    # Old v2 rows carry a schema and no metric_version.  Marking the version
    # explicitly makes the gate recompute rather than trust them.
    fields["metric_version"] = str(
        row.get("metric_version") or row.get("schema") or ""
    )
    if "delta" not in row:
        # The v2 schema serialized accuracy and baseline but not delta.
        fields["delta"] = float(fields.get("accuracy", 0.0)) - float(
            fields.get("random_baseline", 0.0)
        )
    return RankingPoint(**fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    samples = [
        ValueSample(**{key: value for key, value in row.items() if key in SAMPLE_FIELDS})
        for row in payload["samples"]
    ]
    legacy_points = [_legacy_point(row) for row in payload.get("ranking_points", [])]
    points = [revalidate_ranking_point(point) for point in legacy_points]
    comparisons = []
    for legacy, fixed in zip(legacy_points, points):
        comparisons.append({
            "deal_id": legacy.deal_id,
            "token_step": int(legacy.token_step),
            "candidate_count": int(legacy.candidate_count),
            "legacy_serialized": {
                "accuracy": float(legacy.accuracy),
                "random_baseline": float(legacy.random_baseline),
                "delta": float(legacy.delta),
            },
            "tie_aware_recomputed": {
                "accuracy": float(fixed.accuracy),
                "random_baseline": float(fixed.random_baseline),
                "delta": float(fixed.delta),
                "informative": bool(fixed.informative),
                "all_target_tie": bool(fixed.all_target_tie),
                "model_tie": bool(fixed.model_tie),
                "target_top_count": int(fixed.target_top_count),
                "model_top_count": int(fixed.model_top_count),
            },
            "legacy_metric_values_ignored": bool(fixed.legacy_metrics_ignored),
        })
    metadata = dict(payload["metadata"])
    registration = load_value_gate_registration(args.registration)
    artifact = load_v2_policy(args.checkpoint, device="cpu", expected_algorithm="ippo")
    metadata["registration_hash"] = registration["registration_hash"]
    metadata["design_hash"] = registration["design_hash"]
    checkpoint_metadata = dict(artifact.identity)
    # Verified by the gate again from the strictly loaded artifact; not invented.
    checkpoint_metadata["state_dict_sha256"] = _state_dict_weight_hash(
        artifact.model.state_dict()
    )
    metadata["checkpoint"] = checkpoint_metadata
    metadata["sample_target_contract_hash"] = canonical_hash(VALUE_TARGET_CONTRACT)
    metadata["ranking_metric_version"] = VALUE_RANKING_METRIC_VERSION
    metadata["ranking_informative_points"] = sum(
        1 for point in points if point.informative
    )
    metadata["ranking_all_target_tie_points"] = sum(
        1 for point in points if point.all_target_tie
    )
    metadata["ranking_model_tie_points"] = sum(1 for point in points if point.model_tie)
    metadata["ranking_legacy_metric_values_ignored"] = sum(
        1 for point in points if point.legacy_metrics_ignored
    )
    result = QualificationResult(
        samples=samples, ranking_points=points,
        deal_records=list(payload.get("deal_records", [])), metadata=metadata,
    )
    gate = evaluate_value_gate(
        result, registration=registration,
        checkpoint_identities=[artifact.identity],
    )
    evidence = {
        "schema": "guandan-marvel-phase6-reevaluation-evidence-v1",
        "source": str(Path(args.samples).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "registration": str(Path(args.registration).resolve()),
        "recollected": False,
        "ranking_metric_version": VALUE_RANKING_METRIC_VERSION,
        "legacy_metric_values_ignored": sum(
            1 for row in comparisons if row["legacy_metric_values_ignored"]
        ),
        "point_comparisons": comparisons,
        "producer_source_note": (
            "historical samples were produced by an earlier source; no producer identity "
            "is asserted here, so the gate reports it as unbound"
        ),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    (destination.parent / "reevaluation_evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    destination.write_text(
        json.dumps(gate, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "VALUE_GATE": gate["verdict"],
        "SEARCH_STATUS": gate["search_status"],
        "ranking_metric_version": VALUE_RANKING_METRIC_VERSION,
        "legacy_metric_values_ignored": evidence["legacy_metric_values_ignored"],
        "reasons": gate["reasons"],
        "output": str(destination),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
