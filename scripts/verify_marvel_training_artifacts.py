"""Verify M03 checkpoints, resume categories, and smoke summary."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from guandan_marvel.training.checkpoints import (
    CheckpointCompatibilityError, TRAINING_CHECKPOINT_SCHEMA,
    canonical_json_hash, classify_resume, validate_declared_hash,
    validate_source_fingerprint,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_state_dict(state: dict) -> bool:
    return all(not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all()) for value in state.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    algorithms = {}
    for algorithm in ("ippo", "vrpo"):
        record = summary[algorithm]
        checkpoints = []
        for expected in record["checkpoints"]:
            path = Path(expected["path"])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema") != TRAINING_CHECKPOINT_SCHEMA:
                raise AssertionError(f"{path} schema mismatch")
            if payload.get("algorithm") != algorithm:
                raise AssertionError(f"{path} algorithm mismatch")
            if sha256(path) != expected["sha256"]:
                raise AssertionError(f"{path} hash mismatch")
            required = {
                "semantic_contract", "source_hash", "resolved_config",
                "resolved_config_hash", "rng_state", "sampler_state",
                "schedule_state", "merged_policy", "counters",
                "agent_resume_state_dict", "agent_state_dict",
                "agent_optimizer_state_dict", "selection_identity",
                "champion_identity",
            }
            if algorithm == "vrpo":
                required.update({
                    "critic_state_dict", "critic_optimizer_state_dict",
                    "replay_buffer_state", "q_target_contract",
                })
            missing = sorted(required - set(payload))
            if missing:
                raise AssertionError(f"{path} missing {missing}")
            if not finite_state_dict(payload["agent_resume_state_dict"]):
                raise AssertionError(f"{path} non-finite agent state")
            if algorithm == "vrpo" and not finite_state_dict(payload["critic_state_dict"]):
                raise AssertionError(f"{path} non-finite critic state")
            if payload["selection_identity"] is not None or payload["champion_identity"] is not None:
                raise AssertionError("M03 checkpoint must not invent selection/champion identity")
            checkpoints.append({
                "path": str(path), "sha256": expected["sha256"],
                "bytes": path.stat().st_size, "iteration": int(payload["iteration"]),
                "durable_updates": int(payload["durable_updates"]),
                "resume_classification": payload["resume_classification"],
                "merged_policy": payload["merged_policy"],
                "semantic_contract_hash": payload["semantic_contract"]["hash"],
                "buffer_schema": payload["semantic_contract"]["buffer_schema"],
                "sampler_contract": payload["semantic_contract"]["sampler_contract"],
                "replay_persistence": payload.get("replay_persistence"),
            })
        final_payload = torch.load(Path(record["checkpoints"][-1]["path"]), map_location="cpu", weights_only=False)
        contract = final_payload["semantic_contract"]
        validate_declared_hash(final_payload["semantic_contract"], name="semantic_contract")
        validate_declared_hash(final_payload["semantic_contract"]["semantic_config"], name="semantic_config")
        validate_source_fingerprint(final_payload["source_fingerprint"], name="source_fingerprint")
        validate_declared_hash(final_payload["execution_fingerprint"], name="execution_fingerprint")
        exact = classify_resume(
            final_payload,
            expected_contract=contract,
            expected_source_fingerprint=final_payload["source_fingerprint"],
            expected_execution_fingerprint=final_payload["execution_fingerprint"],
            expected_config=final_payload["resolved_config"],
            require_replay=algorithm == "vrpo",
        )
        compatible_config = copy.deepcopy(final_payload["resolved_config"])
        compatible_config["logging_dir"] = "artifact-verifier-different-log-dir"
        compatible_config["run_name"] = "artifact-verifier-different-run-name"
        compatible = classify_resume(
            final_payload,
            expected_contract=contract,
            expected_source_fingerprint=final_payload["source_fingerprint"],
            expected_execution_fingerprint=final_payload["execution_fingerprint"],
            expected_config=compatible_config,
            require_replay=algorithm == "vrpo",
        )
        migration_contract = copy.deepcopy(contract)
        migration_semantic = migration_contract["semantic_config"]
        migration_semantic["discount_gamma"] = 0.5
        migration_semantic["hash"] = canonical_json_hash({
            key: value for key, value in migration_semantic.items() if key != "hash"
        })
        migration_contract["hash"] = canonical_json_hash({
            key: value for key, value in migration_contract.items() if key != "hash"
        })
        migration = classify_resume(
            final_payload,
            expected_contract=migration_contract,
            expected_source_fingerprint=final_payload["source_fingerprint"],
            expected_execution_fingerprint=final_payload["execution_fingerprint"],
            expected_config=final_payload["resolved_config"],
            require_replay=algorithm == "vrpo",
        )
        warm_kind = None
        if algorithm == "vrpo":
            warm_payload = dict(final_payload)
            warm_payload.pop("replay_buffer_state")
            warm_kind = classify_resume(
                warm_payload,
                expected_contract=contract,
                expected_source_fingerprint=final_payload["source_fingerprint"],
                expected_execution_fingerprint=final_payload["execution_fingerprint"],
                expected_config=final_payload["resolved_config"],
                require_replay=True,
            ).kind
        evaluation = classify_resume(
            final_payload,
            expected_contract=contract,
            expected_source_fingerprint=final_payload["source_fingerprint"],
            expected_execution_fingerprint=final_payload["execution_fingerprint"],
            expected_config=final_payload["resolved_config"],
            require_replay=algorithm == "vrpo", for_evaluation=True,
        )
        legacy_rejected = False
        try:
            classify_resume(
                {"protocol_version": "GD-ACTION-0.1|GD-ENV-0.1"},
                expected_contract=contract,
                expected_source_fingerprint=final_payload["source_fingerprint"],
                expected_execution_fingerprint=final_payload["execution_fingerprint"],
                expected_config=final_payload["resolved_config"],
                require_replay=False,
            )
        except CheckpointCompatibilityError:
            legacy_rejected = True
        algorithms[algorithm] = {
            "checkpoints": checkpoints,
            "classification": {
                "exact": exact.kind,
                "compatible": compatible.kind,
                "warm_restart": warm_kind,
                "migration": migration.kind,
                "evaluation": evaluation.kind,
                "legacy_v1_training_resume_rejected": legacy_rejected,
            },
        }
    report = {
        "schema": "guandan-marvel-m03-artifact-verification-v1",
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "algorithms": algorithms,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        algorithm: data["classification"] for algorithm, data in algorithms.items()
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
