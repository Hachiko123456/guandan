# SPDX-License-Identifier: MIT
"""Verify a built MARVEL v14 phase-4 delivery against its own recorded hashes.

Beyond the recorded file/upload hashes this verifier runs two fail-closed gates:

* the structural ``validate_phase4_delivery`` gate (v1.4 target, AMP on,
  production-scale warm start, semantic-contract consistency); and
* the runtime start probe ``validate_phase4_delivery_runtime`` which rebuilds a
  real ``TrainConfig``/``IPPOTrainer`` from the packaged profile, loads the
  packaged warm start, checks the v1.3->v1.4 migration and the pre-step policy
  equivalence, runs at least one local CPU/CUDA iteration and reloads the durable
  checkpoint (v1.4 + AMP) and a resume inheritance check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.kaggle.phase4_delivery import (
    DEFAULT_LOCAL_PROBE_HORIZON, PHASE4_PLAN_NAME,
    TARGET_MODEL_ARCHITECTURE, TARGET_MODEL_PARAMETER_COUNT,
    validate_phase4_delivery, validate_phase4_delivery_runtime,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", default="runs/marvel_v14_phase4/delivery")
    parser.add_argument("--output", default="runs/marvel_v14_phase4/delivery_verification.json")
    parser.add_argument(
        "--runtime", dest="runtime", action="store_true", default=True,
        help="run the fail-closed local CPU/CUDA start probe (default: on)",
    )
    parser.add_argument("--no-runtime", dest="runtime", action="store_false")
    parser.add_argument("--runtime-iterations", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="local runtime probe device; explicit CUDA cannot fall back to CPU")
    parser.add_argument("--runtime-max-game-horizon", type=int, default=DEFAULT_LOCAL_PROBE_HORIZON,
                        help="bounded local smoke horizon; does not change packaged configuration")
    parser.add_argument("--runtime-work-dir", default=None,
                        help="scratch/checkpoint directory outside the delivery package")
    parser.add_argument("--runtime-init-scale", type=float, default=1024.0,
                        help="initial CUDA GradScaler scale for local smoke only; recorded in report")
    args = parser.parse_args(argv)
    root = Path(args.delivery).resolve()
    manifest = json.loads((root / "delivery_manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []

    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append(f"missing file {entry['path']}")
        elif _sha256(path) != entry["sha256"]:
            failures.append(f"sha256 mismatch {entry['path']}")
    for upload in manifest["uploads"]:
        path = root / upload["path"]
        if not path.is_file() or _sha256(path) != upload["sha256"]:
            failures.append(f"upload sha256 mismatch {upload['path']}")
    recorded = {row["path"]: row["sha256"] for row in manifest["files"]}
    for segment in ("start", "resume"):
        dataset = root / "datasets" / segment
        plan_path = dataset / PHASE4_PLAN_NAME
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        rel = plan_path.relative_to(root).as_posix()
        if rel in recorded and recorded[rel] != _sha256(plan_path):
            failures.append(f"plan sha256 mismatch for {segment}")
        if plan["plan_hash"] != manifest["plan_hash"]:
            failures.append(f"plan hash mismatch for {segment}")
        if plan["semantic_contract_hash"] != manifest["semantic_contract_hash"]:
            failures.append(f"semantic contract hash mismatch for {segment}")
        if plan["native_backend_hash"] != manifest["native_backend_hash"]:
            failures.append(f"native backend hash mismatch for {segment}")
    warm = root / "datasets" / "start" / "warm_start_policy.pt"
    if not warm.is_file() or _sha256(warm) != manifest["warm_start"]["sha256"]:
        failures.append("warm-start identity mismatch")

    # --- phase 4A integration gates ------------------------------------------
    try:
        start_manifest = json.loads(
            (root / "datasets" / "start" / "guandan_marvel_dataset.json").read_text(encoding="utf-8")
        )
        training = start_manifest["profiles"]["full"]["training"]
        if str(training["model"]["architecture"]) != TARGET_MODEL_ARCHITECTURE:
            failures.append("start profile is not engineered for the v1.4-macro-fuse target")
        if bool(training.get("amp_enabled", False)) is not True:
            failures.append("start profile does not enable AMP")
        if str(manifest["target_model"]["architecture"]) != TARGET_MODEL_ARCHITECTURE:
            failures.append("delivery manifest target_model is not v1.4-macro-fuse")
        if int(manifest["target_model"]["parameter_count"]) != TARGET_MODEL_PARAMETER_COUNT:
            failures.append("delivery manifest target_model parameter count mismatch")
        if str(manifest["warm_start"]["architecture"]) != "v1.3":
            failures.append("delivery warm start is not a v1.3 policy")
        if int(manifest["warm_start"]["parameter_count"]) != 2_061_314:
            failures.append("delivery warm start is not the production M09 size")
    except (KeyError, TypeError, FileNotFoundError) as error:
        failures.append(f"phase4 profile/manifest inspection failed: {type(error).__name__}: {error}")

    validation = validate_phase4_delivery(root)

    runtime_result: dict[str, object] | None = None
    if args.runtime:
        try:
            runtime_result = validate_phase4_delivery_runtime(
                root, iterations=int(args.runtime_iterations), device=args.device,
                max_game_horizon=args.runtime_max_game_horizon, work_dir=args.runtime_work_dir,
                cuda_init_scale=args.runtime_init_scale,
            )
        except Exception as error:  # noqa: BLE001 - surface any fail-closed rejection
            failures.append(f"runtime start probe failed: {type(error).__name__}: {error}")

    result = {
        "schema": "guandan-marvel-v14-phase4-delivery-verification-v1",
        "delivery_id": manifest["delivery_id"],
        "valid": not failures,
        "failures": failures,
        "file_count": len(manifest["files"]),
        "plan_hash": manifest["plan_hash"],
        "semantic_contract_hash": manifest["semantic_contract_hash"],
        "native_backend_hash": manifest["native_backend_hash"],
        "source_archive_sha256": manifest["source_archive_sha256"],
        "warm_start_sha256": manifest["warm_start"]["sha256"],
        "warm_start": manifest["warm_start"],
        "target_model": manifest["target_model"],
        "validation_schema": validation["schema"],
        "runtime_schema": None if runtime_result is None else runtime_result["schema"],
        "runtime_valid": None if runtime_result is None else runtime_result["valid"],
        "runtime_probe": None if runtime_result is None else runtime_result["runtime_probe"],
        "uploads": manifest["uploads"],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
