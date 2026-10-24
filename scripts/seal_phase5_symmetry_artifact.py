# SPDX-License-Identifier: MIT
"""Seal the frozen phase-5 symmetry diagnostic for training-time consumption.

The diagnostic artifact itself stays byte-identical.  This offline step writes a
separate seal that binds its bytes, design hash, decision, transforms and the
initial policy checkpoint SHA256, so a training run can fail closed on any
tampering without rewriting historical evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "runs/phase5/symmetry/symmetry_diagnostic_r3.json"
DEFAULT_OUTPUT = ROOT / "runs/phase5/closeout_05e/symmetry_artifact_seal.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from guandan_marvel.phase5.symmetry_training import (
        SYMMETRY_ARTIFACT_SEAL_SCHEMA, canonical_hash, load_sealed_symmetry_artifact,
    )

    artifact_path = Path(args.artifact).resolve()
    payload = artifact_path.read_bytes()
    document = json.loads(payload.decode("utf-8"))
    seal = {
        "schema": SYMMETRY_ARTIFACT_SEAL_SCHEMA,
        "artifact_path": str(artifact_path),
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "content_hash": canonical_hash(document),
        "design_hash": document["design_hash"],
        "checkpoint_sha256": document["checkpoint_sha256"],
        "source_policy_identity": document["source_policy_identity"],
        "transforms": sorted(
            str(name) for name in document["diagnostic"]["transforms"]
        ),
        "decision_status": document["decision"]["status"],
        "sealed_note": (
            "Offline seal of frozen evidence. The original artifact is not "
            "rewritten; only its bytes and identity are bound here."
        ),
    }
    seal["seal_hash"] = canonical_hash(seal)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract = load_sealed_symmetry_artifact(output)
    print(json.dumps({
        "status": "SEALED",
        "output": str(output),
        "artifact_sha256": contract.artifact_sha256,
        "content_hash": contract.content_hash,
        "seal_hash": contract.seal_hash,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "transforms": list(contract.transforms),
        "decision_status": contract.decision["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
