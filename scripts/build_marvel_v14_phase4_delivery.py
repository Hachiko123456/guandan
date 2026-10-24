# SPDX-License-Identifier: MIT
"""Build and validate the MARVEL v14 phase-4 segmented Kaggle delivery."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.kaggle.package import render_notebook
from guandan_marvel.kaggle.phase4_delivery import (
    DEFAULT_LOCAL_PROBE_HORIZON, build_phase4_delivery, resolve_phase4_probe_device,
    validate_phase4_delivery, validate_phase4_delivery_runtime,
)

# The formal delivery default is the production-scale M09 v1.3 policy-only
# checkpoint.  The phase-3 synthetic smoke artifact is *not* an acceptable
# default: it cannot load into the production v1.4 start profile.
DEFAULT_WARM_START = "runs/comparison_delivery/dataset/m09_candidate_iter002000.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=".")
    parser.add_argument("--output", default="runs/marvel_v14_phase4/delivery")
    parser.add_argument("--warm-start", default=DEFAULT_WARM_START)
    parser.add_argument("--notebook-template", default=None)
    parser.add_argument(
        "--runtime-probe", action="store_true",
        help="also run the fail-closed local 1-iteration start probe after building",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="local runtime probe device only; never changes the Kaggle profile")
    parser.add_argument("--runtime-max-game-horizon", type=int, default=DEFAULT_LOCAL_PROBE_HORIZON,
                        help="bounded local smoke horizon; truncated games still fail")
    parser.add_argument("--runtime-init-scale", type=float, default=1024.0,
                        help="initial CUDA GradScaler scale for local smoke only; recorded in report")
    args = parser.parse_args(argv)

    if args.runtime_probe:
        # Fail before rebuilding an existing delivery if explicit CUDA is absent.
        resolve_phase4_probe_device(args.device)
    output = Path(args.output).resolve()
    if args.notebook_template is None:
        template = output.parent / "phase4_notebook_template.ipynb"
        template.parent.mkdir(parents=True, exist_ok=True)
        template.write_text(
            json.dumps(render_notebook(), indent=1, sort_keys=True) + "\n", encoding="utf-8",
        )
    else:
        template = Path(args.notebook_template).resolve()

    manifest = build_phase4_delivery(
        args.repository, output,
        warm_start_checkpoint=args.warm_start, notebook_template=template,
    )
    validation = validate_phase4_delivery(output)
    runtime = validate_phase4_delivery_runtime(
        output, device=args.device, max_game_horizon=args.runtime_max_game_horizon,
        cuda_init_scale=args.runtime_init_scale,
    ) if args.runtime_probe else None
    print(json.dumps({
        "delivery_id": manifest["delivery_id"],
        "plan_hash": manifest["plan_hash"],
        "source_archive_sha256": manifest["source_archive_sha256"],
        "semantic_contract_hash": manifest["semantic_contract_hash"],
        "native_backend_hash": manifest["native_backend_hash"],
        "warm_start": manifest["warm_start"],
        "target_model": manifest["target_model"],
        "uploads": manifest["uploads"],
        "validation": validation["schema"],
        "valid": validation["valid"],
        "runtime_validation": None if runtime is None else runtime["schema"],
        "runtime_valid": None if runtime is None else runtime["valid"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
