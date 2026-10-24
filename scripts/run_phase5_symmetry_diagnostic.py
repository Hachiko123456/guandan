# SPDX-License-Identifier: MIT
"""Bounded CUDA symmetry diagnostic; not training and not a stage-6 entry point."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (ROOT / "runs" / "phase5" / "symmetry").resolve()


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda")
    parser.add_argument("--deals", type=positive_int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=7000)
    parser.add_argument("--max-macro-starts-per-deal", type=positive_int, default=24)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--max-wall-seconds", type=float, default=240)
    parser.add_argument(
        "--warm-start", default=str(
            ROOT / "runs/comparison_delivery/dataset/m09_candidate_iter002000.pt"
        ),
    )
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "symmetry_diagnostic.json"))
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    if args.deals > 16:
        parser.error("the symmetry diagnostic pilot is capped at 16 deals")
    if args.start_index < 0:
        parser.error("start-index must be nonnegative")
    if args.max_macro_starts_per_deal > 24:
        parser.error("max-macro-starts-per-deal is capped at the pre-registered 24")
    if args.batch_size > 64:
        parser.error("batch-size is capped at 64 for the 4 GB local GPU")
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 300:
        parser.error("wall budget must be finite and in (0, 300] seconds")
    output = Path(args.output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        parser.error("symmetry diagnostic outputs must stay under runs/phase5/symmetry")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = validate_args(parser, args)
    result = {
        "schema": "guandan-phase5-symmetry-diagnostic-run-v1",
        "status": "FAIL",
        "python_executable": sys.executable,
        "requested_device": args.device,
        "deals_requested": args.deals,
        "start_index": args.start_index,
        "wall_budget_seconds": args.max_wall_seconds,
        "training": False,
        "b0_gate": "NOT RUN",
    }
    started = time.perf_counter()
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from guandan_marvel.config import deserialize_train_config
        from guandan_marvel.kaggle.phase4_delivery import validate_production_warm_start
        from guandan_marvel.model import Agent
        from guandan_marvel.phase5.belief_probe import require_cuda_device
        from guandan_marvel.phase5.design import phase5_design_hash
        from guandan_marvel.phase5.diagnostic_runner import run_symmetry_diagnostic

        device = require_cuda_device(args.device)
        checkpoint = Path(args.warm_start).resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        identity = validate_production_warm_start(payload, source=str(checkpoint))
        cfg = deserialize_train_config(payload["resolved_config"])
        cfg.device = str(device)
        torch.manual_seed(args.base_seed)
        model = Agent(cfg, enable_value_head=True, enable_ntp_head=False).to(device)
        model.load_state_dict(payload["agent_state_dict"], strict=True)
        model.eval()
        before = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        diagnostic = run_symmetry_diagnostic(
            policy=model,
            device=str(device),
            deals=args.deals,
            start_index=args.start_index,
            base_seed=args.base_seed,
            max_macro_starts_per_deal=args.max_macro_starts_per_deal,
            max_wall_seconds=args.max_wall_seconds,
            batch_size=args.batch_size,
        )
        unchanged = all(
            torch.equal(before[key], value.detach().cpu())
            for key, value in model.state_dict().items()
        )
        if not unchanged:
            raise RuntimeError("the frozen policy parameters changed during the diagnostic")
        result.update({
            "status": "COMPLETED",
            "design_hash": phase5_design_hash(),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "parameter_device": str(next(model.parameters()).device),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_policy_identity": identity,
            "model_parameters_unchanged": unchanged,
            "diagnostic": diagnostic,
            "decision": diagnostic["decision"],
            "truncated": diagnostic["truncated"],
            "peak_allocated_bytes": diagnostic["peak_allocated_bytes"],
            "peak_reserved_bytes": diagnostic["peak_reserved_bytes"],
        })
    except Exception as error:  # noqa: BLE001 - reported with real exit status
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    result["total_wall_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        key: value for key, value in result.items() if key not in {"diagnostic", "traceback"}
    }
    if "diagnostic" in result:
        diagnostic = result["diagnostic"]
        summary["aggregate"] = diagnostic.get("aggregate")
        summary["undefined_samples"] = diagnostic.get("undefined_samples")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
