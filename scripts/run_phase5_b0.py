# SPDX-License-Identifier: MIT
"""Bounded phase-5 B0 collection + A/B/C/D probes; never a formal training gate."""
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
OUTPUT_ROOT = (ROOT / "runs" / "phase5" / "b0").resolve()


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("at least three distinct pre-registered seeds are required")
    if len(seeds) > 8:
        raise argparse.ArgumentTypeError("no more than eight probe seeds are allowed")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda")
    parser.add_argument("--deals", type=positive_int, default=512)
    parser.add_argument("--batch-slots", type=positive_int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=90000)
    parser.add_argument("--max-macro-starts-per-deal", type=positive_int, default=8)
    parser.add_argument("--max-collection-seconds", type=float, default=1800)
    parser.add_argument("--max-probe-seconds", type=float, default=1200)
    parser.add_argument("--seeds", type=seed_list, default=[0, 1, 2])
    parser.add_argument("--epochs", type=positive_int, default=40)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--patience", type=positive_int, default=6)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=positive_int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--split-seed", type=int, default=20261019)
    parser.add_argument(
        "--warm-start", default=str(
            ROOT / "runs/comparison_delivery/dataset/m09_candidate_iter002000.pt"
        ),
    )
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "b0_result.json"))
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    if args.deals > 512:
        parser.error("B0 is capped at the pre-registered 512 deals")
    if args.batch_slots > 16:
        parser.error("B0 is capped at 16 collection slots on the 4 GB local GPU")
    if args.start_index < 0:
        parser.error("start-index must be nonnegative")
    if args.max_macro_starts_per_deal > 8:
        parser.error("max-macro-starts-per-deal is capped at the pre-registered 8")
    if not math.isfinite(args.max_collection_seconds) or not 0 < args.max_collection_seconds <= 1800:
        parser.error("collection budget must be finite and in (0, 1800] seconds")
    if not math.isfinite(args.max_probe_seconds) or not 0 < args.max_probe_seconds <= 1200:
        parser.error("probe budget must be finite and in (0, 1200] seconds")
    if args.epochs > 40:
        parser.error("probe epochs are capped at the pre-registered 40")
    if args.batch_size > 256:
        parser.error("probe batch size is capped at the pre-registered 256")
    if args.patience > 6:
        parser.error("early-stop patience is capped at the pre-registered 6")
    if not math.isfinite(args.epsilon) or args.epsilon < 0:
        parser.error("epsilon must be finite and non-negative")
    output = Path(args.output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        parser.error("B0 outputs must stay under runs/phase5/b0")
    artifacts = [
        output,
        output.with_name(output.stem + ".samples.jsonl"),
        output.with_name(output.stem + ".manifest.json"),
        output.with_name(output.stem + ".gate.json"),
    ]
    for artifact in artifacts:
        if artifact.exists():
            parser.error(f"refusing to overwrite existing evidence: {artifact}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = validate_args(parser, args)
    result = {
        "schema": "guandan-phase5-b0-run-v1",
        "status": "FAIL",
        "python_executable": sys.executable,
        "requested_device": args.device,
        "deals_requested": args.deals,
        "start_index": args.start_index,
        "batch_slots": args.batch_slots,
        "collection_budget_seconds": args.max_collection_seconds,
        "probe_budget_seconds": args.max_probe_seconds,
        "probe_seeds": args.seeds,
        "training": False,
        "b1_training": False,
        "b0_gate": "NOT RUN",
        "unexecuted_items": [
            "B1 belief head (only after a PASS)",
            "50/250/500/1000/1500 iteration formal training",
            "Kaggle / dual-T4 / NCCL",
            "final holdout",
            "FableDan evaluation",
            "stage 6 opponent pool / search",
        ],
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
        from guandan_marvel.phase5.b0_runner import (
            canonical_json_hash, evaluate_b0, source_hash,
        )
        from guandan_marvel.phase5.belief_collect import (
            collect_belief_dataset, context_for, dataset_manifest,
        )
        from guandan_marvel.phase5.design import PHASE5_B0_PLAN, phase5_design_hash
        from guandan_marvel.phase5.gate import seal_gate_artifact

        device = torch.device("cuda:0")
        checkpoint = Path(args.warm_start).resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        identity = validate_production_warm_start(payload, source=str(checkpoint))
        resolved_config = payload["resolved_config"]
        cfg = deserialize_train_config(resolved_config)
        cfg.device = str(device)
        torch.manual_seed(args.base_seed)
        model = Agent(cfg, enable_value_head=True, enable_ntp_head=False).to(device)
        model.load_state_dict(payload["agent_state_dict"], strict=True)
        model.eval()
        before = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        devices, amp_states = set(), set()

        def check_device(module, positional, keyword):
            actual = keyword["rollout"]["obs_tokens"].device
            if actual != device or next(module.parameters()).device != device:
                raise RuntimeError("policy or rollout moved off the requested CUDA device")
            devices.add(str(actual))
            amp_states.add(torch.is_autocast_enabled("cuda"))

        handle = model.register_forward_pre_hook(check_device, with_kwargs=True)
        torch.cuda.reset_peak_memory_stats(device)
        collection_started = time.perf_counter()
        try:
            samples, metadata, info = collect_belief_dataset(
                policy=model, device=device, deals=args.deals, base_seed=args.base_seed,
                max_macro_starts_per_deal=args.max_macro_starts_per_deal,
                max_wall_seconds=args.max_collection_seconds,
                start_index=args.start_index, batch_slots=args.batch_slots,
            )
            torch.cuda.synchronize(device)
        finally:
            handle.remove()
        collection_seconds = time.perf_counter() - collection_started
        unchanged = all(
            torch.equal(before[key], value.detach().cpu())
            for key, value in model.state_dict().items()
        )
        if not unchanged:
            raise RuntimeError("the frozen policy parameters changed during collection")
        completed = [row for row in metadata if bool(row.get("finished"))]
        completed_indices = {int(row["deal_index"]) for row in completed}
        completed_samples = [
            sample for sample in samples if int(sample.deal_index) in completed_indices
        ]
        dropped = len(samples) - len(completed_samples)
        if not completed_samples:
            raise RuntimeError("no completed deal produced a belief sample")
        design = phase5_design_hash()
        source = source_hash()
        config = canonical_json_hash(resolved_config)
        evaluate_started = time.perf_counter()
        evaluated = evaluate_b0(
            completed_samples,
            completed_deals=len(completed),
            collection_truncated=bool(info.get("truncated")),
            device=str(device),
            seeds=args.seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            max_probe_seconds=args.max_probe_seconds,
            epsilon=args.epsilon,
            bootstrap={
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "confidence": float(PHASE5_B0_PLAN["gate"]["bootstrap"]["confidence"]),
            },
            source_hash_value=source,
            config_hash_value=config,
        )
        evaluate_seconds = time.perf_counter() - evaluate_started
        gate = seal_gate_artifact(evaluated["gate"])
        split_assignment = evaluated["dataset"]["split_assignment"]
        manifest = dataset_manifest(
            completed_samples, completed,
            collection=info,
            design_hash=design,
            source_hash=source,
            config_hash=config,
            split_assignment=split_assignment,
        )
        index_rows = sorted(int(row["deal_index"]) for row in completed)
        indices_correct = index_rows == list(
            range(args.start_index, args.start_index + len(index_rows))
        )
        result.update({
            "status": "COMPLETED",
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "parameter_device": str(next(model.parameters()).device),
            "forward_devices": sorted(devices),
            "all_forwards_cuda_autocast": amp_states == {True},
            "model_parameters_unchanged": unchanged,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_policy_identity": identity,
            "design_hash": design,
            "source_hash": source,
            "config_hash": config,
            "dataset_split_hash": evaluated["dataset"]["split_hash"],
            "dataset_records_hash": evaluated["dataset"]["records_hash"],
            "sample_count": len(completed_samples),
            "dropped_partial_samples": int(dropped),
            "completed_deals": len(completed),
            "deal_indices": index_rows,
            "deal_indices_correct": bool(indices_correct),
            "contexts_correct": all(
                row["context"] == context_for(int(row["deal_index"])) for row in completed
            ),
            "collection": info,
            "collection_seconds": float(collection_seconds),
            "evaluate_seconds": float(evaluate_seconds),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "leakage_audit": evaluated["leakage_audit"],
            "group_disjoint": evaluated["group_disjoint"],
            "sufficiency": evaluated["sufficiency"],
            "probe": evaluated["probe"],
            "gate": gate,
            "b0_gate": gate["verdict"],
            "dataset": evaluated["dataset"],
            "unexecuted_items": (
                result["unexecuted_items"]
                if gate["verdict"] != "PASS"
                else result["unexecuted_items"]
            ),
        })
        # Evidence artifacts are written once; the output file is never
        # overwritten (validate_args refuses an existing path).
        sample_path = output.with_name(output.stem + ".samples.jsonl")
        manifest_path = output.with_name(output.stem + ".manifest.json")
        gate_path = output.with_name(output.stem + ".gate.json")
        with sample_path.open("x", encoding="utf-8") as stream:
            for sample in completed_samples:
                stream.write(json.dumps(sample.to_record()) + "\n")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        result.update({
            "sample_artifact": str(sample_path),
            "sample_artifact_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            "manifest_artifact": str(manifest_path),
            "manifest_artifact_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "gate_artifact": str(gate_path),
            "gate_artifact_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        })
    except Exception as error:  # noqa: BLE001 - reported with real exit status
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    result["total_wall_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        # Never overwrite a concurrent artifact; the run is invalid.
        print(json.dumps({
            "status": "FAIL", "error": f"output appeared during the run: {output}",
        }, indent=2), flush=True)
        return 1
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        key: value for key, value in result.items()
        if key not in {"probe", "gate", "dataset", "traceback", "leakage_audit"}
    }
    if "gate" in result:
        summary["gate"] = {
            "verdict": result["gate"]["verdict"],
            "reasons": result["gate"]["reasons"],
            "primary": result["gate"]["primary"],
        }
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
