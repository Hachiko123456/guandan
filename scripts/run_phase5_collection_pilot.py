# SPDX-License-Identifier: MIT
"""Bounded CUDA-only collection pilot; not B0 qualification or policy training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cuda:0"), default="cuda")
    parser.add_argument("--deals", type=positive_int, default=16)
    parser.add_argument("--batch-slots", type=positive_int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=90000)
    parser.add_argument("--max-wall-seconds", type=float, default=90)
    parser.add_argument("--max-macro-starts-per-deal", type=positive_int, default=8)
    parser.add_argument("--warm-start", default=str(ROOT / "runs/comparison_delivery/dataset/m09_candidate_iter002000.pt"))
    parser.add_argument("--output", default=str(ROOT / "runs/phase5/pilot/pilot.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    import math
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.deals > 16 or args.batch_slots > 8:
        parser.error("pilot is capped at 16 deals and 8 slots; do not use it as a full B0 runner")
    if args.start_index < 0:
        parser.error("start-index must be nonnegative")
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 180:
        parser.error("pilot wall budget must be finite and in (0, 180] seconds")
    output = Path(args.output).resolve()
    if not output.is_relative_to((ROOT / "runs/phase5/pilot").resolve()):
        parser.error("pilot outputs must stay under runs/phase5/pilot, not formal delivery/evidence")
    result = {"schema": "guandan-phase5-collection-pilot-v2", "status": "FAIL",
              "python_executable": sys.executable, "requested_device": args.device,
              "deals_requested": args.deals, "start_index": args.start_index,
              "batch_slots": args.batch_slots, "wall_budget_seconds": args.max_wall_seconds,
              "training": False, "b0_gate": "NOT RUN"}
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
        from guandan_marvel.phase5.belief_collect import collect_belief_dataset, context_for
        from guandan_marvel.phase5.b0_runner import source_hash as phase5_source_hash
        from guandan_marvel.phase5.design import (
            PHASE5_PUBLIC_LEDGER_SCHEMA, PUBLIC_FEATURE_WIDTH, phase5_design_hash,
        )
        device = torch.device("cuda:0")
        checkpoint = Path(args.warm_start).resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        identity = validate_production_warm_start(payload, source=str(checkpoint))
        cfg = deserialize_train_config(payload["resolved_config"])
        cfg.device = str(device)
        torch.manual_seed(args.base_seed)
        model = Agent(cfg, enable_value_head=True, enable_ntp_head=False).to(device)
        model.load_state_dict(payload["agent_state_dict"], strict=True)
        model.eval()
        before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        devices, amp_states = set(), set()
        def check_device(module, positional, keyword):
            actual = keyword["rollout"]["obs_tokens"].device
            if actual != device or next(module.parameters()).device != device:
                raise RuntimeError("policy or rollout moved off the requested CUDA device")
            devices.add(str(actual))
            amp_states.add(torch.is_autocast_enabled("cuda"))
        handle = model.register_forward_pre_hook(check_device, with_kwargs=True)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            samples, metadata, info = collect_belief_dataset(
                policy=model, device=device, deals=args.deals, base_seed=args.base_seed,
                max_macro_starts_per_deal=args.max_macro_starts_per_deal,
                max_wall_seconds=args.max_wall_seconds, start_index=args.start_index,
                batch_slots=args.batch_slots,
            )
            torch.cuda.synchronize(device)
        finally:
            handle.remove()
        unchanged = all(torch.equal(before[key], value.detach().cpu())
                        for key, value in model.state_dict().items())
        indices = sorted(row["deal_index"] for row in metadata)
        conserved = bool(samples) and all(sample.audit["conserved"] for sample in samples)
        ledger_conserved = bool(samples) and all(
            sample.audit.get("ledger_conserved") for sample in samples
        )
        known_lower_bound = bool(samples) and all(
            sample.audit.get("known_owner_lower_bound") for sample in samples
        )
        ledger_owner_audit = bool(samples) and all(
            sample.audit.get("ledger_owner_matches_engine") for sample in samples
        )
        events_present = bool(samples) and all(
            int(sample.audit.get("public_event_count", -1)) == len(sample.public_events)
            for sample in samples
        )
        contexts_correct = all(row["context"] == context_for(row["deal_index"]) for row in metadata)
        result.update({"torch_version": torch.__version__, "cuda_build": torch.version.cuda,
                       "gpu_name": torch.cuda.get_device_name(device),
                       "parameter_device": str(next(model.parameters()).device),
                       "forward_devices": sorted(devices), "all_forwards_cuda_autocast": amp_states == {True},
                       "samples": len(samples), "groups": len({s.group_id for s in samples}),
                       "finished_deals": sum(bool(row["finished"]) for row in metadata),
                       "deal_indices": indices, "context_levels": sorted({row["context"]["level_rank"] for row in metadata}),
                       "context_leaders": sorted({row["context"]["leader_seat"] for row in metadata}),
                       "tribute_modes": sorted({s.tribute_mode for s in samples}),
                       "contexts_correct": contexts_correct, "conservation_all": conserved,
                       "ledger_conservation_all": ledger_conserved,
                       "known_owner_lower_bound_all": known_lower_bound,
                       "ledger_owner_audit_all": ledger_owner_audit,
                       "public_events_present_all": events_present,
                       "public_ledger_schema": PHASE5_PUBLIC_LEDGER_SCHEMA,
                       "public_feature_width": PUBLIC_FEATURE_WIDTH,
                       "design_hash": phase5_design_hash(),
                       "source_hash": phase5_source_hash(),
                       "model_parameters_unchanged": unchanged, "collection": info,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                       "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       "source_policy_identity": identity})
        if info["truncated"] or result["finished_deals"] != args.deals:
            raise RuntimeError("pilot did not finish within budget; stop and request user confirmation")
        if indices != list(range(args.start_index, args.start_index + args.deals)):
            raise RuntimeError("pilot returned incorrect absolute deal indices")
        if not (conserved and ledger_conserved and known_lower_bound and ledger_owner_audit
                and events_present and contexts_correct and unchanged
                and devices == {"cuda:0"} and amp_states == {True}):
            raise RuntimeError(
                "pilot device/context/conservation/public-ledger/frozen-model gate failed"
            )
        data_path = output.with_name(output.stem + ".samples.jsonl")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with data_path.open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample.to_record()) + "\n")
        result.update(status="PASS", sample_artifact=str(data_path),
                      sample_artifact_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                      metadata=metadata)
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    result["total_wall_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"metadata", "traceback"}}, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
