# SPDX-License-Identifier: MIT
"""Opt-in pilot instrumentation. Does not alter gradients, RNG, or legacy workers."""
from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
import torch
import torch.distributed as dist
from .common import append_jsonl, atomic_json, sha256_file
from guandan_marvel.evaluation.checkpoints import freeze_merged_policy

SCHEMA = "guandan-marvel-t4-pilot-observability-v1"

class PilotObserver:
    def __init__(self, profile, trainer, root: Path, rank_root: Path):
        self.enabled = bool((profile.get("pilot_observability") or {}).get("enabled", False))
        self.trainer, self.root, self.rank_root = trainer, root, rank_root
        self.start = 0.0
        self.updates = self.skips = 0
        self.probe = None
        if not self.enabled:
            return
        cap = (profile.get("pilot_observability") or {}).get("maximum_iteration", 50)
        if isinstance(cap, bool) or cap not in (50, 250) or not 1 <= int(trainer.cfg.total_iterations) <= cap:
            raise ValueError("pilot exceeds its versioned absolute iteration cap (50 or 250)")
        self.stage_seconds = {}
        self.stage_counts = {}
        if (profile.get("pilot_observability") or {}).get("profile_rollout_stages", False):
            from dataclasses import replace
            previous_callback = trainer.sampler.config.timing_callback
            def record_stage(name, seconds):
                self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + float(seconds)
                self.stage_counts[name] = self.stage_counts.get(name, 0) + 1
                if previous_callback is not None:
                    previous_callback(name, seconds)
            trainer.sampler.config = replace(trainer.sampler.config, timing_callback=record_stage)
        if trainer.algorithm != "ippo":
            raise ValueError("pilot observer supports only the IPPO pilot")
        device = trainer.device
        row = {"schema": SCHEMA, "rank": trainer.rank, "local_rank": trainer.local_rank,
               "world_size": trainer.world_size, "device": str(device),
               "communication_backend": dist.get_backend() if dist.is_initialized() else None,
               "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "trim_completed_padding": bool(trainer.cfg.trim_completed_padding),
               "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
               "amp": trainer.amp_enabled,
               "object_collective_backend": getattr(getattr(trainer, "distributed", None), "object_collective_backend", "disabled"),
               "training_semantics": trainer.training_semantics_profile,
               "source_hash": trainer.source_hash}
        atomic_json(rank_root / "pilot_runtime.json", row)

    def export_initial(self, *, mode: str) -> None:
        if not self.enabled or mode == "resume":
            return
        event = None
        if self.trainer.is_primary:
            try:
                path = freeze_merged_policy(self.trainer, self.root / "pilot_initial_policy.pt")
                event = {"status": "complete", "sha256": sha256_file(path), "iteration": self.trainer.iteration}
                atomic_json(self.root / "pilot_initial_identity.json", event)
            except Exception as error:
                event = {"status": "failed", "error": str(error)}
        rows = self.trainer.distributed.all_gather_object(event)
        primary = rows[0]
        if not primary or primary.get("status") != "complete":
            raise RuntimeError(f"initial pilot policy export failed: {primary}")

    def begin(self) -> None:
        if not self.enabled:
            return
        t = self.trainer
        if t.device.type == "cuda":
            torch.cuda.synchronize(t.device)
            torch.cuda.reset_peak_memory_stats(t.device)
        self.stage_seconds.clear()
        self.stage_counts.clear()
        self.start = time.perf_counter()
        self.updates, self.skips = t.learner_updates, t.amp_skipped_steps
        # Tiny sentinel from the actual logit head, not a full state_dict clone.
        self.probe = t.agent_resume.logit_head.weight.detach().clone()

    def finish(self, metrics: Mapping[str, object]) -> dict[str, object]:
        if not self.enabled:
            return {}
        t = self.trainer
        if t.device.type == "cuda":
            torch.cuda.synchronize(t.device)
        elapsed = time.perf_counter() - self.start
        actual_updates = t.learner_updates - self.updates
        row = {"schema": SCHEMA, "rank": t.rank, "local_rank": t.local_rank,
               "iteration": t.iteration, "world_size": t.world_size,
               "timestamp": datetime.now(timezone.utc).isoformat(),
               "device": str(t.device), "iteration_seconds": elapsed,
               "effective_optimizer_updates": actual_updates,
               "amp_skipped_updates": t.amp_skipped_steps - self.skips,
               "amp_scale": float(t.grad_scaler.get_scale()),
               "logit_weight_changed": not torch.equal(self.probe, t.agent_resume.logit_head.weight.detach()),
               "completed_games_total": t.completed_games_total,
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(t.device)) if t.device.type == "cuda" else 0,
               "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(t.device)) if t.device.type == "cuda" else 0,
               "rollout_seconds": metrics.get("perf/rollout_seconds"),
               "learner_seconds": metrics.get("perf/learner_seconds"),
               "trajectory_length_mean": metrics.get("statistics/trajectory_length_mean"),
               "rollout_stage_host_seconds": dict(self.stage_seconds),
               "rollout_stage_calls": dict(self.stage_counts)}
        if actual_updates <= 0:
            raise RuntimeError("pilot iteration had zero effective optimizer updates (AMP may have skipped all)")
        append_jsonl(self.rank_root / "pilot_iterations.jsonl", row)
        self.probe = None
        return {**{f"perf/rollout_stage_host_seconds/{k}": v for k, v in self.stage_seconds.items()},
                "pilot/iteration_seconds": elapsed,
                "pilot/peak_allocated_bytes": row["peak_allocated_bytes"],
                "pilot/peak_reserved_bytes": row["peak_reserved_bytes"],
                "pilot/effective_optimizer_updates": actual_updates,
                "pilot/amp_skipped_updates": row["amp_skipped_updates"],
                "pilot/logit_weight_changed": row["logit_weight_changed"]}

    def console(self, aggregated, *, target: int) -> None:
        if self.enabled and self.trainer.is_primary:
            print("[T4x2 PILOT] "
                  f"iter={self.trainer.iteration}/{target} games={aggregated.get('statistics/completed_games_total')} "
                  f"iter_s={aggregated.get('pilot/iteration_seconds', 0):.2f} "
                  f"rollout_s={aggregated.get('perf/rollout_seconds', 0):.2f} "
                  f"learner_s={aggregated.get('perf/learner_seconds', 0):.2f} "
                  f"pg={aggregated.get('macro/pg_loss')} value={aggregated.get('macro/value_loss')} "
                  f"kl_uniform={aggregated.get('macro/macro_kl')} approx_kl_old_new={aggregated.get('macro/approx_kl_old_new')} clip={aggregated.get('macro/clip_frac')} "
                  f"peak_GiB={aggregated.get('pilot/peak_allocated_bytes', 0)/2**30:.2f}", flush=True)
