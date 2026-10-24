# SPDX-License-Identifier: MIT
"""Low-frequency architecture/training diagnostics (spec section 7).

Nothing here changes the network, the loss, the RNG stream or the optimiser.
The fuse probe runs one ``no_grad`` forward under a temporary module hook and
restores the RNG state and module mode afterwards; every field that is not
actually measured is reported as unavailable instead of being zero-filled.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from .common import append_jsonl, atomic_json, canonical

DIAGNOSTICS_SCHEMA = "guandan-serial-compare-diagnostics-v1"

# Field families the trainer can emit, with the *meaning* frozen here so a
# diagnostic reader cannot conflate ratio with KL.
FIELD_SEMANTICS = {
    "macro/ratio_max": "sampled macro policy ratio r = exp(log_ratio); NOT a KL",
    "macro/approx_kl_old_new": "sample estimate KL(behaviour || current) on the last minibatch",
    "macro/macro_kl": "mean per-macro token KL to the normalised legal-candidate reference (KL to uniform-like prior)",
    "macro/clip_frac": "fraction of macros clipped by PPO eps",
    "macro/value_ev": "1 - Var(target - prediction)/Var(target) on the macro-start sample unit",
    "macro/value_loss": "Huber(delta=10) value loss on the macro-start target",
}


class ProbeError(RuntimeError):
    """Raised when a diagnostic probe could not produce a trustworthy value."""


def probe_fuse_on_rollout(model, rollout, *, max_candidates: int = 256) -> dict[str, object]:
    """Run one bounded forward and measure the residual ratio from the hook."""
    import torch

    from guandan_marvel.training.checkpoints import capture_rng_state, restore_rng_state

    module = getattr(model, "actor_fuse", None)
    if module is None:
        return {"available": False, "reason": "actor_fuse is not enabled in this architecture"}
    captured: dict[str, object] = {}

    def hook(_module, inputs, output):
        with torch.no_grad():
            fused_input = inputs[0].detach().float()
            delta = output.detach().float()
            if fused_input.shape[0] > int(max_candidates):
                fused_input = fused_input[: int(max_candidates)]
                delta = delta[: int(max_candidates)]
            width = fused_input.shape[-1] // 4
            base = fused_input[..., width:2 * width]
            captured["candidate_count"] = int(fused_input.shape[0])
            captured["ratio"] = float(delta.norm().item() / max(base.norm().item(), 1e-12))
            if not math.isfinite(float(captured["ratio"])):
                raise ProbeError("fuse residual ratio is non-finite")

    was_training = bool(getattr(model, "training", False))
    rng = capture_rng_state()
    handle = module.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            model(rollout=rollout)
    finally:
        handle.remove()
        model.train(was_training)
        restore_rng_state(rng)
    if "ratio" not in captured:
        return {"available": False, "reason": "no legal candidate reached the fuse branch"}
    return {"available": True, **captured}


def _value(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
    return values[-1] if values else None


def build_arm_diagnostics(
    rows: Sequence[Mapping[str, object]],
    *,
    arm: str,
    ntp_enabled: bool,
    ranks: int,
    fuse_probe: Mapping[str, object] | None = None,
    context_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble the section-7 diagnostics from the arm's *real* metric rows."""
    if int(ranks) <= 0:
        raise ValueError("ranks must be positive")
    last = dict(rows[-1]) if rows else {}
    gradient_groups = {
        key.split("/", 2)[2]: float(value)
        for key, value in last.items()
        if key.startswith("grad_norm/") and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if ntp_enabled:
        ntp = {
            "status": "RECORDED",
            "loss": _value(rows, "macro/ntp_loss"),
            "pos_rate": _value(rows, "macro/ntp_pos_rate"),
        }
    else:
        ntp = {
            "status": "DISABLED",
            "loss": None,
            "reason": "the NTP auxiliary head is not built for this arm; no zero is reported as a better loss",
        }
    forced = _value(rows, "health/legal_single_action_tokens_last")
    token_count = _value(rows, "macro/token_count_local")
    payload = {
        "schema": DIAGNOSTICS_SCHEMA,
        "arm": str(arm),
        "ranks": int(ranks),
        "frequency": "low (written once per checkpoint interval)",
        "gradients": {
            "groups": gradient_groups,
            "scope": (
                "per-parameter-group gradient norm after AMP unscale and before clip_grad_norm_; "
                "an AMP-skipped step contributes no gradient and is counted separately"
            ),
            "amp_skipped_steps": int(last.get("runtime/amp_skipped_steps", 0) or 0),
            "post_clip": "not separately recorded; the clip target is grad_max_norm in the plan",
        },
        "fuse": dict(fuse_probe) if fuse_probe is not None else {
            "available": False,
            "reason": "probe not scheduled for this checkpoint",
        },
        "value": {
            "ev": _value(rows, "macro/value_ev"),
            "loss": _value(rows, "macro/value_loss"),
            "sample_unit": "macro start (current-owned macros only)",
            "note": "EV is not a gradient signal-to-noise ratio",
        },
        "entropy": {
            "normalized_all_tokens": _value(rows, "macro/normalized_entropy"),
            "normalized_non_forced": _value(rows, "macro/non_forced_normalized_entropy"),
            "non_forced_token_count": _value(rows, "macro/non_forced_token_count"),
            "forced_action_count": forced,
            "forced_action_fraction": (
                float(forced) / float(token_count) if forced is not None and token_count else None
            ),
        },
        "macro_length": {
            "mean": _value(rows, "macro/macro_length_mean"),
            "max": _value(rows, "macro/macro_length_max"),
            "clip_buckets": {
                key.split("/", 2)[2]: float(value)
                for key, value in last.items()
                if key.startswith("clip_frac_by_length/") and isinstance(value, (int, float))
            },
        },
        "ntp": ntp,
        "ratio_and_kl": {
            "ratio_max": _value(rows, "macro/ratio_max"),
            "kl_to_behaviour_last_minibatch": _value(rows, "macro/approx_kl_old_new"),
            "kl_to_uniform_like_reference": _value(rows, "macro/macro_kl"),
            "semantics": dict(FIELD_SEMANTICS),
        },
        "owner_scope": {
            "current_token_count": _value(rows, "macro/token_count_local"),
            "opponent_token_count": _value(rows, "macro/opponent_token_count"),
            "macro_count_local": _value(rows, "macro/macro_count_local"),
            "macro_count_global": _value(rows, "macro/macro_count_global"),
            "rule": "only the current owner's tokens/macros enter policy, value and training statistics",
        },
        "context_provenance": dict(context_provenance) if context_provenance is not None else {
            "available": False,
            "reason": "provenance not supplied for this checkpoint",
        },
        "kl_control": {
            key.split("/", 2)[2]: value
            for key, value in last.items()
            if key.startswith("macro/kl_control/") and isinstance(value, (int, float, bool, dict, type(None)))
        },
    }
    payload["hash"] = canonical(payload)
    return payload


def write_diagnostics(path: str | Path, payload: Mapping[str, object]) -> Path:
    return atomic_json(path, dict(payload))


def append_diagnostics(path: str | Path, payload: Mapping[str, object]) -> None:
    append_jsonl(path, dict(payload))


__all__ = [
    "DIAGNOSTICS_SCHEMA", "FIELD_SEMANTICS", "ProbeError", "append_diagnostics",
    "build_arm_diagnostics", "probe_fuse_on_rollout", "write_diagnostics",
]
