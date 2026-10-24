# SPDX-License-Identifier: MIT
"""Phase-5 symmetry diagnostic runner (CUDA, bounded).

Per sample (one *policy macro start* of a real C++ deal):

1. record the acting seat's observation, the engine's legal next-token
   candidate rows and a serialized engine state for deterministic replay;
2. build the transformed observation with :class:`SeatRotation` (k = 1, 2, 3)
   and, where the deck swap has a legal image, :class:`CopySwap`;
3. replay the recorded engine state and read the transformed state's *own*
   candidate list, then match every source candidate to its image;
4. run the frozen policy on both observations and compare only the legal
   candidates with :func:`aligned_policy_comparison`.

The action unit is declared per sample.  The engine publishes next-token
candidates, not complete macro actions, so every comparison is reported as
``restricted_candidate``.  It is never called a full-action KL.

CUDA is mandatory; there is no CPU fallback and no silent OOM recovery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .design import PHASE5_DIAGNOSTIC_SCHEMA, SYMMETRY_SIGNIFICANCE
from .diagnostics import (
    ACTION_UNIT_RESTRICTED, aligned_policy_comparison, aggregate_diagnostic,
    bucket_key, decide_augmentation,
)
from .symmetry import (
    CARD_TOKEN_BASE, CopySwap, SeatRotation, SymmetryDomainError,
    candidate_image_is_unique, transform_candidate_rows, transform_observation,
    transform_seat,
)

PHASE_NAMES = {0: "PLAY", 1: "TRIBUTE", 2: "RETURN", 3: "TERMINAL"}


class SymmetryDiagnosticError(RuntimeError):
    pass


def require_cuda(device: str) -> torch.device:
    """Resolve an explicit CUDA device; never fall back to CPU."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"symmetry diagnostic requires an explicit CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is allowed")
    index = 0 if resolved.index is None else resolved.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is not available")
    return torch.device("cuda", index)


def tribute_mode(previous_result: Mapping[str, Any] | None, *, anti_tribute: bool) -> str:
    if previous_result is None:
        return "none"
    if previous_result.get("tribute_cancelled"):
        return "cancelled"
    if anti_tribute:
        return "anti_tribute"
    return str(previous_result.get("outcome_class", "unknown"))


@dataclass(frozen=True, slots=True)
class RecordedSample:
    deal_id: str
    token_step: int
    actor: int
    level_rank: int
    leader_seat: int
    tribute_mode: str
    phase: str
    tokens: np.ndarray          # [4, 48]
    channels: np.ndarray        # [4, 512]
    rows: np.ndarray            # [A, 8]
    engine_state: bytes         # serialized C++ batch (single slot)


def _tensor_batch(step_view, batch_size: int, device: torch.device):
    from guandan_marvel.model.tensordict import TensorDict

    return TensorDict(
        {
            "num_steps": torch.ones(batch_size, dtype=torch.long),
            "player_ids": torch.as_tensor(step_view.player_ids)[:, None],
            "obs_tokens": torch.as_tensor(step_view.observations.tokens)[:, None],
            "obs_channels": torch.as_tensor(step_view.observations.channels)[:, None],
            "avail_tokens": torch.as_tensor(step_view.available_actions.tokens)[:, None],
            "avail_weights": torch.as_tensor(step_view.available_actions.weights)[:, None],
        },
        batch_size=[batch_size], device=device,
    )


def collect_recorded_samples(
    *,
    policy,
    device: torch.device,
    deals: int,
    base_seed: int,
    max_macro_starts_per_deal: int,
    max_wall_seconds: float,
    start_index: int = 0,
) -> tuple[list[RecordedSample], list[dict[str, Any]], dict[str, Any]]:
    """Play real C++ deals with the frozen policy and record macro starts.

    Deals are played one at a time (``batch = 1``, ``auto_reset = False``): a
    terminal slot cannot be stepped again, and mixing terminal and live slots in
    one batched step would feed zero observations through the policy.  A
    one-slot stratified context schedule is installed before reset so the
    diagnostic covers level / leader / tribute buckets.
    """
    from nanopokerlib import guandan as cpp

    from guandan_marvel.model import KVCache

    from .belief_collect import context_for

    samples: list[RecordedSample] = []
    deal_meta: list[dict[str, Any]] = []
    started = time.perf_counter()
    truncated = False
    finished_deals = 0
    decoder = policy.decoder.config
    head_dim = decoder.hidden_size // decoder.num_attention_heads
    kv_heads = decoder.num_key_value_heads or decoder.num_attention_heads
    for slot in range(int(deals)):
        if time.perf_counter() - started > float(max_wall_seconds):
            truncated = True
            break
        seed = int(base_seed) + slot
        context = context_for(int(start_index) + slot)
        env = cpp.EnvBatch(1, auto_reset=False)
        env.configure_context_schedule([context], offset=0)
        step = env.reset(seed)
        kv = KVCache(
            policy.decoder_n_layers, 4, max_ctx_len=int(policy.max_game_horizon),
            num_heads=kv_heads, head_dim=head_dim, value_heads=kv_heads,
            value_head_dim=head_dim, dtype=torch.float32, device=device,
        )
        kv.reset(torch.ones(4, dtype=torch.bool, device=device))
        recorded = 0
        token_steps = 0
        while True:
            debug = env.debug_state(0)
            state = debug["state"]
            if not debug["prefix_tokens"] and recorded < int(max_macro_starts_per_deal):
                legal_rows = np.asarray(step.available_actions.weights[0]) > 0
                candidate_rows = np.asarray(
                    step.available_actions.tokens[0], dtype=np.int64,
                )[legal_rows].copy()
                if candidate_rows.shape[0] != len(debug["available_action_rows"]):
                    raise SymmetryDiagnosticError(
                        "observation candidate list disagrees with the engine debug rows"
                    )
                samples.append(RecordedSample(
                    deal_id=f"deal{seed:06d}",
                    token_step=int(debug["token_step"]),
                    actor=int(state["active_seat"]),
                    level_rank=int(state["level_rank"]),
                    leader_seat=int(state["leader_seat"]),
                    tribute_mode=tribute_mode(
                        debug["config"].get("previous_result"),
                        anti_tribute=bool(state["anti_tribute"]),
                    ),
                    phase=PHASE_NAMES[int(state["phase"])],
                    tokens=np.asarray(step.observations.tokens[0], dtype=np.int64).copy(),
                    channels=np.asarray(step.observations.channels[0], dtype=np.uint8).copy(),
                    rows=candidate_rows,
                    engine_state=_slot_payload(env, 0, debug),
                ))
                recorded += 1
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                output = policy(rollout=_tensor_batch(step, 1, device), use_cache=True,
                                past_key_values=kv)
            legal = np.asarray(step.available_actions.weights[0]) > 0
            probabilities = output.avail_probs[0, 0].float()
            if not bool(torch.isfinite(probabilities).all()):
                raise SymmetryDiagnosticError("policy produced a non-finite probability")
            masked = torch.where(
                torch.as_tensor(legal, device=device), probabilities,
                torch.zeros_like(probabilities),
            )
            if float(masked.sum().item()) <= 0:
                raise SymmetryDiagnosticError(
                    f"no legal candidate carries policy probability at deal {seed} step {token_steps}"
                )
            action = int(torch.multinomial(masked, 1).item())
            row = np.asarray(step.available_actions.tokens[0, action], dtype=np.int64)
            next_step = env.step(row[None, :])
            token_steps += 1
            if bool(next_step.is_terminal[0]):
                info = dict(next_step.infos[0])
                deal_meta.append({
                    "deal_id": f"deal{seed:06d}",
                    "finished": True,
                    "token_steps": token_steps,
                    "macro_starts_recorded": recorded,
                    "reward": [float(value) for value in next_step.rewards[0]],
                    "outcome_class": info.get("outcome_class"),
                    "winner_team": info.get("winner_team"),
                    "ranking": list(info.get("ranking", ())),
                })
                finished_deals += 1
                break
            if time.perf_counter() - started > float(max_wall_seconds):
                truncated = True
                break
            step = next_step
        del env, kv
        if truncated:
            deal_meta.append({"deal_id": f"deal{seed:06d}", "finished": False})
            break
    return samples, deal_meta, {
        "environment_seconds": time.perf_counter() - started,
        "finished_deals": int(finished_deals),
        "requested_deals": int(deals),
        "truncated": truncated,
    }


def _slot_payload(env, slot: int, debug: Mapping[str, Any]) -> bytes:
    """Serialized single-slot engine state for deterministic replay."""
    import json

    payload = json.loads(env.serialize().decode("utf-8"))
    game = payload["games"][slot]
    event = payload["events"][slot]
    return json.dumps(
        {"schema": "guandan-marvel-phase5-slot-v1", "game": game, "event": event},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def replay_slot(payload: bytes):
    """Rebuild a one-slot C++ batch from :func:`_slot_payload`."""
    import json

    from nanopokerlib import guandan as cpp

    data = json.loads(payload.decode("utf-8"))
    if data.get("schema") != "guandan-marvel-phase5-slot-v1":
        raise SymmetryDiagnosticError("unknown slot payload schema")
    batch = cpp.EnvBatch(1, auto_reset=False)
    serialized = json.dumps(
        {
            "schema": "guandan-marvel-cpp-runtime-v1",
            "backend": "cpp",
            "action_version": "GD-ACTION-0.2-MARVEL",
            "environment_version": "GD-ENV-0.2-MARVEL",
            "encoding_version": "GD-ENCODING-0.2-MARVEL",
            "auto_reset": False,
            "max_horizon": int(data["game"]["config"]["max_token_steps"]),
            "action_capacity": 256,
            "observation_capacity": 48,
            "trajectory_steps": [int(data["game"]["token_step"])],
            "events": [data["event"]],
            "games": [data["game"]],
        },
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return cpp.EnvBatch.deserialize(serialized)


def transformed_candidate_rows(
    sample: RecordedSample,
    transform,
) -> tuple[np.ndarray, tuple[int, ...]] | None:
    """Return ``(transformed rows, permutation)`` or ``None`` when undefined.

    The engine publishes next-token candidates.  A seat rotation is seat-free on
    the action vocabulary, and the deck swap is a bijection on physical card
    ids; in both cases the transformed legal candidate list is the image of the
    source list, enumerated in the same order.  The induced permutation is
    therefore the identity on that enumeration, and it is only accepted after a
    one-to-one check.  No engine replay is needed or claimed.
    """
    rows = np.asarray(sample.rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 8 or rows.shape[0] == 0:
        return None
    if not candidate_image_is_unique(rows, transform):
        return None
    target = transform_candidate_rows(rows, transform)
    keys = {tuple(int(value) for value in row) for row in target}
    if len(keys) != target.shape[0]:
        return None
    permutation = tuple(range(rows.shape[0]))
    return target, permutation


def _logprobs(policy, tokens: torch.Tensor, channels: torch.Tensor, rows: torch.Tensor,
              player_ids: torch.Tensor) -> torch.Tensor:
    from guandan_marvel.model.tensordict import TensorDict

    n = tokens.shape[0]
    weights = (rows[..., 5] > 0).to(torch.float32)
    rollout = TensorDict(
        {
            "num_steps": torch.ones(n, dtype=torch.long, device=tokens.device),
            "player_ids": player_ids[:, None],
            "obs_tokens": tokens[:, None],
            "obs_channels": channels[:, None],
            "avail_tokens": rows[:, None],
            "avail_weights": weights[:, None],
        },
        batch_size=[n], device=tokens.device,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output = policy(rollout=rollout)
    return output.avail_log_probs.squeeze(1).float()


def run_symmetry_diagnostic(
    *,
    policy,
    device: str = "cuda",
    deals: int = 16,
    base_seed: int = 7000,
    start_index: int = 0,
    max_macro_starts_per_deal: int = SYMMETRY_SIGNIFICANCE["max_macro_starts_per_deal"],
    max_wall_seconds: float = 900.0,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run the bounded diagnostic and return the pre-registered verdict."""
    resolved = require_cuda(device)
    if int(deals) <= 0 or int(batch_size) <= 0:
        raise ValueError("deals and batch_size must be positive")
    if int(start_index) < 0:
        raise ValueError("start_index must be non-negative")
    if float(max_wall_seconds) <= 0:
        raise ValueError("max_wall_seconds must be positive")
    torch.cuda.reset_peak_memory_stats(resolved)
    started = time.perf_counter()
    # Reserve a quarter of the total bounded wall budget for the paired
    # comparisons; collection returns partial data instead of overrunning.
    collection_budget = max(1.0, float(max_wall_seconds) * 0.75)
    samples, deal_meta, environment = collect_recorded_samples(
        policy=policy, device=resolved, deals=int(deals), base_seed=int(base_seed),
        start_index=int(start_index),
        max_macro_starts_per_deal=int(max_macro_starts_per_deal),
        max_wall_seconds=collection_budget,
    )
    learner_started = time.perf_counter()
    truncated = bool(environment.get("truncated"))
    records: list[dict[str, Any]] = []
    undefined: dict[str, int] = {}
    for transform in (SeatRotation(1), SeatRotation(2), SeatRotation(3), CopySwap()):
        if time.perf_counter() - started > float(max_wall_seconds):
            truncated = True
            break
        accepted = []
        for sample in samples:
            if time.perf_counter() - started > float(max_wall_seconds):
                truncated = True
                break
            mapped = transformed_candidate_rows(sample, transform)
            if mapped is None:
                undefined[transform.name] = undefined.get(transform.name, 0) + 1
                continue
            accepted.append((sample, mapped))
        if truncated and not accepted:
            break
        # The transformed observation is computed **once** per sample and reused
        # by every comparison, so the two forward passes below are the only
        # model executions of the diagnostic.
        transformed_observations = [
            transform_observation(item[0].tokens, item[0].channels, transform)
            for item in accepted
        ]
        for start in range(0, len(accepted), int(batch_size)):
            if time.perf_counter() - started > float(max_wall_seconds):
                truncated = True
                break
            chunk = accepted[start:start + int(batch_size)]
            if not chunk:
                continue
            chunk_observations = transformed_observations[start:start + int(batch_size)]
            max_actions = max(
                max(item[0].rows.shape[0], item[1][0].shape[0]) for item in chunk
            )
            n = len(chunk)
            base_rows = np.zeros((n, max_actions, 8), dtype=np.int64)
            target_rows = np.zeros((n, max_actions, 8), dtype=np.int64)
            player_ids = np.zeros(n, dtype=np.int64)
            for index, (sample, (target, _perm)) in enumerate(chunk):
                base_rows[index, : sample.rows.shape[0]] = sample.rows
                target_rows[index, : target.shape[0]] = target
                player_ids[index] = sample.actor
            base_tokens = torch.as_tensor(
                np.stack([item[0].tokens for item in chunk]), dtype=torch.long, device=resolved,
            )
            base_channels = torch.as_tensor(
                np.stack([item[0].channels for item in chunk]), dtype=torch.uint8, device=resolved,
            )
            new_tokens = torch.as_tensor(
                np.stack([obs[0] for obs in chunk_observations]), dtype=torch.long, device=resolved,
            )
            new_channels = torch.as_tensor(
                np.stack([obs[1] for obs in chunk_observations]), dtype=torch.uint8, device=resolved,
            )
            base_logprobs = _logprobs(
                policy, base_tokens, base_channels,
                torch.as_tensor(base_rows, device=resolved),
                torch.as_tensor(player_ids, device=resolved),
            )
            transformed_logprobs = _logprobs(
                policy, new_tokens, new_channels,
                torch.as_tensor(target_rows, device=resolved),
                torch.as_tensor(
                    [transform_seat(transform, sample.actor) for sample, _ in chunk],
                    device=resolved,
                ),
            )
            for index, (sample, (target, permutation)) in enumerate(chunk):
                count = sample.rows.shape[0]
                # Both candidate lists are padded to the chunk maximum; only the
                # real legal prefix participates.  Padding never enters the KL.
                comparison = aligned_policy_comparison(
                    base_logprobs[index, :count], transformed_logprobs[index, :target.shape[0]],
                    permutation, action_unit=ACTION_UNIT_RESTRICTED,
                )
                records.append({
                    "deal_id": sample.deal_id,
                    "token_step": int(sample.token_step),
                    "transform": transform.name if not isinstance(transform, SeatRotation)
                    else f"seat_rotation_{transform.k}",
                    "actor": int(sample.actor),
                    "phase": sample.phase,
                    "action_kind": sample.phase,
                    "level_rank": int(sample.level_rank),
                    "leader_seat": int(sample.leader_seat),
                    "tribute_mode": sample.tribute_mode,
                    "action_unit": comparison.action_unit,
                    "kl": float(comparison.kl),
                    "max_prob_diff": float(comparison.max_prob_diff),
                    "mean_prob_diff": float(comparison.mean_prob_diff),
                    "argmax_agreement": int(comparison.argmax_agreement),
                    "candidate_count": int(comparison.candidate_count),
                    "zero_probability_count": int(comparison.zero_probability_count),
                    "bucket": bucket_key(
                        phase=sample.phase, action_kind=sample.phase,
                        level_rank=sample.level_rank, leader_seat=sample.leader_seat,
                        tribute_mode=sample.tribute_mode,
                    ),
                })
    learner_seconds = time.perf_counter() - learner_started
    if not records:
        if not truncated:
            raise SymmetryDiagnosticError("no transform was defined on any collected sample")
        aggregate = {
            "schema": PHASE5_DIAGNOSTIC_SCHEMA,
            "sample_count": 0,
            "deal_count": 0,
            "deals": [],
            "finite_sample_count": 0,
            "non_finite_kl_count": 0,
            "full_macro_sample_count": 0,
            "restricted_candidate_sample_count": 0,
            "mean_kl": float("inf"),
            "median_kl": float("inf"),
            "max_kl": float("inf"),
            "mean_max_prob_diff": float("inf"),
            "mean_prob_diff": float("inf"),
            "argmax_agreement_rate": float("inf"),
            "argmax_disagreement_rate": float("inf"),
            "buckets": {},
        }
        decision = {
            "schema": PHASE5_DIAGNOSTIC_SCHEMA,
            "status": "INCONCLUSIVE",
            "augmentation_recommended": False,
            "reasons": ["budget_exhausted_before_comparisons"],
            "sample_count": 0,
            "deal_count": 0,
            "non_finite_kl_count": 0,
            "mean_kl": float("inf"),
            "argmax_disagreement_rate": float("inf"),
            "thresholds": dict(SYMMETRY_SIGNIFICANCE),
            "triggered_by": {
                "sample_count": False, "deal_count": False,
                "mean_kl": False, "argmax_disagreement": False,
            },
        }
    else:
        aggregate = aggregate_diagnostic(records)
        decision = decide_augmentation(aggregate, significance=SYMMETRY_SIGNIFICANCE)
    return {
        "schema": PHASE5_DIAGNOSTIC_SCHEMA,
        "device": str(resolved),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(resolved),
        "design": "phase5",
        "significance": dict(SYMMETRY_SIGNIFICANCE),
        "environment": environment,
        "collection_truncated": bool(environment.get("truncated")),
        "truncated": bool(truncated),
        "deal_meta": deal_meta,
        "transforms": sorted({record["transform"] for record in records}),
        "undefined_samples": undefined,
        "aggregate": aggregate,
        "decision": decision,
        "records": records,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(resolved)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(resolved)),
        "environment_seconds": float(environment["environment_seconds"]),
        "learner_seconds": float(learner_seconds),
        "total_seconds": float(time.perf_counter() - started),
    }


__all__ = [
    "PHASE5_DIAGNOSTIC_SCHEMA", "RecordedSample", "SymmetryDiagnosticError",
    "collect_recorded_samples", "replay_slot", "require_cuda",
    "run_symmetry_diagnostic", "transformed_candidate_rows", "tribute_mode",
]
