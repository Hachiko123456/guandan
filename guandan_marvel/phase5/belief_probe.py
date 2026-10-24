# SPDX-License-Identifier: MIT
"""Belief B0 baselines A/B/C/D, metrics, paired bootstrap and the gate decision.

The primary comparison is always the same test split and the same canonical
deal groups:

``improvement = B_error - D_error``

where the error is the mean absolute count error over the 45
(relative seat, printed rank) entries.  D is trained on the *complete causal
public history plus the current observation channels*; B is a rule-based
public-information proportional allocation.  A is the unconditional training
prior and C is a linear probe on the current channels only.

Everything trainable runs on CUDA in FP32.  ``allow_cpu=True`` exists only for
small deterministic unit-test fixtures; the B0 CLI never exposes it and the
formal runner fails fast instead of falling back to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from .design import (
    COUNT_CLASSES, MAX_COPIES_BY_RANK, NUM_PRINTED_RANKS, PHASE5_B0_PLAN,
    PHASE5_BELIEF_GATE_SCHEMA, PHASE5_BELIEF_PROBE_SCHEMA, PRINTED_RANK_LABELS,
    PUBLIC_FEATURE_WIDTH, RELATIVE_SEATS,
)
from .belief_data import decode_public_ledger, validate_public_features

NUM_RELATIVE_SEATS = len(RELATIVE_SEATS)
NUM_ENTRIES = NUM_RELATIVE_SEATS * NUM_PRINTED_RANKS
CHANNEL_WIDTH = 512
PUBLIC_CARD_WIDTH = 108
MAX_COPIES = np.asarray(MAX_COPIES_BY_RANK, dtype=np.float64)
# The 45 probe entries are (relative seat, printed rank).  The three relative
# seats are three distinct hands, each individually bounded by the physical
# per-rank maximum, so the bound is repeated unchanged for every seat.
MAX_COPIES_PER_ENTRY = np.tile(MAX_COPIES, NUM_RELATIVE_SEATS)


class BeliefProbeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Device gate
# ---------------------------------------------------------------------------


def require_cuda_device(device: str | torch.device) -> torch.device:
    """Resolve an explicit CUDA device; never silently fall back to CPU."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"B0 probes require an explicit CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is allowed")
    index = 0 if resolved.index is None else resolved.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is not available")
    # ``is_available`` / ``device_count`` resolve the device count through NVML
    # and deliberately do not create a CUDA context.  Memory statistics
    # (``reset_peak_memory_stats``) are only defined once that context exists:
    # in a fresh process the uninitialised call fails with
    # "RuntimeError: Invalid device argument".  Initialise explicitly here so
    # every CUDA entry point of this module starts from a valid device; never
    # fall back to CPU.
    torch.cuda.init()
    return torch.device("cuda", index)


def _resolve_training_device(
    device: str | torch.device, *, allow_cpu: bool,
) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        return require_cuda_device(resolved)
    if resolved.type == "cpu" and allow_cpu:
        return resolved  # deterministic unit-test fixtures only
    raise ValueError("formal B0 probes must run on CUDA; CPU is test-only")


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def clamp_predictions(predictions: np.ndarray) -> np.ndarray:
    """Clamp to ``[0, physical per-rank maximum]`` preserving the input shape.

    Both ``[..., 45]`` and ``[..., 3, 15]`` are accepted and the result has
    exactly the caller's shape (the previous implementation flattened a
    ``[..., 3, 15]`` input to ``[..., 45]``).  The maxima are the physical
    copies in the two-deck shoe: 8 for each of the 13 normal ranks and 2 for
    each joker rank.
    """
    array = np.asarray(predictions, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("predictions must end in 45 or [3, 15] entries")
    original_shape = array.shape
    if array.shape[-1] == NUM_ENTRIES:
        flat = array
    elif array.shape[-2:] == (NUM_RELATIVE_SEATS, NUM_PRINTED_RANKS):
        flat = array.reshape(array.shape[:-2] + (NUM_ENTRIES,))
    else:
        raise ValueError("predictions must end in 45 or [3, 15] entries")
    reshaped = flat.reshape(flat.shape[:-1] + (NUM_RELATIVE_SEATS, NUM_PRINTED_RANKS))
    clipped = np.minimum(np.maximum(reshaped, 0.0), MAX_COPIES[None, :])
    return clipped.reshape(original_shape)


def _as_counts(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.shape[-1] == NUM_ENTRIES:
        return values.reshape(values.shape[:-1] + (NUM_RELATIVE_SEATS, NUM_PRINTED_RANKS))
    if values.shape[-2:] == (NUM_RELATIVE_SEATS, NUM_PRINTED_RANKS):
        return values
    raise ValueError("count array must end in 45 or [3, 15] entries")


def count_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Pre-registered count metrics (lower is better except accuracy/ECE)."""
    pred = clamp_predictions(predictions)
    truth = _as_counts(labels)
    pred = pred.reshape(truth.shape)
    error = np.abs(pred - truth)
    count_mae = float(error.mean())
    count_l1 = float(error.reshape(error.shape[0], -1).sum(axis=1).mean())
    rank_error = np.abs(
        pred.sum(axis=1) - truth.sum(axis=1)
    )  # [N, 15]
    rank_mae = float(rank_error.mean())
    rounded = np.rint(pred)
    exact = float((rounded == truth).mean())
    conservation = np.abs(
        pred.reshape(pred.shape[0], -1).sum(axis=1)
        - truth.reshape(truth.shape[0], -1).sum(axis=1)
    )
    conservation_error = float(conservation.mean())
    # ECE over nine count bins (0..8); empty bins contribute nothing.
    bins = np.clip(np.rint(pred), 0, COUNT_CLASSES - 1).astype(np.int64)
    total = float(bins.size)
    ece = 0.0
    if total:
        for value in range(COUNT_CLASSES):
            mask = bins == value
            count = int(mask.sum())
            if count:
                ece += (count / total) * abs(float(pred[mask].mean()) - float(truth[mask].mean()))
    result = {
        "count_mae": count_mae,
        "count_l1": count_l1,
        "rank_mae": rank_mae,
        "exact_count_accuracy": exact,
        "conservation_error": conservation_error,
        "calibration_ece": float(ece),
        "sample_count": int(truth.shape[0]),
        "entry_count": int(truth.size),
    }
    if groups is not None:
        group_array = np.asarray([str(group) for group in groups])
        if group_array.shape[0] != truth.shape[0]:
            raise ValueError("groups must match the sample count")
        per_group: dict[str, float] = {}
        sample_error = error.reshape(error.shape[0], -1).mean(axis=1)
        for group in sorted(set(group_array.tolist())):
            per_group[group] = float(sample_error[group_array == group].mean())
        result["per_group_count_mae"] = per_group
    return result


def paired_bootstrap_ci(
    values_by_group: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Cluster bootstrap over canonical deals (one paired value per deal)."""
    groups = sorted(str(group) for group in values_by_group)
    values = np.asarray([float(values_by_group[group]) for group in groups], dtype=np.float64)
    if values.size == 0:
        raise ValueError("bootstrap requires at least one group")
    if not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite")
    count = int(resamples)
    if count <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.default_rng(int(seed))
    if values.size == 1:
        mean = float(values[0])
        return {
            "mean": mean, "lower": mean, "upper": mean,
            "group_count": 1, "resamples": count, "seed": int(seed),
            "confidence": float(confidence),
        }
    indices = rng.integers(0, values.size, size=(count, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "mean": float(values.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "group_count": int(values.size),
        "resamples": count,
        "seed": int(seed),
        "confidence": float(confidence),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaselineA:
    """Unconditional training-set prior per (relative seat, printed rank)."""

    prior: np.ndarray

    @classmethod
    def fit(cls, labels: np.ndarray) -> "BaselineA":
        truth = _as_counts(labels)
        return cls(prior=truth.mean(axis=0))

    def predict(self, sample_count: int) -> np.ndarray:
        return np.broadcast_to(
            self.prior[None, :, :], (int(sample_count), NUM_RELATIVE_SEATS, NUM_PRINTED_RANKS),
        ).copy()


@dataclass(frozen=True, slots=True)
class BaselineB:
    """Public-ledger lower-bounded proportional allocation (no training).

    Derivation (v2 ledger):

        R_s      published remaining count of relative seat s
        K[s, r]  publicly known cards currently held by seat s
        h_s      R_s - sum_r K[s, r]          (unknown hand capacity)
        u_r      residual inventory by rank   (unknown hands + unknown escrow)
        X        escrow cards with unknown identity

    The ledger invariant is ``sum_r u_r == sum_s h_s + X``.  All unknown
    capacity is exchangeable, so the per-rank unknown mass assigned to hands is
    ``u_r * H / (H + X)`` with ``H = sum_s h_s``; splitting it by each seat's
    hand capacity gives

        pred[s, r] = K[s, r] + h_s * u_r / sum_r u_r

    which is never below K and sums over ranks to exactly R_s.  When the unknown
    pool is empty the legal known counts K are returned unchanged.  No hidden
    hand, oracle label or future event participates.
    """

    def predict(self, public: np.ndarray) -> np.ndarray:
        raw = np.asarray(public, dtype=np.float64)
        if raw.ndim != 2:
            raise ValueError("public features must be a [N, width] matrix")
        features = validate_public_features(raw)
        # Every ledger entry is a *physical card count*: it must be a finite,
        # non-negative integer.  A NaN/Inf or a fractional count would make the
        # proportional allocation meaningless and is rejected before any state
        # is touched.
        if not np.isfinite(features).all():
            raise ValueError("public ledger contains a non-finite count")
        fractional = np.abs(features - np.rint(features)) > 1e-9
        if bool(fractional.any()):
            raise ValueError("public ledger counts must be integers")
        decoded = decode_public_ledger(features)
        remaining = decoded["remaining"]
        known = decoded["known"]
        unknown = decoded["unknown"]
        escrow_unknown = decoded["escrow_unknown"]
        if np.any(remaining < 0) or np.any(known < 0) or np.any(unknown < 0):
            raise ValueError("public ledger contains negative counts")
        if np.any(escrow_unknown < 0):
            raise ValueError("public ledger contains a negative escrow capacity")
        max_copies = np.asarray(MAX_COPIES_BY_RANK, dtype=np.float64)
        # Global entity conservation.  ``known`` is indexed by relative seat, so
        # checking each row against the physical per-rank maximum alone does not
        # prevent the *same* physical copies from being counted in several seats.
        # For every printed rank r the 45-entry ledger must satisfy
        #     sum_s K[s, r] <= C_r      (known holders cannot exceed the shoe)
        #     sum_s K[s, r] + u[r] <= C_r  (unknown inventory is disjoint from K)
        # with C_r = 8 for normal ranks and 2 for each joker.
        known_by_rank = known.sum(axis=1)  # [N, 15]
        if np.any(known_by_rank > max_copies[None, :]):
            raise ValueError(
                "known-held cards across seats exceed the global physical per-rank maximum"
            )
        if np.any(known_by_rank + unknown > max_copies[None, :]):
            raise ValueError(
                "known-held plus residual unknown cards exceed the global physical "
                "per-rank maximum"
            )
        if np.any(known > max_copies[None, None, :]):
            raise ValueError("known-held count exceeds the physical per-rank maximum")
        hand_capacity = remaining - known.sum(axis=2)
        if np.any(hand_capacity < 0):
            raise ValueError("known-held lower bound exceeds a published hand size")
        unknown_total = unknown.sum(axis=1)
        capacity_total = hand_capacity.sum(axis=1) + escrow_unknown
        if np.any(np.abs(unknown_total - capacity_total) > 1e-9):
            raise ValueError(
                "public ledger does not conserve: sum(u) must equal H + unknown escrow"
            )
        allocation = np.zeros_like(known)
        positive = unknown_total > 0.0
        if np.any(positive):
            allocation[positive] = (
                hand_capacity[positive, :, None]
                * unknown[positive, None, :]
                / unknown_total[positive, None, None]
            )
        prediction = known + allocation
        if not np.isfinite(prediction).all():
            raise ValueError("baseline B produced a non-finite prediction")
        return clamp_predictions(prediction)


def build_probe_features(
    matrix: Mapping[str, np.ndarray], kind: str,
) -> np.ndarray:
    """Build the exact model-visible probe input for baseline ``C`` or ``D``.

    Labels are never accepted here: only channels, public card bits and public
    balance features participate.
    """
    channels = np.asarray(matrix["channels"], dtype=np.float32)
    if channels.ndim != 2 or channels.shape[1] != CHANNEL_WIDTH:
        raise ValueError("channels must be [N, 512]")
    if kind == "C":
        return np.ascontiguousarray(channels)
    if kind == "D":
        cards = np.asarray(matrix["public_cards"], dtype=np.float32)
        public = np.asarray(matrix["public"], dtype=np.float32)
        if cards.ndim != 2 or cards.shape[1] != PUBLIC_CARD_WIDTH:
            raise ValueError("public_cards must be [N, 108]")
        if public.ndim != 2:
            raise ValueError("public must be a [N, width] matrix")
        validate_public_features(public)
        if cards.shape[0] != channels.shape[0] or public.shape[0] != channels.shape[0]:
            raise ValueError("probe features must have the same sample count")
        return np.ascontiguousarray(np.concatenate([channels, cards, public], axis=1))
    raise ValueError(f"unknown probe kind {kind!r}")


@dataclass
class ProbeFit:
    """One trained linear probe and its audit record."""

    model: nn.Linear
    mean: torch.Tensor
    std: torch.Tensor
    max_copies: torch.Tensor
    device: torch.device
    seed: int
    best_epoch: int
    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    optimizer_updates: int = 0
    parameter_changed: bool = False
    finite_loss: bool = True
    truncated: bool = False
    wall_seconds: float = 0.0
    peak_allocated_bytes: int = 0

    def predict(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        tensor = (tensor - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(tensor)
            logits = torch.minimum(
                torch.maximum(logits, torch.zeros_like(logits)),
                self.max_copies.reshape(1, -1),
            )
        return logits.detach().cpu().numpy().astype(np.float64)

    def audit(self) -> dict[str, Any]:
        return {
            "schema": PHASE5_BELIEF_PROBE_SCHEMA,
            "device": str(self.device),
            "seed": int(self.seed),
            "dtype": "float32",
            "input_dim": int(self.model.in_features),
            "output_dim": int(self.model.out_features),
            "best_epoch": int(self.best_epoch),
            "epochs_run": len(self.train_losses),
            "optimizer_updates": int(self.optimizer_updates),
            "parameter_changed": bool(self.parameter_changed),
            "finite_loss": bool(self.finite_loss),
            "truncated": bool(self.truncated),
            "wall_seconds": float(self.wall_seconds),
            "peak_allocated_bytes": int(self.peak_allocated_bytes),
            "train_losses": [float(value) for value in self.train_losses],
            "validation_losses": [float(value) for value in self.validation_losses],
        }


def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    device: str | torch.device,
    seed: int,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 3.0e-3,
    weight_decay: float = 1.0e-2,
    patience: int = 6,
    max_seconds: float | None = None,
    allow_cpu: bool = False,
) -> ProbeFit:
    """Train one FP32 linear count probe on CUDA (CPU only in unit fixtures)."""
    resolved = _resolve_training_device(device, allow_cpu=allow_cpu)
    train_x = np.asarray(train_features, dtype=np.float32)
    train_y = _as_counts(train_labels).reshape(-1, NUM_ENTRIES).astype(np.float32)
    val_x = np.asarray(validation_features, dtype=np.float32)
    val_y = _as_counts(validation_labels).reshape(-1, NUM_ENTRIES).astype(np.float32)
    if train_x.ndim != 2 or val_x.ndim != 2 or train_x.shape[1] != val_x.shape[1]:
        raise ValueError("probe feature matrices must be [N, D] with matching D")
    if train_x.shape[0] != train_y.shape[0] or val_x.shape[0] != val_y.shape[0]:
        raise ValueError("probe features and labels must have matching sample counts")
    if train_x.shape[0] == 0 or val_x.shape[0] == 0:
        raise ValueError("probe training requires non-empty train and validation splits")
    if int(epochs) <= 0 or int(batch_size) <= 0 or int(patience) <= 0:
        raise ValueError("epochs, batch_size and patience must be positive")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if max_seconds is not None and (
        not math.isfinite(float(max_seconds)) or float(max_seconds) <= 0
    ):
        raise ValueError("max_seconds must be finite and positive")

    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
        torch.cuda.manual_seed_all(int(seed))
    torch.manual_seed(int(seed))
    started = time.perf_counter()

    train_t = torch.as_tensor(train_x, device=resolved)
    train_labels_t = torch.as_tensor(train_y, device=resolved)
    val_t = torch.as_tensor(val_x, device=resolved)
    val_labels_t = torch.as_tensor(val_y, device=resolved)
    mean = train_t.mean(dim=0)
    std = torch.clamp(train_t.std(dim=0, unbiased=False), min=1e-6)
    train_t = (train_t - mean) / std
    val_t = (val_t - mean) / std

    model = nn.Linear(int(train_x.shape[1]), NUM_ENTRIES).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay),
    )
    max_copies = torch.as_tensor(
        MAX_COPIES_PER_ENTRY, dtype=torch.float32, device=resolved,
    ).reshape(1, NUM_ENTRIES)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    updates = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    finite = True
    truncated = False

    def loss_for(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = model(features)
        clamped = torch.minimum(torch.maximum(logits, torch.zeros_like(logits)), max_copies)
        return torch.nn.functional.l1_loss(clamped, labels, reduction="mean")

    for epoch in range(int(epochs)):
        if max_seconds is not None and time.perf_counter() - started > float(max_seconds):
            truncated = True
            break
        model.train()
        order = torch.randperm(train_t.shape[0], generator=generator).to(resolved)
        running = 0.0
        batches = 0
        for start in range(0, train_t.shape[0], int(batch_size)):
            if max_seconds is not None and time.perf_counter() - started > float(max_seconds):
                truncated = True
                break
            index = order[start:start + int(batch_size)]
            batch_x = train_t.index_select(0, index)
            batch_y = train_labels_t.index_select(0, index)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(batch_x, batch_y)
            if not bool(torch.isfinite(loss).item()):
                finite = False
                raise BeliefProbeError("probe produced a non-finite training loss")
            loss.backward()
            optimizer.step()
            updates += 1
            running += float(loss.detach().item())
            batches += 1
        if truncated:
            break
        model.eval()
        with torch.no_grad():
            validation_loss = loss_for(val_t, val_labels_t)
        validation_value = float(validation_loss.detach().item())
        train_value = running / max(batches, 1)
        train_losses.append(train_value)
        validation_losses.append(validation_value)
        if not math.isfinite(train_value) or not math.isfinite(validation_value):
            finite = False
            raise BeliefProbeError("probe produced a non-finite loss")
        if validation_value < best_loss - 1e-12:
            best_loss = validation_value
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= int(patience):
                break
    model.load_state_dict(best_state)
    model.eval()
    parameter_changed = any(
        not torch.equal(initial[name], value.detach())
        for name, value in model.state_dict().items()
    )
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
        peak = int(torch.cuda.max_memory_allocated(resolved))
    else:
        peak = 0
    return ProbeFit(
        model=model,
        mean=mean.detach().clone(),
        std=std.detach().clone(),
        max_copies=max_copies.detach().clone(),
        device=resolved,
        seed=int(seed),
        best_epoch=int(best_epoch),
        train_losses=train_losses,
        validation_losses=validation_losses,
        optimizer_updates=int(updates),
        parameter_changed=bool(parameter_changed),
        finite_loss=bool(finite),
        truncated=bool(truncated),
        wall_seconds=float(time.perf_counter() - started),
        peak_allocated_bytes=peak,
    )


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def split_indices(
    groups: Sequence[str], assignment: Mapping[str, str],
) -> dict[str, np.ndarray]:
    group_array = np.asarray([str(group) for group in groups])
    missing = sorted(set(group_array.tolist()) - set(assignment))
    if missing:
        raise ValueError(f"split assignment is missing groups: {missing[:3]}")
    result: dict[str, np.ndarray] = {}
    for name in ("train", "validation", "test"):
        mask = np.asarray([assignment[str(group)] == name for group in group_array], dtype=bool)
        result[name] = np.flatnonzero(mask)
    return result


def verify_group_disjoint(
    groups: Sequence[str], assignment: Mapping[str, str],
) -> bool:
    seen: dict[str, str] = {}
    for group in {str(value) for value in groups}:
        split = assignment.get(group)
        if split is None:
            return False
        if group in seen:
            return False
        seen[group] = split
    return True


# ---------------------------------------------------------------------------
# B0 evaluation and gate
# ---------------------------------------------------------------------------


def run_b0_probes(
    matrix: Mapping[str, np.ndarray],
    assignment: Mapping[str, str],
    *,
    device: str | torch.device,
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    max_probe_seconds: float,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    """Train A/B/C/D on one locked split and return the gate inputs."""
    resolved = _resolve_training_device(device, allow_cpu=allow_cpu)
    seeds = [int(seed) for seed in seeds]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("B0 requires at least three distinct pre-registered seeds")
    groups = np.asarray([str(group) for group in matrix["groups"]])
    labels = _as_counts(matrix["counts"])
    public = np.asarray(matrix["public"], dtype=np.float64)
    indices = split_indices(groups, assignment)
    train_idx, val_idx, test_idx = indices["train"], indices["validation"], indices["test"]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise BeliefProbeError("locked split has an empty train/validation/test partition")
    train_labels = labels[train_idx]
    validation_labels = labels[val_idx]
    test_labels = labels[test_idx]

    prior = BaselineA.fit(train_labels)
    a_test = prior.predict(len(test_idx))
    b_test = BaselineB().predict(public[test_idx])

    c_features = build_probe_features(matrix, "C")
    d_features = build_probe_features(matrix, "D")
    started = time.perf_counter()
    per_fit_seconds = max(1.0, float(max_probe_seconds) / float(2 * len(seeds)))
    c_fits: list[ProbeFit] = []
    d_fits: list[ProbeFit] = []
    c_predictions: list[np.ndarray] = []
    d_predictions: list[np.ndarray] = []
    for seed in seeds:
        remaining = float(max_probe_seconds) - (time.perf_counter() - started)
        c_fit = train_linear_probe(
            c_features[train_idx], train_labels,
            c_features[val_idx], validation_labels,
            device=resolved, seed=seed, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay,
            patience=patience,
            max_seconds=min(per_fit_seconds, max(remaining, 1.0)),
            allow_cpu=allow_cpu,
        )
        remaining = float(max_probe_seconds) - (time.perf_counter() - started)
        d_fit = train_linear_probe(
            d_features[train_idx], train_labels,
            d_features[val_idx], validation_labels,
            device=resolved, seed=seed, epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay,
            patience=patience,
            max_seconds=min(per_fit_seconds, max(remaining, 1.0)),
            allow_cpu=allow_cpu,
        )
        c_fits.append(c_fit)
        d_fits.append(d_fit)
        c_predictions.append(c_fit.predict(c_features[test_idx]))
        d_predictions.append(d_fit.predict(d_features[test_idx]))

    d_mean = np.mean(np.stack(d_predictions, axis=0), axis=0)
    c_mean = np.mean(np.stack(c_predictions, axis=0), axis=0)
    b_metrics = count_metrics(b_test, test_labels, groups=groups[test_idx])
    a_metrics = count_metrics(a_test, test_labels, groups=groups[test_idx])
    c_metrics = count_metrics(c_mean, test_labels, groups=groups[test_idx])
    d_metrics = count_metrics(d_mean, test_labels, groups=groups[test_idx])
    seed_improvements = [
        float(
            count_metrics(b_test, test_labels)["count_mae"]
            - count_metrics(prediction, test_labels)["count_mae"]
        )
        for prediction in d_predictions
    ]
    # Paired improvement per canonical deal: each deal contributes one value;
    # seeds are averaged inside the deal.
    improvement_by_group: dict[str, float] = {}
    test_groups = groups[test_idx]
    for group in sorted(set(test_groups.tolist())):
        mask = test_groups == group
        b_group = count_metrics(b_test[mask], test_labels[mask])["count_mae"]
        d_group = count_metrics(d_mean[mask], test_labels[mask])["count_mae"]
        improvement_by_group[group] = float(b_group - d_group)
    b_flat = b_test.reshape(-1, NUM_ENTRIES)
    d_flat = d_mean.reshape(-1, NUM_ENTRIES)
    truth_flat = test_labels.reshape(-1, NUM_ENTRIES)
    sample_improvements = (
        np.abs(b_flat - truth_flat).mean(axis=1)
        - np.abs(d_flat - truth_flat).mean(axis=1)
    )
    return {
        "schema": PHASE5_BELIEF_PROBE_SCHEMA,
        "device": str(resolved),
        "seeds": seeds,
        "split_indices": {name: values.tolist() for name, values in indices.items()},
        "fit_audits": {
            "C": [fit.audit() for fit in c_fits],
            "D": [fit.audit() for fit in d_fits],
        },
        "metrics": {
            "A": a_metrics,
            "B": b_metrics,
            "C": c_metrics,
            "D": d_metrics,
        },
        "seed_improvements": seed_improvements,
        "improvement_by_group": improvement_by_group,
        "bucket_sample_improvements": sample_improvements.tolist(),
        "test_groups": sorted(set(test_groups.tolist())),
        "test_labels": test_labels.tolist(),
        "test_groups_array": test_groups.tolist(),
        "bucket_metadata": {
            key: [str(value) for value in matrix[key][test_idx]]
            for key in ("phase", "level_rank", "leader_seat", "tribute_mode")
            if key in matrix
        },
        "probe_seconds": float(time.perf_counter() - started),
        "probe_truncated": any(fit.truncated for fit in c_fits + d_fits),
        "all_parameters_changed": all(
            fit.parameter_changed for fit in c_fits + d_fits
        ),
        "all_losses_finite": all(fit.finite_loss for fit in c_fits + d_fits),
        "peak_allocated_bytes": max(
            [fit.peak_allocated_bytes for fit in c_fits + d_fits] or [0]
        ),
    }


def evaluate_bucket_results(
    probe: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    metadata = probe.get("bucket_metadata", {})
    groups = np.asarray(probe["test_groups_array"])
    # Reconstruct per-sample improvement from the per-group means is impossible;
    # the caller supplies a per-sample vector in ``bucket_improvements``.
    per_sample = np.asarray(probe.get("bucket_sample_improvements", []))
    if per_sample.size != groups.size:
        raise ValueError("bucket evaluation requires one improvement per test sample")
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for name, values in metadata.items():
        values = np.asarray([str(value) for value in values])
        if values.shape[0] != groups.shape[0]:
            raise ValueError("bucket metadata must match the test sample count")
        bucket: dict[str, dict[str, Any]] = {}
        for key in sorted(set(values.tolist())):
            mask = values == key
            sample_count = int(mask.sum())
            if sample_count < 1:
                continue
            by_group: dict[str, float] = {}
            for group in sorted(set(groups[mask].tolist())):
                group_mask = mask & (groups == group)
                by_group[group] = float(per_sample[group_mask].mean())
            ci = paired_bootstrap_ci(
                by_group,
                resamples=int(bootstrap["resamples"]),
                seed=int(bootstrap["seed"]),
                confidence=float(bootstrap["confidence"]),
            )
            bucket[key] = {"sample_count": sample_count, **ci}
        results[name] = bucket
    return results


def severe_buckets(
    buckets: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    min_samples: int = 30,
) -> list[dict[str, Any]]:
    severe: list[dict[str, Any]] = []
    for name, values in buckets.items():
        for key, entry in values.items():
            if int(entry["sample_count"]) >= int(min_samples) and float(entry["upper"]) < 0.0:
                severe.append({
                    "bucket": name, "key": key,
                    "sample_count": int(entry["sample_count"]),
                    "upper": float(entry["upper"]),
                })
    return severe


def b0_gate(
    *,
    probe: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    epsilon: float,
    sufficiency: Mapping[str, Any],
    leakage_pass: bool,
    group_disjoint: bool,
    bucket_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    design_hash: str,
    source_hash: str,
    config_hash: str,
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered PASS / REJECT / INCONCLUSIVE rule."""
    improvements = np.asarray(list(probe["improvement_by_group"].values()), dtype=np.float64)
    ci = paired_bootstrap_ci(
        probe["improvement_by_group"],
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
        confidence=float(bootstrap["confidence"]),
    )
    seed_improvements = [float(value) for value in probe["seed_improvements"]]
    severe = severe_buckets(bucket_results)
    reasons: list[str] = []
    required = [
        ("completed_deals", int(sufficiency.get("completed_deals", 0))),
        ("test_deals", int(sufficiency.get("test_deals", 0))),
        ("test_samples", int(sufficiency.get("test_samples", 0))),
    ]
    if required[0][1] < int(PHASE5_B0_PLAN["max_deals"]):
        reasons.append("insufficient_completed_deals")
    if required[1][1] < int(PHASE5_B0_PLAN["gate"]["sufficiency"]["min_test_deals"]):
        reasons.append("insufficient_test_deals")
    if required[2][1] < int(PHASE5_B0_PLAN["gate"]["sufficiency"]["min_test_samples"]):
        reasons.append("insufficient_test_samples")
    if int(sufficiency.get("test_level_ranks", 0)) < int(
        PHASE5_B0_PLAN["gate"]["sufficiency"]["min_test_level_ranks"]
    ):
        reasons.append("coverage_single_point_level")
    if int(sufficiency.get("test_leader_seats", 0)) < int(
        PHASE5_B0_PLAN["gate"]["sufficiency"]["min_test_leader_seats"]
    ):
        reasons.append("coverage_single_point_leader")
    if int(sufficiency.get("test_tribute_modes", 0)) < int(
        PHASE5_B0_PLAN["gate"]["sufficiency"]["min_test_tribute_modes"]
    ):
        reasons.append("coverage_single_point_tribute")
    if not leakage_pass:
        reasons.append("leakage_check_failed")
    if not group_disjoint:
        reasons.append("group_split_contamination")
    if bool(probe.get("probe_truncated")):
        reasons.append("probe_budget_exhausted")
    if not bool(probe.get("all_losses_finite", True)):
        reasons.append("non_finite_probe_loss")
    sufficient = not reasons
    if not sufficient:
        verdict = "INCONCLUSIVE"
    elif ci["lower"] > float(epsilon) and all(value > 0.0 for value in seed_improvements) and not severe:
        verdict = "PASS"
    elif ci["upper"] <= float(epsilon):
        verdict = "REJECT"
    else:
        verdict = "INCONCLUSIVE"
        if ci["lower"] <= float(epsilon) < ci["upper"]:
            reasons.append("ci_straddles_epsilon")
        if any(value <= 0.0 for value in seed_improvements):
            reasons.append("seed_instability")
        if severe:
            reasons.append("severe_bucket_degradation")
    return {
        "schema": PHASE5_BELIEF_GATE_SCHEMA,
        "verdict": verdict,
        "design_hash": str(design_hash),
        "source_hash": str(source_hash),
        "config_hash": str(config_hash),
        "dataset": dict(dataset),
        "epsilon": float(epsilon),
        "primary": {
            "metric": "count_mae",
            "improvement": "B_error - D_error",
            "mean_improvement": float(improvements.mean()) if improvements.size else 0.0,
            "ci": ci,
            "seed_improvements": seed_improvements,
        },
        "metrics": dict(probe["metrics"]),
        "buckets": {name: dict(values) for name, values in bucket_results.items()},
        "severe_buckets": severe,
        "sufficiency": dict(sufficiency),
        "leakage_pass": bool(leakage_pass),
        "group_disjoint": bool(group_disjoint),
        "reasons": sorted(set(reasons)),
        "probe_audit": {
            "device": probe.get("device"),
            "seeds": probe.get("seeds"),
            "probe_seconds": probe.get("probe_seconds"),
            "probe_truncated": probe.get("probe_truncated"),
            "all_parameters_changed": probe.get("all_parameters_changed"),
            "all_losses_finite": probe.get("all_losses_finite"),
            "peak_allocated_bytes": probe.get("peak_allocated_bytes"),
            "fit_audits": probe.get("fit_audits"),
        },
    }


__all__ = [
    "BaselineA", "BaselineB", "BeliefProbeError", "NUM_ENTRIES",
    "ProbeFit", "b0_gate", "build_probe_features", "clamp_predictions",
    "count_metrics", "evaluate_bucket_results", "paired_bootstrap_ci",
    "require_cuda_device", "run_b0_probes", "severe_buckets", "split_indices",
    "train_linear_probe", "verify_group_disjoint",
]
