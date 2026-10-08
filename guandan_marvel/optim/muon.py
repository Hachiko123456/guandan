# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/optim/muon.py.
# muon.py (Python 3.8+)
from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer

from .common import MultiOptimizer


def _newton_schulz_5(
    G: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    cast_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Approximate the polar/zeroth-power factor of G (i.e., map UΣV^T -> UV^T)
    using the "quintic" Newton–Schulz iteration used by Muon. :contentReference[oaicite:1]{index=1}

    Expects a 2D matrix.
    """
    if G.ndim != 2:
        raise ValueError(f"_newton_schulz_5 expects a 2D tensor, got shape={tuple(G.shape)}")

    a, b, c = coeffs

    # Cast for the Newton–Schulz matmuls (use bf16 for speed on tensor cores if desired).
    X = G.to(dtype = cast_dtype)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True

    # Normalize by Frobenius norm for stability
    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)

    if transposed:
        X = X.t()

    return X


def _newton_schulz_5_batched(
    G: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    cast_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Batched variant of `_newton_schulz_5` for inputs shaped `[N, M, K]`."""
    if G.ndim != 3:
        raise ValueError(f"_newton_schulz_5_batched expects a 3D tensor, got shape={tuple(G.shape)}")

    a, b, c = coeffs

    # Cast for the Newton–Schulz matmuls (use bf16 for speed on tensor cores if desired).
    X = G.to(dtype = cast_dtype)
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.transpose(-1, -2)
        transposed = True

    # Normalize each matrix by Frobenius norm for stability.
    X = X / (X.norm(dim = (-2, -1), keepdim = True) + eps)

    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)

    if transposed:
        X = X.transpose(-1, -2)

    return X


def _as_2d(p: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """
    View `t` as a 2D matrix compatible with Muon.
    For conv weights (out, in, kH, kW), we flatten to (out, in*kH*kW). :contentReference[oaicite:3]{index=3}
    """
    if p.ndim < 2:
        raise ValueError(f"Muon expects parameters with ndim>=2 (got ndim={p.ndim}).")
    if p.ndim == 2:
        return t, lambda x: x
    out_dim = p.size(0)
    return t.reshape(out_dim, -1), lambda x: x.reshape_as(p)


def _muon_scale_factor(
    out_dim: int,
    in_dim: int,
    *,
    adjust_lr_fn: str,
    adamw_beta1: float = 0.9,
) -> float:
    """
    Two common choices:
      - "original": sqrt(out/in) shape scaling (classic Muon implementations). :contentReference[oaicite:4]{index=4}
      - "match_rms_adamw": sqrt(max(out,in)) * sqrt((1-b1)/(1+b1)) (RMS-matching style). :contentReference[oaicite:5]{index=5}
    """
    if adjust_lr_fn == "original":
        return math.sqrt(float(out_dim) / float(in_dim))
    if adjust_lr_fn == "match_rms_adamw":
        # Matches AdamW-like update RMS in expectation (common heuristic for sharing Adam-ish LRs). :contentReference[oaicite:6]{index=6}
        return math.sqrt(float(max(out_dim, in_dim))) * math.sqrt((1.0 - adamw_beta1) / (1.0 + adamw_beta1))
    raise ValueError(f"Unknown adjust_lr_fn={adjust_lr_fn!r} (use 'original' or 'match_rms_adamw').")


class Muon(Optimizer):
    """
    Muon (MomentUm Orthogonalized by Newton–Schulz) for matrix parameters. :contentReference[oaicite:7]{index=7}

    Notes:
      - Intended for hidden weight matrices (ndim>=2). Embeddings / heads / biases / norms often use AdamW instead. :contentReference[oaicite:8]{index=8}
      - Uses decoupled weight decay: p *= (1 - lr*wd).
      - Momentum is EMA-style: m <- beta*m + (1-beta)*g (matches common Muon refs). :contentReference[oaicite:9]{index=9}
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        adjust_lr_fn: str = "match_rms_adamw",
        adamw_beta1_for_rms_match: float = 0.9,
        grad_none_as_zero: bool = False,
        cast_dtype: torch.dtype = torch.float32,
        batch_by_shape: bool = True,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        defaults = dict(
            lr = lr,
            momentum = momentum,
            weight_decay = weight_decay,
            nesterov = nesterov,
            ns_steps = ns_steps,
            ns_coefficients = ns_coefficients,
            eps = eps,
            adjust_lr_fn = adjust_lr_fn,
            adamw_beta1_for_rms_match = adamw_beta1_for_rms_match,
            grad_none_as_zero = grad_none_as_zero,
            cast_dtype = cast_dtype,
            batch_by_shape = batch_by_shape,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["momentum"]
            wd: float = group["weight_decay"]
            nesterov: bool = group["nesterov"]
            ns_steps: int = group["ns_steps"]
            ns_coeffs: Tuple[float, float, float] = group["ns_coefficients"]
            eps: float = group["eps"]
            adjust_lr_fn: str = group["adjust_lr_fn"]
            b1: float = group["adamw_beta1_for_rms_match"]
            grad_none_as_zero: bool = group["grad_none_as_zero"]
            cast_dtype: torch.dtype = group["cast_dtype"]
            batch_by_shape: bool = group["batch_by_shape"]

            if not batch_by_shape:
                for p in group["params"]:
                    if p.grad is None:
                        if not grad_none_as_zero:
                            continue
                        grad = torch.zeros_like(p)
                    else:
                        grad = p.grad

                    if grad.is_sparse:
                        raise RuntimeError("Muon does not support sparse gradients.")

                    # Decoupled weight decay
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    # Momentum buffer (EMA-style)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    m = state["momentum_buffer"]
                    # m <- beta*m + (1-beta)*g
                    m.lerp_(
                        grad,
                        1.0 - beta,
                    )  # equivalent to: m = beta*m + (1-beta)*grad :contentReference[oaicite:10]{index=10}

                    if nesterov:
                        update = torch.lerp(
                            grad,
                            m,
                            beta,
                        )  # (1-beta)*grad + beta*m :contentReference[oaicite:11]{index=11}
                    else:
                        update = m

                    # Orthogonalize (matrix-wise)
                    upd2d, unflatten = _as_2d(p, update)
                    out_dim, in_dim = int(upd2d.size(0)), int(upd2d.size(1))

                    orth2d = _newton_schulz_5(
                        upd2d,
                        steps = ns_steps,
                        eps = eps,
                        coeffs = ns_coeffs,
                        cast_dtype = cast_dtype,
                    ).to(dtype = p.dtype)

                    scale = _muon_scale_factor(
                        out_dim,
                        in_dim,
                        adjust_lr_fn = adjust_lr_fn,
                        adamw_beta1 = b1,
                    )
                    upd = unflatten(orth2d * scale)

                    # Optional per-parameter LR multiplier (handy in some speedrun setups)
                    lr_mul = float(getattr(p, "lr_mul", 1.0))
                    p.add_(upd, alpha = -(lr * lr_mul))
                continue

            # Batched orthogonalization path: group matrices by canonical 2D shape and process as a
            # single batched matmul workload (reduces kernel-launch overhead for many small mats).
            batches: dict[tuple[int, int], dict[str, Any]] = {}
            for p in group["params"]:
                if p.grad is None:
                    if not grad_none_as_zero:
                        continue
                    grad = torch.zeros_like(p)
                else:
                    grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                m = state["momentum_buffer"]
                m.lerp_(grad, 1.0 - beta)

                update = torch.lerp(grad, m, beta) if nesterov else m
                upd2d, unflatten = _as_2d(p, update)
                out_dim, in_dim = int(upd2d.size(0)), int(upd2d.size(1))

                # Canonicalize to (rows<=cols) so the batched kernel doesn't need per-matrix transposes.
                canonical = upd2d
                transposed = False
                if out_dim > in_dim:
                    canonical = upd2d.t()
                    transposed = True

                key = (int(canonical.size(0)), int(canonical.size(1)))
                entry = batches.setdefault(key, {"tensors": [], "metas": []})
                entry["tensors"].append(canonical)
                entry["metas"].append(
                    (
                        p,
                        unflatten,
                        transposed,
                        out_dim,
                        in_dim,
                        float(getattr(p, "lr_mul", 1.0)),
                        p.dtype,
                    )
                )

            for entry in batches.values():
                tensors = entry["tensors"]
                metas = entry["metas"]
                if not tensors:
                    continue

                if len(tensors) == 1:
                    p, unflatten, transposed, out_dim, in_dim, lr_mul, param_dtype = metas[0]
                    orth2d = _newton_schulz_5(
                        tensors[0],
                        steps = ns_steps,
                        eps = eps,
                        coeffs = ns_coeffs,
                        cast_dtype = cast_dtype,
                    ).to(dtype = param_dtype)
                    if transposed:
                        orth2d = orth2d.t()

                    scale = _muon_scale_factor(
                        out_dim,
                        in_dim,
                        adjust_lr_fn = adjust_lr_fn,
                        adamw_beta1 = b1,
                    )
                    upd = unflatten(orth2d * scale)
                    p.add_(upd, alpha = -(lr * lr_mul))
                    continue

                batch = torch.stack([t.to(dtype = cast_dtype) for t in tensors], dim = 0)
                orth_batch = _newton_schulz_5_batched(
                    batch,
                    steps = ns_steps,
                    eps = eps,
                    coeffs = ns_coeffs,
                    cast_dtype = cast_dtype,
                )
                first_dtype = metas[0][6]
                if all(param_dtype == first_dtype for *_rest, param_dtype in metas):
                    orth_batch_cast = orth_batch.to(dtype = first_dtype)
                else:
                    orth_batch_cast = orth_batch

                for idx, (p, unflatten, transposed, out_dim, in_dim, lr_mul, param_dtype) in enumerate(metas):
                    orth2d = orth_batch_cast[idx]
                    if orth2d.dtype != param_dtype:
                        orth2d = orth2d.to(dtype = param_dtype)
                    if transposed:
                        orth2d = orth2d.transpose(-1, -2)

                    scale = _muon_scale_factor(
                        out_dim,
                        in_dim,
                        adjust_lr_fn = adjust_lr_fn,
                        adamw_beta1 = b1,
                    )
                    upd = unflatten(orth2d * scale)
                    p.add_(upd, alpha = -(lr * lr_mul))

        return loss
