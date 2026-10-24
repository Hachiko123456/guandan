# SPDX-License-Identifier: MIT
"""DMC regression learner (stage 3).

One ``train_step`` is exactly:

.. code-block:: text

    batch  = replay.sample(batch_size, rng)               # uniform, no replacement
    q      = Q(s, a_chosen)                               # one candidate per row
    loss   = MSE(q, z)  (+ ntp_weight * BCE(ntp_logits, ntp_targets))
    step   = AdamW(lr=1e-4, weight_decay=0)               # no target network
    clip   = clip_grad_norm_(1.0)
    amp    = fp16 autocast + GradScaler on CUDA; fp32 on CPU

There is deliberately no target network, no GAE, no ratio/clip and no KL
regulariser: ``z`` is the undiscounted team rank score already divided by three
(``guandan_rank_points / 3``) and ``gamma = 1``.

``nonfinite_count`` counts non-finite **forward** values (Q, z, loss, NTP
logits).  A positive count fails closed before any optimizer step.  A fp16
GradScaler overflow is *not* counted: skipping that update is the scaler's
contract, and it is reported as ``step_skipped=True`` with the reduced
``scaler_scale`` instead of being silently turned into a NaN update.
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from guandan_marvel.dmc import DMCRuntimeConfig
from guandan_marvel.dmc.replay import DMCReplay, DMCReplayBatch


class DMCNonFiniteError(RuntimeError):
    """Raised when a DMC forward pass or unscaled gradient becomes non-finite."""


def qz_r2(q, z) -> float:
    """Coefficient of determination of Q against the regression target z.

    ``R2 = 1 - sum((q-z)^2) / sum((z-mean(z))^2)``, evaluated in float64.  A
    constant target has no explained variance: the result is ``1.0`` when Q
    matches it exactly and ``0.0`` otherwise (never ``-inf``/NaN).
    """
    if isinstance(q, torch.Tensor):
        q = q.detach().to("cpu", torch.float64).numpy()
    if isinstance(z, torch.Tensor):
        z = z.detach().to("cpu", torch.float64).numpy()
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if q.shape != z.shape:
        raise ValueError(f"q and z must have the same size, got {q.shape} and {z.shape}")
    if q.size == 0:
        raise ValueError("qz_r2 requires at least one sample")
    residual = float(np.sum((q - z) ** 2))
    total = float(np.sum((z - np.mean(z)) ** 2))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return float(1.0 - residual / total)


class DMCLearner:
    """MSE regression learner for :class:`~guandan_marvel.dmc.q_agent.DMCAgent`."""

    def __init__(
        self,
        q_agent: torch.nn.Module,
        replay: DMCReplay,
        *,
        device: torch.device | str | None = None,
        config: DMCRuntimeConfig | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        weight_decay: float | None = None,
        grad_clip_norm: float | None = None,
        ntp_weight: float | None = None,
        amp: bool | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        if not isinstance(q_agent, torch.nn.Module):
            raise TypeError("q_agent must be a torch.nn.Module exposing q_values(...)")
        if not hasattr(q_agent, "q_values"):
            raise TypeError("q_agent must expose q_values(obs_tokens, obs_channels, action_feats, masks)")
        if not isinstance(replay, DMCReplay):
            raise TypeError("replay must be a DMCReplay")
        runtime = config if config is not None else DMCRuntimeConfig()

        if device is None:
            first = next(q_agent.parameters(), None)
            device = first.device if first is not None else torch.device("cpu")
        self.device = torch.device(device)
        # Reject an incompatible replay before moving (and therefore mutating)
        # the caller's module.
        if replay.action_feature_dim != int(getattr(q_agent, "action_feature_dim", 64)):
            raise ValueError(
                "replay.action_feature_dim does not match the q_agent action feature width"
            )
        agent_channels = int(getattr(q_agent, "num_channels", replay.channel_dim))
        if replay.channel_dim != agent_channels:
            raise ValueError(
                f"replay.channel_dim={replay.channel_dim} does not match "
                f"q_agent.num_channels={agent_channels}; the shipped GameSpec is "
                "512-wide, so stage 4 must construct the replay with the same width"
            )
        agent_players = int(getattr(q_agent, "num_players", replay.num_players))
        if replay.num_players != agent_players:
            raise ValueError(
                f"replay.num_players={replay.num_players} does not match "
                f"q_agent.num_players={agent_players}"
            )
        self.q_agent = q_agent.to(self.device)

        self.replay = replay
        self.batch_size = int(runtime.batch_size if batch_size is None else batch_size)
        self.learning_rate = float(runtime.learning_rate if learning_rate is None else learning_rate)
        self.weight_decay = float(runtime.weight_decay if weight_decay is None else weight_decay)
        self.grad_clip_norm = float(runtime.grad_clip_norm if grad_clip_norm is None else grad_clip_norm)
        self.ntp_weight = float(0.0 if ntp_weight is None else ntp_weight)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.weight_decay < 0.0 or not math.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if self.grad_clip_norm <= 0.0 or not math.isfinite(self.grad_clip_norm):
            raise ValueError("grad_clip_norm must be finite and positive")
        if self.ntp_weight < 0.0 or not math.isfinite(self.ntp_weight):
            raise ValueError("ntp_weight must be finite and non-negative")
        if self.ntp_weight > 0.0 and not getattr(self.q_agent, "enable_ntp", False):
            raise ValueError("ntp_weight > 0 requires a q_agent with the NTP head enabled")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.q_agent.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # CUDA runs fp16 autocast and a GradScaler; CPU runs fp32 without a
        # scaler.  A caller may inject a scaler explicitly (the CPU overflow
        # test does) without enabling autocast on CPU, but requesting CUDA
        # autocast on a CPU device is rejected instead of silently downgraded.
        if amp is None:
            self.amp_enabled = self.device.type == "cuda"
        else:
            if not isinstance(amp, bool):
                raise TypeError("amp must be bool or None")
            if amp and self.device.type != "cuda":
                raise ValueError(
                    "fp16 autocast is only available on CUDA; the CPU path is fp32"
                )
            self.amp_enabled = amp
        if grad_scaler is not None:
            if not isinstance(grad_scaler, torch.amp.GradScaler):
                raise TypeError("grad_scaler must be a torch.amp.GradScaler")
            self.scaler = grad_scaler
        else:
            self.scaler = torch.amp.GradScaler(
                self.device.type, enabled=self.amp_enabled,
            )

        self.rng = rng if rng is not None else np.random.default_rng(seed)
        if not isinstance(self.rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self.step_count = 0
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Batch conversion / forward
    # ------------------------------------------------------------------
    def _to_torch(self, batch: DMCReplayBatch) -> dict[str, torch.Tensor]:
        return {
            "obs_tokens": torch.from_numpy(batch.obs_tokens).to(self.device, torch.long),
            "obs_token_lengths": torch.from_numpy(batch.obs_token_lengths).to(self.device, torch.long),
            "obs_channels": torch.from_numpy(batch.obs_channels).to(self.device),
            "action_feats": torch.from_numpy(batch.action_feats).to(self.device).unsqueeze(1),
            "masks": torch.ones((len(batch), 1), dtype=torch.bool, device=self.device),
            "z": torch.from_numpy(batch.z).to(self.device),
        }

    def _forward(
        self, tensors: dict[str, torch.Tensor], *, ntp_targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp_enabled
            else contextlib.nullcontext()
        )
        with autocast:
            out = self.q_agent.q_values(
                tensors["obs_tokens"],
                tensors["obs_channels"],
                tensors["action_feats"],
                tensors["masks"],
                token_lengths=tensors["obs_token_lengths"],
                return_ntp=self.ntp_weight > 0.0,
            )
            if self.ntp_weight > 0.0:
                q_all, ntp_logits = out
                if ntp_targets is None:
                    raise ValueError(
                        "ntp_weight > 0 requires ntp_targets=[B, num_tokens] for train_step"
                    )
                ntp_targets = ntp_targets.to(device=self.device, dtype=torch.float32)
                if ntp_logits is None:
                    raise ValueError("q_agent returned no NTP logits despite the head being enabled")
                if ntp_targets.shape != ntp_logits.shape:
                    raise ValueError(
                        f"ntp_targets shape {tuple(ntp_targets.shape)} != "
                        f"ntp_logits shape {tuple(ntp_logits.shape)}"
                    )
                ntp_loss = F.binary_cross_entropy_with_logits(ntp_logits.float(), ntp_targets)
            else:
                q_all, ntp_logits, ntp_loss = out, None, None
            q = q_all[:, 0]
            z = tensors["z"]
            loss = F.mse_loss(q, z)
            if ntp_loss is not None:
                loss = loss + self.ntp_weight * ntp_loss
        return q, ntp_loss, loss, z

    @staticmethod
    def _nonfinite_count(*values: torch.Tensor | None) -> int:
        count = 0
        for value in values:
            if value is None:
                continue
            count += int((~torch.isfinite(value)).sum().item())
        return count

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------
    def train_step(self, *, ntp_targets: torch.Tensor | None = None) -> dict[str, Any]:
        """Sample one batch, apply one MSE regression update, return metrics."""
        batch = self.replay.sample(self.batch_size, self.rng)
        tensors = self._to_torch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        q, ntp_loss, loss, z = self._forward(tensors, ntp_targets=ntp_targets)
        nonfinite = self._nonfinite_count(q, z, loss, ntp_loss)
        if nonfinite:
            raise DMCNonFiniteError(
                "non-finite DMC forward value(s): "
                f"q={int((~torch.isfinite(q)).sum())}, "
                f"z={int((~torch.isfinite(z)).sum())}, "
                f"loss={int((~torch.isfinite(loss)).sum())}, "
                f"ntp={0 if ntp_loss is None else int((~torch.isfinite(ntp_loss)).sum())}"
            )

        scaler_enabled = bool(self.scaler.is_enabled())
        old_scale = float(self.scaler.get_scale()) if scaler_enabled else 1.0
        self.scaler.scale(loss).backward()
        if scaler_enabled:
            # ``unscale_`` records per-optimizer inf/NaN so ``step`` skips the
            # update even though clipping may then turn inf gradients into NaN.
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.q_agent.parameters() if p.grad is not None],
            self.grad_clip_norm,
        )
        grad_norm_value = float(grad_norm)
        if scaler_enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            step_skipped = float(self.scaler.get_scale()) < old_scale
            scaler_scale = float(self.scaler.get_scale())
        else:
            if not math.isfinite(grad_norm_value):
                # No scaler to absorb an overflow: fail closed rather than let
                # AdamW write NaN/Inf into the parameters.
                raise DMCNonFiniteError(
                    f"non-finite gradient norm {grad_norm_value} on the fp32 path"
                )
            self.optimizer.step()
            step_skipped = False
            scaler_scale = 1.0

        q_detached = q.detach().float()
        z_detached = z.detach().float()
        metrics: dict[str, Any] = {
            "step": self.step_count,
            "loss": float(loss.detach().float().item()),
            "q_mean": float(q_detached.mean().item()),
            "q_std": float(q_detached.std(unbiased=False).item()),
            "z_mean": float(z_detached.mean().item()),
            "z_std": float(z_detached.std(unbiased=False).item()),
            "qz_r2": qz_r2(q_detached, z_detached),
            "grad_norm": grad_norm_value,
            "scaler_scale": scaler_scale,
            "nonfinite_count": nonfinite,
            "step_skipped": bool(step_skipped),
        }
        if ntp_loss is not None:
            metrics["ntp_loss"] = float(ntp_loss.detach().float().item())

        self.step_count += 1
        if self.log_path is not None:
            # Strict JSON cannot represent inf; a scaler-skipped step reports a
            # non-finite grad_norm because that is exactly why it was skipped.
            safe = {
                key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in metrics.items()
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(safe, sort_keys=True) + "\n")
        return metrics


__all__ = ["DMCLearner", "DMCNonFiniteError", "qz_r2"]
