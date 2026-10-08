# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/utils.py.
from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch


def get_game_attr(game: object, name: str):
    """Read a game capability using snake_case or legacy UPPER_CASE spelling."""
    if hasattr(game, name):
        return getattr(game, name)
    upper = name.upper()
    if hasattr(game, upper):
        return getattr(game, upper)
    raise AttributeError(f"Game contract missing '{name}' (or '{upper}').")


def elementwise_kl(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    a_safe, b_safe = a + eps, b + eps
    return (a_safe * (a_safe.log() - b_safe.log())).sum(dim=-1)


def entropy(a: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    safe = a + eps
    return -(safe * safe.log()).sum(dim=-1)


def set_global_seed(seed: int, *, offset: int = 0) -> None:
    value = int(seed) + int(offset)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def configure_torch(cfg: object, *, log: Callable[[str], None] | None = None) -> None:
    log = print if log is None else log
    deterministic = bool(getattr(cfg, "deterministic", False))
    disable_tf32 = bool(getattr(cfg, "deterministic_disable_tf32", False))
    if deterministic and disable_tf32:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    if not deterministic:
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.device(getattr(cfg, "device", "cpu")).type == "cuda" and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise RuntimeError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG before process start.")
    if not os.environ.get("PYTHONHASHSEED"):
        log("[Determinism] Set PYTHONHASHSEED before process start for stable hashing.")


def unwrap_distributed_module(module: torch.nn.Module | None) -> torch.nn.Module | None:
    current = module
    while isinstance(current, torch.nn.Module):
        original = getattr(current, "_orig_mod", None)
        if isinstance(original, torch.nn.Module):
            current = original
            continue
        inner = getattr(current, "module", None)
        if isinstance(inner, torch.nn.Module):
            current = inner
            continue
        return current
    return current


def compute_commit_step_counts(avail_is_commit: torch.Tensor, action_taken: torch.Tensor, num_steps: torch.Tensor) -> torch.Tensor:
    time = torch.arange(avail_is_commit.shape[1], device=avail_is_commit.device)
    active = time.unsqueeze(0) < num_steps.unsqueeze(1)
    taken = avail_is_commit.gather(2, action_taken.unsqueeze(-1)).squeeze(-1)
    return (taken.bool() & active).sum(dim=1)


__all__ = ["compute_commit_step_counts", "configure_torch", "elementwise_kl", "entropy", "get_game_attr", "set_global_seed", "unwrap_distributed_module"]
