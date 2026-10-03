"""Execution profile propagation, deadlines, and reproducible evidence helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from guandan.execution_profiles import load_execution_profile

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SEED_BASE = 10_000
EVALUATION_SEED_BASE = 1_000_000


class BudgetExceeded(RuntimeError):
    """Profile budget exhausted: execution is incomplete, never passed."""


class Deadline:
    def __init__(self, max_hours: float, *, clock=None, started_at=None):
        if isinstance(max_hours, bool) or not math.isfinite(max_hours) or max_hours <= 0:
            raise ValueError("max_hours must be finite and positive")
        self.clock = clock or time.monotonic
        self.started = self.clock() if started_at is None else started_at
        self.seconds = float(max_hours) * 3600

    @property
    def elapsed(self):
        return self.clock() - self.started

    def check(self, **diagnostic):
        if self.elapsed >= self.seconds:
            raise BudgetExceeded({"elapsed_seconds": self.elapsed, "budget_seconds": self.seconds, **diagnostic})




def resolved_profile(name: str | None = None) -> dict:
    """No environment or profile fallback: canonical runner input must match disk."""
    env_name = os.environ.get("GUANDAN_PROFILE")
    selected = name or env_name or "local_fast"
    result = load_execution_profile(ROOT, selected)
    raw = os.environ.get("GUANDAN_RESOLVED_PROFILE_JSON")
    if raw is not None:
        if env_name is None:
            raise ValueError("GUANDAN_PROFILE required with GUANDAN_RESOLVED_PROFILE_JSON")
        forwarded = json.loads(raw)
        if env_name != selected or forwarded != result:
            raise ValueError("canonical forwarded profile does not match requested profile/on-disk config")
    elif env_name is not None and env_name != selected:
        raise ValueError("requested profile differs from GUANDAN_PROFILE")
    return result


def runtime_metadata(device: str) -> dict:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    return {
        "git_commit": proc.stdout.strip(), "python": sys.executable,
        "python_version": platform.python_version(), "torch": str(torch.__version__),
        "device": str(device), "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_architecture": "mean-pooled-token-MLP; not MARVEL Transformer",
    }


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_evidence(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
