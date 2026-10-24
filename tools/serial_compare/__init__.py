# SPDX-License-Identifier: MIT
"""Single-upload serial recipe comparison (control vs candidate).

Modules are import-light so the build script and the offline verifier can use
the contracts without loading a model, a native extension or CUDA.
"""
from __future__ import annotations

__all__ = [
    "budget", "common", "diagnostics", "evaluation", "markers", "orchestrator",
    "reporting", "training",
]
