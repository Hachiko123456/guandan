"""Pre-fix shim: restore ``RMSNorm = torch.nn.RMSNorm`` for the evidence run.

Used only to demonstrate that ``test_macro_fuse_norm_precision.py`` is a real
counterexample (it fails against the pre-repair code) and never loaded by the
normal suite::

    python -m pytest tests/marvel_training/test_macro_fuse_norm_precision.py \
        -p tests.marvel_training._rmsnorm_pre_fix_plugin -q
"""
from __future__ import annotations

import torch

from guandan_marvel.model import base as base_module

ACTIVE = False


def pytest_configure(config) -> None:
    global ACTIVE
    ACTIVE = True
    # ``ResGatedMLP`` resolves the module-global ``RMSNorm`` at call time, so
    # rebinding it before any model is built reproduces the pre-repair behaviour.
    base_module.RMSNorm = torch.nn.RMSNorm
