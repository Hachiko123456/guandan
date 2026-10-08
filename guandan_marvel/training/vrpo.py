# SPDX-License-Identifier: MIT
"""M03 VRPO trainer entry point."""
from __future__ import annotations

from .local import VRPOTrainer, VRPO_Q_TARGET_CONTRACT

__all__ = ["VRPOTrainer", "VRPO_Q_TARGET_CONTRACT"]
