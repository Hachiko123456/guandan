# SPDX-License-Identifier: MIT
"""Phase-6 conditional features (default off).

This package deliberately keeps its top-level import light: only the frozen
design and the dependency-free configuration type are imported eagerly.  The
pool, value-gate and training-gate modules import the model/evaluation stack
and must be imported explicitly (``guandan_marvel.phase6.opponent_pool`` etc.)
so that ``guandan_marvel.config`` can depend on the configuration type without
creating an import cycle through the trainer.
"""

from .config_types import OpponentPoolConfig
from .design import (
    CURRENT_POLICY_OWNER_ID, FORMAL_CONFIRMATION_TOKEN, FORMAL_GATE_TARGETS,
    PHASE6_DEFAULTS, PHASE6_DESIGN_SCHEMA, SEARCH_IMPLEMENTED,
    SEARCH_STATUS_IMPLEMENTED, SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE,
    SEARCH_STATUS_SKIPPED_VALUE_REJECTED,
    SEARCH_STATUS_VALUE_QUALIFIED_SEARCH_NOT_IMPLEMENTED, phase6_design_hash,
    phase6_design_identity, search_status_for_verdict,
)

__all__ = [
    "CURRENT_POLICY_OWNER_ID", "FORMAL_CONFIRMATION_TOKEN", "FORMAL_GATE_TARGETS",
    "OpponentPoolConfig", "PHASE6_DEFAULTS", "PHASE6_DESIGN_SCHEMA",
    "SEARCH_IMPLEMENTED", "SEARCH_STATUS_IMPLEMENTED",
    "SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE", "SEARCH_STATUS_SKIPPED_VALUE_REJECTED",
    "SEARCH_STATUS_VALUE_QUALIFIED_SEARCH_NOT_IMPLEMENTED",
    "phase6_design_hash", "phase6_design_identity", "search_status_for_verdict",
]
