# SPDX-License-Identifier: MIT
"""Conditional deployment-search gate (phase 6 does *not* implement search).

06A section 4 only permits a search implementation when the value gate has a
PASS artifact produced by the frozen phase-6 design.  The local value pilot is
INCONCLUSIVE (16 < 64 independent deals, one checkpoint, 2 < 64 ranking
points), so no search code is built.  This module is the explicit, fail-closed
entry point: it refuses everything except a *sealed PASS* artifact and reports
``SEARCH_IMPLEMENTED = False``.
"""

from __future__ import annotations

from pathlib import Path

from .design import (
    SEARCH_IMPLEMENTED, SEARCH_STATUS_IMPLEMENTED,
    SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE, SEARCH_STATUS_SKIPPED_VALUE_REJECTED,
    search_status_for_verdict,
)
from .value_gate import ValueGateError, assert_search_allowed, load_value_gate

SEARCH_STATUS = SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE


def require_search_allowed(gate_path: str | Path) -> dict[str, object]:
    """Return the sealed PASS gate or raise; never enables an unimplemented path."""
    if not SEARCH_IMPLEMENTED:
        gate = load_value_gate(gate_path, require_pass=False)
        raise ValueGateError(
            "deployment search is not implemented in phase 6; "
            f"value_gate={gate.get('verdict')!r}, search_status={gate.get('search_status')!r}"
        )
    return assert_search_allowed(gate_path)


__all__ = [
    "SEARCH_IMPLEMENTED", "SEARCH_STATUS", "SEARCH_STATUS_IMPLEMENTED",
    "SEARCH_STATUS_NOT_RUN_VALUE_INCONCLUSIVE", "SEARCH_STATUS_SKIPPED_VALUE_REJECTED",
    "ValueGateError", "load_value_gate", "require_search_allowed",
    "search_status_for_verdict",
]
