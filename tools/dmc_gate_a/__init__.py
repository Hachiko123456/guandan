# SPDX-License-Identifier: MIT
"""Gate A driver package (stage 6).

Shipped inside the Kaggle payload as ``dmc_gate_a/`` next to the source
overlay; imported by the notebook's single run-all cell as
``from dmc_gate_a.runner import run_notebook``.
"""

from .lanes import LANE_DRIVERS, LANE_ROTATIONS, play_lane_games
from .m09 import M09LaneUnavailable, precheck
from .runner import (
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    RESULTS_MANIFEST_SCHEMA,
    RESULTS_ZIP_NAME,
    RESUME_FILENAME,
    canonical,
    evaluate_gates,
    run_notebook,
    stage7_projection,
)

__all__ = [
    "LANE_DRIVERS",
    "LANE_ROTATIONS",
    "M09LaneUnavailable",
    "PLAN_SCHEMA",
    "RESULT_SCHEMA",
    "RESULTS_MANIFEST_SCHEMA",
    "RESULTS_ZIP_NAME",
    "RESUME_FILENAME",
    "canonical",
    "evaluate_gates",
    "play_lane_games",
    "precheck",
    "run_notebook",
    "stage7_projection",
]
