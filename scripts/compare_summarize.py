# SPDX-License-Identifier: MIT
"""Entry point: merge comparison worker shards into the final report."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
OVERLAY = HERE.parents[1]
for candidate in (str(OVERLAY), os.environ.get("COMPARISON_SOURCE_ROOT", "")):
    if candidate and candidate not in sys.path:
        sys.path.insert(0, candidate)

from guandan_marvel.comparison.summarize import main

if __name__ == "__main__":
    raise SystemExit(main())
