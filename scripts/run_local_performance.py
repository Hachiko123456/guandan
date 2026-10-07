"""Local CPU vs GPU gate; opt-in, same seeded workload, no long training."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan.performance_gate import main

if __name__ == "__main__":
    if "--mode" in sys.argv or any(x.startswith("--mode=") for x in sys.argv[1:]):
        raise SystemExit("This wrapper is local only; use the pinned Notebook for remote validation")
    raise SystemExit(main(["--mode", "local", *sys.argv[1:]]))
