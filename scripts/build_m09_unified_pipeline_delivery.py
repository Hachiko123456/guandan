"""Build and validate the long-lived M09 Unified Dataset/Notebook delivery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.kaggle.m09_unified_delivery import build_unified_delivery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--initial-recovery", type=Path, required=True)
    parser.add_argument("--champion-seed", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_unified_delivery(
        args.repository,
        args.output,
        source_dataset_root=args.source_dataset,
        initial_recovery=args.initial_recovery,
        champion_seed=args.champion_seed,
        legacy_checkpoint=args.legacy_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
