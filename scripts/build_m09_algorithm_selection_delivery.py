"""Build or validate the M09 IPPO-vs-VRPO Kaggle selection delivery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.kaggle.m09_selection_delivery import (
    build_selection_delivery, validate_selection_delivery,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ippo-output", type=Path)
    parser.add_argument("--vrpo-output", type=Path)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        result = validate_selection_delivery(args.validate)
    else:
        if args.output is None or args.ippo_output is None or args.vrpo_output is None:
            parser.error("--output, --ippo-output and --vrpo-output are required")
        result = build_selection_delivery(
            args.repository, args.output, ippo_output_root=args.ippo_output,
            vrpo_output_root=args.vrpo_output, commit=args.commit, allow_dirty=args.allow_dirty,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
