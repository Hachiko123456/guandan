"""Build, validate, or safety-check the fresh M09 Kaggle algorithm-pilot delivery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.kaggle.m09_delivery import (
    build_m09_delivery, run_m09_safety_check, validate_m09_delivery,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--safety-check", action="store_true")
    args = parser.parse_args(argv)
    selected = sum((args.validate is not None, bool(args.safety_check), args.output is not None))
    if selected != 1:
        parser.error("choose exactly one of --output, --validate, or --safety-check")
    if args.validate is not None:
        result = validate_m09_delivery(args.validate)
    elif args.safety_check:
        result = run_m09_safety_check(args.repository, require_clean=not args.allow_dirty)
    else:
        result = build_m09_delivery(
            args.repository, args.output, commit=args.commit, allow_dirty=args.allow_dirty,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
