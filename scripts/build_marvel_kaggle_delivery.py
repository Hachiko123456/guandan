"""Build or validate the M07 Kaggle T4x2 delivery package.  Never uploads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.kaggle.package import build_delivery, validate_delivery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        result = validate_delivery(args.validate)
    else:
        if args.output is None:
            parser.error("--output is required when not using --validate")
        result = build_delivery(
            args.repository, args.output, commit=args.commit, allow_dirty=args.allow_dirty,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
