"""Build or validate the M09 bootcamp week-1 Kaggle delivery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guandan_marvel.kaggle.bootcamp_delivery import build_bootcamp_delivery, validate_bootcamp_delivery


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--notebook-template", type=Path)
    parser.add_argument("--locked-config", type=Path)
    parser.add_argument("--expected-warm-start-sha256")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        result = validate_bootcamp_delivery(args.validate)
    else:
        if None in (args.output, args.warm_start, args.notebook_template):
            parser.error("--output --warm-start --notebook-template are required")
        locked = args.locked_config or (ROOT / "configs" / "marvel_m08_locked_config.json")
        result = build_bootcamp_delivery(
            args.repository,
            args.output,
            warm_start_checkpoint=args.warm_start,
            notebook_template=args.notebook_template,
            locked_config=locked,
            expected_warm_start_sha256=args.expected_warm_start_sha256,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
