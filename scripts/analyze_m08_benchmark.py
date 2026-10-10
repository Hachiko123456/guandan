"""Compare downloaded M08 single/dual benchmark_summary.json files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from guandan_marvel.kaggle.benchmark import compare_benchmark_summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    single = json.loads(args.single.read_text(encoding="utf-8"))
    dual = json.loads(args.dual.read_text(encoding="utf-8"))
    result = compare_benchmark_summaries(single, dual)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

