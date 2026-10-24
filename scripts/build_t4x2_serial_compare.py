"""Build the single-upload serial recipe comparison delivery. No training/upload."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.serial_compare.delivery import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-inputs", type=Path, required=True)
    parser.add_argument("--validation-record", type=Path)
    args = parser.parse_args(argv)
    result = build(
        Path(__file__).resolve().parents[1],
        args.output,
        model_dir=args.model_dir,
        base_inputs=args.base_inputs,
        validation_record=args.validation_record,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
