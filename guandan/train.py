"""Local training CLI. A profile controls real executed counts, not labels."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from .training.runtime import TRAIN_SEED_BASE
from .training.trainer import train


def main(argv=None):
    parser = argparse.ArgumentParser(description='GuanDan A05 profile training')
    parser.add_argument('--algorithm', choices=('ippo','vrpo'), required=True)
    parser.add_argument('--profile', choices=('local_fast','remote_full'), default='local_fast')
    parser.add_argument('--device', default=None)
    parser.add_argument('--updates', type=int, default=None, help='Additional updates; override is not full profile evidence')
    parser.add_argument('--checkpoint-dir', default='runs')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=TRAIN_SEED_BASE)
    args = parser.parse_args(argv)
    result = train(**vars(args))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.status == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
