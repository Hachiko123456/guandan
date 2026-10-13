"""Build or validate the segmented M09 main-IPPO Kaggle delivery."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from guandan_marvel.kaggle.m09_main_delivery import build_main_delivery,validate_main_delivery

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path,default=ROOT);p.add_argument('--output',type=Path);p.add_argument('--r2-delivery',type=Path);p.add_argument('--initial-recovery',type=Path);p.add_argument('--validate',type=Path);a=p.parse_args(argv)
 if a.validate is not None:r=validate_main_delivery(a.validate)
 else:
  if None in (a.output,a.r2_delivery,a.initial_recovery):p.error('--output --r2-delivery --initial-recovery are required')
  r=build_main_delivery(a.repository,a.output,r2_delivery_root=a.r2_delivery,initial_recovery=a.initial_recovery)
 print(json.dumps(r,indent=2,sort_keys=True,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
