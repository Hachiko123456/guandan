from __future__ import annotations
import argparse, json
from dataclasses import asdict
from .training.trainer import train

def main():
    p=argparse.ArgumentParser(description="GuanDan A05 training smoke runner")
    p.add_argument("--algorithm",choices=("ippo","vrpo"),required=True)
    p.add_argument("--profile",choices=("local_fast","remote_full"),default="local_fast")
    p.add_argument("--device",default=None); p.add_argument("--updates",type=int,default=None); p.add_argument("--checkpoint-dir",default="runs"); p.add_argument("--resume",default=None); p.add_argument("--hidden-dim",type=int,default=32); p.add_argument("--seed",type=int,default=0); p.add_argument("--rollout-steps",type=int,default=None); p.add_argument("--rollout-envs",type=int,default=None)
    args=p.parse_args(); print(json.dumps(asdict(train(**vars(args))),ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
