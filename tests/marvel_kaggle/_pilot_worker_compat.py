"""Explicit two-rank CPU/Gloo compatibility harness; never target CUDA certification."""
import os,sys,argparse
from pathlib import Path
os.environ.setdefault("USE_LIBUV","0")
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from guandan_marvel.kaggle.worker import main
p=argparse.ArgumentParser();p.add_argument("--rank");p.add_argument("--profile");p.add_argument("--output");p.add_argument("--recovery",default=None)
a=p.parse_args()
args=["--profile",a.profile,"--run-root",a.output,"--package-id","pilot-gloo-compat",
      "--source-commit","a"*40,"--source-archive-sha256","b"*64,"--source-content-manifest-hash","c"*64,"--local-cpu-test"]
if a.recovery:args.extend(["--recovery-root",a.recovery])
raise SystemExit(main(args))
