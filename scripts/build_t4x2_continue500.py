"""Build the source-preserving iter250 -> iter500 delivery. No training/upload."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.continuation500.delivery import build
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--base-inputs',type=Path,required=True);p.add_argument('--validation-record',type=Path)
 a=p.parse_args();print(json.dumps(build(Path(__file__).resolve().parents[1],a.output,a.base_inputs,a.validation_record),ensure_ascii=False,indent=2))
