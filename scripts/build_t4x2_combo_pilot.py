"""Build the new three-parameter T4x2 delivery. Never uploads or starts training."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from guandan_marvel.kaggle.combo_delivery import build_combo_delivery

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warm-start',type=Path,required=True)
    p.add_argument('--validation-record',type=Path)
    a=p.parse_args()
    print(json.dumps(build_combo_delivery(Path(__file__).resolve().parents[1],a.output,a.warm_start,
         validation_record=a.validation_record),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
