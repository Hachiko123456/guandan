from __future__ import annotations
import ast,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'scripts/build_m09_formal_eval_control.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_formal_eval_control_builds_small_reusable_dataset(tmp_path):
 out=tmp_path/'control';subprocess.run([sys.executable,str(SCRIPT),'--output',str(out)],cwd=ROOT,check=True,capture_output=True,text=True);data=json.loads((out/'dataset/m09_formal_eval_control.json').read_text(encoding='utf-8'));body={k:v for k,v in data.items() if k!='manifest_hash'};canonical=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert data['manifest_hash']==canonical;assert sha(out/'dataset'/data['script']['path'])==data['script']['sha256'];assert sha(out/'dataset'/data['legacy_checkpoint']['path'])==data['legacy_checkpoint']['sha256'];assert (out/'dataset').stat().st_size<10_000
 nb=json.loads((out/'guandan_marvel_m09_iter0500_formal_eval.ipynb').read_text(encoding='utf-8'));code='\n'.join(''.join(c.get('source',[])) for c in nb['cells'] if c['cell_type']=='code');ast.parse(code);assert 'segment0500 Notebook Output' in ''.join(nb['cells'][0]['source']);assert 'guandan-marvel-formal-eval-iter0500.zip' in code
def test_formal_eval_script_binds_iter500_and_initial_champion():
 text=(ROOT/'scripts/m09_kaggle_formal_eval.py').read_text(encoding='utf-8');assert "expected['candidate_iteration']" in text;assert 'PromotionPolicy(mode=\'formal\'' in text;assert "counts={'training':0,'selection':32,'final':0}" in text;assert 'final_holdout_used' in text
