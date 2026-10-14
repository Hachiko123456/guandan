from __future__ import annotations
import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'scripts/build_m09_formal_eval_control.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_formal_eval_control_builds_small_reusable_dataset(tmp_path):
 out=tmp_path/'control';subprocess.run([sys.executable,str(SCRIPT),'--output',str(out)],cwd=ROOT,check=True,capture_output=True,text=True);data=json.loads((out/'dataset/m09_formal_eval_control.json').read_text(encoding='utf-8'));body={k:v for k,v in data.items() if k!='manifest_hash'};canonical=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();assert data['manifest_hash']==canonical;assert sha(out/'dataset'/data['script']['path'])==data['script']['sha256'];assert sha(out/'dataset'/data['legacy_checkpoint']['path'])==data['legacy_checkpoint']['sha256'];assert sha(out/'dataset'/data['acceptance_profiles']['path'])==data['acceptance_profiles']['sha256'];assert data['monitoring']=={'heartbeat_seconds':20,'progress_source':'selection_run/games.jsonl','gpu_source':'nvidia-smi'};assert (out/'dataset').stat().st_size<10_000
 nb=json.loads((out/'guandan_marvel_m09_iter0500_formal_eval.ipynb').read_text(encoding='utf-8'));code='\n'.join(''.join(c.get('source',[])) for c in nb['cells'] if c['cell_type']=='code');ast.parse(code);assert 'segment0500 Notebook Output' in ''.join(nb['cells'][0]['source']);assert 'guandan-marvel-formal-eval-iter0500.zip' in code
def test_formal_eval_script_binds_iter500_and_initial_champion():
 text=(ROOT/'scripts/m09_kaggle_formal_eval.py').read_text(encoding='utf-8');assert 'candidate_iteration' in text;assert 'PromotionPolicy(' in text and 'mode="formal"' in text;assert '"selection": 32' in text;assert 'final_holdout_used' in text;assert 'evaluation_heartbeat' in text;assert 'nvidia-smi' in text;assert 'games_completed' in text


def test_formal_eval_materializes_source_at_checkpoint_runtime_path(tmp_path):
 spec=importlib.util.spec_from_file_location('m09_kaggle_formal_eval',ROOT/'scripts/m09_kaggle_formal_eval.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 source=tmp_path/'input/notebooks/owner/run/guandan-marvel-source';(source/'nanopokerlib').mkdir(parents=True);binary=source/'nanopokerlib/_guandan_cpp.cpython-313-x86_64-linux-gnu.so';binary.write_bytes(b'bound-runtime-binary')
 expected={'source_commit':'commit','source_manifest_hash':'manifest','runtime_binary_sha256':sha(binary)};(source/'source_manifest.json').write_text(json.dumps({'git_commit':'commit','manifest_hash':'manifest'}),encoding='utf-8')
 profiles=tmp_path/'acceptance_profiles.json';profiles.write_text('{\"version\":\"profiles-0.2\",\"profiles\":{}}',encoding='utf-8')
 working=tmp_path/'working';working.mkdir();target=module.materialize_source(source,working,expected,profiles)
 assert target==(working/'guandan-marvel-source').resolve();assert not target.is_symlink();assert sha(target/'nanopokerlib'/binary.name)==expected['runtime_binary_sha256'];assert (target/'configs/acceptance_profiles.json').read_bytes()==profiles.read_bytes();assert source.resolve()!=target.resolve()



def test_formal_eval_progress_helpers_report_finite_lane_progress(tmp_path):
 spec=importlib.util.spec_from_file_location('m09_kaggle_formal_eval_progress',ROOT/'scripts/m09_kaggle_formal_eval.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 records=tmp_path/'games.jsonl';records.write_text('{"game":1}\n\n{"game":2}\n',encoding='utf-8')
 assert module.count_jsonl(records)==2
 lanes=('champion','rule','random','legacy','opponent_pool')
 assert module.evaluation_position(0,total=640,lanes=lanes,games_per_lane=128)=={'lane':'champion','lane_index':0,'lane_completed':0}
 assert module.evaluation_position(129,total=640,lanes=lanes,games_per_lane=128)=={'lane':'rule','lane_index':1,'lane_completed':1}
 assert module.evaluation_position(640,total=640,lanes=lanes,games_per_lane=128)=={'lane':'complete','lane_index':5,'lane_completed':128}
