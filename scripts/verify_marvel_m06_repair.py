"""Verify M06 repair run integrity without consuming final holdout."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def payload_hash(value):
 body=dict(value);body.pop('hash',None)
 return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();run=args.run_root.resolve()
 summary=load(run/'repair_exercise_summary.json');result=summary['selection_result'];games=[json.loads(line) for line in Path(result['games_path']).read_text(encoding='utf-8').splitlines()]
 assert summary['training_state_unchanged_during_async_wait'] is True
 assert summary['async_event']['status']=='completed' and result['status']=='complete'
 assert result['promotion_decision']['status']=='inconclusive' and result['promotion_decision']['promote'] is False
 assert result['data_isolation']['formal_selection_eligible'] is True
 assert len(games)==result['expected_games']==20
 assert all(row['token_steps']==row['engine_finish_token_step'] and row['committed_steps']==row['engine_finish_committed_step'] and row['committed_steps']<row['token_steps'] for row in games)
 assert summary['final_holdout_state']=='unused'
 for name in ('deals_manifest.json','champion_manifest.json','opponent_pool_manifest.json','final_holdout_audit.json'):
  value=load(run/name);assert value['hash']==payload_hash(value),name
 for line in (run/'async_hook/evaluation_hook_events.jsonl').read_text(encoding='utf-8').splitlines():
  event=json.loads(line);assert event['hash']==payload_hash(event)
 old=load(ROOT/'docs/marvel_migration/evidence/M06_evaluation_champion/artifact_manifest.json')
 old_valid=True
 for row in old['run_artifacts']:
  path=ROOT/row['path'];old_valid &= path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
 assert old_valid
 protected=['guandan/cards.py','guandan/combos.py','guandan/round.py','guandan/state.py','docs/rules.md']
 diff=subprocess.run(['git','diff','--quiet','10cb3452f69cd23a3bed848498e4353dcfbe3b9c','--',*protected],cwd=ROOT).returncode
 assert diff==0
 reference_status=subprocess.run(['git','-C',r'D:\project\doudizhu','status','--short'],capture_output=True,text=True,check=True).stdout.strip();assert not reference_status
 status=yaml.safe_load((ROOT/'docs/marvel_migration/STATUS.yaml').read_text(encoding='utf-8'))
 m06_status=status['stages']['M06_evaluation_champion']['status'];m07_status=status['stages']['M07_kaggle_t4x2']['status']
 assert (m06_status,m07_status) in {('in_progress','blocked'),('accepted','ready')}
 report={'schema':'guandan-marvel-m06-repair-run-verification-v1','passes':True,'run_root':str(run),'selection_status':result['status'],'promotion_status':result['promotion_decision']['status'],'training_state_unchanged':True,'formal_data_isolation':True,'games':20,'token_steps':sum(x['token_steps'] for x in games),'committed_steps':sum(x['committed_steps'] for x in games),'engine_counts_match':True,'illegal_count':sum(x['illegal_count'] for x in games),'truncated_count':sum(x['truncated_count'] for x in games),'final_holdout_state':'unused','old_m06_artifacts_unchanged':old_valid,'protected_files_unchanged':True,'reference_repo_clean':True,'m06_status':m06_status,'m07_status':m07_status}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
