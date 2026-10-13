from __future__ import annotations
import json
from pathlib import Path
from guandan_marvel.kaggle.m09_main_delivery import MAIN_DELIVERY_ID,SEGMENTS,_profile,build_main_delivery,validate_main_delivery
ROOT=Path(__file__).resolve().parents[2]
R2=Path(r'D:\project\kaggle_train\guandan-marvel-m09-algorithm-pilot-20261011-r2')
RECOVERY=ROOT/'runs/M09_full_training/remote/r2_ippo_pilot_20261013T051332Z/downloaded/guandan-marvel-recovery.zip'

def test_main_plan_freezes_ippo_exact_resume_segments():
 plan=json.loads((ROOT/'configs/marvel_m09_main_training_plan.json').read_text(encoding='utf-8'))
 assert plan['selected_algorithm']=='ippo';assert plan['initial_resume']['classification']=='exact_resume';assert plan['training_distribution']['level_rank']=='Rank.TWO';assert plan['training_distribution']['previous_result'] is None
 assert [x['target_iteration'] for x in plan['segments']]==[500,1000,1500,2000]
 assert [x['target_completed_games'] for x in plan['segments']]==[2000,4000,6000,8000]
 for segment in plan['segments']:
  profile=_profile(plan,segment);assert profile['mode']=='resume';assert profile['expected_algorithm']=='ippo';assert profile['resume']['evaluation_interval']==0;assert profile['completed_game_budget']['target_completed_games']==segment['target_completed_games'];assert profile['m09']['mandatory_evaluation_after_segment'] is True

def test_main_delivery_builds_segmented_exact_resume_package(tmp_path):
 out=tmp_path/MAIN_DELIVERY_ID
 build_main_delivery(ROOT,out,r2_delivery_root=R2,initial_recovery=RECOVERY)
 result=validate_main_delivery(out);assert result['valid'] is True;assert [r['segment'] for r in result['segments']]==list(SEGMENTS);assert result['initial_recovery_sha256']=='8b05b05f0d2f3715f2fb9701cbc4450a43d156a7cf2387eb916c2e9958c8b067'
 assert (out/'datasets/segment0500/guandan-marvel-recovery.zip').is_file();assert not (out/'datasets/segment1000/guandan-marvel-recovery.zip').exists()
