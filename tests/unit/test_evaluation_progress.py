import json
import pytest
from guandan.evaluation_progress import SuiteProgress


def test_progress_adds_workers_without_overwriting_and_keeps_phases_separate(tmp_path):
    m=SuiteProgress(tmp_path, dict(expected_formal_games_total=800,expected_warmup_games_total=80))
    for a in ('ippo','vrpo'):
        m.update(dict(algorithm=a,matchup='old',phase='warmup',event='game_complete',completed_games=30,token_steps=9000))
        m.update(dict(algorithm=a,matchup='cross',phase='warmup',event='game_complete',completed_games=10,token_steps=3000))
    p=m.snapshot(status='warmup_complete',decision={'estimated_remaining_seconds':600})
    assert p['warmup_completed_games']==80 and p['formal_completed_games']==0
    assert p['estimated_finish'].endswith('+08:00')
    m.update(dict(algorithm='ippo',matchup='old',phase='formal',event='formal_started',completed_games=0))
    m.update(dict(algorithm='ippo',matchup='old',phase='formal',event='game_complete',completed_games=1))
    p=m.snapshot()
    assert p['formal_completed_games']==1 and p['warmup_completed_games']==80
    assert p['status']=='formal_running'


def test_progress_no_samples_eta_unknown(tmp_path):
    m=SuiteProgress(tmp_path,dict(expected_formal_games_total=800,expected_warmup_games_total=80))
    p=m.snapshot()
    assert p['estimated_remaining_seconds'] is None
    assert p['estimated_finish'] is None
    assert p['formal_completed_games']==0


def test_live_paths_match_notebook_and_worker_telemetry(tmp_path):
    root=tmp_path/'run'
    system=root/'ippo'/'live'/'system.json'
    system.parent.mkdir(parents=True)
    system.write_text(json.dumps({'cpu':{'status':'ok'},'gpus':[{'index':0,'status':'ok','utilization_percent':20}]}))
    m=SuiteProgress(root,dict(expected_formal_games_total=800,expected_warmup_games_total=80))
    p=m.snapshot()
    assert (root/'live'/'progress.json').is_file()
    assert not (root/'progress.json').exists()
    assert p['system']['ippo']['gpus'][0]['utilization_percent']==20
