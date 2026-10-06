"""Single-writer suite progress aggregation; heartbeats never fabricate game counts."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import json
import time
from pathlib import Path
from .observability import atomic_json


def append_event(path, data):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
        stream.flush()


class SuiteProgress:
    def __init__(self, root, plan):
        self.root = Path(root)/'live'
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan = plan
        self.started = time.monotonic()
        self.rows = {}
        self.last = {}
        self.status = 'starting'
        self.decision = None
        self.formal_started = None
        self.event('suite_started', expected_warmup_games=plan['expected_warmup_games_total'],
                   expected_formal_games=plan['expected_formal_games_total'])

    def event(self, event, **kwargs):
        value = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'event': event, **kwargs}
        append_event(self.root/'events.jsonl', value)
        print('[suite] '+json.dumps(value,ensure_ascii=False),flush=True)

    def update(self, payload):
        algorithm = payload['algorithm']
        phase = payload.get('phase', 'warmup')
        key = (algorithm,payload.get('matchup',''),phase)
        previous = self.rows.get(key,{})
        row = {**previous,**payload}
        if payload.get('event') in {'game_complete','game_advanced','formal_started','warmup_started','opponent_started'}:
            row['status']='running'
        row['completed_games'] = max(previous.get('completed_games',0),payload.get('completed_games',0))
        row['token_steps'] = max(previous.get('token_steps',0),payload.get('token_steps',0))
        row['last_progress_monotonic'] = time.monotonic()
        self.rows[key] = row
        self.last[algorithm] = row
        if payload.get('event') == 'formal_started' and self.formal_started is None:
            self.status='formal_running'
            self.formal_started = time.monotonic()
        append_event(self.root/'metrics.jsonl', payload)
        if payload.get('event') not in {'game_complete','game_advanced','evaluation_progress'}:
            append_event(self.root/'events.jsonl',payload)

    def snapshot(self, *, status=None, decision=None, finals=None):
        if status is not None: self.status=status
        if decision is not None: self.decision=decision
        now=time.monotonic()
        warmup=sum(v.get('completed_games',0) for k,v in self.rows.items() if k[2]=='warmup')
        formal=sum(v.get('completed_games',0) for k,v in self.rows.items() if k[2]=='formal')
        target=self.plan['expected_formal_games_total']
        elapsed=now-self.started
        remaining = self.decision.get('estimated_remaining_seconds') if self.decision else None
        estimate_kind = 'warmup_extrapolation' if remaining is not None else 'unknown'
        if self.formal_started is not None:
            rates=[]
            for alg in ('ippo','vrpo'):
                done=sum(v.get('completed_games',0) for k,v in self.rows.items() if k[0]==alg and k[2]=='formal')
                if done:
                    rates.append(max(0,target/2-done)/ (done/max(now-self.formal_started,1e-9)))
                elif self.decision:
                    rates.append(self.decision['estimated_remaining_seconds'])
            remaining=max(rates) if rates else None
            estimate_kind='running_average' if rates else 'unknown'
        if self.status=='complete': remaining=0
        finish = (datetime.now(timezone(timedelta(hours=8)))+timedelta(seconds=remaining)).isoformat() if remaining is not None else None
        workers={alg:{**{k:v for k,v in value.items() if k!='last_progress_monotonic'},
                      'progress_age_seconds':now-value['last_progress_monotonic']} for alg,value in self.last.items()}
        systems={}
        for alg in ('ippo','vrpo'):
            try:
                record=json.loads((self.root.parent/alg/'live/system.json').read_text(encoding='utf-8'))
                systems[alg]={'cpu':record.get('cpu'),'gpus':record.get('gpus'),'torch_memory':record.get('torch_memory')}
            except (OSError,ValueError):
                systems[alg]={'status':'unavailable'}
        value=dict(status=self.status, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=elapsed,warmup_completed_games=warmup,
                   warmup_target_games=self.plan['expected_warmup_games_total'],
                   formal_completed_games=formal,formal_target_games=target,
                   estimated_remaining_seconds=remaining,estimated_finish=finish,
                   estimate_kind=estimate_kind, workers=workers, system=systems,
                   formal_started=self.formal_started is not None,
                   note='ETA is measured/extrapolated, not a promise; warmup_complete means formal not run')
        atomic_json(self.root/'progress.json',value)
        return value


def progress_line(value):
    """Compact console view; complete structured fields remain in progress.json."""
    eta=value.get('estimated_remaining_seconds')
    head=(f"[suite-progress] status={value.get('status')} "
          f"warmup={value.get('warmup_completed_games',0)}/{value.get('warmup_target_games',0)} "
          f"formal={value.get('formal_completed_games',0)}/{value.get('formal_target_games',0)} "
          f"elapsed={value.get('elapsed_seconds',0)/60:.1f}m "
          f"ETA={'unknown' if eta is None else str(round(eta/60,1))+'m'} "
          f"finish={value.get('estimated_finish') or 'unknown'}")
    parts=[]
    for alg,row in value.get('workers',{}).items():
        gpu=value.get('system',{}).get(alg,{}).get('gpus',[])
        values=[f"gpu{g.get('index','?')}={g.get('utilization_percent','unavailable')}%" for g in gpu if g.get('status')=='ok']
        parts.append(f"{alg}:{row.get('phase')}/{row.get('opponent')} "
                     f"done={row.get('completed_games',0)}/{row.get('target_games',0)} "
                     f"tokens={row.get('token_steps',0)} "
                     f"age={row.get('progress_age_seconds',0):.0f}s "
                     +(' '.join(values) or 'gpu=unavailable'))
    return head+' | '+' | '.join(parts)
