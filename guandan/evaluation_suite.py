"""Two-GPU inference-only ability evaluation with warmup and live progress."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
import json
import multiprocessing as mp
import signal
import threading
from queue import Empty
import os
from pathlib import Path
import random
import re
import time
import traceback
import uuid
import zipfile
from typing import Any

from .evaluation_stats import wilson_interval
from .execution_profiles import load_execution_profile
from .evaluation_inputs import extract_progress, checkpoint_for, sha256
from .evaluation_telemetry import EvaluationTelemetry
from .evaluation_progress import SuiteProgress, progress_line
from .observability import atomic_json
from .deployment.provenance import source_provenance

_WORKER_EVENT_QUEUE = None

_UPDATE_RE = re.compile(r"_update_(\d+)_")


@dataclass(frozen=True)
class EvalWorkerSpec:
    algorithm: str
    candidate: str
    old_snapshot: str
    cross_snapshot: str
    device: str
    profile: str
    batch_size: int
    output_root: str
    seed: int
    warmup_games_per_opponent: int
    target_games_per_opponent: int
    max_estimated_minutes: float | None


def _atomic_json(path, payload):
    atomic_json(path, payload)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _device_index(device: str) -> int | None:
    return int(device.split(":", 1)[1]) if device.startswith("cuda:") and device.split(":", 1)[1].isdigit() else None


def _child_device(device: str) -> str:
    index = _device_index(device)
    if index is None:
        return device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    return "cuda"


def _extract_bundle(bundle, destination):
    return extract_progress(bundle, destination)


def _checkpoint_for(root, algorithm, update):
    return checkpoint_for(root, algorithm, update)


def _cluster_summary(records: list[dict[str, Any]], *, opponent: str, seed: int) -> dict[str, Any]:
    games = [row for row in records if row["opponent"] == opponent]
    wins = [int(row["winner_team"] == row["candidate_team"]) for row in games]
    groups: dict[tuple[int, str], list[int]] = {}
    for row, win in zip(games, wins):
        groups.setdefault((row["deal_group"], row["base_deal_sha256"]), []).append(win)
    values = [sum(rows) / len(rows) for rows in groups.values()]
    rng = random.Random(seed)
    bootstrap = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(2000)) if values else []
    lower = bootstrap[int(.025 * (len(bootstrap) - 1))] if bootstrap else None
    upper = bootstrap[int(.975 * (len(bootstrap) - 1))] if bootstrap else None
    return {
        "opponent": opponent, "games": len(games), "independent_deal_groups": len(groups),
        "candidate_wins": sum(wins), "candidate_win_rate": sum(wins) / len(wins) if wins else None,
        "game_level_wilson95": wilson_interval(sum(wins), len(wins)) if wins else None,
        "deal_cluster_bootstrap95": (lower, upper),
        "mean_token_steps": sum(row["token_steps"] for row in games) / len(games) if games else None,
        "mean_committed_steps": sum(row["committed_steps"] for row in games) / len(games) if games else None,
    }


def _progress_callback(*, root: Path, algorithm: str, label: str, device: str):
    started = time.monotonic()
    output = root / algorithm / label / "live"
    root_live = root / "live"
    previous = {}

    def callback(event: dict[str, Any]) -> None:
        payload = {**previous, **event}
        if event.get('event') in {'game_complete','game_advanced','formal_started','warmup_started','opponent_started'}:
            payload['status']='running'
        if 'opponent' in event and event['opponent'] != previous.get('opponent'):
            payload['completed_opponent_games'] = event.get('completed_per_opponent',{}).get(event['opponent'],0)
        previous.update(payload)
        payload.update({'algorithm': algorithm, 'matchup': label, 'device': device,
                        'observed_at': datetime.now(timezone.utc).isoformat()})
        payload.setdefault('elapsed_seconds', time.monotonic() - started)
        _atomic_json(output / 'progress.json', payload)
        _atomic_json(root_live / f'{algorithm}_{label}_progress.json', payload)
        _append_jsonl(output / 'metrics.jsonl', payload)
        _append_jsonl(root_live / f'{algorithm}_{label}_metrics.jsonl', payload)
        if _WORKER_EVENT_QUEUE is not None:
            _WORKER_EVENT_QUEUE.put({'kind':'progress', 'payload':payload})
        completed, target = payload.get('completed_games', 0), payload.get('target_games', 0)
        rate, eta = payload.get('games_per_second'), payload.get('estimated_remaining_seconds')
        print(f"[eval] algorithm={algorithm} matchup={label} phase={payload.get('phase')} "
              f"opponent={payload.get('opponent')} completed={completed}/{target} "
              f"opponent_completed={payload.get('completed_opponent_games',0)}/{payload.get('target_opponent_games','?')} elapsed={payload.get('elapsed_seconds',0):.1f}s "
              f"rate={'unknown' if rate is None else f'{rate:.3f} games/s'} "
              f"eta={'unknown' if eta is None else f'{eta / 60:.1f}m'} token_steps={payload.get('token_steps', 0)} "
              f"inflight_games={payload.get('inflight_games', 0)} finish={payload.get('estimated_finish')} event={payload.get('event')}", flush=True)
    return callback


def _run_one(label: str, candidate: Path, snapshot: Path, spec: EvalWorkerSpec,
             device: str, out: Path, *, phase: str, opponents: list[str]) -> dict[str, Any]:
    from .evaluation import evaluate_profile, EvaluationIncomplete
    evidence = out / label / f'{phase}_evaluation.json'
    callback = _progress_callback(root=out.parent, algorithm=spec.algorithm, label=label, device=spec.device)
    started = time.monotonic()
    try:
        result = evaluate_profile(
            profile=spec.profile, seed=spec.seed, candidate_checkpoint=candidate,
            snapshot_checkpoint=snapshot, evidence_path=evidence, device=device,
            batch_size=spec.batch_size, progress_callback=callback,
            warmup_games_per_opponent=spec.warmup_games_per_opponent if phase == 'warmup' else 0,
            target_games_per_opponent=spec.target_games_per_opponent,
            max_estimated_minutes=None,
            max_hours=(spec.max_estimated_minutes or 240)/60,
            auto_continue_after_warmup=(phase == 'formal'), opponents=opponents,
        )
        records = [asdict(row) for row in result.records]
        summary = {
            'status': result.status, 'phase': phase, 'label': label, 'algorithm': spec.algorithm,
            'device': spec.device, 'candidate': str(candidate), 'snapshot': str(snapshot),
            'runtime_seconds': result.runtime_seconds, 'wall_seconds': time.monotonic() - started,
            'completed_games_per_opponent': result.completed_games_per_opponent,
            'warmup_games_per_opponent': result.warmup_games_per_opponent,
            'warmup_completed_games_per_opponent': result.warmup_completed_games_per_opponent,
            'target_games_per_opponent_actual': result.target_games_per_opponent,
            'summary': result.summary, 'metadata': result.metadata,
        }
        if phase == 'formal':
            summary['clustered'] = {opponent: _cluster_summary(records, opponent=opponent, seed=spec.seed + index)
                                    for index, opponent in enumerate(opponents)}
    except BaseException as exc:
        summary = {'status': 'incomplete' if isinstance(exc, EvaluationIncomplete) else 'failed', 'phase': phase, 'label': label,
                   'algorithm': spec.algorithm, 'device': spec.device,
                   'runtime_seconds': time.monotonic() - started,
                   'error_type': type(exc).__name__, 'error': str(exc),
                   'traceback': traceback.format_exc()}
    _atomic_json(out / label / f'{phase}_summary.json', summary)
    _atomic_json(out / label / 'ability_summary.json', summary)
    return summary


def _reports_spec(spec: EvalWorkerSpec):
    return (
        (f'{spec.algorithm}_update102_vs_update100', Path(spec.old_snapshot), ['random', 'rule', 'snapshot']),
        (f'{spec.algorithm}_update102_vs_other_update102', Path(spec.cross_snapshot), ['snapshot']),
    )


def _worker(spec: EvalWorkerSpec, result_queue, control_queue) -> None:
    started = time.monotonic(); out = Path(spec.output_root) / spec.algorithm
    global _WORKER_EVENT_QUEUE
    _WORKER_EVENT_QUEUE = result_queue
    warmup_reports = []
    telemetry = None
    try:
        device = _child_device(spec.device); candidate = Path(spec.candidate)
        telemetry = EvaluationTelemetry(out/'live', metadata={'algorithm':spec.algorithm,'device':spec.device}).start()
        for label, snapshot, opponents in _reports_spec(spec):
            report = _run_one(label, candidate, snapshot, spec, device, out,
                              phase='warmup', opponents=opponents)
            warmup_reports.append(report)
            if report.get('status') == 'incomplete':
                raise TimeoutError('evaluation deadline reached during warmup')
            if report.get('status') != 'warmup_complete':
                raise RuntimeError(f"warmup report failed: {label}: {report.get('status')}")
        result_queue.put({'kind': 'warmup_complete', 'algorithm': spec.algorithm,
                          'device': spec.device, 'reports': warmup_reports,
                          'elapsed_seconds': time.monotonic() - started})
        decision = control_queue.get(timeout=max(30.0, (spec.max_estimated_minutes or 60) * 60))
        if not decision.get('continue_formal', False):
            payload = {'kind': 'final', 'status': decision.get('status', 'warmup_complete'),
                       'algorithm': spec.algorithm, 'device': spec.device,
                       'reports': warmup_reports, 'decision': decision,
                       'elapsed_seconds': time.monotonic() - started}
        else:
            formal_reports = []
            for label, snapshot, opponents in _reports_spec(spec):
                report = _run_one(label, candidate, snapshot, spec, device, out,
                                  phase='formal', opponents=opponents)
                formal_reports.append(report)
                if report.get('status') == 'incomplete':
                    raise TimeoutError('evaluation deadline reached during formal phase')
                if report.get('status') != 'complete':
                    raise RuntimeError(f"formal report failed: {label}: {report.get('status')}")
            status = 'complete' if all(row.get('status') == 'complete' for row in formal_reports) else 'incomplete'
            payload = {'kind': 'final', 'status': status, 'algorithm': spec.algorithm,
                       'device': spec.device, 'reports': formal_reports,
                       'warmup_reports': warmup_reports, 'decision': decision,
                       'elapsed_seconds': time.monotonic() - started}
    except BaseException as exc:
        payload = {'kind': 'final', 'status': 'stopped_by_budget' if isinstance(exc, TimeoutError) else 'failed', 'algorithm': spec.algorithm,
                   'device': spec.device, 'reports': warmup_reports,
                   'error_type': type(exc).__name__, 'error': str(exc),
                   'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started}
    if telemetry is not None:
        payload['telemetry'] = telemetry.close()
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / 'result.json', payload)
    result_queue.put(payload)


def _target_for_profile(profile: str, target_games_per_opponent: int) -> int:
    resolved = load_execution_profile(Path(__file__).resolve().parents[1], profile)
    rotations = 4
    if type(target_games_per_opponent) is not int or target_games_per_opponent <= 0 or target_games_per_opponent % rotations:
        raise ValueError('target_games_per_opponent must be a positive multiple of seat rotations')
    return target_games_per_opponent


def build_plan(*, profile: str = 'remote_full', progress_bundle: str | Path = '',
               output_root: str | Path = 'runs/evaluation-suite', batch_size: int = 16,
               device_ippo: str = 'cuda:0', device_vrpo: str = 'cuda:1',
               warmup_games_per_opponent: int = 10, target_games_per_opponent: int = 100,
               max_estimated_minutes: float | None = 60,
               auto_continue_after_warmup: bool = False) -> dict[str, Any]:
    resolved = load_execution_profile(Path(__file__).resolve().parents[1], profile)
    actual_target = _target_for_profile(profile, target_games_per_opponent)
    if type(warmup_games_per_opponent) is not int or warmup_games_per_opponent <= 0:
        raise ValueError('warmup_games_per_opponent must be positive')
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if type(auto_continue_after_warmup) is not bool:
        raise ValueError('auto_continue_after_warmup must be boolean')
    if max_estimated_minutes is not None:
        import math
        if isinstance(max_estimated_minutes, bool) or not isinstance(max_estimated_minutes, (int, float)) or not math.isfinite(max_estimated_minutes) or max_estimated_minutes <= 0:
            raise ValueError('max_estimated_minutes must be positive and finite')
    return {'format': 'guandan-ability-evaluation-plan-v3', 'profile': resolved,
            'progress_bundle': str(progress_bundle), 'output_root': str(Path(output_root).resolve()),
            'batch_size': batch_size, 'workers': {'ippo': device_ippo, 'vrpo': device_vrpo},
            'matchups': {'self_old_snapshot': ['random', 'rule', 'snapshot'],
                         'cross_algorithm_update102': ['snapshot']},
            'warmup_games_per_opponent': warmup_games_per_opponent,
            'target_games_per_opponent_requested': target_games_per_opponent,
            'target_games_per_opponent_actual': actual_target,
            'max_estimated_minutes': max_estimated_minutes,
            'auto_continue_after_warmup': auto_continue_after_warmup,
            'expected_formal_games_total': 8 * actual_target,
            'expected_warmup_games_total': 8 * warmup_games_per_opponent,
            'accepted': False, 'kaggle_verified': False}


def _validate_pair(device_ippo: str, device_vrpo: str) -> None:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('evaluation suite requires two CUDA devices')
    indices = [_device_index(device_ippo), _device_index(device_vrpo)]
    if any(index is None for index in indices) or indices[0] == indices[1] or any(index >= torch.cuda.device_count() for index in indices):
        raise RuntimeError('evaluation workers require distinct visible cuda indices')


def _warmup_decision(warmup_messages: list[dict], *, auto_continue: bool,
                     max_estimated_minutes: float | None) -> dict[str, Any]:
    if len(warmup_messages) != 2 or any(
        len(message.get('reports', [])) != 2 or
        any(report.get('status') != 'warmup_complete' for report in message.get('reports', []))
        for message in warmup_messages
    ):
        return {'continue_formal': False, 'status': 'failed', 'reason': 'warmup incomplete or failed'}
    worker_estimates = []
    for message in warmup_messages:
        estimates = [report.get('metadata', {}).get('warmup', {}).get('estimated_remaining_seconds')
                     for report in message['reports']]
        if any(value is None for value in estimates):
            return {'continue_formal': False, 'status': 'stopped_by_budget', 'reason': 'estimate unavailable'}
        worker_estimates.append(sum(estimates))
    # The workers run concurrently; reports within each worker run sequentially.
    total_estimate = max(worker_estimates)
    decision = {'estimated_remaining_seconds': total_estimate,
                'estimated_finish': (datetime.now(timezone(timedelta(hours=8))) + timedelta(seconds=total_estimate)).isoformat(),
                'worker_estimates_seconds': worker_estimates}
    if max_estimated_minutes is not None and total_estimate > max_estimated_minutes * 60:
        return {**decision, 'continue_formal': False, 'status': 'stopped_by_budget',
                'reason': 'warmup estimate exceeds max_estimated_minutes'}
    return {**decision, 'continue_formal': auto_continue,
            'status': 'formal_starting' if auto_continue else 'warmup_complete'}



# Allow a short drain window for multiprocessing Queue's feeder after process
# exit. A zero exit code is not a substitute for the expected protocol message.
_WORKER_EXIT_GRACE_SECONDS = 3.0


def _check_worker_exits(processes, specs, finals, exited_since, *, phase):
    now = time.monotonic()
    for process, spec in zip(processes, specs):
        algorithm = spec.algorithm
        if algorithm in finals:
            exited_since.pop(algorithm, None)
            continue
        if process.exitcode is None or process.is_alive():
            exited_since.pop(algorithm, None)
            continue
        first_seen = exited_since.setdefault(algorithm, now)
        if now - first_seen >= _WORKER_EXIT_GRACE_SECONDS:
            raise RuntimeError(
                f'{algorithm} worker exited without expected message during {phase}; '
                f'exitcode={process.exitcode}; queue drain grace expired')

def run(*, execute: bool = False, profile: str = 'remote_full', progress_bundle: str = '',
        output_root: str | Path = 'runs/evaluation-suite', batch_size: int = 16,
        device_ippo: str = 'cuda:0', device_vrpo: str = 'cuda:1',
        warmup_games_per_opponent: int = 10, target_games_per_opponent: int = 100,
        max_estimated_minutes: float | None = 60,
        auto_continue_after_warmup: bool = False) -> dict[str, Any]:
    plan = build_plan(profile=profile, progress_bundle=progress_bundle, output_root=output_root,
                      batch_size=batch_size, device_ippo=device_ippo, device_vrpo=device_vrpo,
                      warmup_games_per_opponent=warmup_games_per_opponent,
                      target_games_per_opponent=target_games_per_opponent,
                      max_estimated_minutes=max_estimated_minutes,
                      auto_continue_after_warmup=auto_continue_after_warmup)
    root = Path(output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('evaluation output is not empty; choose a new directory')
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / 'plan.json', plan)
    _atomic_json(root / 'run.json', {'run_id':root.name,'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'source':source_provenance(Path(__file__).resolve().parents[1]),'config':plan})
    if not execute:
        return {'status': 'plan_only', **plan}
    reporter=SuiteProgress(root,plan)
    reporter.snapshot(status='validating_inputs')
    try:
        if device_ippo.startswith('cuda') or device_vrpo.startswith('cuda') or profile == 'remote_full':
            _validate_pair(device_ippo, device_vrpo)
        if not progress_bundle:
            raise ValueError('evaluation requires an explicit progress bundle')
        bundle_root = _extract_bundle(progress_bundle, root / 'progress')
        checkpoints = {f'{alg}{update}':_checkpoint_for(bundle_root, alg, update)
                       for alg in ('ippo','vrpo') for update in (100,102)}
    except Exception as exc:
        result={'status':'failed','error':str(exc),'error_type':type(exc).__name__, 'accepted':False,'kaggle_verified':False}
        _atomic_json(root/'result.json',result)
        reporter.event('suite_failed', **result)
        reporter.snapshot(status='failed')
        return result
    _atomic_json(root/'input_manifest.json', {'read_only':True,'checkpoints':{key:{'path':str(path),'sha256':sha256(path)} for key,path in checkpoints.items()}})
    reporter.snapshot(status='warmup_running')
    context = mp.get_context('spawn'); result_queue = context.Queue(); control_queues = [context.Queue(), context.Queue()]
    specs = [
        EvalWorkerSpec('ippo', str(checkpoints['ippo102']), str(checkpoints['ippo100']), str(checkpoints['vrpo102']), device_ippo, profile, batch_size, str(root), 2_000_000, warmup_games_per_opponent, target_games_per_opponent, max_estimated_minutes),
        EvalWorkerSpec('vrpo', str(checkpoints['vrpo102']), str(checkpoints['vrpo100']), str(checkpoints['ippo102']), device_vrpo, profile, batch_size, str(root), 3_000_000, warmup_games_per_opponent, target_games_per_opponent, max_estimated_minutes),
    ]
    processes = [context.Process(target=_worker, args=(spec, result_queue, control_queues[index]), name=f'guandan-eval-{spec.algorithm}') for index, spec in enumerate(specs)]
    warmups: dict[str, dict] = {}; finals: dict[str, dict] = {}
    exited_since: dict[str, float] = {}
    decision = {'continue_formal': False, 'status': 'failed'}
    suite_started=time.monotonic()
    formal_started=None
    last_monitor=0.0
    last_activity={spec.algorithm:suite_started for spec in specs}

    def monitor():
        nonlocal last_monitor
        now=time.monotonic()
        if now-last_monitor>=10:
            progress=reporter.snapshot()
            print(progress_line(progress),flush=True)
            last_monitor=now
        # These are safety stops, not claims that ETA is exact.
        if now-suite_started > (max_estimated_minutes or 240)*60:
            raise TimeoutError('wall_clock_budget_exceeded')
        for algorithm,last in last_activity.items():
            if algorithm not in finals and (formal_started is not None or algorithm not in warmups) and now-last>300:
                raise RuntimeError(f'worker_stalled: {algorithm}: no token/game progress for 300 seconds')

    signal_handlers={}
    def stop_signal(number, frame):
        raise KeyboardInterrupt(f'evaluation stop signal {number}')
    try:
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                signal_handlers[number]=signal.getsignal(number)
                signal.signal(number,stop_signal)
        for process in processes: process.start()
        while len(warmups) < len(processes):
            monitor()
            try:
                message = result_queue.get(timeout=2)
            except Empty:
                _check_worker_exits(processes, specs, finals, exited_since, phase='warmup barrier')
                continue
            if message.get('kind') == 'progress':
                reporter.update(message['payload'])
                last_activity[message['payload']['algorithm']]=time.monotonic()
                continue
            if message.get('kind') == 'warmup_complete':
                warmups[message['algorithm']] = message
                reports = message.get('reports', [])
                if len(reports) != 2 or any(row.get('status') != 'warmup_complete' for row in reports):
                    raise RuntimeError(f"invalid warmup reports from {message['algorithm']}")
            elif message.get('kind') == 'final':
                finals[message['algorithm']] = message
                if message.get('status') == 'stopped_by_budget':
                    raise TimeoutError(message.get('error','worker budget exhausted'))
                raise RuntimeError(f"{message['algorithm']} worker finalized before warmup barrier: {message.get('status')}")
            _check_worker_exits(processes, specs, finals, exited_since, phase='warmup barrier')
        decision = _warmup_decision(list(warmups.values()), auto_continue=auto_continue_after_warmup,
                                    max_estimated_minutes=max_estimated_minutes)
        decision['warmup_wall_seconds']=time.monotonic()-suite_started
        if decision.get('continue_formal') and max_estimated_minutes is not None and decision.get('estimated_remaining_seconds',0)+decision['warmup_wall_seconds'] > max_estimated_minutes*60:
            decision.update(continue_formal=False,status='stopped_by_budget',reason='warmup + estimated formal exceeds total budget')
        reporter.event('warmup_completed', **decision)
        reporter.snapshot(status=decision['status'],decision=decision)
        if decision['continue_formal']:
            formal_started=time.monotonic()
            last_activity={spec.algorithm:formal_started for spec in specs}
        _atomic_json(root / 'live' / 'warmup_barrier.json', {'event': 'warmup_barrier', **decision, 'workers': warmups})
        for queue in control_queues:
            queue.put(decision)
        while len(finals) < len(processes):
            monitor()
            try:
                message = result_queue.get(timeout=2)
            except Empty:
                _check_worker_exits(processes, specs, finals, exited_since, phase='final result')
                continue
            if message.get('kind') == 'progress':
                reporter.update(message['payload'])
                last_activity[message['payload']['algorithm']]=time.monotonic()
                continue
            if message.get('kind') == 'final':
                finals[message['algorithm']] = message
                if message.get('status') not in {'complete', 'warmup_complete', 'stopped_by_budget'}:
                    raise RuntimeError(f"{message['algorithm']} worker failed: {message.get('status')}")
            _check_worker_exits(processes, specs, finals, exited_since, phase='final result')
    except BaseException as exc:
        failure_status = 'stopped_by_budget' if isinstance(exc, TimeoutError) else 'incomplete' if isinstance(exc, KeyboardInterrupt) else 'failed'
        decision = {'continue_formal': False, 'status': failure_status, 'reason': str(exc),
                    'error_type': type(exc).__name__}
        # Wake barrier waiters before bounded join/terminate cleanup. Do not
        # wait for a warmup message that a failed process can never send.
        for queue in control_queues:
            try:
                queue.put(decision)
            except Exception:
                pass
        for spec in specs:
            finals.setdefault(spec.algorithm, {'kind': 'final', 'status': failure_status,
                'algorithm': spec.algorithm, 'device': spec.device,
                'error': str(exc), 'reason': 'suite aborted',
                'reports': warmups.get(spec.algorithm, {}).get('reports', [])})
        _atomic_json(root / 'live' / 'warmup_barrier.json',
                     {'event': 'suite_failed', **decision, 'workers': warmups})
        reporter.event('suite_stopped' if failure_status!='failed' else 'suite_failed', status=failure_status, reason=str(exc))
        _atomic_json(root / 'result.json', {'status': failure_status, 'error': str(exc),
                     'error_type': type(exc).__name__, 'workers': finals, 'warmups': warmups})
    finally:
        for process in processes:
            # A process whose start() failed must not prevent sibling cleanup.
            try:
                process.join(timeout=3)
                if process.is_alive(): process.terminate(); process.join(timeout=3)
            except (AssertionError,RuntimeError):
                pass
        for number,handler in signal_handlers.items():
            signal.signal(number,handler)
    statuses = [row.get('status') for row in finals.values()]
    if decision.get('status') in {'failed','incomplete','stopped_by_budget'}: status = decision['status']
    elif statuses and all(status == 'complete' for status in statuses): status = 'complete'
    elif any(status == 'failed' for status in statuses): status = 'failed'
    elif decision.get('status') == 'stopped_by_budget': status = 'stopped_by_budget'
    elif statuses and all(value == 'warmup_complete' for value in statuses): status = 'warmup_complete'
    else: status = 'incomplete'
    final = {'format': 'guandan-ability-evaluation-v3', 'status': status, 'profile': profile,
             'checkpoints': {key: str(value) for key, value in checkpoints.items()},
             'workers': finals if finals else warmups, 'warmup_barrier': decision,
             'warmup_games_per_opponent': warmup_games_per_opponent,
             'target_games_per_opponent_actual': plan['target_games_per_opponent_actual'],
             'accepted': False, 'kaggle_verified': False}
    progress=reporter.snapshot(status=status,decision=decision,finals=finals)
    final.update(elapsed_seconds=time.monotonic()-suite_started, completed_warmup_games=progress['warmup_completed_games'], completed_formal_games=progress['formal_completed_games'])
    reporter.event('suite_finished',status=status,completed_warmup_games=final['completed_warmup_games'], completed_formal_games=final['completed_formal_games'])
    _atomic_json(root / 'result.json', final)
    return final


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='GuanDan ability evaluation suite')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--profile', default='remote_full', choices=('local_fast', 'remote_full'))
    parser.add_argument('--progress-bundle', required=True)
    parser.add_argument('--output-root', default='runs/evaluation-suite')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device-ippo', default='cuda:0')
    parser.add_argument('--device-vrpo', default='cuda:1')
    parser.add_argument('--warmup-games-per-opponent', type=int, default=10)
    parser.add_argument('--target-games-per-opponent', type=int, default=100)
    parser.add_argument('--max-estimated-minutes', type=float, default=60)
    parser.add_argument('--auto-continue-after-warmup', action='store_true')
    args = parser.parse_args(argv)
    result = run(**vars(args)); print(json.dumps({key: result.get(key) for key in ('status', 'accepted', 'kaggle_verified')}, ensure_ascii=False))
    return 0 if result['status'] in {'plan_only', 'complete', 'warmup_complete', 'stopped_by_budget'} else 2


if __name__ == '__main__': raise SystemExit(main())
