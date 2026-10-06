"""Profile-driven A06 evaluation with real model snapshots and policy-only inputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from .cards import Rank
from .environment import GameConfig, GuandanEnv
from .environment_types import ACTION_VERSION, RULES_VERSION, ENV_VERSION, ENCODING_VERSION
from .evaluation_agents import EvaluationSession, SnapshotAgent, make_agent, legal_tokens
from .evaluation_stats import summarize_games
from .observability import atomic_json
from .training.runtime import (ROOT, EVALUATION_SEED_BASE, BudgetExceeded, Deadline,
                               resolved_profile, runtime_metadata, file_sha256, write_evidence)


class EvaluationFailure(RuntimeError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__(report.get('reason', 'evaluation failed'))


class EvaluationIncomplete(EvaluationFailure):
    """Timeout/truncation is not a completed evaluation."""


@dataclass(frozen=True)
class GameRecord:
    opponent: str
    deal_group: int
    seat_rotation: int
    policy_seat: int
    candidate_team: int
    seed: int
    done: bool
    truncated: bool
    ranking: list[int]
    winner_team: int
    outcome_class: str
    rewards: list[float]
    team_rewards: list[float]
    token_steps: int
    committed_steps: int
    base_deal_sha256: str
    terminal: bool
    initial_leader: int
    logical_to_physical: list[int]
    policies_by_seat: list[str]
    policy_seeds_by_seat: list[int]
    decisions_by_seat: list[int]
    policy_hand_sha256: str
    action_trace_sha256: str
    agent_metadata: list[dict]


@dataclass(frozen=True)
class EvaluationResult:
    profile: str
    status: str
    target_games_per_opponent: int
    warmup_games_per_opponent: int
    warmup_completed_games_per_opponent: dict[str, int]
    completed_games_per_opponent: dict[str, int]
    total_games: int
    records: tuple[GameRecord, ...]
    runtime_seconds: float
    seed_namespace: int
    summary: dict
    metadata: dict
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def reviewed_checkpoints() -> dict:
    """Resolve EXACT artifacts from the reviewed A05 report, not a latest-file glob."""
    status = (ROOT / 'project_status/STATUS.yaml').read_text(encoding='utf-8')
    match = re.search(r'  A05_training_integration:\n(.*?)(?=  A06_evaluation:)', status, re.S)
    if match is None or not re.search(r'^    local_ready: true$', match[1], re.M):
        raise ValueError('A06 requires reviewed A05.local_ready=true')
    report_match = re.search(r'^        report: ([^\n]+)$', match[1], re.M)
    commit_match = re.search(r'^        code_commit: ([0-9a-f]{40})$', match[1], re.M)
    if report_match is None or commit_match is None:
        raise ValueError('A05 reviewed report/commit missing')
    report_path = (ROOT / report_match[1].strip()).resolve()
    if not report_path.is_relative_to(ROOT):
        raise ValueError('A05 report escapes project')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['status'] != 'passed' or report['profile']['name'] != 'local_fast':
        raise ValueError('A05 report is not passing local_fast evidence')
    if report['provenance_before']['git_commit'] != commit_match[1]:
        raise ValueError('A05 report commit differs from main-reviewed commit')
    manifest = {item['path']: item for item in report['execution_artifacts']}
    summaries = [name for name in manifest if name.endswith('/execution/training_counts.json')]
    if len(summaries) != 1:
        raise ValueError('A05 training-count evidence missing or ambiguous')
    counts_path = ROOT / summaries[0]
    if file_sha256(counts_path) != manifest[summaries[0]]['sha256']:
        raise ValueError('A05 count evidence digest mismatch')
    counts = json.loads(counts_path.read_text(encoding='utf-8'))
    paths = {}
    for role, phase in (('candidate', 'resume'), ('snapshot', 'base')):
        run = counts['algorithms']['ippo'][phase]
        path = Path(run['checkpoint']).resolve()
        relative = path.relative_to(ROOT).as_posix()
        if relative not in manifest or file_sha256(path) != manifest[relative]['sha256']:
            raise ValueError('A05 checkpoint digest missing/mismatched')
        paths[role] = str(path)
    if paths['candidate'] == paths['snapshot']:
        raise ValueError('candidate and frozen opponent snapshot must be distinct checkpoints')
    paths['a05_report'] = str(report_path)
    paths['a05_report_sha256'] = file_sha256(report_path)
    return paths


def _snapshot_path():
    """Compatibility helper: explicit main-reviewed opponent checkpoint."""
    return Path(reviewed_checkpoints()['snapshot'])


def _assignment(rotation: int, opponent: str):
    if type(rotation) is not int or rotation not in range(4):
        raise ValueError('rotation must be one of 0,1,2,3')
    if opponent not in ('random', 'rule', 'snapshot'):
        raise ValueError('unknown opponent; no fallback')
    mapping = [(seat+rotation) % 4 for seat in range(4)]
    policies = [None]*4
    for logical, physical in enumerate(mapping):
        policies[physical] = 'candidate' if logical % 2 == 0 else opponent
    return mapping, policies


def play_game(*, opponent: str, deal_group: int, seat_rotation: int, seed: int,
              candidate_checkpoint, snapshot_checkpoint, deadline=None,
              env_factory=GuandanEnv, agent_factory=make_agent, device='cpu', progress_callback=None) -> GameRecord:
    """Keep the physical deal/leader fixed; rotate the four agent roles instead.

    logical seats 0/2 are the evaluated policy team, 1/3 the opponent team.
    The primary logical seat 0 actually visits each physical seat. Rotations
    with the same team parity still swap the agents' independent RNG streams.
    """
    if type(seed) is not int or seed < EVALUATION_SEED_BASE:
        raise ValueError('evaluation seeds must lie outside reserved training namespace')
    if type(deal_group) is not int or deal_group < 0:
        raise ValueError('deal_group must be a nonnegative integer')
    mapping, policies = _assignment(seat_rotation, opponent)
    deadline = deadline or Deadline(0.5)
    deadline.check()
    env = env_factory(GameConfig(level_rank=Rank.FIVE, seed=seed, leader_seat=0))
    observation = env.reset()
    original = env.full_state()['round_state']['hands']
    if len(original) != 4 or any(len(hand) != 27 for hand in original) or sorted(sum(original, [])) != list(range(108)):
        raise ValueError('counted evaluation requires a complete 108-card 27x4 deal')
    digest = _digest([sorted(hand) for hand in original])
    primary_hand_hash = _digest(sorted(original[seat_rotation]))
    agents = [None]*4
    agent_seeds = [0]*4
    pairing_index = ('random', 'rule', 'snapshot').index(opponent)
    for logical, physical in enumerate(mapping):
        rng_seed = 20_000_000 + pairing_index*1_000_000 + deal_group*1_000 + seat_rotation*10 + logical
        name = policies[physical]
        agents[physical] = agent_factory(
            'snapshot' if name == 'candidate' else name,
            seed=rng_seed, checkpoint=candidate_checkpoint if name == 'candidate' else snapshot_checkpoint,
            device=device,
        )
        agent_seeds[physical] = rng_seed
        for trained in getattr(agents[physical], 'training_deals', []):
            if trained['seed'] == seed or trained['deal_sha256'] == digest:
                raise ValueError('training/evaluation deal overlap')
    deadline.check()
    decisions = [0]*4
    trace = hashlib.sha256()
    result = None
    last_progress = time.monotonic()
    while not env.done:
        deadline.check()
        if env.truncated or observation is None:
            raise EvaluationIncomplete({'status':'incomplete', 'reason':'max_token_steps before true terminal'})
        actor = observation.player_id
        token = agents[actor].select_token(observation)  # observation-only API, no full_state
        if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)) or token not in legal_tokens(observation):
            raise ValueError('agent returned illegal token; no fallback')
        previous_committed, previous_token = env.committed_step, env.token_step
        result = env.step(int(token))
        decisions[actor] += 1
        if result.token_step != previous_token+1 or result.committed_step != previous_committed+int(result.is_commit):
            raise RuntimeError('evaluation transition clocks disagree')
        if not result.done and np.any(result.rewards):
            raise RuntimeError('nonterminal evaluation reward')
        trace.update(json.dumps([actor, int(token), result.token_step, result.committed_step,
                                result.rewards.tolist()], separators=(',', ':')).encode())
        observation = result.observation
        if progress_callback is not None and time.monotonic() - last_progress >= 10:
            progress_callback({'opponent': opponent, 'token_steps': env.token_step, 'inflight_games': 1})
            last_progress = time.monotonic()
        deadline.check()  # final slow transition cannot bypass the time budget
    if result is None or not result.done or result.truncated:
        raise RuntimeError('no real terminal transition')
    info = result.info
    row = GameRecord(
        opponent, deal_group, seat_rotation, seat_rotation, seat_rotation%2, seed, True, False,
        list(info['ranking']), int(info['winner_team']), info['outcome_class'],
        result.rewards.tolist(), list(info['team_reward']), result.token_step, result.committed_step,
        digest, True, 0, mapping, policies, agent_seeds, decisions, primary_hand_hash,
        trace.hexdigest(), [dict(agent.metadata) for agent in agents],
    )
    summarize_games([asdict(row)])  # validate ranking/rewards before counting this game
    return row


@dataclass
class _BatchGame:
    opponent: str
    deal_group: int
    seat_rotation: int
    seed: int
    env: object
    observation: object
    mapping: list[int]
    policies: list[str]
    agents: list[object]
    agent_seeds: list[int]
    decisions: list[int]
    digest: str
    primary_hand_hash: str
    trace: object
    result: object = None


def _new_batch_game(*, opponent, deal_group, seat_rotation, seed, session) -> _BatchGame:
    if type(seed) is not int or seed < EVALUATION_SEED_BASE:
        raise ValueError('evaluation seeds must lie outside reserved training namespace')
    if type(deal_group) is not int or deal_group < 0:
        raise ValueError('deal_group must be a nonnegative integer')
    mapping, policies = _assignment(seat_rotation, opponent)
    env = GuandanEnv(GameConfig(level_rank=Rank.FIVE, seed=seed, leader_seat=0))
    observation = env.reset()
    original = env.full_state()['round_state']['hands']
    if len(original) != 4 or any(len(hand) != 27 for hand in original) or sorted(sum(original, [])) != list(range(108)):
        raise ValueError('counted evaluation requires a complete 108-card 27x4 deal')
    digest = _digest([sorted(hand) for hand in original])
    primary_hand_hash = _digest(sorted(original[seat_rotation]))
    agents = [None] * 4
    agent_seeds = [0] * 4
    pairing_index = ('random', 'rule', 'snapshot').index(opponent)
    for logical, physical in enumerate(mapping):
        rng_seed = 20_000_000 + pairing_index * 1_000_000 + deal_group * 1_000 + seat_rotation * 10 + logical
        name = policies[physical]
        agents[physical] = session.make_agent(
            'snapshot' if name == 'candidate' else name,
            seed=rng_seed,
            checkpoint=session.candidate_checkpoint if name == 'candidate' else session.snapshot_checkpoint,
        )
        agent_seeds[physical] = rng_seed
        for trained in getattr(agents[physical], 'training_deals', []):
            if trained['seed'] == seed or trained['deal_sha256'] == digest:
                raise ValueError('training/evaluation deal overlap')
    return _BatchGame(
        opponent, deal_group, seat_rotation, seed, env, observation, mapping, policies,
        agents, agent_seeds, [0] * 4, digest, primary_hand_hash, hashlib.sha256(),
    )


def _finish_batch_game(state: _BatchGame) -> GameRecord:
    result = state.result
    if result is None or not result.done or result.truncated:
        raise RuntimeError('no real terminal transition')
    info = result.info
    row = GameRecord(
        state.opponent, state.deal_group, state.seat_rotation, state.seat_rotation,
        state.seat_rotation % 2, state.seed, True, False, list(info['ranking']),
        int(info['winner_team']), info['outcome_class'], result.rewards.tolist(),
        list(info['team_reward']), result.token_step, result.committed_step,
        state.digest, True, 0, state.mapping, state.policies, state.agent_seeds,
        state.decisions, state.primary_hand_hash, state.trace.hexdigest(),
        [dict(agent.metadata) for agent in state.agents],
    )
    summarize_games([asdict(row)])
    return row


def _evaluate_batch(tasks, *, session, deadline, on_game=None, on_progress=None) -> list[GameRecord]:
    setup_started=time.monotonic()
    states = [_new_batch_game(**task, session=session) for task in tasks]
    times={'environment_step_seconds':0.0,'policy_inference_seconds':0.0,'game_setup_seconds':time.monotonic()-setup_started}
    records = []
    last_progress = time.monotonic()
    total_tokens = 0

    def finish(state):
        record = _finish_batch_game(state)
        records.append(record)
        if on_game is not None:
            on_game(record)

    while states:
        deadline.check()
        requests = []
        request_indices = []
        tokens = {}
        for index, state in enumerate(states):
            if state.env.done:
                finish(state)
                continue
            if state.env.truncated or state.observation is None:
                raise EvaluationIncomplete({'status': 'incomplete', 'reason': 'max_token_steps before true terminal'})
            actor = state.observation.player_id
            agent = state.agents[actor]
            if isinstance(agent, SnapshotAgent):
                requests.append((agent, state.observation))
                request_indices.append((index, actor))
            else:
                tokens[index] = int(agent.select_token(state.observation))
        if requests:
            inference_started=time.monotonic()
            selected_tokens=SnapshotAgent.select_tokens_batch(requests)
            times['policy_inference_seconds']+=time.monotonic()-inference_started
            for (index, actor), token in zip(request_indices, selected_tokens):
                tokens[index] = token
        survivors = []
        for index, state in enumerate(states):
            if state.env.done:
                continue
            actor = state.observation.player_id
            token = tokens[index]
            if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)) or token not in legal_tokens(state.observation):
                raise ValueError('agent returned illegal token; no fallback')
            previous_committed, previous_token = state.env.committed_step, state.env.token_step
            step_started=time.monotonic()
            result = state.env.step(int(token))
            times['environment_step_seconds']+=time.monotonic()-step_started
            state.decisions[actor] += 1
            if result.token_step != previous_token + 1 or result.committed_step != previous_committed + int(result.is_commit):
                raise RuntimeError('evaluation transition clocks disagree')
            if not result.done and np.any(result.rewards):
                raise RuntimeError('nonterminal evaluation reward')
            state.trace.update(json.dumps([actor, int(token), result.token_step, result.committed_step, result.rewards.tolist()], separators=(',', ':')).encode())
            state.result = result
            state.observation = result.observation
            if state.env.done:
                finish(state)
            else:
                survivors.append(state)
        total_tokens += len(tokens)
        states = survivors
        if on_progress is not None and time.monotonic() - last_progress >= 10:
            on_progress({'opponent': tasks[0]['opponent'], 'token_steps': total_tokens,
                         'inflight_games': len(states), 'timings': dict(times)})
            last_progress = time.monotonic()
    if on_progress is not None:
        on_progress({'opponent':tasks[0]['opponent'],'token_steps':total_tokens,'inflight_games':0,'timings':dict(times)})
    return records


def _validate_game_count(value, *, name: str, allow_zero: bool = True) -> int:
    if type(value) is not int or (value < 0 if allow_zero else value <= 0):
        comparator = 'nonnegative' if allow_zero else 'positive'
        raise ValueError(f'{name} must be a {comparator} integer')
    return value


def _iso_finish(seconds: float | None) -> str | None:
    if seconds is None or seconds < 0:
        return None
    return (datetime.now(timezone(timedelta(hours=8))) + timedelta(seconds=seconds)).isoformat()


def _progress_payload(*, phase: str, opponent: str | None, completed: int,
                      target: int, completed_per_opponent: dict[str, int],
                      started: float, phase_started: float,
                      token_steps: int, warmup_elapsed: float | None = None,
                      warmup_games: int = 0, warmup_token_steps: int = 0,
                      estimated_remaining_seconds: float | None = None) -> dict:
    elapsed = max(0.0, time.monotonic() - started)
    phase_elapsed = max(0.0, time.monotonic() - phase_started)
    rate = completed / phase_elapsed if completed and phase_elapsed > 0 else None
    token_rate = token_steps / phase_elapsed if token_steps and phase_elapsed > 0 else None
    return {
        'event': 'evaluation_progress', 'phase': phase, 'opponent': opponent,
        'completed_games': completed, 'target_games': target,
        'completed_per_opponent': dict(completed_per_opponent),
        'elapsed_seconds': elapsed, 'phase_elapsed_seconds': phase_elapsed,
        'games_per_second': rate, 'token_steps': token_steps,
        'token_steps_per_second': token_rate,
        'warmup_elapsed_seconds': warmup_elapsed, 'warmup_games': warmup_games,
        'warmup_token_steps': warmup_token_steps,
        'estimated_remaining_seconds': estimated_remaining_seconds,
        'estimated_finish': _iso_finish(estimated_remaining_seconds),
    }


def evaluate_profile(*, profile='local_fast', seed=EVALUATION_SEED_BASE,
                     max_hours=None, evidence_path=None, candidate_checkpoint=None,
                     snapshot_checkpoint=None, deadline=None, device=None,
                     progress_callback=None, batch_size=1,
                     warmup_games_per_opponent=0, target_games_per_opponent=None,
                     max_estimated_minutes=None, auto_continue_after_warmup=True,
                     opponents=None) -> EvaluationResult:
    """Evaluate policies with an optional non-counted warmup and ETA reporting."""
    resolved = resolved_profile(profile)
    config = resolved['config']['evaluation']
    device = device or ('cuda' if profile == 'remote_full' else 'cpu')
    if profile == 'remote_full' and device != 'cuda':
        raise ValueError('remote_full requires CUDA evaluation')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested for evaluation but unavailable')
    if type(seed) is not int or seed < EVALUATION_SEED_BASE:
        raise ValueError('invalid evaluation seed namespace')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    warmup_requested = _validate_game_count(warmup_games_per_opponent, name='warmup_games_per_opponent')
    rotations = [0, 1, 2, 3]
    rotation_count = len(rotations)
    if rotation_count < 1:
        raise ValueError('evaluation seat_rotations must not be empty')
    requested_target = config['deal_groups_per_pairing'] * len(config['seat_rotations']) if target_games_per_opponent is None else target_games_per_opponent
    target = _validate_game_count(requested_target, name='target_games_per_opponent', allow_zero=False)
    if target < rotation_count or target % rotation_count:
        raise ValueError('target_games_per_opponent must be a positive multiple of seat rotations')
    if max_estimated_minutes is not None:
        if isinstance(max_estimated_minutes, bool) or not isinstance(max_estimated_minutes, (int, float)):
            raise ValueError('max_estimated_minutes must be a positive finite number')
        if not np.isfinite(max_estimated_minutes) or max_estimated_minutes <= 0:
            raise ValueError('max_estimated_minutes must be a positive finite number')
    if type(auto_continue_after_warmup) is not bool:
        raise ValueError('auto_continue_after_warmup must be a boolean')

    started = time.monotonic()
    deadline = deadline or Deadline(config['max_hours'] if max_hours is None else max_hours)
    opponents = list(config['opponents'] if opponents is None else opponents)
    if not opponents or len(set(opponents)) != len(opponents) or any(name not in {'random', 'rule', 'snapshot'} for name in opponents):
        raise ValueError('opponents must be a non-empty subset of random, rule, snapshot')
    total_target = target * len(opponents)
    records: list[GameRecord] = []
    warmup_completed = {name: 0 for name in opponents}
    completed_per_opponent = {name: 0 for name in opponents}
    output = None if evidence_path is None else Path(evidence_path)
    previous_threads = torch.get_num_threads()
    metadata = {'resolved_profile': resolved, 'runtime': runtime_metadata(device),
                'protocol_versions': [RULES_VERSION, ACTION_VERSION, ENV_VERSION, ENCODING_VERSION],
                'scope': 'pipeline correctness only; fixed FIVE, standalone first hand; no strength claim',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'started_local': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'batch_size': batch_size, 'warmup_games_per_opponent': warmup_requested,
                'target_games_per_opponent_requested': target_games_per_opponent,
                'target_games_per_opponent_actual': target, 'seat_rotation_count': rotation_count,
                'max_estimated_minutes': max_estimated_minutes,
                'auto_continue_after_warmup': auto_continue_after_warmup,
                'opponents': list(opponents)}

    def emit(payload: dict) -> None:
        payload = dict(payload)
        payload.setdefault('target_games_per_opponent', target)
        payload.setdefault('total_target_games', total_target)
        payload.setdefault('target_games', payload['total_target_games'])
        payload.setdefault('completed_games', 0)
        payload.setdefault('estimated_remaining_seconds', None)
        payload.setdefault('estimated_finish', None)
        if progress_callback is not None:
            progress_callback(payload)

    formal_started = [started]
    warmup_started_elapsed = [None]
    warmup_token_steps = [0]
    timing_totals={}

    def run_tasks(tasks, *, phase, phase_started, phase_target, phase_counts,
                  phase_token_steps, record_sink=None):
        phase_records = []
        batch_token_base = 0
        last_opponent = None

        def progress(event, *, name='game_advanced'):
            completed = sum(phase_counts.values())
            rate = completed / max(time.monotonic() - phase_started, 1e-9)
            remaining = (phase_target - completed) / rate if rate else None
            payload = _progress_payload(
                phase=phase, opponent=event['opponent'], completed=completed,
                target=phase_target, completed_per_opponent=phase_counts,
                started=started, phase_started=phase_started,
                token_steps=event['token_steps'],
                warmup_games=sum(warmup_completed.values()),
                warmup_token_steps=warmup_token_steps[0],
                estimated_remaining_seconds=remaining)
            payload.update(event=name, inflight_games=event.get('inflight_games', 0),
                           completed_opponent_games=phase_counts.get(event['opponent'],0),
                           target_opponent_games=warmup_requested if phase=='warmup' else target,
                           avg_game_seconds=1/rate if rate else None, timings=event.get('timings'))
            emit(payload)

        def save(record):
            # Persist before notifying; a later failure in this batch must not
            # discard already-terminated games.
            if output is not None:
                directory = 'warmup_games' if phase == 'warmup' else 'games'
                game_path=output.parent / directory / f'{record.opponent}_{record.deal_group:04d}_{record.seat_rotation}.json'
                if game_path.exists():
                    raise FileExistsError(f'game record already exists: {game_path}')
                atomic_json(game_path, asdict(record))
            phase_records.append(record)
            if record_sink is not None:
                record_sink.append(record)
            phase_counts[record.opponent] += 1
            phase_token_steps[0] += record.token_steps
            progress({'opponent': record.opponent, 'token_steps': phase_token_steps[0]}, name='game_complete')
            per_target = warmup_requested if phase=='warmup' else target
            if phase_counts[record.opponent] == per_target:
                emit({'event':'opponent_completed','phase':phase,'opponent':record.opponent,'completed_games':sum(phase_counts.values()),'target_games':phase_target,'completed_per_opponent':dict(phase_counts)})

        batches = [group[offset:offset+batch_size] for opponent in opponents
                   for group in [[task for task in tasks if task['opponent']==opponent]]
                   for offset in range(0,len(group),batch_size)]
        for batch in batches:
            deadline.check()
            batch_times={}
            if batch[0]['opponent'] != last_opponent:
                last_opponent = batch[0]['opponent']
                emit({'event':'opponent_started','phase':phase,'opponent':last_opponent,'completed_games':sum(phase_counts.values()),'completed_per_opponent':dict(phase_counts),'target_games':phase_target})
            batch_token_base = phase_token_steps[0]
            def advanced(event):
                batch_times.update(event.get('timings',{}))
                progress({**event, 'token_steps': batch_token_base + event['token_steps']})
            if batch_size == 1:
                save(play_game(**batch[0], candidate_checkpoint=candidate_checkpoint,
                     snapshot_checkpoint=snapshot_checkpoint, deadline=deadline,
                     agent_factory=session_agent_factory, device=device, progress_callback=advanced))
            else:
                _evaluate_batch(batch, session=session, deadline=deadline,
                                on_game=save, on_progress=advanced)
            for key, value in batch_times.items():
                timing_totals[key] = timing_totals.get(key, 0.0) + value
        metadata['timings'] = {**timing_totals,
            'scope': 'host wall-clock; inference includes transfers and validation, not GPU kernel time',
            'available': bool(timing_totals)}
        return phase_records

    try:
        emit({'event':'evaluation_started', 'phase':'warmup' if warmup_requested else 'formal', 'status':'loading_models'})
        deadline.check()
        if (candidate_checkpoint is None) != (snapshot_checkpoint is None):
            raise ValueError('supply both candidate and opponent checkpoint, or neither')
        if candidate_checkpoint is None:
            sources = reviewed_checkpoints()
            candidate_checkpoint, snapshot_checkpoint = sources['candidate'], sources['snapshot']
            metadata['source_evidence'] = sources
        session = EvaluationSession(candidate_checkpoint, snapshot_checkpoint, device=device)
        metadata.update(session.metadata)
        metadata['max_hours'] = config['max_hours'] if max_hours is None else max_hours
        # EvaluationSession owns the already-loaded model resources; its factory
        # intentionally does not accept a device argument.  Adapt it to the
        # play_game factory protocol without changing agent/RNG semantics.
        session_agent_factory = lambda kind, seed, checkpoint, device=None: session.make_agent(
            kind, seed=seed, checkpoint=checkpoint)
        torch.set_num_threads(1)

        def make_tasks(*, count: int, seed_offset: int, formal: bool) -> list[dict]:
            tasks: list[dict] = []
            groups = (target // rotation_count) if formal else ((count + rotation_count - 1) // rotation_count)
            for opponent in opponents:
                selected = 0
                for group in range(groups):
                    for rotation in (rotations if formal else rotations[:max(0, min(rotation_count, count - selected))]):
                        if not formal and selected >= count:
                            break
                        tasks.append({'opponent': opponent, 'deal_group': group,
                                      'seat_rotation': rotation, 'seed': seed + seed_offset + group})
                        selected += 1
                    if not formal and selected >= count:
                        break
            return tasks

        if warmup_requested:
            warmup_started = time.monotonic()
            warmup_started_elapsed[0] = warmup_started - started
            warmup_tasks = make_tasks(count=warmup_requested, seed_offset=50_000_000, formal=False)
            emit({'event': 'warmup_started', 'phase': 'warmup',
                  'target_games_per_opponent': warmup_requested,
                  'total_target_games': warmup_requested * len(opponents),
                  'completed_per_opponent': dict(warmup_completed)})
            run_tasks(warmup_tasks, phase='warmup', phase_started=warmup_started,
                      phase_target=warmup_requested * len(opponents),
                      phase_counts=warmup_completed, phase_token_steps=warmup_token_steps)
            warmup_elapsed = time.monotonic() - warmup_started
            warmup_total = sum(warmup_completed.values())
            warmup_rate = warmup_total / warmup_elapsed if warmup_elapsed > 0 else 0.0
            warmup_token_rate = warmup_token_steps[0] / warmup_elapsed if warmup_elapsed > 0 else 0.0
            estimated_remaining = total_target / warmup_rate if warmup_rate > 0 else None
            metadata['warmup'] = {'status': 'complete', 'elapsed_seconds': warmup_elapsed,
                                  'completed_games': warmup_total, 'games_per_second': warmup_rate,
                                  'token_steps': warmup_token_steps[0],
                                  'token_steps_per_second': warmup_token_rate,
                                  'estimated_remaining_seconds': estimated_remaining,
                                  'estimated_finish': _iso_finish(estimated_remaining)}
            emit({'event': 'warmup_completed', 'phase': 'warmup',
                  'completed_games': warmup_total,
                  'target_games': warmup_requested * len(opponents),
                  'completed_per_opponent': dict(warmup_completed),
                  'elapsed_seconds': warmup_elapsed, 'games_per_second': warmup_rate,
                  'token_steps': warmup_token_steps[0], 'token_steps_per_second': warmup_token_rate,
                  'estimated_remaining_seconds': estimated_remaining,
                  'estimated_finish': _iso_finish(estimated_remaining),
                  'formal_target_games': total_target})
            warmup_status = None
            if max_estimated_minutes is not None and (estimated_remaining is None or estimated_remaining > max_estimated_minutes * 60):
                warmup_status = 'stopped_by_budget'
                metadata['warmup']['status'] = warmup_status
            elif not auto_continue_after_warmup:
                warmup_status = 'warmup_complete'
            if warmup_status is not None:
                summary = {'status': warmup_status, 'warmup': metadata['warmup']}
                warmup_result = EvaluationResult(
                    profile, warmup_status, target, warmup_requested, dict(warmup_completed),
                    {opponent: 0 for opponent in opponents}, 0, tuple(),
                    time.monotonic() - started, seed, summary, metadata)
                emit({'event': warmup_status, 'phase': 'warmup', 'status': warmup_status,
                      'completed_games': warmup_total, 'target_games': warmup_requested * len(opponents),
                      'formal_completed_games':0, 'formal_target_games':total_target,
                      'completed_per_opponent': dict(warmup_completed),
                      'estimated_remaining_seconds': estimated_remaining,
                      'estimated_finish': _iso_finish(estimated_remaining)})
                if output is not None:
                    write_evidence(output, warmup_result.as_dict())
                return warmup_result

        formal_started[0] = time.monotonic()
        formal_tasks = make_tasks(count=target, seed_offset=0, formal=True)
        emit({'event': 'formal_started', 'phase': 'formal',
              'target_games_per_opponent': target, 'total_target_games': total_target,
              'completed_per_opponent': dict(completed_per_opponent),
              'warmup': metadata.get('warmup')})
        formal_token_steps = [0]
        run_tasks(formal_tasks, phase='formal', phase_started=formal_started[0],
                  phase_target=total_target, phase_counts=completed_per_opponent,
                  phase_token_steps=formal_token_steps, record_sink=records)
        deadline.check()
        summaries = summarize_games([asdict(row) for row in records])
        counts = {opponent: summaries['per_opponent'][opponent]['completed'] for opponent in opponents}
        if any(value != target for value in counts.values()) or len(records) != total_target:
            raise RuntimeError('completed counts differ from profile')
        metadata['formal'] = {'status': 'complete', 'elapsed_seconds': time.monotonic() - formal_started[0],
                              'completed_games': len(records), 'token_steps': formal_token_steps[0]}
        result = EvaluationResult(profile, 'complete', target, warmup_requested,
                                  dict(warmup_completed), counts, len(records), tuple(records),
                                  time.monotonic() - started, seed, summaries, metadata)
        if output is not None:
            write_evidence(output, result.as_dict())
        emit({'event': 'evaluation_complete', 'phase': 'formal', 'completed_games': len(records),
              'target_games': total_target, 'completed_per_opponent': dict(counts),
              'elapsed_seconds': time.monotonic() - started, 'games_per_second':
              len(records) / max(time.monotonic() - formal_started[0], 1e-9),
              'estimated_remaining_seconds': 0, 'estimated_finish': _iso_finish(0)})
        return result
    except Exception as exc:
        if isinstance(exc, EvaluationIncomplete):
            report = exc.report
        else:
            report = {'status': 'incomplete' if isinstance(exc, BudgetExceeded) else 'failed',
                      'reason': f'{type(exc).__name__}: {exc}', 'completed': len(records),
                      'target': total_target, 'warmup_completed': dict(warmup_completed),
                      'metadata': metadata, 'records': [asdict(row) for row in records]}
        if output is not None:
            write_evidence(output.with_name(output.stem + '_failure.json'), report)
        try:
            emit({'event': 'evaluation_failed', 'status': report['status'],
                  'reason': report.get('reason'), 'completed_games': len(records),
                  'completed_per_opponent': dict(completed_per_opponent)})
        except Exception:
            pass  # Preserve the original failure if the consumer also fails.
        if isinstance(exc, (BudgetExceeded, EvaluationIncomplete)):
            raise EvaluationIncomplete(report) from exc
        raise EvaluationFailure(report) from exc
    finally:
        torch.set_num_threads(previous_threads)


__all__ = ['EvaluationFailure','EvaluationIncomplete','EvaluationResult','GameRecord',
           'evaluate_profile','play_game','reviewed_checkpoints','EVALUATION_SEED_BASE']
