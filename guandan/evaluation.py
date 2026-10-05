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
from .evaluation_agents import SnapshotAgent, make_agent, legal_tokens
from .evaluation_stats import summarize_games
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
              env_factory=GuandanEnv, agent_factory=make_agent, device='cpu') -> GameRecord:
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


def evaluate_profile(*, profile='local_fast', seed=EVALUATION_SEED_BASE,
                     max_hours=None, evidence_path=None, candidate_checkpoint=None,
                     snapshot_checkpoint=None, deadline=None, device=None,
                     progress_callback=None) -> EvaluationResult:
    resolved = resolved_profile(profile)  # consumes/validates runner canonical JSON
    config = resolved['config']['evaluation']
    device = device or ('cuda' if profile == 'remote_full' else 'cpu')
    if profile == 'remote_full' and device != 'cuda':
        raise ValueError('remote_full requires CUDA evaluation')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested for evaluation but unavailable')
    if type(seed) is not int or seed < EVALUATION_SEED_BASE:
        raise ValueError('invalid evaluation seed namespace')
    started = time.monotonic()
    deadline = deadline or Deadline(config['max_hours'] if max_hours is None else max_hours)
    target = config['deal_groups_per_pairing'] * len(config['seat_rotations'])
    records = []
    completed_per_opponent = {name: 0 for name in config['opponents']}
    output = None if evidence_path is None else Path(evidence_path)
    previous_threads = torch.get_num_threads()
    metadata = {'resolved_profile': resolved, 'runtime': runtime_metadata(device),
                'protocol_versions': [RULES_VERSION,ACTION_VERSION,ENV_VERSION,ENCODING_VERSION],
                'scope': 'pipeline correctness only; fixed FIVE, standalone first hand; no strength claim',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'started_local': datetime.now(timezone(timedelta(hours=8))).isoformat()}
    try:
        deadline.check()
        if (candidate_checkpoint is None) != (snapshot_checkpoint is None):
            raise ValueError('supply both candidate and opponent checkpoint, or neither')
        if candidate_checkpoint is None:
            sources = reviewed_checkpoints()
            candidate_checkpoint, snapshot_checkpoint = sources['candidate'], sources['snapshot']
            metadata['source_evidence'] = sources
        # Capture hashes/versions once, before any counted game; load errors fail.
        metadata['candidate'] = SnapshotAgent(candidate_checkpoint,device=device).metadata
        metadata['snapshot_opponent'] = SnapshotAgent(snapshot_checkpoint,device=device).metadata
        metadata['max_hours'] = config['max_hours'] if max_hours is None else max_hours
        torch.set_num_threads(1)
        for opponent in config['opponents']:
            for group in range(config['deal_groups_per_pairing']):
                for rotation in config['seat_rotations']:
                    record = play_game(
                        opponent=opponent, deal_group=group, seat_rotation=rotation, seed=seed+group,
                        candidate_checkpoint=candidate_checkpoint, snapshot_checkpoint=snapshot_checkpoint,
                        deadline=deadline, device=device,
                    )
                    records.append(record)  # completed games ONLY
                    completed_per_opponent[opponent] += 1
                    if progress_callback is not None and (len(records) % 25 == 0 or len(records) == target * len(config['opponents'])):
                        progress_callback({
                            "event": "game_complete", "opponent": opponent,
                            "deal_group": group, "seat_rotation": rotation,
                            "completed_games": len(records),
                            "target_games": target * len(config['opponents']),
                            "completed_per_opponent": dict(completed_per_opponent),
                            "elapsed_seconds": time.monotonic() - started,
                        })
                    if output is not None:
                        write_evidence(output.parent / 'games' / f'{opponent}_{group:04d}_{rotation}.json', asdict(record))
        deadline.check()
        summaries = summarize_games([asdict(row) for row in records])
        counts = {opponent: summaries['per_opponent'][opponent]['completed'] for opponent in config['opponents']}
        if any(value != target for value in counts.values()) or len(records) != target*len(config['opponents']):
            raise RuntimeError('completed counts differ from profile')
        result = EvaluationResult(profile, 'complete', target, counts, len(records), tuple(records),
                                  time.monotonic()-started, seed, summaries, metadata)
        if output is not None:
            write_evidence(output, result.as_dict())
        return result
    except Exception as exc:
        incomplete = isinstance(exc, (BudgetExceeded, EvaluationIncomplete))
        report = {'status': 'incomplete' if incomplete else 'failed',
                  'reason': f'{type(exc).__name__}: {exc}', 'completed': len(records),
                  'target': target*len(config['opponents']), 'metadata': metadata,
                  'records': [asdict(row) for row in records]}
        if output is not None:
            write_evidence(output.with_name(output.stem + '_failure.json'), report)
        if incomplete:
            raise EvaluationIncomplete(report) from exc
        raise EvaluationFailure(report) from exc
    finally:
        torch.set_num_threads(previous_threads)


__all__ = ['EvaluationFailure','EvaluationIncomplete','EvaluationResult','GameRecord',
           'evaluate_profile','play_game','reviewed_checkpoints','EVALUATION_SEED_BASE']
