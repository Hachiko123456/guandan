"""A07 remote_full session orchestration. No upload API and no automatic launch."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

from .provenance import sha256, source_provenance

ROOT = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def _require_input_path(path: Path, label: str, *, working_artifacts_root=None) -> Path:
    resolved = Path(path).expanduser().resolve()
    roots = [KAGGLE_INPUT.resolve()]
    if working_artifacts_root is not None:
        roots.append(Path(working_artifacts_root).resolve())
    if not any(resolved != root and resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"{label} must be under /kaggle/input or the explicit working artifact scope")
    return resolved


def _workflow_scope(args, working_artifacts_root):
    if working_artifacts_root is None:
        return None
    scope = Path(working_artifacts_root).expanduser().resolve()
    working = KAGGLE_WORKING.resolve()
    # Check independently of the local _output_directory test seam. Tests may
    # explicitly substitute simulated mounts, never disable scope validation.
    if scope == working or not scope.is_relative_to(working):
        raise ValueError('working_artifacts_root must be a named subdirectory of /kaggle/working')
    output = _output_directory(args.output_root)
    if scope != output or scope != Path(args.output_root).expanduser().resolve():
        raise ValueError('working_artifacts_root must equal the verified output-root')
    return scope


def _validate_production_paths(args, *, working_artifacts_root=None) -> None:
    _output_directory(args.output_root)
    for value, label in (
        (args.resume, "resume manifest"),
        (args.candidate_checkpoint, "candidate checkpoint"),
        (args.snapshot_checkpoint, "snapshot checkpoint"),
    ):
        if value is not None:
            _require_input_path(value, label, working_artifacts_root=working_artifacts_root)


def _recovery_origin(resume, working_artifacts_root):
    if resume is None:
        return 'none'
    path = Path(resume).expanduser().resolve()
    if working_artifacts_root is not None and path.is_relative_to(working_artifacts_root):
        return 'current_working_output'
    if path.is_relative_to(KAGGLE_INPUT.resolve()):
        return 'mounted_input'
    # Only an explicitly injected runtime seam can supply an unscoped local
    # fixture. Do not label that fixture as evidence of a new Kaggle session.
    return 'none'


def _profile():
    from ..execution_profiles import load_execution_profile
    return load_execution_profile(ROOT, 'remote_full')


@contextmanager
def canonical_profile(profile):
    keys = ('GUANDAN_PROFILE','GUANDAN_RESOLVED_PROFILE_JSON')
    previous = {key:os.environ.get(key) for key in keys}
    os.environ[keys[0]] = 'remote_full'
    os.environ[keys[1]] = json.dumps(profile,sort_keys=True)
    try:
        yield
    finally:
        for key,value in previous.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value


def resolve_recovery(manifest_path: Path, *, algorithm: str) -> dict:
    """Resolve only a sibling basename: old machine absolute paths are irrelevant."""
    from ..training.checkpoint import CURRENT_PROTOCOL_VERSIONS, load_checkpoint
    manifest_path = Path(manifest_path).resolve()
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    if data.get('format') != 'guandan-recovery-v1' or data.get('profile') != 'remote_full' or data.get('algorithm') != algorithm:
        raise ValueError('incompatible recovery manifest format/profile/algorithm')
    name = data.get('checkpoint_filename')
    if not isinstance(name,str) or not name or Path(name).name!=name or '/' in name or '\\' in name or name in {'.','..'}:
        raise ValueError('recovery must identify one sibling checkpoint basename')
    checkpoint = (manifest_path.parent / name).resolve()
    if checkpoint.parent != manifest_path.parent or not checkpoint.is_file():
        raise ValueError('recovery checkpoint missing or escapes manifest directory')
    if sha256(checkpoint)!=data.get('checkpoint_sha256'):
        raise ValueError('checkpoint digest mismatch')
    loaded = load_checkpoint(checkpoint,expected_profile='remote_full',expected_algorithm=algorithm,
                             expected_model_config=data['model_config'],map_location='cpu')
    if loaded.metadata.git_commit != data.get('source_git_commit'):
        raise ValueError('checkpoint source commit differs from manifest')
    if data.get('protocol_versions')!=CURRENT_PROTOCOL_VERSIONS:
        raise ValueError('recovery protocol versions mismatch')
    compatibility = loaded.training_state.get('compatibility')
    if compatibility!=data.get('compatibility'):
        raise ValueError('recovery compatibility fields differ')
    actual = loaded.training_state['engine']
    if loaded.metadata.update_count!=data.get('durable_updates') or actual['update_count']!=loaded.metadata.update_count:
        raise ValueError('recovery update counters disagree')
    if actual['collector']['total_token_steps']!=data.get('durable_token_steps'):
        raise ValueError('recovery token counters disagree')
    profile = _profile()
    settings = profile['config']['training']
    if compatibility['profile_sha256']!=profile['sha256'] or compatibility['envs']!=settings['rollout_envs'] or compatibility['token_steps_per_update']!=settings['token_steps_per_update']:
        raise ValueError('remote_full profile counts/hash changed')
    if actual['collector']['total_token_steps'] != loaded.metadata.update_count*settings['token_steps_per_update']:
        raise ValueError('checkpoint contains partial/unaccounted training update')
    return {**data,'checkpoint_path':str(checkpoint),'manifest_path':str(manifest_path)}


def training_plan(algorithm, recovery=None):
    profile = _profile()
    cfg = profile['config']['training']
    if algorithm not in profile['config']['algorithms']:
        raise ValueError('unknown algorithm')
    start = 0 if recovery is None else recovery['durable_updates']
    base, extra = cfg['updates_per_algorithm'], cfg['resume_updates']
    if type(start) is not int or start<0 or start>=base+extra:
        raise ValueError('checkpoint already completed target or has invalid counter')
    goal = base if start<base else base+extra
    return {'profile':profile,'algorithm':algorithm,'start_update':start,'goal_update':goal,
            'updates_this_session_if_time_allows':goal-start,'base_target':base,'resume_target':extra,
            'phase':'base' if recovery is None else ('base_continuation' if start<base else 'resume_verification'),
            'automatic_target_reduction':False}


def _output_directory(output_root: Path) -> Path:
    output_root = Path(output_root).resolve()
    kaggle_working = KAGGLE_WORKING.resolve()
    if not output_root.is_relative_to(kaggle_working) or output_root==kaggle_working:
        raise ValueError('execution output-root must be a named subdirectory of /kaggle/working')
    return output_root


def smoke_environment() -> dict:
    """Real A04 rule/environment smoke with a normal 108-card FIVE deal."""
    from ..environment import GuandanEnv,GameConfig
    from ..action_state import TokenCodec,COMMIT_TOKEN
    from ..cards import Rank
    from ..state import IllegalActionError
    env=GuandanEnv(GameConfig(level_rank=Rank.FIVE,seed=10001))
    obs=env.reset()
    hands=env.full_state()['round_state']['hands']
    if [len(hand) for hand in hands]!=[27]*4 or sorted(sum(hands,[]))!=list(range(108)):
        raise RuntimeError('physical card deal invalid')
    before=env.serialize()
    try:
        env.step(0)
    except IllegalActionError:
        pass
    else:
        raise RuntimeError('PAD was accepted')
    if env.serialize()!=before:
        raise RuntimeError('invalid action mutated state')
    env.step(TokenCodec.family('single'))
    env.step(TokenCodec.card(hands[0][0]))
    result=env.step(COMMIT_TOKEN)
    restored=GuandanEnv.deserialize(env.serialize())
    if restored.serialize()!=env.serialize() or not result.is_commit or result.committed_step!=1:
        raise RuntimeError('step/serialization smoke failed')
    return {'status':'passed','kind':'A04_real_environment_smoke','card_count':108,
            'token_steps':env.token_step,'committed_steps':env.committed_step,
            'level_rank':int(Rank.FIVE),'previous_result':None,'kaggle_verified':False}


def _verify_training_completion(result, plan: dict, algorithm: str, source: dict,
                                *, checkpoint_root: Path) -> dict:
    """Verify every complete result, including injected/local simulations.

    TrainingResult.updates and total_token_steps are the actual counts for this
    invocation; lifetime and durable counts include the recovered prefix.
    No status/counter claim substitutes for re-reading the durable checkpoint.
    """
    goal = plan["goal_update"]
    start = plan["start_update"]
    settings = plan["profile"]["config"]["training"]
    per_update = settings["token_steps_per_update"]
    actual_updates = goal - start
    actual_token_steps = actual_updates * per_update
    lifetime_steps = goal * per_update
    if result.algorithm != algorithm or result.profile != 'remote_full':
        raise ValueError("complete training has incompatible profile/algorithm")
    if plan['updates_this_session_if_time_allows'] != actual_updates:
        raise ValueError("training plan update counters disagree")
    expected = {
        "start_update": start,
        "target_updates": actual_updates,
        "updates": actual_updates,
        "total_token_steps": actual_token_steps,
        "lifetime_updates": goal,
        "lifetime_token_steps": lifetime_steps,
        "durable_updates": goal,
        "durable_token_steps": lifetime_steps,
        "rollout_envs": settings["rollout_envs"],
        "token_steps_per_update": per_update,
        "discarded_partial_steps": 0,
        "discarded_optimizer_steps": 0,
    }
    profile_counts_match = getattr(result, "profile_counts_match", None)
    if type(profile_counts_match) is not bool:
        raise ValueError("complete training profile_counts_match must be a bool")
    # This flag means the invocation itself matched one canonical whole-phase
    # target. A segmented continuation (99->100 or 101->102) is valid even
    # when the trainer correctly reports False; all actual/durable counters
    # below remain mandatory evidence.
    canonical_target = settings["updates_per_algorithm"] if start == 0 else settings["resume_updates"]
    if profile_counts_match and actual_updates != canonical_target:
        raise ValueError("profile_counts_match flag disagrees with continuation target")
    # If a result adapter also exposes explicitly named actual counters, these
    # must agree with TrainingResult's canonical fields, not override them.
    for name, value in (("actual_updates", actual_updates),
                        ("actual_token_steps", actual_token_steps)):
        if hasattr(result, name):
            expected[name] = value
    for name, value in expected.items():
        observed = getattr(result, name, None)
        if type(observed) is not int or observed != value:
            raise ValueError(f"complete training {name} counter disagrees with plan: {observed!r} != {value}")
    manifest_name = getattr(result, "recovery_manifest", None)
    if not manifest_name:
        raise ValueError("complete training did not produce a recovery manifest")
    manifest_path = Path(manifest_name).expanduser().resolve()
    artifact_root = Path(checkpoint_root).resolve()
    if not manifest_path.is_relative_to(artifact_root):
        raise ValueError("complete training recovery manifest escapes session checkpoint directory")
    if not manifest_path.is_file():
        raise ValueError("complete training recovery manifest is missing")
    # resolve_recovery checks the manifest SHA against file bytes, checkpoint
    # metadata/engine counters, model/protocol compatibility and source commit.
    resolved = resolve_recovery(manifest_path, algorithm=algorithm)
    for name, value in (("durable_updates", goal), ("durable_token_steps", lifetime_steps)):
        if type(resolved.get(name)) is not int or resolved[name] != value:
            raise ValueError(f"recovery manifest {name} does not reach the training target")
    if not source.get('git_commit') or resolved['source_git_commit'] != source['git_commit']:
        raise ValueError("completed checkpoint source commit differs from session source")
    checkpoint_name = getattr(result, "checkpoint", None)
    if not checkpoint_name:
        raise ValueError("complete training did not produce a checkpoint")
    checkpoint = Path(checkpoint_name).expanduser().resolve()
    if not checkpoint.is_relative_to(artifact_root):
        raise ValueError("complete training checkpoint escapes session checkpoint directory")
    if checkpoint != Path(resolved["checkpoint_path"]).resolve():
        raise ValueError("training result checkpoint differs from recovery manifest")
    if sha256(checkpoint) != resolved["checkpoint_sha256"]:
        raise ValueError("training result checkpoint digest mismatch")
    return resolved


def _verify_evaluation_checkpoint(loaded, *, label, update, algorithm, profile, source):
    """Verify persisted provenance and engine counters, not filenames/flags."""
    metadata = loaded.metadata
    if metadata.profile != 'remote_full' or metadata.algorithm != algorithm:
        raise ValueError(f'{label} checkpoint profile/algorithm differs from requested evaluation')
    if type(metadata.update_count) is not int or metadata.update_count != update:
        raise ValueError('remote evaluation requires update102 candidate and update100 frozen snapshot')
    if not source.get('git_commit') or metadata.git_commit != source['git_commit']:
        raise ValueError(f'{label} checkpoint source commit differs from current source')
    settings = profile['config']['training']
    state = loaded.training_state
    compatibility = state.get('compatibility', {})
    if compatibility.get('profile_sha256') != profile['sha256']:
        raise ValueError(f'{label} checkpoint profile hash differs from canonical profile')
    for field, expected in (('envs', settings['rollout_envs']),
                            ('token_steps_per_update', settings['token_steps_per_update'])):
        value = compatibility.get(field)
        if type(value) is not int or value != expected:
            raise ValueError(f'{label} checkpoint canonical compatibility {field} differs')
    engine = state.get('engine', {})
    if engine.get('algorithm') != algorithm:
        raise ValueError(f'{label} engine algorithm differs from checkpoint')
    if type(engine.get('update_count')) is not int or engine['update_count'] != update:
        raise ValueError(f'{label} engine update counter differs from checkpoint')
    collector = engine.get('collector', {})
    expected_steps = update * settings['token_steps_per_update']
    if type(collector.get('total_token_steps')) is not int or collector['total_token_steps'] != expected_steps:
        raise ValueError(f'{label} engine token counter differs from canonical completed updates')
    envs, ticks = collector.get('envs'), collector.get('ticks_per_env')
    count = settings['rollout_envs']
    if not isinstance(envs, (list, tuple)) or len(envs) != count:
        raise ValueError(f'{label} collector environment count differs from canonical profile')
    if (not isinstance(ticks, (list, tuple)) or len(ticks) != count
            or any(type(tick) is not int or tick != expected_steps // count for tick in ticks)
            or sum(ticks) != expected_steps or collector.get('incomplete') is not False):
        raise ValueError(f'{label} collector token counters are incomplete or inconsistent')
    return {'profile': metadata.profile, 'algorithm': metadata.algorithm,
            'source_git_commit': metadata.git_commit, 'updates': update,
            'engine_token_steps': expected_steps, 'rollout_envs': count}


def _verify_evaluation_result(report, profile):
    config = profile['config']['evaluation']
    opponents = config['opponents']
    rotations = config['seat_rotations']
    per_opponent = config['deal_groups_per_pairing'] * len(rotations)
    total = per_opponent * len(opponents)
    if report.get('profile') != 'remote_full' or report.get('status') != 'complete' or report.get('errors'):
        raise ValueError('evaluation did not return a complete remote_full result')
    for name, expected in (('target_games_per_opponent', per_opponent), ('total_games', total)):
        if type(report.get(name)) is not int or report[name] != expected:
            raise ValueError(f'evaluation {name} differs from canonical profile')
    counts = report.get('completed_games_per_opponent')
    if (not isinstance(counts, dict) or set(counts) != set(opponents)
            or any(type(value) is not int or value != per_opponent for value in counts.values())
            or sum(counts.values()) != total):
        raise ValueError('evaluation completed_games_per_opponent differs from total/profile')
    records = report.get('records')
    if not isinstance(records, (list, tuple)) or len(records) != total:
        raise ValueError('evaluation completed game records differ from reported total')
    observed = {opponent: 0 for opponent in opponents}
    seen = set()
    for row in records:
        if (not isinstance(row, dict) or row.get('done') is not True
                or row.get('terminal') is not True or row.get('truncated') is not False):
            raise ValueError('evaluation contains a non-completed game record')
        opponent, group, rotation = row.get('opponent'), row.get('deal_group'), row.get('seat_rotation')
        if (opponent not in opponents or type(group) is not int
                or not 0 <= group < config['deal_groups_per_pairing']
                or type(rotation) is not int or rotation not in rotations):
            raise ValueError('evaluation game identity is outside the canonical profile')
        key = (opponent, group, rotation)
        if key in seen:
            raise ValueError('evaluation contains duplicate completed game identities')
        seen.add(key)
        observed[opponent] += 1
    if observed != counts:
        raise ValueError('evaluation actual completed games disagree with reported counts')


def run(args, *, runtime_check=None, checkpoint_controller=None, working_artifacts_root=None) -> dict:
    """Fail-closed remote entry; only an explicit callback supplies a local seam.

    runtime_check permits simulated runtime facts and local paths, but never
    bypasses durable completion verification. The production CLI cannot supply
    this callback. plan remains configuration-only and never requires a GPU.
    A workflow may explicitly share its controller and its exact output-root
    scope. This never implies a new Kaggle session or resets the shared clock.
    """
    # Reject missing consent before runtime imports, checkpoint reads or writes.
    if args.action not in ('plan', 'system', 'smoke', 'train', 'evaluate'):
        raise ValueError('unknown session action')
    if args.action in ('smoke', 'train', 'evaluate') and not args.execute:
        raise ValueError('--execute is required for smoke/train/evaluate; no automatic launch')
    if args.profile != 'remote_full':
        raise ValueError('A07 remote entry only accepts remote_full')
    if runtime_check is not None and not callable(runtime_check):
        raise TypeError('runtime_check must be an explicitly injected callable')
    scope = _workflow_scope(args, working_artifacts_root)
    if runtime_check is None or scope is not None:
        _validate_production_paths(args, working_artifacts_root=scope)
    # Lazy imports keep --help and an unresumed plan independent of Torch.
    from scripts.kaggle_environment_check import inspect_environment, require_remote_runtime
    profile = _profile()
    system = None
    if args.action != 'plan':
        system = inspect_environment()
        (require_remote_runtime if runtime_check is None else runtime_check)(system)
    recovery = resolve_recovery(args.resume,algorithm=args.algorithm) if args.resume is not None else None
    plan = training_plan(args.algorithm,recovery)
    origin = _recovery_origin(args.resume, scope)
    recovery_evidence = {'recovery_origin': origin,
                         'new_session_resume': origin == 'mounted_input',
                         'genuine_new_kaggle_session_verified': False}
    if args.action=='plan':
        return {'status':'configuration_only','plan':plan,'upload':False,'executed':False,'accepted':False,'kaggle_verified':False,**recovery_evidence}
    if args.action=='system':
        return {'status':'environment_inspected','system':system,'executed_training':False,'accepted':False,'kaggle_verified':False,**recovery_evidence}
    # The local seam may monkeypatch _output_directory for a temporary fixture;
    # production still uses the same /kaggle/working boundary check.
    output = _output_directory(args.output_root)
    from .control import CheckpointController
    controller = (CheckpointController(args.session_hours,args.save_margin_seconds,args.checkpoint_seconds)
                  if checkpoint_controller is None else checkpoint_controller)
    source = source_provenance(ROOT)
    if not source.get('git_worktree_clean'):
        raise ValueError('remote execution requires clean committed/exported source')
    if recovery is not None and recovery['source_git_commit']!=source['git_commit']:
        raise ValueError('use the same exported source commit to resume this checkpoint')
    session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex
    directory = output / session_id
    directory.mkdir(parents=True, exist_ok=False)
    record = {'session_id':session_id,'action':args.action,'system':system,'source':source,
              'profile':profile,'plan':plan,'session_hours':controller.hard_seconds / 3600,
              'save_margin_seconds':controller.hard_seconds - controller.soft_seconds,
              'checkpoint_seconds':controller.interval,
              'checkpoint_controller_reused':checkpoint_controller is not None,
              'working_artifacts_root':None if scope is None else str(scope),
              'outputs':str(directory),'kaggle_verified':False,'accepted':False,
              **recovery_evidence,
              'limitations_document':'docs/prerequisite_defects_A05_A06.md',
              'origin':'executed_session; location indicators are not supervisory acceptance'}
    from ..training.runtime import write_evidence
    write_evidence(directory/'session_started.json',record)
    try:
        with canonical_profile(profile), controller.signal_handlers():
            controller.check(stage='session_start')
            record['smoke'] = smoke_environment()
            controller.check(stage='smoke_complete')
            if args.action=='smoke':
                record['status']='smoke_complete'
            elif args.action=='train':
                from ..training.trainer import train
                compatibility = {} if recovery is None else recovery['compatibility']
                options = {'algorithm':args.algorithm,'profile':'remote_full','device':'cuda',
                           'updates':plan['updates_this_session_if_time_allows'],
                           'checkpoint_dir':directory/'checkpoints', 'checkpoint_controller':controller,
                           'seed':compatibility.get('seed',10000 if args.algorithm=='ippo' else 60000),
                           'hidden_dim':32 if recovery is None else recovery['model_config']['hidden_dim'],
                           'epochs':compatibility.get('epochs',2), 'gamma':compatibility.get('gamma',1.0),
                           'gae_lambda':compatibility.get('gae_lambda',0.95),
                           'resume':None if recovery is None else recovery['checkpoint_path']}
                result = train(**options)
                record['training'] = asdict(result)
                complete = result.status == 'complete'
                if complete:
                    verified = _verify_training_completion(
                        result, plan, args.algorithm, source,
                        checkpoint_root=directory/'checkpoints')
                    record['durable_verification'] = verified
                elif result.status != 'incomplete':
                    raise ValueError(f"trainer returned unsuccessful status: {result.status!r}")
                record['status']='phase_complete' if complete else 'incomplete'
                record['recovery_manifest'] = result.recovery_manifest
            elif args.action=='evaluate':
                from ..evaluation import evaluate_profile
                if args.candidate_checkpoint is None or args.snapshot_checkpoint is None:
                    raise ValueError('remote evaluation requires explicit candidate and frozen snapshot checkpoint paths')
                from ..training.checkpoint import load_checkpoint
                candidate = load_checkpoint(args.candidate_checkpoint, expected_profile='remote_full',
                                            expected_algorithm=args.algorithm, map_location='cpu')
                frozen = load_checkpoint(args.snapshot_checkpoint, expected_profile='remote_full',
                                         expected_algorithm=args.algorithm, map_location='cpu')
                required_base = profile['config']['training']['updates_per_algorithm']
                required_resume = profile['config']['training']['resume_updates']
                record['evaluation_checkpoints'] = {
                    'candidate': _verify_evaluation_checkpoint(
                        candidate, label='candidate', update=required_base + required_resume,
                        algorithm=args.algorithm, profile=profile, source=source),
                    'snapshot': _verify_evaluation_checkpoint(
                        frozen, label='snapshot', update=required_base,
                        algorithm=args.algorithm, profile=profile, source=source),
                }
                result=evaluate_profile(profile='remote_full',candidate_checkpoint=args.candidate_checkpoint,
                                        snapshot_checkpoint=args.snapshot_checkpoint,
                                        evidence_path=directory/'evaluation'/'summary.json',deadline=controller,device='cuda')
                record['evaluation']=result.as_dict()
                _verify_evaluation_result(record['evaluation'], profile)
                record['status']='evaluation_complete'
        write_evidence(directory/'session_result.json',record)
        return record
    except BaseException as exc:
        from ..training.runtime import BudgetExceeded
        from ..evaluation import EvaluationIncomplete
        record['status']='incomplete' if isinstance(exc,(KeyboardInterrupt,BudgetExceeded,EvaluationIncomplete)) else 'failed'
        record['error']={'type':type(exc).__name__,'message':str(exc)}
        write_evidence(directory/'session_failure.json',record)
        raise


def parser():
    result=argparse.ArgumentParser(description='A07 remote_full Kaggle entry; never uploads or marks accepted')
    result.add_argument('--action',choices=('plan','system','smoke','train','evaluate'),default='plan')
    result.add_argument('--profile',choices=('remote_full',),default='remote_full')
    result.add_argument('--execute',action='store_true')
    result.add_argument('--algorithm',choices=('ippo','vrpo'),default='ippo')
    result.add_argument('--output-root',type=Path,default=Path('/kaggle/working/guandan'))
    result.add_argument('--resume',type=Path)
    result.add_argument('--session-hours',type=float,default=10.0)
    result.add_argument('--save-margin-seconds',type=float,default=300.0)
    result.add_argument('--checkpoint-seconds',type=float,default=600.0)
    result.add_argument('--candidate-checkpoint',type=Path)
    result.add_argument('--snapshot-checkpoint',type=Path)
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        result=run(args)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 2 if result['status']=='incomplete' else 0
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'error':str(exc),
                          'accepted':False,'kaggle_verified':False}),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
