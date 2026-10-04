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


def _kaggle_production_context() -> bool:
    root = ROOT.resolve()
    return (KAGGLE_INPUT.is_dir() and KAGGLE_WORKING.is_dir()
            and (root.is_relative_to(KAGGLE_INPUT.resolve())
                 or root.is_relative_to(KAGGLE_WORKING.resolve())))


def _require_input_path(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = KAGGLE_INPUT.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"{label} must be under /kaggle/input")
    return resolved


def _validate_production_paths(args) -> None:
    _output_directory(args.output_root)
    for value, label in (
        (args.resume, "resume manifest"),
        (args.candidate_checkpoint, "candidate checkpoint"),
        (args.snapshot_checkpoint, "snapshot checkpoint"),
    ):
        if value is not None:
            _require_input_path(value, label)


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
    kaggle_working = Path('/kaggle/working').resolve()
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


def _verify_training_completion(result, plan: dict, algorithm: str) -> dict:
    """Re-read the durable artifact before claiming a production phase complete."""
    goal = plan["goal_update"]
    settings = plan["profile"]["config"]["training"]
    expected_steps = goal * settings["token_steps_per_update"]
    if getattr(result, "durable_updates", None) != goal:
        raise ValueError("complete training has incorrect durable update counter")
    if getattr(result, "durable_token_steps", None) != expected_steps:
        raise ValueError("complete training has incorrect durable token counter")
    manifest_name = getattr(result, "recovery_manifest", None)
    if not manifest_name:
        raise ValueError("complete training did not produce a recovery manifest")
    manifest_path = Path(manifest_name).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError("complete training recovery manifest is missing")
    resolved = resolve_recovery(manifest_path, algorithm=algorithm)
    if resolved["durable_updates"] != goal or resolved["durable_token_steps"] != expected_steps:
        raise ValueError("recovery manifest durable counters do not reach the training target")
    checkpoint_name = getattr(result, "checkpoint", None)
    if not checkpoint_name:
        raise ValueError("complete training did not produce a checkpoint")
    checkpoint = Path(checkpoint_name).expanduser().resolve()
    if checkpoint != Path(resolved["checkpoint_path"]).resolve():
        raise ValueError("training result checkpoint differs from recovery manifest")
    if sha256(checkpoint) != resolved["checkpoint_sha256"]:
        raise ValueError("training result checkpoint digest mismatch")
    return resolved


def run(args, *, runtime_check=None, production=False) -> dict:
    # Lazy imports keep --help and plan usable without torch/dependency installation.
    from scripts.kaggle_environment_check import inspect_environment, require_remote_runtime
    production = bool(production or _kaggle_production_context())
    profile = _profile()
    if args.profile != 'remote_full':
        raise ValueError('A07 remote entry only accepts remote_full')
    if production:
        _validate_production_paths(args)
    recovery = resolve_recovery(args.resume,algorithm=args.algorithm) if args.resume is not None else None
    plan = training_plan(args.algorithm,recovery)
    if args.action=='plan':
        return {'status':'configuration_only','plan':plan,'upload':False,'executed':False,'accepted':False,'kaggle_verified':False}
    system = inspect_environment()
    if args.action=='system':
        # A direct local run remains an observation-only diagnostic. The real
        # CLI enters production mode in a Kaggle input/working layout; tests
        # may explicitly inject runtime_check as the sole local seam.
        if production or runtime_check is not None:
            (runtime_check or require_remote_runtime)(system)
        return {'status':'environment_inspected','system':system,'executed_training':False,'accepted':False,'kaggle_verified':False}
    if not args.execute:
        raise ValueError('--execute is required for smoke/train/evaluate; no automatic launch')
    # Test-only seam used with fake runtime; CLI never exposes a bypass flag.
    (runtime_check or require_remote_runtime)(system)
    output = _output_directory(args.output_root)
    from .control import CheckpointController
    controller = CheckpointController(args.session_hours,args.save_margin_seconds,args.checkpoint_seconds)
    source = source_provenance(ROOT)
    if not source.get('git_worktree_clean'):
        raise ValueError('remote execution requires clean committed/exported source')
    if recovery is not None and recovery['source_git_commit']!=source['git_commit']:
        raise ValueError('use the same exported source commit to resume this checkpoint')
    session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex
    directory = output / session_id
    directory.mkdir(parents=True, exist_ok=False)
    record = {'session_id':session_id,'action':args.action,'system':system,'source':source,
              'profile':profile,'plan':plan,'session_hours':args.session_hours,
              'save_margin_seconds':args.save_margin_seconds,'checkpoint_seconds':args.checkpoint_seconds,
              'outputs':str(directory),'kaggle_verified':False,'accepted':False,
              'limitations_document':'docs/prerequisite_defects_A05_A06.md',
              'origin':'executed_session; location indicators are not supervisory acceptance'}
    from ..training.runtime import write_evidence
    write_evidence(directory/'session_started.json',record)
    try:
        with canonical_profile(profile), controller.signal_handlers():
            record['smoke'] = smoke_environment()
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
                complete = result.status == 'complete' and result.lifetime_updates == plan['goal_update']
                if complete and production:
                    verified = _verify_training_completion(result, plan, args.algorithm)
                    record['durable_verification'] = verified
                record['status']='phase_complete' if complete else 'incomplete'
                record['recovery_manifest'] = result.recovery_manifest
                record['new_session_resume'] = args.resume is not None
            elif args.action=='evaluate':
                from ..evaluation import evaluate_profile
                if args.candidate_checkpoint is None or args.snapshot_checkpoint is None:
                    raise ValueError('remote evaluation requires explicit candidate and frozen snapshot checkpoint paths')
                from ..training.checkpoint import load_checkpoint
                candidate = load_checkpoint(args.candidate_checkpoint,expected_profile='remote_full')
                frozen = load_checkpoint(args.snapshot_checkpoint,expected_profile='remote_full')
                if candidate.metadata.algorithm != frozen.metadata.algorithm:
                    raise ValueError('candidate and frozen snapshot must use the same algorithm')
                required_base = profile['config']['training']['updates_per_algorithm']
                required_resume = profile['config']['training']['resume_updates']
                if candidate.metadata.update_count != required_base+required_resume or frozen.metadata.update_count != required_base:
                    raise ValueError('remote evaluation requires update102 candidate and update100 frozen snapshot')
                result=evaluate_profile(profile='remote_full',candidate_checkpoint=args.candidate_checkpoint,
                                        snapshot_checkpoint=args.snapshot_checkpoint,
                                        evidence_path=directory/'evaluation'/'summary.json',deadline=controller,device='cuda')
                record['evaluation']=result.as_dict()
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
        result=run(args, production=_kaggle_production_context())
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 2 if result['status']=='incomplete' else 0
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'error':str(exc),
                          'accepted':False,'kaggle_verified':False}),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
