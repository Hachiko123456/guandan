"""Synthetic orchestration/ZIP tests, never remote training or Kaggle evidence."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import stat
import zipfile

import pytest

from guandan.deployment import workflow as w
from guandan.deployment.control import CheckpointController


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fake_resolver(path, *, algorithm):
    row = json.loads(Path(path).read_text())
    checkpoint = Path(path).parent / row['checkpoint_filename']
    assert row['algorithm'] == algorithm
    assert w.digest(checkpoint) == row['checkpoint_sha256']
    return {**row, 'checkpoint_path': str(checkpoint)}


@pytest.fixture(autouse=True)
def patch_session_recovery(monkeypatch):
    # Synthetic workflow artifacts still go through session._verify_training_completion;
    # only its low-level checkpoint resolver is replaced by this deterministic fixture.
    monkeypatch.setattr(w.session, 'resolve_recovery', fake_resolver)


def synthetic_evaluation(profile, algorithm, candidate, snapshot):
    cfg = profile['config']['evaluation']
    records = []
    for op in cfg['opponents']:
        for group in range(cfg['deal_groups_per_pairing']):
            for rotation in cfg['seat_rotations']:
                records.append(dict(opponent=op, deal_group=group, seed=900000+group,
                    seat_rotation=rotation, policy_seat=rotation, candidate_team=rotation % 2,
                    winner_team=0, done=True, truncated=False, ranking=[1, 2, 3, 4],
                    rewards=[1, -1, 1, -1], team_rewards=[1, -1], token_steps=4,
                    committed_steps=1, base_deal_sha256=f'{group:064x}', terminal=True))
    target = cfg['deal_groups_per_pairing'] * len(cfg['seat_rotations'])
    return dict(status='complete', profile='remote_full', target_games_per_opponent=target, total_games=len(records), records=records,
        completed_games_per_opponent={op: target for op in cfg['opponents']},
        metadata={'candidate': {'sha256': w.digest(candidate), 'algorithm': algorithm, 'update': 102},
                  'snapshot_opponent': {'sha256': w.digest(snapshot), 'algorithm': algorithm, 'update': 100}})


def harness(tmp_path, *, incomplete_once=False, bad=False):
    cfg = w.session._profile()
    source = {'git_commit': 'a'*40}
    clock = [0.0]
    controller = CheckpointController(clock=lambda: clock[0])
    calls = []
    def executor(args, *, checkpoint_controller, working_artifacts_root):
        assert checkpoint_controller is controller
        assert Path(working_artifacts_root) == args.output_root
        assert args.execute and args.profile == 'remote_full'
        calls.append((args.action, args.algorithm))
        clock[0] += 1
        out = args.output_root / f'run-{len(calls):04d}'
        out.mkdir(parents=True)
        record = {'outputs': str(out), 'status': 'smoke_complete', 'action': args.action, 'algorithm': args.algorithm}
        if args.action == 'train':
            from guandan.training.trainer import TrainingResult
            start = 0 if args.resume is None else fake_resolver(args.resume, algorithm=args.algorithm)['durable_updates']
            count = 100 if start < 100 else 102
            incomplete = incomplete_once and len(calls) == 2
            durable = 7 if incomplete else count
            checkpoint = out / 'checkpoints' / f'{args.algorithm}-{durable}.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(f'SYNTHETIC ONLY {args.algorithm} {durable}'.encode())
            row = {'format': 'test_synthetic_not_remote_evidence', 'algorithm': args.algorithm,
                   'profile': 'remote_full', 'source_git_commit': source['git_commit'],
                   'durable_updates': durable, 'durable_token_steps': durable*1024,
                   'checkpoint_filename': checkpoint.name, 'checkpoint_sha256': w.digest(checkpoint)}
            write(checkpoint.with_suffix('.recovery.json'), row)
            additional = max(0, durable - start)
            result = TrainingResult(
                algorithm=args.algorithm, profile='remote_full', updates=additional,
                total_token_steps=additional*1024, last_loss=0.25,
                checkpoint=str(checkpoint), status='incomplete' if incomplete else 'complete',
                lifetime_updates=durable, lifetime_token_steps=durable*1024,
                start_update=start, target_updates=additional, rollout_envs=8,
                token_steps_per_update=1024,
                profile_counts_match=(start in (0, 100) and not incomplete),
                elapsed_seconds=1.0, metrics=[], checkpoints=[],
                resumed_from=None if args.resume is None else str(
                    Path(args.resume).parent / json.loads(Path(args.resume).read_text())['checkpoint_filename']),
                resume_sha256=None if args.resume is None else fake_resolver(
                    args.resume, algorithm=args.algorithm)['checkpoint_sha256'],
                runtime={}, reason='synthetic incomplete' if incomplete else None,
                evidence=None, durable_updates=durable,
                durable_token_steps=durable*1024, discarded_partial_steps=0,
                discarded_optimizer_steps=0,
                recovery_manifest=str(checkpoint.with_suffix('.recovery.json')),
            )
            record['training'] = asdict(result)
            record['recovery_manifest'] = result.recovery_manifest
            record['status'] = 'phase_complete' if not incomplete else 'incomplete'
        elif args.action == 'evaluate':
            result = synthetic_evaluation(cfg, args.algorithm, args.candidate_checkpoint, args.snapshot_checkpoint)
            if bad:
                result['total_games'] -= 1
            record.update(status='evaluation_complete', action='evaluate', evaluation=result)
            write(out / 'evaluation' / 'summary.json', result)
        write(out / 'session_result.json', record)
        return record
    return cfg, source, controller, calls, clock, executor



def test_real_trainer_nested_checkpoint_layout_maps_to_stage(tmp_path):
    sessions = tmp_path / 'sessions'
    manifest = sessions / 'run-1' / 'checkpoints' / 'remote_full' / 'ippo' / 'ippo_update_000100_x.recovery.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}', encoding='utf-8')
    assert w._stage_directory_from_manifest(manifest, sessions) == sessions / 'run-1'


def test_flat_synthetic_checkpoint_layout_still_maps_to_stage(tmp_path):
    sessions = tmp_path / 'sessions'
    manifest = sessions / 'run-1' / 'checkpoints' / 'ippo_update_000100_x.recovery.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}', encoding='utf-8')
    assert w._stage_directory_from_manifest(manifest, sessions) == sessions / 'run-1'

def test_run_all_sequences_both_algorithms_with_one_controller_and_real_artifact_paths(tmp_path):
    profile, source, controller, calls, clock, executor = harness(tmp_path)
    flow = w.Workflow(tmp_path/'flow', profile, source, controller, executor=executor, resolver=fake_resolver)
    result = flow.run()
    assert calls == [('smoke','ippo'), ('train','ippo'), ('train','ippo'), ('train','vrpo'),
                     ('train','vrpo'), ('evaluate','ippo'), ('evaluate','vrpo')]
    assert result['status'] == 'pipeline_complete_pending_remote_review'
    assert result['accepted'] is result['kaggle_verified'] is result['genuine_new_kaggle_session_verified'] is False
    assert result['elapsed_seconds'] == 7
    assert flow.archive.is_file()
    for algorithm in ('ippo','vrpo'):
        rows = flow.recoveries(algorithm)
        assert set(rows) == {100, 102}
        assert rows[100]['checkpoint_path'] != rows[102]['checkpoint_path']
        path = w.resolve_reference(flow.root, result['evaluations'][algorithm])
        assert json.loads(path.read_text())['total_games'] == 3000


def test_incomplete_stops_without_running_next_algorithm_and_keeps_durable_bundle(tmp_path):
    profile, source, controller, calls, clock, executor = harness(tmp_path, incomplete_once=True)
    flow = w.Workflow(tmp_path/'flow', profile, source, controller, executor=executor, resolver=fake_resolver)
    result = flow.run()
    assert result['status'] == 'incomplete'
    assert calls == [('smoke','ippo'), ('train','ippo')]
    assert set(flow.recoveries('ippo')) == {7}
    assert flow.archive.is_file()
    assert not result['evaluations']


def test_relocated_bundle_recovers_without_old_absolute_paths_or_repeating_finished_training(tmp_path):
    profile, source, controller, calls, clock, executor = harness(tmp_path)
    old = w.Workflow(tmp_path/'old', profile, source, controller, executor=executor, resolver=fake_resolver)
    old.run()
    destination = tmp_path/'new'
    w.restore_progress(old.archive, destination, commit=source['git_commit'], profile_sha256=profile['sha256'])
    # Remove access to old path by renaming, not by testing against retained originals.
    old.root.rename(tmp_path/'old-unavailable')
    new_calls=[]
    def new_executor(args, **kwargs):
        assert args.action == 'smoke', 'verified stages must not be retrained'
        new_calls.append(args.action)
        out = args.output_root / 'new-smoke'
        out.mkdir()
        record={'outputs':str(out),'status':'smoke_complete'}
        write(out/'session_result.json',record)
        return record
    new = w.Workflow(destination, profile, source, CheckpointController(), executor=new_executor, resolver=fake_resolver)
    result = new.run()
    assert new_calls == ['smoke']
    assert result['status'] == 'pipeline_complete_pending_remote_review'
    assert result['previous_run_id'] == old.state['run_id']
    assert result['genuine_new_kaggle_session_verified'] is False


def test_deadline_is_not_reset_per_stage(tmp_path):
    profile, source, controller, calls, clock, executor = harness(tmp_path)
    flow = w.Workflow(tmp_path/'flow', profile, source, controller, executor=executor, resolver=fake_resolver)
    clock[0] = controller.soft_seconds
    result = flow.run()
    assert result['status'] == 'incomplete' and calls == []
    assert flow.archive.is_file()


def test_false_evaluation_total_explicitly_fails_not_completed(tmp_path):
    profile, source, controller, calls, clock, executor = harness(tmp_path, bad=True)
    flow = w.Workflow(tmp_path/'flow', profile, source, controller, executor=executor, resolver=fake_resolver)
    with pytest.raises(ValueError, match='total_games'):
        flow.run()
    assert flow.state['status'] == 'failed'
    assert 'vrpo' not in flow.state['evaluations']
    assert flow.archive.is_file()


@pytest.mark.parametrize('name', ['../escape','/absolute','C:/escape','a\\b','a/../b','./x'])
def test_progress_paths_reject_escape(name):
    with pytest.raises(ValueError):
        w.safe_relative(name)


def test_bad_source_bundle_and_tamper_rejected_before_destination(tmp_path):
    profile, source, controller, calls, clock, executor = harness(tmp_path)
    flow = w.Workflow(tmp_path/'flow', profile, source, controller, executor=executor, resolver=fake_resolver)
    w.export_progress(flow.root, flow.archive)
    with pytest.raises(ValueError, match='source/profile'):
        w.restore_progress(flow.archive, tmp_path/'new', commit='b'*40, profile_sha256=profile['sha256'])
    assert not (tmp_path/'new').exists()
    forged=tmp_path/'forged.zip'
    with zipfile.ZipFile(flow.archive) as original, zipfile.ZipFile(forged,'w') as target:
        for info in original.infolist():
            target.writestr(info, b'x'*info.file_size if info.filename==w.STATE else original.read(info))
    with pytest.raises(ValueError, match='SHA256'):
        w.restore_progress(forged,tmp_path/'new',commit=source['git_commit'],profile_sha256=profile['sha256'])
    assert not (tmp_path/'new').exists()


def test_progress_zip_rejects_link_member(tmp_path):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as archive:
        info=zipfile.ZipInfo('linked')
        info.external_attr=(stat.S_IFLNK|0o777)<<16
        archive.writestr(info,'outside')
    with pytest.raises(ValueError, match='linked/special'):
        w.restore_progress(path,tmp_path/'new',commit='a'*40,profile_sha256='b'*64)
    assert not (tmp_path/'new').exists()


def test_no_time_limit_controller_only_stops_on_explicit_signal():
    from guandan.deployment.control import CheckpointController, SessionStop
    clock = [0.0]
    controller = CheckpointController(None, save_margin_seconds=300, checkpoint_seconds=10,
                                      no_time_limit=True, clock=lambda: clock[0])
    clock[0] = 10_000_000
    controller.check()
    controller.before_update()
    controller.after_update(1.0)
    controller.request_stop('explicit')
    with pytest.raises(SessionStop, match='explicit'):
        controller.check()


def test_plan_never_executes_remote_runtime_or_writes(tmp_path, monkeypatch, capsys):
    def prohibited(*a,**kw):
        raise AssertionError('runtime should not be checked for plan')
    monkeypatch.setattr('scripts.kaggle_environment_check.require_remote_runtime', prohibited)
    assert w.main(['--output-root', str(tmp_path/'no-write')]) == 0
    assert not (tmp_path/'no-write').exists()
    assert json.loads(capsys.readouterr().out)['status'] == 'plan_only'


def test_expanded_dataset_bundle_restores_and_preserves_input(tmp_path):
    profile, source, controller, calls, clock, executor = harness(tmp_path)
    flow = w.Workflow(tmp_path/'flow', profile, source, controller, executor=executor, resolver=fake_resolver)
    w.export_progress(flow.root, flow.archive)
    expanded = tmp_path/'input'/'saved-output'
    with zipfile.ZipFile(flow.archive) as bundle:
        for info in bundle.infolist():
            path=expanded/info.filename
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(bundle.read(info))
    before={p.relative_to(expanded).as_posix():w.digest(p) for p in expanded.rglob('*') if p.is_file()}
    dest=tmp_path/'restored'
    w.restore_progress(expanded,dest,commit=source['git_commit'],profile_sha256=profile['sha256'])
    assert (dest/w.STATE).read_bytes()==(expanded/w.STATE).read_bytes()
    assert before=={p.relative_to(expanded).as_posix():w.digest(p) for p in expanded.rglob('*') if p.is_file()}


def test_recorded_failure_not_silently_retried(tmp_path):
    profile, source, controller, calls, clock, executor = harness(tmp_path, bad=True)
    flow=w.Workflow(tmp_path/'flow',profile,source,controller,executor=executor,resolver=fake_resolver)
    with pytest.raises(ValueError):
        flow.run()
    before=(flow.root/w.STATE).read_bytes()
    with pytest.raises(ValueError,match='--retry-failed'):
        w.Workflow(flow.root,profile,source,controller,executor=executor,resolver=fake_resolver)
    assert (flow.root/w.STATE).read_bytes()==before


def test_generator_renders_one_notebook_with_shared_deadline_and_no_per_algorithm_modes():
    from scripts.prepare_kaggle_unified import render_notebook, ROOT
    import ast
    metadata={'source_git_commit':'a'*40,'source_zip_sha256':'b'*64,
              'source_manifest_sha256':'c'*64,'checker_sha256':'d'*64}
    template=(ROOT/'notebooks/kaggle_unified_bootstrap.py').read_text(encoding='utf-8-sig')
    doc=render_notebook(metadata,template)
    assert doc['nbformat']==4 and len(doc['cells'])==5
    code='\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code')
    ast.parse(code)
    assert 'guandan.deployment.workflow' in code
    assert 'remaining_seconds/3600' in code
    assert 'runtime_check=' not in code
    assert '__ZIP_SHA256__' not in code
    assert 'MODE =' not in code
    assert all(c['execution_count'] is None and c['outputs']==[] for c in doc['cells'] if c['cell_type']=='code')


class patch_export:
    def __init__(self, function):
        self.function = function
    def __enter__(self):
        self.patch = pytest.MonkeyPatch()
        self.patch.setattr(w, 'export_progress', self.function)
        return self
    def __exit__(self, exc_type, exc, tb):
        self.patch.undo()
        return False



def test_failed_training_gate_requires_explicit_retry_and_retries_from_parent(tmp_path):
    profile, source, controller, calls, clock, base_executor = harness(tmp_path)
    failed_once = [False]

    def executor(args, **kwargs):
        record = base_executor(args, **kwargs)
        if args.action == 'train' and args.algorithm == 'ippo' and args.resume is not None and not failed_once[0]:
            failed_once[0] = True
            path = Path(record['outputs']) / 'session_result.json'
            payload = json.loads(path.read_text())
            payload['training']['updates'] = 0
            path.write_text(json.dumps(payload), encoding='utf-8')
            record['training']['updates'] = 0
        return record

    flow = w.Workflow(tmp_path/'flow', profile, source, controller,
                      executor=executor, resolver=fake_resolver)
    with pytest.raises(ValueError, match='training completion'):
        flow.run()
    assert flow.state['status'] == 'failed'
    before = len(calls)
    retried = w.Workflow(flow.root, profile, source, controller,
                         executor=executor, resolver=fake_resolver, retry_failed=True)
    result = retried.run()
    assert result['status'] == 'pipeline_complete_pending_remote_review'
    assert ('train', 'ippo') in calls[before:]


def test_cached_evaluation_with_nonempty_errors_is_rejected(tmp_path):
    profile = w.session._profile()
    candidate = tmp_path / 'candidate.pt'
    snapshot = tmp_path / 'snapshot.pt'
    candidate.write_bytes(b'candidate')
    snapshot.write_bytes(b'snapshot')
    report = synthetic_evaluation(profile, 'ippo', candidate, snapshot)
    report['errors'] = ['synthetic postcondition error']
    with pytest.raises(ValueError, match='complete remote_full'):
        w.checked_evaluation(report, profile, 'ippo', candidate, snapshot)


def test_parent_chain_rejects_candidate_from_unrelated_update100_branch(tmp_path):
    profile, source, controller, calls, clock, executor = harness(tmp_path)
    flow = w.Workflow(tmp_path/'flow', profile, source, controller,
                      executor=executor, resolver=fake_resolver)
    parent = {'durable_updates': 100, 'checkpoint_sha256': 'a' * 64,
              'stage_input': None, 'start_update': 0}
    unrelated_candidate = {'durable_updates': 102, 'checkpoint_sha256': 'c' * 64,
                           'stage_input': {'durable_updates': 100, 'checkpoint_sha256': 'b' * 64},
                           'start_update': 100}
    by_count = {100: [parent], 102: [unrelated_candidate]}
    assert flow._lineage_verified(parent, by_count, {}, set()) is True
    assert flow._lineage_verified(unrelated_candidate, by_count, {}, set()) is False


def test_export_over_hard_deadline_preserves_previous_zip(tmp_path):
    root = tmp_path / 'flow'
    root.mkdir()
    write(root / w.STATE, {'source_git_commit': 'a' * 40, 'profile_sha256': 'b' * 64})
    archive = tmp_path / 'flow-progress.zip'
    w.export_progress(root, archive)
    previous = archive.read_bytes()
    clock = [0.0]
    controller = CheckpointController(session_hours=1, save_margin_seconds=60,
                                       checkpoint_seconds=10, clock=lambda: clock[0])
    clock[0] = controller.hard_seconds + 1
    with pytest.raises(w.SessionStop, match='hard deadline'):
        w.export_progress(root, archive, controller=controller)
    assert archive.read_bytes() == previous
    assert not archive.with_suffix('.partial.zip').exists()


def test_workflow_export_over_hard_deadline_is_incomplete_with_external_receipt(tmp_path):
    profile, source, controller, calls, clock, base_executor = harness(tmp_path)
    flow = w.Workflow(tmp_path/'flow', profile, source, controller,
                      executor=base_executor, resolver=fake_resolver)
    export_calls = [0]

    def export(root, archive, *, controller=None):
        export_calls[0] += 1
        if export_calls[0] == 7:  # final export after six phase exports
            controller.request_stop('synthetic hard deadline')
            raise w.SessionStop('progress export exceeded hard deadline')
        return {'path': str(archive), 'sha256': 'a' * 64, 'files': 1, 'uploaded': False,
                'export_seconds': 1.0, 'total_elapsed_seconds': 6.0}

    with patch_export(export):
        result = flow.run()
    assert result['status'] == 'incomplete'
    assert result['progress_export_over_budget'] is True
    receipt = Path(result['export_receipt'])
    assert receipt.is_file()
    receipt_data = json.loads(receipt.read_text())
    assert receipt_data['status'] == 'incomplete'
    assert receipt_data['total_elapsed_seconds'] >= receipt_data['export_seconds']
