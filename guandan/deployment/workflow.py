"""Single-notebook A07 workflow. No uploads and no automatic acceptance.

Uses the existing session/trainer/evaluator, a single cooperative controller, and
relative artifact references. A same-process reload is NOT a new Kaggle session.
Incomplete evaluations are preserved but restarted (no partial-game aggregation).
"""
from __future__ import annotations

import argparse
from dataclasses import MISSING, fields as dataclass_fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import time
import uuid
import zipfile

from . import session
from .control import CheckpointController, SessionStop
from .provenance import source_provenance

STATE = 'workflow.json'
BUNDLE = 'WORKFLOW_BUNDLE.json'
STAGE = 'WORKFLOW_STAGE.json'
RECEIPT_SUFFIX = '.receipt.json'
MAX_BUNDLE_BYTES = 16 * 1024**3
MAX_MEMBERS = 50000


def _hard_check(controller):
    if controller is None:
        return
    if controller.clock() >= controller.hard_deadline:
        controller.request_stop('progress_export_hard_deadline')
        raise SessionStop('progress export exceeded hard deadline')


def digest(path, *, controller=None):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            _hard_check(controller)
            value.update(chunk)
    _hard_check(controller)
    return value.hexdigest()


def safe_relative(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('unsafe bundle path')
    path = PurePosixPath(name)
    if path.is_absolute() or str(path) != name or '..' in path.parts or name == '.':
        raise ValueError('unsafe bundle path')
    return name


def linked(path):
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reference(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    return {'path': path.relative_to(root).as_posix(), 'sha256': digest(path)}


def resolve_reference(root, ref):
    root = Path(root).resolve()
    path = root / safe_relative(ref['path'])
    if (linked(path) or not path.resolve().is_relative_to(root)
            or not path.is_file() or digest(path) != ref['sha256']):
        raise ValueError('workflow artifact digest/path mismatch')
    return path.resolve()


def export_progress(root, archive, *, controller=None):
    """Create an atomic local download ZIP; never upload.

    Only the hard deadline interrupts export: a soft deadline must still allow
    the already-durable progress bundle to be written.  A failed export leaves
    the last complete archive untouched.
    """
    started = controller.clock() if controller is not None else time.monotonic()
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if archive.is_relative_to(root):
        raise ValueError('progress ZIP must be outside its source tree')
    _hard_check(controller)
    state = json.loads((root / STATE).read_text(encoding='utf-8'))
    files = {}
    total = 0
    temporary = archive.with_suffix('.partial.zip')
    try:
        for path in sorted(root.rglob('*')):
            _hard_check(controller)
            if linked(path):
                raise ValueError('linked progress entry')
            if path.is_dir():
                continue
            if not path.is_file() or path.suffix == '.tmp':
                raise ValueError('incomplete or non-regular progress entry')
            name = safe_relative(path.relative_to(root).as_posix())
            if name in {BUNDLE, STAGE} and name == BUNDLE:
                raise ValueError('reserved progress filename')
            total += path.stat().st_size
            if total > MAX_BUNDLE_BYTES or len(files) >= MAX_MEMBERS:
                raise ValueError('progress bundle exceeds size limit')
            files[name] = {'sha256': digest(path, controller=controller), 'bytes': path.stat().st_size}
        manifest = {'format': 'guandan-workflow-bundle-v1', 'source_git_commit': state['source_git_commit'],
                    'profile_sha256': state['profile_sha256'], 'files': files,
                    'accepted': False, 'kaggle_verified': False}
        archive.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            temporary.unlink()
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
            _hard_check(controller)
            bundle.writestr(BUNDLE, json.dumps(manifest, sort_keys=True))
            for name in files:
                _hard_check(controller)
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_STORED
                with bundle.open(info, 'w') as target, (root / name).open('rb') as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        _hard_check(controller)
                        target.write(chunk)
        _hard_check(controller)
        archive_sha256 = digest(temporary, controller=controller)
        _hard_check(controller)
        os.replace(temporary, archive)
        finished = controller.clock() if controller is not None else time.monotonic()
        return {'path': str(archive), 'sha256': archive_sha256, 'files': len(files), 'uploaded': False,
                'export_seconds': float(finished - started),
                'total_elapsed_seconds': float((finished - controller.started) if controller is not None else finished - started)}
    except BaseException:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
        raise


def export_receipt_path(archive):
    return Path(archive).resolve().with_suffix(RECEIPT_SUFFIX)


def write_export_receipt(archive, *, status, export_seconds, total_elapsed_seconds,
                         error=None, bundle=None):
    archive = Path(archive).resolve()
    payload = {'format': 'guandan-workflow-export-receipt-v1', 'status': status,
               'archive_path': str(archive),
               'archive_sha256': (bundle.get('sha256') if bundle is not None else None),
               'export_seconds': float(export_seconds),
               'total_elapsed_seconds': float(total_elapsed_seconds),
               'uploaded': False, 'accepted': False, 'kaggle_verified': False}
    if bundle is not None:
        payload['bundle'] = {key: value for key, value in bundle.items() if key != 'path'}
    if error is not None:
        payload['error'] = {'type': type(error).__name__, 'message': str(error)}
    receipt = export_receipt_path(archive)
    atomic_json(receipt, payload)
    return receipt


def restore_progress(archive, destination, *, commit, profile_sha256):
    """Restore only into a new directory. Check every member before final rename."""
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('restore requires a new destination; never overwrite local progress')
    if archive.is_dir():
        return restore_expanded_progress(archive, destination, commit=commit, profile_sha256=profile_sha256)
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > MAX_MEMBERS + 1 or sum(i.file_size for i in infos) > MAX_BUNDLE_BYTES:
            raise ValueError('progress bundle exceeds size limit')
        seen = set()
        for info in infos:
            name = safe_relative(info.filename)
            if name.casefold() in seen or info.is_dir() or info.flag_bits & 1:
                raise ValueError('duplicate/directory/encrypted progress entry')
            seen.add(name.casefold())
            if stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG):
                raise ValueError('linked/special progress entry')
        if BUNDLE not in bundle.namelist() or bundle.getinfo(BUNDLE).file_size > 16 * 1024**2:
            raise ValueError('missing/oversized bundle manifest')
        manifest = json.loads(bundle.read(BUNDLE))
        if (manifest.get('format') != 'guandan-workflow-bundle-v1'
                or manifest.get('source_git_commit') != commit
                or manifest.get('profile_sha256') != profile_sha256
                or manifest.get('accepted') is not False or manifest.get('kaggle_verified') is not False):
            raise ValueError('progress source/profile/format mismatch; reuse the SAME source ZIP')
        files = manifest['files']
        if not isinstance(files, dict) or STATE not in files or set(files) | {BUNDLE} != set(bundle.namelist()):
            raise ValueError('progress file set mismatch')
        destination.parent.mkdir(parents=True, exist_ok=True)
        # TemporaryDirectory is always a fresh child of the selected working parent.
        with tempfile.TemporaryDirectory(prefix='guandan-restore-', dir=destination.parent) as temp:
            stage = Path(temp) / 'verified'
            stage.mkdir()
            for name, expected in files.items():
                target = stage / safe_relative(name)
                if expected['bytes'] != bundle.getinfo(name).file_size:
                    raise ValueError('progress size mismatch')
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as src, target.open('xb') as dst:
                    for chunk in iter(lambda: src.read(1024 * 1024), b''):
                        dst.write(chunk)
                if digest(target) != expected['sha256']:
                    raise ValueError('progress SHA256 mismatch')
            state = json.loads((stage / STATE).read_text(encoding='utf-8'))
            if state['source_git_commit'] != commit or state['profile_sha256'] != profile_sha256:
                raise ValueError('workflow state and bundle disagree')
            stage.rename(destination)
    return manifest



def restore_expanded_progress(source, destination, *, commit, profile_sha256):
    """Support a Dataset that exposes the contents of the downloaded progress ZIP."""
    manifest_path = source / BUNDLE
    if not manifest_path.is_file() or manifest_path.stat().st_size > 16 * 1024**2:
        raise ValueError('missing/oversized expanded progress manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('format') != 'guandan-workflow-bundle-v1'
            or manifest.get('source_git_commit') != commit or manifest.get('profile_sha256') != profile_sha256
            or manifest.get('accepted') is not False or manifest.get('kaggle_verified') is not False):
        raise ValueError('progress source/profile/format mismatch')
    files = manifest['files']
    if not isinstance(files, dict) or STATE not in files or len(files) > MAX_MEMBERS:
        raise ValueError('invalid expanded progress file set')
    actual, total = set(), 0
    for path in source.rglob('*'):
        if linked(path) or not path.resolve().is_relative_to(source):
            raise ValueError('linked/escaping expanded progress entry')
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('non-regular expanded progress entry')
        actual.add(safe_relative(path.relative_to(source).as_posix()))
        total += path.stat().st_size
        if total > MAX_BUNDLE_BYTES or len(actual) > MAX_MEMBERS + 1:
            raise ValueError('progress bundle exceeds size limit')
    if actual != set(files) | {BUNDLE}:
        raise ValueError('expanded progress file set mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='guandan-restore-', dir=destination.parent) as temp:
        stage = Path(temp)/'verified'
        stage.mkdir()
        for name, expected in files.items():
            src, target = source / safe_relative(name), stage / name
            if src.stat().st_size != expected['bytes'] or digest(src) != expected['sha256']:
                raise ValueError('expanded progress SHA256/size mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            with src.open('rb') as reader, target.open('xb') as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b''):
                    writer.write(chunk)
            if digest(target) != expected['sha256']:
                raise ValueError('progress changed during copy')
        state = json.loads((stage/STATE).read_text(encoding='utf-8'))
        if state['source_git_commit'] != commit or state['profile_sha256'] != profile_sha256:
            raise ValueError('workflow state and bundle disagree')
        stage.rename(destination)
    return manifest



def equivalent_checkpoints(first, second):
    """Duplicate saves may have different ZIP bytes/timestamps; compare actual state."""
    from ..training.checkpoint import load_checkpoint
    import torch
    def equal(left, right):
        if type(left) is not type(right):
            return False
        if isinstance(left, torch.Tensor):
            return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, (tuple, list)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right
    left, right = load_checkpoint(first, map_location='cpu'), load_checkpoint(second, map_location='cpu')
    return all(equal(getattr(left, name), getattr(right, name))
               for name in ('model_state', 'optimizer_state', 'training_state', 'model_config'))


def _safe_file_reference(root, path):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if linked(path) or not path.is_file() or not path.is_relative_to(root):
        raise ValueError('workflow stage artifact must be a regular file under workflow root')
    return reference(root, path)


def _stage_directory_from_manifest(manifest_path, sessions):
    manifest_path = Path(manifest_path).resolve()
    sessions = Path(sessions).resolve()
    if linked(manifest_path) or not manifest_path.is_relative_to(sessions):
        raise ValueError('workflow recovery manifest escapes sessions')
    # Trainer layout: <stage>/checkpoints/<profile>/<algorithm>/*.recovery.json.
    # Synthetic fixtures may use the older flat <stage>/checkpoints/ layout.
    if (manifest_path.parent.name == 'checkpoints'
            and manifest_path.parent.parent.parent == sessions):
        stage = manifest_path.parent.parent
    elif (manifest_path.parent.parent.name == 'remote_full'
          and manifest_path.parent.parent.parent.name == 'checkpoints'):
        stage = manifest_path.parent.parent.parent.parent
    else:
        raise ValueError('workflow recovery manifest has an invalid checkpoint directory')
    if stage.parent != sessions:
        raise ValueError('workflow recovery manifest has an invalid stage directory')
    return stage


def _training_result_from_record(record):
    from ..training.trainer import TrainingResult
    training = record.get('training')
    if not isinstance(training, dict):
        raise ValueError('successful training session is missing TrainingResult')
    names = {field.name for field in dataclass_fields(TrainingResult)}
    if set(training) - names:
        raise ValueError('training result contains unknown fields')
    missing = [name for name in names if name not in training and field_default(TrainingResult, name) is _MISSING]
    if missing:
        raise ValueError(f'training result is missing fields: {missing}')
    return TrainingResult(**training)


_MISSING = object()


def field_default(cls, name):
    for field in dataclass_fields(cls):
        if field.name == name:
            if field.default is not MISSING:
                return field.default
            if field.default_factory is not MISSING:
                return field.default_factory()
            return _MISSING
    return _MISSING


def _workflow_stage_path(stage_dir):
    return Path(stage_dir) / STAGE



def checked_evaluation(result, profile, algorithm, candidate, snapshot):
    """Validate a completed report with the same gate used by session.run."""
    # Keep cached/skip validation at least as strict as first-run validation.
    session._verify_evaluation_result(result, profile)
    from ..evaluation_stats import summarize_games
    cfg = profile['config']['evaluation']
    target = cfg['deal_groups_per_pairing'] * len(cfg['seat_rotations'])
    counts = {name: target for name in cfg['opponents']}
    if (result.get('errors') or result['status'] != 'complete' or result['profile'] != 'remote_full'
            or result['completed_games_per_opponent'] != counts
            or result['total_games'] != sum(counts.values())):
        raise ValueError('evaluation incomplete or wrong actual counts')
    records = result['records']
    expected = {(op, g, r) for op in cfg['opponents'] for g in range(cfg['deal_groups_per_pairing'])
                for r in cfg['seat_rotations']}
    actual = {(r['opponent'], r['deal_group'], r['seat_rotation']) for r in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError('evaluation deal/rotation coverage mismatch')
    summary = summarize_games(records)
    if {name: summary['per_opponent'][name]['completed'] for name in counts} != counts:
        raise ValueError('terminal record counts mismatch')
    for role, checkpoint, update in [('candidate', candidate, 102), ('snapshot_opponent', snapshot, 100)]:
        meta = result['metadata'][role]
        if meta['sha256'] != digest(checkpoint) or meta['algorithm'] != algorithm or meta['update'] != update:
            raise ValueError('evaluation refers to different checkpoints')
    return result


class Workflow:
    """Coordinator; executor/resolver injection is for bounded offline unit tests only."""
    def __init__(self, root, profile, source, controller, *, executor=None, resolver=None, retry_failed=False):
        self.root = Path(root).resolve()
        self.sessions = self.root / 'sessions'
        self.profile, self.source, self.controller = profile, source, controller
        self.retry_failed = bool(retry_failed)
        self.executor = executor or session.run
        self.resolver = resolver or session.resolve_recovery
        self.archive = self.root.parent / (self.root.name + '-progress.zip')
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / STATE
        if path.exists():
            self.state = json.loads(path.read_text(encoding='utf-8'))
            if (self.state.get('format') != 'guandan-unified-v1'
                    or self.state['source_git_commit'] != source['git_commit']
                    or self.state['profile_sha256'] != profile['sha256']):
                raise ValueError('existing workflow source/profile mismatch')
            if self.state.get('status') == 'failed' and not retry_failed:
                raise ValueError('previous workflow failed; fix the reported error before explicit --retry-failed')
            self.previous_run = self.state['run_id']
        else:
            self.state = {'format': 'guandan-unified-v1', 'source_git_commit': source['git_commit'],
                          'profile_sha256': profile['sha256'], 'evaluations': {}, 'history': []}
            self.previous_run = None
        if 'error' in self.state:
            self.state.setdefault('previous_errors', []).append(self.state.pop('error'))
        self.state.update(run_id=uuid.uuid4().hex, previous_run_id=self.previous_run,
                          status='running', accepted=False, kaggle_verified=False,
                          genuine_new_kaggle_session_verified=False,
                          resume_semantics='checkpoint reload; not proof of a new Kaggle session',
                          incomplete_evaluation_policy='preserve records, restart incomplete algorithm evaluation')
        self.save()

    def save(self):
        atomic_json(self.root / STATE, self.state)

    def _input_metadata(self, manifest):
        if manifest is None:
            return None
        manifest = Path(manifest).resolve()
        if not manifest.is_relative_to(self.root):
            raise ValueError('workflow stage input must be under the current workflow root')
        raw = json.loads(manifest.read_text(encoding='utf-8'))
        data = self.resolver(manifest, algorithm=raw['algorithm'])
        checkpoint = Path(data['checkpoint_path']).resolve()
        return {
            'manifest': _safe_file_reference(self.root, manifest),
            'checkpoint': _safe_file_reference(self.root, checkpoint),
            'durable_updates': data['durable_updates'],
            'checkpoint_sha256': data['checkpoint_sha256'],
        }

    def _discover_stage_directory(self, before):
        candidates = [path for path in self.sessions.iterdir() if path.is_dir() and path not in before]
        return sorted(candidates)[-1] if candidates else None

    def _write_stage_metadata(self, output, *, action, algorithm, stage_input,
                              record=None, error=None):
        output = None if output is None else Path(output).resolve()
        if output is not None and not output.is_relative_to(self.root):
            raise ValueError('stage output escapes workflow root')
        session_result = None if output is None else output / 'session_result.json'
        session_failure = None if output is None else output / 'session_failure.json'
        evidence = session_result if session_result is not None and session_result.is_file() else session_failure
        stage = {
            'format': 'guandan-workflow-stage-v1',
            'action': action,
            'algorithm': algorithm,
            'status': None if record is None else record.get('status'),
            'input': stage_input,
            'evidence': None if evidence is None else _safe_file_reference(self.root, evidence),
            'recovery_manifest': None,
            'checkpoint': None,
            'verification': {'status': 'not_run'},
        }
        if error is not None:
            stage['error'] = {'type': type(error).__name__, 'message': str(error)}
        if record is not None and output is not None and record.get('status') == 'phase_complete':
            manifest_name = record.get('recovery_manifest')
            training = record.get('training') or {}
            if manifest_name is None:
                manifest_name = training.get('recovery_manifest')
            checkpoint_name = training.get('checkpoint')
            if manifest_name and checkpoint_name:
                manifest = Path(manifest_name).expanduser().resolve()
                checkpoint = Path(checkpoint_name).expanduser().resolve()
                if (manifest.is_file() and checkpoint.is_file()
                        and manifest.is_relative_to(output)
                        and checkpoint.is_relative_to(output)):
                    stage['recovery_manifest'] = _safe_file_reference(self.root, manifest)
                    stage['checkpoint'] = _safe_file_reference(self.root, checkpoint)
        path = _workflow_stage_path(output) if output is not None else None
        if path is not None:
            atomic_json(path, stage)
        return path, stage

    def _verify_training_stage(self, stage_path, expected_manifest=None):
        stage_path = Path(stage_path).resolve()
        stage = json.loads(stage_path.read_text(encoding='utf-8'))
        if stage.get('format') != 'guandan-workflow-stage-v1' or stage.get('action') != 'train':
            raise ValueError('training stage metadata is missing or incompatible')
        if stage.get('status') != 'phase_complete' or stage.get('verification', {}).get('status') == 'failed':
            raise ValueError('training stage was not recorded as successful')
        evidence_ref = stage.get('evidence')
        manifest_ref = stage.get('recovery_manifest')
        checkpoint_ref = stage.get('checkpoint')
        if not evidence_ref or not manifest_ref or not checkpoint_ref:
            raise ValueError('successful training stage lacks relative artifact references')
        evidence_path = resolve_reference(self.root, evidence_ref)
        manifest_path = resolve_reference(self.root, manifest_ref)
        checkpoint_path = resolve_reference(self.root, checkpoint_ref)
        if expected_manifest is not None and manifest_path != Path(expected_manifest).resolve():
            raise ValueError('training stage manifest reference differs from recovery manifest')
        record = json.loads(evidence_path.read_text(encoding='utf-8'))
        if record.get('status') != 'phase_complete' or record.get('action') != 'train':
            raise ValueError('training session_result is not a successful phase result')
        training_data = dict(record.get('training') or {})
        if training_data.get('status') != 'complete':
            raise ValueError('training session_result contains an incomplete TrainingResult')
        input_meta = stage.get('input')
        recovery = None
        if input_meta is not None:
            input_manifest = resolve_reference(self.root, input_meta['manifest'])
            recovery = self.resolver(input_manifest, algorithm=stage['algorithm'])
            if recovery['checkpoint_sha256'] != input_meta.get('checkpoint_sha256'):
                raise ValueError('training stage input checkpoint digest changed')
            if recovery['durable_updates'] != input_meta.get('durable_updates'):
                raise ValueError('training stage input update changed')
            if training_data.get('resume_sha256') != input_meta.get('checkpoint_sha256'):
                raise ValueError('training resume SHA does not match selected parent checkpoint')
            if (not training_data.get('resumed_from')
                    or Path(training_data['resumed_from']).name != Path(recovery['checkpoint_path']).name):
                raise ValueError('training resume path does not match selected parent checkpoint')
        plan = session.training_plan(stage['algorithm'], recovery)
        training_data['checkpoint'] = str(checkpoint_path)
        training_data['recovery_manifest'] = str(manifest_path)
        from ..training.trainer import TrainingResult
        result = TrainingResult(**training_data)
        if self.resolver is session.resolve_recovery:
            verified = session._verify_training_completion(
                result, plan, stage['algorithm'], self.source,
                checkpoint_root=manifest_path.parent,
            )
        else:
            # Explicit synthetic resolver seam used only by offline tests; keep
            # counters, digest, path, profile and source assertions mandatory.
            recovery = self.resolver(manifest_path, algorithm=stage['algorithm'])
            if (recovery.get('source_git_commit') != self.source['git_commit']
                    or recovery.get('checkpoint_path') != str(checkpoint_path)
                    or recovery.get('checkpoint_sha256') != digest(checkpoint_path)
                    or recovery.get('durable_updates') != result.durable_updates
                    or recovery.get('durable_token_steps') != result.durable_token_steps):
                raise ValueError('synthetic training stage recovery evidence mismatch')
            verified = recovery
        stage['verification'] = {'status': 'passed', 'durable_updates': verified['durable_updates'],
                                'durable_token_steps': verified['durable_token_steps']}
        atomic_json(stage_path, stage)
        return verified

    def _lineage_verified(self, entry, by_count, memo, active):
        key = (entry['durable_updates'], entry['checkpoint_sha256'])
        if key in memo:
            return memo[key]
        if key in active:
            raise ValueError('cyclic workflow checkpoint lineage')
        active.add(key)
        parent = entry.get('stage_input')
        if parent is None:
            value = entry['start_update'] == 0
        else:
            candidates = [item for item in by_count.get(parent['durable_updates'], [])
                          if item['checkpoint_sha256'] == parent['checkpoint_sha256']]
            value = bool(candidates) and any(self._lineage_verified(item, by_count, memo, active)
                                             for item in candidates)
        active.remove(key)
        memo[key] = value
        return value

    def recoveries(self, algorithm):
        by_count = {}
        for path in sorted(self.sessions.rglob('*.recovery.json')):
            self.controller.check()
            if linked(path) or not path.resolve().is_relative_to(self.sessions):
                raise ValueError('escaping workflow recovery manifest')
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['algorithm'] not in self.profile['config']['algorithms']:
                raise ValueError('unknown recovery algorithm')
            if row['algorithm'] != algorithm:
                continue
            data = self.resolver(path, algorithm=algorithm)
            count = data['durable_updates']
            if (data['source_git_commit'] != self.source['git_commit']
                    or type(count) is not int or not 0 <= count <= 102
                    or data['durable_token_steps'] != count * 1024):
                raise ValueError('incompatible durable workflow checkpoint')
            checkpoint = Path(data['checkpoint_path']).resolve()
            if linked(checkpoint) or not checkpoint.is_relative_to(self.sessions):
                raise ValueError('checkpoint escapes workflow sessions')
            stage_dir = _stage_directory_from_manifest(path, self.sessions)
            stage_path = _workflow_stage_path(stage_dir)
            stage = json.loads(stage_path.read_text(encoding='utf-8')) if stage_path.is_file() else None
            success_verified = False
            verification_error = None
            if stage is not None and stage.get('action') == 'train' and stage.get('status') == 'phase_complete':
                try:
                    self._verify_training_stage(stage_path, expected_manifest=path)
                    success_verified = True
                except Exception as exc:
                    verification_error = f'{type(exc).__name__}: {exc}'
            stage_input = None if stage is None else stage.get('input')
            start_update = 0 if stage_input is None else stage_input.get('durable_updates')
            entry = {**data, 'manifest_path': str(path.resolve()), 'stage_path': str(stage_path.resolve()),
                     'stage_input': stage_input, 'start_update': start_update,
                     'success_verified': success_verified, 'verification_error': verification_error}
            by_count.setdefault(count, []).append(entry)

        latest = {}
        for count, entries in by_count.items():
            verified = [entry for entry in entries if entry['success_verified']]
            pool = verified or entries
            selected = sorted(pool, key=lambda entry: entry['manifest_path'])[-1]
            for entry in pool:
                if entry['checkpoint_sha256'] == selected['checkpoint_sha256']:
                    continue
                if equivalent_checkpoints(selected['checkpoint_path'], entry['checkpoint_path']):
                    continue
                if selected['success_verified'] != entry['success_verified']:
                    if entry['success_verified']:
                        selected = entry
                    continue
                raise ValueError('ambiguous checkpoint lineage: same algorithm/update but different state')
            latest[count] = selected
        memo = {}
        for count, entries in by_count.items():
            for entry in entries:
                entry['lineage_verified'] = self._lineage_verified(entry, by_count, memo, set())
        for count, entry in latest.items():
            entry['lineage_verified'] = self._lineage_verified(entry, by_count, memo, set())
        return latest

    def invoke(self, action, algorithm, **inputs):
        self.controller.check()
        args = session.parser().parse_args(['--action', action, '--profile', 'remote_full', '--execute',
                                            '--algorithm', algorithm, '--output-root', str(self.sessions)])
        args.session_hours = self.controller.hard_seconds / 3600
        args.save_margin_seconds = self.controller.hard_seconds - self.controller.soft_seconds
        args.checkpoint_seconds = self.controller.interval
        for key, value in inputs.items():
            setattr(args, key, Path(value))
        stage_input = self._input_metadata(inputs.get('resume')) if action == 'train' else None
        self.state['active'] = {'action': action, 'algorithm': algorithm}
        self.save()
        before = set(self.sessions.iterdir()) if self.sessions.exists() else set()
        print(f'[workflow] START {algorithm} {action}; elapsed={self.controller.clock()-self.controller.started:.1f}s', flush=True)
        try:
            record = self.executor(args, checkpoint_controller=self.controller, working_artifacts_root=self.sessions)
        except BaseException as exc:
            output = self._discover_stage_directory(before)
            stage_path, stage = self._write_stage_metadata(output, action=action, algorithm=algorithm,
                                                           stage_input=stage_input, error=exc)
            history = {'action': action, 'algorithm': algorithm, 'status': 'raised', 'input': stage_input,
                       'stage': None if stage_path is None else _safe_file_reference(self.root, stage_path),
                       'error': {'type': type(exc).__name__, 'message': str(exc)}}
            self.state['history'].append(history)
            self.state.pop('active', None)
            self.save()
            raise
        output = Path(record['outputs']).resolve()
        if not output.is_relative_to(self.root):
            raise ValueError('stage output escapes workflow root')
        stage_path, stage = self._write_stage_metadata(output, action=action, algorithm=algorithm,
                                                       stage_input=stage_input, record=record)
        evidence = output / 'session_result.json'
        if not evidence.is_file():
            evidence = output / 'session_failure.json'
        history = {'action': action, 'algorithm': algorithm, 'status': record['status'], 'input': stage_input,
                   'stage': _safe_file_reference(self.root, stage_path),
                   'result': _safe_file_reference(self.root, evidence) if evidence.is_file() else None}
        self.state['history'].append(history)
        self.state.pop('active', None)
        self.save()
        print(f'[workflow] END {algorithm} {action}: {record["status"]}', flush=True)
        return record

    def run(self):
        from ..evaluation import EvaluationIncomplete
        from ..training.runtime import BudgetExceeded
        try:
            self.invoke('smoke', 'ippo')
            for algorithm in self.profile['config']['algorithms']:
                while True:
                    recoveries = self.recoveries(algorithm)
                    count = max(recoveries, default=-1)
                    if count > 100 and 100 not in recoveries:
                        raise ValueError('missing frozen update100 snapshot; cannot skip base evidence')
                    if count == 102:
                        entry = recoveries[102]
                        if entry['success_verified'] and entry['lineage_verified']:
                            print(f'[workflow] SKIP {algorithm} training: verified update102', flush=True)
                            break
                        if self.state.get('status') == 'failed' and not self.retry_failed:
                            raise ValueError('update102 gate failed; retry requires explicit --retry-failed')
                        parent = entry.get('stage_input')
                        if not parent or not parent.get('manifest'):
                            raise ValueError('update102 lacks a verifiable parent recovery; cannot retry or skip')
                        resume = resolve_reference(self.root, parent['manifest'])
                        count = parent['durable_updates']
                        inputs = {'resume': resume}
                    else:
                        inputs = {} if count < 0 else {'resume': recoveries[count]['manifest_path']}
                    record = self.invoke('train', algorithm, **inputs)
                    export_progress(self.root, self.archive, controller=self.controller)
                    if record['status'] != 'phase_complete':
                        raise SessionStop('training incomplete; restore saved progress in next session')
                    after = self.recoveries(algorithm)
                    expected = 100 if count < 100 else 102
                    if (expected not in after or not after[expected]['success_verified']
                            or not after[expected]['lineage_verified']):
                        raise ValueError('training completion/recovery-lineage gate failed; cannot continue or skip')
            for algorithm in self.profile['config']['algorithms']:
                rows = self.recoveries(algorithm)
                if 100 not in rows or 102 not in rows:
                    raise ValueError(f'missing candidate/snapshot checkpoints for {algorithm}')
                candidate, snapshot = rows[102]['checkpoint_path'], rows[100]['checkpoint_path']
                prior = self.state['evaluations'].get(algorithm)
                if prior:
                    self.controller.check()
                    path = resolve_reference(self.root, prior)
                    checked_evaluation(json.loads(path.read_text(encoding='utf-8')), self.profile,
                                      algorithm, candidate, snapshot)
                    print(f'[workflow] SKIP {algorithm} evaluation: verified complete records', flush=True)
                    continue
                record = self.invoke('evaluate', algorithm,
                                     candidate_checkpoint=candidate, snapshot_checkpoint=snapshot)
                if record['status'] != 'evaluation_complete':
                    raise ValueError('evaluation did not complete')
                result = record['evaluation']
                checked_evaluation(result, self.profile, algorithm, candidate, snapshot)
                path = Path(record['outputs']) / 'evaluation' / 'summary.json'
                self.state['evaluations'][algorithm] = reference(self.root, path)
                self.save()
                export_progress(self.root, self.archive, controller=self.controller)
            self.controller.check()
            self.state['status'] = 'pipeline_complete_pending_remote_review'
        except (BudgetExceeded, EvaluationIncomplete, KeyboardInterrupt) as exc:
            self.state.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        except Exception as exc:
            self.state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            active_exception = sys.exc_info()[0] is not None
            started_elapsed = self.controller.clock() - self.controller.started
            self.state['elapsed_seconds'] = float(started_elapsed)
            self.state['export_status'] = 'pending'
            self.save()
            export_started = self.controller.clock()
            try:
                bundle = export_progress(self.root, self.archive, controller=self.controller)
            except SessionStop as export_stop:
                finished = self.controller.clock()
                export_seconds = float(finished - export_started)
                total_elapsed = float(finished - self.controller.started)
                if self.state.get('status') != 'failed':
                    self.state.update(status='incomplete', error=f'{type(export_stop).__name__}: {export_stop}')
                self.state['export_status'] = 'failed'
                self.state['progress_export_over_budget'] = True
                self.save()
                receipt = write_export_receipt(
                    self.archive, status=self.state['status'], export_seconds=export_seconds,
                    total_elapsed_seconds=total_elapsed, error=export_stop)
                self.state.update(export_seconds=export_seconds, total_elapsed_seconds=total_elapsed,
                                  export_receipt=str(receipt))
                if not active_exception:
                    print(f'[workflow] {self.state["status"]}; progress export failed: {receipt}', flush=True)
            except Exception as export_error:
                finished = self.controller.clock()
                export_seconds = float(finished - export_started)
                total_elapsed = float(finished - self.controller.started)
                if self.state.get('status') != 'failed':
                    self.state.update(status='failed', error=f'{type(export_error).__name__}: {export_error}')
                self.state['export_status'] = 'failed'
                self.save()
                receipt = write_export_receipt(
                    self.archive, status=self.state['status'], export_seconds=export_seconds,
                    total_elapsed_seconds=total_elapsed, error=export_error)
                self.state.update(export_seconds=export_seconds, total_elapsed_seconds=total_elapsed,
                                  export_receipt=str(receipt))
                if not active_exception:
                    raise
            else:
                receipt = write_export_receipt(
                    self.archive, status=self.state['status'], export_seconds=bundle['export_seconds'],
                    total_elapsed_seconds=bundle['total_elapsed_seconds'], bundle=bundle)
                self.state.update(export_status='complete', export_seconds=bundle['export_seconds'],
                                  total_elapsed_seconds=bundle['total_elapsed_seconds'],
                                  export_receipt=str(receipt))
                print(f'[workflow] {self.state["status"]}; download {bundle["path"]}', flush=True)
        return self.state


def main(argv=None):
    parser = argparse.ArgumentParser(description='One Run All, shared budget; never uploads or accepts stages')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--retry-failed', action='store_true', help='explicit retry only after fixing a recorded failure')
    parser.add_argument('--output-root', type=Path, default=Path('/kaggle/working/guandan-unified'))
    parser.add_argument('--resume-bundle', type=Path)
    parser.add_argument('--session-hours', type=float, default=10.0)
    parser.add_argument('--save-margin-seconds', type=float, default=300.0)
    parser.add_argument('--checkpoint-seconds', type=float, default=600.0)
    args = parser.parse_args(argv)
    profile = session._profile()
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'profile': profile,
                          'order': ['ippo100', 'ippo102', 'vrpo100', 'vrpo102', 'ippo_eval3000', 'vrpo_eval3000'],
                          'accepted': False, 'new_kaggle_session_proven': False}, ensure_ascii=False))
        return 0
    # Created before setup/restore. One clock for smoke + both algorithms + both evaluations.
    controller = CheckpointController(args.session_hours, args.save_margin_seconds, args.checkpoint_seconds)
    from scripts.kaggle_environment_check import inspect_environment, require_remote_runtime
    require_remote_runtime(inspect_environment())
    root = session._output_directory(args.output_root)
    source = source_provenance(session.ROOT)
    if not source['git_worktree_clean']:
        raise ValueError('workflow requires clean committed or verified exported source')
    if args.resume_bundle:
        archive = session._require_input_path(args.resume_bundle, 'workflow resume bundle')
        restore_progress(archive, root, commit=source['git_commit'], profile_sha256=profile['sha256'])
    workflow = Workflow(root, profile, source, controller, retry_failed=args.retry_failed)
    workflow.state['restore_input'] = None if not args.resume_bundle else {
        'path': str(archive), 'kind': 'expanded_dataset' if archive.is_dir() else 'zip_dataset'}
    workflow.state['runtime'] = inspect_environment()
    workflow.state['profile'] = profile
    workflow.save()
    result = workflow.run()
    print(json.dumps({key: result[key] for key in ('status', 'run_id', 'elapsed_seconds', 'accepted', 'kaggle_verified')}, ensure_ascii=False))
    return 1 if result['status'] == 'failed' else (2 if result['status'] == 'incomplete' else 0)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'[workflow] FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
