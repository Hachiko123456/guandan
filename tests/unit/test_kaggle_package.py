"""Local-only packaging checks; fixtures/mock clocks are not Kaggle evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

from guandan.deployment.package import build_package


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts/kaggle_entry.py"
CHECKER = ROOT / "scripts/kaggle_environment_check.py"
NOTEBOOK = ROOT / "notebooks/kaggle_training.py"


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checker():
    return load_script(CHECKER, "a07_checker_test")


@pytest.fixture
def clean_cwd(tmp_path):
    cwd = tmp_path / "unrelated cwd"
    cwd.mkdir()
    return cwd


def cli(script, *arguments, cwd, no_site=True, env=None):
    return subprocess.run(
        [sys.executable, "-B", *(["-S"] if no_site else []), str(script), *arguments],
        cwd=cwd, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", **(env or {})},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45,
    )


@pytest.fixture
def exported_source(tmp_path):
    """A local source-copy fixture, not a clean-commit or remote acceptance claim."""
    source = tmp_path / "input mount" / "source"
    shutil.copytree(ROOT / "guandan", source / "guandan", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("scripts/kaggle_entry.py", "scripts/kaggle_environment_check.py",
                 "notebooks/kaggle_training.py", "notebooks/kaggle_unified_bootstrap.py",
                 "pyproject.toml", "configs/acceptance_profiles.json"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    files = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(source.rglob("*")) if path.is_file()}
    (source / "SOURCE_MANIFEST.json").write_text(json.dumps({
        "format": "guandan-source-v1", "git_commit": "a" * 40, "files": files,
        "profile": "remote_full", "uploaded": False, "accepted": False, "kaggle_verified": False,
    }), encoding="utf-8")
    return source


def git_fixture(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode("utf-8").strip()


def package_fixture_repo(tmp_path):
    root = tmp_path / "package source"
    root.mkdir()
    git_fixture(root, "init", "-q")
    git_fixture(root, "config", "user.name", "Package Fixture")
    git_fixture(root, "config", "user.email", "package@example.invalid")
    shutil.copytree(ROOT / "guandan", root / "guandan", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("scripts/kaggle_entry.py", "scripts/kaggle_environment_check.py",
                 "notebooks/kaggle_unified_bootstrap.py", "pyproject.toml", "configs/acceptance_profiles.json"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    git_fixture(root, "add", ".")
    git_fixture(root, "commit", "-qm", "package fixture")
    return root


@pytest.mark.parametrize("path", [ENTRY, CHECKER, NOTEBOOK])
def test_module_import_has_no_dependency_or_execution_side_effects(path, clean_cwd):
    code = '''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','guandan'}:
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0,Block())
runpy.run_path(sys.argv[1],run_name='inspection_only')
'''
    result = subprocess.run([sys.executable, "-B", "-S", "-c", code, str(path)],
                            cwd=clean_cwd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert list(clean_cwd.iterdir()) == []


@pytest.mark.parametrize("path", [ENTRY, CHECKER, NOTEBOOK])
def test_help_works_without_any_site_packages(path, clean_cwd):
    result = cli(path, "--help", cwd=clean_cwd)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert list(clean_cwd.iterdir()) == []


def test_checker_show_system_reports_missing_torch_without_importing_project(clean_cwd):
    result = cli(CHECKER, "--show-system", cwd=clean_cwd)
    assert result.returncode == 0, result.stderr
    system = json.loads(result.stdout)
    assert system["python_version_info"][:2] >= [3, 12]
    assert system["torch_status"] == "missing"
    assert system["cuda_available"] is False
    assert system["gpus"] == []
    assert system["accepted"] is system["kaggle_verified"] is False
    assert list(clean_cwd.iterdir()) == []


@pytest.mark.parametrize("arguments", [[], ["--action", "plan", "--execute"]])
def test_default_and_plan_never_execute_even_with_hostile_profile_env(exported_source, clean_cwd, arguments):
    result = cli(exported_source / "scripts/kaggle_entry.py", *arguments, cwd=clean_cwd,
                 env={"GUANDAN_PROFILE": "local_fast", "GUANDAN_RESOLVED_PROFILE_JSON": "bad inherited value"})
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "configuration_only"
    assert report["executed"] is False
    assert report["plan"]["profile"]["name"] == "remote_full"
    assert report["plan"]["updates_this_session_if_time_allows"] == 100
    assert report["plan"]["automatic_target_reduction"] is False
    assert not (exported_source / "runs").exists()
    assert list(clean_cwd.iterdir()) == []


@pytest.mark.parametrize("action", ["smoke", "train", "evaluate"])
def test_expensive_action_without_execute_is_refused(action, exported_source, clean_cwd):
    result = cli(exported_source / "scripts/kaggle_entry.py", "--action", action, cwd=clean_cwd)
    assert result.returncode != 0
    assert "--execute" in result.stderr
    assert list(clean_cwd.iterdir()) == []


def test_session_system_without_torch_fails_closed(exported_source, clean_cwd):
    result = cli(exported_source / "scripts/kaggle_entry.py", "--action", "system", cwd=clean_cwd)
    assert result.returncode != 0
    report = json.loads(result.stderr)
    assert report["status"] == "failed"
    assert report["accepted"] is False
    assert "PyTorch" in report["error"] or "CUDA" in report["error"] or "Kaggle" in report["error"]


def test_entry_delegates_argv_exit_code_and_preserves_environment(tmp_path, clean_cwd):
    root = tmp_path / "stub source"
    for name in ("scripts", "configs", "guandan/deployment"):
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ENTRY, root / "scripts/kaggle_entry.py")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "configs/acceptance_profiles.json").write_text("{}", encoding="utf-8")
    (root / "guandan/__init__.py").write_text("", encoding="utf-8")
    (root / "guandan/deployment/__init__.py").write_text("", encoding="utf-8")
    (root / "guandan/deployment/session.py").write_text(
        "import json,os,sys\ndef main(argv=None):\n"
        " print(json.dumps({'argv':sys.argv[1:] if argv is None else argv, 'cwd':os.getcwd(), 'env':os.getenv('GUANDAN_PROFILE'), 'root':sys.path[0]}))\n"
        " return 7\n", encoding="utf-8")
    result = cli(root / "scripts/kaggle_entry.py", "--action", "train", "--resume", "a b/manifest.json",
                 cwd=clean_cwd, env={"GUANDAN_PROFILE": "sentinel"})
    assert result.returncode == 7
    report = json.loads(result.stdout)
    assert report["argv"] == ["--action", "train", "--resume", "a b/manifest.json"]
    assert Path(report["root"]) == root
    assert Path(report["cwd"]) == clean_cwd
    assert report["env"] == "sentinel"


def test_bootstrap_rejects_incomplete_root_and_loaded_other_checkout(tmp_path, monkeypatch):
    entry = load_script(ENTRY, "a07_entry_test")
    monkeypatch.setattr(entry, "__file__", str(tmp_path / "scripts/kaggle_entry.py"))
    with pytest.raises(RuntimeError, match="incomplete source"):
        entry.bootstrap_root()
    monkeypatch.setattr(entry, "__file__", str(ENTRY))
    monkeypatch.setitem(sys.modules, "guandan", types.SimpleNamespace(__file__=str(tmp_path / "old/guandan/__init__.py")))
    with pytest.raises(RuntimeError, match="another guandan"):
        entry.bootstrap_root()


def valid_system():
    return {"python_version_info": [3, 12, 0], "torch_status": "ok", "cuda_available": True,
            "gpu_count": 1, "gpus": [{"index": 0, "name": "fixture only"}],
            "kaggle": {"input_exists": True, "working_exists": True, "working_writable": True},
            "accepted": False, "kaggle_verified": False}


@pytest.mark.parametrize("change", [
    {"python_version_info": [3, 11, 9]}, {"python_version_info": []},
    {"torch_status": "missing"}, {"torch_status": "error"},
    {"cuda_available": False}, {"gpu_count": 0}, {"gpus": []}, {"kaggle": {}},
])
def test_require_remote_runtime_fails_closed(checker, change):
    with pytest.raises(RuntimeError):
        checker.require_remote_runtime({**valid_system(), **change})


def test_runtime_marker_validation_never_grants_acceptance(checker):
    system = valid_system()
    assert checker.require_remote_runtime(system) is None
    assert system["accepted"] is system["kaggle_verified"] is False


def test_inspection_uses_actual_lazy_cuda_api_not_environment_gpu_names(checker, monkeypatch, tmp_path):
    calls = []
    fake_torch = types.SimpleNamespace(__version__="test-wheel", version=types.SimpleNamespace(cuda="12.test"),
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
             get_device_properties=lambda index: types.SimpleNamespace(total_memory=1234),
             get_device_name=lambda index: "measured device", get_device_capability=lambda index: (8, 6)))
    def importer(name):
        calls.append(name)
        return fake_torch
    monkeypatch.setattr(checker.importlib, "import_module", importer)
    monkeypatch.setattr(checker, "KAGGLE_INPUT", tmp_path / "absent input")
    monkeypatch.setattr(checker, "KAGGLE_WORKING", tmp_path / "absent work")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "secret marker content")
    result = checker.inspect_environment()
    assert calls == ["torch"]
    assert result["gpus"][0]["name"] == "measured device"
    assert result["gpus"][0]["total_memory_bytes"] == 1234
    assert result["torch_status"] == "ok"
    assert "secret marker content" not in json.dumps(result)
    with pytest.raises(RuntimeError, match="directories"):
        checker.require_remote_runtime(result)


def test_inspection_reports_broken_cuda_as_error(checker, monkeypatch):
    def broken(_):
        raise OSError("missing runtime DLL")
    monkeypatch.setattr(checker.importlib, "import_module", broken)
    report = checker.inspect_environment()
    assert report["torch_status"] == "error"
    assert "missing runtime DLL" in report["torch_error"]


def test_install_dry_run_never_copies_or_invokes_pip(checker, exported_source, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not invoke subprocess/pip")
    monkeypatch.setattr(subprocess, "run", forbidden)
    destination = tmp_path / "working" / "source"
    result = checker.install_source(exported_source, destination, install_deps=True, dry_run=True, enforce_kaggle_paths=False)
    assert result["status"] == "planned"
    assert not destination.exists()
    command = result["commands"][0]
    assert command[:4] == [sys.executable, "-m", "pip", "install"]
    assert "--only-binary=:all:" in command and "numpy>=2" in command
    assert result["uploaded"] is result["kaggle_verified"] is False


def test_install_rejects_extra_unmanifested_files_before_copying(checker, exported_source, tmp_path, monkeypatch):
    for name in ("runs/x.pt", "logs/log.txt", "checkpoints/model.pt", ".git/config", "__pycache__/cache.pyc", ".env"):
        extra = exported_source / name
        extra.parent.mkdir(exist_ok=True, parents=True)
        extra.write_text("should not be copied", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("implicit pip"))
    destination = tmp_path / "writable" / "source"
    with pytest.raises(ValueError, match="file-set mismatch"):
        checker.install_source(exported_source, destination, enforce_kaggle_paths=False)
    assert not destination.exists()


def test_install_explicit_pip_failure_remains_failure(checker, exported_source, tmp_path, monkeypatch):
    calls = []
    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return types.SimpleNamespace(returncode=19)
    monkeypatch.setattr(subprocess, "run", fake_run)
    report = checker.install_source(exported_source, tmp_path / "fresh", install_deps=True, enforce_kaggle_paths=False)
    assert len(calls) == 1
    assert report["dependency_returncode"] == 19
    assert report["status"] == "dependency_install_failed"
    monkeypatch.setattr(checker, "install_source", lambda *args: report)
    assert checker.main(["--source-root", "input", "--dest-root", "output", "--install-deps"]) == 19


def test_install_rejects_modified_or_incomplete_source(checker, exported_source, tmp_path):
    target = tmp_path / "target"
    (exported_source / "guandan/__init__.py").write_text("# modified", encoding="utf-8")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        checker.install_source(exported_source, target, enforce_kaggle_paths=False)
    assert not target.exists()
    with pytest.raises(ValueError, match="incomplete source root"):
        checker.install_source(tmp_path, target, enforce_kaggle_paths=False)


@pytest.mark.parametrize("unsafe", ["../escape.py", "/absolute.py", "C:/host.py", "sub\\host.py"])
def test_installer_rejects_manifest_path_escape_before_writing(checker, exported_source, tmp_path, unsafe):
    manifest_path = exported_source / "SOURCE_MANIFEST.json"
    data = json.loads(manifest_path.read_text())
    data["files"][unsafe] = "0" * 64
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe source manifest"):
        checker.install_source(exported_source, tmp_path / "target", enforce_kaggle_paths=False)
    assert not (tmp_path / "target").exists()


def test_installer_rejects_nested_input_dest_and_linked_source(checker, exported_source, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="separate"):
        checker.install_source(exported_source, exported_source / "copy", enforce_kaggle_paths=False)
    monkeypatch.setattr(checker, "KAGGLE_INPUT", tmp_path / "readonly")
    with pytest.raises(ValueError, match="read-only"):
        checker.install_source(exported_source, tmp_path / "readonly/copy", enforce_kaggle_paths=False)
    monkeypatch.setattr(checker, "_is_link", lambda path: path.name == "pyproject.toml")
    with pytest.raises(ValueError, match="linked source"):
        checker.install_source(exported_source, tmp_path / "target", enforce_kaggle_paths=False)


def test_notebook_commands_are_explicit_and_script_friendly():
    notebook = load_script(NOTEBOOK, "a07_notebook_test")
    plan = notebook.session_command(root=ROOT)
    assert "--execute" not in plan and plan[plan.index("--action") + 1] == "plan"
    resume = notebook.session_command("train", root=ROOT, execute=True, resume="/new-mount/manifest.json")
    assert "--execute" in resume and resume[-1] == "/new-mount/manifest.json"
    evaluation = notebook.session_command("evaluate", execute=True, candidate_checkpoint="update102.pt", snapshot_checkpoint="update100.pt")
    assert evaluation[-4:] == ["--candidate-checkpoint", "update102.pt", "--snapshot-checkpoint", "update100.pt"]


def test_canonical_profile_environment_is_restored_even_on_failure(monkeypatch):
    from guandan.deployment.session import canonical_profile, _profile
    monkeypatch.setenv("GUANDAN_PROFILE", "local_fast")
    monkeypatch.setenv("GUANDAN_RESOLVED_PROFILE_JSON", "sentinel")
    with pytest.raises(RuntimeError, match="test failure"):
        with canonical_profile(_profile()):
            assert os.environ["GUANDAN_PROFILE"] == "remote_full"
            assert json.loads(os.environ["GUANDAN_RESOLVED_PROFILE_JSON"])["name"] == "remote_full"
            raise RuntimeError("test failure")
    assert os.environ["GUANDAN_PROFILE"] == "local_fast"
    assert os.environ["GUANDAN_RESOLVED_PROFILE_JSON"] == "sentinel"


def test_timeout_and_signal_checkpoint_hooks_use_upstream_controller(monkeypatch):
    from guandan.deployment import control
    clock = [0.0]
    controller = control.CheckpointController(session_hours=1, save_margin_seconds=60, checkpoint_seconds=30, clock=lambda: clock[0])
    assert not controller.checkpoint_due()
    clock[0] = 31.0
    assert controller.checkpoint_due()
    controller.checkpoint_saved()
    assert not controller.checkpoint_due()
    controller.after_update(10.0)
    clock[0] = 3530.0
    with pytest.raises(control.SessionStop, match="insufficient_time"):
        controller.before_update()
    timed = control.CheckpointController(session_hours=1, save_margin_seconds=60, checkpoint_seconds=30, clock=lambda: clock[0])
    clock[0] += 3540.0
    with pytest.raises(control.SessionStop, match="session_save_margin"):
        timed.check()
    signal_controller = control.CheckpointController(clock=lambda: clock[0])
    previous = object()
    handlers = {}
    def fake_signal(number, callback):
        old = handlers.get(number, previous)
        handlers[number] = callback
        return old
    monkeypatch.setattr(control.signal, "signal", fake_signal)
    with signal_controller.signal_handlers():
        handlers[control.signal.SIGTERM](control.signal.SIGTERM, None)
        assert signal_controller.reason.startswith("signal_")
        with pytest.raises(control.SessionStop, match="signal_"):
            signal_controller.check()
    assert all(value is previous for value in handlers.values())


def test_build_package_unzip_install_source_round_trip(checker, tmp_path):
    root = package_fixture_repo(tmp_path)
    archive_path = tmp_path / "source.zip"
    result = build_package(root, archive_path)
    unpacked = tmp_path / "unpacked"
    with __import__("zipfile").ZipFile(result["archive"]) as archive:
        archive.extractall(unpacked)
    source = unpacked / "guandan"
    destination = tmp_path / "installed source"
    report = checker.install_source(source, destination, enforce_kaggle_paths=False)
    assert report["status"] == "copied"
    assert report["source"]["source_kind"] == "verified_export"
    assert report["uploaded"] is report["accepted"] is report["kaggle_verified"] is False
    assert (destination / "SOURCE_MANIFEST.json").read_bytes() == (source / "SOURCE_MANIFEST.json").read_bytes()
    assert (destination / "scripts/kaggle_entry.py").is_file()


def test_zip_dataset_upload_round_trip_installs_without_folder_upload(checker, tmp_path):
    root = package_fixture_repo(tmp_path)
    archive_path = tmp_path / "guandan-source.zip"
    result = build_package(root, archive_path)
    destination = tmp_path / "working" / "guandan-source"
    report = checker.install_source_archive(
        archive_path,
        destination,
        enforce_kaggle_paths=False,
    )
    assert report["status"] == "copied"
    assert report["input_kind"] == "zip_source_export"
    assert report["source"]["source_kind"] == "verified_export"
    assert (destination / "SOURCE_MANIFEST.json").is_file()
    assert (destination / "scripts/kaggle_entry.py").is_file()
    assert not list((tmp_path / "working").glob("guandan-archive-*"))


def test_zip_dataset_upload_rejects_extra_member_before_destination(checker, tmp_path):
    root = package_fixture_repo(tmp_path)
    archive_path = tmp_path / "guandan-source.zip"
    result = build_package(root, archive_path)
    import zipfile
    forged = tmp_path / "forged.zip"
    with zipfile.ZipFile(result["archive"]) as source, zipfile.ZipFile(forged, "w") as target:
        for info in source.infolist():
            target.writestr(info, source.read(info))
        target.writestr("guandan/unlisted.py", b"forged")
    destination = tmp_path / "working" / "guandan-source"
    with pytest.raises(ValueError, match="file-set|hash/path mismatch"):
        checker.install_source_archive(forged, destination, enforce_kaggle_paths=False)
    assert not destination.exists()


@pytest.mark.parametrize("missing", [
    "scripts/kaggle_entry.py",
    "guandan/deployment/control.py",
])
def test_build_package_rejects_missing_required_file_before_output(tmp_path, missing):
    root = package_fixture_repo(tmp_path)
    (root / missing).unlink()
    git_fixture(root, "add", "-A")
    git_fixture(root, "commit", "-qm", "remove required deployment file")
    output = tmp_path / "missing.zip"
    with pytest.raises(ValueError, match="incomplete source export"):
        build_package(root, output)
    assert not output.exists()


def test_build_package_rejects_non_remote_full_profile(tmp_path):
    root = package_fixture_repo(tmp_path)
    profile_path = root / "configs/acceptance_profiles.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["profiles"]["remote_full"]["mode"] = "local"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    git_fixture(root, "add", "configs/acceptance_profiles.json")
    git_fixture(root, "commit", "-qm", "forge remote profile")
    output = tmp_path / "profile.zip"
    with pytest.raises(ValueError, match="remote_full"):
        build_package(root, output)
    assert not output.exists()


def test_build_package_rejects_wrong_remote_counts_before_output(tmp_path):
    root = package_fixture_repo(tmp_path)
    profile_path = root / "configs/acceptance_profiles.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["profiles"]["remote_full"]["training"]["updates_per_algorithm"] = 99
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    git_fixture(root, "add", "configs/acceptance_profiles.json")
    git_fixture(root, "commit", "-qm", "forge remote counts")
    output = tmp_path / "counts.zip"
    with pytest.raises(ValueError, match="remote_full configuration mismatch"):
        build_package(root, output)
    assert not output.exists()

# Evaluation delivery: local rendering/stub-process tests, not Kaggle evidence.
@pytest.fixture
def evaluation_packager():
    return load_script(ROOT / 'scripts/prepare_kaggle_evaluation.py', 'evaluation_packager_test')


def evaluation_namespace(module, tmp_path):
    import ast
    import math
    namespace = {'Path': Path, 'sys': sys, 'math': math}
    exec(module.CONFIG, namespace)
    namespace.update(INPUT_ROOT=tmp_path, WORKING_ROOT=tmp_path,
                     OUTPUT_ROOT=tmp_path / 'evaluation', RUN_ID='test')
    tree = ast.parse(module.RUN)
    helpers = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    exec(compile(helpers, '<notebook-helpers>', 'exec'), namespace)
    return namespace


def test_evaluation_notebook_defaults_and_ast(evaluation_packager):
    metadata = {'source_git_commit': 'a' * 40, 'source_zip_sha256': 'b' * 64,
                'source_manifest_sha256': 'c' * 64, 'checker_sha256': 'd' * 64}
    notebook = evaluation_packager.render_notebook(
        metadata, (ROOT / 'notebooks/kaggle_unified_bootstrap.py').read_text(encoding='utf-8-sig'))
    assert notebook['nbformat'] == 4
    assert len(notebook['cells']) == 5
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')
            assert cell['execution_count'] is None and cell['outputs'] == []
    ns = {}
    exec(evaluation_packager.CONFIG, ns)
    assert ns['WARMUP_GAMES_PER_OPPONENT'] == 10
    assert ns['TARGET_GAMES_PER_OPPONENT'] == 100
    assert ns['AUTO_CONTINUE_AFTER_WARMUP'] is False
    assert ns['PROGRESS_BUNDLE'] == ''
    assert ns['EVAL_PROFILE'] == 'local_fast'
    assert '__COMMIT__' not in ''.join(notebook['cells'][2]['source'])
    assert 'input(' not in evaluation_packager.RUN
    assert 'GUANDAN_EVAL_' not in evaluation_packager.RUN  # no silently ignored env fallback


@pytest.mark.parametrize('expanded', [False, True])
def test_evaluation_config_supports_explicit_zip_or_expanded(evaluation_packager, tmp_path, expanded):
    ns = evaluation_namespace(evaluation_packager, tmp_path)
    path = tmp_path / ('expanded' if expanded else 'progress.zip')
    if expanded:
        (path / 'workers').mkdir(parents=True)
    else:
        path.write_bytes(b'zip validated by suite')
    ns['PROGRESS_BUNDLE'] = str(path)
    assert ns['_validate_config']() == path.resolve()
    payload = ns['_config_payload'](path)
    assert payload['warmup_total_games'] == 80
    assert payload['formal_total_games'] == 800


@pytest.mark.parametrize(('name', 'value'), [
    ('PROGRESS_BUNDLE', ''), ('WARMUP_GAMES_PER_OPPONENT', 0),
    ('TARGET_GAMES_PER_OPPONENT', 101), ('BATCH_SIZE', 0),
    ('MAX_ESTIMATED_MINUTES', float('nan')), ('MAX_ESTIMATED_MINUTES', float('inf')),
    ('AUTO_CONTINUE_AFTER_WARMUP', 'False'), ('EXECUTE', 'True'),
])
def test_evaluation_bad_config_fails_before_launch(evaluation_packager, tmp_path, name, value):
    ns = evaluation_namespace(evaluation_packager, tmp_path)
    path = tmp_path / 'progress.zip'
    path.touch()
    ns['PROGRESS_BUNDLE'] = str(path)
    ns[name] = value
    with pytest.raises(ValueError):
        ns['_validate_config']()


def test_evaluation_bundle_must_be_under_input_and_not_plain_directory(evaluation_packager, tmp_path):
    ns = evaluation_namespace(evaluation_packager, tmp_path)
    ns['PROGRESS_BUNDLE'] = str(tmp_path)
    with pytest.raises(ValueError, match='workers'):
        ns['_validate_config']()
    ns['INPUT_ROOT'] = tmp_path / 'input'
    ns['PROGRESS_BUNDLE'] = str(tmp_path / 'elsewhere.zip')
    with pytest.raises(ValueError, match='/kaggle/input'):
        ns['_validate_config']()


@pytest.mark.parametrize('execute', [False, True])
@pytest.mark.parametrize('auto_continue', [False, True])
def test_evaluation_flags_pass_exact_config(evaluation_packager, tmp_path, execute, auto_continue):
    ns = evaluation_namespace(evaluation_packager, tmp_path)
    ns.update(EXECUTE=execute, AUTO_CONTINUE_AFTER_WARMUP=auto_continue,
              TARGET_GAMES_PER_OPPONENT=120, WARMUP_GAMES_PER_OPPONENT=7,
              MAX_ESTIMATED_MINUTES=45, BATCH_SIZE=8, EVAL_PROFILE='remote_full')
    command = ns['_command'](tmp_path / 'progress.zip')
    for flag, value in [('--warmup-games-per-opponent', '7'), ('--target-games-per-opponent', '120'),
                        ('--max-estimated-minutes', '45'), ('--batch-size', '8'), ('--profile', 'remote_full')]:
        assert command[command.index(flag) + 1] == value
    assert ('--execute' in command) == execute
    assert ('--auto-continue-after-warmup' in command) == auto_continue
    assert 'guandan.evaluation_suite' in command
    assert 'guandan.experiment' not in command


def execute_evaluation_notebook(module, tmp_path, *, execute=False):
    ns = {'WORKING_ROOT': tmp_path, 'WORK_ROOT': ROOT}
    exec(module.CONFIG, ns)
    bundle = tmp_path / 'progress.zip'
    bundle.touch(exist_ok=True)
    ns.update(EXECUTE=execute, PROGRESS_BUNDLE=str(bundle))
    # Replace only the platform root in the test; all other generated code executes unchanged.
    code = module.RUN.replace("INPUT_ROOT = Path('/kaggle/input').resolve()",
                              f'INPUT_ROOT = Path({str(tmp_path)!r}).resolve()')
    return ns, code


def test_evaluation_plan_does_not_launch_and_uses_new_run_directory(evaluation_packager, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('plan must not launch an evaluation subprocess')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    ns, code = execute_evaluation_notebook(evaluation_packager, tmp_path)
    exec(code, ns)
    first = ns['OUTPUT_ROOT']
    assert first.name.startswith('guandan-evaluation-')
    assert json.loads((first / 'plan.json').read_text())['status'] == 'plan_only'
    assert not ns['RESULT_ZIP'].exists()
    exec(code, ns)
    assert ns['OUTPUT_ROOT'] != first
    assert (first / 'plan.json').is_file()


@pytest.mark.parametrize(('status', 'code'), [('warmup_complete', 0), ('stopped_by_budget', 2), ('complete', 0), ('failed', 1)])
def test_evaluation_exit_packs_logs_and_keeps_status(evaluation_packager, tmp_path, monkeypatch, capsys, status, code):
    import io
    import zipfile
    class Process:
        def __init__(self, command, **kwargs):
            self.stdout = io.StringIO('completed_games=10 ETA=30s\n')
            root = Path(command[command.index('--output-root') + 1])
            # The suite is allowed to require a previously absent output root.
            root.mkdir()
            (root / 'result.json').write_text(json.dumps({'status': status}))
            (root / 'games').mkdir()
            (root / 'games' / 'one.json').write_text('{"completed":true}')
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def wait(self, **kwargs): return code
    monkeypatch.setattr(subprocess, 'Popen', Process)
    ns, source = execute_evaluation_notebook(evaluation_packager, tmp_path, execute=True)
    if code == 1:
        with pytest.raises(RuntimeError, match='failed'):
            exec(source, ns)
    else:
        exec(source, ns)
    with zipfile.ZipFile(ns['RESULT_ZIP']) as archive:
        assert json.loads(archive.read('result.json'))['status'] == status
        assert 'notebook.log' in archive.namelist()
        assert 'games/one.json' in archive.namelist()
        assert 'evaluation_config.json' in archive.namelist()
    output = capsys.readouterr().out
    if status == 'warmup_complete':
        assert '尚未运行 formal' in output
    if status == 'stopped_by_budget':
        assert '预算保护停止' in output


def test_evaluation_package_build_includes_committed_tutorial(evaluation_packager, tmp_path, monkeypatch):
    # Source fixture is committed in an isolated repository; production tree is not committed/packaged here.
    root = package_fixture_repo(tmp_path)
    for name in ('docs/kaggle_evaluation_tutorial.md', 'scripts/prepare_kaggle_evaluation.py'):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    git_fixture(root, 'add', '.')
    git_fixture(root, 'commit', '-qm', 'evaluation delivery fixture')
    monkeypatch.setattr(evaluation_packager, 'ROOT', root)
    out = tmp_path / 'release-fixture'
    metadata = evaluation_packager.build(out)
    notebook = json.loads((out / 'guandan_evaluation.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')
    tutorial = (root / 'docs/kaggle_evaluation_tutorial.md').read_text(encoding='utf-8-sig')
    assert (out / '使用说明.txt').read_text(encoding='utf-8') == tutorial
    assert (out / 'kaggle_evaluation_tutorial.md').read_text(encoding='utf-8') == tutorial
    assert metadata['source_zip_sha256'] == hashlib.sha256((out / 'guandan-source.zip').read_bytes()).hexdigest()
    assert metadata['uploaded'] is metadata['accepted'] is metadata['kaggle_verified'] is False


def test_evaluation_interruption_preserves_already_written_games(evaluation_packager, tmp_path, monkeypatch):
    import zipfile
    state = {'terminated': False}
    class InterruptedOutput:
        def __iter__(self): return self
        def __next__(self): raise KeyboardInterrupt()
    class Process:
        def __init__(self, command, **kwargs):
            self.stdout = InterruptedOutput()
            root = Path(command[command.index('--output-root') + 1])
            (root / 'games').mkdir(parents=True)
            (root / 'games' / 'completed.json').write_text('{}')
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def terminate(self): state['terminated'] = True
        def wait(self, **kwargs): return -15
    monkeypatch.setattr(subprocess, 'Popen', Process)
    ns, code = execute_evaluation_notebook(evaluation_packager, tmp_path, execute=True)
    with pytest.raises(KeyboardInterrupt):
        exec(code, ns)
    assert state['terminated'] is True
    with zipfile.ZipFile(ns['RESULT_ZIP']) as archive:
        assert 'games/completed.json' in archive.namelist()
        assert 'notebook.log' in archive.namelist()
        assert 'result.json' not in archive.namelist()  # never invent a completed result


def test_evaluation_warmup_and_formal_counts_are_independent(evaluation_packager, tmp_path):
    ns = evaluation_namespace(evaluation_packager, tmp_path)
    bundle = tmp_path / 'progress.zip'
    bundle.touch()
    ns.update(PROGRESS_BUNDLE=str(bundle), WARMUP_GAMES_PER_OPPONENT=10, TARGET_GAMES_PER_OPPONENT=4)
    assert ns['_validate_config']() == bundle.resolve()
    assert ns['_config_payload'](bundle)['formal_total_games'] == 32


def test_evaluation_plan_without_progress_does_not_launch(evaluation_packager,tmp_path,monkeypatch):
    ns,source=execute_evaluation_notebook(evaluation_packager,tmp_path,execute=False)
    ns['PROGRESS_BUNDLE']=''
    def forbidden(*args,**kwargs):
        raise AssertionError('plan must not launch child')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    exec(source,ns)
    assert json.loads((ns['OUTPUT_ROOT']/'plan.json').read_text(encoding='utf-8'))['status']=='plan_only'
