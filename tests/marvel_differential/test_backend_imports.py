from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def _run(code: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd or ROOT), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def test_import_guandan_marvel_then_cpp_backend_is_complete():
    result = _run(
        "import guandan_marvel as g; "
        "assert g.game.BACKEND == 'python_reference'; "
        "assert g.default_game.BACKEND == 'cpp'; "
        "assert hasattr(g.default_game, 'EnvBatch'); "
        "assert g.get_game_backend().EnvBatch is g.default_game.EnvBatch; "
        "print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_import_native_first_then_default_backend_is_complete():
    result = _run(
        "from nanopokerlib import guandan as n; import guandan_marvel as g; "
        "assert hasattr(n, 'EnvBatch'); "
        "assert hasattr(g.default_game, 'EnvBatch'); "
        "assert hasattr(g.get_game_backend(), 'EnvBatch'); "
        "print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_public_module_identity_and_historical_import_contract():
    result = _run(
        "import importlib, guandan_marvel as g; "
        "h=importlib.import_module('guandan_marvel.game'); "
        "p=importlib.import_module('guandan_marvel.python_game'); "
        "c=importlib.import_module('guandan_marvel.cpp_game'); "
        "assert g.game is h; assert h.EnvBatch is p.EnvBatch; "
        "assert g.python_game is p; assert g.cpp_game is c; "
        "assert g.default_game is c; print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_config_import_does_not_load_native_extension():
    result = _run(
        "import sys; import guandan_marvel.config; "
        "assert 'nanopokerlib._guandan_cpp' not in sys.modules; "
        "assert 'guandan_marvel.cpp_game' not in sys.modules; print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_missing_extension_keeps_explicit_python_oracle_and_rejects_cpp():
    code = r'''
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "_guandan_cpp" or name.endswith("._guandan_cpp"):
        raise ImportError("blocked native extension")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import guandan_marvel as g
python_backend = g.get_game_backend("python_reference")
assert python_backend.BACKEND == "python_reference"
assert hasattr(python_backend, "EnvBatch")
try:
    g.get_game_backend("cpp")
except g.BackendUnavailableError:
    print("ok")
else:
    raise AssertionError("cpp request silently fell back")
'''
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
