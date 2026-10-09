"""Verify M05 backend imports from a clean isolated wheel installation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_python(code: str, *, cwd: Path, pythonpath: Path) -> dict[str, object]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pythonpath)
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return {
        "command": [sys.executable, "-c", code],
        "cwd": str(cwd), "pythonpath": str(pythonpath),
        "exit_code": process.returncode,
        "stdout": process.stdout, "stderr": process.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    wheel = Path(args.wheel).resolve()
    run_dir = Path(args.run_dir).resolve()
    install_dir = run_dir / "isolated-wheel"
    no_native_dir = run_dir / "isolated-no-native"
    outside = run_dir / "outside-cwd"
    for directory in (install_dir, no_native_dir, outside):
        directory.mkdir(parents=True, exist_ok=True)
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(install_dir), str(wheel)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    scenarios = []
    scenarios.append(run_python(
        "import guandan_marvel as g; "
        "assert g.game.BACKEND=='python_reference'; "
        "assert g.default_game.BACKEND=='cpp'; assert hasattr(g.default_game,'EnvBatch'); "
        "print(g.__file__); print(g.default_game.get_backend_build_identity()['extension_path'])",
        cwd=outside, pythonpath=install_dir,
    ))
    scenarios.append(run_python(
        "from nanopokerlib import guandan as n; import guandan_marvel as g; "
        "assert hasattr(n,'EnvBatch'); assert hasattr(g.get_game_backend(),'EnvBatch'); "
        "print(n.__file__)",
        cwd=outside, pythonpath=install_dir,
    ))
    scenarios.append(run_python(
        "import guandan_marvel.config,sys; "
        "assert 'nanopokerlib._guandan_cpp' not in sys.modules; print('config-only-ok')",
        cwd=outside, pythonpath=install_dir,
    ))

    for name in ("guandan", "guandan_marvel", "nanopokerlib"):
        source = install_dir / name
        target = no_native_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("_guandan_cpp*.pyd", "_guandan_cpp*.so"))
    scenarios.append(run_python(
        "import guandan_marvel as g; p=g.get_game_backend('python_reference'); "
        "assert p.BACKEND=='python_reference' and hasattr(p,'EnvBatch')\n"
        "try: g.get_game_backend('cpp')\n"
        "except g.BackendUnavailableError: print('no-native-ok')\n"
        "else: raise AssertionError('silent fallback')",
        cwd=outside, pythonpath=no_native_dir,
    ))
    valid = install.returncode == 0 and all(item["exit_code"] == 0 for item in scenarios)
    result = {
        "schema": "guandan-marvel-m05-isolated-wheel-imports-v1",
        "wheel": {"path": str(wheel), "bytes": wheel.stat().st_size, "sha256": sha256(wheel)},
        "install": {"exit_code": install.returncode, "stdout": install.stdout, "stderr": install.stderr},
        "scenarios": scenarios,
        "valid": valid,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": valid, "wheel": str(wheel), "output": str(output)}, sort_keys=True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
