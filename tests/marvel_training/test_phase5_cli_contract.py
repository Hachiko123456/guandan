"""Phase-5 CLI budgets and output boundaries are fail-closed."""
from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B0 = _load("run_phase5_b0")
SYMMETRY = _load("run_phase5_symmetry_diagnostic")


def _unique_b0_output() -> str:
    return str(ROOT / "runs" / "phase5" / "b0" / f"contract-{uuid.uuid4().hex}.json")


def _unique_symmetry_output() -> str:
    return str(ROOT / "runs" / "phase5" / "symmetry" / f"contract-{uuid.uuid4().hex}.json")


def test_b0_parser_defaults_are_the_pre_registered_budget():
    parser = B0.build_parser()
    args = parser.parse_args(["--output", _unique_b0_output()])
    assert args.deals == 512 and args.batch_slots == 16
    assert args.max_collection_seconds == 1800
    assert args.max_probe_seconds == 1200
    assert args.seeds == [0, 1, 2]
    assert args.epochs == 40 and args.batch_size == 256 and args.patience == 6
    B0.validate_args(parser, args)


def test_b0_parser_rejects_unbounded_or_unsafe_budgets():
    parser = B0.build_parser()
    invalid = [
        ["--deals", "513"],
        ["--batch-slots", "17"],
        ["--max-macro-starts-per-deal", "9"],
        ["--max-collection-seconds", "1801"],
        ["--max-probe-seconds", "1201"],
        ["--epochs", "41"],
        ["--batch-size", "257"],
        ["--patience", "7"],
        ["--seeds", "0,1"],
        ["--seeds", "0,0,1"],
        ["--epsilon", "-0.1"],
        ["--device", "cpu"],
        ["--output", str(ROOT / "runs" / "phase5" / "pilot" / "escape.json")],
    ]
    for arguments in invalid:
        command = list(arguments)
        if "--output" not in command:
            command += ["--output", _unique_b0_output()]
        with pytest.raises(SystemExit):
            parsed = parser.parse_args(command)
            B0.validate_args(parser, parsed)


def test_b0_refuses_to_overwrite_existing_evidence(tmp_path):
    parser = B0.build_parser()
    existing = ROOT / "runs" / "phase5" / "b0" / f"contract-{uuid.uuid4().hex}.json"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("{}", encoding="utf-8")
    try:
        args = parser.parse_args(["--output", str(existing)])
        with pytest.raises(SystemExit):
            B0.validate_args(parser, args)
    finally:
        existing.unlink(missing_ok=True)


def test_symmetry_parser_rejects_unbounded_budgets():
    parser = SYMMETRY.build_parser()
    args = parser.parse_args(["--output", _unique_symmetry_output()])
    assert args.deals == 16
    assert args.max_macro_starts_per_deal == 24
    assert args.max_wall_seconds == 240
    SYMMETRY.validate_args(parser, args)
    invalid = [
        ["--deals", "17"],
        ["--max-macro-starts-per-deal", "25"],
        ["--batch-size", "65"],
        ["--max-wall-seconds", "301"],
        ["--device", "cpu"],
        ["--output", str(ROOT / "runs" / "phase5" / "b0" / "escape.json")],
    ]
    for arguments in invalid:
        command = list(arguments)
        if "--output" not in command:
            command += ["--output", _unique_symmetry_output()]
        with pytest.raises(SystemExit):
            parsed = parser.parse_args(command)
            SYMMETRY.validate_args(parser, parsed)


def _guandan_marvel_imports(path: Path):
    """Every ``(module, symbol)`` an ``ImportFrom`` at ``path`` pulls in.

    Only absolute imports rooted at ``guandan_marvel`` are collected; ``from
    __future__ import ...``, third-party imports and star imports are skipped.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pairs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 0:
            continue
        module = node.module or ""
        if not module.startswith("guandan_marvel"):
            continue
        for alias in node.names:
            if alias.name != "*":
                pairs.append((module, alias.name))
    return pairs


def test_phase5_clis_only_import_symbols_their_modules_actually_export():
    # The CLI imports live inside ``main()`` (so loading the module never runs
    # them), which is exactly why a name that does not exist in its target
    # module -- ``require_cuda_device`` from ``diagnostic_runner`` -- survived
    # until the first real GPU run.  Resolving every guandan_marvel ImportFrom
    # against the real module catches that class of defect without executing
    # main(), any engine, or any CUDA code.
    scripts = sorted((ROOT / "scripts").glob("run_phase5_*.py"))
    assert scripts, "no phase-5 CLI scripts found under scripts/"
    checked = 0
    for script in scripts:
        for module_name, symbol in _guandan_marvel_imports(script):
            module = importlib.import_module(module_name)
            assert hasattr(module, symbol), (
                f"{script.name} imports {symbol!r} from {module_name!r}, which "
                f"does not provide it"
            )
            checked += 1
    assert checked > 0, "expected the phase-5 CLIs to import guandan_marvel symbols"


def test_import_resolver_flags_the_historic_dangling_guandan_marvel_import(tmp_path):
    # The pre-fix symmetry CLI contained exactly this import and
    # ``diagnostic_runner`` never provided ``require_cuda_device``.  Running the
    # same resolver used above on that historic source proves the general check
    # fails on the old script instead of being a tautology.
    historic = tmp_path / "run_phase5_symmetry_diagnostic_old.py"
    historic.write_text(
        "from guandan_marvel.phase5.diagnostic_runner import (\n"
        "    require_cuda_device, run_symmetry_diagnostic,\n"
        ")\n",
        encoding="utf-8",
    )
    dangling = [
        (module_name, symbol)
        for module_name, symbol in _guandan_marvel_imports(historic)
        if not hasattr(importlib.import_module(module_name), symbol)
    ]
    assert dangling == [
        ("guandan_marvel.phase5.diagnostic_runner", "require_cuda_device"),
    ]
