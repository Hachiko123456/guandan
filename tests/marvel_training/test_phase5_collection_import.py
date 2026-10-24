"""Phase-5 import preflight: syntax errors must fail before any GPU collection."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PHASE5_SOURCES = sorted((ROOT / "guandan_marvel" / "phase5").glob("*.py"))


@pytest.mark.parametrize("source", PHASE5_SOURCES, ids=lambda path: path.name)
def test_phase5_sources_compile_without_starting_collection(source):
    # Compile in memory: no dataset collection, model allocation or cache writes.
    # The old joined append/del statement fails this exact preflight.
    compile(source.read_text(encoding="utf-8-sig"), str(source), "exec")


def test_belief_collector_imports_before_loading_any_checkpoint():
    module = importlib.import_module("guandan_marvel.phase5.belief_collect")
    assert callable(module.collect_belief_dataset)


def test_phase5_declared_public_exports_all_exist():
    package = importlib.import_module("guandan_marvel.phase5")
    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert missing == []
    namespace = {}
    exec("from guandan_marvel.phase5 import *", namespace)
    assert namespace["transform_candidate_rows"] is package.transform_candidate_rows
