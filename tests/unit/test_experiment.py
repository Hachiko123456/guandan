import json
from pathlib import Path

import pytest

from guandan.experiment import build_plan, run


def test_plan_is_explicit_and_never_executes(tmp_path):
    result = run(profile="local_fast", output_root=tmp_path / "plan")
    assert result["status"] == "plan_only"
    assert result["workers"]["ippo"]["device"] == "cuda:0"
    assert result["workers"]["vrpo"]["device"] == "cuda:1"
    assert result["accepted"] is False
    assert json.loads((tmp_path / "plan" / "plan.json").read_text())['format'] == 'guandan-dual-gpu-plan-v1'


def test_plan_has_distinct_seeds_and_profile_counts(tmp_path):
    plan = build_plan(profile="local_fast", output_root=tmp_path, base_updates=1, resume_updates=1)
    assert plan["workers"]["ippo"]["seed"] != plan["workers"]["vrpo"]["seed"]
    assert plan["workers"]["ippo"]["base_updates"] == 1
    assert plan["workers"]["ippo"]["resume_updates"] == 1
    assert plan["dual_gpu_required"] is True
