"""Prune execution remains explicit and scoped to a verified local plan."""
from pathlib import Path
import shutil

import pytest

from tools.kaggle_pipeline import core, delivery_registry


def test_prune_preview_never_removes_and_execute_removes_exact_candidates(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    parent = repo / "runs/deliveries"
    pipeline = repo / "runs/pipeline"
    current = parent / "delivery_v3"
    old = parent / "delivery_v1"
    old_build = parent / "delivery_v1_build"
    for path, data in ((current, b"current"), (old, b"old"), (old_build, b"build")):
        path.mkdir(parents=True)
        (path / "data.bin").write_bytes(data)
    pipeline.mkdir(parents=True)
    config = {
        "delivery_parent": str(parent), "output_root": str(pipeline),
        "keep_previous_deliveries": 0,
    }
    current_record = {"delivery_dir": "runs/deliveries/delivery_v3"}
    monkeypatch.setattr(core, "ROOT", repo)
    monkeypatch.setattr(core, "current_delivery", lambda _config: (current, current_record))

    preview = core.prune_deliveries(config, execute=False)
    assert preview["executed"] is False and preview["remote_resources_deleted"] is False
    assert old.is_dir() and old_build.is_dir() and current.is_dir()
    assert {row["relative_path"] for row in preview["candidates"]} == {
        "runs/deliveries/delivery_v1", "runs/deliveries/delivery_v1_build",
    }

    result = core.prune_deliveries(config, execute=True)
    assert result["executed"] is True and result["recoverable"] is False
    assert result["removed_bytes"] == len(b"old") + len(b"build")
    assert not old.exists() and not old_build.exists() and current.is_dir()


def test_prune_stops_if_candidate_changes_after_planning(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    parent = repo / "runs/deliveries"
    current = parent / "delivery_v2"
    old = parent / "delivery_v1"
    current.mkdir(parents=True)
    old.mkdir()
    config = {"delivery_parent": str(parent), "output_root": str(repo / "runs/pipeline"),
              "keep_previous_deliveries": 0}
    monkeypatch.setattr(core, "ROOT", repo)
    monkeypatch.setattr(core, "current_delivery", lambda _config: (
        current, {"delivery_dir": "runs/deliveries/delivery_v2"},
    ))
    real_plan = delivery_registry.cleanup_plan
    def changed_plan(*args, **kwargs):
        plan = real_plan(*args, **kwargs)
        shutil.rmtree(old)
        old.write_text("changed target", encoding="utf-8")
        return plan
    monkeypatch.setattr(delivery_registry, "cleanup_plan", changed_plan)
    with pytest.raises(core.PipelineError, match="changed after planning"):
        core.prune_deliveries(config, execute=True)
    assert current.is_dir()
