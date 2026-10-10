from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import pytest

from guandan.cards import Rank
from guandan_marvel.kaggle.evaluation import run_smoke_evaluation, run_smoke_evaluation_isolated


def _rng_hashes() -> dict[str, str]:
    return {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }


def test_eval_only_real_cpp_entry_uses_rank_two(m07_cpp_evaluation_checkpoint, tmp_path):
    result = run_smoke_evaluation(
        m07_cpp_evaluation_checkpoint,
        output_root=tmp_path / "eval-only",
        base_deals=1,
        seed_base=887700,
        device="cpu",
        opponents=("random",),
        seat_rotations=(0,),
    )
    assert result["status"] == "complete"
    assert result["level_ranks"] == [int(Rank.TWO)]
    assert result["games"] == result["complete_games"] == 1
    assert result["illegal_count"] == result["truncated_count"] == 0


def test_spawned_real_cpp_evaluation_preserves_parent_rng(m07_cpp_evaluation_checkpoint, tmp_path):
    before = _rng_hashes()
    result = run_smoke_evaluation_isolated(
        m07_cpp_evaluation_checkpoint,
        output_root=tmp_path / "isolated",
        base_deals=1,
        seed_base=887710,
        device="cpu",
        opponents=("random",),
        seat_rotations=(0,),
        timeout_seconds=120,
    )
    after = _rng_hashes()
    assert result["status"] == "complete"
    assert before == after
    assert (tmp_path / "isolated" / "evaluation_process_result.json").is_file()

def test_spawned_real_loader_failure_preserves_parent_rng(m07_cpp_evaluation_checkpoint, tmp_path):
    before = _rng_hashes()
    with pytest.raises(RuntimeError, match="isolated evaluation failed"):
        run_smoke_evaluation_isolated(
            m07_cpp_evaluation_checkpoint,
            output_root=tmp_path / "isolated-failure",
            base_deals=1,
            seed_base=887720,
            device="cpu",
            opponents=("invalid-opponent",),
            seat_rotations=(0,),
            timeout_seconds=120,
        )
    assert before == _rng_hashes()
