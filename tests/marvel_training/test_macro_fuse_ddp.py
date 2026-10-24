"""Phase-3 actor-fuse DDP invariants under two-rank CPU gloo."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.marvel_training._ddp_launcher import launch_two_ranks
from tests.marvel_training._fuse_ddp_worker import PROBE_PARAMS, TRUNK_PROBES

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests" / "marvel_training" / "_fuse_ddp_worker.py"


@pytest.mark.slow
def test_two_rank_fuse_ddp_six_invariants(tmp_path: Path):
    out = tmp_path / "fuse_ddp.json"
    results = launch_two_ranks(
        WORKER, "--mode", "fuse_ddp", "--out", str(out), cwd=ROOT, timeout=1800,
    )
    for result in results:
        assert result.returncode == 0, result.stdout
    payloads = [
        json.loads((tmp_path / f"fuse_ddp.json.rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    for payload in payloads:
        assert payload["world_size"] == 2
        assert payload["find_unused_parameters"] is False
        assert payload["amp_enabled"] is False
        assert payload["scenario"] == "fuse_ddp"

    # (1) identical new-parameter initialisation on every rank.
    assert payloads[0]["init_fuse_hash"] == payloads[1]["init_fuse_hash"]
    assert payloads[0]["init_probe_digest"] == payloads[1]["init_probe_digest"]
    # The migration precondition: the residual gate starts at strict zero.
    assert payloads[0]["init_output_proj_absmax"] == 0.0
    assert payloads[1]["init_output_proj_absmax"] == 0.0

    # The two ranks really hold different probe data (otherwise an all-reduce is
    # indistinguishable from a local-only backward).
    assert (
        payloads[0]["local_grad"]["actor_fuse.output_proj.weight"]["hash"]
        != payloads[1]["local_grad"]["actor_fuse.output_proj.weight"]["hash"]
    )

    # First-backward gate: W_out gradient non-zero, trunk exactly zero.
    for payload in payloads:
        local_gate = payload["local_grad"]["actor_fuse.output_proj.weight"]
        assert local_gate["present"] and local_gate["absmax"] > 0.0
        for name in TRUNK_PROBES:
            assert payload["local_grad"][name]["present"]
            assert payload["local_grad"][name]["absmax"] == 0.0, name

    # (2) the first backward W_out gradient is synchronized (and equals the mean
    # of the two local gradients).
    first_gate = [payload["first_post_grad"]["actor_fuse.output_proj.weight"] for payload in payloads]
    assert first_gate[0]["hash"] == first_gate[1]["hash"]
    assert first_gate[0]["absmax"] > 0.0
    for name in PROBE_PARAMS:
        left = payloads[0]["first_post_grad"][name]
        right = payloads[1]["first_post_grad"][name]
        assert left["hash"] == right["hash"], name
        local_mean_sum = 0.5 * (
            payloads[0]["local_grad"][name]["sum"] + payloads[1]["local_grad"][name]["sum"]
        )
        assert left["sum"] == pytest.approx(local_mean_sum, rel=1e-5, abs=1e-12), name
    # The trunk is still exactly zero after the first DDP backward.
    for payload in payloads:
        for name in TRUNK_PROBES:
            assert payload["first_post_grad"][name]["absmax"] == 0.0, name

    # (3) after one optimizer step the parameter hashes agree across ranks.
    assert payloads[0]["post_step_fuse_hash"] == payloads[1]["post_step_fuse_hash"]
    assert payloads[0]["post_step_probe_digest"] == payloads[1]["post_step_probe_digest"]
    assert payloads[0]["output_proj_nonzero_after_step"] > 0
    assert payloads[1]["output_proj_nonzero_after_step"] > 0

    # (4) the second backward produces non-zero trunk gradients, synced across ranks.
    for name in TRUNK_PROBES:
        left = payloads[0]["second_post_grad"][name]
        right = payloads[1]["second_post_grad"][name]
        assert left["hash"] == right["hash"], name
        assert left["absmax"] > 0.0, name
    assert (
        payloads[0]["second_post_grad"]["actor_fuse.input_proj.weight"]["absmax"] > 0.0
    )

    # (5) a full production iteration ran with find_unused_parameters=False.
    for payload in payloads:
        assert payload["ddp_enabled"] is True
        assert payload["iteration"] == 1
        assert payload["durable_updates"] == 1
        assert payload["profile"] == "macro_decision_v1"
        assert math.isfinite(payload["grad_norm"]) and payload["grad_norm"] > 0.0
        assert payload["macro_count_global"] >= 1.0
    assert payloads[0]["macro_count_global"] == payloads[1]["macro_count_global"]
    assert payloads[0]["rank"] == 0 and payloads[1]["rank"] == 1
