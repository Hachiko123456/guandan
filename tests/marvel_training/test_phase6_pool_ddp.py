"""Phase-6 CPU two-rank gloo: empty current-macro rank uses a connected zero."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tests.marvel_training._ddp_launcher import launch_two_ranks
from tests.marvel_training._phase6_pool_ddp_worker import (
    build_case, scaled_log_probs,
)
from guandan_marvel.training.macro_ppo import build_macro_batch_view, compute_macro_pg_loss

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests" / "marvel_training" / "_phase6_pool_ddp_worker.py"


@pytest.mark.slow
def test_gloo_mixed_one_and_two_current_macros_match_single_process(tmp_path: Path):
    output = tmp_path / "mixed.json"
    results = launch_two_ranks(
        WORKER, "--mode", "mixed", "--out", str(output), cwd=ROOT, timeout=600,
    )
    for result in results:
        assert result.returncode == 0, result.stdout
    payloads = [
        json.loads((tmp_path / f"mixed.json.rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert [payload["local_trainable_macros"] for payload in payloads] == [1, 2]
    assert [payload["global_trainable_macros"] for payload in payloads] == [3, 3]
    reference = _mixed_reference()
    averaged_grad = sum(payload["grad"] for payload in payloads) / 2.0
    assert averaged_grad == pytest.approx(reference["grad"], rel=1e-5, abs=1e-7)
    averaged_loss = sum(payload["loss"] for payload in payloads) / 2.0
    assert averaged_loss == pytest.approx(reference["loss"], rel=1e-5, abs=1e-7)


def _mixed_reference():
    from guandan_marvel.model.tensordict import TensorDict
    from guandan_marvel.phase6.policy_owner import segment_pooled_macros
    from guandan_marvel.training.macro_ppo import (
        build_macro_batch_view, compute_macro_pg_loss,
    )
    from tests.marvel_training._phase6_pool_ddp_worker import build_case, scaled_log_probs

    cases = [build_case(0, "mixed"), build_case(1, "mixed")]
    length = max(case[0]["player_ids"].shape[1] for case in cases)
    fields = {}
    for name in cases[0][0].keys():
        tensors = []
        for rollout, _seg, _prob, _train in cases:
            tensor = rollout[name]
            if tensor.dim() == 1:
                tensors.append(tensor)
                continue
            if tensor.shape[1] < length:
                padding = torch.zeros(
                    (tensor.shape[0], length - tensor.shape[1], *tensor.shape[2:]),
                    dtype=tensor.dtype,
                )
                tensor = torch.cat([tensor, padding], dim=1)
            tensors.append(tensor)
        fields[name] = torch.cat(tensors, dim=0)
    # Padding must not claim ownership or trainability.
    time = torch.arange(length).unsqueeze(0)
    active = time < fields["num_steps"].unsqueeze(1)
    for name in (
        "policy_owner", "opponent_checkpoint_id", "opponent_policy_version",
        "assignment_id",
    ):
        fields[name] = torch.where(active, fields[name], torch.full_like(fields[name], -1))
    fields["assigned_seats"] = torch.where(
        active.unsqueeze(-1), fields["assigned_seats"],
        torch.full_like(fields["assigned_seats"], -1),
    )
    fields["policy_train_mask"] = fields["policy_train_mask"] & active
    fields["on_policy_valid"] = fields["on_policy_valid"] & active
    stacked = TensorDict(fields, batch_size=[len(cases)])
    probabilities = []
    for _rollout, _seg, prob, _train in cases:
        probabilities.append(torch.cat([
            prob,
            torch.zeros((1, length - prob.shape[1], prob.shape[2])),
        ], dim=1))
    probabilities = torch.cat(probabilities, dim=0)
    segmentation = segment_pooled_macros(stacked)
    view = build_macro_batch_view(segmentation, torch.arange(2))
    advantage = torch.zeros((2, segmentation.max_macros))
    for batch in range(2):
        sign = 1.0
        for index in range(segmentation.max_macros):
            if bool(segmentation.macro_train_mask[batch, index]):
                advantage[batch, index] = sign * (index + 1.0)
                sign = -sign
    w = torch.tensor(0.7, requires_grad=True)
    log_probs = scaled_log_probs(stacked, probabilities, w)
    pg = compute_macro_pg_loss(
        stacked, view=view, avail_log_probs_new=log_probs,
        macro_advantage=advantage, clip_coef=0.2, world_size=1,
    )
    pg.pg_loss_mean.backward()
    assert int(pg.macro_count_global) == 3
    return {"loss": float(pg.pg_loss_mean.detach().item()), "grad": float(w.grad.item())}


def _reference():
    rollout, segmentation, probabilities, train = build_case(1)
    view = build_macro_batch_view(segmentation, torch.arange(segmentation.batch_size))
    trainable = segmentation.macro_train_mask
    advantage = torch.zeros((segmentation.batch_size, segmentation.max_macros))
    sign = 1.0
    for index in range(segmentation.max_macros):
        if bool(trainable[0, index]):
            advantage[0, index] = sign * (index + 1.0)
            sign = -sign
    w = torch.tensor(0.7, requires_grad=True)
    log_probs = scaled_log_probs(rollout, probabilities, w)
    pg = compute_macro_pg_loss(
        rollout, view=view, avail_log_probs_new=log_probs,
        macro_advantage=advantage, clip_coef=0.2, world_size=1,
    )
    pg.pg_loss_mean.backward()
    return {
        "loss": float(pg.pg_loss_mean.detach().item()),
        "grad": float(w.grad.item()),
        "global": int(pg.macro_count_global),
        "local": int(pg.macro_count_local),
        "current_tokens": int(train.sum().item()),
    }


@pytest.mark.slow
def test_gloo_empty_current_rank_zero_anchor_and_global_weighting(tmp_path: Path):
    output = tmp_path / "phase6.json"
    results = launch_two_ranks(
        WORKER, "--out", str(output), cwd=ROOT, timeout=600,
    )
    for result in results:
        assert result.returncode == 0, result.stdout
    payloads = [
        json.loads((tmp_path / f"phase6.json.rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    reference = _reference()
    assert payloads[0]["local_current_tokens"] == 0
    assert payloads[1]["local_current_tokens"] == 2
    assert payloads[0]["local_active_macros"] == 2
    assert [payload["local_trainable_macros"] for payload in payloads] == [0, 2]
    assert [payload["global_trainable_macros"] for payload in payloads] == [2, 2]
    assert sum(payload["local_trainable_macros"] for payload in payloads) == 2
    assert payloads[0]["empty_local_batch"] is True
    assert payloads[1]["empty_local_batch"] is False
    assert payloads[0]["finite"] and payloads[1]["finite"]
    assert payloads[0]["loss"] == 0.0
    assert payloads[0]["grad"] == 0.0
    assert reference["global"] == 2 and reference["local"] == 2
    # DDP averages the two rank gradients; rank 0 contributes a connected zero.
    averaged_grad = sum(payload["grad"] for payload in payloads) / 2.0
    assert averaged_grad == pytest.approx(reference["grad"], rel=1e-5, abs=1e-7)
    averaged_loss = sum(payload["loss"] for payload in payloads) / 2.0
    assert averaged_loss == pytest.approx(reference["loss"], rel=1e-5, abs=1e-7)
    assert not any(
        torch.isnan(torch.tensor(payload["loss"])) for payload in payloads
    )
