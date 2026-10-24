"""Phase-3 repair: ResGatedMLP RMSNorm must compute FP32 statistics under AMP.

``torch.nn.RMSNorm`` accumulates the variance in FP32 but stores its result in the
*input* dtype, so under ``torch.autocast`` a bf16 activation leaves the module
bf16 and the FP32 statistics are rounded away at the module boundary.  The
residual-MLP norm used by ``state_mlp`` / ``action_mlp`` / ``actor_fuse.trunk``
must instead upcast and emit an FP32 activation, exactly like
``TransformerRMSNorm``.

Reproduce the defect (pre-fix)::

    python -m pytest tests/marvel_training/test_macro_fuse_norm_precision.py \
        -p tests.marvel_training._rmsnorm_pre_fix_plugin -q
"""
from __future__ import annotations

import math
import time

import pytest
import torch

from guandan_marvel.model.base import RMSNorm
from guandan_marvel.model.transformer import TransformerRMSNorm
from guandan_marvel.training.local import IPPOTrainer
from tests.marvel_training._fuse_fixtures import (
    FUSE_ARCH, LEGACY_ARCH, capture_logits, deterministic_agent, migrate_legacy_into_fuse,
    smoke_trainer_config, tiny_config, tiny_rollout,
)

NORM_GROUPS = ("state_mlp", "action_mlp", "actor_fuse")


def _fused_model_and_rollout():
    torch.manual_seed(101)
    model = deterministic_agent(tiny_config(FUSE_ARCH, fuse_layers=2), seed=103).eval()
    return model, tiny_rollout(batch=2, steps=3)


def _capture_norms(model, rollout, *, norm_type):
    """Run the model under bf16 autocast and capture every norm invocation."""
    records: list[dict] = []
    handles = []

    def make_hook(name):
        def _hook(module, inputs, output):
            raw_eps = getattr(module, "eps", None)
            records.append({
                "name": name,
                "x": inputs[0].detach().clone(),
                "out": output.detach().clone(),
                "weight": module.weight.detach().clone(),
                # The required semantics: the statistics are FP32, so a None eps
                # (torch.nn.RMSNorm's default) means finfo(float32).eps.
                "eps": torch.finfo(torch.float32).eps if raw_eps is None else float(raw_eps),
            })
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, norm_type):
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model(rollout=rollout)
    finally:
        for handle in handles:
            handle.remove()
    return records


def _fp32_statistics_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Independent reference: variance and normalisation evaluated in FP32.

    Written with explicit torch ops in the spec's formula (upcast, mean of
    squares, ``rsqrt``, cast back to the input dtype, then multiply by the FP32
    weight) - it never calls the production norm.
    """
    upcast = x.to(torch.float32)
    variance = upcast.pow(2).mean(dim=-1, keepdim=True)
    normalised = upcast * torch.rsqrt(variance + eps)
    return weight * normalised.to(x.dtype)


def _reduced_precision_statistics_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The forbidden variant: the mean of squares is carried in the input dtype."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


# ---------------------------------------------------------------------------
# Decisive counterexample
# ---------------------------------------------------------------------------


def test_resgated_norm_upcasts_and_emits_fp32_statistics_under_autocast():
    model, rollout = _fused_model_and_rollout()
    records = _capture_norms(model, rollout, norm_type=RMSNorm)
    assert records, "the fused model has no ResGatedMLP norms to inspect"
    groups = {record["name"].split(".")[0] for record in records}
    assert groups >= set(NORM_GROUPS), groups

    bf16_reference_hits = 0
    for record in records:
        x, out, weight, eps = record["x"], record["out"], record["weight"], record["eps"]
        # Precondition: autocast really lowered the activation into bf16.
        assert x.dtype is torch.bfloat16, record["name"]
        assert weight.dtype is torch.float32, record["name"]

        # (1) The FP32 statistics survive the module: it emits FP32, not bf16.
        assert out.dtype is torch.float32, (
            f"{record['name']} emitted {out.dtype}; the FP32 statistics were rounded "
            "away at the module boundary"
        )
        # (2) Bitwise equal to an independently written FP32-statistics reference.
        reference = _fp32_statistics_reference(x, weight, eps)
        assert torch.equal(out, reference), record["name"]

        # (3) It is *not* a reduced-precision-statistics result.
        low = _reduced_precision_statistics_reference(x, weight, eps)
        if not torch.equal(out.double(), low.double()):
            bf16_reference_hits += 1
    assert bf16_reference_hits > 0, "the norms accidentally agree with a bf16-statistics result"

    # The transformer norm is the behavioural reference this fix matches: it emits
    # FP32 for both reduced-precision and FP32 activations.
    reference_norm = TransformerRMSNorm(24)
    assert reference_norm(torch.randn(3, 24).to(torch.bfloat16)).dtype is torch.float32
    assert reference_norm(torch.randn(3, 24)).dtype is torch.float32

    transformer_records = _capture_norms(model, rollout, norm_type=TransformerRMSNorm)
    assert transformer_records
    assert all(record["out"].dtype is torch.float32 for record in transformer_records)


def test_norm_output_is_closer_to_the_fp64_ground_truth_than_a_bf16_variance():
    model, rollout = _fused_model_and_rollout()
    records = _capture_norms(model, rollout, norm_type=RMSNorm)
    improved = 0
    for record in records:
        x, out, weight, eps = record["x"], record["out"], record["weight"], record["eps"]
        ground_truth = _fp32_statistics_reference(x.double(), weight.double(), eps)
        low = _reduced_precision_statistics_reference(x, weight, eps)
        err_out = float((out.double() - ground_truth).abs().max())
        err_low = float((low.double() - ground_truth).abs().max())
        assert math.isfinite(err_out) and math.isfinite(err_low)
        if err_out <= err_low:
            improved += 1
    assert improved > len(records) // 2


# ---------------------------------------------------------------------------
# FP32 equivalence preserved
# ---------------------------------------------------------------------------


def test_upcast_norm_is_bitwise_identical_to_torch_rmsnorm_in_fp32():
    """Upcasting is a no-op in FP32, and the state_dict key set is unchanged."""
    for dim in (8, 24, 128):
        torch.manual_seed(dim)
        upcast = RMSNorm(dim)
        reference = torch.nn.RMSNorm(dim)
        with torch.no_grad():
            reference.weight.normal_()
        with torch.no_grad():
            upcast.weight.copy_(reference.weight)
        x = torch.randn(512, dim)
        with torch.no_grad():
            assert torch.equal(upcast(x), reference(x)), dim
    assert list(RMSNorm(24).state_dict()) == ["weight"]
    assert list(torch.nn.RMSNorm(24).state_dict()) == ["weight"]


def test_fp32_v13_to_v14_migration_is_still_bitwise_identical():
    torch.manual_seed(107)
    legacy = deterministic_agent(tiny_config(LEGACY_ARCH), seed=109).eval()
    fused = deterministic_agent(tiny_config(FUSE_ARCH), seed=109).eval()
    missing, unexpected = migrate_legacy_into_fuse(fused, legacy)
    assert unexpected == [] and all(key.startswith("actor_fuse.") for key in missing)

    rollout = tiny_rollout(batch=2, steps=3)
    with torch.no_grad():
        old_out, old_logits = capture_logits(legacy, rollout)
        new_out, new_logits = capture_logits(fused, rollout)
    assert torch.equal(old_logits, new_logits)
    assert torch.equal(old_out.avail_probs, new_out.avail_probs)
    assert torch.equal(
        torch.argmax(old_out.avail_probs, dim=-1), torch.argmax(new_out.avail_probs, dim=-1),
    )
    assert torch.equal(old_out.avail_log_probs, new_out.avail_log_probs)


# ---------------------------------------------------------------------------
# Autocast coverage decision + full-iteration timing (auditable)
# ---------------------------------------------------------------------------


def _reduced_precision_is_active() -> bool:
    """API-independent probe: would an autocast-eligible op run in reduced precision?"""
    probe = torch.ones(2, 2).matmul(torch.ones(2, 2))
    return probe.dtype is torch.bfloat16


def test_rollout_forward_is_fp32_while_the_learner_forward_is_autocast(tmp_path):
    """Documents the deliberate autocast scope: learner only, rollout stays FP32.

    ``runtime/sampler.py`` (which performs the rollout forward) is outside this
    phase's modification range, so the coverage boundary is asserted explicitly
    here rather than left implicit.
    """
    cfg, game = smoke_trainer_config(FUSE_ARCH, run_name="amp-scope", amp_enabled=True)
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "run")
    try:
        assert trainer.amp_enabled is True
        rollout_states: list[bool] = []
        handle = trainer.agent_resume.register_forward_hook(
            lambda _m, _i, _o: rollout_states.append(_reduced_precision_is_active()),
        )
        try:
            rollout, *_rest = trainer._collect(trainer.agent_model)
        finally:
            handle.remove()
        assert rollout_states, "the rollout path never called the agent"
        assert not any(rollout_states), "rollout forwards must stay FP32"

        learner_states: list[bool] = []
        handle = trainer.agent_resume.register_forward_hook(
            lambda _m, _i, _o: learner_states.append(_reduced_precision_is_active()),
        )
        try:
            with torch.no_grad():
                trainer._forward_agent(rollout)
        finally:
            handle.remove()
        assert learner_states, "the learner forward never called the agent"
        assert all(learner_states), "learner forwards must run under autocast"
        assert _reduced_precision_is_active() is False  # the scope is closed again
    finally:
        trainer.close()


@pytest.mark.slow
def test_full_iteration_wall_time_amp_off_vs_on(tmp_path):
    """§七 '完整 iter 耗时': reported for both AMP settings, once each."""
    timings: dict[str, dict[str, float]] = {}
    for label, enabled in (("off", False), ("on", True)):
        cfg, game = smoke_trainer_config(FUSE_ARCH, run_name=f"it-{label}", amp_enabled=enabled)
        trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / label)
        try:
            trainer.train_iteration()  # warm up (allocator / sampler first touch)
            started = time.perf_counter()
            metrics = trainer.train_iteration()
            timings[label] = {
                "full_iteration_seconds": time.perf_counter() - started,
                "learner_seconds": float(metrics["perf/learner_seconds"]),
                "rollout_seconds": float(metrics.get("statistics/rollout_seconds", float("nan"))),
            }
        finally:
            trainer.close()
    for label in ("off", "on"):
        assert timings[label]["full_iteration_seconds"] > 0.0
        assert timings[label]["learner_seconds"] > 0.0
    ratio = timings["off"]["full_iteration_seconds"] / timings["on"]["full_iteration_seconds"]
    assert math.isfinite(ratio)
    print(
        "AMP_FULL_ITERATION "
        f"off={timings['off']['full_iteration_seconds']:.4f}s "
        f"on={timings['on']['full_iteration_seconds']:.4f}s ratio={ratio:.3f} "
        f"| learner off={timings['off']['learner_seconds']:.4f}s on={timings['on']['learner_seconds']:.4f}s"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
