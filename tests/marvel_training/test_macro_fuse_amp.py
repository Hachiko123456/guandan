"""Phase-3 mixed-precision acceptance on explicit CPU and optional CUDA devices.

CPU BF16 tolerances remain unchanged.  CUDA hosts run a separate real FP16 /
GradScaler path; CPU tests never infer their device from host availability.
"""
from __future__ import annotations

import math
import time

import pytest
import torch

from guandan_marvel.model import TensorDict
from guandan_marvel.training.losses import compute_clipped_pg_loss, compute_value_loss
from guandan_marvel.training.local import IPPOTrainer
from tests.marvel_training._fuse_fixtures import (
    FUSE_ARCH, capture_logits, deterministic_agent, smoke_trainer_config, tiny_config,
    tiny_rollout,
)

# Pre-declared bfloat16 tolerances (the CPU autocast dtype).
LOGIT_ABS_TOL = 5e-2
LOGIT_REL_TOL = 2e-2
PROB_ABS_TOL = 1e-2
LOGPROB_ABS_TOL = 5e-2
VALUE_ABS_TOL = 1e-3
LOSS_ABS_TOL = 1e-2
GRAD_NORM_REL_TOL = 5e-2
ARGMAX_MIN_AGREEMENT = 0.99

# fp16-specific acceptance items that cannot be run on a CPU-only host.
FP16_NOT_RUN = (
    "CUDA fp16 autocast forward/loss parity",
    "fp16 GradScaler scale growth / backoff",
    "fp16 skipped-step accounting under real overflow",
    "fp16 learner wall-time speedup",
)


def _pg_rollout() -> TensorDict:
    base = tiny_rollout(batch=2, steps=3)
    fields = {key: base[key] for key in base.keys()}
    batch, steps, actions = fields["avail_weights"].shape
    action_taken = torch.zeros((batch, steps), dtype=torch.long)
    behavior = torch.full((batch, steps), 0.5)
    policy = torch.zeros((batch, steps, actions))
    policy[..., 0] = 0.5
    policy[..., 1] = 0.5
    fields["action_taken"] = action_taken
    fields["behavior_action_prob"] = behavior
    fields["policy_probs"] = policy
    fields["on_policy_valid"] = torch.ones((batch, steps), dtype=torch.bool)
    return TensorDict(fields, batch_size=[batch])


def _loss_outputs(model, rollout, *, autocast: bool):
    context = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        if autocast else torch.autocast(device_type="cpu", enabled=False)
    )
    with context:
        output = model(rollout=rollout)
    # Loss-side math always runs outside autocast: pass FP32 tensors explicitly.
    probs = output.avail_probs.float()
    pg = compute_clipped_pg_loss(
        rollout, clip_coef=0.2, avail_probs_new=probs,
        player_advantage=torch.linspace(-1.0, 1.0, rollout["avail_weights"].shape[0] * 3).reshape(
            rollout["avail_weights"].shape[0], 3,
        ),
    )
    value = compute_value_loss(
        rollout,
        player_state_value_pred=output.player_state_value.float(),
        player_state_value_target=torch.zeros_like(output.player_state_value.float()),
    )
    total = pg.pg_loss_mean + 0.2 * pg.kl_unif_mean + 0.5 * value.value_loss_mean
    return output, pg, value, total


def test_cpu_amp_uses_bfloat16_and_never_enables_the_fp16_scaler(tmp_path):
    cfg, game = smoke_trainer_config(FUSE_ARCH, run_name="amp-cpu", amp_enabled=True)
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "run")
    try:
        assert trainer.amp_enabled is True
        assert trainer.amp_device_type == "cpu"
        assert trainer.amp_dtype is torch.bfloat16
        assert trainer.grad_scaler.is_enabled() is False
        assert trainer.amp_scaler_enabled is False
        metrics = trainer.train_iteration()
        assert metrics["runtime/amp_enabled"] is True
        assert metrics["runtime/amp_autocast_dtype"] == "torch.bfloat16"
        assert metrics["runtime/amp_scaler_enabled"] is False
        assert metrics["runtime/amp_skipped_steps"] == 0
        for key in ("agent/grad_norm", "macro_initial/pg_loss"):
            assert math.isfinite(float(metrics[key]))
    finally:
        trainer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fp16 acceptance requires a GPU")
def test_cuda_amp_uses_fp16_scaler_and_real_optimizer_update(tmp_path):
    cfg, game = smoke_trainer_config(FUSE_ARCH, run_name="amp-cuda", amp_enabled=True)
    cfg.device = "cuda:0"
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "cuda")
    try:
        assert next(trainer.agent_resume.parameters()).device.type == "cuda"
        assert trainer.amp_dtype is torch.float16
        assert trainer.grad_scaler.is_enabled()
        parameter = trainer.agent_resume.logit_head.weight
        before = parameter.detach().clone()
        from guandan_marvel.kaggle.phase4_delivery import _fixed_probe_rollout
        rollout = _fixed_probe_rollout(game, batch=1).to(trainer.device)
        # FP16 can legitimately overflow the initial scale.  Exercise actual
        # GradScaler backoff until one finite update, without lowering tolerances
        # or changing the production initial scale.
        for _ in range(8):
            with trainer._amp_autocast():
                output = trainer.agent_model(rollout=rollout)
            assert output.player_state_value.dtype is torch.float16
            assert output.avail_log_probs.dtype is torch.float32
            loss = -output.avail_log_probs[..., 0].mean() + output.player_state_value.float().square().mean()
            norm = trainer._optimizer_step(loss=loss, module=trainer.agent_resume,
                                          optimizer=trainer.agent_optimizer, name="cuda_fp16")
            assert math.isfinite(norm)
            if trainer.learner_updates:
                break
        assert trainer.learner_updates == 1
        assert not torch.equal(parameter, before)
        scaler_state = trainer.grad_scaler.state_dict()
        restored = torch.amp.GradScaler("cuda")
        restored.load_state_dict(scaler_state)
        assert restored.state_dict() == scaler_state
    finally:
        trainer.close()


def test_amp_off_on_parity_on_identical_fixed_data():
    torch.manual_seed(101)
    model = deterministic_agent(tiny_config(FUSE_ARCH), seed=103)
    model.eval()
    rollout = _pg_rollout()

    with torch.no_grad():
        _, fp32_logits = capture_logits(model, rollout)
        output_off, pg_off, value_off, _ = _loss_outputs(model, rollout, autocast=False)
        _, bf16_logits = capture_logits(model, rollout)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output_on = model(rollout=rollout)
            _, bf16_logits = capture_logits(model, rollout)
        output_on, pg_on, value_on, _ = _loss_outputs(model, rollout, autocast=True)

    assert fp32_logits.dtype is torch.float32
    assert bf16_logits.dtype is torch.bfloat16
    # AMP genuinely changed the arithmetic (a no-op autocast would be bitwise
    # identical and would pass any pure-tolerance assertion for free).
    assert not torch.equal(fp32_logits, bf16_logits.float())

    logit_diff = (fp32_logits - bf16_logits.float()).abs()
    assert float(logit_diff.max()) <= LOGIT_ABS_TOL
    scale = float(fp32_logits.abs().max())
    assert float(logit_diff.max()) / max(scale, 1e-9) <= LOGIT_REL_TOL

    prob_diff = (output_off.avail_probs.float() - output_on.avail_probs.float()).abs()
    assert float(prob_diff.max()) <= PROB_ABS_TOL
    logprob_diff = (
        output_off.avail_log_probs.float() - output_on.avail_log_probs.float()
    ).abs()
    assert float(logprob_diff.max()) <= LOGPROB_ABS_TOL
    assert float(
        (output_off.player_state_value.float() - output_on.player_state_value.float()).abs().max()
    ) <= VALUE_ABS_TOL

    agreement = float(
        (torch.argmax(output_off.avail_probs, -1) == torch.argmax(output_on.avail_probs, -1))
        .float().mean()
    )
    assert agreement >= ARGMAX_MIN_AGREEMENT

    for name, off, on in (
        ("pg_loss", float(pg_off.pg_loss_mean), float(pg_on.pg_loss_mean)),
        ("kl_unif", float(pg_off.kl_unif_mean), float(pg_on.kl_unif_mean)),
        ("value_loss", float(value_off.value_loss_mean), float(value_on.value_loss_mean)),
    ):
        assert math.isfinite(off) and math.isfinite(on), name
        # Per-sample PPO/value terms are O(1) after the internal advantage
        # z-normalization (and value targets are O(1)), so the *mean* is compared
        # with an absolute tolerance on the O(1) term scale rather than on the
        # near-cancelling mean itself.
        assert abs(on - off) <= LOSS_ABS_TOL, (name, off, on)
        print(f"AMP_LOSS {name} off={off:.8f} on={on:.8f} abs_diff={abs(on - off):.3e}")
    assert math.isfinite(float(logit_diff.max()))


def test_amp_gradient_norm_parity_on_identical_fixed_data():
    torch.manual_seed(107)
    model = deterministic_agent(tiny_config(FUSE_ARCH), seed=109)
    model.train()
    rollout = _pg_rollout()

    def grad_norm(autocast: bool) -> float:
        model.zero_grad(set_to_none=True)
        _, _, _, total = _loss_outputs(model, rollout, autocast=autocast)
        total.backward()
        return float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).detach()
        )

    off = grad_norm(False)
    on = grad_norm(True)
    assert math.isfinite(off) and math.isfinite(on)
    assert abs(on - off) <= GRAD_NORM_REL_TOL * max(abs(off), 1e-9)


def test_amp_learner_wall_time_is_measured_and_finite():
    torch.manual_seed(113)
    model = deterministic_agent(tiny_config(FUSE_ARCH), seed=127)
    model.train()
    rollout = _pg_rollout()

    def run(autocast: bool, repeats: int) -> float:
        started = time.perf_counter()
        for _ in range(repeats):
            model.zero_grad(set_to_none=True)
            _, _, _, total = _loss_outputs(model, rollout, autocast=autocast)
            total.backward()
        return time.perf_counter() - started

    off = run(False, 4)
    on = run(True, 4)
    assert off > 0.0 and on > 0.0
    assert math.isfinite(off / on)
    print(f"AMP_LEARNER_TIMING off={off:.4f}s on={on:.4f}s ratio={off / on:.3f}")


# ---------------------------------------------------------------------------
# GradScaler state machine (exercised on CPU with a deliberately enabled scaler)
# ---------------------------------------------------------------------------


def test_scaler_path_unscales_before_clip_and_steps(tmp_path):
    cfg, game = smoke_trainer_config(FUSE_ARCH, run_name="scaler-path")
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "run")
    try:
        # Test-only: force the scaler on so the scale/unscale/state code runs.
        trainer.grad_scaler = torch.amp.GradScaler("cpu", enabled=True)
        assert trainer.grad_scaler.is_enabled() is True

        module = trainer.agent_resume
        parameter = module.logit_head.weight
        with torch.no_grad():
            parameter.fill_(0.25)
        before = parameter.detach().clone()
        # grad = 1.0 per element -> the clip norm (default 1.0) must bound the
        # update after unscaling; a clip applied to *scaled* gradients would
        # shrink the step by a factor of `scale`.
        loss = parameter.sum()
        norm = trainer._optimizer_step(
            loss=loss, module=module, optimizer=trainer.agent_optimizer, name="scaler_probe",
        )
        assert math.isfinite(norm) and norm > 0.0
        assert trainer.amp_skipped_steps == 0
        assert not torch.equal(parameter.detach(), before)
        assert all(
            torch.isfinite(p.grad if p.grad is not None else torch.zeros(1)).all()
            for p in module.parameters()
        )
        assert float(trainer.grad_scaler.get_scale()) == pytest.approx(65536.0)
    finally:
        trainer.close()


def test_scaler_skips_and_counts_an_overflowing_step(tmp_path):
    cfg, game = smoke_trainer_config(FUSE_ARCH, run_name="scaler-overflow")
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "run")
    try:
        trainer.grad_scaler = torch.amp.GradScaler("cpu", enabled=True)
        module = trainer.agent_resume
        parameter = module.logit_head.weight
        before = parameter.detach().clone()
        # 1e35 * scale(65536) overflows fp32 -> the scaled gradient is +inf.
        loss = (parameter * 1e35).sum()
        norm = trainer._optimizer_step(
            loss=loss, module=module, optimizer=trainer.agent_optimizer, name="scaler_overflow",
        )
        assert norm == 0.0
        assert trainer.amp_skipped_steps == 1
        # A skipped step must not mutate any parameter.
        assert torch.equal(parameter.detach(), before)
        # ... and it must be counted as no learner update at all.
        assert trainer.learner_updates == 0
        # The scaler backs off after the overflow.
        assert float(trainer.grad_scaler.get_scale()) < 65536.0
    finally:
        trainer.close()


def test_scaler_state_round_trips_through_a_real_optimizer_step(tmp_path):
    cfg, game = smoke_trainer_config(FUSE_ARCH, run_name="scaler-state")
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "run")
    try:
        trainer.grad_scaler = torch.amp.GradScaler("cpu", enabled=True)
        parameter = trainer.agent_resume.logit_head.weight
        trainer._optimizer_step(
            loss=parameter.sum(), module=trainer.agent_resume,
            optimizer=trainer.agent_optimizer, name="scaler_state",
        )
        saved = trainer.grad_scaler.state_dict()
        assert saved["scale"] == pytest.approx(65536.0)
        restored = torch.amp.GradScaler("cpu", enabled=True)
        restored.load_state_dict(saved)
        assert float(restored.get_scale()) == float(trainer.grad_scaler.get_scale())

        disabled = torch.amp.GradScaler("cpu", enabled=False)
        disabled.load_state_dict(saved)  # must not raise for a disabled scaler
        assert disabled.is_enabled() is False
    finally:
        trainer.close()


def test_amp_trainer_checkpoint_restores_scaler_state(tmp_path):
    cfg, game = smoke_trainer_config(FUSE_ARCH, run_name="scaler-ckpt")
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "a")
    try:
        trainer.grad_scaler = torch.amp.GradScaler("cpu", enabled=True)
        trainer.train_iteration()
        scale = float(trainer.grad_scaler.get_scale())
        path = trainer.save_checkpoint(tmp_path / "amp.pt")
    finally:
        trainer.close()

    reloaded_cfg, reloaded_game = smoke_trainer_config(FUSE_ARCH, run_name="scaler-ckpt")
    reloaded = IPPOTrainer(reloaded_cfg, game=reloaded_game, run_dir=tmp_path / "b")
    try:
        reloaded.grad_scaler = torch.amp.GradScaler("cpu", enabled=True)
        decision = reloaded.load_checkpoint(path)
        assert decision.kind in {"exact_resume", "compatible_resume"}
        assert float(reloaded.grad_scaler.get_scale()) == pytest.approx(scale)
    finally:
        reloaded.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
