"""Fresh-initialization policy import and optional macro target-KL (phase 7).

These tests are CPU-only and use the Python-reference smoke game so they never
depend on the C++ extension.  The KL reference values are hand-computed with
``math.expm1`` (never with the production helper), including the two unequal
macro-count counterexamples from the task specification.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from guandan_marvel.config import (
    MACRO_DECISION_TRAINING_SEMANTICS_PROFILE, MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
    PPOConfig, ModelConfig, TrainConfig,
)
from guandan_marvel.model import Agent
from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.phase6.opponent_pool import OpponentPoolError, PoolPolicyRouter
from guandan_marvel.training.checkpoints import (
    build_backend_identity, canonical_json_hash, capture_rng_state, serialize_config,
)
from guandan_marvel.training.local import IPPOTrainer
from guandan_marvel.training.macro_ppo import MacroPPOError, macro_sample_kl_control
from guandan_marvel.training.policy_import import (
    ALLOWED_DROP_PREFIXES, PolicyImportError, import_policy_as_fresh_run,
    initial_state_fingerprint,
)
from guandan_marvel.training.smoke import PythonReferenceSmokeGame

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _game() -> PythonReferenceSmokeGame:
    return PythonReferenceSmokeGame(16)


def _config(*, ntp_weight: float = 0.0, seed: int = 11, target_kl: float | None = None) -> TrainConfig:
    return TrainConfig(
        game=_game().spec,
        algo_name="ippo",
        device="cpu",
        seed=seed,
        run_name=f"phase7-import-{seed}-{int(ntp_weight * 1000)}",
        rollout_batch_size=1,
        collect_batch_size=1,
        mini_batches=1,
        max_sampler_bytes_per_rank=256 * 2**20,
        ntp_weight=float(ntp_weight),
        model=ModelConfig(
            architecture="v1.4-macro-fuse", history_dim=8, feature_dim=8,
            decoder_layers=1, decoder_heads=1, state_layers=1,
            action_layers=1, fuse_layers=1, ffn_ratio=2,
        ),
        ppo=PPOConfig(
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
            target_kl=target_kl,
        ),
    )


def _trainer(tmp_path: Path, name: str, *, ntp_weight: float = 0.0) -> IPPOTrainer:
    return IPPOTrainer(_config(ntp_weight=ntp_weight), game=_game(), run_dir=tmp_path / name)


def _source_payload(cfg: TrainConfig, state: dict, **overrides) -> dict[str, object]:
    resolved = serialize_config(cfg)
    payload: dict[str, object] = {
        "schema": "guandan-marvel-evaluation-checkpoint-v1",
        "algorithm": "ippo",
        "architecture": str(cfg.model.architecture),
        "protocol_version": str(cfg.game.protocol_version),
        "backend": "python_reference",
        "backend_identity": build_backend_identity(_game()),
        "resolved_config": resolved,
        "resolved_config_hash": canonical_json_hash(resolved),
        "agent_state_dict": {key: value.detach().clone() for key, value in state.items()},
        "iteration": 375,
        "merged_policy": {"identity": "current_policy_copy", "beta": 0.0},
        "resume_classification": "evaluation_only",
        "training_resume_allowed": False,
        "source_hash": "a" * 64,
        "source_parent_commit": "b" * 40,
    }
    payload.update(overrides)
    return payload


def _log_ratio_for_kl(target_kl: float) -> float:
    """Invert ``expm1(x) - x`` by bisection (independent of the implementation)."""
    low, high = 0.0, 50.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if math.expm1(middle) - middle < target_kl:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _macro_tensor(values: list[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = torch.tensor(values, dtype=torch.float32).view(1, -1)
    valid = torch.ones_like(ratio, dtype=torch.bool)
    lengths = torch.ones_like(ratio, dtype=torch.long)
    return ratio, valid, lengths


# ---------------------------------------------------------------------------
# target-KL mathematics (hand-computed, unequal macro counts)
# ---------------------------------------------------------------------------
def test_global_kl_weights_by_macro_count_not_by_rank_mean():
    """rank0: 1 macro with KL=.04, rank1: 9 macros with KL=0 -> global .004."""
    lr = _log_ratio_for_kl(0.04)
    ratio = torch.tensor([[lr] + [0.0] * 9], dtype=torch.float32)
    expected_local = math.expm1(lr) - lr
    assert expected_local == pytest.approx(0.04, rel=1e-6)
    decision = macro_sample_kl_control(
        ratio, torch.ones_like(ratio, dtype=torch.bool), torch.ones_like(ratio, dtype=torch.long),
        world_size=1, target_kl=None,
    )
    assert decision.global_count == 10
    assert decision.global_mean == pytest.approx(0.04 / 10, rel=1e-6)
    assert decision.global_mean != pytest.approx(0.02, rel=1e-3)


def test_global_kl_second_counterexample_nine_vs_one():
    """rank0: 9 macros with KL=.02, rank1: 1 macro with KL=0 -> global .018."""
    lr = _log_ratio_for_kl(0.02)
    ratio = torch.tensor([[lr] * 9 + [0.0]], dtype=torch.float32)
    decision = macro_sample_kl_control(
        ratio, torch.ones_like(ratio, dtype=torch.bool), torch.ones_like(ratio, dtype=torch.long),
        world_size=1, target_kl=None,
    )
    assert decision.global_count == 10
    assert decision.global_mean == pytest.approx(0.018, rel=1e-6)
    # A naive mean of rank means would be .01 and would not stop at .015.
    assert decision.global_mean > 0.015


def test_stop_is_strictly_greater_than_the_target():
    ratio = torch.tensor([[_log_ratio_for_kl(0.04)] + [0.0] * 9], dtype=torch.float32)
    valid = torch.ones_like(ratio, dtype=torch.bool)
    lengths = torch.ones_like(ratio, dtype=torch.long)
    mean = 0.004
    at_target = macro_sample_kl_control(ratio, valid, lengths, world_size=1, target_kl=mean)
    assert at_target.stop is False  # exactly at the threshold does not stop
    below = macro_sample_kl_control(ratio, valid, lengths, world_size=1, target_kl=mean - 1e-6)
    assert below.stop is True
    above = macro_sample_kl_control(ratio, valid, lengths, world_size=1, target_kl=mean + 1e-3)
    assert above.stop is False


def test_empty_and_partial_macro_sets_are_safe():
    ratio = torch.zeros((1, 4), dtype=torch.float32)
    valid = torch.zeros_like(ratio, dtype=torch.bool)
    lengths = torch.ones_like(ratio, dtype=torch.long)
    empty = macro_sample_kl_control(ratio, valid, lengths, world_size=1, target_kl=1e-4)
    assert empty.global_count == 0
    assert empty.global_mean is None
    assert empty.stop is False
    valid[0, 1:3] = True
    ratio[0, 1] = _log_ratio_for_kl(0.02)
    partial = macro_sample_kl_control(ratio, valid, lengths, world_size=1, target_kl=0.02)
    assert partial.global_count == 2
    assert partial.global_mean == pytest.approx(0.01, rel=1e-6)
    assert partial.stop is False


def test_nonfinite_log_ratio_fails_closed_instead_of_being_clamped():
    ratio = torch.tensor([[float("inf"), 0.0]], dtype=torch.float32)
    valid = torch.tensor([[True, True]])
    lengths = torch.tensor([[1, 1]])
    with pytest.raises(MacroPPOError, match="non-finite"):
        macro_sample_kl_control(ratio, valid, lengths, world_size=1, target_kl=0.01)


def test_target_kl_none_never_stops_and_records_no_target():
    lr = _log_ratio_for_kl(10.0)
    ratio = torch.tensor([[lr]], dtype=torch.float32)
    decision = macro_sample_kl_control(
        ratio, torch.ones_like(ratio, dtype=torch.bool), torch.ones_like(ratio, dtype=torch.long),
        world_size=1, target_kl=None,
    )
    assert decision.stop is False
    assert decision.target_kl is None


def _run_ddp_scenario(tmp_path: Path, scenario: str, target: float) -> list[dict[str, object]]:
    from tests.marvel_training._ddp_launcher import launch_two_ranks

    output = tmp_path / f"kl_{scenario}_{target}.json"
    environment = dict(os.environ)
    environment.update({"KL_SCENARIO": scenario, "KL_TARGET": str(target), "KL_OUTPUT": str(output)})
    os.environ.update(environment)
    try:
        completed = launch_two_ranks(
            REPO_ROOT / "tests/marvel_training/_target_kl_ddp_worker.py",
            cwd=REPO_ROOT, timeout=300.0,
        )
    finally:
        for key in ("KL_SCENARIO", "KL_TARGET", "KL_OUTPUT"):
            os.environ.pop(key, None)
    failures = [item for item in completed if item.returncode != 0]
    assert not failures, failures[0].stderr[-2000:] if failures else ""
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_two_rank_unequal_macro_counts_agree_and_do_not_deadlock(tmp_path):
    ranks = _run_ddp_scenario(tmp_path, "one_vs_nine", 0.0035)
    assert {row["rank"] for row in ranks} == {0, 1}
    assert all(row["global_count"] == 10 for row in ranks)
    assert all(row["global_mean"] == pytest.approx(0.004, rel=1e-6) for row in ranks)
    # A mean-of-rank-means (.02) and the global mean (.004) disagree below the
    # threshold, so only the global statistic keeps both ranks training.
    assert all(row["stop"] is True for row in ranks)
    steadier = _run_ddp_scenario(tmp_path, "one_vs_nine", 0.01)
    assert all(row["stop"] is False for row in steadier)
    reverse = _run_ddp_scenario(tmp_path, "nine_vs_one", 0.015)
    assert all(row["global_mean"] == pytest.approx(0.018, rel=1e-6) for row in reverse)
    assert all(row["stop"] is True for row in reverse)


@pytest.mark.slow
def test_two_rank_with_one_empty_rank_does_not_deadlock(tmp_path):
    ranks = _run_ddp_scenario(tmp_path, "empty_rank", 0.001)
    assert all(row["global_count"] == 4 for row in ranks)
    assert all(row["stop"] is False for row in ranks)


# ---------------------------------------------------------------------------
# configuration surface
# ---------------------------------------------------------------------------
def test_target_kl_none_is_bitwise_inert_and_enabled_is_semantic():
    from guandan_marvel.training.checkpoints import build_semantic_configuration

    inert = _config().to_dict()
    assert "target_kl" not in inert["ppo"]
    enabled_cfg = _config(target_kl=0.015)
    enabled = enabled_cfg.to_dict()
    assert enabled["ppo"]["target_kl"] == 0.015
    assert build_semantic_configuration(_config(), algorithm="ippo")["hash"] != build_semantic_configuration(
        enabled_cfg, algorithm="ippo"
    )["hash"]
    assert "target_kl" in build_semantic_configuration(enabled_cfg, algorithm="ippo")["training_semantics"]


def test_target_kl_rejects_legacy_profile_and_invalid_values():
    with pytest.raises(ValueError, match="macro training_semantics_profile"):
        PPOConfig(target_kl=0.01)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="target_kl"):
            PPOConfig(
                training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
                policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
                target_kl=bad,
            )


# ---------------------------------------------------------------------------
# fresh-initialization import
# ---------------------------------------------------------------------------
def test_import_manifest_contract(tmp_path):
    source_cfg = _config(ntp_weight=0.02)
    torch.manual_seed(102)
    source_model = Agent(source_cfg, enable_value_head=True, enable_ntp_head=True)
    payload = _source_payload(source_cfg, source_model.state_dict())
    trainer = _trainer(tmp_path, "candidate2", ntp_weight=0.0)
    rng_before = canonical_json_hash(capture_rng_state())
    manifest = import_policy_as_fresh_run(
        trainer, payload, source_role="candidate:p375", allowed_drop_prefixes=("ntp_head.",),
        curriculum_offset=375,
    )
    assert manifest["schema"] == "guandan-marvel-policy-import-manifest-v1"
    assert manifest["migration_version"] == "guandan-marvel-policy-import-v1"
    assert manifest["migration"]["dropped_keys"] == sorted(
        key for key in payload["agent_state_dict"] if key.startswith("ntp_head.")
    )
    assert manifest["migration"]["drop_reason"] is not None
    assert manifest["kept_key_count"] + manifest["dropped_key_count"] == len(payload["agent_state_dict"])
    assert manifest["bitwise_verified"] is True
    assert manifest["optimizer"]["covered_exactly_once"] is True
    assert manifest["rng_stream_untouched"] is True
    assert canonical_json_hash(capture_rng_state()) == rng_before
    assert trainer.curriculum_offset == 375
    assert trainer.iteration == 0 and trainer.durable_updates == 0 and trainer.learner_updates == 0
    # Every kept tensor is exactly the source tensor in both models.
    resolved = trainer.agent_resume.state_dict()
    for key, value in payload["agent_state_dict"].items():
        if key.startswith("ntp_head."):
            continue
        assert torch.equal(resolved[key], value)
    assert manifest["logical_origin"]["origin_iteration"] == 375
    assert manifest["logical_origin"]["new_iterations"] == 0


def test_both_arms_import_a_bitwise_identical_shared_initial_policy(tmp_path):
    source_cfg = _config(ntp_weight=0.02)
    torch.manual_seed(103)
    source_model = Agent(source_cfg, enable_value_head=True, enable_ntp_head=True)
    payload = _source_payload(source_cfg, source_model.state_dict())
    control = _trainer(tmp_path, "control", ntp_weight=0.02)
    candidate = _trainer(tmp_path, "candidate", ntp_weight=0.0)
    import_policy_as_fresh_run(control, payload, source_role="control:p375")
    import_policy_as_fresh_run(
        candidate, payload, source_role="candidate:p375", allowed_drop_prefixes=("ntp_head.",),
    )
    assert initial_state_fingerprint(control, include_ntp=False) == initial_state_fingerprint(
        candidate, include_ntp=False
    )
    assert initial_state_fingerprint(control, include_ntp=True) != initial_state_fingerprint(
        candidate, include_ntp=True
    )


@pytest.mark.parametrize("mutation", ["unexpected", "missing", "shape", "protocol", "semantics", "schema"])
def test_import_rejects_non_whitelisted_or_mismatched_sources(tmp_path, mutation):
    source_cfg = _config(ntp_weight=0.02)
    torch.manual_seed(104)
    source_model = Agent(source_cfg, enable_value_head=True, enable_ntp_head=True)
    state = source_model.state_dict()
    overrides: dict[str, object] = {}
    if mutation == "unexpected":
        state = dict(state)
        state["mystery_head.weight"] = torch.zeros(3, 3)
    elif mutation == "missing":
        state = {key: value for key, value in state.items() if key != "logit_head.weight"}
    elif mutation == "shape":
        state = dict(state)
        state["logit_head.weight"] = torch.zeros(5, 5)
    elif mutation == "protocol":
        overrides["protocol_version"] = "GD-ACTION-0.0-BOGUS|GD-ENV-0.0-BOGUS"
    elif mutation == "semantics":
        legacy = _config()
        legacy.ppo.training_semantics_profile = "macro_sequence_ablation_v1"
        overrides["resolved_config"] = serialize_config(legacy)
        overrides["resolved_config_hash"] = canonical_json_hash(serialize_config(legacy))
    elif mutation == "schema":
        overrides["schema"] = "guandan-marvel-training-checkpoint-v5"
    payload = _source_payload(source_cfg, state, **overrides)
    trainer = _trainer(tmp_path, f"reject-{mutation}", ntp_weight=0.0)
    with pytest.raises(PolicyImportError):
        import_policy_as_fresh_run(
            trainer, payload, source_role="candidate:p375", allowed_drop_prefixes=("ntp_head.",),
        )


def test_import_rejects_a_drop_prefix_outside_the_whitelist(tmp_path):
    source_cfg = _config(ntp_weight=0.02)
    torch.manual_seed(105)
    source_model = Agent(source_cfg, enable_value_head=True, enable_ntp_head=True)
    payload = _source_payload(source_cfg, source_model.state_dict())
    trainer = _trainer(tmp_path, "bad-whitelist", ntp_weight=0.0)
    with pytest.raises(PolicyImportError, match="whitelist"):
        import_policy_as_fresh_run(
            trainer, payload, source_role="candidate:p375", allowed_drop_prefixes=("value_head.",),
        )


def test_import_requires_an_unstarted_trainer(tmp_path):
    source_cfg = _config(ntp_weight=0.02)
    torch.manual_seed(106)
    source_model = Agent(source_cfg, enable_value_head=True, enable_ntp_head=True)
    payload = _source_payload(source_cfg, source_model.state_dict())
    trainer = _trainer(tmp_path, "started", ntp_weight=0.0)
    trainer.iteration = 3
    with pytest.raises(PolicyImportError, match="unstarted"):
        import_policy_as_fresh_run(trainer, payload, source_role="candidate:p375")


def test_old_policy_warm_start_still_rejects_pool_mode(tmp_path):
    trainer = _trainer(tmp_path, "pool-guard", ntp_weight=0.0)
    # Exercise the historical guard directly: with a router present the old
    # policy-only warm start must still refuse instead of pretending to rebuild
    # pool assignments.
    trainer.policy_router = object()
    with pytest.raises(ValueError, match="policy-only warm start cannot reconstruct"):
        trainer.load_warm_start_policy({
            "schema": "guandan-marvel-evaluation-checkpoint-v1",
            "algorithm": "ippo",
            "architecture": "v1.4-macro-fuse",
            "protocol_version": str(_config().game.protocol_version),
            "agent_state_dict": {},
        })


def test_removing_the_ntp_head_does_not_change_policy_or_value_outputs(tmp_path):
    source_cfg = _config(ntp_weight=0.02)
    torch.manual_seed(107)
    source_model = Agent(source_cfg, enable_value_head=True, enable_ntp_head=True)
    payload = _source_payload(source_cfg, source_model.state_dict())
    trainer = _trainer(tmp_path, "equivalence", ntp_weight=0.0)
    rollout = _synthetic_rollout(_game())
    manifest = import_policy_as_fresh_run(
        trainer, payload, source_role="candidate:p375", allowed_drop_prefixes=("ntp_head.",),
        equivalence_rollout=rollout,
    )
    equivalence = manifest["inference_equivalence"]
    assert equivalence is not None
    assert set(equivalence["compared_fields"]) >= {"avail_probs", "player_state_value", "state_values"}
    assert all(value == "exact" for value in equivalence["compared_fields"].values())


def _synthetic_rollout(game: PythonReferenceSmokeGame, batch: int = 1, steps: int = 4) -> TensorDict:
    players = int(game.num_players)
    data = {
        "num_steps": torch.tensor([steps] * batch, dtype=torch.long),
        "player_ids": torch.tensor([[index % players for index in range(steps)]] * batch, dtype=torch.long),
        "obs_tokens": torch.randint(0, int(game.num_tokens), (batch, steps, players, int(game.max_observation_tokens))),
        "obs_channels": torch.randint(0, 3, (batch, steps, players, int(game.num_channels))),
        "avail_tokens": torch.randint(0, int(game.num_tokens), (batch, steps, int(game.max_available_actions), int(game.max_action_tokens))),
        "avail_weights": torch.ones(batch, steps, int(game.max_available_actions), dtype=torch.float32),
    }
    return TensorDict(data, batch_size=[batch], device="cpu")


def test_router_capacity_must_cover_every_resident_entry():
    class _FakePool:
        entries = [object(), object()]

    with pytest.raises(OpponentPoolError, match="max_resident_policies"):
        PoolPolicyRouter(
            pool=_FakePool(), batch_size=1, num_players=4, max_ctx_len=16,
            device="cpu", max_resident_policies=1,
        )


# ---------------------------------------------------------------------------
# end-to-end macro iteration (CPU, Python-reference game)
# ---------------------------------------------------------------------------
def _training_game():
    return PythonReferenceSmokeGame(512)


def _iteration_config(*, target_kl, seed: int, run_name: str) -> TrainConfig:
    game = _training_game()
    return TrainConfig(
        game=game.spec, algo_name="ippo", device="cpu", seed=seed, run_name=run_name,
        rollout_batch_size=2, collect_batch_size=2, mini_batches=1,
        max_sampler_bytes_per_rank=1024 * 2**20,
        model=ModelConfig(
            architecture="v1.4-macro-fuse", history_dim=16, feature_dim=16,
            decoder_layers=1, decoder_heads=2, state_layers=1, action_layers=1,
            fuse_layers=1, ffn_ratio=2,
        ),
        ppo=PPOConfig(
            training_semantics_profile=MACRO_DECISION_TRAINING_SEMANTICS_PROFILE,
            policy_regularizer_mode=MACRO_NORMALIZED_POLICY_REGULARIZER_MODE,
            target_kl=target_kl,
        ),
    )


@pytest.mark.slow
def test_target_kl_none_adds_no_metric_and_enabled_stops_then_resumes(tmp_path):
    from guandan_marvel.training.local import IPPOTrainer

    game = _training_game()
    plain = IPPOTrainer(
        _iteration_config(target_kl=None, seed=31, run_name="kl-none"),
        game=game, run_dir=tmp_path / "none",
    )
    metrics = plain.train_iteration()
    assert not any(key.startswith("macro/kl_control") for key in metrics)

    strict = IPPOTrainer(
        _iteration_config(target_kl=1e-12, seed=31, run_name="kl-strict"),
        game=game, run_dir=tmp_path / "strict",
    )
    stopped = strict.train_iteration()
    assert stopped["macro/kl_control/stopped"] is True
    assert stopped["macro/kl_control/skipped_steps"] >= 1
    assert stopped["macro/kl_control/attempted_steps"] == (
        stopped["macro/kl_control/applied_steps"]
        + stopped["macro/kl_control/skipped_steps"]
        + stopped["macro/kl_control/amp_skipped_steps"]
    )
    duration = stopped["macro/kl_control/stop_position"]
    assert duration is not None and duration["target_kl"] == 1e-12
    assert stopped["macro/kl_control/iteration_global_count"] >= stopped["macro/kl_control/last_minibatch_global_count"]

    checkpoint = strict.save_checkpoint(tmp_path / "strict.pt")
    restored = IPPOTrainer(
        _iteration_config(target_kl=1e-12, seed=31, run_name="kl-strict"),
        game=game, run_dir=tmp_path / "strict-resume",
    )
    restored.load_checkpoint(checkpoint)
    assert restored.iteration == strict.iteration
    assert restored.learner_updates == strict.learner_updates
    assert restored.curriculum_offset == strict.curriculum_offset


@pytest.mark.slow
def test_no_target_kl_rewards_are_bitwise_unchanged(tmp_path):
    """``target_kl=None`` must not alter the update itself (same seed, same loss)."""
    from guandan_marvel.training.local import IPPOTrainer

    game = _training_game()
    # Build and train each trainer in isolation: constructing a second trainer
    # re-seeds the global RNG, so both must be created *and* trained serially.
    first = IPPOTrainer(
        _iteration_config(target_kl=None, seed=41, run_name="kl-baseline-a"),
        game=game, run_dir=tmp_path / "baseline-a",
    )
    left = first.train_iteration()
    second = IPPOTrainer(
        _iteration_config(target_kl=None, seed=41, run_name="kl-baseline-b"),
        game=game, run_dir=tmp_path / "baseline-b",
    )
    right = second.train_iteration()
    for key in ("macro/pg_loss", "macro/clip_frac", "macro/macro_count_global", "statistics/learner_updates"):
        assert left[key] == right[key], key
    a = first.agent_resume.state_dict()
    b = second.agent_resume.state_dict()
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_pool_loading_does_not_consume_the_learner_rng_stream(tmp_path, monkeypatch):
    """A pool load that draws randomness must not shift the learner init stream.

    ``OpponentPool.from_config`` is replaced by a stub that deliberately consumes
    the global RNG (and ``write_manifest``/``PoolPolicyRouter`` by inert stubs),
    so the only thing that can keep the stream identical is the trainer's own
    capture/restore around pool construction.
    """
    from guandan_marvel.phase6 import opponent_pool as pool_module

    class _StubPool:
        manifest = {"manifest_hash": "stub-manifest"}
        manifest_hash = "stub-manifest"
        entries: tuple = ()

    def _noisy_from_config(*args, **kwargs):
        torch.rand(64)  # simulate weight loading / sampling draws
        torch.randn(32)
        return _StubPool()

    class _StubRouter:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self, **_kwargs):
            return None

        def policy_probabilities(self, **_kwargs):
            raise AssertionError("no rollout is collected in this test")

        def on_commit(self, _mask):
            return None

        def state_dict(self):
            return {"schema": "stub-router"}

    monkeypatch.setattr(pool_module.OpponentPool, "from_config", staticmethod(_noisy_from_config))
    monkeypatch.setattr(pool_module, "write_manifest", lambda payload, path: None)
    monkeypatch.setattr(pool_module, "PoolPolicyRouter", _StubRouter)

    game = _game()
    plain_cfg = _config()
    pool_cfg = _config()
    pool_cfg.opponent_pool = pool_module.OpponentPoolConfig(
        enabled=True,
        entries=({
            "id": "p250", "kind": "v2_evaluation", "checkpoint": str(tmp_path / "unused.pt"),
            "checkpoint_sha256": "0" * 64, "architecture": "v1.4-macro-fuse",
            "protocol_version": str(game.spec.protocol_version), "backend": "python_reference",
            "policy_version": 250, "weight": 1.0, "active": True, "tags": ["recent"],
        },),
        sampling_seed=1, max_resident_policies=1,
    )
    plain = IPPOTrainer(plain_cfg, game=game, run_dir=tmp_path / "plain")
    after_plain = canonical_json_hash(capture_rng_state())
    pooled = IPPOTrainer(pool_cfg, game=game, run_dir=tmp_path / "pooled")
    after_pooled = canonical_json_hash(capture_rng_state())
    assert after_plain == after_pooled
    assert pooled.policy_router is not None


def test_incremental_kv_cache_matches_full_history_recompute():
    """The step-wise cache path and a full-history recompute must agree.

    Tolerance is FP32-reduction-order based (rtol=1e-6, atol=1e-7); the
    comparison is on the last step's policy distribution only.
    """
    from guandan_marvel.model import Agent, KVCache
    from guandan_marvel.model.tensordict import TensorDict

    game = _game()
    cfg = _config()
    torch.manual_seed(2026)
    model = Agent(cfg, enable_value_head=True).eval()
    players = int(game.num_players)
    steps = 4
    torch.manual_seed(7)
    obs = torch.randint(0, int(game.num_tokens), (1, steps, players, int(game.max_observation_tokens)))
    channels = torch.randint(0, 3, (1, steps, players, int(game.num_channels)))
    avail = torch.randint(0, int(game.num_tokens), (1, steps, int(game.max_available_actions), int(game.max_action_tokens)))
    weights = torch.ones(1, steps, int(game.max_available_actions), dtype=torch.float32)
    player_ids = torch.tensor([[index % players for index in range(steps)]], dtype=torch.long)

    decoder = model.decoder.config
    heads = decoder.num_key_value_heads or decoder.num_attention_heads
    head_dim = decoder.hidden_size // decoder.num_attention_heads
    kv = KVCache(
        model.decoder_n_layers, 1 * players, max_ctx_len=16,
        num_heads=heads, head_dim=head_dim, value_heads=heads, value_head_dim=head_dim,
        dtype=next(model.parameters()).dtype, device="cpu",
    )
    incremental = []
    with torch.no_grad():
        for index in range(steps):
            step = TensorDict(
                {
                    "num_steps": torch.tensor([index + 1], dtype=torch.long),
                    "player_ids": player_ids[:, index:index + 1],
                    "obs_tokens": obs[:, index:index + 1],
                    "obs_channels": channels[:, index:index + 1],
                    "avail_tokens": avail[:, index:index + 1],
                    "avail_weights": weights[:, index:index + 1],
                },
                batch_size=[1],
            )
            output = model(rollout=step, use_cache=True, past_key_values=kv)
            incremental.append(output.avail_probs.squeeze(1).clone())
        full = TensorDict(
            {
                "num_steps": torch.tensor([steps], dtype=torch.long),
                "player_ids": player_ids,
                "obs_tokens": obs,
                "obs_channels": channels,
                "avail_tokens": avail,
                "avail_weights": weights,
            },
            batch_size=[1],
        )
        reference = model(rollout=full)
    assert torch.allclose(
        incremental[-1], reference.avail_probs[:, steps - 1], rtol=1e-6, atol=1e-7,
    ), "incremental KV cache diverged from the full-history recompute"
