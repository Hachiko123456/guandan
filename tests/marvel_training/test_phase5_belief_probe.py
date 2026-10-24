"""Phase-5 B0 baselines, paired bootstrap, gate and probe smoke tests."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from guandan_marvel.phase5.belief_probe import (
    BaselineA, BaselineB, b0_gate, build_probe_features, clamp_predictions,
    count_metrics, evaluate_bucket_results, paired_bootstrap_ci,
    require_cuda_device, run_b0_probes, train_linear_probe,
)
from guandan_marvel.phase5.design import MAX_COPIES_BY_RANK, phase5_design_hash
from guandan_marvel.phase5.gate import (
    BeliefGateError, assert_belief_head_allowed, load_b0_gate, seal_gate_artifact,
)


def _truth(sample_count=4):
    truth = np.zeros((sample_count, 3, 15), dtype=np.float64)
    truth[:, 0, 0] = 3.0
    truth[:, 1, 1] = 2.0
    truth[:, 2, 2] = 1.0
    return truth


def test_baseline_a_is_the_training_mean_only():
    train = np.stack([_truth(2), _truth(2) + 1.0], axis=0).reshape(-1, 45)
    baseline = BaselineA.fit(train)
    prediction = baseline.predict(5)
    assert prediction.shape == (5, 3, 15)
    assert prediction[0, 0, 0] == pytest.approx(3.5)
    assert prediction[0, 1, 1] == pytest.approx(2.5)
    assert prediction[0, 2, 2] == pytest.approx(1.5)
    assert prediction[4, 0, 0] == prediction[0, 0, 0]


def _public_ledger_row(remaining, known=None, unknown=None, escrow_unknown=0):
    remaining = np.asarray(remaining, dtype=np.float64)
    known_array = (
        np.zeros((3, 15), dtype=np.float64)
        if known is None else np.asarray(known, dtype=np.float64)
    )
    if unknown is None:
        hand_capacity = remaining - known_array.sum(axis=1)
        unknown_array = np.zeros(15, dtype=np.float64)
        unknown_array[0] = hand_capacity.sum() + float(escrow_unknown)
    else:
        unknown_array = np.asarray(unknown, dtype=np.float64)
    return np.concatenate([
        remaining, known_array.reshape(-1), unknown_array,
        np.asarray([float(escrow_unknown)]),
    ])


def test_baseline_b_uses_the_v2_ledger_and_keeps_known_held_lower_bounds():
    # No known-held cards: the old hand-computed proportional allocation is
    # still the correct special case (R=5, H=5, X=0).
    remaining = (1.0, 1.0, 3.0)
    unseen = [4.0] + [0.0] * 13 + [1.0]
    public = _public_ledger_row(remaining, unknown=unseen).reshape(1, -1)
    prediction = BaselineB().predict(public)
    assert prediction.shape == (1, 3, 15)
    for seat, seat_remaining in enumerate(remaining):
        assert prediction[0, seat, 0] == pytest.approx(4.0 * seat_remaining / 5.0)
        assert prediction[0, seat, 14] == pytest.approx(1.0 * seat_remaining / 5.0)
        np.testing.assert_array_equal(prediction[0, seat, 1:14], np.zeros(13))
        assert prediction[0, seat, :].sum() == pytest.approx(seat_remaining)
    np.testing.assert_allclose(prediction[0, :, 0], np.asarray([0.8, 0.8, 2.4]))
    np.testing.assert_allclose(prediction[0, :, 14], np.asarray([0.2, 0.2, 0.6]))

    # Counterexample for the old implementation: both big jokers are public and
    # one is still held by each of two relative seats, so the unknown rank-14
    # pool is exactly zero.  The known-held cards remain *hard* lower bounds
    # (the old 18-wide code predicted 0, 0, 0).
    known = np.zeros((3, 15))
    known[0, 14] = 1.0  # next holds a publicly known big joker
    known[1, 14] = 1.0  # partner holds the other
    row = _public_ledger_row((1.0, 1.0, 0.0), known=known, unknown=[0.0] * 15)
    bounded = BaselineB().predict(row.reshape(1, -1))
    assert bounded[0, 0, 14] == pytest.approx(1.0)
    assert bounded[0, 1, 14] == pytest.approx(1.0)
    assert bounded[0, 2, 14] == pytest.approx(0.0)
    np.testing.assert_allclose(bounded.sum(axis=(0, 2)), np.asarray([1.0, 1.0, 0.0]))

    zero = BaselineB().predict(np.zeros((1, 64)))
    np.testing.assert_array_equal(zero, np.zeros((1, 3, 15)))
    # Legacy 18-wide rows cannot express ownership and are rejected.
    with pytest.raises(ValueError):
        BaselineB().predict(np.zeros((1, 18)))
    # Malformed ledgers fail closed instead of being clamped into a prediction.
    malformed = np.zeros((1, 64))
    malformed[0, 48] = 1.0  # u rank 0 is positive while H + X is zero
    with pytest.raises(ValueError):
        BaselineB().predict(malformed)
    broken = _public_ledger_row((2.0, 2.0, 2.0), unknown=[1.0] * 15)
    with pytest.raises(ValueError):
        BaselineB().predict(broken.reshape(1, -1))


def test_clamp_predictions_preserves_input_shape_and_physical_maxima():
    # The old implementation reshaped a [N, 3, 15] input to [N, 45] and
    # returned that intermediate shape, silently losing the seat axis.
    raw = np.full((2, 3, 15), 99.0, dtype=np.float64)
    raw[0, 0, 0] = -1.0
    clamped = clamp_predictions(raw)
    assert clamped.shape == (2, 3, 15)
    maxima = np.asarray(MAX_COPIES_BY_RANK, dtype=np.float64)
    assert maxima.tolist() == [8.0] * 13 + [2.0, 2.0]
    np.testing.assert_array_equal(
        clamped[0, 0], np.concatenate([[0.0], maxima[1:]]),
    )
    np.testing.assert_array_equal(
        clamped[0, 1:], np.broadcast_to(maxima, (2, 15)),
    )
    # A flat [N, 45] input keeps its own shape, and the per-rank physical bound
    # is repeated for each of the three relative seats (distinct hands).
    flat = np.full((2, 45), 99.0, dtype=np.float64)
    flat[1, :] = -5.0
    flat_clamped = clamp_predictions(flat)
    assert flat_clamped.shape == (2, 45)
    np.testing.assert_array_equal(flat_clamped[0], np.tile(maxima, 3))
    np.testing.assert_array_equal(flat_clamped[1], np.zeros(45))
    with pytest.raises(ValueError):
        clamp_predictions(np.zeros((2, 44)))
    with pytest.raises(ValueError):
        clamp_predictions(np.asarray(1.0))


def test_count_metrics_are_hand_computed():
    truth = _truth(2)
    perfect = count_metrics(truth, truth)
    assert perfect["count_mae"] == 0.0
    assert perfect["count_l1"] == 0.0
    assert perfect["rank_mae"] == 0.0
    assert perfect["exact_count_accuracy"] == 1.0
    assert perfect["conservation_error"] == 0.0
    assert perfect["calibration_ece"] == 0.0
    offset = truth.copy()
    offset[0, 0, 0] += 2.0
    metrics = count_metrics(offset, truth)
    assert metrics["count_mae"] == pytest.approx(1.0 / 45.0)
    assert metrics["count_l1"] == pytest.approx(2.0 / 2.0)
    assert metrics["exact_count_accuracy"] == pytest.approx(1.0 - 1.0 / 90.0)
    assert metrics["conservation_error"] == pytest.approx(1.0)


def test_paired_bootstrap_clusters_by_deal_and_is_deterministic():
    values = {"deal-a": 1.0, "deal-b": 2.0, "deal-c": 3.0}
    first = paired_bootstrap_ci(values, resamples=500, seed=7)
    second = paired_bootstrap_ci(values, resamples=500, seed=7)
    assert first == second
    assert first["group_count"] == 3
    assert first["lower"] <= first["mean"] <= first["upper"]
    single = paired_bootstrap_ci({"only": 0.5}, resamples=50, seed=1)
    assert single["lower"] == single["mean"] == single["upper"] == 0.5
    with pytest.raises(ValueError):
        paired_bootstrap_ci({}, resamples=10, seed=1)
    with pytest.raises(ValueError):
        paired_bootstrap_ci({"a": float("inf")}, resamples=10, seed=1)


def _probe_payload(improvements, seeds):
    return {
        "improvement_by_group": {f"deal-{index}": value for index, value in enumerate(improvements)},
        "seed_improvements": list(seeds),
        "metrics": {"B": {"count_mae": 1.0}, "D": {"count_mae": 0.5}},
        "probe_truncated": False,
        "all_losses_finite": True,
        "all_parameters_changed": True,
        "device": "cuda:0",
        "seeds": [0, 1, 2],
        "probe_seconds": 1.0,
        "peak_allocated_bytes": 1,
        "fit_audits": {"C": [], "D": []},
        "bucket_metadata": {},
        "test_groups_array": [],
        "bucket_sample_improvements": [],
    }


def _sufficiency(**overrides):
    value = {
        "completed_deals": 512,
        "collection_truncated": False,
        "test_deals": 64,
        "test_samples": 512,
        "test_level_ranks": 13,
        "test_leader_seats": 4,
        "test_tribute_modes": 4,
    }
    value.update(overrides)
    return value


def test_b0_gate_pass_reject_and_inconclusive_rules():
    bootstrap = {"resamples": 500, "seed": 12345, "confidence": 0.95}
    dataset = {"records_hash": "r", "split_hash": "s", "sample_count": 1, "group_count": 1}
    pass_gate = b0_gate(
        probe=_probe_payload([1.0, 1.1, 0.9, 1.2], [1.0, 1.0, 1.0]),
        bootstrap=bootstrap, epsilon=0.05, sufficiency=_sufficiency(),
        leakage_pass=True, group_disjoint=True, bucket_results={},
        design_hash="d", source_hash="s", config_hash="c", dataset=dataset,
    )
    assert pass_gate["verdict"] == "PASS"
    reject_gate = b0_gate(
        probe=_probe_payload([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        bootstrap=bootstrap, epsilon=0.05, sufficiency=_sufficiency(),
        leakage_pass=True, group_disjoint=True, bucket_results={},
        design_hash="d", source_hash="s", config_hash="c", dataset=dataset,
    )
    assert reject_gate["verdict"] == "REJECT"
    # Insufficient sample must never be laundered into REJECT.
    short = b0_gate(
        probe=_probe_payload([0.0, 0.0], [0.0, 0.0, 0.0]),
        bootstrap=bootstrap, epsilon=0.05,
        sufficiency=_sufficiency(completed_deals=100, test_deals=8, test_samples=8),
        leakage_pass=True, group_disjoint=True, bucket_results={},
        design_hash="d", source_hash="s", config_hash="c", dataset=dataset,
    )
    assert short["verdict"] == "INCONCLUSIVE"
    assert "insufficient_completed_deals" in short["reasons"]
    # A CI that straddles epsilon is inconclusive.
    straddle = b0_gate(
        probe=_probe_payload([-0.2, 0.2, -0.1, 0.1], [0.1, -0.1, 0.0]),
        bootstrap=bootstrap, epsilon=0.05, sufficiency=_sufficiency(),
        leakage_pass=True, group_disjoint=True, bucket_results={},
        design_hash="d", source_hash="s", config_hash="c", dataset=dataset,
    )
    assert straddle["verdict"] == "INCONCLUSIVE"
    # A severe bucket or a leakage failure blocks PASS.
    severe_bucket = {
        "phase": {"PLAY": {"sample_count": 40, "mean": -0.1,
                           "lower": -0.2, "upper": -0.05, "group_count": 10,
                           "resamples": 500, "seed": 1, "confidence": 0.95}},
    }
    severe = b0_gate(
        probe=_probe_payload([1.0, 1.1, 0.9, 1.2], [1.0, 1.0, 1.0]),
        bootstrap=bootstrap, epsilon=0.05, sufficiency=_sufficiency(),
        leakage_pass=True, group_disjoint=True, bucket_results=severe_bucket,
        design_hash="d", source_hash="s", config_hash="c", dataset=dataset,
    )
    assert severe["verdict"] != "PASS"
    leaked = b0_gate(
        probe=_probe_payload([1.0, 1.1, 0.9, 1.2], [1.0, 1.0, 1.0]),
        bootstrap=bootstrap, epsilon=0.05, sufficiency=_sufficiency(),
        leakage_pass=False, group_disjoint=True, bucket_results={},
        design_hash="d", source_hash="s", config_hash="c", dataset=dataset,
    )
    assert leaked["verdict"] == "INCONCLUSIVE"
    assert "leakage_check_failed" in leaked["reasons"]


def _write_gate(tmp_path, verdict, design_hash=None):
    payload = seal_gate_artifact({
        "schema": "guandan-marvel-phase5-belief-gate-v1",
        "verdict": verdict,
        "design_hash": design_hash or phase5_design_hash(),
        "source_hash": "source",
        "config_hash": "config",
        "dataset": {"records_hash": "r", "split_hash": "s"},
        "epsilon": 0.05,
        "primary": {"improvement": 1.0},
        "metrics": {},
        "sufficiency": {},
        "reasons": [],
    })
    path = tmp_path / f"gate-{verdict}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_gate_loader_is_fail_closed_and_hides_b1_behind_pass(tmp_path):
    pass_path = _write_gate(tmp_path, "PASS")
    loaded = load_b0_gate(pass_path)
    assert loaded["verdict"] == "PASS"
    assert load_b0_gate(pass_path, expected_design_hash=phase5_design_hash())["design_hash"]
    with pytest.raises(BeliefGateError):
        load_b0_gate(_write_gate(tmp_path, "REJECT"))
    with pytest.raises(BeliefGateError):
        load_b0_gate(_write_gate(tmp_path, "INCONCLUSIVE"))
    with pytest.raises(BeliefGateError):
        load_b0_gate(pass_path, expected_design_hash="0" * 64)
    with pytest.raises(BeliefGateError):
        load_b0_gate(tmp_path / "missing.json")
    tampered = json.loads(pass_path.read_text(encoding="utf-8"))
    tampered["verdict"] = "REJECT"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(BeliefGateError):
        load_b0_gate(tampered_path)

    class Belief:
        build_head = True
        belief_weight = 0.1
        gate_artifact = str(pass_path)

    assert assert_belief_head_allowed(Belief())["verdict"] == "PASS"
    Belief.gate_artifact = str(_write_gate(tmp_path, "REJECT"))
    with pytest.raises(BeliefGateError):
        assert_belief_head_allowed(Belief())
    Belief.build_head = False
    assert assert_belief_head_allowed(Belief()) is None


def _synthetic_matrix(samples=60, features=512):
    rng = np.random.default_rng(20261019)
    public = np.stack([
        _public_ledger_row(
            (27.0, 27.0, 27.0),
            unknown=[8.0] * 10 + [1.0] + [0.0] * 4,
        )
        for _ in range(samples)
    ]).astype(np.float32)
    matrix = {
        "channels": rng.integers(0, 2, size=(samples, features), dtype=np.uint8),
        "public_cards": rng.integers(0, 2, size=(samples, 108), dtype=np.uint8),
        "public": public,
        "counts": rng.integers(0, 4, size=(samples, 3, 15)).astype(np.float32),
        "groups": np.asarray([f"g{index % 6}" for index in range(samples)]),
        "phase": np.asarray(["PLAY"] * samples),
        "level_rank": np.asarray([str(3 + index % 3) for index in range(samples)]),
        "leader_seat": np.asarray([str(index % 2) for index in range(samples)]),
        "tribute_mode": np.asarray(["none"] * samples),
    }
    return matrix


def test_probe_features_use_only_visible_inputs_and_have_expected_widths():
    matrix = _synthetic_matrix(20)
    c = build_probe_features(matrix, "C")
    d = build_probe_features(matrix, "D")
    assert c.shape == (20, 512)
    assert d.shape == (20, 512 + 108 + 64)
    # Changing labels cannot change the model-visible features.
    baseline = d.copy()
    matrix["counts"] = np.zeros_like(matrix["counts"])
    np.testing.assert_array_equal(build_probe_features(matrix, "D"), baseline)
    matrix["public"] = np.zeros((20, 18), dtype=np.float32)
    with pytest.raises(ValueError):
        build_probe_features(matrix, "D")
    with pytest.raises(ValueError):
        build_probe_features(matrix, "X")


def test_sample_matrix_projects_the_actor_row_not_the_observer_stack():
    from guandan_marvel.phase5.belief_collect import BeliefSample, sample_matrix

    # Each sample carries the full [4, 512] observer stack; the four rows are
    # distinguishable.  The probe input must be the acting seat's own row only.
    # The old implementation stacked the stack ([N, 4, 512]) and
    # build_probe_features raised ValueError("channels must be [N, 512]").
    actors = [0, 1, 2, 3]
    samples = []
    for index, actor in enumerate(actors):
        channels = np.zeros((4, 512), dtype=np.uint8)
        for seat in range(4):
            channels[seat, 0] = seat  # distinguishable observer rows
        channels[actor, 1] = index + 1
        samples.append(BeliefSample(
            group_id=f"group-{index}",
            deal_index=index,
            token_step=0,
            actor=actor,
            phase="PLAY",
            level_rank=3,
            leader_seat=0,
            tribute_mode="none",
            tokens=np.zeros((4, 48), dtype=np.int64),
            channels=channels,
            public=np.zeros(64, dtype=np.int64),
            public_cards=np.zeros(108, dtype=np.uint8),
            public_events=(),
            ledger={},
            label_counts=np.zeros((3, 15), dtype=np.int64),
            label_mask=np.zeros((3, 15, 9), dtype=bool),
            audit={},
        ))
    matrix = sample_matrix(samples)
    assert matrix["channels"].shape == (len(samples), 512)
    for index, sample in enumerate(samples):
        np.testing.assert_array_equal(
            matrix["channels"][index], sample.channels[sample.actor],
        )
    # The pre-registered C probe now accepts the matrix (width 512).
    features = build_probe_features(matrix, "C")
    assert features.shape == (len(samples), 512)
    for index, sample in enumerate(samples):
        np.testing.assert_array_equal(
            features[index].astype(np.uint8), sample.channels[sample.actor],
        )


def test_linear_probe_cpu_fixture_changes_parameters_and_is_seed_reproducible():
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(64, 16)).astype(np.float32)
    val_x = rng.normal(size=(16, 16)).astype(np.float32)
    train_y = rng.integers(0, 4, size=(64, 45)).astype(np.float32)
    val_y = rng.integers(0, 4, size=(16, 45)).astype(np.float32)
    first = train_linear_probe(
        train_x, train_y, val_x, val_y, device="cpu", seed=0,
        epochs=3, batch_size=16, patience=2, allow_cpu=True,
    )
    second = train_linear_probe(
        train_x, train_y, val_x, val_y, device="cpu", seed=0,
        epochs=3, batch_size=16, patience=2, allow_cpu=True,
    )
    assert first.parameter_changed and first.optimizer_updates > 0
    assert first.finite_loss and not first.truncated
    # The clamp tensor must cover all 45 entries, one physical maximum per
    # (relative seat, printed rank): 8 copies for 13 normal ranks, 2 per joker.
    expected_copies = np.tile(np.asarray(MAX_COPIES_BY_RANK, dtype=np.float32), 3)
    assert tuple(first.max_copies.shape) == (1, 45)
    np.testing.assert_array_equal(
        first.max_copies.detach().cpu().numpy().reshape(45), expected_copies,
    )
    assert first.predict(val_x).shape == (16, 45)
    np.testing.assert_allclose(first.predict(val_x), second.predict(val_x), atol=0)
    with pytest.raises(ValueError):
        train_linear_probe(
            train_x, train_y, val_x, val_y, device="cpu", seed=0,
            epochs=1, allow_cpu=False,
        )


def test_run_b0_probes_integrates_a_b_c_d_on_cpu_fixture():
    matrix = _synthetic_matrix(60)
    assignment = {
        "g0": "train", "g1": "train", "g2": "train", "g3": "train",
        "g4": "validation", "g5": "test",
    }
    result = run_b0_probes(
        matrix, assignment, device="cpu", seeds=[0, 1, 2],
        epochs=2, batch_size=16, learning_rate=1e-2, weight_decay=0.0,
        patience=1, max_probe_seconds=60, allow_cpu=True,
    )
    assert set(result["metrics"]) == {"A", "B", "C", "D"}
    assert len(result["seed_improvements"]) == 3
    assert result["all_losses_finite"] and result["all_parameters_changed"]
    assert len(result["bucket_sample_improvements"]) == len(result["test_groups_array"])
    buckets = evaluate_bucket_results(
        result, bootstrap={"resamples": 50, "seed": 1, "confidence": 0.95},
    )
    assert "phase" in buckets
    with pytest.raises(ValueError):
        run_b0_probes(
            matrix, assignment, device="cpu", seeds=[0, 1],
            epochs=1, batch_size=16, learning_rate=1e-2, weight_decay=0.0,
            patience=1, max_probe_seconds=10, allow_cpu=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the probe smoke")
def test_cuda_linear_probe_really_updates_on_gpu():
    device = require_cuda_device("cuda")
    rng = np.random.default_rng(11)
    train_x = rng.normal(size=(64, 32)).astype(np.float32)
    val_x = rng.normal(size=(16, 32)).astype(np.float32)
    train_y = rng.integers(0, 4, size=(64, 45)).astype(np.float32)
    val_y = rng.integers(0, 4, size=(16, 45)).astype(np.float32)
    fit = train_linear_probe(
        train_x, train_y, val_x, val_y, device=device, seed=0,
        epochs=2, batch_size=16, patience=2, allow_cpu=False,
    )
    assert fit.device.type == "cuda"
    assert fit.parameter_changed and fit.optimizer_updates > 0
    assert fit.peak_allocated_bytes > 0
    assert fit.predict(val_x).shape == (16, 45)
    with pytest.raises(ValueError):
        require_cuda_device("cpu")


_ROOT = Path(__file__).resolve().parents[2]


def _fresh_process_probe_script() -> str:
    return textwrap.dedent(
        """
        import json
        import numpy as np
        import torch

        # This is a genuinely fresh process: importing torch / asking for
        # availability does not create a CUDA context.
        assert not torch.cuda.is_initialized(), "child must start uninitialised"
        from guandan_marvel.phase5.belief_probe import (
            require_cuda_device, train_linear_probe,
        )
        device = require_cuda_device("cuda")
        assert torch.cuda.is_initialized(), "CUDA context must exist after the gate"
        rng = np.random.default_rng(0)
        features = rng.normal(size=(8, 4)).astype(np.float32)
        labels = rng.integers(0, 3, size=(8, 45)).astype(np.float32)
        fit = train_linear_probe(
            features, labels, features, labels, device=device, seed=0,
            epochs=1, batch_size=4, patience=1,
        )
        assert fit.device.type == "cuda"
        assert int(fit.peak_allocated_bytes) > 0
        print(json.dumps({"device": str(fit.device),
                          "initialized": torch.cuda.is_initialized()}))
        """
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the probe smoke")
def test_cuda_probe_initialises_a_fresh_process_before_memory_stats():
    # Counterexample for the old code: the first CUDA operation was
    # reset_peak_memory_stats, which raises "Invalid device argument" when no
    # context exists yet.  A subprocess is required because the parent pytest
    # process may already be initialised.
    completed = subprocess.run(
        [sys.executable, "-c", _fresh_process_probe_script()],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"initialized": true' in completed.stdout
