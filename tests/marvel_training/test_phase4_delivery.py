# SPDX-License-Identifier: MIT
"""MARVEL v14 phase-4 coverage and segmented-delivery counterexamples.

Fails against the pre-phase-4 implementation where (a) selection coverage was
declared from the *request* metadata rather than the generated deal rows, and
(b) there was no versioned segmented delivery with a rank-point reward plan,
52-context distribution and ``budget_stopped`` wall-clock contract.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from guandan.cards import PRINTED_RANKS
from guandan_marvel.evaluation.common import payload_hash
from guandan_marvel.evaluation.manifests import build_deals_manifest, validate_deals_manifest
from guandan_marvel.training.smoke import CppSmokeGame
from guandan_marvel.kaggle.phase4_delivery import (
    ABSOLUTE_FINAL_ITERATION, PHASE4_PLAN_NAME, RUNTIME_PROBE_ITERATION_BUDGET,
    TARGET_MODEL_ARCHITECTURE, TARGET_MODEL_PARAMETER_COUNT, TARGET_MODEL_STATE_DICT_KEYS,
    WARM_START_NAME, build_phase4_plan, run_phase4_runtime_probe, validate_phase4_delivery,
    validate_production_warm_start, _evaluation_block, _training_block,
)


def test_worker_runs_selection_tier_with_expected_parameters(monkeypatch, tmp_path):
    from guandan_marvel.kaggle import worker

    captured: dict[str, object] = {}

    def fake(checkpoint, **kwargs):
        captured.update(kwargs)
        return {"status": "ok", "selection_dataset_hash": "0" * 64}

    monkeypatch.setattr(worker, "run_smoke_evaluation_isolated", fake)
    evaluation_cfg = _evaluation_block()
    summary, parameters = worker.run_due_evaluation(
        tmp_path / "ckpt.pt", evaluation_cfg, output_root=tmp_path, device="cpu",
    )
    assert summary["status"] == "ok"
    assert parameters["role"] == "selection"
    assert parameters["base_deals"] == 52
    assert parameters["level_ranks"] == tuple(range(3, 16))
    assert parameters["leader_seats"] == (0, 1, 2, 3)
    assert parameters["seat_rotations"] == (0, 1, 2, 3)
    assert parameters["cluster_by"] == "deal_id"
    # the actually forwarded call must use the selection tier, not the anchor
    assert captured["base_deals"] == 52
    assert captured["level_ranks"] == tuple(range(3, 16))
    assert captured["leader_seats"] == (0, 1, 2, 3)
    assert captured["seat_rotations"] == (0, 1, 2, 3)
    assert evaluation_cfg["fast_anchor"]["base_deals"] == 8
    assert evaluation_cfg["final_holdout"]["generated"] is False

ROOT = Path(__file__).resolve().parents[2]
# The phase-3 synthetic smoke policy is *small* (feature_dim=24/history=16) and
# must never be a valid delivery warm start; the production-scale M09 iter2000
# v1.3 policy-only checkpoint is the formal warm start.
SYNTHETIC_WARM_START = ROOT / "runs" / "marvel_v14_phase3" / "ab-smoke" / "synthetic-v13-warm-start.pt"
M09_WARM_START = ROOT / "runs" / "comparison_delivery" / "dataset" / "m09_candidate_iter002000.pt"
M09_SHA256 = "af0df1eaeb67c88e467ee78bf6b0494c49d000f5733d655ee81b40351d4ca9c8"
BUILD_DELIVERY_SCRIPT = ROOT / "scripts" / "build_marvel_v14_phase4_delivery.py"


def _preview_profile(training: dict) -> dict:
    return {
        "schema": "guandan-marvel-kaggle-profile-v1",
        "mode": "full",
        "algorithm": "ippo",
        "ddp": {"launcher": "torchrun", "world_size": 1, "backend": "gloo", "require_cuda": False},
        "training": training,
    }


@pytest.fixture(scope="module")
def delivery(tmp_path_factory):
    if not M09_WARM_START.is_file():
        pytest.skip("production M09 warm-start checkpoint is absent")
    from guandan_marvel.kaggle.package import render_notebook
    from guandan_marvel.kaggle.phase4_delivery import build_phase4_delivery

    base = tmp_path_factory.mktemp("phase4-delivery")
    template = base / "phase4.ipynb"
    template.write_text(json.dumps(render_notebook()), encoding="utf-8")
    root = base / "delivery"
    manifest = build_phase4_delivery(
        ROOT, root, warm_start_checkpoint=M09_WARM_START, notebook_template=template,
    )
    return {"root": root, "manifest": manifest, "template": template, "base": base}


def test_selection_coverage_is_derived_from_generated_rows_not_request_metadata():
    levels = tuple(int(rank) for rank in PRINTED_RANKS)  # 13 levels requested
    manifest = build_deals_manifest(
        counts={"training": 0, "selection": 8, "final": 0},  # only 8 deals generated
        level_ranks=levels, leader_seats=(0, 1, 2, 3),
    )
    coverage = manifest["coverage"]
    assert manifest["deal_counts"]["selection"] == 8
    # eight rows can only realise the first eight levels, never all thirteen
    assert coverage["level_ranks"] == [3, 4, 5, 6, 7, 8, 9, 10]
    assert len(coverage["level_ranks"]) != 13
    assert coverage["leader_seats"] == [0, 1, 2, 3]

    # a manifest may not *declare* coverage its own rows do not realise
    tampered = dict(manifest)
    tampered["coverage"] = {
        "level_ranks": list(range(3, 16)), "leader_seats": [0, 1, 2, 3], "tribute_modes": ["none"],
    }
    tampered["hash"] = payload_hash(tampered)
    with pytest.raises(ValueError):
        validate_deals_manifest(tampered)


def test_phase4_plan_records_reward_context_and_budget_stop():
    plan = build_phase4_plan(
        warm_start_sha256="a" * 64, warm_start_iteration=7,
        native_backend_hash="b" * 64, semantic_contract_hash="c" * 64,
    )
    assert plan["reward"]["semantics"] == "guandan_rank_points"
    assert plan["reward"]["training_scale"] == 1.0 / 3.0
    assert plan["reward"]["raw_range"] == [-3.0, 3.0]
    assert plan["training_distribution"]["context_coverage"]["context_count"] == 52
    assert plan["training_distribution"]["context_coverage"]["tribute_modes"] == ["none"]
    assert plan["budget_stop_semantics"]["max_wall_seconds"] == 12_600
    assert plan["budget_stop_semantics"]["on_stop"].startswith("save_checkpoint_then_atomic_recovery")
    assert plan["budget_stop_semantics"]["repeatable_until"] == ABSOLUTE_FINAL_ITERATION
    assert plan["evaluation"]["final_holdout"]["generated"] is False
    assert plan["training_distribution"]["tribute_7B"] == "runtime_enabled_no_delivery_coverage"
    training = plan["training"]
    assert training["training_context_schedule"]["contexts"]
    assert len(training["training_context_schedule"]["contexts"]) == 52
    assert training["training_context_curriculum"]["schema"]
    assert training["training_semantics_profile"] == "macro_decision_v1"
    assert training["policy_regularizer_mode"] == "macro_normalized_kl_uniform_v1"
    assert training["reward_semantics"] == "guandan_rank_points"


def _copy_delivery(delivery: dict, destination: Path) -> Path:
    import shutil

    shutil.copytree(Path(delivery["root"]), destination)
    return destination


def test_phase4_delivery_builds_validates_and_is_tamper_evident(delivery, tmp_path):
    delivery_root = Path(delivery["root"])
    manifest = delivery["manifest"]
    assert manifest["absolute_final_iteration"] == ABSOLUTE_FINAL_ITERATION
    assert manifest["max_wall_seconds"] == 12_600
    assert {row["segment"] for row in manifest["segments"]} == {"start", "resume"}
    validation = validate_phase4_delivery(delivery_root)
    assert validation["valid"] is True
    # one shared source archive across both segments
    assert {upload["segment"] for upload in manifest["uploads"]} == {"start", "resume"}
    start_plan = json.loads((delivery_root / "datasets" / "start" / PHASE4_PLAN_NAME).read_text(encoding="utf-8"))
    assert start_plan["plan_hash"] == manifest["plan_hash"]
    # the recorded semantic contract is the *real* one for the start config
    from guandan_marvel.kaggle.phase4_delivery import build_start_semantic_contract
    start_dataset = json.loads(
        (delivery_root / "datasets" / "start" / "guandan_marvel_dataset.json").read_text(encoding="utf-8")
    )
    training = start_dataset["profiles"]["full"]["training"]
    expected_contract = build_start_semantic_contract(dict(training), run_root=delivery_root)
    assert expected_contract["hash"] == manifest["semantic_contract_hash"]
    assert manifest["semantic_contract"]["hash"] == manifest["semantic_contract_hash"]
    # tampering a semantic field changes the contract hash
    tampered_training = dict(training)
    tampered_curriculum = dict(training["training_context_curriculum"])
    tampered_curriculum["warmup_iterations"] = 5
    tampered_training["training_context_curriculum"] = tampered_curriculum
    assert build_start_semantic_contract(tampered_training, run_root=delivery_root)["hash"] != manifest["semantic_contract_hash"]
    # a tampered resume target is rejected by the delivery validator (on a copy so
    # the shared module fixture stays pristine for the other tests)
    copy = _copy_delivery(delivery, tmp_path / "tampered")
    manifest_path = copy / "datasets" / "resume" / "guandan_marvel_dataset.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["profiles"]["resume"]["resume"]["target_iterations"] = ABSOLUTE_FINAL_ITERATION + 1
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_phase4_delivery(copy)


# ---------------------------------------------------------------------------
# Phase 4A: delivery must integrate the phase-3 v1.4 fuse + AMP and a real M09
# warm start, and the verifier must reject the pre-phase-4 v1.3/AMP-off and
# synthetic-shape packages.  Every test below fails against the old delivery.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime_probe(delivery):
    if not M09_WARM_START.is_file():
        pytest.skip("production M09 warm-start checkpoint is absent")
    from guandan_marvel import cpp_game

    start_dataset = json.loads(
        (Path(delivery["root"]) / "datasets" / "start" / "guandan_marvel_dataset.json").read_text(
            encoding="utf-8",
        )
    )
    training = start_dataset["profiles"]["full"]["training"]
    report = run_phase4_runtime_probe(
        training,
        warm_start_path=Path(delivery["root"]) / "datasets" / "start" / WARM_START_NAME,
        game=CppSmokeGame(512),
        work_dir=Path(delivery["base"]) / "runtime-probe",
        iterations=1,
        iteration_budget=RUNTIME_PROBE_ITERATION_BUDGET, device="auto",
    )
    return report


def test_phase4_delivery_targets_v14_fuse_with_amp(tmp_path):
    block = _training_block()
    assert block["model"]["architecture"] == TARGET_MODEL_ARCHITECTURE == "v1.4-macro-fuse"
    assert block["amp_enabled"] is True
    from guandan_marvel import cpp_game
    from guandan_marvel.kaggle.profiles import build_train_config

    cfg = build_train_config(_preview_profile(block), game=cpp_game, run_root=tmp_path, local_cpu_test=True)
    assert cfg.model.architecture == "v1.4-macro-fuse"
    assert cfg.amp_enabled is True


def test_phase4_delivery_default_warm_start_is_production_m09():
    spec = importlib.util.spec_from_file_location("phase4_build_delivery", BUILD_DELIVERY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    default = Path(module.DEFAULT_WARM_START)
    assert default.name == "m09_candidate_iter002000.pt"
    assert "synthetic" not in str(default)
    resolved = default if default.is_absolute() else ROOT / default
    assert resolved.is_file()
    assert resolved.resolve() == M09_WARM_START.resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    identity = validate_production_warm_start(payload)
    assert identity["architecture"] == "v1.3"
    assert identity["parameter_count"] == 2_061_314
    assert identity["state_dict_keys"] == 78


def test_phase4_packaged_warm_start_loads_into_real_profile(runtime_probe):
    assert runtime_probe["valid"] is True
    assert runtime_probe["target_architecture"] == TARGET_MODEL_ARCHITECTURE
    assert runtime_probe["target_parameter_count"] == TARGET_MODEL_PARAMETER_COUNT
    assert runtime_probe["target_state_dict_keys"] == TARGET_MODEL_STATE_DICT_KEYS
    migration = runtime_probe["migration"]
    assert migration["source_architecture"] == "v1.3"
    assert migration["target_architecture"] == "v1.4-macro-fuse"
    assert migration["fuse_migrated"] is True
    assert migration["output_proj_zero_verified"] is True
    assert runtime_probe["warm_start"]["parameter_count"] == 2_061_314
    assert runtime_probe["output_proj_zero"] is True


def test_phase4_packaged_warm_start_rejects_synthetic_small_checkpoint(tmp_path):
    if not SYNTHETIC_WARM_START.is_file():
        pytest.skip("synthetic smoke warm-start is absent")
    synthetic = torch.load(SYNTHETIC_WARM_START, map_location="cpu", weights_only=False)
    # The delivery gate must reject the small synthetic checkpoint outright...
    with pytest.raises(ValueError):
        validate_production_warm_start(synthetic, source="synthetic")
    # ... and building a delivery from it must fail (it cannot load into v1.4).
    from guandan_marvel.kaggle.package import render_notebook
    from guandan_marvel.kaggle.phase4_delivery import build_phase4_delivery

    template = tmp_path / "phase4-reject.ipynb"
    template.write_text(json.dumps(render_notebook()), encoding="utf-8")
    with pytest.raises(ValueError):
        build_phase4_delivery(
            ROOT, tmp_path / "reject-delivery",
            warm_start_checkpoint=SYNTHETIC_WARM_START, notebook_template=template,
        )


def test_phase4_runtime_probe_is_bitwise_equal_before_first_step(runtime_probe):
    equivalence = runtime_probe["bitwise_policy_equivalence"]
    assert equivalence == {
        "logits": True, "avail_probs": True, "avail_log_probs": True,
        "player_state_value": True, "argmax": True,
    }
    assert runtime_probe["iterations"] >= 1
    assert runtime_probe["optimizer_updates"] >= 1
    assert runtime_probe["parameter_changed"] is True
    assert runtime_probe["runtime_adaptation"]["equivalence_dtype"] == "torch.float32"
    for key in ("agent/grad_norm", "macro_initial/pg_loss"):
        value = runtime_probe["loss_metrics"][key]
        assert value is not None and torch.isfinite(torch.tensor(float(value)))
    checkpoint = runtime_probe["checkpoint"]
    assert checkpoint["architecture"] == TARGET_MODEL_ARCHITECTURE
    assert checkpoint["amp_enabled"] is True
    assert checkpoint["resume_kind"] in {"exact_resume", "compatible_resume"}
    assert checkpoint["reload_amp_enabled"] is True
    assert checkpoint["grad_scaler_state_restored"] is True
    assert checkpoint["model_states_restored"] is True
    assert checkpoint["optimizer_state_restored"] is True
    assert checkpoint["counters_restored"] is True
    assert checkpoint["reload_parameter_device"] == runtime_probe["device"]["parameter_device"]
    assert runtime_probe["runtime_adaptation"]["kaggle_dual_t4_verified"] is False
    if torch.cuda.is_available():
        assert runtime_probe["device"]["selected"].startswith("cuda:")
        assert runtime_probe["amp_metrics"]["dtype"] == "torch.float16"
        assert runtime_probe["amp_metrics"]["scaler_enabled"] is True
        assert runtime_probe["memory"]["peak_allocated_bytes"] > 0
    else:
        assert runtime_probe["device"]["selected"] == "cpu"
        assert runtime_probe["amp_metrics"]["dtype"] == "torch.bfloat16"


def test_phase4_resume_inherits_v14_amp_semantics(delivery, runtime_probe, tmp_path):
    from guandan_marvel import cpp_game
    from guandan_marvel.kaggle.profiles import build_resume_train_config
    from guandan_marvel.training.checkpoints import TrainingCheckpointStore

    resume_profile = json.loads(
        (Path(delivery["root"]) / "datasets" / "resume" / "guandan_marvel_dataset.json").read_text(
            encoding="utf-8",
        )
    )["profiles"]["resume"]
    # the resume profile must not smuggle any training-semantic override
    for key in ("training", "model", "amp_enabled", "reward_semantics", "reward_scale"):
        assert key not in resume_profile
    payload = TrainingCheckpointStore.load(runtime_probe["checkpoint"]["path"], device="cpu")
    assert payload["architecture"] == TARGET_MODEL_ARCHITECTURE
    assert payload["amp"]["enabled"] is True
    inherited = build_resume_train_config(
        resume_profile, checkpoint_payload=payload,
        game=CppSmokeGame(512),
        run_root=tmp_path,
    )
    assert inherited.model.architecture == TARGET_MODEL_ARCHITECTURE
    assert inherited.amp_enabled is True
    # a resume profile that tries to pin an AMP override is rejected by the
    # delivery verifier (the semantic value must be inherited, never declared)
    copy = _copy_delivery(delivery, tmp_path / "resume_override")
    target = copy / "datasets" / "resume" / "guandan_marvel_dataset.json"
    forged_payload = json.loads(target.read_text(encoding="utf-8"))
    forged_payload["profiles"]["resume"]["amp_enabled"] = False
    target.write_text(json.dumps(forged_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="override inherited training semantics"):
        validate_phase4_delivery(copy)


def test_phase4_verifier_rejects_v13_or_amp_off_plan(delivery, tmp_path):
    dataset = Path(delivery["root"]) / "datasets" / "start" / "guandan_marvel_dataset.json"
    base_payload = json.loads(dataset.read_text(encoding="utf-8"))

    v13 = _copy_delivery(delivery, tmp_path / "v13")
    target = v13 / "datasets" / "start" / "guandan_marvel_dataset.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["profiles"]["full"]["training"]["model"]["architecture"] = "v1.3"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="v1.4-macro-fuse"):
        validate_phase4_delivery(v13)

    amp_off = _copy_delivery(delivery, tmp_path / "amp_off")
    target = amp_off / "datasets" / "start" / "guandan_marvel_dataset.json"
    payload = dict(base_payload)
    payload["profiles"]["full"]["training"]["amp_enabled"] = False
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AMP"):
        validate_phase4_delivery(amp_off)


def test_phase4_verifier_rejects_shape_mismatch_even_when_hashes_match(delivery, tmp_path):
    if not M09_WARM_START.is_file():
        pytest.skip("production M09 warm-start checkpoint is absent")
    # (a) a warm start whose *metadata* still claims the production policy but
    # whose tensors have the wrong shape must be rejected on shape, not on hash.
    payload = torch.load(M09_WARM_START, map_location="cpu", weights_only=False)
    forged = dict(payload)
    state = dict(payload["agent_state_dict"])
    state["state_tokens_embed.weight"] = torch.zeros(320, 16)  # production is [320,128]
    forged["agent_state_dict"] = state
    assert forged["architecture"] == "v1.3"  # metadata untouched
    assert forged["resolved_config"]["model"]["feature_dim"] == 256
    with pytest.raises(ValueError):
        validate_production_warm_start(forged, source="shape-mismatch")

    # (b) tampering the packaged warm-start bytes is rejected by the delivery
    # validator even though the profile/manifest are untouched.
    copy = _copy_delivery(delivery, tmp_path / "hash_tamper")
    warm = copy / "datasets" / "start" / WARM_START_NAME
    blob = bytearray(warm.read_bytes())
    blob[-1] ^= 0xFF
    warm.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="warm-start"):
        validate_phase4_delivery(copy)


def test_phase4_runtime_work_directory_cannot_pollute_delivery(delivery):
    from guandan_marvel.kaggle.phase4_delivery import validate_phase4_delivery_runtime

    work = Path(delivery["root"]) / "must-not-create"
    with pytest.raises(ValueError, match="outside the delivery"):
        validate_phase4_delivery_runtime(delivery["root"], work_dir=work, device="cpu")
    assert not work.exists()
