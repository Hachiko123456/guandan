"""Phase-2 legacy compatibility: fixed probe, parameter count, state_dict keys, warm start."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from guandan_marvel.config import deserialize_train_config
from guandan_marvel.evaluation.checkpoints import load_v2_policy
from guandan_marvel.training.local import IPPOTrainer
from guandan_marvel.training.smoke import CppSmokeGame

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts" / "build_marvel_v14_phase1_baseline.py"
MANIFEST = ROOT / "docs" / "marvel_v14" / "phase1_baseline_manifest.json"
PROBE_OUTPUTS = ROOT / "runs" / "marvel_v14_phase1" / "probe_outputs.pt"
M09_WARM_START = (
    ROOT / "runs" / "bootcamp_week1_delivery" / "datasets" / "session1_start" / "warm_start_policy.pt"
)
M09_CANDIDATE = ROOT / "runs" / "comparison_delivery" / "dataset" / "m09_candidate_iter002000.pt"


def _load_builder():
    spec = importlib.util.spec_from_file_location("phase1_baseline_builder_compat", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_fixed_probe_parameter_count_and_state_dict_are_unchanged():
    if not (MANIFEST.is_file() and PROBE_OUTPUTS.is_file()):
        pytest.skip("phase-1 probe artifacts are not present in this workspace")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    builder = _load_builder()
    result = builder.run_probe(builder.DEFAULT_CHECKPOINT)
    outputs = result["outputs"]
    assert outputs["parameter_counts"]["total"] == 2_061_314
    assert outputs["parameter_counts"]["total"] == manifest["probe"]["parameter_counts"]["total"]
    assert outputs["checkpoint_state_dict_sha256"] == manifest["probe"]["checkpoint_state_dict_sha256"]
    assert outputs["model_output_content_sha256"] == manifest["probe"]["model_output_content_sha256"]
    assert outputs["probability_match_detail"]["reconstructed_softmax_matches_avail_probs"] is True

    frozen = torch.load(PROBE_OUTPUTS, map_location="cpu", weights_only=False)
    assert torch.equal(outputs["avail_probs"], frozen["avail_probs"].float())
    assert torch.equal(outputs["argmax_action"], frozen["argmax_action"])
    assert torch.equal(outputs["selected_prob"], frozen["selected_prob"])
    assert torch.equal(outputs["raw_head_logits"], frozen["raw_head_logits"].float())


def test_agent_state_dict_keys_are_unchanged_by_phase2():
    if not M09_CANDIDATE.is_file():
        pytest.skip("M09 candidate checkpoint is not present in this workspace")
    payload = torch.load(M09_CANDIDATE, map_location="cpu", weights_only=False)
    frozen_keys = tuple(sorted(payload["agent_state_dict"]))
    cfg = deserialize_train_config(payload["resolved_config"])
    assert cfg is not None
    from guandan_marvel.model import Agent

    model = Agent(cfg, enable_value_head=True, enable_ntp_head=Agent.ntp_enabled_for_config(cfg))
    assert tuple(sorted(model.state_dict())) == frozen_keys
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_061_314
    from guandan_marvel.model.types import ActiveAgentOutput

    assert "avail_log_probs" in ActiveAgentOutput.__dataclass_fields__


def test_m09_policy_only_warm_start_loads_and_one_macro_iteration_runs(tmp_path: Path):
    if not M09_WARM_START.is_file():
        pytest.skip("M09 warm start checkpoint is not present in this workspace")
    payload = torch.load(M09_WARM_START, map_location="cpu", weights_only=False)
    resolved = dict(payload["resolved_config"])
    cfg = deserialize_train_config(resolved)
    assert cfg is not None
    game = CppSmokeGame(max_game_horizon=512)
    cfg.device = "cpu"
    cfg.logging_dir = str(tmp_path / "logging")
    cfg.run_name = "warm-start-macro"
    cfg.rollout_batch_size = 1
    cfg.collect_batch_size = 1
    cfg.replay_buffer_scale = 1
    cfg.mini_batches = 1
    cfg.total_iterations = 1
    cfg.max_sampler_bytes_per_rank = 512 * 2**20
    cfg.amp_enabled = False
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    cfg.ppo.agent_epoch = 1
    cfg.ppo.training_semantics_profile = "macro_decision_v1"
    cfg.ppo.policy_regularizer_mode = "legacy_token_kl_uniform_v1"
    cfg.ddp.enabled = False
    cfg.ddp.require_cuda = False
    cfg.ddp.backend = "gloo"
    trainer = IPPOTrainer(cfg, game=game, run_dir=tmp_path / "run")
    info = trainer.load_warm_start_policy(payload)
    assert info["source_iteration"] == 2000
    assert trainer.iteration == 0 and trainer.learner_updates == 0
    metrics = trainer.train_iteration()
    assert metrics["macro/profile"] == "macro_decision_v1"
    assert metrics["macro_initial/macro_count_global"] >= 1
    trainer.close()
    # The official strict loader keeps its phase-1 frozen behaviour for this legacy
    # checkpoint (a different native backend identity); it is not weakened.
    with pytest.raises(Exception) as info_error:
        load_v2_policy(M09_CANDIDATE, device="cpu", expected_algorithm="ippo")
    assert "identity" in str(info_error.value).lower() or "backend" in str(info_error.value).lower()
