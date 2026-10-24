"""NTP auxiliary head, next-observation loss, warm-start tolerance, plateau gate."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from guandan_marvel.config import GameSpec
from guandan_marvel.kaggle.worker import _evaluate_plateau_gate, _restore_anchor_history
from guandan_marvel.model import TensorDict
from guandan_marvel.model.agent import Agent
from guandan_marvel.testing.fake_game import FAKE_GAME_SPEC, fake_rollout, fake_train_config
from guandan_marvel.training.local import LocalTrainerBase
from guandan_marvel.training.losses import compute_next_observation_token_loss

PAD = FAKE_GAME_SPEC.pad_token


def _controlled_rollout() -> TensorDict:
    V = 6
    obs = torch.zeros((1, 3, 4, 4), dtype=torch.long)
    # player 0 sees {1} -> {1,2} -> {1,2,3}
    obs[0, 0, 0, :2] = torch.tensor([1, 0])
    obs[0, 1, 0, :3] = torch.tensor([1, 2, 0])
    obs[0, 2, 0, :4] = torch.tensor([1, 2, 3, 0])
    # player 1 sees {4} -> {4,5} at its decision steps
    obs[0, 0, 1, :2] = torch.tensor([4, 0])
    obs[0, 1, 1, :2] = torch.tensor([4, 0])
    obs[0, 2, 1, :3] = torch.tensor([4, 5, 0])
    assert V > int(obs.max().item())
    return TensorDict(
        {
            "num_steps": torch.tensor([3], dtype=torch.long),
            "player_ids": torch.tensor([[0, 1, -1]], dtype=torch.long),
            "obs_tokens": obs,
        },
        batch_size=[1],
    )


def test_next_observation_loss_matches_reference_and_excludes_known_tokens():
    rollout = _controlled_rollout()
    V, pos_weight = 6, 3.0
    logits = torch.zeros((1, 3, V), dtype=torch.float64)
    out = compute_next_observation_token_loss(
        rollout, ntp_logits=logits, pad_token=PAD, pos_weight=pos_weight,
    )
    # t0 excludes known {1}: 4 negatives + 1 positive (class 2).
    # t1 excludes known {4}: 4 negatives + 1 positive (class 5).
    ln2 = math.log(2.0)
    expected = ln2 * (4.0 + pos_weight) / 5.0
    assert out.ntp_step_count == 2
    assert out.ntp_loss_mean.item() == pytest.approx(expected, rel=1e-6)
    assert out.ntp_pos_rate.item() == pytest.approx(0.2, rel=1e-6)

    # Known-token logits must not affect the loss.
    known_moved = logits.clone()
    known_moved[0, 0, 1] = 50.0
    known_moved[0, 1, 4] = -50.0
    same = compute_next_observation_token_loss(
        rollout, ntp_logits=known_moved, pad_token=PAD, pos_weight=pos_weight,
    )
    assert same.ntp_loss_mean.item() == pytest.approx(expected, rel=1e-6)

    # A never-seen token (class 0 for t0) does affect the loss.
    unseen_moved = logits.clone()
    unseen_moved[0, 0, 0] = 50.0
    moved = compute_next_observation_token_loss(
        rollout, ntp_logits=unseen_moved, pad_token=PAD, pos_weight=pos_weight,
    )
    assert moved.ntp_loss_mean.item() > expected


def test_agent_ntp_head_shapes_gradients_and_disabled_path():
    cfg = fake_train_config()
    cfg.ntp_weight = 0.03
    agent = Agent(cfg, enable_value_head=True, enable_ntp_head=True)
    rollout = fake_rollout(batch_size=2, steps=3)
    output = agent(rollout=rollout)
    assert output.ntp_logits is not None
    assert output.ntp_logits.shape == (2, 3, FAKE_GAME_SPEC.num_tokens)
    loss = compute_next_observation_token_loss(
        rollout, ntp_logits=output.ntp_logits, pad_token=PAD,
        pos_weight=float(cfg.ntp_pos_weight),
    )
    assert torch.isfinite(loss.ntp_loss_mean)
    loss.ntp_loss_mean.backward()
    assert agent.ntp_head is not None
    assert agent.ntp_head.weight.grad is not None
    assert torch.isfinite(agent.ntp_head.weight.grad).all()

    disabled = Agent(fake_train_config(), enable_value_head=True, enable_ntp_head=False)
    assert disabled.ntp_head is None
    assert disabled(rollout=rollout).ntp_logits is None
    assert Agent.ntp_enabled_for_config(cfg) is True
    assert Agent.ntp_enabled_for_config(fake_train_config()) is False


def test_bootcamp_profile_disables_ntp_and_freezes_record_only_gate():
    from guandan_marvel.kaggle.bootcamp_delivery import _plateau_gate_block, _training_block
    gate = _plateau_gate_block()
    assert float(_training_block()["ntp_weight"]) == 0.0
    assert "stop_authority" not in gate
    assert gate["enabled"] is True and gate["metric"] == "rule_win_rate"


def test_ntp_weight_zero_model_returns_to_2061314_parameters():
    import guandan_marvel.game as game
    from guandan_marvel.config import GameSpec, ModelConfig, TrainConfig
    cfg = TrainConfig(
        game=GameSpec.from_game(game), device="cpu", algo_name="ippo", run_name="param-check",
        logging_dir=".", seed=1, rollout_batch_size=2, collect_batch_size=2, mini_batches=1,
        total_iterations=1, ntp_weight=0.0,
        model=ModelConfig(architecture="v1.3", history_dim=128, feature_dim=256, decoder_layers=4,
                          decoder_heads=4, state_layers=3, action_layers=2, fuse_layers=2),
    )
    agent = Agent(cfg, enable_value_head=True, enable_ntp_head=Agent.ntp_enabled_for_config(cfg))
    assert agent.ntp_head is None
    assert sum(p.numel() for p in agent.parameters()) == 2_061_314


class _Body(nn.Module):
    def __init__(self, *, with_ntp: bool, with_tail: bool = False) -> None:
        super().__init__()
        self.body = nn.Linear(2, 2)
        self.ntp_head = nn.Linear(2, 2) if with_ntp else None
        self.tail = nn.Linear(2, 2) if with_tail else None


class _Merge:
    def __init__(self) -> None:
        self.iteration = None

    def set_reference_iteration(self, value) -> None:
        self.iteration = value


def _warm_start_host(*, with_ntp: bool, with_tail: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        algorithm="ippo",
        cfg=SimpleNamespace(
            model=SimpleNamespace(architecture="v1.3"),
            game=GameSpec.from_game(FAKE_GAME_SPEC),
        ),
        agent_resume=_Body(with_ntp=with_ntp, with_tail=with_tail),
        agent_merged=_Body(with_ntp=with_ntp, with_tail=with_tail),
        merge_updater=_Merge(),
        _require_healthy=lambda: None,
    )


def _warm_start_payload(state) -> dict:
    return {
        "schema": "guandan-marvel-evaluation-checkpoint-v1",
        "algorithm": "ippo",
        "architecture": "v1.3",
        "protocol_version": FAKE_GAME_SPEC.protocol_version,
        "agent_state_dict": state,
        "iteration": 2000,
        "resolved_config_hash": "a" * 64,
        "source_hash": "b" * 64,
    }


def test_warm_start_allows_missing_ntp_head_only():
    source = _Body(with_ntp=False).state_dict()
    host = _warm_start_host(with_ntp=True)
    result = LocalTrainerBase.load_warm_start_policy(host, _warm_start_payload(source))
    assert result["source_iteration"] == 2000
    assert host.merge_updater.iteration == 0

    unexpected = dict(source)
    unexpected["bogus.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        LocalTrainerBase.load_warm_start_policy(
            _warm_start_host(with_ntp=True), _warm_start_payload(unexpected),
        )

    host_tail = _warm_start_host(with_ntp=False, with_tail=True)
    with pytest.raises(ValueError, match="missing"):
        LocalTrainerBase.load_warm_start_policy(host_tail, _warm_start_payload(source))


def _write_evaluation(run_root, *, rule_wins: int, rule_games: int) -> None:
    rows = []
    for index in range(rule_games):
        rows.append(json.dumps({
            "status": "complete",
            "opponent_id": "rule",
            "candidate_win": index < rule_wins,
        }))
    rows.append(json.dumps({"status": "complete", "opponent_id": "random", "candidate_win": True}))
    (run_root / "evaluation_games.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_plateau_gate_records_alert_and_never_stops(tmp_path):
    gate = {
        "schema": "guandan-marvel-plateau-gate-v1",
        "enabled": True,
        "metric": "rule_win_rate",
        "min_evaluations": 3,
        "patience": 2,
        "min_delta": 0.01,
    }
    _write_evaluation(tmp_path, rule_wins=10, rule_games=20)
    alert, payload = _evaluate_plateau_gate(tmp_path, gate, iteration=250)
    assert alert is False and payload["stale_count"] == 0 and payload["plateau_alert"] is False

    _write_evaluation(tmp_path, rule_wins=11, rule_games=20)
    alert, payload = _evaluate_plateau_gate(tmp_path, gate, iteration=500)
    assert alert is False and payload["stale_count"] == 0 and payload["plateau_alert"] is False

    _write_evaluation(tmp_path, rule_wins=11, rule_games=20)
    alert, payload = _evaluate_plateau_gate(tmp_path, gate, iteration=750)
    assert alert is False and payload["stale_count"] == 1 and payload["plateau_alert"] is False

    _write_evaluation(tmp_path, rule_wins=10, rule_games=20)
    alert, payload = _evaluate_plateau_gate(tmp_path, gate, iteration=1000)
    assert alert is True and payload["stale_count"] == 2
    assert payload["plateau_alert"] is True
    assert "plateau_stop" not in payload
    assert "stop_authority" not in payload
    assert len(payload["entries"]) == 4
    history = json.loads((tmp_path / "anchor_history.json").read_text(encoding="utf-8"))
    assert history["stale_count"] == 2
    assert history["plateau_alert"] is True
    assert "plateau_stop" not in history
    assert "stop_authority" not in history


def test_plateau_gate_rejects_stop_authority_field():
    from guandan_marvel.kaggle.worker import _plateau_gate_config
    gate = {
        "schema": "guandan-marvel-plateau-gate-v1",
        "enabled": True,
        "metric": "rule_win_rate",
        "min_evaluations": 2,
        "patience": 1,
        "min_delta": 0.01,
        "stop_authority": True,
    }
    with pytest.raises(ValueError, match="record-only"):
        _plateau_gate_config({"plateau_gate": gate})


def test_plateau_gate_ignores_missing_opponent(tmp_path):
    gate = {
        "schema": "guandan-marvel-plateau-gate-v1",
        "enabled": True,
        "metric": "rule_win_rate",
        "min_evaluations": 2,
        "patience": 1,
        "min_delta": 0.01,
    }
    (tmp_path / "evaluation_games.jsonl").write_text(
        json.dumps({"status": "complete", "opponent_id": "random", "candidate_win": True}) + "\n",
        encoding="utf-8",
    )
    alert, payload = _evaluate_plateau_gate(tmp_path, gate, iteration=250)
    assert alert is False and payload["plateau_alert"] is False
    assert "plateau_stop" not in payload
    assert "stop_authority" not in payload
    assert payload["entries"] == []


def test_retention_prunes_interrupted_tmp_artifacts(tmp_path):
    from guandan_marvel.kaggle.checkpoints import prune_run_artifacts

    checkpoint = tmp_path / "checkpoints" / "iter000050"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
    stale_checkpoint = tmp_path / "checkpoints" / "iter000060.tmp"
    stale_checkpoint.mkdir()
    (stale_checkpoint / "rank000.pt").write_text("partial", encoding="utf-8")
    evaluation_root = tmp_path / "evaluation_checkpoints"
    evaluation_root.mkdir()
    (evaluation_root / "iter000250.pt").write_text("eval", encoding="utf-8")
    stale_evaluation = evaluation_root / "iter000250.pt.tmp-4242"
    stale_evaluation.write_text("partial", encoding="utf-8")
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps({
        "schema": "guandan-marvel-checkpoint-manifest-v2",
        "latest": {"iteration": 50, "checkpoint": "checkpoints/iter000050"},
        "history": [{"iteration": 50, "checkpoint": "checkpoints/iter000050"}],
    }), encoding="utf-8")

    retention = prune_run_artifacts(
        tmp_path, checkpoint_keep_last=1, evaluation_keep_last=1,
    )
    assert not stale_checkpoint.exists()
    assert not stale_evaluation.exists()
    assert checkpoint.is_dir()
    assert (evaluation_root / "iter000250.pt").is_file()
    assert retention["retained_checkpoints"] == ["checkpoints/iter000050"]
    assert retention["retained_evaluations"] == ["evaluation_checkpoints/iter000250.pt"]


def test_restore_anchor_history_from_recovery(tmp_path):
    recovery = tmp_path / "recovery"
    (recovery / "artifacts").mkdir(parents=True)
    history = recovery / "artifacts" / "anchor_history.json"
    history.write_text('{"schema": "guandan-marvel-anchor-history-v1"}', encoding="utf-8")
    declared = {"schema": "guandan-marvel-recovery-manifest-v3", "files": [{"path": "artifacts/anchor_history.json"}]}

    run_root = tmp_path / "run"
    run_root.mkdir()
    restored = _restore_anchor_history(recovery, run_root, recovery_payload=declared)
    assert restored is not None and restored.is_file()
    assert "anchor-history" in restored.read_text(encoding="utf-8")

    # Undeclared artifacts are never copied.
    other = tmp_path / "run_undeclared"
    other.mkdir()
    assert _restore_anchor_history(recovery, other, recovery_payload={"files": []}) is None
    assert not (other / "anchor_history.json").exists()

    # An existing run-root history always wins.
    restored.write_text("local", encoding="utf-8")
    again = _restore_anchor_history(recovery, run_root, recovery_payload=declared)
    assert again is not None and again.read_text(encoding="utf-8") == "local"
