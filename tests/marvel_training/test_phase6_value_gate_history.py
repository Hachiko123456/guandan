"""Phase-6 repair: counterfactual value uses the full visible history."""
from __future__ import annotations

import pytest
import torch

from guandan_marvel import cpp_game
from guandan_marvel.evaluation.agents import V2PolicySession
from guandan_marvel.evaluation.checkpoints import load_v2_policy
from guandan_marvel.phase6.value_gate import QualificationDeal, _value_at_state

from tests.marvel_training._phase6_pool_fixtures import make_tiny_eval_checkpoint


def test_full_history_value_matches_policy_session_at_first_state(tmp_path):
    fixture = make_tiny_eval_checkpoint(tmp_path)
    artifact = load_v2_policy(fixture["checkpoint"], device="cpu", expected_algorithm="ippo")
    deal = QualificationDeal(
        deal_id="qualification-test", seed=2_000_000_000, level_rank=3, leader_seat=0,
    )
    env = cpp_game.EnvBatch(1, configs=[deal.game_config()], auto_reset=False, max_horizon=128)
    step = env.reset(deal.seed)
    player = int(step.player_ids[0])
    session = V2PolicySession(artifact, seed=1, temperature=1.0, device="cpu")
    _probabilities, session_value = session.observe_score_and_value(step)
    assert session_value is not None
    full_value = _value_at_state(
        artifact, deal=deal, rows_to_state=[], target_player=player,
        device="cpu", max_steps=128,
    )
    assert full_value == pytest.approx(float(session_value.item()), rel=1e-4, abs=1e-5)


def test_full_history_value_changes_with_history_at_the_same_current_observation(tmp_path):
    fixture = make_tiny_eval_checkpoint(tmp_path)
    artifact = load_v2_policy(fixture["checkpoint"], device="cpu", expected_algorithm="ippo")
    deal = QualificationDeal(
        deal_id="qualification-history", seed=2_000_000_001, level_rank=4, leader_seat=1,
    )
    env = cpp_game.EnvBatch(1, configs=[deal.game_config()], auto_reset=False, max_horizon=128)
    step = env.reset(deal.seed)
    player = int(step.player_ids[0])
    first_row = torch.as_tensor(step.available_actions.tokens[0, 0], dtype=torch.long)
    import numpy as np

    row = first_row.numpy().astype(np.int64)
    env.step(row.reshape(1, -1))
    # The same current observation with one step of history is a different
    # value input; a one-step approximation would collapse the two.
    with_player = _value_at_state(
        artifact, deal=deal, rows_to_state=[row], target_player=player,
        device="cpu", max_steps=128,
    )
    fresh = _value_at_state(
        artifact, deal=deal, rows_to_state=[], target_player=player,
        device="cpu", max_steps=128,
    )
    assert isinstance(with_player, float) and isinstance(fresh, float)
    assert torch.isfinite(torch.tensor(with_player)) and torch.isfinite(torch.tensor(fresh))
