from __future__ import annotations

from guandan.action_state import COMMIT_TOKEN, StepwiseActionState, canonical_tokens
from guandan.cards import Rank
from guandan.round import enumerate_legal_actions
from guandan.state import RoundState
from scripts.run_full_game_regression import run_regression


def test_seed_two_six_has_unique_canonical_action_sequences() -> None:
    protocol = StepwiseActionState.deal(seed=2, level_rank=Rank.SIX)
    candidates = protocol._candidate_sequences()
    sequences = [sequence for sequence, _action in candidates]
    assert len(sequences) == len(set(sequences))
    assert protocol.legal_next_tokens().size > 0
    for sequence, _action in candidates:
        for split, token in enumerate(sequence):
            if token == COMMIT_TOKEN:
                matching = [item for item in candidates if item[0][:split] == sequence[:split] and len(item[0]) == split + 1 and item[0][-1] == COMMIT_TOKEN]
                assert len(matching) == 1


def test_seed_two_six_round_trip_encoding_for_short_actions() -> None:
    state = RoundState.deal(seed=2, level_rank=Rank.SIX)
    actions = enumerate_legal_actions(state)
    sampled = tuple(sorted(actions, key=lambda action: len(canonical_tokens(action)))[:2])
    for action in sampled:
        protocol = StepwiseActionState(state.clone())
        tokens = canonical_tokens(action)
        before_public = protocol.round_state.serialize()
        for index, token in enumerate(tokens):
            result = protocol.step(token)
            if index < len(tokens) - 1:
                assert not result.is_commit
                assert protocol.round_state.serialize() == before_public
        assert result.is_commit
        assert canonical_tokens(result.committed_action.atomic_action) == tokens


def test_scoped_complete_game_regression_reaches_terminal_states() -> None:
    report = run_regression(4, start_seed=0, max_steps=512)
    assert report["count_requested"] == 4
    assert report["count_completed"] == 4
    assert report["count_truncated"] == 0
    assert report["count_with_illegal_actions"] == 0
    assert report["count_failed_checks"] == 0
    assert report["passed"] is True
    assert all(game["committed_steps"] > 3 for game in report["games"])
