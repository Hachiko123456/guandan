from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from guandan.cards import Rank
from guandan.state import RoundState
from guandan_marvel import DEFAULT_BACKEND, default_game, game, get_game_backend, python_game
from guandan_marvel.testing.differential import (
    CORPUS_SCHEMA,
    FIXTURE_SCHEMA,
    DifferentialMismatch,
    build_case,
    compare_step,
    corpus_record,
    cpp_batch_from_case,
    python_batch_from_case,
    replay_corpus_record,
    run_game,
)
from nanopokerlib import guandan as cpp_game


def test_cpp_default_and_python_oracle_paths_are_unambiguous():
    assert DEFAULT_BACKEND == "cpp"
    assert default_game.BACKEND == "cpp"
    assert game.BACKEND == "python_reference"
    assert game.EnvBatch is python_game.EnvBatch
    assert get_game_backend().BACKEND == "cpp"
    assert get_game_backend("python_reference") is python_game
    assert get_game_backend("cpp").BACKEND == "cpp"
    with pytest.raises(ValueError):
        get_game_backend("automatic")


def test_matched_random_game_corpus_roundtrip_has_zero_diff(tmp_path: Path):
    result = run_game(build_case(11), intermediate_quota=7, mismatch_dir=tmp_path / "mismatches")
    assert result["intermediate_sample_count"] == 7
    assert result["serialization_roundtrips"] == 1
    assert result["terminal"]["outcome_class"] in {"DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST"}
    record = corpus_record(result)
    assert record["schema"] == CORPUS_SCHEMA
    replayed = replay_corpus_record(record)
    assert replayed["trajectory_digest"] == result["trajectory_digest"]
    assert not list((tmp_path / "mismatches").glob("*.json"))

    corpus = tmp_path / "corpus.jsonl.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    with gzip.open(corpus, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline())["trajectory_digest"] == result["trajectory_digest"]


def test_all_four_observers_hide_other_hand_reorders_cross_backend():
    base = RoundState.deal(seed=301, level_rank=Rank.SIX, leader_seat=0)
    for observer in range(4):
        hidden = [seat for seat in range(4) if seat != observer]
        alternate = base.clone()
        alternate.hands[hidden[0]][0], alternate.hands[hidden[1]][0] = (
            alternate.hands[hidden[1]][0], alternate.hands[hidden[0]][0]
        )
        alternate.check_invariants()
        cases = []
        for index, state in enumerate((base, alternate)):
            case = build_case(100 + observer * 2 + index)
            case = type(case)(
                game_index=case.game_index,
                seed=case.seed,
                level_rank=int(state.level_rank),
                leader_seat=state.leader_seat,
                previous_result=None,
                deal=tuple(tuple(card.card_id for card in hand) for hand in state.hands),
            )
            py_batch, py_step = python_batch_from_case(case)
            cpp_batch, cpp_step = cpp_batch_from_case(case)
            compare_step(
                py_step, cpp_step, py_batch.critic_state(), cpp_batch.critic_state(),
                step_index=0,
            )
            cases.append((py_step, cpp_step))
        np.testing.assert_array_equal(
            cases[0][0].observations.channels[0, observer],
            cases[1][0].observations.channels[0, observer],
        )
        np.testing.assert_array_equal(
            cases[0][1].observations.channels[0, observer],
            cases[1][1].observations.channels[0, observer],
        )
        np.testing.assert_array_equal(
            cases[0][0].observations.tokens[0, observer],
            cases[1][0].observations.tokens[0, observer],
        )


def test_mismatch_fixture_schema_is_minimal_and_replayable(tmp_path: Path, monkeypatch):
    from guandan_marvel.testing import differential

    original = differential.compare_step
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DifferentialMismatch("injected", 1, {"reason": "test"})
        return original(*args, **kwargs)

    monkeypatch.setattr(differential, "compare_step", fail_once)
    with pytest.raises(DifferentialMismatch):
        run_game(build_case(3), intermediate_quota=1, mismatch_dir=tmp_path)
    fixtures = list(tmp_path.glob("*.json"))
    assert len(fixtures) == 1
    payload = json.loads(fixtures[0].read_text(encoding="utf-8"))
    assert payload["schema"] == FIXTURE_SCHEMA
    assert payload["mismatch"]["category"] == "injected"
    assert payload["case"]["deal"]
    assert isinstance(payload["steps"], list)
