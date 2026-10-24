"""Stage-2 DMC candidate enumeration / GD-DMC-ACTION-1 contract tests.

The fuzz corpus contains 220 decision positions drawn from random deals and
random play.  Positions are accepted only when a decision has at most
``CANDIDATE_CAP`` complete candidates: replaying *every* candidate token by
token through ``legal_token_rows`` costs O(candidates^2) in the accepted A03
protocol, and an opening 27-card lead already has ~2000 candidates (minutes
per position).  The cap is an explicit runtime bound, not a tolerance
change: within the corpus every candidate is replayed, and separate tests
cover the large-lead and tribute/return decision points that the cap excludes.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from guandan.action_state import COMMIT_TOKEN, PASS_TOKEN, canonical_tokens
from guandan.cards import Card, Rank
from guandan.environment import _CachedProtocol
from guandan.round import apply_action
from guandan.state import Phase, PreviousHandResult, RoundState
from guandan_marvel.dmc.candidates import (
    ACTION_FEATURE_DIM,
    ACTION_FEATURE_VERSION,
    FAMILY_ORDER,
    CandidateAction,
    encode_action_features,
    enumerate_candidates,
)

CANDIDATE_CAP = 256
FUZZ_POSITIONS = 220
FUZZ_SEEDS = (0, 1, 2, 3, 7, 11, 19, 23)
LEVELS = (Rank.TWO, Rank.FIVE, Rank.SIX, Rank.ACE)


def _deal(seed: int, *, level: Rank = Rank.TWO, leader: int = 0, previous=None) -> _CachedProtocol:
    return _CachedProtocol(
        RoundState.deal(seed=seed, level_rank=level, leader_seat=leader, previous_result=previous),
    )


def _collect_positions(limit: int = FUZZ_POSITIONS, cap: int = CANDIDATE_CAP):
    """Random deals + random play; snapshot decision points below the runtime cap."""

    rng = random.Random(20261024)
    positions: list[tuple[_CachedProtocol, tuple[CandidateAction, ...]]] = []
    for game_index, seed in enumerate(FUZZ_SEEDS):
        if len(positions) >= limit:
            break
        state = _deal(seed, level=LEVELS[game_index % len(LEVELS)], leader=seed % 4)
        while not state.done and len(positions) < limit:
            candidates = enumerate_candidates(state)
            assert candidates, "a live decision point must have at least one legal candidate"
            if len(candidates) <= cap:
                positions.append((state.clone(), candidates))
            for token in rng.choice(candidates).tokens:
                state.step(token)
    return positions


def _token_paths(state: _CachedProtocol) -> set[tuple[int, ...]]:
    """DFS over the protocol's legal-next-token view from the empty prefix.

    Non-commit tokens provably leave the public round state untouched (A03
    checks this on every step), so the DFS advances the prefix directly and
    does not clone.  ``_assert_submittable`` separately replays every complete
    candidate through ``legal_token_rows()`` and ``step()``; together they
    cover both the token-view completeness and the commit path.
    """

    found: set[tuple[int, ...]] = set()
    saved_prefix = state._prefix_tokens

    def walk(prefix: tuple[int, ...]) -> None:
        state._prefix_tokens = prefix
        for token in state.legal_next_tokens():
            token = int(token)
            if token in (PASS_TOKEN, COMMIT_TOKEN):
                found.add(prefix + (token,))
            else:
                walk(prefix + (token,))

    try:
        walk(())
    finally:
        state._prefix_tokens = saved_prefix
    return found


def _assert_candidate_fields(candidate: CandidateAction) -> None:
    assert candidate.card_ids == tuple(sorted(candidate.card_ids))
    assert 0 <= candidate.family <= len(FAMILY_ORDER)
    if candidate.is_pass:
        assert candidate.tokens == (PASS_TOKEN,)
        assert candidate.family == 0
        assert candidate.card_ids == ()
        assert candidate.comparison_rank == 0
        assert candidate.wild_assignments == ()
    else:
        assert candidate.tokens and candidate.tokens[-1] == COMMIT_TOKEN
        assert candidate.card_ids, "a play/exchange candidate must select physical cards"


def _assert_submittable(base: _CachedProtocol, candidate: CandidateAction) -> None:
    trial = base.clone()
    result = None
    for index, token in enumerate(candidate.tokens):
        rows = trial.legal_token_rows()
        assert token in {row.token_id for row in rows}, (
            f"candidate token {token} at step {index} is not a legal token row"
        )
        result = trial.step(token)
        if index < len(candidate.tokens) - 1:
            assert not result.is_commit
            assert result.committed_action is None
    assert result is not None and result.is_commit
    public = result.committed_action
    assert public is not None
    atomic = public.atomic_action

    # Token path and public action describe the same CommittedAction.
    assert canonical_tokens(atomic) == candidate.tokens
    assert tuple(sorted(int(card.card_id) for card in atomic.cards)) == candidate.card_ids
    assert tuple(sorted(int(card_id) for card_id in public.card_ids)) == candidate.card_ids
    if candidate.is_pass:
        assert atomic.kind == "pass"
        assert atomic.declaration is None
    elif candidate.family == 0:
        # Tribute/return exchange card: a control action, never a declaration.
        assert atomic.kind in {"tribute", "return"}
        assert atomic.declaration is None
    else:
        assert atomic.kind is FAMILY_ORDER[candidate.family - 1]
        declaration = atomic.declaration
        assert declaration is not None
        assert int(declaration.comparison_rank) == candidate.comparison_rank
        assert candidate.wild_assignments == tuple(
            (int(card_id), int(rank), int(suit))
            for card_id, (rank, suit) in sorted(declaration.wild_assignments.items())
        )

    # Committing through tokens equals applying the atomic action directly.
    baseline = base.clone()
    apply_action(baseline.round_state, atomic)
    assert trial.round_state.serialize() == baseline.round_state.serialize()


def _assert_features(candidate: CandidateAction, state: _CachedProtocol) -> None:
    features = encode_action_features(candidate, state.round_state)
    assert features.shape == (ACTION_FEATURE_DIM,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))
    assert np.array_equal(features, encode_action_features(candidate, state.round_state))

    assert features[0] == (1.0 if candidate.is_pass else 0.0)
    assert features[1:12].sum() == 1.0
    assert int(np.argmax(features[1:12])) == candidate.family
    assert np.all(features[53:] == 0.0)

    count = len(candidate.card_ids)
    assert features[27] == 0.0  # block padding inside the locked 12..27 range
    assert features[12:27].sum() == pytest.approx(count / 4.0)
    assert features[32:47].sum() == pytest.approx(count / 4.0)
    assert features[47:52].sum() == pytest.approx(count / 27.0)
    assert features[28] == pytest.approx(count / 27.0)
    assert features[30] == pytest.approx(candidate.comparison_rank / 15.0)

    level = state.round_state.level_rank
    wilds = sum(1 for card_id in candidate.card_ids if Card(card_id).is_level_wild(level))
    assert features[29] == pytest.approx(wilds / 2.0)
    assert features[52] == pytest.approx(0.5 if wilds else 0.0)

    has_face = state.round_state.current_winning is not None
    expected_beat = 1.0 if has_face and 1 <= candidate.family <= len(FAMILY_ORDER) else 0.0
    assert features[31] == expected_beat
    if has_face and candidate.family >= 1:
        # Legal enumeration invariant: every non-pass play beats the live face.
        assert features[31] == 1.0


def _assert_position(state: _CachedProtocol, candidates: tuple[CandidateAction, ...]) -> None:
    assert candidates == enumerate_candidates(state)

    round_state = state.round_state
    pass_allowed = round_state.phase is Phase.PLAY and round_state.current_winning is not None
    assert any(candidate.is_pass for candidate in candidates) is pass_allowed
    if round_state.phase is not Phase.PLAY:
        assert all(not candidate.is_pass for candidate in candidates)

    paths = _token_paths(state)
    assert paths == {candidate.tokens for candidate in candidates}
    assert len(paths) == len(candidates)

    for candidate in candidates:
        _assert_candidate_fields(candidate)
        _assert_submittable(state, candidate)
        _assert_features(candidate, state)


def test_fuzz_random_positions_cover_the_candidate_contract():
    positions = _collect_positions()
    assert len(positions) >= 200
    seen_lead = seen_follow = seen_pass = False
    for state, candidates in positions:
        _assert_position(state, candidates)
        has_face = state.round_state.current_winning is not None
        seen_follow |= has_face
        seen_lead |= not has_face
        seen_pass |= any(candidate.is_pass for candidate in candidates)
    assert seen_lead and seen_follow and seen_pass


def test_large_lead_state_is_complete_and_candidates_are_submittable():
    """The 27-card opening lead (~2000 candidates) is excluded from the capped corpus."""

    state = _deal(0, level=Rank.TWO)
    candidates = enumerate_candidates(state)
    assert len(candidates) > CANDIDATE_CAP
    assert candidates == enumerate_candidates(state)
    assert all(not candidate.is_pass for candidate in candidates)

    paths = _token_paths(state)
    assert paths == {candidate.tokens for candidate in candidates}

    # Deterministic spread sample, replayed through legal_token_rows.
    sample = candidates[:: max(1, len(candidates) // 64)][:64]
    assert sample
    for candidate in sample:
        _assert_submittable(state, candidate)
        _assert_features(candidate, state)


def test_tribute_and_return_phase_candidates_are_exchanges_not_passes():
    previous = PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN")
    state = _deal(7, level=Rank.SIX, previous=previous)
    assert state.round_state.phase is Phase.TRIBUTE
    while state.round_state.phase is Phase.TRIBUTE:
        tribute = enumerate_candidates(state)
        assert len(tribute) == 1
        assert not tribute[0].is_pass and tribute[0].family == 0
        assert _token_paths(state) == {tribute[0].tokens}
        _assert_submittable(state, tribute[0])
        _assert_features(tribute[0], state)
        for token in tribute[0].tokens:
            state.step(token)

    assert state.round_state.phase is Phase.RETURN
    returns = enumerate_candidates(state)
    assert returns and all(not candidate.is_pass and candidate.family == 0 for candidate in returns)
    assert _token_paths(state) == {candidate.tokens for candidate in returns}
    for candidate in returns:
        _assert_candidate_fields(candidate)
        _assert_submittable(state, candidate)
        _assert_features(candidate, state)


def test_pass_is_only_emitted_when_the_rules_allow_it():
    state = _deal(1, level=Rank.TWO)
    assert state.round_state.current_winning is None
    assert not any(candidate.is_pass for candidate in enumerate_candidates(state))

    lead = enumerate_candidates(state)[0]
    for token in lead.tokens:
        state.step(token)
    assert state.round_state.current_winning is not None
    candidates = enumerate_candidates(state)
    assert any(candidate.is_pass for candidate in candidates)
    pass_candidate = next(candidate for candidate in candidates if candidate.is_pass)
    assert pass_candidate.tokens == (PASS_TOKEN,)


def test_feature_version_and_encoding_are_stable():
    assert ACTION_FEATURE_VERSION == "GD-DMC-ACTION-1"
    assert ACTION_FEATURE_DIM == 64
    state = _deal(3, level=Rank.ACE)
    candidates = enumerate_candidates(state)
    for candidate in candidates[:16]:
        first = encode_action_features(candidate, state.round_state)
        second = encode_action_features(
            candidate,
            {
                "level_rank": int(state.round_state.level_rank),
                "current_winning": state.round_state.current_winning,
            },
        )
        assert first.shape == (64,) and first.dtype == np.float32
        assert np.array_equal(first, second)
