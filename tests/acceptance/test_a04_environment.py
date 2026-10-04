"""Demand-driven GD-ENV-0.1 acceptance suite.

The case IDs below are a traceability inventory, not an enum traversal. Each
behavioral test below asserts a contract clause against the frozen public API;
missing implementation must fail as a real assertion.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import importlib
import json

import numpy as np
import pytest

from guandan.action_state import (
    COMMIT_TOKEN, PASS_TOKEN, ProtocolCommittedAction, ProtocolError,
    StepwiseActionState, TokenCodec, canonical_tokens,
)
from guandan.cards import Card, Rank, Suit, cards_for_rank, full_deck
from guandan.combos import CombinationKind, require_combination
from guandan.round import enumerate_legal_actions, return_cards_for, tribute_card_for
from guandan.state import CommittedAction, EpisodeTerminatedError, IllegalActionError, Phase, PreviousHandResult, RoundState

pytestmark = pytest.mark.acceptance

CASE_IDS = (
    "api-exports", "spec-fixed", "obs-fields", "step-fields", "batch-fields",
    "reset-explicit", "reset-deal", "reset-seed", "reset-leader", "reset-previous-result",
    "reset-fresh-a03", "reset-fresh-env", "reset-no-resume", "reset-transaction",
    "legal-mask-complete", "legal-mask-padded", "legal-mask-no-reserved", "legal-mask-no-pass-lead",
    "legal-mask-follow-pass", "legal-mask-tribute", "legal-mask-return", "legal-mask-no-overflow",
    "privacy-opponent", "privacy-teammate", "privacy-self-visible", "privacy-no-debug",
    "privacy-tribute-receipt", "privacy-return-receipt", "privacy-resolved-transfer",
    "prefix-token-clock", "prefix-zero-reward", "commit-clock", "commit-receipt", "pass-commit",
    "terminal-reward", "terminal-ranking", "base-no-autoreset", "truncation-reward",
    "post-terminal-guard", "post-truncation-guard", "invalid-token-rollback", "invalid-type-rollback",
    "duplicate-card-rollback", "clone-independent", "serialize-prefix", "serialize-terminal",
    "serialize-rng", "serialize-version", "batch-equivalence", "batch-leading-dims",
    "batch-independent", "batch-reset-at", "batch-inactive-pad", "batch-invalid-atomic",
    "batch-load-atomic", "batch-auto-reset-terminal", "batch-auto-reset-truncated",
    "batch-auto-reset-other-live", "batch-heterogeneous", "overflow-observation",
    "overflow-action", "overflow-legal", "overflow-prefix", "overflow-commit", "overflow-batch",
)

OBS_ARRAYS = {
    "observation_tokens": ((4096,), np.int32),
    "state_channels": ((256,), np.float32),
    "legal_next_tokens": ((256,), np.int32),
    "legal_next_mask": ((256,), np.bool_),
    "legal_next_is_commit": ((256,), np.bool_),
    "legal_next_kinds": ((256,), np.int32),
    "legal_next_values": ((256,), np.int32),
}
SINGLE = TokenCodec.family(CombinationKind.SINGLE)
PAIR = TokenCodec.family(CombinationKind.PAIR)
PREVIOUS = PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN")


@pytest.fixture
def api():
    return importlib.import_module("guandan.environment")


def _card(rank, index=0):
    return cards_for_rank(rank)[index]


def _checkpoint():
    low = _card(Rank.THREE)
    medium = list(cards_for_rank(Rank.FOUR)[:2])
    high = _card(Rank.ACE)
    used = {low.card_id, high.card_id, *(c.card_id for c in medium)}
    hands = [[low], medium, [high], [c for c in full_deck() if c.card_id not in used]]
    return RoundState.from_hands(hands, level_rank=Rank.SIX)


def _fresh(api, state=None, *, max_token_steps=4096, spec=None):
    state = _checkpoint() if state is None else state
    spec = api.ObservationSpec() if spec is None else spec
    protocol = StepwiseActionState(
        state.clone(), max_action_tokens=spec.max_action_tokens,
        max_legal_next_tokens=spec.max_legal_next_tokens,
        max_token_steps=max_token_steps,
    )
    assert protocol.token_step == protocol.committed_step == 0
    assert protocol.prefix.tokens == () and not protocol.done and not protocol.truncated
    config = api.GameConfig(
        level_rank=int(state.level_rank), seed=301, leader_seat=state.active_seat,
        previous_result=state.previous_result, max_token_steps=max_token_steps,
    )
    env = api.GuandanEnv(config, observation_spec=spec)
    obs = env.reset(initial_state=protocol.serialize())
    return env, obs


def _deal_state(seed=301, *, previous=None):
    return RoundState.deal(seed=seed, level_rank=Rank.SIX, leader_seat=0, previous_result=previous)


def _swap_cards(state, left, right):
    other = state.clone()
    other.hands[left][0], other.hands[right][0] = other.hands[right][0], other.hands[left][0]
    other.check_invariants()
    return other


def _same(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype and left.shape == right.shape
        np.testing.assert_array_equal(left, right)
    elif is_dataclass(left):
        assert type(left) is type(right)
        for field in fields(left):
            _same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        assert left == right


def _check_obs(api, obs):
    assert isinstance(obs, api.Observation)
    assert 0 <= obs.player_id < 4 and obs.team_id == (obs.player_id % 2)
    assert obs.phase in {"LEAD_PLAY", "FOLLOW_PLAY", "TRIBUTE", "RETURN"}
    assert obs.private_hand_card_ids is None
    for name, (shape, dtype) in OBS_ARRAYS.items():
        value = getattr(obs, name)
        assert isinstance(value, np.ndarray), name
        assert value.shape == shape and value.dtype == np.dtype(dtype), name
    assert np.isfinite(obs.state_channels).all()
    assert np.all((obs.observation_tokens >= 0) & (obs.observation_tokens < 256))
    mask = obs.legal_next_mask
    assert mask.any()
    assert np.all(obs.legal_next_tokens[mask] > 0)
    assert np.all(obs.legal_next_tokens[mask] < 256)
    for name in ("legal_next_tokens", "legal_next_is_commit", "legal_next_kinds", "legal_next_values"):
        assert np.all(getattr(obs, name)[~mask] == 0), name
    assert len(set(obs.legal_next_tokens[mask].tolist())) == int(mask.sum())
    np.testing.assert_array_equal(obs.legal_next_is_commit[mask], np.isin(obs.legal_next_tokens[mask], [PASS_TOKEN, COMMIT_TOKEN]))


def _check_result(api, result):
    assert isinstance(result, api.StepResult)
    assert result.rewards.shape == (4,) and result.rewards.dtype == np.float32
    assert np.isfinite(result.rewards).all()
    assert all(type(getattr(result, name)) is bool for name in ("done", "truncated", "is_commit"))
    assert not (result.done and result.truncated)
    assert isinstance(result.info, dict)
    if not result.is_commit:
        assert result.committed_action is None
        np.testing.assert_array_equal(result.rewards, np.zeros(4, np.float32))
    if result.done or result.truncated:
        assert result.observation is None
    else:
        _check_obs(api, result.observation)
        assert result.observation.token_step == result.token_step
        assert result.observation.committed_step == result.committed_step
    if result.truncated:
        assert result.info["termination_reason"]


def _policy_safe(value):
    """Policy transport must not carry privileged engine objects or hidden maps."""
    assert not isinstance(value, (RoundState, StepwiseActionState, CommittedAction, ProtocolCommittedAction))
    if isinstance(value, np.ndarray):
        assert value.dtype != object
    elif is_dataclass(value):
        for field in fields(value):
            _policy_safe(getattr(value, field.name))
    elif isinstance(value, dict):
        forbidden = {"round_state", "hands", "protocol", "rng_state", "tribute_escrow", "tribute_selections"}
        assert not forbidden.intersection(value)
        for item in value.values():
            _policy_safe(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _policy_safe(item)
    elif value is not None:
        assert isinstance(value, (str, bool, int, float, np.generic))

def _play_single(env, card_id):
    env.step(SINGLE)
    env.step(TokenCodec.card(card_id))
    return env.step(COMMIT_TOKEN)


def _before_terminal(api, *, max_token_steps=4096):
    """Reach seat 2's winning single ONLY by the legal seat0/PASS path."""
    env, _ = _fresh(api, max_token_steps=max_token_steps)
    first = _play_single(env, _card(Rank.THREE).card_id)
    assert not first.done and first.observation.player_id == 1
    passed = env.step(PASS_TOKEN)
    assert not passed.done and passed.observation.player_id == 2
    assert env.full_state()["round_state"]["finished_ranks"] == [0]
    assert (env.token_step, env.committed_step) == (4, 2)
    return env


def _exchange_env(api):
    env = api.GuandanEnv(api.GameConfig(level_rank=Rank.SIX, seed=0, previous_result=PREVIOUS))
    obs = env.reset()
    assert (obs.phase, obs.player_id) == ("TRIBUTE", 1)
    return env


def _exchange_commit(env, card_id=None):
    token = int(env.legal_tokens()[0]) if card_id is None else TokenCodec.card(card_id)
    assert token in env.legal_tokens()
    prefix = env.step(token)
    assert not prefix.is_commit and prefix.committed_action is None
    assert env.legal_tokens().tolist() == [COMMIT_TOKEN]
    result = env.step(COMMIT_TOKEN)
    assert result.is_commit and not result.done and not result.truncated
    return result


def _return_env(api):
    env = _exchange_env(api)
    first = _exchange_commit(env)
    assert (first.observation.phase, first.observation.player_id) == ("TRIBUTE", 3)
    second = _exchange_commit(env)
    assert (second.observation.phase, second.observation.player_id) == ("RETURN", 0)
    assert (env.token_step, env.committed_step) == (4, 2)
    return env


def _partition(fixed):
    """Fresh reduced-hand fixture retaining every physical card exactly once."""
    used = {c.card_id for hand in fixed.values() for c in hand}
    assert len(used) == sum(map(len, fixed.values()))
    hands = [list(fixed.get(seat, ())) for seat in range(4)]
    open_seats = [seat for seat in range(4) if seat not in fixed]
    for index, card in enumerate(c for c in full_deck() if c.card_id not in used):
        hands[open_seats[index % len(open_seats)]].append(card)
    return RoundState.from_hands(hands, level_rank=Rank.SIX)


def _batch_from(api, envs, *, auto_reset=False):
    batch = api.GuandanEnvBatch([env.config for env in envs], auto_reset=auto_reset,
                                observation_spec=envs[0].observation_spec)
    batch.load_serialized([env.serialize() for env in envs])
    return batch


def _masked(obs):
    return obs.legal_next_tokens[obs.legal_next_mask]


def _used_tokens(obs):
    nonzero = np.flatnonzero(obs.observation_tokens)
    return int(nonzero[-1] + 1) if nonzero.size else 0


def _choose(env):
    legal = env.legal_tokens().tolist()
    for token in (PASS_TOKEN, COMMIT_TOKEN, SINGLE, PAIR):
        if token in legal:
            return token
    return int(legal[0])


def test_A04_001_config_is_validated_and_preserved_by_environment(api):
    config = api.GameConfig(level_rank=Rank.SIX, seed=301, leader_seat=2, max_token_steps=23)
    env = api.GuandanEnv(config)
    assert env.config == config
    assert env.config.max_token_steps == 23
    with pytest.raises((ValueError, TypeError)):
        api.GameConfig(level_rank=Rank.SIX, leader_seat=4)
    with pytest.raises((ValueError, TypeError)):
        api.GameConfig(level_rank=Rank.SIX, max_token_steps=0)


def test_A04_002_public_api_and_type_reexports(api):
    types = importlib.import_module("guandan.environment_types")
    for name in ("GameConfig", "ObservationSpec", "Observation", "StepResult", "GuandanEnv", "GuandanEnvBatch"):
        assert hasattr(api, name), name
    for name in ("GameConfig", "ObservationSpec", "Observation", "StepResult"):
        assert getattr(api, name) is getattr(types, name)


def test_A04_003_fixed_observation_spec(api):
    spec = api.ObservationSpec()
    assert {field.name: getattr(spec, field.name) for field in fields(spec)} == {
        "token_vocab_size": 256, "pad_token_id": 0, "max_observation_tokens": 4096,
        "max_action_tokens": 32, "max_legal_next_tokens": 256, "num_state_channels": 256,
    }


def test_A04_004_explicit_reset_is_required(api):
    env = api.GuandanEnv(api.GameConfig(level_rank=Rank.SIX, seed=301))
    assert env.observe() is None
    assert (env.token_step, env.committed_step) == (0, 0)
    with pytest.raises(RuntimeError):
        env.step(SINGLE)
    with pytest.raises(RuntimeError):
        env.legal_tokens()
    with pytest.raises(RuntimeError):
        env.serialize()


def test_A04_005_reset_observation_has_exact_fields_shapes_and_dtypes(api):
    env = api.GuandanEnv(api.GameConfig(level_rank=Rank.SIX, seed=301, leader_seat=2))
    obs = env.reset()
    _check_obs(api, obs)
    assert (obs.player_id, obs.team_id, obs.phase, obs.token_step, obs.committed_step) == (2, 0, "LEAD_PLAY", 0, 0)


def test_A04_006_reset_deals_108_cards_as_four_27_card_hands(api):
    env = api.GuandanEnv(api.GameConfig(level_rank=Rank.SIX, seed=301))
    env.reset()
    hands = env.full_state()["round_state"]["hands"]
    assert [len(hand) for hand in hands] == [27] * 4
    assert sorted(card for hand in hands for card in hand) == list(range(108))


def test_A04_007_seed_and_leader_are_deterministic(api):
    config = api.GameConfig(level_rank=Rank.SIX, seed=301, leader_seat=3)
    a, b = api.GuandanEnv(config), api.GuandanEnv(config)
    _same(a.reset(), b.reset())
    assert a.serialize() == b.serialize()
    assert a.observe().player_id == 3


def test_A04_008_explicit_seed_overrides_config_without_coercion(api):
    a = api.GuandanEnv(api.GameConfig(level_rank=Rank.SIX, seed=301))
    b = api.GuandanEnv(api.GameConfig(level_rank=Rank.SIX, seed=999))
    _same(a.reset(seed=41), b.reset(seed=41))
    with pytest.raises(ValueError):
        a.reset(seed=True)


def test_A04_009_known_previous_result_starts_at_first_tribute_donor(api):
    env = _exchange_env(api)
    obs = env.observe()
    assert (obs.phase, obs.player_id, obs.team_id) == ("TRIBUTE", 1, 1)
    assert obs.token_step == obs.committed_step == 0
    assert not env.full_state()["round_state"]["anti_tribute"]


def test_A04_010_fresh_a03_and_environment_payloads_are_accepted(api):
    env, obs = _fresh(api)
    a03 = StepwiseActionState(_checkpoint()).serialize()
    clone_a = api.GuandanEnv(env.config, observation_spec=env.observation_spec)
    clone_b = api.GuandanEnv(env.config, observation_spec=env.observation_spec)
    _same(clone_a.reset(initial_state=a03), obs)
    _same(clone_b.reset(initial_state=env.serialize()), obs)
    assert clone_a.serialize() == clone_b.serialize()


def test_A04_011_reset_rejects_resume_payload_and_preserves_state(api):
    env, _ = _fresh(api)
    env.step(SINGLE)
    before = env.serialize()
    with pytest.raises(ProtocolError, match="fresh|resume"):
        env.reset(initial_state=before)
    assert env.serialize() == before


def test_A04_012_observation_legal_rows_are_padded_zero_and_unique(api):
    env, obs = _fresh(api)
    assert env.legal_tokens().dtype == np.int32
    np.testing.assert_array_equal(env.legal_tokens(), _masked(obs))
    assert _masked(obs).size <= 256
    assert not np.any(obs.legal_next_tokens[~obs.legal_next_mask])
    assert not np.any(obs.legal_next_is_commit[~obs.legal_next_mask])
    assert not np.any(obs.legal_next_kinds[~obs.legal_next_mask])
    assert not np.any(obs.legal_next_values[~obs.legal_next_mask])


def test_A04_013_lead_legal_mask_excludes_pad_phase_reserved_pass_and_incomplete_commit(api):
    _, obs = _fresh(api)
    tokens = set(_masked(obs).tolist())
    forbidden = {0, 1, 2, 3, 4, 5, PASS_TOKEN, COMMIT_TOKEN, *range(49, 64), *range(224, 256)}
    assert not tokens & forbidden
    assert not obs.legal_next_is_commit.any()


def test_A04_014_follow_legal_mask_contains_pass_as_commit(api):
    env, _ = _fresh(api)
    card_id = env.full_state()["round_state"]["hands"][0][0]
    _play_single(env, card_id)
    obs = env.observe()
    assert obs.phase == "FOLLOW_PLAY"
    assert PASS_TOKEN in _masked(obs)
    row = np.flatnonzero(obs.legal_next_tokens == PASS_TOKEN)
    assert row.size == 1 and bool(obs.legal_next_is_commit[row[0]])


def test_A04_015_legal_mask_is_complete_for_single_card_prefix(api):
    env, _ = _fresh(api)
    state = env.full_state()["round_state"]
    env.step(SINGLE)
    expected = sorted(TokenCodec.card(card_id) for card_id in state["hands"][0])
    assert _masked(env.observe()).tolist() == expected


def test_A04_016_tribute_mask_is_exact_and_includes_eligible_jokers(api):
    env = _exchange_env(api)
    state = RoundState.from_serialized(env.full_state()["round_state"])
    expected = tribute_card_for(state, 1)
    assert expected is not None
    assert _masked(env.observe()).tolist() == [TokenCodec.card(expected.card_id)]
    assert expected.is_joker
    assert not expected.is_level_wild(Rank.SIX)
    assert not env.observe().legal_next_is_commit.any()
    prefix = env.step(TokenCodec.card(expected.card_id))
    assert _masked(prefix.observation).tolist() == [COMMIT_TOKEN]
    assert prefix.observation.legal_next_is_commit[prefix.observation.legal_next_mask].tolist() == [True]


def test_A04_017_return_mask_is_the_complete_natural_2_to_10_set_excluding_current_level(api):
    env = _return_env(api)
    obs = env.observe()
    assert (obs.phase, obs.player_id) == ("RETURN", 0)
    state = RoundState.from_serialized(env.full_state()["round_state"])
    ranks = {Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
             Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN}
    expected = sorted(TokenCodec.card(c.card_id) for c in state.hands[0]
                      if c.rank in ranks and c.rank is not Rank.SIX and not c.is_level_wild(Rank.SIX))
    assert expected and any(c.rank is Rank.TWO for c in return_cards_for(state, 0))
    assert expected == sorted(TokenCodec.card(c.card_id) for c in return_cards_for(state, 0))
    assert _masked(obs).tolist() == env.legal_tokens().tolist() == expected
    assert not obs.legal_next_is_commit.any()


def test_A04_018_policy_observation_has_no_private_hand_field(api):
    _, obs = _fresh(api)
    assert obs.private_hand_card_ids is None
    assert obs.state_channels.dtype == np.float32


def test_A04_019_hidden_opponent_swap_preserves_all_policy_observation_fields(api):
    state = _deal_state()
    swapped = _swap_cards(state, 1, 3)  # both opponents of the active seat 0
    a, obs_a = _fresh(api, state)
    b, obs_b = _fresh(api, swapped)
    assert state.serialize() != swapped.serialize()
    assert state.hands[0] == swapped.hands[0]
    _same(obs_a, obs_b)
    _same(a.step(SINGLE), b.step(SINGLE))


def test_A04_020_hidden_teammate_swap_preserves_all_policy_observation_fields(api):
    state = _deal_state()
    swapped = _swap_cards(state, 1, 2)  # seat 2 is active seat 0's teammate
    a, obs_a = _fresh(api, state)
    b, obs_b = _fresh(api, swapped)
    assert state.hands[2] != swapped.hands[2]
    assert state.hands[0] == swapped.hands[0]
    _same(obs_a, obs_b)
    _same(a.step(SINGLE), b.step(SINGLE))


def test_A04_021_active_self_hand_change_is_visible(api):
    state = _deal_state()
    swapped = _swap_cards(state, 0, 1)
    _, obs_a = _fresh(api, state)
    _, obs_b = _fresh(api, swapped)
    assert not (np.array_equal(obs_a.observation_tokens, obs_b.observation_tokens)
                and np.array_equal(obs_a.state_channels, obs_b.state_channels))


def test_A04_022_full_state_is_separate_debug_channel(api):
    env, obs = _fresh(api)
    assert "round_state" in env.full_state()
    _policy_safe(obs)
    with pytest.raises(TypeError):
        env.observe(include_private=True)
    result = _play_single(env, _card(Rank.THREE).card_id)
    assert not hasattr(result.committed_action, "atomic_action")
    _policy_safe(result)

def test_A04_023_unresolved_tribute_receipt_redacts_card_identity(api):
    env = _exchange_env(api)
    result = _exchange_commit(env)
    _check_result(api, result)
    assert (result.observation.phase, result.observation.player_id) == ("TRIBUTE", 3)
    receipt = result.committed_action
    assert (receipt.actor, receipt.phase, receipt.kind) == (1, "TRIBUTE", "TRIBUTE")
    assert receipt.card_ids == receipt.declared_ranks == receipt.wild_assignments == ()
    assert receipt.family is None and receipt.declared_suit is None
    assert "resolved_transfers" not in result.info
    _policy_safe(result)

def test_A04_024_unresolved_tribute_hidden_worlds_have_identical_results(api):
    state = _deal_state(seed=2, previous=PREVIOUS)
    first = tribute_card_for(state, 1)
    assert first is not None
    other = next(card for card in state.hands[0] if card.rank is first.rank and not card.is_level_wild(Rank.SIX))
    owner = 0
    alternate = state.clone()
    donor_index, owner_index = alternate.hands[1].index(first), alternate.hands[owner].index(other)
    alternate.hands[1][donor_index], alternate.hands[owner][owner_index] = other, first
    alternate.check_invariants()
    assert tribute_card_for(alternate, 1) == other
    a, _ = _fresh(api, state); b, _ = _fresh(api, alternate)
    result_a, result_b = _exchange_commit(a), _exchange_commit(b)
    assert result_a.observation.player_id == result_b.observation.player_id == 3
    assert a.full_state()["round_state"] != b.full_state()["round_state"]
    _same(result_a, result_b)
    assert "resolved_transfers" not in result_a.info

def test_A04_025_prefix_increments_token_only_and_has_zero_reward(api):
    env, initial = _fresh(api)
    before_round = env.full_state()["round_state"]
    before_rng = env.full_state()["rng_state"]
    for index, token in enumerate((SINGLE, TokenCodec.card(_card(Rank.THREE).card_id)), 1):
        result = env.step(token)
        _check_result(api, result)
        assert (result.token_step, result.committed_step) == (index, 0)
        assert not result.is_commit and result.committed_action is None
        assert result.observation.player_id == initial.player_id
        assert env.full_state()["round_state"] == before_round
        assert env.full_state()["rng_state"] == before_rng
    assert env.full_state()["protocol"]["prefix_tokens"] == [SINGLE, TokenCodec.card(_card(Rank.THREE).card_id)]
    assert not np.array_equal(initial.observation_tokens, result.observation.observation_tokens)

def test_A04_026_play_commit_increments_both_counters_and_clears_prefix(api):
    env, _ = _fresh(api)
    card_id = env.full_state()["round_state"]["hands"][0][0]
    result = _play_single(env, card_id)
    _check_result(api, result)
    assert result.is_commit and (result.token_step, result.committed_step) == (3, 1)
    assert result.committed_action.card_ids == (card_id,)
    assert env.full_state()["protocol"]["prefix_tokens"] == []


def test_A04_027_pass_is_a_single_committed_step_with_zero_nonterminal_reward(api):
    env, _ = _fresh(api)
    card_id = env.full_state()["round_state"]["hands"][0][0]
    _play_single(env, card_id)
    before = env.full_state()["round_state"]["hands"]
    result = env.step(PASS_TOKEN)
    _check_result(api, result)
    assert result.is_commit and result.committed_action.kind == "PASS"
    assert (result.token_step, result.committed_step) == (4, 2)
    assert env.full_state()["round_state"]["hands"] == before


def test_A04_028_step_result_exposes_all_frozen_fields(api):
    env, _ = _fresh(api)
    result = env.step(SINGLE)
    assert tuple(field.name for field in fields(result)) == (
        "observation", "rewards", "done", "truncated", "is_commit",
        "committed_action", "token_step", "committed_step", "info",
    )
    _check_result(api, result)


def test_A04_029_terminal_commit_has_shared_zero_sum_reward_and_ranking(api):
    env = _before_terminal(api)
    result = _play_single(env, _card(Rank.ACE).card_id)
    _check_result(api, result)
    assert result.done and not result.truncated and result.is_commit
    np.testing.assert_array_equal(result.rewards, np.array([1, -1, 1, -1], np.float32))
    assert float(result.rewards.sum()) == 0.0
    assert (result.token_step, result.committed_step) == (7, 3)
    state = env.full_state()["round_state"]
    assert state["ranking"] == [1, 4, 2, 3]
    assert state["winner_team"] == 0 and state["outcome_class"] == "DOUBLE_DOWN"
    assert result.info["ranking"] == (1, 4, 2, 3)
    assert result.info["winner_team"] == 0 and result.info["outcome_class"] == "DOUBLE_DOWN"
    assert result.info["team_reward"] == (1.0, -1.0)
    assert result.info["finish_token_step"] == state["finish_token_step"] == 7
    assert result.info["finish_committed_step"] == state["finish_committed_step"] == 3
    assert env.full_state()["protocol"]["prefix_tokens"] == []
    RoundState.from_serialized(state).check_invariants()

def test_A04_030_base_terminal_does_not_autoreset_or_replay_reward(api):
    env = _before_terminal(api)
    result = _play_single(env, _card(Rank.ACE).card_id)
    assert result.done and env.done
    terminal = env.serialize()
    assert env.observe() is None and env.legal_tokens().size == 0
    for token in (PASS_TOKEN, COMMIT_TOKEN, 0):
        with pytest.raises(EpisodeTerminatedError):
            env.step(token)
        assert env.serialize() == terminal

def test_A04_031_max_token_steps_truncates_without_reward(api):
    env, _ = _fresh(api, max_token_steps=1)
    result = env.step(SINGLE)
    _check_result(api, result)
    assert result.truncated and not result.done
    assert result.info["termination_reason"] == "max_token_steps"
    assert (result.token_step, result.committed_step) == (1, 0)
    np.testing.assert_array_equal(result.rewards, np.zeros(4, dtype=np.float32))


def test_A04_032_post_truncation_step_is_rejected_without_mutation(api):
    env, _ = _fresh(api, max_token_steps=1)
    env.step(SINGLE)
    before = env.serialize()
    with pytest.raises(EpisodeTerminatedError):
        env.step(COMMIT_TOKEN)
    assert env.serialize() == before and env.observe() is None


@pytest.mark.parametrize("token", [True, False, 0, 1, 49, 255, 256, -1, 2**40, PASS_TOKEN, COMMIT_TOKEN],
                         ids=["bool-true", "bool-false", "pad", "bos", "reserved-low", "reserved-high", "vocab-overflow", "negative", "int-overflow", "lead-pass", "incomplete-commit"])
def test_A04_033_invalid_token_rolls_back_every_byte(api, token):
    env, obs = _fresh(api)
    before = env.serialize()
    with pytest.raises(IllegalActionError):
        env.step(token)
    assert env.serialize() == before
    _same(env.observe(), obs)
    assert (env.token_step, env.committed_step) == (0, 0)


def test_A04_034_duplicate_physical_card_rolls_back_partial_pair_prefix(api):
    pair = sorted(cards_for_rank(Rank.THREE)[:2], key=lambda card: card.card_id)
    state = _partition({0: pair + [_card(Rank.NINE)]})
    assert sorted(card.card_id for hand in state.hands for card in hand) == list(range(108))
    env, _ = _fresh(api, state)
    for token in (PAIR, TokenCodec.rank(Rank.THREE), TokenCodec.card(pair[0].card_id)):
        assert token in env.legal_tokens()
        env.step(token)
    assert env.legal_tokens().tolist() == [TokenCodec.card(pair[1].card_id)]
    before, obs = env.serialize(), env.observe()
    with pytest.raises(IllegalActionError):
        env.step(TokenCodec.card(pair[0].card_id))
    assert env.serialize() == before
    _same(env.observe(), obs)
    assert (env.token_step, env.committed_step) == (3, 0)
    env.step(TokenCodec.card(pair[1].card_id))
    committed = env.step(COMMIT_TOKEN)
    assert committed.is_commit and committed.committed_action.card_ids == tuple(card.card_id for card in pair)

@pytest.mark.parametrize("value", [None, "8", [8], np.array(8, dtype=np.int32), np.array([8], dtype=np.int32)])
def test_A04_035_non_scalar_token_is_not_coerced(api, value):
    env, _ = _fresh(api)
    before = env.serialize()
    with pytest.raises(IllegalActionError):
        env.step(value)
    assert env.serialize() == before


def test_A04_036_clone_is_independent_and_equivalent(api):
    env, _ = _fresh(api)
    env.step(SINGLE)
    clone = env.clone()
    assert clone is not env and clone.serialize() == env.serialize()
    card_id = env.full_state()["round_state"]["hands"][0][0]
    _same(env.step(TokenCodec.card(card_id)), clone.step(TokenCodec.card(card_id)))
    assert env.serialize() == clone.serialize()


def test_A04_037_serialization_continues_mid_prefix_exactly(api):
    env, _ = _fresh(api)
    env.step(SINGLE)
    payload = env.serialize()
    restored = api.GuandanEnv.deserialize(payload)
    assert restored.serialize() == payload
    card_id = env.full_state()["round_state"]["hands"][0][0]
    for token in (TokenCodec.card(card_id), COMMIT_TOKEN, PASS_TOKEN):
        _same(env.step(token), restored.step(token))
        assert env.serialize() == restored.serialize()


def test_A04_038_serialization_preserves_terminal_and_truncated_guards(api):
    truncated, _ = _fresh(api, max_token_steps=1)
    truncated.step(SINGLE)
    restored = api.GuandanEnv.deserialize(truncated.serialize())
    assert restored.truncated and restored.observe() is None
    with pytest.raises(EpisodeTerminatedError):
        restored.step(COMMIT_TOKEN)


def test_A04_039_serialization_preserves_rng_and_future_resets(api):
    a = api.GuandanEnv(api.GameConfig(level_rank=Rank.SIX, seed=41))
    a.reset()
    restored = api.GuandanEnv.deserialize(a.serialize())
    _same(a.reset(), restored.reset())
    _same(a.reset(seed=55), restored.reset(seed=55))
    assert a.serialize() == restored.serialize()


@pytest.mark.parametrize("field", ["rules_version", "action_version", "environment_version", "encoding_version", "serialization_version"])
def test_A04_040_incompatible_serialization_version_is_rejected(api, field):
    env, _ = _fresh(api)
    payload = json.loads(env.serialize())
    payload[field] = "future" if field != "serialization_version" else 999
    with pytest.raises(ProtocolError):
        api.GuandanEnv.deserialize(json.dumps(payload).encode())


def test_A04_041_array_and_full_state_results_are_detached(api):
    env, obs = _fresh(api)
    before = env.serialize()
    for field in fields(obs):
        value = getattr(obs, field.name)
        if isinstance(value, np.ndarray) and value.flags.writeable:
            value.fill(0)
    debug = env.full_state()
    debug["protocol"]["round_state"]["hands"][0].clear()
    assert env.serialize() == before
    _check_obs(api, env.observe())


def test_A04_042_batch_observation_and_result_leading_dimensions(api):
    configs = [api.GameConfig(level_rank=Rank.SIX, seed=301 + i, leader_seat=i) for i in range(3)]
    batch = api.GuandanEnvBatch(configs)
    obs = batch.reset(seeds=[41, 42, 43])
    for name, (shape, dtype) in OBS_ARRAYS.items():
        assert getattr(obs, name).shape == (3, *shape)
        assert getattr(obs, name).dtype == np.dtype(dtype)
    for name in ("player_id", "team_id", "token_step", "committed_step"):
        assert getattr(obs, name).shape == (3,)
    assert obs.present.dtype == np.bool_ and obs.present.tolist() == [True, True, True]
    result = batch.step(np.array([SINGLE, SINGLE, SINGLE], dtype=np.int32))
    assert result.rewards.shape == (3, 4) and result.rewards.dtype == np.float32
    for name in ("done", "truncated", "is_commit"):
        assert getattr(result, name).shape == (3,) and getattr(result, name).dtype == np.bool_
    assert result.token_step.shape == result.committed_step.shape == (3,)


def test_A04_043_batch_matches_single_slots_for_seeded_token_trace(api):
    configs = [api.GameConfig(level_rank=Rank.SIX, seed=301 + i, leader_seat=i) for i in range(2)]
    singles = [api.GuandanEnv(c) for c in configs]
    batch = api.GuandanEnvBatch(configs)
    batch_obs = batch.reset(seeds=[41, 42])
    for index, env in enumerate(singles):
        _same(batch_obs.observation_tokens[index], env.reset(seed=41 + index).observation_tokens)
    for _ in range(5):
        tokens = np.array([_choose(env) for env in singles], dtype=np.int32)
        single_results = [env.step(int(token)) for env, token in zip(singles, tokens)]
        batch_result = batch.step(tokens)
        for index, single_result in enumerate(single_results):
            _same(batch_result.observation.observation_tokens[index], single_result.observation.observation_tokens if single_result.observation is not None else np.zeros(4096, np.int32))
            assert batch_result.done[index] == single_result.done
            assert batch_result.truncated[index] == single_result.truncated
            np.testing.assert_array_equal(batch_result.rewards[index], single_result.rewards)
            assert batch_result.committed_action[index] == single_result.committed_action
        assert batch.serialize() == [env.serialize() for env in singles]


def test_A04_044_batch_slots_are_independent_and_rng_is_not_shared(api):
    configs = [api.GameConfig(level_rank=Rank.SIX, seed=301), api.GameConfig(level_rank=Rank.SIX, seed=302)]
    batch = api.GuandanEnvBatch(configs)
    batch.reset()
    before = batch.serialize()
    assert batch.envs[0] is not batch.envs[1]
    batch.envs[0].step(SINGLE)
    assert batch.serialize()[0] != before[0]
    assert batch.serialize()[1] == before[1]
    batch.reset_at(0, seed=777)
    assert batch.serialize()[1] == before[1]


def test_A04_045_batch_reset_at_only_replaces_requested_slot(api):
    config = api.GameConfig(level_rank=Rank.SIX, seed=301)
    batch = api.GuandanEnvBatch([config, config])
    batch.reset(seeds=[11, 12])
    untouched = batch.serialize()[1]
    observation = batch.reset_at(0, seed=13)
    _check_obs(api, observation)
    assert batch.serialize()[1] == untouched
    assert batch.envs[0].token_step == 0 and batch.envs[1].token_step == 0


def test_A04_046_batch_inactive_slot_accepts_only_pad_and_is_transactional(api):
    truncated, _ = _fresh(api, max_token_steps=1)
    truncated.step(SINGLE)
    live, _ = _fresh(api)
    batch = _batch_from(api, [truncated, live])
    before = batch.serialize()
    with pytest.raises(EpisodeTerminatedError):
        batch.step(np.array([SINGLE, SINGLE], dtype=np.int32))
    assert batch.serialize() == before
    result = batch.step(np.array([0, SINGLE], dtype=np.int32))
    assert result.truncated.tolist() == [True, False]
    assert result.info[0]["inactive"] is True


def test_A04_047_batch_later_invalid_token_rolls_back_earlier_slot(api):
    a, _ = _fresh(api); b, _ = _fresh(api)
    batch = _batch_from(api, [a, b])
    before = batch.serialize()
    with pytest.raises(IllegalActionError):
        batch.step(np.array([SINGLE, COMMIT_TOKEN], dtype=np.int32))
    assert batch.serialize() == before


@pytest.mark.parametrize("tokens", [np.array([SINGLE]), np.array([[SINGLE, SINGLE]], dtype=np.int32), np.array([8.0, 8.0]), np.array([True, True])], ids=["short", "matrix", "float", "bool"])
def test_A04_048_batch_input_shape_dtype_is_strict(api, tokens):
    env, _ = _fresh(api)
    batch = _batch_from(api, [env, env.clone()])
    before = batch.serialize()
    with pytest.raises((TypeError, ValueError)):
        batch.step(tokens)
    assert batch.serialize() == before


def test_A04_049_batch_load_serialized_continues_each_prefix(api):
    a, _ = _fresh(api); b, _ = _fresh(api)
    a.step(SINGLE); b.step(SINGLE)
    batch = _batch_from(api, [a, b])
    restored = api.GuandanEnvBatch([a.config, b.config])
    restored.load_serialized(batch.serialize())
    assert restored.serialize() == batch.serialize()
    for _ in range(2):
        tokens = np.array([_choose(env) for env in batch.envs], dtype=np.int32)
        left, right = batch.step(tokens), restored.step(tokens)
        _same(left, right)


def test_A04_050_batch_load_is_atomic_on_later_bad_payload(api):
    env, _ = _fresh(api)
    batch = _batch_from(api, [env, env.clone()])
    before = batch.serialize()
    changed = env.clone(); changed.step(SINGLE)
    with pytest.raises((TypeError, ValueError, ProtocolError)):
        batch.load_serialized([changed.serialize(), b"not-json"])
    assert batch.serialize() == before


def test_A04_051_auto_reset_returns_terminal_transition_and_new_observation(api):
    env, _ = _fresh(api, max_token_steps=1)
    batch = _batch_from(api, [env], auto_reset=True)
    result = batch.step(np.array([SINGLE], dtype=np.int32))
    assert result.truncated.tolist() == [True] and result.done.tolist() == [False]
    assert result.info[0]["auto_reset"] is True
    assert result.info[0]["final_info"]["termination_reason"] == "max_token_steps"
    assert result.observation.present.tolist() == [True]
    assert result.observation.token_step.tolist() == [0]
    assert batch.envs[0].token_step == 0 and not batch.envs[0].truncated


def test_A04_052_auto_reset_preserves_other_live_slot(api):
    ending, _ = _fresh(api, max_token_steps=1)
    live, _ = _fresh(api)
    batch = _batch_from(api, [ending, live], auto_reset=True)
    expected = live.step(SINGLE)
    result = batch.step(np.array([SINGLE, SINGLE], dtype=np.int32))
    assert result.truncated.tolist() == [True, False]
    assert batch.envs[1].serialize() == live.serialize()
    assert result.observation.token_step.tolist() == [0, expected.token_step]




def test_A04_053_first_return_receipt_remains_redacted_until_exchange_resolves(api):
    env = _return_env(api)
    state_before = RoundState.from_serialized(env.full_state()["round_state"])
    expected = sorted(TokenCodec.card(card.card_id) for card in return_cards_for(state_before, 0))
    assert Rank.TWO in {card.rank for card in return_cards_for(state_before, 0)}
    assert _masked(env.observe()).tolist() == expected
    first = _exchange_commit(env, TokenCodec.card_from_token(int(expected[0])))
    _check_result(api, first)
    assert (first.observation.phase, first.observation.player_id) == ("RETURN", 2)
    assert first.committed_action.phase == "RETURN"
    assert first.committed_action.card_ids == ()
    assert "resolved_transfers" not in first.info


def test_A04_054_final_return_publishes_complete_resolved_transfer_identities(api):
    env = _return_env(api)
    first_state = RoundState.from_serialized(env.full_state()["round_state"])
    first_card = return_cards_for(first_state, 0)[0]
    first_transfer = (0, first_state.return_obligations[0][1], first_card.card_id)
    _exchange_commit(env, first_card.card_id)
    assert env.observe().phase == "RETURN" and env.observe().player_id == 2
    final_state = RoundState.from_serialized(env.full_state()["round_state"])
    final_cards = return_cards_for(final_state, 2)
    assert final_cards and Rank.TWO in {card.rank for card in final_cards}
    second_transfer = (2, final_state.return_obligations[0][1], final_cards[0].card_id)
    result = _exchange_commit(env, final_cards[0].card_id)
    assert result.observation.phase == "LEAD_PLAY"
    assert result.committed_action.phase == "RETURN"
    assert result.committed_action.card_ids == (final_cards[0].card_id,)
    assert "resolved_transfers" in result.info
    resolved = result.info["resolved_transfers"]
    after_state = RoundState.from_serialized(env.full_state()["round_state"])
    expected_tribute = tuple((donor, recipient, card.card_id)
                             for donor, recipient, card in after_state.tribute_transfers)
    expected_return = tuple((recipient, donor, card.card_id)
                            for recipient, donor, card in after_state.return_transfers)
    assert tuple(resolved["tribute"]) == expected_tribute
    assert tuple(resolved["return"]) == expected_return
    assert expected_return == (first_transfer, second_transfer)
    assert len(resolved["tribute"]) == 2 and len(resolved["return"]) == 2

def test_A04_055_completed_commit_payload_is_rejected_by_reset_but_deserialize_resumes(api):
    env, _ = _fresh(api)
    _play_single(env, _card(Rank.THREE).card_id)
    payload = env.serialize()
    target, _ = _fresh(api)
    before = target.serialize()
    with pytest.raises(ProtocolError, match="fresh|zero counters|resume"):
        target.reset(initial_state=payload)
    assert target.serialize() == before
    restored = api.GuandanEnv.deserialize(payload)
    assert restored.serialize() == payload
    assert restored.committed_step == 1 and restored.token_step == 3


def test_A04_056_seed2_level_six_natural_heart_straight_flush_commits_once(api):
    env = api.GuandanEnv(api.GameConfig(level_rank=Rank.SIX, seed=2))
    obs = env.reset()
    assert (obs.phase, obs.player_id) == ("LEAD_PLAY", 0)
    # Seed 2 / level SIX has the natural 5H-9H collision: the heart SIX is
    # both a physical level wild and a natural straight-flush card.
    tokens = (16, 35, 20, 74, 78, 82, 140, 144, 186, COMMIT_TOKEN)
    for token in tokens[:-1]:
        result = env.step(token)
        assert not result.is_commit
    result = env.step(tokens[-1])
    assert result.is_commit and result.committed_action is not None
    assert result.committed_action.kind == "STRAIGHT_FLUSH"
    assert result.committed_action.card_ids == (10, 14, 18, 76, 80)
    assert result.committed_action.declared_suit == int(Suit.HEARTS)
    assert result.committed_action.wild_assignments == ((14, int(Rank.SIX), int(Suit.HEARTS)),)
    assert env.committed_step == 1 and env.token_step == len(tokens)
    assert env.full_state()["round_state"]["history"][-1]["cards"] == [10, 14, 18, 76, 80]


def test_A04_057_observation_capacity_overflow_is_explicit_and_transactional(api):
    with pytest.raises(ProtocolError, match="max_observation_tokens|capacity|overflow"):
        _fresh(api, spec=api.ObservationSpec(max_observation_tokens=1))
    env, _ = _fresh(api)
    before = env.serialize()
    original = importlib.import_module("guandan.environment").encode_observation
    module = importlib.import_module("guandan.environment")
    module.encode_observation = lambda *args, **kwargs: (_ for _ in ()).throw(ProtocolError("observation capacity overflow"))
    try:
        with pytest.raises(ProtocolError, match="observation|capacity|overflow"):
            env.step(SINGLE)
    finally:
        module.encode_observation = original
    assert env.serialize() == before


def test_A04_058_action_capacity_overflow_is_not_truncation(api):
    with pytest.raises(ProtocolError, match="action|max_action_tokens|capacity|overflow"):
        _fresh(api, spec=api.ObservationSpec(max_action_tokens=1))


def test_A04_059_legal_next_capacity_overflow_is_not_top_n_filtering(api):
    env = api.GuandanEnv(api.GameConfig(level_rank=Rank.SIX, seed=301), observation_spec=api.ObservationSpec(max_legal_next_tokens=1))
    with pytest.raises(ProtocolError, match="legal|max_legal_next_tokens|capacity|overflow"):
        env.reset()


def test_A04_060_batch_overflow_does_not_publish_any_slot(api):
    spec = api.ObservationSpec(max_observation_tokens=1)
    batch = api.GuandanEnvBatch([api.GameConfig(level_rank=Rank.SIX, seed=301),
                                 api.GameConfig(level_rank=Rank.SIX, seed=302)],
                                observation_spec=spec)
    with pytest.raises(ProtocolError, match="observation|capacity|overflow"):
        batch.reset(seeds=[1, 2])
    assert all(env.observe() is None for env in batch.envs)
