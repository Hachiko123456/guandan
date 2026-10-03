"""Single-slot A04 transaction, checkpoint and RNG regressions (main-owned)."""
from dataclasses import fields
import json

import numpy as np
import pytest

from guandan.action_state import COMMIT_TOKEN, PASS_TOKEN, StepwiseActionState, TokenCodec
from guandan.cards import Rank, cards_for_rank, full_deck
from guandan.combos import CombinationKind
from guandan.environment import GameConfig, GuandanEnv, ObservationSpec, ProtocolError
from guandan.state import EpisodeTerminatedError, IllegalActionError, RoundState


def checkpoint():
    low = cards_for_rank(Rank.THREE)[0]
    medium = list(cards_for_rank(Rank.FOUR)[:2])
    high = cards_for_rank(Rank.ACE)[0]
    used = {low.card_id, high.card_id, *(c.card_id for c in medium)}
    state = RoundState.from_hands([[low], medium, [high], [c for c in full_deck() if c.card_id not in used]], level_rank=Rank.SIX)
    return StepwiseActionState(state).serialize()


def fresh(**kwargs):
    env = GuandanEnv(GameConfig(level_rank=Rank.SIX, seed=301, **kwargs))
    env.reset(initial_state=checkpoint())
    return env


def compare_observations(a, b):
    if a is None or b is None:
        assert a is b
        return
    for item in fields(a):
        left, right = getattr(a, item.name), getattr(b, item.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def single_path(env, card_id):
    for token in (TokenCodec.family(CombinationKind.SINGLE), TokenCodec.card(card_id), COMMIT_TOKEN):
        result = env.step(token)
    return result


def test_explicit_reset_and_no_implicit_reset():
    env = GuandanEnv(GameConfig())
    assert env.observe() is None
    with pytest.raises(RuntimeError):
        env.step(8)
    with pytest.raises(RuntimeError):
        env.serialize()


@pytest.mark.parametrize('token', [True, False, 8.0, '8', None, [8], np.asarray(8), -1, 0, 255])
def test_invalid_scalar_preserves_every_byte(token):
    env = fresh()
    before = env.serialize()
    with pytest.raises(IllegalActionError):
        env.step(token)
    assert env.serialize() == before


def test_prefix_checkpoint_resume_next_commit_exact():
    env = fresh()
    first = env.step(TokenCodec.family(CombinationKind.SINGLE))
    assert not first.is_commit and first.token_step == 1 and first.committed_step == 0
    assert not first.rewards.any()
    restored = GuandanEnv.deserialize(env.serialize())
    assert env.serialize() == restored.serialize()
    compare_observations(env.observe(), restored.observe())
    card_id = env.full_state()['round_state']['hands'][0][0]
    for token in (TokenCodec.card(card_id), COMMIT_TOKEN):
        a, b = env.step(token), restored.step(token)
        compare_observations(a.observation, b.observation)
        np.testing.assert_array_equal(a.rewards, b.rewards)
        assert a.committed_action == b.committed_action
        assert env.serialize() == restored.serialize()
    assert a.committed_step == 1 and a.token_step == 3


def test_terminal_metadata_token_clock_and_no_reward_replay():
    env = fresh()
    low = env.full_state()['round_state']['hands'][0][0]
    single_path(env, low)
    env.step(PASS_TOKEN)
    high = env.full_state()['round_state']['hands'][2][0]
    result = single_path(env, high)
    assert result.done and not result.truncated and result.observation is None
    assert result.info['ranking'] == (1, 4, 2, 3)
    assert result.info['finish_token_step'] == 7
    assert result.info['finish_committed_step'] == 3
    np.testing.assert_array_equal(result.rewards, [1, -1, 1, -1])
    assert env.full_state()['round_state']['finish_token_step'] == 7
    restored = GuandanEnv.deserialize(env.serialize())
    assert restored.done and restored.observe() is None and restored.serialize() == env.serialize()
    before = restored.serialize()
    with pytest.raises(EpisodeTerminatedError):
        restored.step(PASS_TOKEN)
    assert restored.serialize() == before


def test_truncation_is_zero_reward_and_serializable():
    env = fresh(max_token_steps=1)
    result = env.step(TokenCodec.family(CombinationKind.SINGLE))
    assert result.truncated and not result.done
    assert result.observation is None and not result.rewards.any()
    assert result.info['termination_reason'] == 'max_token_steps'
    restored = GuandanEnv.deserialize(env.serialize())
    assert restored.serialize() == env.serialize()
    with pytest.raises(EpisodeTerminatedError):
        restored.step(COMMIT_TOKEN)


@pytest.mark.parametrize('version', ['rules_version', 'action_version', 'environment_version', 'encoding_version', 'serialization_version'])
def test_checkpoint_version_rejected(version):
    env = fresh()
    payload = json.loads(env.serialize())
    payload[version] = 'future-incompatible'
    with pytest.raises(ProtocolError):
        GuandanEnv.deserialize(json.dumps(payload).encode())


def test_reset_is_not_resume_and_failure_preserves_rng():
    env = fresh()
    env.step(TokenCodec.family(CombinationKind.SINGLE))
    before = env.serialize()
    with pytest.raises(ProtocolError, match='fresh'):
        env.reset(initial_state=before)
    with pytest.raises(ValueError):
        env.reset(seed=True)
    assert env.serialize() == before


def test_reset_encoder_failure_rolls_back(monkeypatch):
    env = fresh()
    before = env.serialize()
    def fail(*args, **kwargs):
        raise ProtocolError('deliberate encoder overflow')
    monkeypatch.setattr('guandan.environment.encode_observation', fail)
    with pytest.raises(ProtocolError):
        env.reset(seed=991)
    assert env.serialize() == before


def test_commit_post_observation_failure_rolls_back(monkeypatch):
    env = fresh()
    low = env.full_state()['round_state']['hands'][0][0]
    env.step(TokenCodec.family(CombinationKind.SINGLE))
    env.step(TokenCodec.card(low))
    before = env.serialize()
    def fail(*args, **kwargs):
        raise ProtocolError('deliberate encoder overflow after commit')
    monkeypatch.setattr('guandan.environment.encode_observation', fail)
    with pytest.raises(ProtocolError):
        env.step(COMMIT_TOKEN)
    assert env.serialize() == before


def test_seed_reset_streams_and_clone_are_independent():
    a = GuandanEnv(GameConfig(level_rank=Rank.SIX, seed=41))
    b = GuandanEnv(GameConfig(level_rank=Rank.SIX, seed=41))
    compare_observations(a.reset(), b.reset())
    original = a.serialize()
    clone = GuandanEnv.deserialize(original)
    compare_observations(a.reset(), b.reset())
    assert a.full_state()['round_state']['hands'] != json.loads(original)['protocol']['round_state']['hands']
    compare_observations(clone.reset(), a.observe())
    before = b.serialize()
    a.reset(seed=55)
    assert b.serialize() == before
    compare_observations(a.reset(seed=55), a.observe())


def test_full_state_and_policy_arrays_are_detached():
    env = fresh()
    before = env.serialize()
    debug = env.full_state()
    debug['round_state']['hands'][0].clear()
    observation = env.observe()
    for item in fields(observation):
        value = getattr(observation, item.name)
        if isinstance(value, np.ndarray) and value.flags.writeable:
            value.fill(0)
    assert env.serialize() == before
    assert env.observe().private_hand_card_ids is None

@pytest.mark.parametrize('snapshot_kind', ['environment', 'protocol'])
def test_empty_prefix_after_commit_is_resume_not_reset(snapshot_kind):
    env = fresh()
    low = env.full_state()['round_state']['hands'][0][0]
    single_path(env, low)
    before = env.serialize()
    initial = before if snapshot_kind == 'environment' else json.dumps(json.loads(before)['protocol']).encode()
    with pytest.raises(ProtocolError, match='fresh'):
        env.reset(initial_state=initial)
    assert env.serialize() == before


def test_real_observation_overflow_after_commit_rolls_back_all_state():
    # Fresh self hand fits; next player's hand + the public play/trick will not.
    env = GuandanEnv(GameConfig(level_rank=Rank.SIX), observation_spec=ObservationSpec(max_observation_tokens=26))
    env.reset(initial_state=checkpoint())
    low = env.full_state()['round_state']['hands'][0][0]
    env.step(TokenCodec.family(CombinationKind.SINGLE))
    env.step(TokenCodec.card(low))
    before = env.serialize()
    with pytest.raises(ProtocolError, match='max_observation_tokens'):
        env.step(COMMIT_TOKEN)
    assert env.serialize() == before


def test_real_observation_overflow_after_prefix_does_not_increment_clock():
    probe = fresh()
    used = int(probe.observe().state_channels[24])
    env = GuandanEnv(probe.config, observation_spec=ObservationSpec(max_observation_tokens=used))
    env.reset(initial_state=checkpoint())
    before = env.serialize()
    with pytest.raises(ProtocolError, match='max_observation_tokens'):
        env.step(TokenCodec.family(CombinationKind.SINGLE))
    assert env.serialize() == before


@pytest.mark.parametrize('corruption', [
    'token-float', 'committed-float', 'actor-float', 'history-clock',
    'dead-prefix', 'rank-index', 'truncated-string', 'early-truncation',
    'over-limit', 'orphan-winner',
])
def test_corrupted_checkpoint_rejected_without_coercion(corruption):
    env = fresh()
    before = env.serialize()
    payload = json.loads(before)
    raw = payload['protocol']
    state = raw['round_state']
    if corruption == 'token-float': raw['token_step'] = 0.75
    elif corruption == 'committed-float': state['committed_step'] = 0.75
    elif corruption == 'actor-float': state['active_seat'] = 0.75
    elif corruption == 'history-clock': state['committed_step'] = raw['token_step'] = 1
    elif corruption == 'dead-prefix': raw['prefix_tokens'] = [COMMIT_TOKEN]; raw['token_step'] = 1
    elif corruption == 'rank-index': state['ranking'] = [1, 0, 0, 0]
    elif corruption == 'truncated-string': raw['truncated'] = 'false'
    elif corruption == 'early-truncation': raw['truncated'] = True
    elif corruption == 'over-limit': raw['token_step'] = raw['max_token_steps'] + 1
    elif corruption == 'orphan-winner': state['current_winner_seat'] = 0
    with pytest.raises((ProtocolError, ValueError)):
        GuandanEnv.deserialize(json.dumps(payload).encode())
    assert env.serialize() == before


def test_unknown_or_forged_current_declaration_rejected_on_resume():
    env = fresh()
    low = env.full_state()['round_state']['hands'][0][0]
    single_path(env, low)
    before = env.serialize()
    payload = json.loads(before)
    payload['protocol']['round_state']['current_winning']['comparison_key'] = [999]
    with pytest.raises(ProtocolError):
        GuandanEnv.deserialize(json.dumps(payload).encode())
    assert env.serialize() == before


def test_cached_candidates_invalidate_after_private_hand_permutation():
    # Debug mutation is not a policy API, but it must never yield stale paths.
    from guandan.action_state import canonical_tokens
    from guandan.round import enumerate_legal_actions
    env = fresh()
    protocol = env._protocol
    original = {sequence for sequence, _ in protocol._candidate_sequences()}
    protocol.round_state.hands[0][0], protocol.round_state.hands[1][0] = (
        protocol.round_state.hands[1][0], protocol.round_state.hands[0][0])
    paths = {sequence for sequence, _ in protocol._candidate_sequences()}
    reference = {canonical_tokens(action) for action in enumerate_legal_actions(protocol.round_state)}
    assert paths == reference and paths != original
