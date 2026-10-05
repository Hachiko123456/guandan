from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pytest

from guandan.action_state import (
    BOS_TOKEN, COMMIT_TOKEN, FAMILY_ORDER, FOLLOW_PLAY_TOKEN, LEAD_PLAY_TOKEN, PASS_TOKEN,
    RETURN_TOKEN, TRIBUTE_TOKEN, ProtocolError, StepwiseActionState, TokenCodec,
    canonical_tokens,
)
from guandan.cards import Card, Rank, Suit, full_deck
from guandan.combos import CombinationKind, recognize_combinations
from guandan.encoding import (
    FINISHING_RANKS_SLICE, PREFIX_TOKENS_SLICE, REMAINING_COUNTS_SLICE,
    SELF_CARD_IDS_SLICE, encode_observation,
)
from guandan.environment_types import Observation, ObservationSpec
from guandan.round import apply_action, enumerate_legal_actions
from guandan.state import CommittedAction, EpisodeTerminatedError, Phase, PreviousHandResult, RoundState


ARRAY_FIELDS = (
    "observation_tokens", "state_channels", "legal_next_tokens", "legal_next_mask",
    "legal_next_is_commit", "legal_next_kinds", "legal_next_values",
)


def make_state() -> RoundState:
    small = [[Card(i) for i in ids] for ids in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))]
    used = {card.card_id for hand in small for card in hand}
    return RoundState.from_hands(small + [[card for card in full_deck() if card.card_id not in used]], level_rank=Rank.KING)


def same_observation(left: Observation, right: Observation) -> None:
    for field in fields(Observation):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=field.name)
        else:
            assert a == b, field.name


def split_wire(observation: Observation) -> tuple[list[int], list[list[int]]]:
    """Independent framing parser; it does not call encoder internals."""
    used = int(observation.state_channels[24])
    stream = observation.observation_tokens[:used].tolist()
    assert stream[:1] == [BOS_TOKEN] and stream[4] == COMMIT_TOKEN
    sections = []
    offset = 5
    for section_id in range(5):
        assert stream[offset:offset + 2] == [BOS_TOKEN, TokenCodec.length(section_id)]
        offset += 2
        start = offset
        if section_id < 3:
            while stream[offset] != COMMIT_TOKEN:
                offset += 1
        else:
            while stream[offset] == BOS_TOKEN:
                offset += 1
                while stream[offset] != COMMIT_TOKEN:
                    offset += 1
                offset += 1
        sections.append(stream[start:offset])
        assert stream[offset] == COMMIT_TOKEN
        offset += 1
    assert offset == used
    assert not observation.observation_tokens[used:].any()
    return stream[:5], sections


def records(section: list[int]) -> list[list[int]]:
    result = []
    offset = 0
    while offset < len(section):
        assert section[offset] == BOS_TOKEN
        end = section.index(COMMIT_TOKEN, offset)
        result.append(section[offset + 1:end])
        offset = end + 1
    return result


def decode_play(payload: list[int], level: Rank):
    """Parse the observation grammar, not Scheme B or encoder internals."""
    kind = TokenCodec.family_from_token(payload[0])
    comparison_rank = TokenCodec.rank_from_token(payload[1])
    cards, assignments, natural = [], {}, []
    offset = 2
    while offset < len(payload):
        card_id = TokenCodec.card_from_token(payload[offset])
        assert not cards or cards[-1].card_id < card_id
        cards.append(Card(card_id))
        offset += 1
        if offset < len(payload) and 172 <= payload[offset] <= 223:
            assignments[card_id] = TokenCodec.wild_from_token(payload[offset])
            offset += 1
        if offset < len(payload) and payload[offset] == TokenCodec.length(1):
            natural.append(card_id)
            offset += 1
    matches = [d for d in recognize_combinations(cards, level_rank=level)
               if d.kind is kind and d.comparison_rank is comparison_rank
               and dict(d.wild_assignments) == assignments and list(d.natural_wild_ids) == natural]
    assert len(matches) == 1
    return matches[0]


def declaration_key(d) -> tuple:
    return (d.kind, tuple(sorted(d.physical_ids())), d.level_rank, d.comparison_rank,
            d.comparison_key, tuple(sorted(d.wild_assignments.items())), d.natural_wild_ids)


def test_default_shapes_dtypes_vocab_padding_and_legal_metadata() -> None:
    protocol = StepwiseActionState(make_state())
    observation = encode_observation(protocol)
    assert observation.private_hand_card_ids is None
    assert observation.player_id == observation.team_id == 0
    assert observation.phase == "LEAD_PLAY"
    for name in ARRAY_FIELDS:
        array = getattr(observation, name)
        assert array.shape == ((4096,) if name == "observation_tokens" else (256,))
        dtype = np.float32 if name == "state_channels" else np.bool_ if name in ("legal_next_mask", "legal_next_is_commit") else np.int32
        assert array.dtype == dtype and array.flags.owndata
    valid = observation.legal_next_mask
    for name in ARRAY_FIELDS[2:]:
        assert not getattr(observation, name)[~valid].any()
    assert np.all(observation.legal_next_tokens[valid] > 0)
    assert np.all(np.diff(observation.legal_next_tokens[valid]) > 0)
    for actual, expected in zip((getattr(observation, name) for name in ARRAY_FIELDS[2:]), protocol.legal_token_arrays(), strict=True):
        np.testing.assert_array_equal(actual, expected)
    used = int(observation.state_channels[24])
    payload = observation.observation_tokens[:used]
    assert np.all(((1 <= payload) & (payload <= 48)) | ((64 <= payload) & (payload <= 223)))
    header, sections = split_wire(observation)
    assert header == [BOS_TOKEN, LEAD_PLAY_TOKEN, TokenCodec.rank(Rank.KING), TokenCodec.length(0), COMMIT_TOKEN]
    assert [TokenCodec.card_from_token(t) for t in sections[0]] == [0, 1, 2, 3]
    assert sections[1:] == [[], [], [], []]
    np.testing.assert_array_equal(observation.state_channels[REMAINING_COUNTS_SLICE], [4, 4, 4, 96])
    np.testing.assert_array_equal(observation.state_channels[FINISHING_RANKS_SLICE], [0, 0, 0, 0])
    assert np.flatnonzero(observation.state_channels[SELF_CARD_IDS_SLICE]).tolist() == [0, 1, 2, 3]
    assert observation.state_channels[81:96].sum() == observation.state_channels[96:101].sum() == 4
    np.testing.assert_array_equal(observation.state_channels[48:52], [0, 1, 0, 1])


def test_self_prefix_survives_losslessly_without_public_state_mutation() -> None:
    protocol = StepwiseActionState(make_state())
    before = protocol.round_state.serialize()
    action = next(a for a in enumerate_legal_actions(protocol.round_state) if a.kind is CombinationKind.PAIR)
    path = canonical_tokens(action)
    for index, token in enumerate(path[:-1], 1):
        protocol.step(token)
        observation = encode_observation(protocol)
        _, sections = split_wire(observation)
        assert sections[1] == list(path[:index])
        assert sections[3] == []
        np.testing.assert_array_equal(observation.state_channels[PREFIX_TOKENS_SLICE][:index], path[:index])
        assert observation.token_step == index and observation.committed_step == 0
        assert protocol.round_state.serialize() == before
    assert observation.state_channels[22] == observation.state_channels[254] == 1
    assert observation.legal_next_tokens[observation.legal_next_mask].tolist() == [COMMIT_TOKEN]
    assert observation.legal_next_is_commit[0] and observation.legal_next_kinds[0] == 2
    protocol.step(COMMIT_TOKEN)
    assert split_wire(encode_observation(protocol))[1][1] == []


def test_public_history_and_current_trick_are_framed_and_decodable() -> None:
    state = make_state()
    apply_action(state, CommittedAction(0, CombinationKind.PAIR, (Card(0), Card(1))))
    apply_action(state, CommittedAction(1, "pass"))
    observation = encode_observation(StepwiseActionState(state))
    header, sections = split_wire(observation)
    assert header[1] == FOLLOW_PLAY_TOKEN and observation.phase == "FOLLOW_PLAY"
    assert sections[2][0] == TokenCodec.length(0)
    assert declaration_key(decode_play(sections[2][1:], state.level_rank)) == declaration_key(state.current_winning)
    history = records(sections[3])
    assert len(history) == 2
    assert history[0][:2] == [LEAD_PLAY_TOKEN, TokenCodec.length(0)]
    assert declaration_key(decode_play(history[0][2:], state.level_rank)) == declaration_key(state.history[0].declaration)
    assert history[1] == [FOLLOW_PLAY_TOKEN, TokenCodec.length(1), PASS_TOKEN]
    assert observation.state_channels[7] == 0 and observation.state_channels[20] == 1
    assert observation.legal_next_tokens[0] == PASS_TOKEN
    assert observation.legal_next_is_commit[0] and observation.legal_next_kinds[0] == 1
    apply_action(state, CommittedAction(2, "pass"))
    apply_action(state, CommittedAction(3, "pass"))
    reset = encode_observation(StepwiseActionState(state))
    assert reset.phase == "LEAD_PLAY" and split_wire(reset)[1][2] == []
    assert len(records(split_wire(reset)[1][3])) == 4
    assert reset.state_channels[7] == -1


@pytest.mark.parametrize("natural", [False, True])
def test_history_roundtrip_preserves_physical_wild_assignments_and_natural_use(natural: bool) -> None:
    level = Rank.FOUR
    wild = next(c for c in full_deck() if c.rank is level and c.suit is Suit.HEARTS)
    partner = next(c for c in full_deck() if c.rank is (level if natural else Rank.THREE) and c.suit is Suit.CLUBS)
    rest = [c for c in full_deck() if c not in (wild, partner)]
    state = RoundState.from_hands([[wild, partner, rest[0]], rest[1:4], rest[4:7], rest[7:]], level_rank=level)
    declaration = next(d for d in recognize_combinations((wild, partner), level_rank=level)
                       if d.kind is CombinationKind.PAIR and bool(d.natural_wild_ids) == natural
                       and d.comparison_rank is partner.rank)
    apply_action(state, CommittedAction(0, CombinationKind.PAIR, (wild, partner), declaration))
    observation = encode_observation(StepwiseActionState(state))
    record = records(split_wire(observation)[1][3])[0]
    assert declaration_key(decode_play(record[2:], level)) == declaration_key(declaration)
    assert declaration_key(decode_play(split_wire(observation)[1][2][1:], level)) == declaration_key(declaration)


@dataclass(frozen=True)
class PlayCase:
    name: str
    family: CombinationKind
    card_ids: tuple[int, ...]
    comparison_rank: Rank
    assignments: tuple[tuple[int, Rank, Suit], ...] = ()
    natural: tuple[int, ...] = ()


# All cases use level FOUR: the two physical level-wild cards are 6 and 60.
# Input card order intentionally differs from physical-ID order in several
# cases so canonical group/window ordering cannot accidentally pass this test.
PLAY_CASES = (
    PlayCase("single", CombinationKind.SINGLE, (0,), Rank.THREE),
    PlayCase("single-natural-wild", CombinationKind.SINGLE, (6,), Rank.FOUR,
             ((6, Rank.FOUR, Suit.HEARTS),), (6,)),
    PlayCase("pair-duplicate-decks", CombinationKind.PAIR, (54, 0), Rank.THREE),
    PlayCase("triple", CombinationKind.TRIPLE, (2, 0, 1), Rank.THREE),
    PlayCase("full-house", CombinationKind.FULL_HOUSE, (8, 9, 10, 0, 1), Rank.FIVE),
    PlayCase("full-house-two-wild-targets", CombinationKind.FULL_HOUSE,
             (60, 0, 2, 6, 1), Rank.THREE,
             ((6, Rank.FIVE, Suit.CLUBS), (60, Rank.FIVE, Suit.CLUBS))),
    PlayCase("full-house-natural-and-substituted", CombinationKind.FULL_HOUSE,
             (60, 0, 2, 6, 1), Rank.THREE,
             ((6, Rank.FOUR, Suit.HEARTS), (60, Rank.FOUR, Suit.CLUBS)), (6,)),
    PlayCase("straight-start-three-high-seven", CombinationKind.STRAIGHT,
             (0, 4, 8, 12, 16), Rank.SEVEN),
    PlayCase("straight-low-ace-high-five", CombinationKind.STRAIGHT,
             (44, 48, 0, 4, 8), Rank.FIVE),
    PlayCase("pair-sequence-start-three-high-five", CombinationKind.PAIR_SEQUENCE,
             (0, 54, 4, 58, 8, 62), Rank.FIVE),
    PlayCase("triple-sequence-start-three-high-four", CombinationKind.TRIPLE_SEQUENCE,
             (0, 1, 2, 4, 5, 7), Rank.FOUR),
    PlayCase("rank-bomb-four", CombinationKind.RANK_BOMB, (3, 2, 1, 0), Rank.THREE),
    PlayCase("rank-bomb-nine", CombinationKind.RANK_BOMB,
             (0, 1, 2, 3, 54, 55, 56, 57, 6), Rank.THREE,
             ((6, Rank.THREE, Suit.CLUBS),)),
    PlayCase("rank-bomb-ten", CombinationKind.RANK_BOMB,
             (0, 1, 2, 3, 54, 55, 56, 57, 6, 60), Rank.THREE,
             ((6, Rank.THREE, Suit.CLUBS), (60, Rank.THREE, Suit.CLUBS))),
    PlayCase("straight-flush-spades", CombinationKind.STRAIGHT_FLUSH,
             (3, 7, 11, 15, 19), Rank.SEVEN),
    PlayCase("straight-flush-suited-wild", CombinationKind.STRAIGHT_FLUSH,
             (3, 6, 11, 15, 19), Rank.SEVEN, ((6, Rank.FOUR, Suit.SPADES),)),
    PlayCase("straight-flush-natural-heart-wild", CombinationKind.STRAIGHT_FLUSH,
             (2, 6, 10, 14, 18), Rank.SEVEN,
             ((6, Rank.FOUR, Suit.HEARTS),), (6,)),
    PlayCase("four-kings", CombinationKind.FOUR_KINGS, (107, 52, 106, 53), Rank.BIG_JOKER),
)


def test_roundtrip_case_inventory_covers_all_ten_families() -> None:
    assert {case.family for case in PLAY_CASES} == set(CombinationKind)
    assert {len(case.card_ids) for case in PLAY_CASES
            if case.family is CombinationKind.RANK_BOMB} >= {4, 9, 10}


def test_straight_flush_identical_wild_declaration_canonicalizes_to_natural() -> None:
    level = Rank.FOUR
    cards = tuple(Card(card_id) for card_id in (2, 6, 10, 14, 18))
    declarations = [d for d in recognize_combinations(cards, level_rank=level)
                    if d.kind is CombinationKind.STRAIGHT_FLUSH
                    and d.comparison_rank is Rank.SEVEN
                    and dict(d.wild_assignments) == {6: (Rank.FOUR, Suit.HEARTS)}]
    assert len(declarations) == 1
    assert declarations[0].natural_wild_ids == (6,)


@pytest.mark.parametrize("case", PLAY_CASES, ids=lambda case: case.name)
def test_all_families_history_and_winning_roundtrip(case: PlayCase) -> None:
    level = Rank.FOUR
    cards = tuple(Card(card_id) for card_id in case.card_ids)
    expected_assignments = {card_id: (rank, suit) for card_id, rank, suit in case.assignments}
    declarations = [d for d in recognize_combinations(cards, level_rank=level)
                    if d.kind is case.family and d.comparison_rank is case.comparison_rank
                    and dict(d.wild_assignments) == expected_assignments
                    and d.natural_wild_ids == case.natural]
    assert len(declarations) == 1
    declaration = declarations[0]
    unused = [card for card in full_deck() if card.card_id not in case.card_ids]
    state = RoundState.from_hands(
        [list(cards) + unused[:1], unused[1:4], unused[4:7], unused[7:]],
        level_rank=level,
    )
    apply_action(state, CommittedAction(0, case.family, cards, declaration))
    protocol = StepwiseActionState(state)
    before = protocol.serialize()
    observation = encode_observation(protocol)
    assert protocol.serialize() == before
    header, sections = split_wire(observation)
    assert header[1] == FOLLOW_PLAY_TOKEN
    history = records(sections[3])
    assert len(history) == 1 and history[0][:2] == [LEAD_PLAY_TOKEN, TokenCodec.length(0)]
    assert sections[2][0] == TokenCodec.length(0)
    payload = history[0][2:]
    assert payload == sections[2][1:]
    assert payload[:2] == [TokenCodec.family(case.family), TokenCodec.rank(case.comparison_rank)]
    assert [TokenCodec.card_from_token(t) for t in payload if 64 <= t <= 171] == sorted(case.card_ids)
    assert COMMIT_TOKEN not in payload and BOS_TOKEN not in payload
    # No A03 START, SUIT, LENGTH or second full-house rank can appear here.
    assert len([t for t in payload if 18 <= t <= 32]) == 1
    assert not any(33 <= t <= 37 for t in payload)
    assert len([t for t in payload if 38 <= t <= 48]) == len(case.natural)
    for card_id, (rank, suit) in expected_assignments.items():
        position = payload.index(TokenCodec.card(card_id))
        assert payload[position + 1] == TokenCodec.wild(rank, suit)
    for card_id in case.natural:
        position = payload.index(TokenCodec.card(card_id)) + 1 + int(card_id in expected_assignments)
        assert payload[position] == TokenCodec.length(1)
    for body in (payload, sections[2][1:]):
        assert declaration_key(decode_play(body, level)) == declaration_key(declaration)
    np.testing.assert_array_equal(observation.state_channels[251:254],
                                  [FAMILY_ORDER.index(case.family) + 1, int(case.comparison_rank), len(cards)])
    used = int(observation.state_channels[24])
    # Header + five sections + one history record: no hidden nested terminator.
    assert np.count_nonzero(observation.observation_tokens[:used] == COMMIT_TOKEN) == 7
    # Reconstruct framing independently, exactly up to the PAD boundary.
    rebuilt = list(header)
    for section_id, body in enumerate(sections):
        rebuilt.extend([BOS_TOKEN, TokenCodec.length(section_id), *body, COMMIT_TOKEN])
    assert rebuilt == observation.observation_tokens[:used].tolist()


@pytest.mark.parametrize("other_seat", [1, 2])
def test_opponent_and_teammate_rearrangements_do_not_change_any_policy_field(other_seat: int) -> None:
    left = StepwiseActionState(make_state())
    left.step(TokenCodec.family(CombinationKind.PAIR))
    right = left.clone()
    right.round_state.hands[other_seat][0], right.round_state.hands[3][0] = right.round_state.hands[3][0], right.round_state.hands[other_seat][0]
    right.round_state.check_invariants()
    assert left.round_state.serialize() != right.round_state.serialize()
    same_observation(encode_observation(left), encode_observation(right))


def exchange_state() -> RoundState:
    # Each donor has one distinct non-wild candidate; jokers stay on recipient
    # seats so the donor-side anti-tribute rule does not cancel the exchange.
    small = [[Card(0), Card(1)], [Card(10), Card(6)], [Card(2), Card(3)]]
    used = {card.card_id for hand in small for card in hand} | {Card(11).card_id, Card(7).card_id}
    small[0] += [c for c in full_deck() if c.card_id not in used]
    return RoundState.from_hands(small + [[Card(11), Card(7)]], level_rank=Rank.FOUR,
                                previous_result=PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN"))


def commit_atomic(state: RoundState, kind: str) -> None:
    action = next(a for a in enumerate_legal_actions(state) if a.kind == kind)
    apply_action(state, action)


def test_hidden_other_donor_choice_counterfactual_and_unresolved_history_redaction() -> None:
    left, right = exchange_state(), exchange_state()
    # Change another donor's selected physical card, preserving the current
    # viewer's hand, every remaining count, and all public context.
    high = next(card for card in right.hands[0] if card.rank is Rank.ACE and card.card_id != 11)
    right.hands[0].remove(high)
    right.hands[0].append(Card(10))
    right.hands[1].remove(Card(10))
    right.hands[1].append(high)
    commit_atomic(left, "tribute")
    commit_atomic(right, "tribute")
    assert left.active_seat == right.active_seat == 3
    assert left.tribute_selections != right.tribute_selections
    before_left, before_right = left.serialize(), right.serialize()
    a, b = encode_observation(StepwiseActionState(left)), encode_observation(StepwiseActionState(right))
    same_observation(a, b)
    assert a.phase == "TRIBUTE" and a.committed_step == 1
    assert split_wire(a)[1][3:] == [[], []]
    assert a.state_channels[23] == a.state_channels[42] == a.state_channels[43] == 0
    assert left.serialize() == before_left and right.serialize() == before_right


def test_return_phase_hides_transfers_and_only_whole_exchange_reveals_identities() -> None:
    state = exchange_state()
    commit_atomic(state, "tribute")
    commit_atomic(state, "tribute")
    assert state.phase is Phase.RETURN
    # Finish the first return so the observer has a tiny hand, avoiding a
    # full-deal play enumeration; the exchange is still not public.
    commit_atomic(state, "return")
    assert state.phase is Phase.RETURN
    unresolved = StepwiseActionState(state)
    redacted = encode_observation(unresolved)
    assert split_wire(redacted)[1][3:] == [[], []]
    assert redacted.state_channels[27] == redacted.state_channels[42] == redacted.state_channels[43] == 0
    alternative = unresolved.clone()
    # Perturb only hidden bookkeeping. None of these fields are a policy input.
    alternative.round_state.tribute_order.reverse()
    alternative.round_state.tribute_transfers.reverse()
    alternative.round_state.return_transfers[:] = [(0, 3, Card(106))]
    alternative.round_state.tribute_selections.clear()
    alternative.round_state.tribute_escrow[:] = [Card(52)]
    alternative.round_state.history[:] = [CommittedAction(1, "tribute", (Card(106),))]
    same_observation(redacted, encode_observation(alternative))
    commit_atomic(state, "return")
    assert state.phase is Phase.PLAY
    resolved = encode_observation(StepwiseActionState(state))
    history = records(split_wire(resolved)[1][3])
    transfers = records(split_wire(resolved)[1][4])
    assert len(history) == len(transfers) == 4
    assert [r[0] for r in history] == [TRIBUTE_TOKEN, TRIBUTE_TOKEN, RETURN_TOKEN, RETURN_TOKEN]
    for record, action in zip(history, state.history, strict=True):
        assert record[1] == TokenCodec.length(action.player_id)
        assert record[2:] == [TokenCodec.card(card.card_id) for card in action.cards]
    for record, (source, target, card) in zip(transfers, state.tribute_transfers + state.return_transfers, strict=True):
        assert record[1:] == [TokenCodec.length(source), TokenCodec.length(target), TokenCodec.card(card.card_id)]
    assert resolved.state_channels[27] == 1
    assert resolved.state_channels[42] == resolved.state_channels[43] == 2


def test_observation_capacity_exact_fit_and_overflow_never_truncate_history() -> None:
    protocol = StepwiseActionState(make_state())
    baseline = encode_observation(protocol)
    used = int(baseline.state_channels[24])
    exact = encode_observation(protocol, ObservationSpec(max_observation_tokens=used))
    assert exact.observation_tokens.tolist() == baseline.observation_tokens[:used].tolist()
    before = protocol.serialize()
    for capacity in (1, used - 1):
        with pytest.raises(ProtocolError, match="max_observation_tokens"):
            encode_observation(protocol, ObservationSpec(max_observation_tokens=capacity))
    assert protocol.serialize() == before
    # Explicit over-capacity restored history: do not retain only the tail.
    protocol.round_state.history = [CommittedAction(i % 4, "pass") for i in range(900)]
    protocol.round_state.committed_step = 900
    before = protocol.serialize()
    with pytest.raises(ProtocolError, match="max_observation_tokens"):
        encode_observation(protocol)
    assert protocol.serialize() == before


@pytest.mark.parametrize("spec, message", [
    (ObservationSpec(max_legal_next_tokens=1), "max_legal_next_tokens"),
    (ObservationSpec(max_action_tokens=1), "max_action_tokens"),
])
def test_legal_and_action_capacity_overflow_without_mutating_protocol(spec, message) -> None:
    protocol = StepwiseActionState(make_state())
    before = protocol.serialize()
    with pytest.raises(ProtocolError, match=message):
        encode_observation(protocol, spec)
    assert protocol.serialize() == before


def test_float32_counter_capacity_is_not_silently_rounded() -> None:
    protocol = StepwiseActionState(make_state())
    protocol.token_step = 2**24 + 1
    with pytest.raises(ProtocolError, match="float32"):
        encode_observation(protocol)


@pytest.mark.parametrize("flag", ["done", "truncated", "terminal_phase"])
def test_terminal_and_truncated_raise_without_enumeration(flag, monkeypatch) -> None:
    protocol = StepwiseActionState(make_state())
    if flag == "done":
        protocol.round_state.done = True
    elif flag == "truncated":
        protocol.truncated = True
    else:
        protocol.round_state.phase = Phase.TERMINAL
    def fail(*args):
        raise AssertionError("terminal encoding must not enumerate")
    monkeypatch.setattr(protocol, "legal_token_arrays", fail)
    with pytest.raises(EpisodeTerminatedError):
        encode_observation(protocol)


def test_arrays_are_detached_and_inputs_untouched_even_after_caller_mutation() -> None:
    protocol = StepwiseActionState(make_state())
    before = protocol.serialize()
    first, second = encode_observation(protocol), encode_observation(protocol)
    same_observation(first, second)
    for name in ARRAY_FIELDS:
        a, b = getattr(first, name), getattr(second, name)
        assert not np.shares_memory(a, b)
        a[:] = False if a.dtype == np.bool_ else -1
    assert protocol.serialize() == before
    same_observation(second, encode_observation(protocol))


def test_projection_never_serializes_state_and_enumerates_once(monkeypatch) -> None:
    protocol = StepwiseActionState(make_state())
    original = protocol._candidate_sequences
    calls = 0
    def counted():
        nonlocal calls
        calls += 1
        return original()
    def forbidden(*args, **kwargs):
        raise AssertionError("policy projection must not serialize engine state or call prefix")
    monkeypatch.setattr(protocol, "_candidate_sequences", counted)
    monkeypatch.setattr(StepwiseActionState, "prefix", property(forbidden))
    monkeypatch.setattr(StepwiseActionState, "serialize", forbidden)
    monkeypatch.setattr(RoundState, "serialize", forbidden)
    observation = encode_observation(protocol)
    assert calls == 1
    assert observation.private_hand_card_ids is None


def test_smaller_source_legal_arrays_are_repacked_to_spec_shape() -> None:
    protocol = StepwiseActionState(make_state(), max_legal_next_tokens=10)
    observation = encode_observation(protocol)
    assert observation.legal_next_tokens.shape == (256,)
    assert observation.legal_next_mask.sum() == len(protocol.legal_next_tokens())
