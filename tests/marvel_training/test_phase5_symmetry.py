"""Phase-5 symmetry transforms: independent manual expectations and counterexamples.

The expected values below are hand-computed from the documented channel/token
layout; the tests do not call a production helper to decide what "correct"
means.  Round-trip checks are paired with those manual assertions.
"""
from __future__ import annotations

import numpy as np
import pytest

from guandan_marvel.phase5.symmetry import (
    ACTIVE_ONE_HOT, BOS_TOKEN, CARD_TOKEN_BASE, COMMIT_TOKEN, ComposedTransform,
    CopySwap, CURRENT_WINNING, EMPTIED_ONE_HOT, EVENT_END,
    EVENT_EXCHANGE_RESOLVED, EVENT_PRIVATE_PREFIX, EVENT_PUBLIC_COMMIT,
    EVENT_RESET, FINISHING_RANKS, LEADER_ONE_HOT, LENGTH_TOKEN_BASE,
    OBSERVER_ONE_HOT, PREVIOUS_RANKING, PREVIOUS_WINNER_TEAM, RANK_TOKEN_BASE,
    REMAINING_COUNTS, RETURN_TOKEN, SELF_EXCHANGE, SELF_HAND, SELF_PREFIX,
    SINGLE_BITS, SeatRotation, SymmetryDomainError, TEAM_OF, TEAM_ONE_HOT,
    TRIBUTE_DONOR, TRIBUTE_ORDER, TRIBUTE_TOKEN, WILD_TOKEN_BASE, WINNER_ONE_HOT,
    action_permutation, candidate_image_is_unique, check_transform_involution,
    compose_transforms, event_seat_positions, inverse_candidate_rows,
    macro_action_permutation, transform_candidate_rows, transform_event_tokens,
    transform_observation, transform_team,
)


def _empty_observation():
    tokens = np.zeros((4, 48), dtype=np.int64)
    channels = np.zeros((4, 512), dtype=np.uint8)
    return tokens, channels


def _rotated_fixture():
    tokens, channels = _empty_observation()
    for observer in range(4):
        tokens[observer, 0] = BOS_TOKEN
        tokens[observer, 1] = EVENT_RESET
        tokens[observer, 2] = 2
        tokens[observer, 3] = RANK_TOKEN_BASE + 12
        tokens[observer, 4] = LENGTH_TOKEN_BASE + observer
        tokens[observer, 5] = EVENT_END
        channels[observer, OBSERVER_ONE_HOT.start + observer] = 1
        channels[observer, ACTIVE_ONE_HOT.start + (observer + 1) % 4] = 1
        channels[observer, LEADER_ONE_HOT.start + (observer + 2) % 4] = 1
        channels[observer, WINNER_ONE_HOT.start + (observer + 3) % 4] = 1
        channels[observer, REMAINING_COUNTS.start + observer] = 10 + observer
        channels[observer, FINISHING_RANKS.start + observer] = observer + 1
        channels[observer, PREVIOUS_RANKING.start + observer] = observer + 1
        channels[observer, TEAM_ONE_HOT.start + TEAM_OF[observer]] = 1
        channels[observer, PREVIOUS_WINNER_TEAM.start] = 1  # team 0
        channels[observer, TRIBUTE_DONOR.start + observer] = 1
        channels[observer, TRIBUTE_ORDER.start + observer] = observer + 1
        channels[observer, SINGLE_BITS["is_observer"]] = 1
        channels[observer, SINGLE_BITS["has_previous_result"]] = 1
    return tokens, channels


def test_seat_rotation_is_manual_and_invertible():
    tokens, channels = _rotated_fixture()
    new_tokens, new_channels = transform_observation(tokens, channels, SeatRotation(1))
    # Original observer 0 moves to seat 1.
    row = 1
    assert int(new_tokens[row, 4]) == LENGTH_TOKEN_BASE + 1
    assert int(new_channels[row, OBSERVER_ONE_HOT.start + row]) == 1
    assert int(new_channels[row, ACTIVE_ONE_HOT.start + 2]) == 1
    assert int(new_channels[row, LEADER_ONE_HOT.start + 3]) == 1
    assert int(new_channels[row, WINNER_ONE_HOT.start + 0]) == 1
    assert int(new_channels[row, REMAINING_COUNTS.start + 1]) == 10
    assert int(new_channels[row, FINISHING_RANKS.start + 1]) == 1
    assert int(new_channels[row, PREVIOUS_RANKING.start + 1]) == 1
    # An odd rotation swaps the two teams and the previous winner's team bit.
    assert int(new_channels[row, TEAM_ONE_HOT.start + TEAM_OF[1]]) == 1
    assert int(new_channels[row, TEAM_ONE_HOT.start + TEAM_OF[0]]) == 0
    assert int(new_channels[row, PREVIOUS_WINNER_TEAM.start + 1]) == 1
    assert int(new_channels[row, PREVIOUS_WINNER_TEAM.start + 0]) == 0
    # Tribute order stores seat+1 and must be rotated as a value.
    assert int(new_channels[row, TRIBUTE_ORDER.start + 0]) == 2
    assert int(new_channels[row, SINGLE_BITS["is_observer"]]) == 1
    # Rotation-by-three is the inverse.
    back_tokens, back_channels = transform_observation(
        new_tokens, new_channels, SeatRotation(3),
    )
    np.testing.assert_array_equal(back_tokens, tokens)
    np.testing.assert_array_equal(back_channels, channels)


def test_copy_swap_maps_card_ids_but_never_rank_or_wild_tokens():
    tokens, channels = _empty_observation()
    tokens[0, 0] = BOS_TOKEN
    tokens[0, 1] = EVENT_PUBLIC_COMMIT
    tokens[0, 2] = 2
    tokens[0, 3] = LENGTH_TOKEN_BASE + 0
    tokens[0, 4] = CARD_TOKEN_BASE + 54
    tokens[0, 5] = EVENT_END
    channels[0, SELF_HAND.start + 5] = 1
    channels[0, CURRENT_WINNING.start + 7] = 1
    channels[0, SELF_EXCHANGE.start + 8] = 1
    channels[0, SELF_PREFIX.start + 0] = CARD_TOKEN_BASE + 9
    channels[0, SELF_PREFIX.start + 1] = RANK_TOKEN_BASE + 3
    channels[0, SELF_PREFIX.start + 2] = WILD_TOKEN_BASE + 4
    new_tokens, new_channels = transform_observation(tokens, channels, CopySwap())
    assert int(new_tokens[0, 1]) == EVENT_PUBLIC_COMMIT
    assert int(new_tokens[0, 3]) == LENGTH_TOKEN_BASE + 0
    assert int(new_tokens[0, 4]) == CARD_TOKEN_BASE + 0  # 54 + 54 mod 108
    assert int(new_channels[0, SELF_HAND.start + 59]) == 1
    assert int(new_channels[0, SELF_HAND.start + 5]) == 0
    assert int(new_channels[0, CURRENT_WINNING.start + 61]) == 1
    assert int(new_channels[0, SELF_EXCHANGE.start + 62]) == 1
    assert int(new_channels[0, SELF_PREFIX.start + 0]) == CARD_TOKEN_BASE + 63
    assert int(new_channels[0, SELF_PREFIX.start + 1]) == RANK_TOKEN_BASE + 3
    assert int(new_channels[0, SELF_PREFIX.start + 2]) == WILD_TOKEN_BASE + 4
    back_tokens, back_channels = transform_observation(
        new_tokens, new_channels, CopySwap(),
    )
    np.testing.assert_array_equal(back_tokens, tokens)
    np.testing.assert_array_equal(back_channels, channels)


def test_rotation_event_seat_fields_include_tribute_records_and_ignore_length_payload():
    exchange = [
        BOS_TOKEN, EVENT_EXCHANGE_RESOLVED, 1, LENGTH_TOKEN_BASE + 2,
        TRIBUTE_TOKEN, LENGTH_TOKEN_BASE + 0, LENGTH_TOKEN_BASE + 1,
        CARD_TOKEN_BASE + 3, COMMIT_TOKEN,
        RETURN_TOKEN, LENGTH_TOKEN_BASE + 2, LENGTH_TOKEN_BASE + 3,
        CARD_TOKEN_BASE + 4, COMMIT_TOKEN, EVENT_END,
    ]
    assert event_seat_positions(exchange) == (3, 5, 6, 10, 11)
    rotated = transform_event_tokens(exchange, SeatRotation(1))
    # Seat fields are protocol tokens ``LENGTH_TOKEN_BASE + seat``
    # (guandan.action_state.TokenCodec.length), not bare seat integers.  A
    # rotation by one maps seat s to (s + 1) mod 4, so the decoded seats are
    # independently recomputed here and the encoded tokens checked as well.
    assert [int(rotated[index]) - LENGTH_TOKEN_BASE for index in (3, 5, 6, 10, 11)] == [
        3, 1, 2, 3, 0,
    ]
    assert [int(rotated[index]) for index in (3, 5, 6, 10, 11)] == [
        LENGTH_TOKEN_BASE + 3, LENGTH_TOKEN_BASE + 1, LENGTH_TOKEN_BASE + 2,
        LENGTH_TOKEN_BASE + 3, LENGTH_TOKEN_BASE + 0,
    ]
    # Card tokens are physical and must not be touched by a seat rotation.
    assert [int(rotated[index]) for index in (7, 12)] == [CARD_TOKEN_BASE + 3, CARD_TOKEN_BASE + 4]
    swapped = transform_event_tokens(exchange, CopySwap())
    assert [int(swapped[index]) for index in (7, 12)] == [
        CARD_TOKEN_BASE + 57, CARD_TOKEN_BASE + 58,
    ]
    # A deck swap leaves seats (and therefore their encoded LENGTH tokens) alone.
    assert [int(swapped[index]) - LENGTH_TOKEN_BASE for index in (3, 5, 6, 10, 11)] == [
        2, 0, 1, 2, 3,
    ]

    # A LENGTH token inside a declared action sequence is a length, not a seat.
    declared = [
        BOS_TOKEN, EVENT_PUBLIC_COMMIT, 2, LENGTH_TOKEN_BASE + 1,
        LENGTH_TOKEN_BASE + 5, EVENT_END,
    ]
    rotated_declared = transform_event_tokens(declared, SeatRotation(1))
    assert int(rotated_declared[3]) == LENGTH_TOKEN_BASE + 2
    assert int(rotated_declared[4]) == LENGTH_TOKEN_BASE + 5

    # A private prefix event never contributes a public card.
    private = [BOS_TOKEN, EVENT_PRIVATE_PREFIX, 2, LENGTH_TOKEN_BASE + 0,
               CARD_TOKEN_BASE + 9, EVENT_END]
    rotated_private = transform_event_tokens(private, SeatRotation(1))
    assert int(rotated_private[3]) == LENGTH_TOKEN_BASE + 1
    assert int(rotated_private[4]) == CARD_TOKEN_BASE + 9


def test_hostile_padding_does_not_change_the_event_grammar():
    event = [BOS_TOKEN, EVENT_PUBLIC_COMMIT, 2, LENGTH_TOKEN_BASE + 1,
             CARD_TOKEN_BASE + 3, EVENT_END]
    padded = event + [0] * 10
    assert event_seat_positions(padded) == (3,)
    rotated = transform_event_tokens(padded, SeatRotation(1))
    assert int(rotated[3]) == LENGTH_TOKEN_BASE + 2
    assert int(rotated[4]) == CARD_TOKEN_BASE + 3
    assert all(int(value) == 0 for value in rotated[len(event):])
    with pytest.raises(SymmetryDomainError):
        event_seat_positions([BOS_TOKEN, EVENT_PUBLIC_COMMIT, 2,
                              RANK_TOKEN_BASE + 0, EVENT_END])


def test_action_rows_are_one_to_one_and_round_trip():
    rows = np.asarray([
        [262, 2, 263, 296, 304, 6, 306, 307],       # PASS
        [262, 2, 263, 296, 304, CARD_TOKEN_BASE + 3, 306, 307],
        [262, 2, 263, 296, 304, CARD_TOKEN_BASE + 57, 306, 307],
        [262, 2, 263, 296, 304, WILD_TOKEN_BASE + 2, 306, 307],
    ], dtype=np.int64)
    image = transform_candidate_rows(rows, CopySwap())
    assert [int(value) for value in image[:, 5]] == [
        6, CARD_TOKEN_BASE + 57, CARD_TOKEN_BASE + 3, WILD_TOKEN_BASE + 2,
    ]
    assert candidate_image_is_unique(rows, CopySwap())
    permutation = action_permutation(rows, CopySwap(), target_rows=image)
    assert permutation == (0, 1, 2, 3)
    assert macro_action_permutation(rows, image, CopySwap()) == (0, 1, 2, 3)
    back = inverse_candidate_rows(image, CopySwap())
    np.testing.assert_array_equal(back, rows)
    # A duplicate source row cannot define a one-to-one action permutation.
    duplicated = np.asarray([rows[0], rows[0]], dtype=np.int64)
    assert not candidate_image_is_unique(duplicated, CopySwap())


def test_composed_transform_group_is_closed_and_invertible():
    composed = compose_transforms(SeatRotation(1), CopySwap())
    assert isinstance(composed, ComposedTransform)
    assert composed.rotation == 1 and composed.swap is True
    assert compose_transforms(SeatRotation(1), SeatRotation(2)) == SeatRotation(3)
    assert compose_transforms(CopySwap(), CopySwap()) == SeatRotation(0)
    assert compose_transforms(CopySwap(), SeatRotation(1)) == ComposedTransform(1, True)
    assert transform_team(composed, 0) == 1
    tokens, channels = _rotated_fixture()
    channels[0, SELF_HAND.start + 5] = 1
    moved_tokens, moved_channels = transform_observation(tokens, channels, composed)
    # Card 5 -> 59 in the row moved from seat 0 to seat 1.
    assert int(moved_channels[1, SELF_HAND.start + 59]) == 1
    assert int(moved_channels[1, SELF_HAND.start + 5]) == 0
    back_tokens, back_channels = transform_observation(
        moved_tokens, moved_channels, composed.inverse(),
    )
    np.testing.assert_array_equal(back_tokens, tokens)
    np.testing.assert_array_equal(back_channels, channels)


@pytest.mark.parametrize(
    "transform",
    [SeatRotation(0), SeatRotation(1), SeatRotation(2), SeatRotation(3),
     CopySwap(), ComposedTransform(1, True), ComposedTransform(3, True)],
)
def test_involution_helper_round_trips_every_group_element(transform):
    assert check_transform_involution(transform, trials=3)


def test_transform_observation_rejects_ambiguous_or_malformed_encodings():
    tokens, channels = _empty_observation()
    channels[0, OBSERVER_ONE_HOT.start + 0] = 1
    channels[0, ACTIVE_ONE_HOT.start + 0] = 1
    channels[0, ACTIVE_ONE_HOT.start + 1] = 1  # two active seats: ambiguous
    with pytest.raises(SymmetryDomainError):
        transform_observation(tokens, channels, SeatRotation(1))
    with pytest.raises(SymmetryDomainError):
        transform_observation(tokens, np.zeros((4, 511), dtype=np.uint8), SeatRotation(1))
    with pytest.raises(ValueError):
        SeatRotation(4)
    with pytest.raises(ValueError):
        ComposedTransform(4, False)
