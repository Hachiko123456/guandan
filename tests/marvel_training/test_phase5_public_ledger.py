"""Decisive phase-5 counterexamples for public transfer ownership and Baseline B.

The tests are deliberately independent of the production tracker where they can
be: hand-built 64-wide ledger rows are used for the Baseline B arithmetic, and
the real C++ engine cases compare the public ledger against the engine's hidden
hands *after* the visible features have been produced.
"""
from __future__ import annotations

import numpy as np
import pytest

from guandan_marvel.phase5.belief_data import (
    PublicLedger, PublicLedgerError, parse_public_event_row,
    public_event_from_observations, public_ledger_features,
    public_ledger_snapshot, validate_public_features,
)
from guandan_marvel.phase5.belief_probe import BaselineB
from guandan_marvel.phase5.design import (
    MAX_COPIES_BY_RANK, PHASE5_PUBLIC_LEDGER_SCHEMA, PUBLIC_FEATURE_WIDTH,
    phase5_design_hash,
)
from guandan_marvel.phase5.gate import BeliefGateError, load_b0_gate

OLD_DESIGN_HASH = "407431941204e41ecde208e34a89f380ba0e2b62021bd8f0a41c44c371d1608e"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _channels(actor, own_cards, remaining, *, phase=0, winning=(), self_exchange=()):
    channels = np.zeros((4, 512), dtype=np.uint8)
    for card in own_cards:
        channels[actor, int(card)] = 1
    for card in winning:
        channels[actor, 108 + int(card)] = 1
    for card in self_exchange:
        channels[actor, 216 + int(card)] = 1
    channels[:, 356:360] = np.asarray(remaining, dtype=np.uint8)
    channels[actor, 384 + int(phase)] = 1
    return channels


def _transfer(*records):
    transfers = [
        {"kind": kind, "source": source, "target": target, "card": card}
        for kind, source, target, card in records
    ]
    return {"kind": "exchange_resolved", "transfers": transfers}


def _commit(seat, cards):
    return {"kind": "commit", "seat": int(seat), "cards": list(cards)}


def _feature_row(remaining, known=None, unknown=None, escrow_unknown=0):
    known_array = (
        np.zeros((3, 15)) if known is None else np.asarray(known, dtype=np.float64)
    )
    if unknown is None:
        hand_capacity = np.asarray(remaining, dtype=np.float64) - known_array.sum(axis=1)
        unknown_array = np.zeros(15, dtype=np.float64)
        unknown_array[0] = hand_capacity.sum() + float(escrow_unknown)
    else:
        unknown_array = np.asarray(unknown, dtype=np.float64)
    return np.concatenate([
        np.asarray(remaining, dtype=np.float64),
        known_array.reshape(-1),
        unknown_array,
        np.asarray([float(escrow_unknown)]),
    ])


def _actor_zero_state():
    return list(range(0, 27)), [27, 27, 27, 27]


# ---------------------------------------------------------------------------
# 1. Known held transfer card is a hard Baseline B lower bound
# ---------------------------------------------------------------------------


def test_known_transfer_survives_even_when_the_unknown_rank_pool_is_zero():
    # next holds one publicly known big joker, partner holds one; the unknown
    # rank-14 pool is exactly zero.  Baseline B must still predict >= 1.
    row = np.zeros(PUBLIC_FEATURE_WIDTH)
    row[:3] = [1.0, 1.0, 0.0]
    row[3 + 0 * 15 + 14] = 1.0  # next K
    row[3 + 1 * 15 + 14] = 1.0  # partner K
    row[48 + 14] = 0.0          # u[14] == 0
    prediction = BaselineB().predict(row.reshape(1, -1))
    assert prediction[0, 0, 14] >= 1.0
    assert prediction[0, 1, 14] >= 1.0
    assert prediction[0, 2, 14] == pytest.approx(0.0)
    np.testing.assert_allclose(prediction[0, :, 14], np.asarray([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(prediction.sum(axis=(0, 2)), np.asarray([1.0, 1.0, 0.0]))


# ---------------------------------------------------------------------------
# 2. Same entity: transfer -> owner change -> play; self-received / self-sent
# ---------------------------------------------------------------------------


def test_same_entity_transfer_owner_change_and_play_never_double_counts():
    own, remaining = _actor_zero_state()
    ledger = PublicLedger()
    ledger.apply_event(_transfer(("tribute", 1, 2, 98)))
    first = public_ledger_snapshot(
        _channels(0, own, remaining), actor=0, ledger=ledger,
    )
    assert first["known_counts"][1, 11] == 1  # partner = seat 2 holds 98
    assert first["known_counts"][0].sum() == 0
    ledger.apply_event(_transfer(("return", 2, 1, 98)))  # same physical entity
    second = public_ledger_snapshot(
        _channels(0, own, remaining), actor=0, ledger=ledger,
    )
    assert second["known_counts"][0, 11] == 1  # next = seat 1 owns it now
    assert second["known_counts"][1].sum() == 0
    ledger.apply_event(_commit(1, [98]))
    third = public_ledger_snapshot(
        _channels(0, own, [27, 26, 27, 27], winning=[98]), actor=0, ledger=ledger,
    )
    assert third["known_counts"].sum() == 0
    assert third["played_counts"][11] == 1
    assert third["played_total"] == 1
    assert third["ledger_conserved"] is True
    # The entity is counted once in every snapshot.
    for snapshot in (first, second, third):
        assert int(snapshot["known_total"]) + int(snapshot["played_total"]) <= 108


def test_self_received_and_self_sent_transfers_are_attributed_to_the_other_hand():
    own, remaining = _actor_zero_state()
    # Self-received tribute 98 (kept in own hand) and self-sent return 99.
    received_own = [card for card in own if card != 26] + [98]
    ledger = PublicLedger()
    ledger.apply_event(_transfer(
        ("tribute", 1, 0, 98), ("return", 0, 1, 99),
    ))
    snapshot = public_ledger_snapshot(
        _channels(0, received_own, remaining), actor=0, ledger=ledger,
    )
    assert snapshot["own_counts"][11] == 1          # 98 is in the actor hand
    assert snapshot["known_counts"][0, 11] == 1     # 99 is held by next (seat 1)
    assert snapshot["known_counts"][1].sum() == 0
    assert snapshot["known_owner"][98] == 0
    assert snapshot["known_owner"][99] == 1

    # Self-sent tribute 98 and self-received return 99.
    sent_own = [card for card in own if card != 26] + [99]
    ledger = PublicLedger()
    ledger.apply_event(_transfer(
        ("tribute", 0, 2, 98), ("return", 2, 0, 99),
    ))
    snapshot = public_ledger_snapshot(
        _channels(0, sent_own, remaining), actor=0, ledger=ledger,
    )
    assert snapshot["own_counts"][11] == 1          # 99 is in the actor hand
    assert snapshot["known_counts"][1, 11] == 1     # 98 is held by partner (seat 2)
    assert snapshot["known_owner"][98] == 2
    assert snapshot["known_owner"][99] == 0


# ---------------------------------------------------------------------------
# 3. Identical visible history, different hidden worlds; private/future data
# ---------------------------------------------------------------------------


def test_identical_visible_history_has_identical_b_input_but_labels_may_differ():
    own, remaining = _actor_zero_state()
    channels = _channels(0, own, remaining)
    ledger_a = PublicLedger().apply_event(_transfer(("tribute", 1, 2, 98)))
    ledger_b = PublicLedger().apply_event(_transfer(("tribute", 1, 2, 98)))
    features_a = public_ledger_features(channels, actor=0, ledger=ledger_a)
    features_b = public_ledger_features(channels, actor=0, ledger=ledger_b)
    np.testing.assert_array_equal(features_a, features_b)
    prediction_a = BaselineB().predict(features_a.reshape(1, -1))
    prediction_b = BaselineB().predict(features_b.reshape(1, -1))
    np.testing.assert_array_equal(prediction_a, prediction_b)
    # Two hidden worlds with the same public state but different rank labels.
    label_a = np.zeros((3, 15), dtype=np.int64)
    label_b = np.zeros((3, 15), dtype=np.int64)
    label_a[1, 11] = 1
    label_b[1, 11] = 0
    label_b[1, 12] = 1
    assert not np.array_equal(label_a, label_b)
    # Private events publish nothing.
    private_prefix = [1, 257, 0, 39, 65, 308]
    private_exchange = [1, 259, 1, 39, 164, 7, 308]
    assert public_event_from_observations([private_prefix] * 4) is None
    assert public_event_from_observations([private_exchange] * 4) is None
    before = ledger_a.to_records()
    assert public_event_from_observations([private_prefix] * 4) is None
    assert ledger_a.to_records() == before
    # A future event appended afterwards cannot change the replay of the prefix
    # that was actually visible at the sample.
    ledger_a.apply_event(_commit(1, [100]))
    prefix_replay = PublicLedger.from_records(before)
    np.testing.assert_array_equal(
        public_ledger_features(channels, actor=0, ledger=prefix_replay), features_a,
    )


# ---------------------------------------------------------------------------
# 4. R == U, R > U, R < U are all legal; differences come from K / escrow
# ---------------------------------------------------------------------------


def test_r_equals_u_r_greater_than_u_and_r_less_than_u_are_legal():
    own, remaining = _actor_zero_state()
    # R == U: no public transfer and no escrow.
    ledger_equal = PublicLedger()
    equal = public_ledger_snapshot(
        _channels(0, own, remaining), actor=0, ledger=ledger_equal,
    )
    r_equal = int(equal["remaining"].sum())
    u_equal = int(equal["unknown_counts"].sum())
    assert r_equal == u_equal == 81
    prediction_equal = BaselineB().predict(equal["features"].reshape(1, -1))
    np.testing.assert_allclose(
        prediction_equal.sum(axis=(0, 2)), np.asarray([27.0, 27.0, 27.0]),
    )

    # R > U: the publicly resolved card 98 is still held by a relative seat.
    ledger_greater = PublicLedger().apply_event(_transfer(("tribute", 1, 2, 98)))
    greater = public_ledger_snapshot(
        _channels(0, own, remaining), actor=0, ledger=ledger_greater,
    )
    r_greater = int(greater["remaining"].sum())
    u_greater = int(greater["unknown_counts"].sum())
    assert (r_greater, u_greater) == (81, 80)
    assert greater["known_counts"][1, 11] == 1
    prediction_greater = BaselineB().predict(greater["features"].reshape(1, -1))
    assert prediction_greater[0, 1, 11] >= 1.0
    np.testing.assert_allclose(
        prediction_greater.sum(axis=(0, 2)), np.asarray([27.0, 27.0, 27.0]),
    )

    # R < U: one donor has already put a tribute card into escrow.
    escrow_channels = _channels(0, own, [27, 27, 26, 27], phase=1)
    less = public_ledger_snapshot(escrow_channels, actor=0, ledger=PublicLedger())
    r_less = int(less["remaining"].sum())
    u_less = int(less["unknown_counts"].sum())
    assert (r_less, u_less, int(less["escrow_unknown"])) == (80, 81, 1)
    prediction_less = BaselineB().predict(less["features"].reshape(1, -1))
    np.testing.assert_allclose(
        prediction_less.sum(axis=(0, 2)), np.asarray([27.0, 26.0, 27.0]),
    )
    assert float(prediction_less.sum()) == pytest.approx(80.0)  # escrow not assigned


# ---------------------------------------------------------------------------
# 4b. Actor-visible private escrow vs other-observer private state
# ---------------------------------------------------------------------------


def test_own_escrow_selection_is_excluded_from_opponent_allocation():
    # The actor is a donor that has just selected its tribute card: the card is
    # in the observable escrow, visible in its own SELF_EXCHANGE slice, and no
    # longer in its hand.  It must be E_known, never allocated to an opponent.
    own = list(range(0, 26))  # 26 cards after the selection
    channels = _channels(
        0, own, [26, 27, 27, 27], phase=1, self_exchange=[100],
    )
    snapshot = public_ledger_snapshot(channels, actor=0, ledger=PublicLedger())
    assert snapshot["escrow_total"] == 1
    assert snapshot["escrow_unknown"] == 0
    assert snapshot["escrow_known_counts"][11] == 1
    assert int(snapshot["unknown_counts"][11]) == 7  # 8 - known escrow card
    prediction = BaselineB().predict(snapshot["features"].reshape(1, -1))
    assert float(prediction.sum()) == pytest.approx(81.0)  # only opponent hands


def test_other_observers_private_exchange_never_enters_the_ledger():
    own, remaining = _actor_zero_state()
    base = _channels(0, own, remaining, phase=1)
    mutated = base.copy()
    mutated[1, 216 + 100] = 1  # another observer's private selection
    mutated[2, 216 + 101] = 1
    np.testing.assert_array_equal(
        public_ledger_features(base, actor=0, ledger=PublicLedger()),
        public_ledger_features(mutated, actor=0, ledger=PublicLedger()),
    )


def test_double_down_return_self_exchange_owner_stays_unassigned():
    # DOUBLE_DOWN has two candidate donors; the actor's own return card cannot
    # be attributed to one of them without hidden information.  It stays in the
    # residual pool (documented INCONCLUSIVE path), never guessed into K.
    own = list(range(0, 27))
    channels = _channels(
        0, own, [27, 27, 27, 27], phase=2, self_exchange=[100],
    )
    channels[0, 497] = 1             # previous_result present
    channels[0, 409] = 1             # DOUBLE_DOWN outcome one-hot
    for seat, rank in enumerate([1, 3, 2, 4]):
        channels[0, 412 + seat] = rank
    snapshot = public_ledger_snapshot(channels, actor=0, ledger=PublicLedger())
    assert snapshot["known_counts"].sum() == 0
    assert int(snapshot["unknown_counts"][11]) == 8  # visible but unassigned
    assert snapshot["ledger_conserved"] is True


# ---------------------------------------------------------------------------
# 6. Empty pool, finished seats, illegal inventory, joker cap, rotation, swap
# ---------------------------------------------------------------------------


def test_empty_unknown_pool_and_finished_seat_are_returned_directly():
    row = np.zeros(PUBLIC_FEATURE_WIDTH)
    row[:3] = [2.0, 0.0, 0.0]
    row[3 + 0 * 15 + 0] = 2.0
    prediction = BaselineB().predict(row.reshape(1, -1))
    np.testing.assert_allclose(prediction[0, 0, 0], 2.0)
    np.testing.assert_allclose(prediction[0, 1:], np.zeros((2, 15)))
    # A finished seat has zero remaining and receives no mass, while the two
    # live seats share the four unknown cards.
    finished = _feature_row((2.0, 0.0, 2.0), unknown=[4.0] + [0.0] * 14)
    prediction = BaselineB().predict(finished.reshape(1, -1))
    np.testing.assert_allclose(prediction[0, 1], np.zeros(15))
    np.testing.assert_allclose(prediction.sum(axis=(0, 2)), np.asarray([2.0, 0.0, 2.0]))
    # A known-held card on the finished seat is physically impossible.
    known = np.zeros((3, 15))
    known[1, 0] = 1.0
    impossible = _feature_row((2.0, 0.0, 2.0), known=known, unknown=[3.0] + [0.0] * 14)
    with pytest.raises(ValueError):
        BaselineB().predict(impossible.reshape(1, -1))


def test_negative_inventory_duplicate_owner_and_physical_caps_fail_closed():
    # K sum exceeding a published hand size.
    known = np.zeros((3, 15))
    known[0, 0] = 3.0
    row = _feature_row((2.0, 2.0, 2.0), known=known, unknown=[2.0] * 15)
    with pytest.raises(ValueError):
        BaselineB().predict(row.reshape(1, -1))
    # Known joker count above the physical maximum of two.
    known = np.zeros((3, 15))
    known[0, 14] = 3.0
    row = _feature_row((3.0, 0.0, 0.0), known=known, unknown=[0.0] * 15)
    with pytest.raises(ValueError):
        BaselineB().predict(row.reshape(1, -1))
    # Negative residual inventory.
    row = _feature_row((2.0, 2.0, 2.0), unknown=[-1.0] + [0.0] * 14)
    with pytest.raises(ValueError):
        BaselineB().predict(row.reshape(1, -1))
    # Duplicate ownership / double counting is rejected by the ledger.
    ledger = PublicLedger().apply_event(_transfer(("tribute", 1, 2, 98)))
    with pytest.raises(PublicLedgerError):
        ledger.apply_event(_transfer(("tribute", 3, 0, 98)))
    with pytest.raises(PublicLedgerError):
        ledger.apply_event(_commit(1, [98]))  # owner is seat 2
    with pytest.raises(PublicLedgerError):
        ledger.apply_event(_commit(2, [98]))
        ledger.apply_event(_commit(2, [98]))
    with pytest.raises(PublicLedgerError):
        repeated = PublicLedger()
        repeated.apply_event(_commit(0, [5]))
        repeated.apply_event(_commit(0, [5]))
    with pytest.raises(PublicLedgerError):
        PublicLedger().apply_event(_commit(0, [5])).apply_event(
            _transfer(("tribute", 0, 1, 5))
        )


def test_relative_rotation_and_copy_id_rename_preserve_rank_accounting():
    own, remaining = _actor_zero_state()
    ledger = PublicLedger().apply_event(_transfer(("tribute", 1, 2, 98)))
    snapshot_actor_zero = public_ledger_snapshot(
        _channels(0, own, remaining), actor=0, ledger=ledger,
    )
    # For actor 1, seat 2 is the *next* seat instead of the partner.
    snapshot_actor_one = public_ledger_snapshot(
        _channels(1, own, remaining), actor=1, ledger=ledger,
    )
    assert snapshot_actor_zero["known_counts"][1, 11] == 1
    assert snapshot_actor_one["known_counts"][0, 11] == 1
    # Copy-ID rename: card_id -> card_id + 54 (mod 108) changes ids, not ranks.
    renamed_own = [(card + 54) % 108 for card in own]
    renamed_ledger = PublicLedger().apply_event(_transfer(("tribute", 1, 2, 44)))
    renamed = public_ledger_snapshot(
        _channels(0, renamed_own, remaining), actor=0, ledger=renamed_ledger,
    )
    np.testing.assert_array_equal(
        snapshot_actor_zero["known_counts"], renamed["known_counts"],
    )
    np.testing.assert_array_equal(
        snapshot_actor_zero["unknown_counts"], renamed["unknown_counts"],
    )
    np.testing.assert_array_equal(
        snapshot_actor_zero["remaining"], renamed["remaining"],
    )


# ---------------------------------------------------------------------------
# 7. Ledger lifecycle: no cross-deal contamination / broadcast counted once
# ---------------------------------------------------------------------------


def test_public_event_is_counted_once_and_ledgers_do_not_cross_deals():
    from guandan_marvel.phase5 import belief_collect as collect

    event_row = [1, 258, 0, 39, 164, 7, 308]
    event = public_event_from_observations([event_row] * 4)
    assert event == {"kind": "commit", "seat": 1, "cards": [100]}
    first = collect._start_deal(
        {"config": {"level_rank": 3},
         "state": {"hands": [list(range(27 * seat, 27 * (seat + 1))) for seat in range(4)]}},
        0, collect.context_for(0),
    )
    second = collect._start_deal(
        {"config": {"level_rank": 4},
         "state": {"hands": [list(range(27 * seat, 27 * (seat + 1))) for seat in range(4)]}},
        1, collect.context_for(1),
    )
    assert first["ledger"] is not second["ledger"]
    first["ledger"].apply_event(event)
    assert second["ledger"].to_records() == []
    assert first["ledger"].to_records() == [event]


# ---------------------------------------------------------------------------
# 8. Real C++ engine cases: single/double tribute, same entity, anti-tribute
# ---------------------------------------------------------------------------


def _base_deal():
    return [
        list(range(0, 27)),
        list(range(27, 54)),
        list(range(54, 81)),
        list(range(81, 108)),
    ]


def _move_card(deal, card, seat):
    for source, hand in enumerate(deal):
        if card in hand:
            if source == seat:
                return deal
            swap = deal[seat][0]
            hand[hand.index(card)] = swap
            deal[seat][0] = card
            return deal
    raise AssertionError(f"card {card} is absent")


def _engine_reset(deal, *, level=6, leader=0, previous=None):
    from nanopokerlib import guandan as cpp

    env = cpp.EnvBatch(1, auto_reset=False)
    step = env.reset_from_deals(
        [deal], level_ranks=[level], leader_seats=[leader],
        previous_results=[previous],
    )
    return env, step


def _apply_step_event(step, ledger):
    event = public_event_from_observations(step.observations.tokens[0])
    if event is not None:
        ledger.apply_event(event)
    return event


def _step_next(env, step, ledger, token_id=None):
    """Emit exactly one protocol token (the engine is stepwise, not macro)."""
    rows = np.asarray(step.available_actions.tokens[0])
    weights = np.asarray(step.available_actions.weights[0])
    legal = [int(index) for index in np.flatnonzero(weights > 0)]
    if token_id is not None:
        legal = [index for index in legal if int(rows[index, 5]) == int(token_id)]
        assert len(legal) == 1, f"token {token_id} is not a unique legal next token"
    assert legal, "engine reported no legal next token"
    step = env.step(rows[legal[0]][None, :])
    _apply_step_event(step, ledger)
    return step


def _commit_action(env, step, ledger, tokens=()):
    """Drive one action to COMMIT, optionally fixing its first tokens."""
    before = int(env.debug_state(0)["state"]["committed_step"])
    for index in range(64):
        desired = tokens[index] if index < len(tokens) else None
        step = _step_next(env, step, ledger, desired)
        if int(env.debug_state(0)["state"]["committed_step"]) > before:
            return step
    raise AssertionError("action did not commit")


def _drive_to_phase(env, step, ledger, target_phase, *, max_actions=32):
    for _ in range(max_actions):
        if int(env.debug_state(0)["state"]["phase"]) == target_phase:
            return step
        step = _commit_action(env, step, ledger)
    raise AssertionError(f"engine did not reach phase {target_phase}")


def _single_tribute_previous():
    return {
        "ranking": [1, 2, 3, 4], "winner_team": 0, "outcome_class": "HEAD_THIRD",
    }


def _double_tribute_previous():
    return {
        "ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN",
    }


def test_real_engine_single_tribute_big_joker_stays_with_its_owner():
    # Single tribute: seat 3 gives its highest eligible card, the big joker 53,
    # to seat 0.  The old 18-wide baseline subtracted both visible jokers and
    # predicted zero for the owner's hand; the v2 ledger must keep K = 1.
    deal = _base_deal()
    _move_card(deal, 107, 2)   # keep both big jokers away from the donor
    _move_card(deal, 53, 3)    # donor seat 3 now holds 53
    env, step = _engine_reset(deal, previous=_single_tribute_previous())
    ledger = PublicLedger()
    step = _drive_to_phase(env, step, ledger, 0)
    debug = env.debug_state(0)
    assert debug["state"]["tribute_transfers"] == [
        {"source": 3, "target": 0, "card_id": 53},
    ]
    assert 53 in debug["state"]["hands"][0]
    assert ledger.known_owner.get(53) == 0
    channels = np.asarray(step.observations.channels[0])
    # Actor 1 sees seat 0 as its 'previous' relative seat.
    snapshot = public_ledger_snapshot(channels, actor=1, ledger=ledger)
    assert snapshot["known_counts"][2, 14] == 1
    prediction = BaselineB().predict(snapshot["features"].reshape(1, -1))
    assert prediction[0, 2, 14] >= 1.0
    assert snapshot["ledger_conserved"] is True


def test_real_engine_double_tribute_tracks_all_four_transfers():
    deal = _base_deal()
    _move_card(deal, 107, 2)   # avoid anti-tribute (only 53 remains with a donor)
    env, step = _engine_reset(deal, previous=_double_tribute_previous())
    ledger = PublicLedger()
    step = _drive_to_phase(env, step, ledger, 0)
    debug = env.debug_state(0)
    tributes = [row for row in debug["state"]["tribute_transfers"]]
    returns = [row for row in debug["state"]["return_transfers"]]
    assert len(tributes) == 2 and len(returns) == 2
    tribute_cards = {row["card_id"]: row for row in tributes}
    assert tribute_cards[53]["target"] == 0
    for row in tributes:
        card = row["card_id"]
        assert ledger.known_owner.get(card) == row["target"] or card in ledger.played
    for row in returns:
        card = row["card_id"]
        assert ledger.known_owner.get(card) == row["target"] or card in ledger.played
    for card, owner in ledger.known_owner.items():
        assert card in debug["state"]["hands"][owner]


def test_real_engine_same_entity_is_returned_and_then_played_once():
    # Craft a donor hand whose highest eligible card is a returnable normal
    # rank (SEVEN), so the recipient can return the *same physical card*.
    deck = list(range(108))
    low = [
        card for card in deck
        if card % 54 < 52 and (card % 54) // 4 in (0, 1, 2, 4)  # THREE..FIVE, SEVEN
    ]
    donor = low[:27]
    assert len(donor) == 27
    remaining = [card for card in deck if card not in donor]
    other_hands = [remaining[0:27], remaining[27:54], remaining[54:81]]
    deal = [other_hands[0], other_hands[1], other_hands[2], donor]
    env, step = _engine_reset(deal, previous=_single_tribute_previous())
    ledger = PublicLedger()
    # First action: the deterministic tribute selection (card token + COMMIT).
    step = _commit_action(env, step, ledger)
    debug = env.debug_state(0)
    assert int(debug["state"]["phase"]) == 2  # RETURN
    selection = debug["state"]["tribute_selections"][0]
    tribute_card = selection[1]
    assert ledger.known_owner == {}  # unresolved exchange is private
    # Recipient seat 0 returns the same physical entity.
    step = _commit_action(env, step, ledger, tokens=(64 + tribute_card,))
    debug = env.debug_state(0)
    assert int(debug["state"]["phase"]) == 0  # PLAY
    assert ledger.known_owner.get(tribute_card) == 3  # returned to the donor
    assert tribute_card in debug["state"]["hands"][3]
    assert tribute_card not in debug["state"]["hands"][0]
    # The donor (leader after the exchange) plays the same entity again.
    # A single action is FAMILY=8, CARD, COMMIT.
    step = _commit_action(env, step, ledger, tokens=(8, 64 + tribute_card))
    assert tribute_card in ledger.played
    assert tribute_card not in ledger.known_owner
    # No double counting: the engine's played set and the ledger agree.
    debug = env.debug_state(0)
    engine_played = {
        card for action in debug["state"]["history"]
        if action["kind"] not in ("pass", "tribute", "return")
        for card in action["card_ids"]
    }
    assert set(ledger.played) == engine_played


def test_real_engine_anti_tribute_publishes_no_transfer():
    deal = _base_deal()  # seat 1 holds 53, seat 3 holds 107
    env, step = _engine_reset(deal, previous=_double_tribute_previous())
    debug = env.debug_state(0)
    assert debug["state"]["anti_tribute"] is True
    assert int(debug["state"]["phase"]) == 0
    ledger = PublicLedger()
    _apply_step_event(step, ledger)
    assert ledger.to_records() == []


def test_cancelled_previous_flag_is_historical_and_publishes_no_transfer():
    # The C++ engine records ``tribute_cancelled`` from the previous hand but
    # still initializes *this* round's exchange from its donors (unless the
    # current anti-tribute condition holds).  The ledger must follow the real
    # phase and publish nothing before resolution, rather than inventing a
    # cancel from the flag.
    cancelled_previous = _double_tribute_previous()
    cancelled_previous["tribute_cancelled"] = True
    deal = _base_deal()
    _move_card(deal, 107, 2)  # current donors do not hold both big jokers
    env, step = _engine_reset(deal, previous=cancelled_previous)
    debug = env.debug_state(0)
    assert int(debug["state"]["phase"]) == 1  # TRIBUTE for the current round
    ledger = PublicLedger()
    _apply_step_event(step, ledger)  # reset event publishes no card
    assert ledger.to_records() == []


# ---------------------------------------------------------------------------
# 10. Versioning: old 18-wide rows and the old design hash are rejected
# ---------------------------------------------------------------------------


def test_legacy_features_and_old_gate_hash_are_rejected(tmp_path):
    assert PUBLIC_FEATURE_WIDTH == 64
    assert PHASE5_PUBLIC_LEDGER_SCHEMA.endswith("-v2")
    with pytest.raises(ValueError):
        validate_public_features(np.zeros(18, dtype=np.int64))
    with pytest.raises(ValueError):
        BaselineB().predict(np.zeros((1, 18)))
    gate = {
        "schema": "guandan-marvel-phase5-belief-gate-v1",
        "verdict": "PASS",
        "design_hash": OLD_DESIGN_HASH,
        "source_hash": "source",
        "config_hash": "config",
        "dataset": {"records_hash": "r", "split_hash": "s"},
        "epsilon": 0.05,
        "primary": {"improvement": 1.0},
        "metrics": {},
        "sufficiency": {},
        "reasons": [],
    }
    path = tmp_path / "old-gate.json"
    path.write_text(__import__("json").dumps(gate), encoding="utf-8")
    with pytest.raises(BeliefGateError):
        load_b0_gate(path, expected_design_hash=phase5_design_hash())
