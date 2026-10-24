"""Phase-5 belief labels, public-ledger accounting, conservation and group split."""
from __future__ import annotations

import numpy as np
import pytest

from guandan_marvel.phase5.belief_data import (
    MAX_COPIES_BY_RANK, PublicLedger, PublicLedgerError, build_label,
    canonical_context_identity, canonical_deal_identity, canonical_physical_deal,
    conservation_audit, dataset_hash, group_split, label_class_mask,
    leakage_probe_identity, parse_public_event_row, printed_rank_index,
    public_balance_features, public_history_bits, public_history_counts,
    public_ledger_features, public_ledger_snapshot, public_relative_remaining,
    public_unseen_by_rank, rank_counts, split_hash, validate_public_features,
    visible_and_played_cards,
)
from guandan_marvel.phase5.design import PUBLIC_FEATURE_WIDTH


def _full_deal():
    deck = list(range(108))
    return [deck[0:27], deck[27:54], deck[54:81], deck[81:108]]


def _rotate(hands, k):
    return [list(hands[(seat - k) % 4]) for seat in range(4)]


def _swap(hands):
    return [[(card + 54) % 108 for card in hand] for hand in hands]


def _channels(actor=0):
    channels = np.zeros((4, 512), dtype=np.uint8)
    return channels


def _full_channels(actor, own_cards, remaining, *, phase=0, winning=()):
    channels = np.zeros((4, 512), dtype=np.uint8)
    for card in own_cards:
        channels[actor, int(card)] = 1
    for card in winning:
        channels[actor, 108 + int(card)] = 1
    channels[:, 356:360] = np.asarray(remaining, dtype=np.uint8)
    channels[actor, 384 + int(phase)] = 1
    return channels


def _transfer_event(transfers):
    return {"kind": "exchange_resolved", "transfers": list(transfers)}


def _commit_event(seat, cards):
    return {"kind": "commit", "seat": int(seat), "cards": list(cards)}


def test_printed_rank_layout_matches_the_physical_two_deck_scheme():
    assert [printed_rank_index(card) for card in (0, 1, 2, 3)] == [0, 0, 0, 0]
    assert printed_rank_index(4) == 1
    assert printed_rank_index(48) == 12
    assert printed_rank_index(51) == 12
    assert [printed_rank_index(card) for card in (52, 53)] == [13, 14]
    assert [printed_rank_index(card) for card in (54, 106, 107)] == [0, 13, 14]
    counts = rank_counts([0, 4, 52, 54, 107])
    assert counts[0] == 2
    assert counts[1] == 1
    assert counts[13] == 1
    assert counts[14] == 1


def test_label_counts_printed_rank_and_joker_max_and_class_mask():
    # Card 14 is SIX of hearts in deck 0: if level SIX, it is a wild heart, but
    # the label must still count its *printed* rank, never the declaration.
    hands = [[], [14], [], []]
    label = build_label(hands, actor=0)
    assert int(label.counts[0, 3]) == 1
    assert int(label.counts.sum()) == 1
    assert label.class_mask.shape == (3, 15, 9)
    for rank, maximum in enumerate(MAX_COPIES_BY_RANK):
        assert label.class_mask[0, rank, : maximum + 1].all()
        assert not label.class_mask[0, rank, maximum + 1:].any()
    with pytest.raises(ValueError):
        build_label([[], [52, 52, 52], [], []], actor=0)  # three joker copies
    # Two physical copies per joker and eight per normal rank are the physical
    # maxima, not the 13-class level/declaration axis.
    assert MAX_COPIES_BY_RANK == (8,) * 13 + (2, 2)


def test_public_ledger_counts_known_held_transfers_and_conserves():
    # Actor 0 holds cards 0..26 (27 cards).  Seat 1 first plays card 100 (rank
    # 11), then the resolved exchange reveals 98 held by seat 2 and 99 held by
    # seat 1.  The old 18-wide balance subtracted all three from "unseen" and
    # could not keep 98/99 with their current owners.
    ledger = PublicLedger()
    ledger.apply_event(_transfer_event([
        {"kind": "tribute", "source": 1, "target": 2, "card": 98},
        {"kind": "return", "source": 2, "target": 1, "card": 99},
    ]))
    ledger.apply_event(_commit_event(1, [100]))
    channels = _full_channels(
        0, range(0, 27), [27, 26, 27, 27], phase=0, winning=[100],
    )
    features = public_balance_features(channels, actor=0, ledger=ledger)
    assert features.shape == (PUBLIC_FEATURE_WIDTH,) == (64,)
    assert features[:3].tolist() == [26, 27, 27]
    known = features[3:48].reshape(3, 15)
    assert known[0, 11] == 1  # next = seat 1 holds the returned card 99
    assert known[1, 11] == 1  # partner = seat 2 holds the tribute card 98
    assert known[2].sum() == 0
    unknown = features[48:63]
    assert int(unknown[11]) == 5  # 8 - played 100 - known 98/99
    assert int(features[63]) == 0
    snapshot = public_ledger_snapshot(channels, actor=0, ledger=ledger)
    assert snapshot["ledger_conserved"] is True
    assert snapshot["played_total"] == 1
    assert snapshot["known_total"] == 2
    assert int(snapshot["unknown_counts"].sum()) == int(
        snapshot["hand_capacity"].sum()
    )
    np.testing.assert_array_equal(
        public_unseen_by_rank(channels, actor=0, ledger=ledger), unknown,
    )
    np.testing.assert_array_equal(
        public_ledger_features(channels, actor=0, ledger=ledger), features,
    )
    # Relative order is next = actor+1, partner = actor+2, previous = actor+3;
    # the actor's own count must never appear.
    np.testing.assert_array_equal(
        public_relative_remaining(channels, actor=2),
        np.asarray([27, 27, 26]),
    )
    np.testing.assert_array_equal(
        public_relative_remaining(channels, actor=0),
        np.asarray([26, 27, 27]),
    )
    with pytest.raises(ValueError):
        validate_public_features(np.zeros(18, dtype=np.int64))


def test_public_history_bits_do_not_double_count_and_are_binary():
    bits = public_history_bits([5, 5, 8, 107])
    assert bits.shape == (108,)
    assert int(bits.sum()) == 3
    assert int(bits[5]) == 1 and int(bits[8]) == 1 and int(bits[107]) == 1
    assert public_history_counts([5, 5, 8, 107]).tolist()[1] == 2
    assert public_history_counts([5, 5, 8, 107]).tolist()[2] == 1
    with pytest.raises(ValueError):
        public_history_bits([108])


def _full_hands_for_actor(actor, own_cards, next_cards, partner_cards, previous_cards):
    hands = [None] * 4
    hands[actor] = list(own_cards)
    hands[(actor + 1) % 4] = list(next_cards)
    hands[(actor + 2) % 4] = list(partner_cards)
    hands[(actor + 3) % 4] = list(previous_cards)
    return hands


def test_hidden_hand_counterfactual_cannot_change_probe_inputs():
    # The two deals differ only in the hidden opponent hands (the 'next' and the
    # 'previous' seat); the actor's own 27 cards and every published size are
    # identical.
    own = list(range(0, 27))
    next_cards = list(range(27, 54))
    partner = list(range(54, 81))
    previous = list(range(81, 108))
    deal_a = _full_hands_for_actor(1, own, next_cards, partner, previous)
    # A hidden-only swap between the next seat (absolute seat 2) and the previous
    # seat (absolute seat 0) exchanges two entities of *different* printed rank:
    # 27 is rank 6 ((27 % 54) // 4 == 6) and 84 is rank 7 ((84 % 54) // 4 == 7).
    # The old fixture paired 27 with its deck-1 twin 81 (both rank 6, because
    # 81 % 54 == 27), so the 15-rank label could not change.  Both opponent rank
    # profiles change now, while the actor's visible state stays bit-identical.
    deal_b = [list(hand) for hand in deal_a]
    deal_b[2][deal_b[2].index(27)] = 84
    deal_b[0][deal_b[0].index(84)] = 27
    # The counterfactual is still a legal 108-card partition: every physical
    # entity appears exactly once.
    assert sorted(card for hand in deal_b for card in hand) == list(range(108))
    channels = _full_channels(1, own, [27, 27, 27, 27], phase=0)
    tokens = np.zeros((4, 48), dtype=np.int64)
    tokens[1, 0] = 1  # BOS_TOKEN
    ledger = PublicLedger()
    features = public_ledger_features(channels, actor=1, ledger=ledger)
    np.testing.assert_array_equal(
        features, public_ledger_features(channels, actor=1, ledger=PublicLedger()),
    )
    assert leakage_probe_identity(tokens, channels) == leakage_probe_identity(
        tokens, channels,
    )
    # The hidden hands really differ, in both opponent seats ...
    assert deal_a[2] != deal_b[2]
    assert deal_a[0] != deal_b[0]
    # ... and the label changes in *both* opponent relative seats ('next' and
    # 'previous'), because the exchanged entities have different printed ranks.
    label_a = build_label(deal_a, 1)
    label_b = build_label(deal_b, 1)
    assert not np.array_equal(label_a.counts, label_b.counts)
    assert int(label_a.counts[0].sum()) == 27
    assert int(label_b.counts[0].sum()) == 27
    np.testing.assert_array_equal(label_a.counts[1], rank_counts(partner))
    np.testing.assert_array_equal(label_a.counts[2], rank_counts(previous))
    assert label_a.counts[0].tolist() != label_b.counts[0].tolist()
    assert label_a.counts[2].tolist() != label_b.counts[2].tolist()
    channels_b = _channels(actor=1)
    channels_b[1, 5] = 1
    own_visible, exposed = visible_and_played_cards(channels_b, actor=1, public_cards=[10])
    assert own_visible == {5}
    assert exposed == {10}


def test_event_parser_rejects_private_cards_and_malformed_grammar():
    # Private prefix and private exchange commits publish no card.
    private_prefix = [1, 257, 2, 39, 65, 308]
    private_exchange = [1, 259, 1, 39, 164, 7, 308]
    assert parse_public_event_row(private_prefix) is None
    assert parse_public_event_row(private_exchange) is None
    # A public commit row publishes its cards once.
    assert parse_public_event_row([1, 258, 0, 39, 164, 7, 308]) == {
        "kind": "commit", "seat": 1, "cards": [100],
    }
    # A tribute record missing its COMMIT terminator must fail closed.
    with pytest.raises(PublicLedgerError):
        parse_public_event_row([1, 260, 1, 39, 4, 39, 40, 164, 308])


def test_conservation_audit_uses_finished_seats_and_current_owners():
    hands = _full_deal()
    audit = conservation_audit(hands=hands, played=[], escrow=[], actor=0)
    assert audit["conserved"] and not audit["missing_cards"] and not audit["duplicated_cards"]
    assert audit["finished_players_zero"] is True  # no seat declared finished
    audit_finished = conservation_audit(
        hands=hands, played=[], escrow=[], actor=0, finished_seats=[0],
    )
    assert audit_finished["finished_players_zero"] is False
    finished_hands = [[], hands[1], hands[2], hands[3]]
    played = list(hands[0])
    audit_ok = conservation_audit(
        hands=finished_hands, played=played, escrow=[], actor=0,
        finished_seats=[0],
    )
    assert audit_ok["conserved"] and audit_ok["finished_players_zero"] is True
    duplicated = [list(hands[0]) + [0], hands[1], hands[2], hands[3]]
    assert not conservation_audit(
        hands=duplicated, played=[], escrow=[], actor=0,
    )["conserved"]
    missing = [hands[0][1:], hands[1], hands[2], hands[3]]
    assert conservation_audit(
        hands=missing, played=[], escrow=[], actor=0,
    )["missing_cards"] == [hands[0][0]]
    with pytest.raises(ValueError):
        conservation_audit(hands=hands[:3], played=[], escrow=[], actor=0)


def test_group_identity_is_context_rotation_and_copy_swap_invariant():
    hands = _full_deal()
    base = canonical_deal_identity(
        hands, level_rank=3, leader_seat=0, previous_result=None,
    )
    assert canonical_deal_identity(
        hands, level_rank=15, leader_seat=3,
        previous_result={"ranking": [1, 2, 3, 4]},
    ) == base
    for k in range(4):
        assert canonical_deal_identity(
            _rotate(hands, k), level_rank=7, leader_seat=1,
            previous_result={"outcome_class": "HEAD_THIRD"},
        ) == base
    assert canonical_deal_identity(_swap(hands)) == base
    assert canonical_deal_identity(_swap(_rotate(hands, 2))) == base
    changed = [list(hand) for hand in hands]
    changed[0][0], changed[1][0] = changed[1][0], changed[0][0]
    assert canonical_deal_identity(changed) != base
    canonical = canonical_physical_deal(hands)
    assert canonical == canonical_physical_deal(_rotate(_swap(hands), 3))
    assert canonical_context_identity(level_rank=3, leader_seat=0, previous_result=None) != (
        canonical_context_identity(level_rank=4, leader_seat=0, previous_result=None)
    )


def test_group_split_is_deal_level_deterministic_and_disjoint():
    groups = [f"deal-{index}" for index in range(100)]
    assignment = group_split(
        groups, seed=20261019,
        fractions={"train": 0.70, "validation": 0.15, "test": 0.15},
    )
    assert set(assignment) == set(groups)
    counts = {name: sum(1 for value in assignment.values() if value == name)
              for name in ("train", "validation", "test")}
    assert counts == {"train": 70, "validation": 15, "test": 15}
    assert group_split(
        groups, seed=20261019,
        fractions={"train": 0.70, "validation": 0.15, "test": 0.15},
    ) == assignment
    # A duplicated group id (a replay / augmentation) cannot cross splits.
    assert group_split(
        groups + groups[:5], seed=20261019,
        fractions={"train": 0.70, "validation": 0.15, "test": 0.15},
    ) == assignment
    assert split_hash(assignment) == split_hash(dict(reversed(list(assignment.items()))))
    with pytest.raises(ValueError):
        group_split(groups, seed=1, fractions={"train": 0.5})


def test_dataset_hash_is_order_sensitive_and_stable():
    records = [{"a": 1, "b": [2, 3]}, {"a": 4, "b": [5]}]
    assert dataset_hash(records) == dataset_hash(list(records))
    assert dataset_hash(records) != dataset_hash(list(reversed(records)))
