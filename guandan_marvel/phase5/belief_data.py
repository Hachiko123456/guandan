# SPDX-License-Identifier: MIT
"""Belief B0 labels, leakage boundary and grouped split.

The *label* channel is deliberately separate from every model input:

* ``label_remaining`` is produced from hidden full state by an oracle counter and
  is written into the dataset under its own key;
* the policy/probe input is exactly ``obs_tokens`` + ``obs_channels`` at that
  macro start (what the acting seat can see) plus, for the *public* baselines
  only, quantities derivable from public information;
* :func:`public_information_balance` recomputes the public view from the
  observation alone and refuses to look at hidden hands.

Label definition (spec 4.1): for each of the three *relative* seats
(``next`` = actor+1, ``partner`` = actor+2, ``previous`` = actor+3) the number of
remaining physical cards per *printed rank*, i.e. 13 normal ranks (0..8) plus
small/big joker (0..2).  A wild card counts under its **printed** rank, never
under the rank it was declared as; a declared wild assignment is recorded
separately.  Finished players have zero remaining cards.

Physical conservation is audited per sample over ``own + opponents + played``
with publicly transferred cards counted exactly once (current owner).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .design import (
    CARDS_PER_DECK, JOKER_COUNT_MAX, MAX_COPIES_BY_RANK, NUM_PRINTED_RANKS,
    NUM_RELATIVE_SEATS, PHASE5_BELIEF_DATASET_SCHEMA, PHASE5_PUBLIC_LEDGER_SCHEMA,
    PRINTED_RANK_LABELS, PUBLIC_ESCROW_UNKNOWN_INDEX, PUBLIC_FEATURE_WIDTH,
    PUBLIC_KNOWN_SLICE, PUBLIC_REMAINING_SLICE, PUBLIC_UNKNOWN_SLICE,
    RELATIVE_SEAT_OFFSETS, RELATIVE_SEATS, TOTAL_PHYSICAL_CARDS,
)
from .symmetry import (
    BOS_TOKEN, CARD_TOKEN_BASE, COMMIT_TOKEN, EVENT_END, EVENT_EXCHANGE_RESOLVED,
    EVENT_PRIVATE_EXCHANGE_COMMIT, EVENT_PRIVATE_PREFIX, EVENT_PUBLIC_COMMIT,
    EVENT_RESET, LENGTH_TOKEN_BASE, NUM_PHYSICAL_CARDS, PHASE_ONE_HOT,
    PREVIOUS_OUTCOME, PREVIOUS_RANKING, REMAINING_COUNTS, RETURN_TOKEN,
    TRIBUTE_TOKEN,
)

# Physical card layout: id // 54 is the deck, id % 54 the deck offset; the first
# 52 offsets are the four suits of PRINTED_RANKS[offset // 4], the last two are
# the jokers.
NORMAL_CARDS_PER_DECK = 52
# Version tag of the records-hash canonicalisation.  A change to the canonical
# form must bump this tag and produce a new derived provenance artifact instead
# of silently re-hashing an existing delivery.
RECORDS_CANONICALIZATION = "guandan-marvel-phase5-records-canonical-v1"


def printed_rank_index(card_id: int) -> int:
    """Printed-rank index 0..14 of a physical card id (jokers are 13, 14)."""
    value = int(card_id)
    if not 0 <= value < TOTAL_PHYSICAL_CARDS:
        raise ValueError(f"card id {value} is outside 0..107")
    offset = value % CARDS_PER_DECK
    if offset < NORMAL_CARDS_PER_DECK:
        return offset // 4
    return 13 + (offset - NORMAL_CARDS_PER_DECK)


def deck_index(card_id: int) -> int:
    return int(card_id) // CARDS_PER_DECK


def rank_counts(card_ids: Sequence[int]) -> np.ndarray:
    """Count physical cards per printed rank (15 entries)."""
    counts = np.zeros(NUM_PRINTED_RANKS, dtype=np.int64)
    for card_id in card_ids:
        counts[printed_rank_index(int(card_id))] += 1
    return counts


def relative_seat(actor: int, offset: int) -> int:
    return (int(actor) + int(offset)) % 4


@dataclass(frozen=True, slots=True)
class BeliefLabel:
    """One macro-start label block.

    ``counts`` is ``[3, 15]`` int64 in relative-seat order; ``class_mask`` is
    ``[3, 15, 9]`` bool, False for count classes that no legal hand can realise
    (joker classes 3..8 are masked; normal ranks keep 0..8).
    """

    counts: np.ndarray
    class_mask: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts.tolist(),
            "class_mask": self.class_mask.tolist(),
        }


def label_class_mask() -> np.ndarray:
    """Precomputed legal count classes ``[15, 9]``."""
    mask = np.zeros((NUM_PRINTED_RANKS, 9), dtype=bool)
    for rank in range(NUM_PRINTED_RANKS):
        maximum = MAX_COPIES_BY_RANK[rank]
        mask[rank, : maximum + 1] = True
    return mask


def build_label(hands: Sequence[Sequence[int]], actor: int) -> BeliefLabel:
    """Oracle label for the three seats relative to ``actor``."""
    if len(hands) != 4:
        raise ValueError("a deal requires four hands")
    counts = np.zeros((len(RELATIVE_SEATS), NUM_PRINTED_RANKS), dtype=np.int64)
    for index, offset in enumerate(RELATIVE_SEAT_OFFSETS):
        seat = relative_seat(actor, offset)
        counts[index] = rank_counts(hands[seat])
    for rank in range(NUM_PRINTED_RANKS):
        if int(counts[:, rank].max()) > MAX_COPIES_BY_RANK[rank]:
            raise ValueError(
                f"label count for rank {PRINTED_RANK_LABELS[rank]} exceeds the physical "
                f"maximum {MAX_COPIES_BY_RANK[rank]}"
            )
    mask = np.broadcast_to(label_class_mask(), counts.shape + (9,)).copy()
    return BeliefLabel(counts=counts, class_mask=mask)


# ---------------------------------------------------------------------------
# Causal public-information ledger (v2)
# ---------------------------------------------------------------------------
#
# Entity states at one acting seat's macro start:
#
#   O  own hand, exactly the visible SELF_HAND bits of the actor's channel row;
#   P  cards committed in public play actions (PASS adds nothing);
#   K  cards publicly revealed by a resolved tribute/return event whose current
#      owner is still a relative seat and which have not since been played;
#   E  known non-hand cards (the actor's own tribute selection while it is in
#      the public escrow bucket);
#   X  escrow cards whose identity is still private (another donor's selection);
#   U  residual physical inventory not attributable to any bucket above.
#
# The buckets are disjoint and partition the 108 physical cards.  ``u_r`` is
# the per-rank count of U; because U is the disjoint union of unknown opponent
# hand capacity H and unknown escrow capacity X, ``sum_r u_r == H + X``.
#
# ``K`` is updated causally: a resolved-event transfer moves a card from its
# previous owner (if known) to the event's target, and a later public play
# moves it to P.  A private prefix/exchange-selection event never changes it.


class PublicLedgerError(ValueError):
    """A public event or ledger snapshot is malformed or inconsistent."""


def _trim_event(row: Sequence[int]) -> list[int]:
    tokens = [int(token) for token in row]
    while tokens and tokens[-1] == 0:
        tokens.pop()
    return tokens


def parse_public_event_row(row: Sequence[int]) -> dict[str, Any] | None:
    """Parse one causal observation event row.

    Returns ``None`` for reset and private events (they publish no card to the
    acting seat), otherwise a compact public record.  The full exchange grammar
    is parsed, so ownership is never inferred from "a CARD token appeared".
    """
    tokens = _trim_event(row)
    if not tokens:
        return None
    if tokens[0] != BOS_TOKEN or tokens[-1] != EVENT_END:
        raise PublicLedgerError("observation event is not BOS..EVENT_END framed")
    if len(tokens) < 2:
        raise PublicLedgerError("observation event has no kind")
    kind = tokens[1]
    if kind in (
        EVENT_RESET, EVENT_PRIVATE_PREFIX, EVENT_PRIVATE_EXCHANGE_COMMIT,
    ):
        return None
    if len(tokens) < 4:
        raise PublicLedgerError("public event is missing its phase/seat header")
    seat = int(tokens[3]) - LENGTH_TOKEN_BASE
    if not 0 <= seat < 4:
        raise PublicLedgerError(f"public event has an invalid seat token {tokens[3]}")
    if kind == EVENT_PUBLIC_COMMIT:
        cards = sorted({
            int(token) - CARD_TOKEN_BASE
            for token in tokens
            if CARD_TOKEN_BASE <= token < CARD_TOKEN_BASE + NUM_PHYSICAL_CARDS
        })
        return {"kind": "commit", "seat": seat, "cards": cards}
    if kind == EVENT_EXCHANGE_RESOLVED:
        transfers: list[dict[str, Any]] = []
        index = 4
        # Each transfer record is exactly five tokens:
        # TRIBUTE|RETURN, source-length, target-length, card, COMMIT.
        while index < len(tokens) - 1:
            if index + 5 > len(tokens) - 1:
                raise PublicLedgerError("truncated exchanged-transfer record")
            transfer_kind = int(tokens[index])
            if transfer_kind not in (TRIBUTE_TOKEN, RETURN_TOKEN):
                raise PublicLedgerError(
                    f"unexpected exchange record token {transfer_kind}"
                )
            source = int(tokens[index + 1]) - LENGTH_TOKEN_BASE
            target = int(tokens[index + 2]) - LENGTH_TOKEN_BASE
            card = int(tokens[index + 3]) - CARD_TOKEN_BASE
            commit = int(tokens[index + 4])
            if not 0 <= source < 4 or not 0 <= target < 4:
                raise PublicLedgerError("exchange transfer has an invalid seat token")
            if not 0 <= card < TOTAL_PHYSICAL_CARDS:
                raise PublicLedgerError("exchange transfer has an invalid card token")
            if commit != COMMIT_TOKEN:
                raise PublicLedgerError("exchange transfer record is not COMMIT-terminated")
            transfers.append({
                "kind": "tribute" if transfer_kind == TRIBUTE_TOKEN else "return",
                "source": source,
                "target": target,
                "card": card,
            })
            index += 5
        if index != len(tokens) - 1:
            raise PublicLedgerError("exchange event has trailing tokens")
        return {"kind": "exchange_resolved", "transfers": transfers}
    raise PublicLedgerError(f"unknown public event kind {kind}")


def public_event_from_observations(
    rows: Sequence[Sequence[int]],
) -> dict[str, Any] | None:
    """Extract the one public event of a step from the four observer rows.

    Public events are broadcast identically to all four observers; private
    events appear only in the acting seat's row (and parse to ``None``).  The
    four rows must not disagree, so a public event can never be counted twice
    or silently dropped.
    """
    parsed = [
        event for event in (parse_public_event_row(row) for row in rows)
        if event is not None
    ]
    if not parsed:
        return None
    first = parsed[0]
    if any(event != first for event in parsed[1:]):
        raise PublicLedgerError("public event rows disagree")
    return first


def _normalized_event(event: Mapping[str, Any]) -> dict[str, Any]:
    kind = event.get("kind")
    if kind == "commit":
        cards = sorted({int(card) for card in event.get("cards", ())})
        if len(cards) != len(event.get("cards", ())):
            raise PublicLedgerError("commit event repeats a physical card")
        return {"kind": "commit", "seat": int(event["seat"]), "cards": cards}
    if kind == "exchange_resolved":
        transfers = []
        for transfer in event.get("transfers", ()):
            transfers.append({
                "kind": str(transfer["kind"]),
                "source": int(transfer["source"]),
                "target": int(transfer["target"]),
                "card": int(transfer["card"]),
            })
        return {"kind": "exchange_resolved", "transfers": transfers}
    raise PublicLedgerError(f"unknown public event kind {kind!r}")


@dataclass
class PublicLedger:
    """Incremental, actor-independent public ownership ledger."""

    played: dict[int, int] = field(default_factory=dict)
    known_owner: dict[int, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def apply_event(self, event: Mapping[str, Any]) -> "PublicLedger":
        normalized = _normalized_event(event)
        if normalized["kind"] == "commit":
            self._apply_commit(normalized["seat"], normalized["cards"])
        else:
            self._apply_exchange(normalized["transfers"])
        self.events.append(normalized)
        return self

    def _apply_commit(self, seat: int, cards: Sequence[int]) -> None:
        if not 0 <= int(seat) < 4:
            raise PublicLedgerError(f"commit has an invalid seat {seat}")
        validated: list[int] = []
        for card in cards:
            card = int(card)
            if not 0 <= card < TOTAL_PHYSICAL_CARDS:
                raise PublicLedgerError(f"commit has an invalid card id {card}")
            if card in self.played:
                raise PublicLedgerError(f"card {card} was publicly played twice")
            owner = self.known_owner.get(card)
            if owner is not None and owner != int(seat):
                raise PublicLedgerError(
                    f"card {card} is known to be held by seat {owner} but was "
                    f"played by seat {seat}"
                )
            validated.append(card)
        # Apply only after every card passed, so a rejected event never leaves
        # a partially mutated ledger.
        for card in validated:
            self.known_owner.pop(card, None)
            self.played[card] = int(seat)

    def _apply_exchange(self, transfers: Sequence[Mapping[str, Any]]) -> None:
        played = dict(self.played)
        known_owner = dict(self.known_owner)
        for transfer in transfers:
            kind = str(transfer["kind"])
            if kind not in ("tribute", "return"):
                raise PublicLedgerError(f"unknown exchange transfer kind {kind!r}")
            source = int(transfer["source"])
            target = int(transfer["target"])
            card = int(transfer["card"])
            if not 0 <= source < 4 or not 0 <= target < 4:
                raise PublicLedgerError("exchange transfer has an invalid seat")
            if source == target:
                raise PublicLedgerError("exchange transfer source equals target")
            if not 0 <= card < TOTAL_PHYSICAL_CARDS:
                raise PublicLedgerError(f"exchange transfer has an invalid card id {card}")
            if card in played:
                raise PublicLedgerError(f"card {card} was transferred after being played")
            owner = known_owner.pop(card, None)
            if owner is not None and owner != source:
                raise PublicLedgerError(
                    f"card {card} is known to be held by seat {owner} but was "
                    f"transferred by seat {source}"
                )
            known_owner[card] = target
        # Commit the whole resolved event atomically.
        self.played = played
        self.known_owner = known_owner

    def public_cards(self) -> set[int]:
        """Cards revealed by public events so far (played union still-held)."""
        return set(self.played) | set(self.known_owner)

    def to_records(self) -> list[dict[str, Any]]:
        return [_normalized_event(event) for event in self.events]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "PublicLedger":
        ledger = cls()
        for event in records:
            ledger.apply_event(event)
        return ledger


def channel_phase(channels: np.ndarray, *, actor: int) -> int:
    """Decode the one-hot phase channel of the acting observer."""
    row = np.asarray(channels)
    if row.ndim != 2 or row.shape[0] != 4:
        raise ValueError("obs_channels must be [4, 512]")
    one_hot = np.asarray(row[int(actor), PHASE_ONE_HOT], dtype=np.int64)
    if int(one_hot.sum()) != 1:
        raise PublicLedgerError("phase channel is not one-hot")
    return int(np.argmax(one_hot))


def _single_tribute_partner(
    channels: np.ndarray, *, actor: int,
) -> tuple[int, int] | None:
    """``(donor, recipient)`` when the public previous result is single-tribute.

    ``None`` for no previous result, a cancelled tribute, anti-tribute, or a
    DOUBLE_DOWN relation (two donors/two recipients make an individual
    transfer's owner genuinely ambiguous without hidden data).
    """
    row = np.asarray(channels)[int(actor)]
    if not bool(row[497]):
        return None
    if bool(row[486]) or bool(row[488]):
        return None
    outcome = np.asarray(row[PREVIOUS_OUTCOME], dtype=np.int64)
    if int(outcome.sum()) != 1:
        raise PublicLedgerError("previous outcome channel is not one-hot")
    if int(np.argmax(outcome)) == 0:
        return None  # DOUBLE_DOWN
    ranking = [int(value) for value in row[PREVIOUS_RANKING]]
    if sorted(ranking) != [1, 2, 3, 4]:
        raise PublicLedgerError("previous ranking channels are not a permutation")
    seat_of_rank = {rank: seat for seat, rank in enumerate(ranking)}
    return seat_of_rank[4], seat_of_rank[1]


def _relative_index(actor: int, seat: int) -> int | None:
    offset = (int(seat) - int(actor)) % 4
    for index, expected in enumerate(RELATIVE_SEAT_OFFSETS):
        if offset == expected:
            return index
    return None


def public_ledger_snapshot(
    obs_channels: np.ndarray,
    *,
    actor: int,
    ledger: PublicLedger,
) -> dict[str, Any]:
    """Build the auditable bucket snapshot for one acting seat.

    The snapshot is defined for the seat whose macro start is being recorded.
    In the accepted encoder a donor's just-received return card is deliberately
    hidden from its own ``SELF_HAND`` slice during the RETURN phase; the
    ``remaining count == visible hand`` check therefore fail-closes instead of
    guessing that card.  Only the actor's channel row and the causal public
    ledger are read.  The function also fails closed on negative inventory,
    duplicate ownership, a conservation mismatch, or a row that claims
    ownership of a card the actor cannot currently see.
    """
    channels = np.asarray(obs_channels)
    if channels.ndim != 2 or channels.shape[0] != 4:
        raise ValueError("obs_channels must be [4, 512]")
    actor = int(actor)
    if not 0 <= actor < 4:
        raise ValueError("actor must be 0..3")
    if not isinstance(ledger, PublicLedger):
        raise TypeError("ledger must be a PublicLedger")
    row = channels[actor]
    max_copies = np.asarray(MAX_COPIES_BY_RANK, dtype=np.int64)

    own_cards = {
        card for card in range(TOTAL_PHYSICAL_CARDS) if int(row[card])
    }
    own_counts = np.zeros(NUM_PRINTED_RANKS, dtype=np.int64)
    for card in own_cards:
        own_counts[printed_rank_index(card)] += 1
    winning = {
        card for card in range(TOTAL_PHYSICAL_CARDS) if int(row[108 + card])
    }
    played_cards = set(int(card) for card in ledger.played)
    if winning - played_cards:
        raise PublicLedgerError(
            "current-winning channel contains cards absent from the public play history"
        )

    known_counts = np.zeros((NUM_RELATIVE_SEATS, NUM_PRINTED_RANKS), dtype=np.int64)
    for card, owner in sorted(ledger.known_owner.items()):
        owner = int(owner)
        rank = printed_rank_index(int(card))
        if owner == actor:
            if int(card) not in own_cards:
                raise PublicLedgerError(
                    f"card {card} is known to be owned by the actor but is not in "
                    "its visible hand"
                )
            continue
        index = _relative_index(actor, owner)
        if index is None:
            raise PublicLedgerError(f"known owner {owner} is not a relative seat")
        known_counts[index, rank] += 1

    remaining_absolute = np.asarray(row[REMAINING_COUNTS], dtype=np.int64)
    remaining = np.asarray(
        [remaining_absolute[(actor + offset) % 4] for offset in RELATIVE_SEAT_OFFSETS],
        dtype=np.int64,
    )
    if int(remaining_absolute[actor]) != len(own_cards):
        raise PublicLedgerError(
            "published own remaining count disagrees with the visible hand"
        )

    phase = channel_phase(channels, actor=actor)
    escrow_known_counts = np.zeros(NUM_PRINTED_RANKS, dtype=np.int64)
    escrow_known_cards: list[int] = []
    self_exchange = {
        card for card in range(TOTAL_PHYSICAL_CARDS) if int(row[216 + card])
    }
    if phase == 1:  # TRIBUTE: a self-selected tribute card is in escrow.
        for card in sorted(self_exchange):
            if (
                card not in own_cards
                and card not in played_cards
                and card not in ledger.known_owner
            ):
                escrow_known_counts[printed_rank_index(card)] += 1
                escrow_known_cards.append(int(card))
    elif phase == 2:  # RETURN: a single-tribute partner is deterministic.
        partner = _single_tribute_partner(channels, actor=actor)
        if partner is not None:
            donor, recipient = partner
            if actor == donor:
                owner = recipient  # own tribute selection now held by recipient
            elif actor == recipient:
                owner = donor  # own return transfer now held by the donor
            else:
                owner = None
            if owner is not None:
                index = _relative_index(actor, owner)
                if index is None:
                    raise PublicLedgerError("single-tribute partner is not relative")
                for card in self_exchange:
                    if (
                        card not in own_cards
                        and card not in played_cards
                        and card not in ledger.known_owner
                    ):
                        known_counts[index, printed_rank_index(card)] += 1

    known_total = int(known_counts.sum())
    if known_total:
        owner_sums = known_counts.sum(axis=1)
        if np.any(owner_sums > remaining):
            raise PublicLedgerError("known-held lower bound exceeds a published hand size")
    if np.any(known_counts > max_copies[None, :]):
        raise PublicLedgerError("known-held count exceeds a physical per-rank maximum")

    played_counts = np.zeros(NUM_PRINTED_RANKS, dtype=np.int64)
    for card in played_cards:
        played_counts[printed_rank_index(card)] += 1

    escrow_total = (
        TOTAL_PHYSICAL_CARDS
        - int(remaining_absolute.sum())
        - len(played_cards)
    )
    if escrow_total < 0:
        raise PublicLedgerError("negative escrow capacity (hands + played exceed 108)")
    escrow_known_total = int(escrow_known_counts.sum())
    escrow_unknown = int(escrow_total - escrow_known_total)
    if escrow_unknown < 0:
        raise PublicLedgerError("known escrow cards exceed the escrow capacity")
    if phase != 1 and escrow_total != 0:
        raise PublicLedgerError("escrow capacity must be zero outside the tribute phase")

    unknown = (
        max_copies
        - own_counts
        - played_counts
        - known_counts.sum(axis=0)
        - escrow_known_counts
    )
    if np.any(unknown < 0):
        raise PublicLedgerError("negative residual inventory")
    hand_capacity = remaining - known_counts.sum(axis=1)
    if np.any(hand_capacity < 0):
        raise PublicLedgerError("negative unknown hand capacity")
    conservation = int(unknown.sum()) - int(hand_capacity.sum()) - escrow_unknown
    if conservation != 0:
        raise PublicLedgerError(
            "public ledger does not conserve: sum(u) != H + unknown_escrow"
        )

    features = np.concatenate([
        remaining.astype(np.int64),
        known_counts.reshape(-1).astype(np.int64),
        unknown.astype(np.int64),
        np.asarray([escrow_unknown], dtype=np.int64),
    ])
    if features.shape != (PUBLIC_FEATURE_WIDTH,):
        raise PublicLedgerError("public ledger feature width is inconsistent")
    return {
        "schema": PHASE5_PUBLIC_LEDGER_SCHEMA,
        "phase": int(phase),
        "remaining": remaining,
        "own_counts": own_counts,
        "played_counts": played_counts,
        "known_counts": known_counts,
        "escrow_known_counts": escrow_known_counts,
        "hand_capacity": hand_capacity,
        "unknown_counts": unknown,
        "escrow_total": int(escrow_total),
        "escrow_unknown": int(escrow_unknown),
        "escrow_known_cards": list(escrow_known_cards),
        "played_total": len(played_cards),
        "known_total": known_total,
        "known_owner": {
            int(card): int(owner) for card, owner in sorted(ledger.known_owner.items())
        },
        "ledger_conserved": True,
        "features": features,
    }


def public_ledger_features(
    obs_channels: np.ndarray,
    *,
    actor: int,
    ledger: PublicLedger,
) -> np.ndarray:
    """The v2 ``[64]`` public-ledger feature vector for one macro start."""
    return public_ledger_snapshot(
        obs_channels, actor=actor, ledger=ledger,
    )["features"]


def validate_public_features(features: np.ndarray) -> np.ndarray:
    """Fail closed on anything that is not a v2 64-wide public ledger.

    Legacy 18-wide ``[remaining | unseen]`` rows are *not* reinterpreted: they
    cannot express current ownership, so silently accepting them would rebuild
    the original bug.
    """
    array = np.asarray(features)
    if array.shape[-1] != PUBLIC_FEATURE_WIDTH:
        raise ValueError(
            f"public ledger features must have width {PUBLIC_FEATURE_WIDTH}; "
            f"got {array.shape[-1]} (legacy 18-wide public features are not a "
            "v2 ownership ledger)"
        )
    return array


def decode_public_ledger(features: np.ndarray) -> dict[str, np.ndarray]:
    """Split a validated feature array into its four documented slices."""
    array = validate_public_features(features)
    known = array[..., PUBLIC_KNOWN_SLICE].reshape(
        array.shape[:-1] + (NUM_RELATIVE_SEATS, NUM_PRINTED_RANKS)
    )
    return {
        "remaining": array[..., PUBLIC_REMAINING_SLICE],
        "known": known,
        "unknown": array[..., PUBLIC_UNKNOWN_SLICE],
        "escrow_unknown": array[..., PUBLIC_ESCROW_UNKNOWN_INDEX],
    }


def public_relative_remaining(obs_channels: np.ndarray, *, actor: int) -> np.ndarray:
    """Published remaining hand sizes of next/partner/previous relative to actor."""
    channels = np.asarray(obs_channels)
    if channels.ndim != 2 or channels.shape[0] != 4:
        raise ValueError("obs_channels must be [4, 512]")
    row = channels[int(actor)]
    absolute = np.asarray(row[REMAINING_COUNTS], dtype=np.int64)
    return np.asarray(
        [absolute[(int(actor) + offset) % 4] for offset in RELATIVE_SEAT_OFFSETS],
        dtype=np.int64,
    )


def public_unseen_by_rank(
    obs_channels: np.ndarray,
    *,
    actor: int,
    ledger: PublicLedger,
) -> np.ndarray:
    """Residual unknown inventory ``u_r`` of the v2 public ledger."""
    return public_ledger_snapshot(
        obs_channels, actor=actor, ledger=ledger,
    )["unknown_counts"]


def public_balance_features(
    obs_channels: np.ndarray,
    *,
    actor: int,
    ledger: PublicLedger,
) -> np.ndarray:
    """Full v2 public-ledger features (64 entries)."""
    return public_ledger_features(obs_channels, actor=actor, ledger=ledger)


def public_remaining_totals(
    obs_channels: np.ndarray,
    *,
    actor: int,
    ledger: PublicLedger,
) -> np.ndarray:
    """Residual unknown inventory (kept name, corrected v2 semantics)."""
    return public_unseen_by_rank(obs_channels, actor=actor, ledger=ledger)


def public_history_bits(public_cards: Sequence[int]) -> np.ndarray:
    """``[108]`` uint8 bits: public physical cards received from deal start."""
    bits = np.zeros(TOTAL_PHYSICAL_CARDS, dtype=np.uint8)
    for card in public_cards:
        value = int(card)
        if not 0 <= value < TOTAL_PHYSICAL_CARDS:
            raise ValueError(f"card id {value} is outside 0..107")
        bits[value] = 1
    return bits


def public_history_counts(public_cards: Sequence[int]) -> np.ndarray:
    """``[15]`` int64 public physical-card counts by printed rank."""
    counts = np.zeros(NUM_PRINTED_RANKS, dtype=np.int64)
    for card in public_cards:
        value = int(card)
        if not 0 <= value < TOTAL_PHYSICAL_CARDS:
            raise ValueError(f"card id {value} is outside 0..107")
        counts[printed_rank_index(value)] += 1
    return counts


def visible_and_played_cards(
    obs_channels: np.ndarray,
    *,
    actor: int,
    public_cards: Sequence[int] = (),
) -> tuple[set[int], set[int]]:
    """Split the acting seat's public card knowledge into (own hand, exposed)."""
    row = np.asarray(obs_channels)[int(actor)]
    own = {card_id for card_id in range(TOTAL_PHYSICAL_CARDS) if row[card_id]}
    exposed = {int(card) for card in public_cards}
    for card_id in range(TOTAL_PHYSICAL_CARDS):
        if row[108 + card_id] or row[216 + card_id]:
            exposed.add(card_id)
    return own, exposed


def conservation_audit(
    *,
    hands: Sequence[Sequence[int]],
    played: Sequence[int],
    escrow: Sequence[int],
    actor: int,
    finished_seats: Sequence[int] = (),
) -> dict[str, Any]:
    """Physical-card conservation with no double counting of public transfers.

    ``hands`` are the *current owners* (post-transfer), ``played`` the committed
    non-exchange plays and ``escrow`` the cards still in the public tribute
    escrow.  Every one of the 108 physical cards must appear exactly once.
    ``finished_seats`` is the engine's ranking order; a finished player's hand
    must be empty (the old tautological check never looked at that set).
    """
    if len(hands) != 4:
        raise ValueError("conservation audit requires four hands")
    seen: dict[int, int] = {}
    for hand in hands:
        for card_id in hand:
            seen[int(card_id)] = seen.get(int(card_id), 0) + 1
    for card_id in played:
        seen[int(card_id)] = seen.get(int(card_id), 0) + 1
    for card_id in escrow:
        seen[int(card_id)] = seen.get(int(card_id), 0) + 1
    duplicated = sorted(card for card, count in seen.items() if count > 1)
    missing = [card for card in range(TOTAL_PHYSICAL_CARDS) if card not in seen]
    finished = {int(seat) for seat in finished_seats}
    if any(not 0 <= seat < 4 for seat in finished):
        raise ValueError("finished seat is outside 0..3")
    finished_zero = all(len(hands[seat]) == 0 for seat in finished)
    return {
        "physical_cards_seen": len(seen),
        "duplicated_cards": duplicated,
        "missing_cards": missing,
        "conserved": not duplicated and not missing,
        "finished_players_zero": finished_zero,
        "finished_seats": sorted(finished),
        "actor": int(actor),
    }


def _rotate_hands(hands: Sequence[Sequence[int]], k: int) -> tuple[tuple[int, ...], ...]:
    """New state after rotating old seat ``s`` to ``(s + k) % 4``."""
    return tuple(
        tuple(sorted(int(card) for card in hands[(seat - k) % 4]))
        for seat in range(4)
    )


def _swap_hands(hands: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sorted((int(card) + CARDS_PER_DECK) % TOTAL_PHYSICAL_CARDS for card in hand))
        for hand in hands
    )


def canonical_physical_deal(hands: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Canonical representative of a physical deal modulo the symmetry group.

    The group is the four seat rotations and the two deck swaps.  The
    lexicographically smallest representation is used, so a rotation or a
    copy-ID augmentation of one physical deal hashes identically.  Context is
    not part of the deal identity.
    """
    if len(hands) != 4:
        raise ValueError("a deal requires four hands")
    cards = [int(card) for hand in hands for card in hand]
    if len(cards) != TOTAL_PHYSICAL_CARDS:
        raise ValueError("a canonical physical deal must contain all 108 cards")
    if len(set(cards)) != len(cards) or any(
        not 0 <= card < TOTAL_PHYSICAL_CARDS for card in cards
    ):
        raise ValueError("a canonical physical deal must be a permutation of 0..107")
    candidates = []
    for swap in (False, True):
        base = _swap_hands(hands) if swap else tuple(
            tuple(sorted(int(card) for card in hand)) for hand in hands
        )
        for rotation in range(4):
            candidates.append(_rotate_hands(base, rotation))
    return min(candidates)


def canonical_deal_identity(
    hands: Sequence[Sequence[int]],
    *,
    level_rank: int | None = None,
    leader_seat: int | None = None,
    previous_result: Any = None,
) -> str:
    """Group id: a hash of the canonical *physical deal* only.

    ``level_rank`` / ``leader_seat`` / ``previous_result`` are accepted for call
    compatibility but deliberately excluded: changing the context or naming a
    replay differently must not move the same physical deal across splits.
    """
    del level_rank, leader_seat, previous_result
    payload = {
        "schema": "guandan-marvel-phase5-deal-identity-v1",
        "hands": [list(hand) for hand in canonical_physical_deal(hands)],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_context_identity(
    *,
    level_rank: int,
    leader_seat: int,
    previous_result: Any,
) -> str:
    """Hash of the context, kept separate from the deal identity."""
    payload = {
        "schema": "guandan-marvel-phase5-context-identity-v1",
        "level_rank": int(level_rank),
        "leader_seat": int(leader_seat),
        "previous_result": previous_result,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def group_split(
    group_ids: Sequence[str],
    *,
    seed: int,
    fractions: Mapping[str, float],
) -> dict[str, str]:
    """Deterministic, seeded, deal-level train/validation/test assignment.

    The unit is the canonical deal id, so every decision, rotation, copy-swap
    augmentation and replay of one deal stays in one split.  A deal id that
    appears twice yields the same split; a context/name change cannot move it
    because only the physical deal identity is hashed.
    """
    total = sum(float(value) for value in fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    unique = sorted(set(str(value) for value in group_ids))
    if not unique:
        raise ValueError("group_split requires at least one deal")
    order = sorted(
        unique,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    n = len(order)
    n_train = int(round(n * float(fractions["train"])))
    n_validation = int(round(n * float(fractions["validation"])))
    # Guarantee at least one group in every split when the budget allows it.
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_validation = max(1, min(n_validation, n - n_train - 1))
    assignment: dict[str, str] = {}
    for index, group in enumerate(order):
        if index < n_train:
            assignment[group] = "train"
        elif index < n_train + n_validation:
            assignment[group] = "validation"
        else:
            assignment[group] = "test"
    return assignment


def split_hash(assignment: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(assignment.items())), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_records_json(records: Sequence[Mapping[str, Any]]) -> str:
    """The one canonical JSON encoding of dataset records.

    Nested mappings (``ledger.known_owner``) are keyed by *physical card id*
    (an ``int``) in memory, while JSON only has string keys.  Hashing the two
    forms with ``json.dumps(sort_keys=True)`` therefore disagreed: integers sort
    numerically (``2`` before ``10``) and strings lexicographically (``"10"``
    before ``"2"``), so a delivered JSONL file could not reproduce the hash of
    the in-memory records.

    The canonical form is defined explicitly:

    * mapping keys are converted to strings (``int`` -> ``str``) and the
      converted keys are sorted lexicographically; a conversion collision
      (both ``1`` and ``"1"`` present) is a hard error, never a silent overwrite;
    * mapping key types other than ``int``/``str`` are rejected;
    * list/tuple order is preserved exactly (sample, feature and label order
      never changes);
    * integers/strings/bools/``None`` are kept verbatim;
    * floats must be finite (``NaN``/``Inf`` are rejected, since JSON cannot
      round-trip them faithfully);
    * any other Python type is rejected instead of being stringified.
    """
    return json.dumps(
        canonical_records_value(records),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=True,
    )


def canonical_records_value(value: Any) -> Any:
    """Recursively normalise ``value`` to the canonical JSON value tree."""
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (int, str)):
                raise ValueError(
                    f"dataset record mapping key must be int or str, got {type(key).__name__}"
                )
            text = str(int(key)) if isinstance(key, int) else str(key)
            if text in converted:
                raise ValueError(
                    f"dataset record mapping key collision after int->str conversion: {text!r}"
                )
            converted[text] = canonical_records_value(item)
        return {
            key: converted[key]
            for key in sorted(converted)
        }
    if isinstance(value, (list, tuple)):
        return [canonical_records_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("dataset record contains a non-finite JSON number")
        return number
    raise TypeError(f"dataset record contains a non-JSON value of type {type(value).__name__}")


def dataset_hash(records: Sequence[Mapping[str, Any]]) -> str:
    payload = canonical_records_json(records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def leakage_probe_identity(
    obs_tokens: np.ndarray,
    obs_channels: np.ndarray,
) -> str:
    """Hash of exactly the model-visible bytes of one sample.

    The hidden-hand counterfactual test mutates hidden hands while keeping the
    visible observation fixed; this hash must then be unchanged.
    """
    tokens = np.ascontiguousarray(np.asarray(obs_tokens, dtype=np.int64))
    channels = np.ascontiguousarray(np.asarray(obs_channels, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(tokens.shape)).encode("ascii"))
    digest.update(tokens.tobytes())
    digest.update(json.dumps(list(channels.shape)).encode("ascii"))
    digest.update(channels.tobytes())
    return digest.hexdigest()


__all__ = [
    "BeliefLabel", "PHASE5_BELIEF_DATASET_SCHEMA", "PHASE5_PUBLIC_LEDGER_SCHEMA",
    "PublicLedger", "PublicLedgerError", "RECORDS_CANONICALIZATION", "build_label",
    "canonical_context_identity", "canonical_deal_identity",
    "canonical_physical_deal", "canonical_records_json", "canonical_records_value",
    "channel_phase", "conservation_audit",
    "dataset_hash", "deck_index", "decode_public_ledger", "group_split",
    "label_class_mask", "leakage_probe_identity", "parse_public_event_row",
    "printed_rank_index", "public_balance_features", "public_event_from_observations",
    "public_history_bits", "public_history_counts", "public_ledger_features",
    "public_ledger_snapshot", "public_relative_remaining",
    "public_remaining_totals", "public_unseen_by_rank", "rank_counts",
    "relative_seat", "split_hash", "validate_public_features",
    "visible_and_played_cards",
]
