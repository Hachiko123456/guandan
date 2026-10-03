"""GD-ENCODING-0.1: an explicit, policy-only projection of A03 state.

The wire grammar and every channel index are specified in docs/encoding_layout.md.
No engine snapshot, opponent cards, or unresolved exchange records leave this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_state import (
    BOS_TOKEN, CARD_TOKEN_BASE, COMMIT_TOKEN, FAMILY_ORDER, FAMILY_TOKEN_BASE,
    FOLLOW_PLAY_TOKEN, LEAD_PLAY_TOKEN, PASS_TOKEN, RETURN_TOKEN, TRIBUTE_TOKEN,
    WILD_TOKEN_BASE, ProtocolError, StepwiseActionState, TokenCodec,
)
from .cards import Card, PRINTED_RANKS, Rank
from .combos import Combination
from .environment_types import Observation, ObservationSpec
from .state import CommittedAction as AtomicCommittedAction, EpisodeTerminatedError, Phase, TEAM_OF

REMAINING_COUNTS_SLICE = slice(8, 12)
FINISHING_RANKS_SLICE = slice(12, 16)
SELF_CARD_IDS_SLICE = slice(101, 209)
PREFIX_TOKENS_SLICE = slice(209, 241)

_PHASES = {
    LEAD_PLAY_TOKEN: "LEAD_PLAY", FOLLOW_PLAY_TOKEN: "FOLLOW_PLAY",
    TRIBUTE_TOKEN: "TRIBUTE", RETURN_TOKEN: "RETURN",
}
_OUTCOMES = {"DOUBLE_DOWN": 1, "HEAD_THIRD": 2, "HEAD_LAST": 3}


@dataclass(frozen=True, slots=True)
class _PolicyView:
    """Only detached public/self primitives; deliberately contains no RoundState."""

    player: int
    phase: int
    level: int
    token_step: int
    committed_step: int
    max_token_steps: int
    leader: int
    winner: int
    passes: int
    hand: tuple[int, ...]
    prefix: tuple[int, ...]
    counts: tuple[int, ...]
    ranks: tuple[int, ...]
    emptied: tuple[int, ...]
    # present, winner, outcome, rank[4], previous cancellation
    previous: tuple[int, ...]
    anti_tribute: bool
    exchanges_public: bool
    winning: tuple[int, ...]
    winning_family: int
    winning_rank: int
    winning_size: int
    history: tuple[tuple[int, ...], ...]
    transfers: tuple[tuple[int, ...], ...]
    play_pass_count: int
    played_card_count: int
    tribute_count: int
    return_count: int


def _play_tokens(declaration: Combination) -> tuple[int, ...]:
    """Lossless public declaration body, independent of A03's action grammar.

    Emit FAMILY, comparison RANK, then physical cards in ID order with each
    represented WILD target and optional natural-use LENGTH(1) immediately
    after its CARD. The enclosing record/section owns the COMMIT delimiter.
    """
    body = [TokenCodec.family(declaration.kind), TokenCodec.rank(declaration.comparison_rank)]
    assignments = declaration.wild_assignments
    natural = set(declaration.natural_wild_ids)
    for card in sorted(declaration.cards, key=lambda item: item.card_id):
        body.append(TokenCodec.card(card.card_id))
        if card.card_id in assignments:
            rank, suit = assignments[card.card_id]
            body.append(TokenCodec.wild(rank, suit))
        if card.card_id in natural:
            body.append(TokenCodec.length(1))
    return tuple(body)


def _action_tokens(action: AtomicCommittedAction) -> tuple[int, ...]:
    """Return only a public action body; never include a framing COMMIT."""
    if action.is_pass:
        return (PASS_TOKEN,)
    if action.kind in ("tribute", "return"):
        if len(action.cards) != 1:
            raise ProtocolError("exchange history action must contain one card")
        return (TokenCodec.card(action.cards[0].card_id),)
    if action.declaration is None:
        raise ProtocolError("public play history requires its declaration")
    return _play_tokens(action.declaration)


def _project(protocol: StepwiseActionState) -> _PolicyView:
    state = protocol.round_state
    if protocol.done or protocol.truncated or state.phase is Phase.TERMINAL:
        raise EpisodeTerminatedError("cannot encode a terminal or truncated protocol")
    phase = (
        TRIBUTE_TOKEN if state.phase is Phase.TRIBUTE else
        RETURN_TOKEN if state.phase is Phase.RETURN else
        LEAD_PLAY_TOKEN if state.current_winning is None else FOLLOW_PLAY_TOKEN
    )
    exchanges_public = state.phase in (Phase.PLAY, Phase.TERMINAL)
    history = []
    play_pass_count = played_card_count = 0
    for action in state.history:
        # Gate BEFORE inspecting cards or declarations. No redacted placeholder,
        # escrow size, hidden-history length, pairing, or selection is emitted.
        if action.kind in ("tribute", "return"):
            if not exchanges_public:
                continue
            tag = TRIBUTE_TOKEN if action.kind == "tribute" else RETURN_TOKEN
            payload = _action_tokens(action)
        elif action.is_pass:
            tag, payload = FOLLOW_PLAY_TOKEN, _action_tokens(action)
            play_pass_count += 1
        else:
            if action.declaration is None:
                raise ProtocolError("public play history requires its declaration")
            tag, payload = LEAD_PLAY_TOKEN, _action_tokens(action)
            play_pass_count += 1
            played_card_count += len(action.cards)
        history.append((BOS_TOKEN, tag, TokenCodec.length(action.player_id), *payload, COMMIT_TOKEN))
    transfers = []
    tribute_count = return_count = 0
    if exchanges_public:
        # These fields are never even inspected during TRIBUTE or RETURN.
        for source, target, card in state.tribute_transfers:
            transfers.append((BOS_TOKEN, TRIBUTE_TOKEN, TokenCodec.length(source), TokenCodec.length(target), TokenCodec.card(card.card_id), COMMIT_TOKEN))
            tribute_count += 1
        for source, target, card in state.return_transfers:
            transfers.append((BOS_TOKEN, RETURN_TOKEN, TokenCodec.length(source), TokenCodec.length(target), TokenCodec.card(card.card_id), COMMIT_TOKEN))
            return_count += 1
    previous = state.previous_result
    previous_fields = (0, -1, 0, 0, 0, 0, 0, 0) if previous is None else (
        1, previous.winner_team, _OUTCOMES[previous.outcome_class],
        *previous.ranking, int(previous.tribute_cancelled),
    )
    winning = state.current_winning
    return _PolicyView(
        player=state.active_seat, phase=phase, level=int(state.level_rank),
        token_step=protocol.token_step, committed_step=protocol.committed_step,
        max_token_steps=protocol.max_token_steps, leader=state.leader_seat,
        winner=-1 if state.current_winner_seat is None else state.current_winner_seat,
        passes=state.consecutive_passes,
        hand=tuple(sorted(card.card_id for card in state.hands[state.active_seat])),
        # A03's public prefix property enumerates all candidates. Read its
        # immutable backing tuple instead, without mutating it, so legal arrays
        # are the sole candidate enumeration in this encoding operation.
        prefix=tuple(protocol._prefix_tokens),
        counts=tuple(len(hand) for hand in state.hands),
        ranks=tuple(state.ranking), emptied=tuple(state.emptied_seats),
        previous=previous_fields, anti_tribute=bool(state.anti_tribute),
        exchanges_public=exchanges_public,
        winning=() if winning is None else _play_tokens(winning),
        winning_family=0 if winning is None else FAMILY_ORDER.index(winning.kind) + 1,
        winning_rank=0 if winning is None else int(winning.comparison_rank),
        winning_size=0 if winning is None else len(winning.cards),
        history=tuple(history), transfers=tuple(transfers),
        play_pass_count=play_pass_count, played_card_count=played_card_count,
        tribute_count=tribute_count, return_count=return_count,
    )


def _encode_tokens(view: _PolicyView, spec: ObservationSpec) -> tuple[np.ndarray, int]:
    tokens = [BOS_TOKEN, view.phase, TokenCodec.rank(view.level), TokenCodec.length(view.player), COMMIT_TOKEN]

    def extend(items: tuple[int, ...] | list[int]) -> None:
        if len(tokens) + len(items) > spec.max_observation_tokens:
            raise ProtocolError("max_observation_tokens capacity overflow")
        tokens.extend(items)

    sections = (
        (tuple(TokenCodec.card(card_id) for card_id in view.hand),),
        (view.prefix,),
        (() if not view.winning else (TokenCodec.length(view.winner), *view.winning),),
        view.history,
        view.transfers,
    )
    for section_id, records in enumerate(sections):
        extend((BOS_TOKEN, TokenCodec.length(section_id)))
        for record in records:
            extend(record)
        extend((COMMIT_TOKEN,))
    if any(not (1 <= token <= 48 or 64 <= token <= 223) for token in tokens):
        raise ProtocolError("observation contains a PAD or reserved token in its payload")
    result = np.zeros(spec.max_observation_tokens, dtype=np.int32)
    result[:len(tokens)] = tokens
    return result, len(tokens)


def _legal_arrays(protocol: StepwiseActionState, view: _PolicyView, spec: ObservationSpec) -> tuple[np.ndarray, ...]:
    if len(view.prefix) + 1 > spec.max_action_tokens:
        raise ProtocolError("max_action_tokens capacity overflow")
    # A smaller diagnostic spec must check ALL live action paths, not filter
    # long actions or mutate the original protocol's capacities. This temporary
    # wrapper is read-only and is never part of a policy value.
    source = protocol
    if protocol.max_action_tokens > spec.max_action_tokens or protocol.max_legal_next_tokens > spec.max_legal_next_tokens:
        source = protocol.clone()
        source.max_action_tokens = min(protocol.max_action_tokens, spec.max_action_tokens)
        source.max_legal_next_tokens = min(protocol.max_legal_next_tokens, spec.max_legal_next_tokens)
        source._prefix_tokens = view.prefix
    raw = source.legal_token_arrays()
    count = int(np.count_nonzero(raw[1]))
    if not count:
        raise ProtocolError("nonterminal prefix has no legal continuation")
    if count > spec.max_legal_next_tokens:
        raise ProtocolError("max_legal_next_tokens capacity overflow")
    result = []
    for array, dtype in zip(raw, (np.int32, np.bool_, np.bool_, np.int32, np.int32), strict=True):
        detached = np.zeros(spec.max_legal_next_tokens, dtype=dtype)
        detached[:count] = array[raw[1]]
        result.append(detached)
    return tuple(result)


def _encode_channels(view: _PolicyView, used: int, legal: tuple[np.ndarray, ...]) -> np.ndarray:
    # Python integers until the final conversion prevent an intermediate
    # float32 rounding from concealing counter overflow.
    channels = [0] * 256
    channels[:8] = [view.player, TEAM_OF[view.player], view.phase, view.level,
                    view.token_step, view.committed_step, view.leader, view.winner]
    channels[REMAINING_COUNTS_SLICE] = view.counts
    channels[FINISHING_RANKS_SLICE] = view.ranks
    channels[16:20] = [int(seat in view.emptied) for seat in range(4)]
    complete = bool(np.any(legal[0][legal[1]] == COMMIT_TOKEN))
    channels[20:29] = [view.passes, len(view.prefix), int(complete), len(view.history),
                       used, len(view.hand), int(bool(view.winning)),
                       int(view.exchanges_public), int(view.anti_tribute)]
    channels[29:37] = view.previous
    channels[37:45] = [int(np.count_nonzero(legal[1])), view.max_token_steps,
                       sum(rank != 0 for rank in view.ranks), view.played_card_count,
                       view.play_pass_count, view.tribute_count, view.return_count,
                       view.winning_size]
    prefix_family = next((token - FAMILY_TOKEN_BASE + 1 for token in view.prefix
                          if FAMILY_TOKEN_BASE <= token < FAMILY_TOKEN_BASE + 10), 0)
    channels[45:48] = [sum(CARD_TOKEN_BASE <= token < CARD_TOKEN_BASE + 108 for token in view.prefix),
                       prefix_family, sum(WILD_TOKEN_BASE <= token < WILD_TOKEN_BASE + 52 for token in view.prefix)]
    channels[48:52] = TEAM_OF
    channels[52 + view.player] = 1
    channels[56 + view.leader] = 1
    if view.winner >= 0:
        channels[60 + view.winner] = 1
    channels[64 + view.phase - LEAD_PLAY_TOKEN] = 1
    channels[68 + PRINTED_RANKS.index(Rank(view.level))] = 1
    for card_id in view.hand:
        card = Card(card_id)
        channels[81 + TokenCodec.rank(card.rank) - 18] += 1
        channels[96 + int(card.suit)] += 1
        channels[101 + card_id] = 1
    if len(view.prefix) > 32:
        raise ProtocolError("prefix exceeds fixed channel layout")
    channels[209:209 + len(view.prefix)] = view.prefix
    if prefix_family:
        channels[241 + prefix_family - 1] = 1
    channels[251:256] = [view.winning_family, view.winning_rank, view.winning_size, int(complete), 1]
    if any(abs(value) > 2**24 for value in channels):
        raise ProtocolError("state channel exceeds exact float32 integer capacity")
    return np.asarray(channels, dtype=np.float32)


def encode_observation(
    protocol: StepwiseActionState, spec: ObservationSpec = ObservationSpec(),
) -> Observation:
    """Encode the current actor without mutation, truncation, or hidden-state output.

    Terminal/truncated protocols raise EpisodeTerminatedError; the environment
    is responsible for returning None instead. All seven arrays own their data.
    """
    if not isinstance(protocol, StepwiseActionState):
        raise TypeError("protocol must be StepwiseActionState")
    if not isinstance(spec, ObservationSpec):
        raise TypeError("spec must be ObservationSpec")
    view = _project(protocol)
    tokens, used = _encode_tokens(view, spec)
    legal = _legal_arrays(protocol, view, spec)
    channels = _encode_channels(view, used, legal)
    return Observation(
        player_id=view.player, team_id=TEAM_OF[view.player], phase=_PHASES[view.phase],
        token_step=view.token_step, committed_step=view.committed_step,
        observation_tokens=tokens, state_channels=channels,
        legal_next_tokens=legal[0], legal_next_mask=legal[1],
        legal_next_is_commit=legal[2], legal_next_kinds=legal[3], legal_next_values=legal[4],
        private_hand_card_ids=None,
    )


__all__ = ["encode_observation", "REMAINING_COUNTS_SLICE", "FINISHING_RANKS_SLICE",
           "SELF_CARD_IDS_SLICE", "PREFIX_TOKENS_SLICE"]
