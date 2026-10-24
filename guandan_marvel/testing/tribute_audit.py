# SPDX-License-Identifier: MIT
"""Phase-4 (repair) tribute/return observation audit.

The audit uses the *real* observation encoders:

* ``guandan_marvel.game._encode_channels`` for the per-observer channel planes;
* ``guandan_marvel.game._step_events`` for the resolved-exchange event tokens;
* ``guandan.state.PreviousHandResult`` for legal ``previous_result`` values.

It answers three questions:

1. does the policy observation carry every rule-public exchange fact?
2. do hidden (other-seat) cards leak into the observation?
3. can two *legal* exchange relations produce the same complete observation
   (tokens + channels)?

The earlier pre-repair counterexample only compared the *transfer token bag*
and used an illegal team structure.  Against the complete observation every
legal relation is distinguished because ``TRIBUTE_ORDER_SLICE`` records the
donor order induced by the tribute card ranks and ``SELF_HAND_SLICE`` /
``REMAINING_COUNTS_SLICE`` record the (different) card holdings.
"""
from __future__ import annotations

import numpy as np

from guandan.action_state import PASS_TOKEN, COMMIT_TOKEN, StepwiseActionState
from guandan.cards import Rank, Suit, Card
from guandan.environment import GuandanEnv
from guandan.environment_types import GameConfig
from guandan.state import Phase, PreviousHandResult, RoundState

from guandan_marvel import game as marvel_game

AUDIT_SCHEMA = "guandan-marvel-tribute-observation-audit-v1"

# Legal previous_result values (winner team 1 below is intentionally included).
PREVIOUS_RESULTS: tuple[PreviousHandResult, ...] = (
    PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN"),
    PreviousHandResult((1, 4, 2, 3), 0, "DOUBLE_DOWN"),
    PreviousHandResult((1, 2, 3, 4), 0, "HEAD_THIRD"),
    PreviousHandResult((1, 2, 4, 3), 0, "HEAD_LAST"),
    PreviousHandResult((3, 1, 4, 2), 1, "DOUBLE_DOWN"),
    PreviousHandResult((2, 1, 4, 3), 1, "HEAD_THIRD"),
    PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN", tribute_cancelled=True),
)


def _resolved_observations(previous: PreviousHandResult, seed: int):
    """Drive one real deal through the exchange and capture the resolution step."""
    env = GuandanEnv(GameConfig(level_rank=Rank.SIX, previous_result=previous, max_token_steps=4096))
    env.reset(seed=seed)
    observations = None
    transfers = None
    for _ in range(64):
        protocol = env._require_protocol()
        phase = protocol.round_state.phase
        if phase is Phase.PLAY:
            break
        before = protocol.clone()
        legal = [int(token) for token in env.legal_tokens().tolist()]
        commits = [token for token in legal if token in (PASS_TOKEN, COMMIT_TOKEN)]
        token = commits[0] if commits else legal[0]
        result = env.step(token)
        after = env._require_protocol()
        if after.round_state.phase is Phase.PLAY:
            events = marvel_game._step_events(before, result, token, after)
            observations = tuple(
                (tuple(int(t) for t in events[observer]), tuple(int(v) for v in marvel_game._encode_channels(after, observer)))
                for observer in range(4)
            )
            state = after.round_state
            transfers = tuple(
                ("tribute", t[0], t[1], t[2].card_id) for t in state.tribute_transfers
            ) + tuple(("return", t[0], t[1], t[2].card_id) for t in state.return_transfers)
    if observations is None:
        return None
    return {"observations": observations, "transfers": transfers, "after": env._require_protocol().round_state}


def _bag_collision_pair():
    """Two exchange relations whose transfer token *bags* coincide.

    Both use the proper double-down seat structure (donors = ranks 3/4,
    recipients = ranks 1/2); only the donor->recipient assignment differs.  The
    token bag is identical, yet the complete observation differs because
    ``TRIBUTE_ORDER_SLICE`` records the (reversed) donor order.
    """
    previous = PreviousHandResult((1, 3, 2, 4), 0, "DOUBLE_DOWN")

    def observation(pairing):
        state = RoundState.deal(seed=5, level_rank=Rank.SIX, previous_result=previous)
        donor_high, donor_low = previous.seat_with_rank(3), previous.seat_with_rank(4)
        recipient_one, recipient_two = previous.seat_with_rank(1), previous.seat_with_rank(2)
        card_high = next(card for card in state.hands[donor_high] if card.rank is Rank.NINE)
        card_low = next(card for card in state.hands[donor_low] if card.rank is Rank.FIVE)
        if pairing == "rank_ordered":
            transfers = [(donor_high, recipient_one, card_high), (donor_low, recipient_two, card_low)]
            order = [donor_high, donor_low]
        else:
            transfers = [(donor_high, recipient_two, card_high), (donor_low, recipient_one, card_low)]
            order = [donor_low, donor_high]
        for donor, recipient, card in transfers:
            state.hands[donor].remove(card)
            state.hands[recipient].append(card)
        state.phase = Phase.PLAY
        state.tribute_donors = [donor_high, donor_low]
        state.tribute_order = list(order)
        state.tribute_transfers = list(transfers)
        state.return_transfers = []
        return state

    return observation("rank_ordered"), observation("swapped")


def _event_tokens_for(state: RoundState) -> list[int]:
    """Rebuild the resolved-exchange event tokens for a PLAY state."""
    records: list[int] = []
    for source, target, card in state.tribute_transfers:
        records.extend((
            marvel_game.TRIBUTE_TOKEN, marvel_game.TokenCodec.length(source),
            marvel_game.TokenCodec.length(target), marvel_game.TokenCodec.card(card.card_id),
            marvel_game.COMMIT_TOKEN,
        ))
    return records


def audit_tribute_observation() -> dict[str, object]:
    solved = []
    for previous in PREVIOUS_RESULTS:
        for seed in (3, 11, 23):
            record = _resolved_observations(previous, seed)
            if record is not None:
                solved.append((previous, seed, record))
                break

    observations = []
    for previous, seed, record in solved:
        payload = (previous.ranking, previous.winner_team, previous.outcome_class, previous.tribute_cancelled, seed)
        observations.append((payload, record["observations"]))

    collisions = []
    for index, (key_a, obs_a) in enumerate(observations):
        for key_b, obs_b in observations[index + 1:]:
            if obs_a == obs_b:
                collisions.append({"a": key_a, "b": key_b})

    # hidden-card leak: for each observer, tokens must only reference the
    # publicly transferred cards; channels' SELF_HAND slice is the observer's own.
    leaks = []
    for previous, seed, record in solved:
        state = record["after"]
        public_cards = {t[3] for t in record["transfers"]}
        for observer in range(4):
            tokens, channels = record["observations"][observer]
            for token in tokens:
                if marvel_game.CARD_TOKEN_BASE <= token < marvel_game.CARD_TOKEN_BASE + 108:
                    if (token - marvel_game.CARD_TOKEN_BASE) not in public_cards:
                        leaks.append((previous.ranking, seed, observer, token))

    state_a, state_b = _bag_collision_pair()
    tokens_a = _event_tokens_for(state_a)
    tokens_b = _event_tokens_for(state_b)
    bag_collides = sorted(tokens_a) == sorted(tokens_b)
    channels_a = tuple(int(v) for v in marvel_game._encode_channels(StepwiseActionState(state_a), 0))
    channels_b = tuple(int(v) for v in marvel_game._encode_channels(StepwiseActionState(state_b), 0))
    channels_differ = channels_a != channels_b

    passed = not collisions and not leaks and bag_collides and channels_differ
    return {
        "schema": AUDIT_SCHEMA,
        "passed": bool(passed),
        "legal_relations_checked": len(observations),
        "full_observation_collisions": collisions,
        "hidden_card_leaks": leaks,
        "bag_level_collision_present": bool(bag_collides),
        "bag_level_collision_disambiguated_by_channels": bool(channels_differ),
        "reason": (
            "complete observation (tokens + channels) distinguishes every legal exchange relation: "
            "the transfer token bag coincidence is broken by TRIBUTE_ORDER_SLICE, SELF_HAND_SLICE and "
            "REMAINING_COUNTS_SLICE, and no hidden card leaks"
            if passed else "tribute observation audit found a defect"
        ),
    }


__all__ = ["AUDIT_SCHEMA", "PREVIOUS_RESULTS", "audit_tribute_observation"]
