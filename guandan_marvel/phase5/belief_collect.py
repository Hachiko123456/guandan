# SPDX-License-Identifier: MIT
"""Belief B0 dataset collection from the real C++ engine.

One sample = one *policy macro start* of one independent canonical deal:

* ``obs_tokens`` / ``obs_channels`` — exactly the acting seat's observation at
  that macro start (the only model input);
* ``public`` — the v2 public-ledger features the B baseline may use: published
  remaining counts, publicly known *currently held* transfer cards (K), the
  residual unknown inventory by rank and the unknown escrow capacity;
* ``public_events`` / ``ledger`` — the visible causal event trajectory and the
  bucket snapshot, so another process can independently replay the ledger;
* ``label`` — the oracle remaining-card counts of the three relative seats
  (separate channel; never an input);
* ``group_id`` — the canonical initial-deal hash (split unit);
* ``audit`` — physical conservation, finished-player zeros and the leakage
  counterfactual identity.

Deals are played in parallel auto-resetting slots (``collection_batch_slots``,
16 by default, measured ~2.3 GB peak on the 4 GB local GPU) so a terminal slot
is re-dealt by the engine and never fed through the policy as a zero
observation.  Wall-clock and deal budgets are explicit; exceeding them returns
``truncated`` instead of silently extending the run.
"""

from __future__ import annotations

import time
import math
import operator
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .belief_data import (
    PublicLedger, build_label, canonical_deal_identity, conservation_audit,
    dataset_hash, leakage_probe_identity, public_event_from_observations,
    public_history_bits, public_history_counts, public_ledger_features,
    public_ledger_snapshot,
)
from .design import (
    PHASE5_BELIEF_DATASET_SCHEMA, PHASE5_PUBLIC_LEDGER_SCHEMA,
    PUBLIC_FEATURE_WIDTH, TOTAL_PHYSICAL_CARDS,
)


class BeliefCollectionError(RuntimeError):
    pass


_PREVIOUS_RESULTS: tuple[Mapping[str, Any] | None, ...] = (
    None,
    {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN"},
    {"ranking": [1, 2, 3, 4], "winner_team": 0, "outcome_class": "HEAD_THIRD"},
    {"ranking": [1, 2, 4, 3], "winner_team": 0, "outcome_class": "HEAD_LAST"},
    {"ranking": [3, 1, 4, 2], "winner_team": 1, "outcome_class": "DOUBLE_DOWN"},
    {"ranking": [2, 1, 4, 3], "winner_team": 1, "outcome_class": "HEAD_THIRD"},
    {"ranking": [2, 1, 3, 4], "winner_team": 1, "outcome_class": "HEAD_LAST"},
    {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN",
     "tribute_cancelled": True},
)


def context_for(index: int) -> dict[str, Any]:
    """Deterministic stratified context for deal ``index``.

    Level ranks cycle over the 13 printed ranks, leader seats over 0..3 and
    tribute modes over the eight legal ``previous_result`` values; the
    combination walks every cell before repeating (``index mod 13*4*8``).
    """
    index = int(index)
    level = 3 + (index % 13)
    leader = (index // 13) % 4
    previous = _PREVIOUS_RESULTS[(index // (13 * 4)) % len(_PREVIOUS_RESULTS)]
    return {
        "level_rank": int(level),
        "leader_seat": int(leader),
        "previous_result": None if previous is None else dict(previous),
        "tribute_mode": "none" if previous is None else (
            "cancelled" if previous.get("tribute_cancelled") else previous["outcome_class"]
        ),
    }


@dataclass(frozen=True, slots=True)
class BeliefSample:
    group_id: str
    deal_index: int
    token_step: int
    actor: int
    phase: str
    level_rank: int
    leader_seat: int
    tribute_mode: str
    tokens: np.ndarray
    channels: np.ndarray
    public: np.ndarray
    public_cards: np.ndarray
    public_events: tuple[dict[str, Any], ...]
    ledger: Mapping[str, Any]
    label_counts: np.ndarray
    label_mask: np.ndarray
    audit: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PHASE5_BELIEF_DATASET_SCHEMA,
            "public_ledger_schema": PHASE5_PUBLIC_LEDGER_SCHEMA,
            "group_id": self.group_id,
            "deal_index": int(self.deal_index),
            "token_step": int(self.token_step),
            "actor": int(self.actor),
            "phase": self.phase,
            "level_rank": int(self.level_rank),
            "leader_seat": int(self.leader_seat),
            "tribute_mode": self.tribute_mode,
            "obs_tokens": self.tokens.astype(np.int64).tolist(),
            "obs_channels": self.channels.astype(np.uint8).tolist(),
            "public": self.public.astype(np.int64).tolist(),
            "public_cards": self.public_cards.astype(np.uint8).tolist(),
            "public_events": [dict(event) for event in self.public_events],
            "ledger": _serialize_ledger(self.ledger),
            "label_counts": self.label_counts.astype(np.int64).tolist(),
            "label_mask": self.label_mask.astype(bool).tolist(),
            "audit": dict(self.audit),
        }


def _serialize_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in ledger.items():
        if isinstance(item, np.ndarray):
            value[key] = item.tolist()
        elif isinstance(item, dict):
            value[key] = {
                (int(name) if isinstance(name, (np.integer, int)) else str(name)): int(entry)
                for name, entry in item.items()
            }
        else:
            value[key] = item
    return value


def public_features(
    channels: np.ndarray,
    *,
    actor: int,
    ledger: PublicLedger,
) -> np.ndarray:
    """v2 public-ledger baseline features (never reads hidden hands)."""
    return public_ledger_features(channels, actor=actor, ledger=ledger)


def _tensor_batch(step_view, batch_size: int, device: torch.device):
    from guandan_marvel.model.tensordict import TensorDict

    return TensorDict(
        {
            "num_steps": torch.ones(batch_size, dtype=torch.long),
            "player_ids": torch.as_tensor(step_view.player_ids)[:, None],
            "obs_tokens": torch.as_tensor(step_view.observations.tokens)[:, None],
            "obs_channels": torch.as_tensor(step_view.observations.channels)[:, None],
            "avail_tokens": torch.as_tensor(step_view.available_actions.tokens)[:, None],
            "avail_weights": torch.as_tensor(step_view.available_actions.weights)[:, None],
        },
        batch_size=[batch_size], device=device,
    )


def deterministic_deal(seed: int) -> list[list[int]]:
    """A reproducible 4x27 physical deal generated outside the engine RNG.

    The deal is a pure function of ``seed`` (NumPy PCG64), so the canonical
    deal identity, the group split and a replay never depend on engine state.
    """
    deck = np.arange(TOTAL_PHYSICAL_CARDS, dtype=np.int64)
    np.random.default_rng(int(seed)).shuffle(deck)
    return [deck[seat * 27:(seat + 1) * 27].tolist() for seat in range(4)]


def collect_belief_dataset(
    *,
    policy,
    device: torch.device,
    deals: int,
    base_seed: int,
    max_macro_starts_per_deal: int,
    max_wall_seconds: float,
    start_index: int = 0,
    batch_slots: int = 16,
) -> tuple[list[BeliefSample], list[dict[str, Any]], dict[str, Any]]:
    """Play real C++ deals and record belief samples at policy macro starts.

    The engine runs ``auto_reset=True`` in ``batch_slots`` parallel slots with a
    context schedule installed *before* the first reset.  Each slot therefore
    consumes the next stratified context whenever it re-deals, so the level /
    leader / tribute distribution is exactly the manifest's schedule.  A deal's
    samples are buffered until the deal terminates; a deal truncated by the
    wall-clock budget is reported as unfinished and its partial samples are kept
    only in the returned sample list (never in a manifest claiming completion).
    The canonical deal identity is computed from the engine's own initial hands
    (``debug_state``), never from a seed guess.
    """
    from nanopokerlib import guandan as cpp

    from guandan_marvel.model import KVCache

    def integer(name: str, value: int, *, minimum: int) -> int:
        try:
            result = operator.index(value)
        except TypeError as error:
            raise ValueError(f"{name} must be an integer >= {minimum}") from error
        if isinstance(value, bool) or result < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return result

    # Fail before touching a policy, allocating buffers or creating an engine.
    total = integer("deals", deals, minimum=1)
    slots = min(integer("batch_slots", batch_slots, minimum=1), total)
    next_index = integer("start_index", start_index, minimum=0)
    max_macro_starts_per_deal = integer(
        "max_macro_starts_per_deal", max_macro_starts_per_deal, minimum=1,
    )
    if not math.isfinite(float(max_wall_seconds)) or float(max_wall_seconds) <= 0:
        raise ValueError("max_wall_seconds must be finite and positive")
    decoder = policy.decoder.config
    head_dim = decoder.hidden_size // decoder.num_attention_heads
    kv_heads = decoder.num_key_value_heads or decoder.num_attention_heads

    samples: list[BeliefSample] = []
    deals_meta: list[dict[str, Any]] = []
    started = time.perf_counter()
    truncated = False
    finished_deals = 0
    recorded_deals = 0
    # The whole stratified schedule is installed up front: the engine consumes
    # one context per deal in order (slot 0..n-1 at reset, then one per
    # auto-reset). ``issued`` below is an *absolute* deal index, including
    # start_index, and mirrors every engine reset, even an unrecorded slot.
    # ``slots`` extra entries absorb the auto-reset that follows the final
    # planned deal (the engine re-deals even when no further sample is wanted).
    schedule = [
        context_for(next_index + index) for index in range(total + slots)
    ]
    env = cpp.EnvBatch(slots, auto_reset=True)
    env.configure_context_schedule(schedule, offset=0)
    step = env.reset(int(base_seed))
    kv = KVCache(
        policy.decoder_n_layers, slots * 4, max_ctx_len=int(policy.max_game_horizon),
        num_heads=kv_heads, head_dim=head_dim, value_heads=kv_heads,
        value_head_dim=head_dim, dtype=torch.float32, device=device,
    )
    kv.reset(torch.ones(slots * 4, dtype=torch.bool, device=device))
    active: dict[int, dict[str, Any]] = {}
    buffered: dict[int, list[BeliefSample]] = {slot: [] for slot in range(slots)}
    for slot in range(slots):
        active[slot] = _start_deal(
            env.debug_state(slot), next_index + slot, context_for(next_index + slot),
        )
    issued = next_index + slots
    while recorded_deals < total:
        if time.perf_counter() - started > float(max_wall_seconds):
            truncated = True
            break
        for slot in list(active):
            if active[slot].get("out_of_range"):
                continue
            debug = env.debug_state(slot)
            if not debug["prefix_tokens"]:
                record = _record_sample(
                    slot, step, debug, active[slot], max_macro_starts_per_deal,
                    ledger=active[slot]["ledger"],
                )
                if record is not None:
                    buffered[slot].append(record)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            output = policy(rollout=_tensor_batch(step, slots, device), use_cache=True,
                            past_key_values=kv)
        legal = np.asarray(step.available_actions.weights) > 0
        probabilities = output.avail_probs.squeeze(1).float()
        if not bool(torch.isfinite(probabilities).all()):
            raise BeliefCollectionError("policy produced a non-finite probability")
        masked = torch.where(
            torch.as_tensor(legal, device=device), probabilities,
            torch.zeros_like(probabilities),
        )
        sums = masked.sum(-1)
        for slot in range(slots):
            if float(sums[slot].item()) <= 0:
                raise BeliefCollectionError(
                    f"no legal candidate carries policy probability in slot {slot}"
                )
        # A slot that just terminated still sits in ``active`` for this step (the
        # terminal transition is observed below), so every sampled row belongs to
        # a live slot and is a legal row by construction.
        actions = torch.multinomial(masked, 1).squeeze(1).cpu().numpy()
        rows = np.asarray(step.available_actions.tokens)[
            np.arange(slots), actions
        ].astype(np.int64)
        next_step = env.step(rows)
        terminal = np.asarray(next_step.is_terminal, dtype=bool)
        # Accumulate the causal public ledger of every still-live deal from the
        # event just produced.  Private prefix/exchange-selection events publish
        # no card (public_event_from_observations returns None); a public event
        # is broadcast identically to the four observer rows and is applied once.
        for slot in range(slots):
            meta = active[slot]
            if meta.get("out_of_range") or terminal[slot]:
                continue
            event = public_event_from_observations(next_step.observations.tokens[slot])
            if event is not None:
                meta["ledger"].apply_event(event)
        # The engine auto-resets terminating slots in ascending slot order and
        # consumes one scheduled context per reset; mirror that order exactly so
        # the recorded deal index is the engine's own context index.
        for slot in range(slots):
            meta = active[slot]
            if not terminal[slot]:
                if not meta.get("out_of_range"):
                    meta["token_steps"] = int(meta["token_steps"]) + 1
                continue
            if not meta.get("out_of_range"):
                meta["token_steps"] = int(meta["token_steps"]) + 1
                meta["finished"] = True
                meta["samples"] = len(buffered[slot])
                meta["reward"] = [float(value) for value in next_step.rewards[slot]]
                info = dict(next_step.infos[slot])
                meta["outcome_class"] = info.get("outcome_class")
                meta["winner_team"] = info.get("winner_team")
                meta["ranking"] = list(info.get("ranking", ()))
                samples.extend(buffered[slot])
                buffered[slot] = []
                meta["public_cards"] = sorted(
                    int(card) for card in meta["ledger"].public_cards()
                )
                meta["public_events"] = meta["ledger"].to_records()
                meta["public_ledger_schema"] = PHASE5_PUBLIC_LEDGER_SCHEMA
                meta.pop("ledger", None)
                deals_meta.append(meta)
                finished_deals += 1
                recorded_deals += 1
            # Even exhausted slots keep advancing the native batch, and each
            # reset consumes one scheduled context. Never inspect or publish a
            # padding deal: the finite context schedule can wrap while another
            # requested deal is still finishing.
            if issued >= next_index + total:
                active[slot] = {"out_of_range": True}
            else:
                active[slot] = _start_deal(env.debug_state(slot), issued, context_for(issued))
            issued += 1
        if terminal.any():
            kv.reset(torch.as_tensor(terminal).repeat_interleave(4).to(device))
        step = next_step
    for slot, pending in buffered.items():
        for record in pending:
            samples.append(record)
    for slot, meta in active.items():
        if meta.get("out_of_range"):
            continue
        meta["finished"] = False
        meta["samples"] = len(buffered[slot])
        meta["public_cards"] = sorted(
            int(card) for card in meta["ledger"].public_cards()
        )
        meta["public_events"] = meta["ledger"].to_records()
        meta["public_ledger_schema"] = PHASE5_PUBLIC_LEDGER_SCHEMA
        meta.pop("ledger", None)
        deals_meta.append(meta)
    del env, kv
    return samples, deals_meta, {
        "environment_seconds": time.perf_counter() - started,
        "finished_deals": int(finished_deals),
        "requested_deals": int(total),
        "truncated": bool(truncated),
        "batch_slots": int(slots),
    }


def _start_deal(debug: Mapping[str, Any], index: int, context: Mapping[str, Any]) -> dict[str, Any]:
    hands = [list(hand) for hand in debug["state"]["hands"]]
    if int(debug["config"]["level_rank"]) != int(context["level_rank"]):
        raise BeliefCollectionError("engine context installation failed")
    return {
        "deal_index": int(index),
        "group_id": canonical_deal_identity(
            hands, level_rank=int(context["level_rank"]),
            leader_seat=int(context["leader_seat"]),
            previous_result=context["previous_result"],
        ),
        "context": dict(context),
        "samples": 0,
        "token_steps": 0,
        "ledger": PublicLedger(),
        "public_ledger_schema": PHASE5_PUBLIC_LEDGER_SCHEMA,
    }


def _audit_ledger(
    *,
    ledger: PublicLedger,
    snapshot: Mapping[str, Any],
    hands: Sequence[Sequence[int]],
    played: Sequence[int],
    escrow: Sequence[int],
    actor: int,
) -> dict[str, Any]:
    """Independent engine-state audit of the visible ledger (never an input).

    The hidden hands, played pile and escrow are read only here, after the
    public features and labels have been produced.  Any disagreement raises
    instead of publishing a sample whose public claim is false.
    """
    from .belief_data import rank_counts

    failures: list[str] = []
    absolute_hands = [set(int(card) for card in hand) for hand in hands]
    if set(ledger.played) != {int(card) for card in played}:
        failures.append("played_set")
    for card, owner in ledger.known_owner.items():
        if int(card) not in absolute_hands[int(owner)]:
            failures.append(f"known_owner_card:{int(card)}")
    if int(snapshot["escrow_total"]) != len(escrow):
        failures.append("escrow_total")
    for card in snapshot.get("escrow_known_cards", ()):
        if int(card) not in {int(value) for value in escrow}:
            failures.append(f"escrow_known_card:{int(card)}")
    own_expected = rank_counts(hands[int(actor)])
    if not np.array_equal(own_expected, snapshot["own_counts"]):
        failures.append("own_counts")
    played_expected = rank_counts(played)
    if not np.array_equal(played_expected, snapshot["played_counts"]):
        failures.append("played_counts")
    lower_bound = True
    for index, offset in enumerate((1, 2, 3)):
        seat = (int(actor) + offset) % 4
        actual = rank_counts(hands[seat])
        if np.any(snapshot["known_counts"][index] > actual):
            lower_bound = False
            failures.append(f"known_lower_bound:{seat}")
    if not bool(snapshot.get("ledger_conserved")):
        failures.append("ledger_conserved")
    if failures:
        raise BeliefCollectionError(
            "public ledger disagrees with the engine state: " + ", ".join(failures)
        )
    return {
        "ledger_owner_matches_engine": True,
        "known_owner_lower_bound": bool(lower_bound),
        "played_set_matches_engine": True,
        "escrow_total_matches_engine": True,
        "ledger_audit_failures": [],
    }


def _record_sample(
    slot: int,
    step,
    debug: Mapping[str, Any],
    meta: Mapping[str, Any],
    max_macro_starts_per_deal: int,
    *,
    ledger: PublicLedger,
) -> BeliefSample | None:
    if int(meta["samples"]) >= int(max_macro_starts_per_deal):
        return None
    state = debug["state"]
    actor = int(state["active_seat"])
    hands = [list(hand) for hand in state["hands"]]
    label = build_label(hands, actor)
    played = [
        int(card)
        for action in state["history"]
        if action["kind"] not in ("pass", "tribute", "return")
        for card in action["card_ids"]
    ]
    escrow = [int(card) for card in state["tribute_escrow"]]
    audit = conservation_audit(
        hands=hands, played=played, escrow=escrow, actor=actor,
        finished_seats=state.get("finished_ranks", ()),
    )
    channels = np.asarray(step.observations.channels[slot], dtype=np.uint8)
    tokens = np.asarray(step.observations.tokens[slot], dtype=np.int64)
    snapshot = public_ledger_snapshot(channels, actor=actor, ledger=ledger)
    features = snapshot["features"]
    if features.shape != (PUBLIC_FEATURE_WIDTH,):
        raise BeliefCollectionError("public ledger features have the wrong width")
    public_cards_sorted = tuple(sorted(int(card) for card in ledger.public_cards()))
    ledger_audit = _audit_ledger(
        ledger=ledger, snapshot=snapshot, hands=hands, played=played,
        escrow=escrow, actor=actor,
    )
    public_events = tuple(ledger.to_records())
    meta["samples"] = int(meta["samples"]) + 1
    return BeliefSample(
        group_id=str(meta["group_id"]),
        deal_index=int(meta["deal_index"]),
        token_step=int(debug["token_step"]),
        actor=actor,
        phase=str(state["phase"]),
        level_rank=int(state["level_rank"]),
        leader_seat=int(state["leader_seat"]),
        tribute_mode=str(meta["context"]["tribute_mode"]),
        tokens=tokens.copy(),
        channels=channels.copy(),
        public=features.copy(),
        public_cards=public_history_bits(public_cards_sorted),
        public_events=public_events,
        ledger=snapshot,
        label_counts=label.counts,
        label_mask=label.class_mask,
        audit={
            **audit,
            **ledger_audit,
            "leakage_identity": leakage_probe_identity(tokens, channels),
            "public_ledger_schema": PHASE5_PUBLIC_LEDGER_SCHEMA,
            "public_feature_width": int(features.shape[0]),
            "public_event_count": len(public_events),
            "public_card_count": len(public_cards_sorted),
            "public_rank_counts": public_history_counts(public_cards_sorted).tolist(),
            "escrow_total": int(snapshot["escrow_total"]),
            "escrow_unknown": int(snapshot["escrow_unknown"]),
            "known_held_total": int(snapshot["known_total"]),
            "unknown_total": int(snapshot["unknown_counts"].sum()),
            "hand_capacity_total": int(snapshot["hand_capacity"].sum()),
            "ledger_conserved": bool(snapshot["ledger_conserved"]),
        },
    )


def dataset_manifest(
    samples: Sequence[BeliefSample],
    deals_meta: Sequence[Mapping[str, Any]],
    *,
    collection: Mapping[str, Any],
    design_hash: str,
    source_hash: str,
    config_hash: str,
    split_assignment: Mapping[str, str],
) -> dict[str, Any]:
    """Auditable dataset manifest (labels never claim unmeasured coverage)."""
    records = [sample.to_record() for sample in samples]
    groups = sorted({sample.group_id for sample in samples})
    levels = sorted({int(sample.level_rank) for sample in samples})
    leaders = sorted({int(sample.leader_seat) for sample in samples})
    modes = sorted({str(sample.tribute_mode) for sample in samples})
    phases = sorted({str(sample.phase) for sample in samples})
    audits = [sample.audit for sample in samples]
    return {
        "schema": PHASE5_BELIEF_DATASET_SCHEMA,
        "design_hash": design_hash,
        "source_hash": source_hash,
        "config_hash": config_hash,
        "collection": dict(collection),
        "sample_count": len(records),
        "deal_count": len(groups),
        "deal_group_ids": groups,
        "coverage": {
            "level_ranks": levels,
            "leader_seats": leaders,
            "tribute_modes": modes,
            "phases": phases,
            "uncovered": {
                "level_ranks": sorted(set(range(3, 16)) - set(levels)),
                "leader_seats": sorted(set(range(4)) - set(leaders)),
                "tribute_modes": sorted(
                    {"none", "DOUBLE_DOWN", "HEAD_THIRD", "HEAD_LAST", "cancelled",
                     "anti_tribute"} - set(modes)
                ),
            },
        },
        "conservation": {
            "all_conserved": all(bool(audit["conserved"]) for audit in audits),
            "violations": [
                {"deal_index": int(sample.deal_index), "token_step": int(sample.token_step)}
                for sample, audit in zip(samples, audits) if not audit["conserved"]
            ],
            "finished_zero_all": all(bool(audit["finished_players_zero"]) for audit in audits),
        },
        "public_ledger": {
            "schema": PHASE5_PUBLIC_LEDGER_SCHEMA,
            "feature_width": PUBLIC_FEATURE_WIDTH,
            "all_conserved": all(bool(audit.get("ledger_conserved")) for audit in audits),
            "known_lower_bound_all": all(
                bool(audit.get("known_owner_lower_bound")) for audit in audits
            ),
            "owner_audit_all": all(
                bool(audit.get("ledger_owner_matches_engine")) for audit in audits
            ),
            "events_present": all(
                int(sample.audit.get("public_event_count", -1)) == len(sample.public_events)
                for sample in samples
            ),
            "unresolved_paths": [
                "DOUBLE_DOWN RETURN self-exchange ownership remains unassigned "
                "(retained in the residual unknown pool, never guessed)"
            ],
        },
        "label_definition": {
            "seats": "next/partner/previous relative to the acting seat",
            "ranks": "13 printed ranks + small/big joker",
            "wild": "a wild card counts under its printed rank, never its declaration",
            "joker_max": 2,
        },
        "split": {
            "unit": "group_id",
            "assignment": dict(sorted(split_assignment.items())),
            "counts": {
                name: sum(1 for value in split_assignment.values() if value == name)
                for name in ("train", "validation", "test")
            },
        },
        "deals": [dict(item) for item in deals_meta],
        "records_hash": dataset_hash(records),
    }


def sample_matrix(samples: Sequence[BeliefSample]) -> dict[str, np.ndarray]:
    """Stack the model-visible inputs, public ledger and labels."""
    if not samples:
        raise ValueError("no belief samples")
    public = np.stack([sample.public for sample in samples]).astype(np.int64)
    if public.shape[1] != PUBLIC_FEATURE_WIDTH:
        raise ValueError(
            f"belief samples must carry {PUBLIC_FEATURE_WIDTH}-wide v2 public "
            f"ledger features, got {public.shape[1]}"
        )
    from .belief_data import decode_public_ledger

    decoded = decode_public_ledger(public)
    return {
        # Probe input is the acting seat's own channel row.  ``sample.channels``
        # stays the full ``[4, 512]`` observer stack (the ledger/leakage audit
        # needs it); stacking the stack here produced ``[N, 4, 512]`` and made
        # ``build_probe_features`` reject the matrix as "channels must be
        # [N, 512]".  Project to ``sample.channels[sample.actor]`` (``[512]``).
        "channels": np.stack([
            np.asarray(sample.channels)[int(sample.actor)] for sample in samples
        ]).astype(np.uint8),
        "tokens": np.stack([sample.tokens for sample in samples]).astype(np.int64),
        "public": public.astype(np.float32),
        "public_cards": np.stack(
            [sample.public_cards for sample in samples]
        ).astype(np.uint8),
        "known_counts": decoded["known"].astype(np.int64),
        "unknown_counts": decoded["unknown"].astype(np.int64),
        "escrow_unknown": decoded["escrow_unknown"].astype(np.int64),
        "history_counts": np.stack([
            public_history_counts(np.flatnonzero(sample.public_cards))
            for sample in samples
        ]).astype(np.int64),
        "counts": np.stack([sample.label_counts for sample in samples]).astype(np.float32),
        "mask": np.stack([sample.label_mask for sample in samples]).astype(bool),
        "groups": np.asarray([sample.group_id for sample in samples]),
    }


__all__ = [
    "BeliefCollectionError", "BeliefSample", "collect_belief_dataset",
    "context_for", "dataset_manifest", "public_features", "sample_matrix",
]
