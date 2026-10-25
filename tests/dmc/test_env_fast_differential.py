# SPDX-License-Identifier: MIT
"""Stage 07C differential equivalence suite: ``DMCEnvBatch`` vs ``EnvBatch``.

The fast driver (``guandan_marvel/dmc/env_fast.py``) removes redundant adapter
validation, so its safety rests entirely on this suite.  There is no tolerance
and no state sampling: every comparison is byte equality, and the first
divergence fails with the batch step, the slot and both values.

Coverage of one differential game (all slots driven in lockstep by a random
legal-candidate policy):

* every **acting-seat observation token row** of every step (all four seats'
  token rows are compared, a superset of the required acting-seat stream);
* every **acting-seat channel row** (the 512-wide actor encoding of the seat
  that acts at the next decision — the only row the sampler/evaluator read);
* every **legal action sequence** at every decision point (both live
  protocols, full canonical token tuples);
* the **public state digest** and the **full protocol digest** after every
  executed action (commit boundary), plus byte-identical ``serialize()``;
* **terminal rewards / ranking / outcome** for every finished game.

The suite id is :data:`guandan_marvel.dmc.env_fast.DIFFERENTIAL_SUITE` and the
Gate L plan declares it, so a delivery cannot claim an equivalence it did not
run.
"""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from guandan.combos import CombinationKind
from guandan_marvel.backends import get_game_backend
from guandan_marvel.config import ModelConfig, PPOConfig, TrainConfig
from guandan_marvel.dmc.candidates import enumerate_candidates
from guandan_marvel.dmc.env_fast import (
    DIFFERENTIAL_SUITE,
    DRIVER_VERSION,
    DMCEnvBatch,
)
from guandan_marvel.dmc.q_agent import DMCAgent
from guandan_marvel.dmc.replay import DMCReplay
from guandan_marvel.dmc.sampler import DMCSampler, DMCSamplerConfig
from guandan_marvel.game import EnvBatch, _action_row

#: The acceptance floor: at least 200 random games must agree byte-for-byte.
DIFFERENTIAL_GAMES = 200
DIFFERENTIAL_BATCH = 8
DIFFERENTIAL_SEED = 20_261_025

GAME = get_game_backend("python_reference")

TINY_MODEL = {
    "history_dim": 16,
    "feature_dim": 24,
    "decoder_layers": 1,
    "decoder_heads": 4,
    "state_layers": 1,
    "action_layers": 1,
    "fuse_layers": 1,
}


# ---------------------------------------------------------------------------
# Canonical projections
# ---------------------------------------------------------------------------
def _declaration_key(declaration) -> list | None:
    if declaration is None:
        return None
    return [
        declaration.kind.value,
        int(declaration.level_rank),
        int(declaration.comparison_rank),
        [int(value) for value in declaration.comparison_key],
        [int(card.card_id) for card in declaration.cards],
        sorted([int(card_id), int(rank), int(suit)]
               for card_id, (rank, suit) in declaration.wild_assignments.items()),
        sorted(int(card_id) for card_id in declaration.natural_wild_ids),
    ]


def public_state(protocol) -> dict:
    """Every public field of the round plus per-seat hand *sizes*.

    Hand identities stay out (they are private); the full protocol digest
    below covers them as a strictly stronger check.
    """
    state = protocol.round_state
    previous = state.previous_result
    return {
        "phase": state.phase.value,
        "level_rank": int(state.level_rank),
        "active_seat": int(state.active_seat),
        "leader_seat": int(state.leader_seat),
        "current_winning": _declaration_key(state.current_winning),
        "current_winner_seat": state.current_winner_seat,
        "consecutive_passes": int(state.consecutive_passes),
        "finished_ranks": [int(seat) for seat in state.finished_ranks],
        "emptied_seats": [int(seat) for seat in state.emptied_seats],
        "ranking": [int(value) for value in state.ranking],
        "done": bool(state.done),
        "outcome_class": state.outcome_class,
        "winner_team": state.winner_team,
        "rewards": [float(value) for value in state.rewards],
        "team_rewards": [float(value) for value in state.team_rewards],
        "committed_step": int(state.committed_step),
        "token_step": int(protocol.token_step),
        "prefix_tokens": [int(token) for token in protocol._prefix_tokens],
        "truncated": bool(protocol.truncated),
        "finish_committed_step": state.finish_committed_step,
        "finish_token_step": state.finish_token_step,
        "hand_sizes": [len(hand) for hand in state.hands],
        "previous_result": None if previous is None else [
            [int(value) for value in previous.ranking],
            int(previous.winner_team),
            previous.outcome_class,
            bool(previous.tribute_cancelled),
        ],
        "tribute_donors": [int(seat) for seat in state.tribute_donors],
        "tribute_order": [int(seat) for seat in state.tribute_order],
        "anti_tribute": bool(state.anti_tribute),
        "tribute_selections": {
            str(seat): int(card.card_id)
            for seat, card in sorted(state.tribute_selections.items())
        },
        "tribute_escrow": [int(card.card_id) for card in state.tribute_escrow],
        "tribute_transfers": [
            [int(donor), int(recipient), int(card.card_id)]
            for donor, recipient, card in state.tribute_transfers
        ],
        "return_obligations": [
            [int(recipient), int(donor)] for recipient, donor in state.return_obligations
        ],
        "return_transfers": [
            [int(recipient), int(donor), int(card.card_id)]
            for recipient, donor, card in state.return_transfers
        ],
        "history": [
            [
                int(action.player_id),
                action.kind.value if isinstance(action.kind, CombinationKind) else str(action.kind),
                [int(card.card_id) for card in action.cards],
                _declaration_key(action.declaration),
            ]
            for action in state.history
        ],
    }


def _digest(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def public_digest(protocol) -> str:
    return _digest(public_state(protocol))


def full_digest(protocol) -> str:
    """Digest of the complete protocol serialization (private hands included)."""
    return hashlib.sha256(protocol.serialize()).hexdigest()


def _fail(step: int, slot: int, field: str, reference, fast) -> None:
    raise AssertionError(
        f"stage-07C differential divergence: {field} differs at batch step "
        f"{step}, slot {slot}\n  EnvBatch   = {reference!r}\n  DMCEnvBatch= {fast!r}"
    )


# ---------------------------------------------------------------------------
# The lockstep differential harness
# ---------------------------------------------------------------------------
def _compare_step(ref_step, fast_step, step: int) -> int:
    """Compare one batch step; returns the number of terminal slots."""
    for slot in range(ref_step.observations.tokens.shape[0]):
        if not np.array_equal(ref_step.observations.tokens[slot],
                              fast_step.observations.tokens[slot]):
            _fail(step, slot, "observation token rows (4 seats)",
                  ref_step.observations.tokens[slot].tolist(),
                  fast_step.observations.tokens[slot].tolist())
        acting = int(ref_step.player_ids[slot])
        if int(fast_step.player_ids[slot]) != acting:
            _fail(step, slot, "player_ids", acting, int(fast_step.player_ids[slot]))
        if acting >= 0:
            if not np.array_equal(ref_step.observations.channels[slot, acting],
                                  fast_step.observations.channels[slot, acting]):
                reference_row = ref_step.observations.channels[slot, acting]
                fast_row = fast_step.observations.channels[slot, acting]
                differing = np.flatnonzero(reference_row != fast_row).tolist()
                _fail(step, slot, "acting-seat channel row",
                      f"differing channel indices {differing}",
                      "see EnvBatch row")
        if not np.array_equal(ref_step.rewards[slot], fast_step.rewards[slot]):
            _fail(step, slot, "rewards", ref_step.rewards[slot].tolist(),
                  fast_step.rewards[slot].tolist())
        if bool(ref_step.is_terminal[slot]) != bool(fast_step.is_terminal[slot]):
            _fail(step, slot, "is_terminal", bool(ref_step.is_terminal[slot]),
                  bool(fast_step.is_terminal[slot]))
    return int(np.count_nonzero(ref_step.is_terminal))


def _compare_terminal_info(ref_step, fast_step, step: int, slot: int) -> None:
    ref_info = ref_step.infos[slot]
    fast_info = fast_step.infos[slot]
    for key in (
        "ranking", "winner_team", "outcome_class", "team_reward",
        "finish_token_step", "finish_committed_step",
    ):
        if ref_info.get(key) != fast_info.get(key):
            _fail(step, slot, f"terminal info[{key!r}]", ref_info.get(key), fast_info.get(key))


def _compare_commit(reference_env, fast_env, step: int, slot: int) -> None:
    ref_protocol = reference_env._require_protocol()
    fast_protocol = fast_env._require_protocol()
    ref_public, fast_public = public_digest(ref_protocol), public_digest(fast_protocol)
    if ref_public != fast_public:
        _fail(step, slot, "public state digest after the executed action",
              public_state(ref_protocol), public_state(fast_protocol))
    ref_full, fast_full = full_digest(ref_protocol), full_digest(fast_protocol)
    if ref_full != fast_full:
        _fail(step, slot, "full protocol digest after the executed action",
              ref_full, fast_full)


def run_differential(
    *, games: int = DIFFERENTIAL_GAMES, batch: int = DIFFERENTIAL_BATCH,
    seed: int = DIFFERENTIAL_SEED,
) -> dict:
    """Drive both drivers in lockstep for ``games`` random games.

    Returns the transcript summary (game/step/commit counts and one sha256 over
    every compared record) so the report can cite a concrete evidence digest.
    """
    reference = EnvBatch(batch, auto_reset=True, max_horizon=int(GAME.max_game_horizon))
    fast = DMCEnvBatch(batch, auto_reset=True, max_horizon=int(GAME.max_game_horizon))
    ref_out = reference.reset(seed)
    fast_out = fast.reset(seed)
    _compare_step(ref_out, fast_out, 0)
    transcript = hashlib.sha256()

    rng = np.random.default_rng(seed)
    pending: list[list[int]] = [[] for _ in range(batch)]
    completed = 0
    steps = 0
    commits = 0
    while completed < games:
        tokens = np.zeros(batch, dtype=np.int64)
        for slot in range(batch):
            if not pending[slot]:
                ref_protocol = reference.envs[slot]._require_protocol()
                fast_protocol = fast.envs[slot]._require_protocol()
                ref_sequences = tuple(
                    tuple(candidate.tokens) for candidate in enumerate_candidates(ref_protocol)
                )
                fast_sequences = tuple(
                    tuple(candidate.tokens) for candidate in enumerate_candidates(fast_protocol)
                )
                if ref_sequences != fast_sequences:
                    _fail(steps, slot, "legal action sequences", ref_sequences, fast_sequences)
                if not ref_sequences:
                    raise AssertionError(
                        f"slot {slot} reached a decision point with no legal candidate "
                        f"at batch step {steps}"
                    )
                transcript.update(repr(ref_sequences).encode("utf-8"))
                choice = int(rng.integers(len(ref_sequences)))
                pending[slot] = list(ref_sequences[choice])
            tokens[slot] = pending[slot].pop(0)
        rows = np.stack([
            _action_row(reference.envs[slot]._require_protocol(), int(tokens[slot])).row
            for slot in range(batch)
        ])
        ref_step = reference.step(rows)
        fast_step = fast.step_tokens(tokens)
        steps += 1
        transcript.update(np.ascontiguousarray(ref_step.observations.tokens).tobytes())
        transcript.update(np.ascontiguousarray(fast_step.observations.tokens).tobytes())
        _compare_step(ref_step, fast_step, steps)
        for slot in range(batch):
            if bool(ref_step.is_terminal[slot]):
                completed += 1
                _compare_terminal_info(ref_step, fast_step, steps, slot)
                transcript.update(
                    repr(ref_step.infos[slot].get("ranking")).encode("utf-8")
                    + repr(ref_step.rewards[slot].tolist()).encode("utf-8")
                )
            if not pending[slot]:
                commits += 1
                _compare_commit(reference.envs[slot], fast.envs[slot], steps, slot)
                transcript.update(public_digest(reference.envs[slot]._require_protocol()).encode())
                transcript.update(full_digest(reference.envs[slot]._require_protocol()).encode())
        if steps > 200_000:  # pragma: no cover - runaway guard, never hit
            raise AssertionError("differential harness exceeded its step budget")
    if reference.serialize() != fast.serialize():
        raise AssertionError("batch serialization diverged after the differential run")
    return {
        "suite": DIFFERENTIAL_SUITE,
        "driver_version": DRIVER_VERSION,
        "games": completed,
        "batch_steps": steps,
        "commits": commits,
        "batch": batch,
        "seed": seed,
        "transcript_sha256": transcript.hexdigest(),
    }


# ---------------------------------------------------------------------------
# Core acceptance test
# ---------------------------------------------------------------------------
def test_differential_games_are_byte_identical() -> None:
    report = run_differential()
    assert report["suite"] == DIFFERENTIAL_SUITE
    assert report["games"] >= 200, "the acceptance floor is 200 random games"
    assert report["commits"] >= report["games"]
    assert report["batch_steps"] > 0
    assert len(report["transcript_sha256"]) == 64


def test_differential_is_deterministic() -> None:
    first = run_differential(games=4, batch=2, seed=7)
    second = run_differential(games=4, batch=2, seed=7)
    assert first["transcript_sha256"] == second["transcript_sha256"]
    assert first["batch_steps"] == second["batch_steps"]


# ---------------------------------------------------------------------------
# Terminal state (auto_reset=False keeps it inspectable)
# ---------------------------------------------------------------------------
def test_terminal_state_matches_without_auto_reset() -> None:
    reference = EnvBatch(1, auto_reset=False, max_horizon=int(GAME.max_game_horizon))
    fast = DMCEnvBatch(1, auto_reset=False, max_horizon=int(GAME.max_game_horizon))
    _compare_step(reference.reset(31), fast.reset(31), 0)
    rng = np.random.default_rng(31)
    pending: list[int] = []
    step = 0
    while True:
        if not pending:
            candidates = enumerate_candidates(reference.envs[0]._require_protocol())
            pending = list(candidates[int(rng.integers(len(candidates)))].tokens)
        token = pending.pop(0)
        row = _action_row(reference.envs[0]._require_protocol(), token).row
        ref_step = reference.step(np.asarray([row], dtype=np.int64))
        fast_step = fast.step_tokens(np.asarray([token], dtype=np.int64))
        step += 1
        _compare_step(ref_step, fast_step, step)
        if bool(ref_step.is_terminal[0]):
            break
    ref_protocol = reference.envs[0]._require_protocol()
    fast_protocol = fast.envs[0]._require_protocol()
    assert ref_protocol.serialize() == fast_protocol.serialize()
    assert ref_protocol.round_state.ranking == fast_protocol.round_state.ranking
    assert ref_protocol.round_state.outcome_class == fast_protocol.round_state.outcome_class
    assert ref_protocol.round_state.winner_team == fast_protocol.round_state.winner_team
    assert np.array_equal(ref_protocol.round_state.rewards, fast_protocol.round_state.rewards)
    assert ref_protocol.round_state.finish_token_step == fast_protocol.round_state.finish_token_step
    _compare_terminal_info(ref_step, fast_step, step, 0)
    # A terminal slot is not stepped again (fail-loud on both drivers).
    with pytest.raises(Exception):
        reference.step(np.asarray([row], dtype=np.int64))
    with pytest.raises(Exception):
        fast.step_tokens(np.asarray([token], dtype=np.int64))


# ---------------------------------------------------------------------------
# Transport: one payload, two drivers
# ---------------------------------------------------------------------------
def test_serialize_is_byte_identical_and_cross_loadable() -> None:
    reference = EnvBatch(2, auto_reset=True, max_horizon=int(GAME.max_game_horizon))
    fast = DMCEnvBatch(2, auto_reset=True, max_horizon=int(GAME.max_game_horizon))
    reference.reset(11)
    fast.reset(11)
    assert reference.serialize() == fast.serialize()
    rng = np.random.default_rng(11)
    pending: list[list[int]] = [[], []]
    steps = 0
    # Stop on a sequence boundary so the restored drivers start a fresh decision.
    while steps < 40 or any(pending):
        tokens = np.zeros(2, dtype=np.int64)
        for slot in range(2):
            if not pending[slot]:
                candidates = enumerate_candidates(reference.envs[slot]._require_protocol())
                pending[slot] = list(candidates[int(rng.integers(len(candidates)))].tokens)
            tokens[slot] = pending[slot].pop(0)
        rows = np.stack([
            _action_row(reference.envs[slot]._require_protocol(), int(tokens[slot])).row
            for slot in range(2)
        ])
        reference.step(rows)
        fast.step_tokens(tokens)
        steps += 1
        assert reference.serialize() == fast.serialize(), f"payload diverged at step {steps}"

    payload = reference.serialize()
    restored_fast = DMCEnvBatch.deserialize(payload)
    assert restored_fast.serialize() == payload
    restored_reference = EnvBatch.deserialize(fast.serialize())
    assert restored_reference.serialize() == payload

    # Both restorations continue identically.
    next_tokens = np.asarray([
        int(enumerate_candidates(restored_fast.envs[slot]._require_protocol())[0].tokens[0])
        for slot in range(2)
    ], dtype=np.int64)
    fast_out = restored_fast.step_tokens(next_tokens)
    reference_out = restored_reference.step(
        np.stack([
            _action_row(restored_reference.envs[slot]._require_protocol(), int(next_tokens[slot])).row
            for slot in range(2)
        ])
    )
    assert np.array_equal(fast_out.observations.tokens, reference_out.observations.tokens)
    assert restored_fast.serialize() == restored_reference.serialize()


# ---------------------------------------------------------------------------
# Context schedule (per-deal counter semantics)
# ---------------------------------------------------------------------------
def test_context_schedule_matches_reference() -> None:
    from guandan_marvel.context_schedule import expand_context_schedule, validate_context_schedule

    schedule = validate_context_schedule({
        "schema": "guandan-marvel-training-context-schedule-v1",
        "contexts": [
            {"level_rank": 3, "leader_seat": 0, "weight": 1},
            {"level_rank": 7, "leader_seat": 2, "weight": 2},
            {"level_rank": 15, "leader_seat": 3, "weight": 1},
        ],
    })
    expanded = expand_context_schedule(schedule)
    reference = EnvBatch(2, auto_reset=True, max_horizon=int(GAME.max_game_horizon))
    fast = DMCEnvBatch(2, auto_reset=True, max_horizon=int(GAME.max_game_horizon))
    reference.configure_context_schedule(expanded, offset=1)
    fast.configure_context_schedule(expanded, offset=1)
    _compare_step(reference.reset(5), fast.reset(5), 0)
    rng = np.random.default_rng(5)
    pending: list[list[int]] = [[], []]
    for step in range(120):
        tokens = np.zeros(2, dtype=np.int64)
        for slot in range(2):
            if not pending[slot]:
                candidates = enumerate_candidates(reference.envs[slot]._require_protocol())
                pending[slot] = list(candidates[int(rng.integers(len(candidates)))].tokens)
            tokens[slot] = pending[slot].pop(0)
        rows = np.stack([
            _action_row(reference.envs[slot]._require_protocol(), int(tokens[slot])).row
            for slot in range(2)
        ])
        _compare_step(reference.step(rows), fast.step_tokens(tokens), step + 1)
    assert reference.serialize() == fast.serialize()


# ---------------------------------------------------------------------------
# Horizon truncation: a branch normal play never reaches, compared fail-closed
# ---------------------------------------------------------------------------
def test_truncation_horizon_fails_closed_on_both_drivers() -> None:
    """With a tiny horizon both drivers must truncate at the same token step.

    This is the one branch the differential games cannot reach (a real game ends
    long before ``max_token_steps``), so it is pinned separately: identical state
    up to the truncating token, the same step, the same error type.

    The *post-failure* states differ by design and are asserted as such:
    ``EnvBatch`` is transactional (a failed step publishes nothing, so its slot
    keeps the pre-step state), while the fast driver advances in place and
    poisons itself.  Both are fail-closed; ``DMCSampler`` poisons on any
    exception, so no supported path observes the difference.
    """
    reference = EnvBatch(1, auto_reset=True, max_horizon=20)
    fast = DMCEnvBatch(1, auto_reset=True, max_horizon=20)
    reference.reset(17)
    fast.reset(17)
    rng = np.random.default_rng(17)
    pending: list[int] = []
    step = 0
    while True:
        if not pending:
            candidates = enumerate_candidates(reference.envs[0]._require_protocol())
            pending = list(candidates[int(rng.integers(len(candidates)))].tokens)
        token = pending.pop(0)
        row = _action_row(reference.envs[0]._require_protocol(), token).row
        step += 1
        assert reference.envs[0]._require_protocol().serialize() == (
            fast.envs[0]._require_protocol().serialize()
        ), f"state diverged before the truncating step {step}"
        reference_error = fast_error = None
        try:
            reference.step(np.asarray([row], dtype=np.int64))
        except Exception as error:  # noqa: BLE001 - the point is the failure
            reference_error = error
        try:
            fast.step_tokens(np.asarray([token], dtype=np.int64))
        except Exception as error:  # noqa: BLE001 - the point is the failure
            fast_error = error
        assert (reference_error is None) == (fast_error is None), (
            f"truncation diverged at token step {step}: "
            f"EnvBatch={reference_error!r} DMCEnvBatch={fast_error!r}"
        )
        if reference_error is not None:
            assert type(reference_error).__name__ == "TrajectoryHorizonError"
            assert type(fast_error).__name__ == "TrajectoryHorizonError"
            assert step == 20, f"truncation must fire at max_token_steps, got step {step}"
            break
        assert step < 25, "the tiny horizon never truncated; the test is vacuous"
    # Fail-closed on both sides, with the documented transactional asymmetry.
    assert fast.poisoned is True
    assert fast.envs[0].truncated is True
    assert reference.envs[0].truncated is False
    assert reference.envs[0].token_step == 19
    assert fast.envs[0].token_step == 20


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------
def test_forged_sequence_fails_closed_and_poisons() -> None:
    fast = DMCEnvBatch(1, auto_reset=True, max_horizon=int(GAME.max_game_horizon))
    fast.reset(3)
    protocol = fast.envs[0]._require_protocol()
    legal = {int(token) for token in protocol.legal_next_tokens().tolist()}
    forged = next(token for token in range(8, 200) if token not in legal)
    with pytest.raises(Exception):
        fast.step_tokens(np.asarray([forged], dtype=np.int64))
    assert fast.poisoned is True
    with pytest.raises(RuntimeError, match="unusable"):
        fast.step_tokens(np.asarray([forged], dtype=np.int64))


def test_commit_of_a_noncanonical_prefix_fails_closed() -> None:
    fast = DMCEnvBatch(1, auto_reset=True, max_horizon=int(GAME.max_game_horizon))
    fast.reset(3)
    protocol = fast.envs[0]._require_protocol()
    candidates = enumerate_candidates(protocol)
    longest = max(candidates, key=lambda candidate: len(candidate.tokens))
    # A >=3-token canonical action guarantees that ``prefix + (COMMIT,)`` is not
    # itself canonical (a 2-token action such as a tribute would close legally).
    assert len(longest.tokens) >= 3
    # Feed the first token of a real action, then a COMMIT that does not close
    # any canonical sequence from that prefix.
    fast.step_tokens(np.asarray([int(longest.tokens[0])], dtype=np.int64))
    with pytest.raises(Exception):
        fast.step_tokens(np.asarray([7], dtype=np.int64))
    assert fast.poisoned is True


# ---------------------------------------------------------------------------
# The sampler on both drivers pushes byte-identical samples
# ---------------------------------------------------------------------------
def _make_sampler(driver: str, *, num_envs: int = 2) -> DMCSampler:
    torch.manual_seed(11)
    agent = DMCAgent(TrainConfig(
        game=GAME,
        algo_name="dmc",
        device="cpu",
        ppo=PPOConfig(training_semantics_profile="dmc_v1"),
        model=ModelConfig(**TINY_MODEL),
    ))
    replay = DMCReplay(
        4096,
        max_obs_tokens=256,
        channel_dim=int(GAME.num_channels),
        num_tokens=int(GAME.num_tokens),
        pad_token=int(GAME.pad_token),
        num_players=4,
        seed=0,
    )
    return DMCSampler(
        q_agent=agent,
        replay=replay,
        config=DMCSamplerConfig(
            num_envs=num_envs,
            device="cpu",
            env_seed=0,
            seed=0,
            env_driver=driver,
            max_obs_tokens=256,
            obs_head_tokens=64,
        ),
    )


def test_sampler_fast_and_reference_push_identical_samples() -> None:
    fast = _make_sampler("fast")
    reference = _make_sampler("reference")
    assert fast.health()["env_driver"] == "fast"
    assert reference.health()["env_driver"] == "reference"

    fast_stats = fast.collect_games(6)
    reference_stats = reference.collect_games(6)

    assert fast_stats["samples_pushed"] == reference_stats["samples_pushed"] > 0
    assert fast_stats["decisions"] == reference_stats["decisions"]
    for key in (
        "games_started", "games_completed", "decisions", "single_candidate_decisions",
        "multi_candidate_decisions", "samples_pushed", "argmax_hit_rate",
        "illegal", "truncated", "nonfinite", "obs_head_drops", "obs_tail_drops",
    ):
        assert fast_stats["health"][key] == reference_stats["health"][key], key

    rows = len(fast._replay)
    assert rows == len(reference._replay)
    for name in (
        "_obs_tokens", "_obs_token_lengths", "_obs_channels", "_action_feats",
        "_z", "_seat", "_game_id",
    ):
        assert np.array_equal(
            getattr(fast._replay, name)[:rows], getattr(reference._replay, name)[:rows],
        ), name
    assert fast.game_records() == reference.game_records()

    # Resuming a fast sampler from a reference payload continues identically.
    state = reference.state_dict()
    assert state["env_driver"] == "reference"
    resumed = _make_sampler("fast")
    resumed.load_state_dict(copy.deepcopy(state))
    assert resumed.health()["games_completed"] == 6
    resumed_stats = resumed.collect_games(2)
    assert resumed_stats["samples_pushed"] > 0


def test_sampler_rejects_unknown_driver() -> None:
    with pytest.raises(ValueError, match="env_driver"):
        DMCSamplerConfig(num_envs=1, env_driver="cpp")
