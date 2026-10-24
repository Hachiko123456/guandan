"""Fixed-seed fuzz: macro segmentation versus an independent token-stream reference.

The reference here is a plain-Python token-stream analysis: it derives macro
boundaries, per-player successors and tau from the *declarative* macro specs
used to generate each trajectory.  It never calls the segmentation
implementation nor reproduces its (player, ordinal) lookup, so agreement is
real evidence rather than self-comparison.

Coverage per group: batch size 1-4, 0-60 macros per game, macro length 1-32,
chance/environment tokens inside and between macros, different ``num_steps``
per trajectory, and hostile padding (player ids from {-2, -1, 0, 3, 99, 999},
action indices around +/-2**40, garbage availability rows and terminal flags).
"""
from __future__ import annotations

import numpy as np
import torch

from guandan_marvel.training.macro_actions import (
    ACTION_KIND_BASE,
    ACTION_PAYLOAD,
    ACTION_ROW_BEGIN,
    ACTION_ROW_END,
    COMMIT_KIND_COMMIT,
    COMMIT_KIND_PASS,
    KIND_COMMIT,
    KIND_PASS,
    segment_macro_actions,
)

P = 4
ACTION_COUNT = 8
NON_COMMIT_KIND = 2
NON_COMMIT_KINDS = (2, 3, 4, 5, 6, 7)
PADDING_PLAYER_CHOICES = (-2, -1, 0, 3, 99, 999)
FUZZ_SEED = 20261019
FUZZ_GROUPS = 1100
MIN_REAL_MACROS_CHECKED = 1000


def _row(kind_code: int) -> list[int]:
    return [
        ACTION_ROW_BEGIN, 100, 263, ACTION_KIND_BASE + kind_code,
        ACTION_PAYLOAD, 7,
        306 if kind_code in (KIND_PASS, KIND_COMMIT) else 305,
        ACTION_ROW_END,
    ]


def _generate_trajectory(rng):
    """Return ``(tokens, macros)``; tokens are ``(player_id, kind_or_None)``.

    ``macros`` records the declarative boundary of every real macro in start
    order: player, start index, commit (end) index, last token index, token
    count, and commit kind.
    """
    tokens: list[tuple[int, int | None]] = []
    macros: list[dict] = []
    for _ in range(int(rng.integers(0, 61))):
        if rng.random() < 0.35:
            for _ in range(int(rng.integers(1, 5))):
                tokens.append((-1 if rng.random() < 0.5 else -2, None))
        player = int(rng.integers(0, P))
        length = int(rng.integers(1, 33))
        kind = KIND_PASS if (length == 1 and rng.random() < 0.3) else KIND_COMMIT
        start = len(tokens)
        for step in range(length):
            if step > 0 and rng.random() < 0.15:
                tokens.append((-1 if rng.random() < 0.5 else -2, None))
            tokens.append((player, kind if step == length - 1 else NON_COMMIT_KIND))
        end = len(tokens) - 1
        owned = [t for t in range(start, end + 1) if tokens[t][0] == player]
        macros.append({
            "player": player,
            "start": start,
            "end": end,
            "last": owned[-1],
            "length": len(owned),
            "kind": kind,
        })
    if rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 4))):
            tokens.append((-1 if rng.random() < 0.5 else -2, None))
    return tokens, macros


def _expected_token_to_macro(tokens, macros) -> list[int]:
    expected = [-1] * len(tokens)
    for macro_id, macro in enumerate(macros):
        for t in range(macro["start"], macro["end"] + 1):
            if tokens[t][0] == macro["player"]:
                expected[t] = macro_id
    return expected


def _expected_successors(macros):
    by_player: dict[int, list[int]] = {}
    for macro_id, macro in enumerate(macros):
        by_player.setdefault(macro["player"], []).append(macro_id)
    next_index = {}
    tau = {}
    for ids in by_player.values():
        for position, macro_id in enumerate(ids):
            if position + 1 < len(ids):
                successor = ids[position + 1]
                next_index[macro_id] = successor
                tau[macro_id] = macros[successor]["start"] - macros[macro_id]["start"]
            else:
                next_index[macro_id] = -1
                tau[macro_id] = 0
    return next_index, tau


def _check_group(rng) -> int:
    B = int(rng.integers(1, 5))
    per_traj = [_generate_trajectory(rng) for _ in range(B)]
    lengths = [len(tokens) for tokens, _ in per_traj]
    T = max(max(lengths), 1) + int(rng.integers(0, 6))

    player_ids = np.zeros((B, T), dtype=np.int64)
    action_taken = np.zeros((B, T), dtype=np.int64)
    avail_tokens = np.tile(
        np.asarray(_row(NON_COMMIT_KIND), dtype=np.int64), (B, T, ACTION_COUNT, 1)
    )
    for action in range(ACTION_COUNT):
        kind = NON_COMMIT_KINDS[action % len(NON_COMMIT_KINDS)]
        avail_tokens[:, :, action, 3] = ACTION_KIND_BASE + kind
    for b, (tokens, _macros) in enumerate(per_traj):
        for t, (player, kind) in enumerate(tokens):
            player_ids[b, t] = player
            if kind is not None:
                avail_tokens[b, t, 0, 3] = ACTION_KIND_BASE + kind
    avail_is_commit = np.isin(
        avail_tokens[..., 3],
        (ACTION_KIND_BASE + KIND_PASS, ACTION_KIND_BASE + KIND_COMMIT),
    )
    is_terminal = np.zeros((B, T), dtype=np.bool_)
    for b, n in enumerate(lengths):
        if n > 0:
            is_terminal[b, n - 1] = True
        # Hostile garbage on the padding region; it must never be read.
        avail_tokens[b, n:] = 314
        avail_is_commit[b, n:] = True
        is_terminal[b, n:] = True
        player_ids[b, n:] = rng.choice(PADDING_PLAYER_CHOICES, size=T - n)
        magnitudes = rng.integers(1 << 33, 1 << 40, size=T - n)
        signs = rng.choice([-1, 1], size=T - n)
        action_taken[b, n:] = magnitudes * signs

    segmentation = segment_macro_actions(
        num_steps=torch.as_tensor(lengths, dtype=torch.long),
        player_ids=torch.as_tensor(player_ids),
        action_taken=torch.as_tensor(action_taken),
        avail_is_commit=torch.as_tensor(avail_is_commit),
        avail_tokens=torch.as_tensor(avail_tokens),
        is_terminal=torch.as_tensor(is_terminal),
    )

    active_players = [
        int(player_ids[b, t])
        for b in range(B)
        for t in range(lengths[b])
        if player_ids[b, t] >= 0
    ]
    assert segmentation.num_players == ((max(active_players) + 1) if active_players else 0)
    assert segmentation.batch_size == B
    assert segmentation.horizon == T
    assert segmentation.incomplete_count == 0
    assert segmentation.player_switch_violation_count == 0

    macro_checks = 0
    for b in range(B):
        tokens, macros = per_traj[b]
        n = lengths[b]
        next_index, tau = _expected_successors(macros)
        expected_ttm = _expected_token_to_macro(tokens, macros)
        assert int(segmentation.num_macros[b]) == len(macros)
        for t in range(T):
            expected = expected_ttm[t] if t < n else -1
            assert int(segmentation.token_to_macro_id[b, t]) == expected
            if t < n:
                kind = tokens[t][1]
                expected_commit = kind in (KIND_PASS, KIND_COMMIT)
                assert bool(segmentation.chosen_is_commit[b, t]) == expected_commit
            else:
                assert not bool(segmentation.chosen_is_commit[b, t])
        for macro_id, macro in enumerate(macros):
            macro_checks += 1
            assert bool(segmentation.macro_valid[b, macro_id])
            assert int(segmentation.macro_start_index[b, macro_id]) == macro["start"]
            assert int(segmentation.macro_end_index[b, macro_id]) == macro["end"]
            assert int(segmentation.macro_last_index[b, macro_id]) == macro["last"]
            assert int(segmentation.macro_length[b, macro_id]) == macro["length"]
            assert int(segmentation.macro_player[b, macro_id]) == macro["player"]
            assert int(segmentation.macro_next_index[b, macro_id]) == next_index[macro_id]
            assert int(segmentation.macro_tau[b, macro_id]) == tau[macro_id]
            assert bool(segmentation.macro_is_last_for_player[b, macro_id]) == (
                next_index[macro_id] < 0
            )
            assert bool(segmentation.macro_is_terminal[b, macro_id]) == (macro["end"] == n - 1)
            expected_kind = (
                COMMIT_KIND_PASS if macro["kind"] == KIND_PASS else COMMIT_KIND_COMMIT
            )
            assert int(segmentation.macro_commit_kind[b, macro_id]) == expected_kind
        for macro_id in range(len(macros), segmentation.max_macros):
            assert not bool(segmentation.macro_valid[b, macro_id])
            assert not bool(segmentation.macro_is_last_for_player[b, macro_id])
            assert not bool(segmentation.macro_is_terminal[b, macro_id])
            assert int(segmentation.macro_player[b, macro_id]) == -1
            assert int(segmentation.macro_next_index[b, macro_id]) == -1
            assert int(segmentation.macro_tau[b, macro_id]) == 0
            assert int(segmentation.macro_length[b, macro_id]) == 0
            assert int(segmentation.macro_start_index[b, macro_id]) == -1
            assert int(segmentation.macro_end_index[b, macro_id]) == -1
    return macro_checks


def test_macro_segmentation_matches_independent_reference_fuzz():
    rng = np.random.default_rng(FUZZ_SEED)
    macro_checks = 0
    for group in range(FUZZ_GROUPS):
        try:
            macro_checks += _check_group(rng)
        except AssertionError as error:  # pragma: no cover - failure reporting
            raise AssertionError(
                f"fuzz group {group} failed (seed={FUZZ_SEED}): {error}"
            ) from error
    assert macro_checks >= MIN_REAL_MACROS_CHECKED, macro_checks
    print(
        f"fuzz seed={FUZZ_SEED} groups={FUZZ_GROUPS} real_macros_checked={macro_checks}"
    )
