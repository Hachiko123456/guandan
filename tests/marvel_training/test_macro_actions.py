"""Phase-1 macro segmentation and legacy GAE oracle (synthetic rollouts)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.training.legacy_gae import (
    compute_legacy_token_advantages,
    compute_legacy_token_deltas,
    expand_macro_start_advantages,
    legacy_gae_macro_oracle,
)
from guandan_marvel.training.macro_actions import (
    ACTION_KIND_BASE,
    ACTION_PAYLOAD,
    ACTION_ROW_BEGIN,
    ACTION_ROW_END,
    COMMIT_KIND_COMMIT,
    COMMIT_KIND_PASS,
    KIND_COMMIT,
    KIND_PASS,
    macro_diagnostics,
    segment_rollout_macros,
)

P = 4
ACTION_COUNT = 8
NON_COMMIT_KINDS = (2, 3, 4, 5, 6, 7)


def _row(kind_code: int) -> list[int]:
    return [
        ACTION_ROW_BEGIN, 100, 263, ACTION_KIND_BASE + kind_code,
        ACTION_PAYLOAD, 7, 306 if kind_code in (KIND_PASS, KIND_COMMIT) else 305,
        ACTION_ROW_END,
    ]


def _make_rollout(
    tokens,
    *,
    dtype=torch.float64,
    terminal_rewards=None,
    reward_overrides=(),
    horizon=None,
    num_steps=None,
    values_seed=0,
    with_rows=True,
    behavior_action_prob=None,
):
    """tokens: list of (player_id, selected_kind_code); commit iff kind in {0, 1}."""
    batch = 1
    T = len(tokens) if horizon is None else horizon
    assert T >= len(tokens)
    player_ids = torch.zeros((batch, T), dtype=torch.long)
    action_taken = torch.zeros((batch, T), dtype=torch.long)
    avail_is_commit = torch.zeros((batch, T, ACTION_COUNT), dtype=torch.bool)
    avail_weights = torch.ones((batch, T, ACTION_COUNT), dtype=torch.float32)
    avail_tokens = torch.zeros((batch, T, ACTION_COUNT, 8), dtype=torch.long)
    behavior = torch.full((batch, T), 0.2, dtype=dtype)
    for t, (player, kind) in enumerate(tokens):
        player_ids[0, t] = player
        chosen_kind = kind if kind is not None else 3
        for a in range(ACTION_COUNT):
            row_kind = chosen_kind if a == 0 else NON_COMMIT_KINDS[(a + t) % len(NON_COMMIT_KINDS)]
            avail_tokens[0, t, a] = torch.tensor(_row(row_kind), dtype=torch.long)
            avail_is_commit[0, t, a] = row_kind in (KIND_PASS, KIND_COMMIT)
        behavior[0, t] = 0.2 + 0.01 * ((t % 7) - 3)
    if behavior_action_prob is not None:
        behavior = torch.as_tensor(behavior_action_prob, dtype=dtype).reshape(batch, T)

    rewards = torch.zeros((batch, T, P), dtype=dtype)
    if terminal_rewards is not None:
        rewards[0, len(tokens) - 1] = torch.tensor(terminal_rewards, dtype=dtype)
    for (t, player, value) in reward_overrides:
        rewards[0, t, player] = value

    generator = torch.Generator().manual_seed(values_seed)
    values = torch.randn((batch, T, P), generator=generator, dtype=torch.float64).to(dtype) * 0.1
    steps = num_steps if num_steps is not None else len(tokens)
    for t in range(T):
        player = int(player_ids[0, t])
        for q in range(P):
            if t >= steps or player < 0 or q != player:
                values[0, t, q] = 0.0

    terminal = torch.zeros((batch, T), dtype=torch.bool)
    if steps > 0:
        terminal[0, steps - 1] = True

    rollout = TensorDict(
        {
            "num_steps": torch.tensor([steps], dtype=torch.long),
            "player_ids": player_ids,
            "action_taken": action_taken,
            "avail_is_commit": avail_is_commit,
            "avail_tokens": avail_tokens if with_rows else torch.zeros((batch, T, ACTION_COUNT, 8), dtype=torch.long),
            "avail_weights": avail_weights,
            "behavior_action_prob": behavior,
            "rewards": rewards,
            "is_terminal": terminal,
        },
        batch_size=[batch],
    )
    return rollout, values


def _make_batch_rollout(
    trajectories,
    *,
    dtype=torch.float64,
    values_seed=0,
    pad=5,
    garbage_padding=True,
    with_rows=True,
    rewards_overrides=(),
    terminal_rewards_per_traj=None,
):
    """Batch of token trajectories with independent ``num_steps``.

    ``trajectories`` is a list of token lists ``[(player_id, selected_kind)]``
    with possibly different lengths.  Unless ``garbage_padding=False`` every
    padding position carries deliberately hostile garbage (player_id=-2,
    action_taken=999, bogus availability flags/rewards/values) that must never
    influence segmentation or the oracle.
    """
    B = len(trajectories)
    lengths = [len(tokens) for tokens in trajectories]
    T = max(lengths) + pad
    player_ids = torch.zeros((B, T), dtype=torch.long)
    action_taken = torch.zeros((B, T), dtype=torch.long)
    avail_is_commit = torch.zeros((B, T, ACTION_COUNT), dtype=torch.bool)
    avail_tokens = torch.zeros((B, T, ACTION_COUNT, 8), dtype=torch.long)
    avail_weights = torch.ones((B, T, ACTION_COUNT), dtype=torch.float32)
    behavior = torch.full((B, T), 0.2, dtype=dtype)
    rewards = torch.zeros((B, T, P), dtype=dtype)
    terminal = torch.zeros((B, T), dtype=torch.bool)
    for b, tokens in enumerate(trajectories):
        n = lengths[b]
        for t, (player, kind) in enumerate(tokens):
            player_ids[b, t] = player
            chosen_kind = kind if kind is not None else 3
            for a in range(ACTION_COUNT):
                row_kind = chosen_kind if a == 0 else NON_COMMIT_KINDS[(a + t) % len(NON_COMMIT_KINDS)]
                avail_tokens[b, t, a] = torch.tensor(_row(row_kind), dtype=torch.long)
                avail_is_commit[b, t, a] = row_kind in (KIND_PASS, KIND_COMMIT)
            behavior[b, t] = 0.2 + 0.01 * ((t % 7) - 3)
        if terminal_rewards_per_traj is not None:
            rewards[b, n - 1] = torch.tensor(terminal_rewards_per_traj[b], dtype=dtype)
        terminal[b, n - 1] = True
    for (b, t, player, value) in rewards_overrides:
        rewards[b, t, player] = value

    if values_seed is None:
        values = torch.zeros((B, T, P), dtype=dtype)
    else:
        generator = torch.Generator().manual_seed(values_seed)
        values = (torch.randn((B, T, P), generator=generator, dtype=torch.float64) * 0.1).to(dtype)
        for b in range(B):
            for t in range(T):
                player = int(player_ids[b, t])
                for q in range(P):
                    if t >= lengths[b] or player < 0 or q != player:
                        values[b, t, q] = 0.0

    if garbage_padding:
        for b, n in enumerate(lengths):
            player_ids[b, n:] = -2
            action_taken[b, n:] = 999
            avail_is_commit[b, n:] = True
            avail_tokens[b, n:] = 314
            rewards[b, n:] = 123.0
            values[b, n:] = 456.0

    fields = {
        "num_steps": torch.tensor(lengths, dtype=torch.long),
        "player_ids": player_ids,
        "action_taken": action_taken,
        "avail_is_commit": avail_is_commit,
        "avail_weights": avail_weights,
        "behavior_action_prob": behavior,
        "rewards": rewards,
        "is_terminal": terminal,
    }
    if with_rows:
        fields["avail_tokens"] = avail_tokens
    rollout = TensorDict(fields, batch_size=[B])
    return rollout, values


def _all_commit_specs(count: int, *, kind=KIND_COMMIT):
    return [(index % P, kind) for index in range(count)]


def _macro_spec_tokens(specs):
    """specs: (player, length, kind_code_at_commit)."""
    tokens = []
    for (player, length, kind) in specs:
        for step in range(length):
            tokens.append((player, kind if step == length - 1 else 2))
    return tokens


def _assert_macro_invariants(segmentation, rollout):
    player_ids = rollout["player_ids"]
    num_steps = rollout["num_steps"]
    active = torch.arange(segmentation.horizon).unsqueeze(0) < num_steps.unsqueeze(1)
    assert segmentation.incomplete_count == 0
    assert segmentation.player_switch_violation_count == 0
    assert bool((segmentation.token_to_macro_id[~active] == -1).all())
    valid = segmentation.macro_start_index >= 0
    for b in range(segmentation.batch_size):
        for m in range(int(segmentation.num_macros[b])):
            start = int(segmentation.macro_start_index[b, m])
            end = int(segmentation.macro_end_index[b, m])
            last = int(segmentation.macro_last_index[b, m])
            player = int(segmentation.macro_player[b, m])
            assert active[b, start] and active[b, end]
            assert bool(segmentation.chosen_is_commit[b, end])
            assert int(player_ids[b, start]) == int(player_ids[b, end]) == player
            tokens = segmentation.token_to_macro_id[b] == m
            assert bool((player_ids[b][tokens] == player).all())
            assert int(tokens.sum()) == int(segmentation.macro_length[b, m])
            assert int(torch.arange(segmentation.horizon)[tokens].max()) == last
            if segmentation.macro_next_index[b, m] >= 0:
                successor = segmentation.macro_next_index[b, m]
                assert int(segmentation.macro_player[b, successor]) == player
                assert int(segmentation.macro_start_index[b, successor]) - start == int(
                    segmentation.macro_tau[b, m]
                )


def test_single_player_macro_length_one():
    tokens = [(0, KIND_COMMIT) for _ in range(60)]
    rollout, values = _make_rollout(tokens, terminal_rewards=[1, -1, 1, -1])
    segmentation = segment_rollout_macros(rollout)
    assert int(segmentation.num_macros[0]) == 60
    assert bool((segmentation.macro_length[0] == 1).all())
    assert bool((segmentation.macro_tau[0, :-1] == 1).all())
    assert segmentation.macro_is_last_for_player[0, -1]
    _assert_macro_invariants(segmentation, rollout)
    oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=1.0, gae_lambda=0.95)
    assert oracle.max_abs_diff < 1e-10
    diagnostics = macro_diagnostics(segmentation)
    assert diagnostics["macro/count"] == 60
    assert diagnostics["macro/count_by_player/0"] == 60
    assert diagnostics["macro/tau_mean"] == 1.0


def test_four_player_strict_rotation_macro_length_one():
    tokens = _all_commit_specs(60)
    rollout, values = _make_rollout(tokens, terminal_rewards=[-1, 1, -1, 1])
    segmentation = segment_rollout_macros(rollout)
    assert int(segmentation.num_macros[0]) == 60
    assert bool((segmentation.macro_length[0] == 1).all())
    has_next = segmentation.macro_next_index[0] >= 0
    assert bool((segmentation.macro_tau[0][has_next] == 4).all())
    assert int(has_next.sum()) == 60 - P
    assert [int(segmentation.macro_player[0, m]) for m in range(60)] == [m % P for m in range(60)]
    _assert_macro_invariants(segmentation, rollout)
    oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=1.0, gae_lambda=0.95)
    assert oracle.max_abs_diff < 1e-10
    diagnostics = macro_diagnostics(segmentation)
    assert [diagnostics[f"macro/count_by_player/{p}"] for p in range(P)] == [15] * P
    assert diagnostics["macro/tau_mean"] == 4.0


def test_sixty_step_all_commit_matches_legacy_with_gamma_lambda_tau():
    tokens = _all_commit_specs(60)
    rollout, values = _make_rollout(tokens, terminal_rewards=[1, -1, 1, -1])
    gamma, lam = 1.0, 0.95
    segmentation = segment_rollout_macros(rollout)
    oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=gamma, gae_lambda=lam)
    assert oracle.max_abs_diff < 1e-10

    # The plan's gamma^tau * lambda clock must NOT be claimed equivalent to the
    # legacy token GAE even when every macro has length 1 (four-player rotation
    # gives tau = 4 > 1).
    delta = oracle.token_delta
    wrong = torch.zeros_like(oracle.expanded_at_macro_start)
    starts = segmentation.macro_start_index
    for m in range(segmentation.max_macros - 1, -1, -1):
        successor = segmentation.macro_next_index[:, m]
        successor_value = wrong.gather(1, successor.clamp(min=0).unsqueeze(1)).squeeze(1)
        tau = segmentation.macro_tau[:, m].to(delta.dtype)
        wrong[:, m] = delta.gather(1, starts[:, m : m + 1]).squeeze(1) + torch.where(
            successor >= 0,
            (gamma ** tau) * lam * successor_value,
            torch.zeros_like(successor_value),
        )
    wrong_diff = (oracle.legacy_at_macro_start - wrong).abs().max().item()
    assert wrong_diff > 1e-3
    assert abs(lam ** 1 - lam) < 1e-12 and abs(lam ** 4 - lam) > 0.1


def test_variable_macro_lengths_and_tau():
    lengths = [1, 2, 3, 4, 5, 6, 7, 8]
    specs = []
    for index in range(32):
        specs.append(((index * 3) % P, lengths[index % len(lengths)], KIND_COMMIT))
    tokens = _macro_spec_tokens(specs)
    for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 1e-6)):
        rollout, values = _make_rollout(
            tokens, dtype=dtype, terminal_rewards=[1, -1, 1, -1], values_seed=3,
        )
        segmentation = segment_rollout_macros(rollout)
        observed_lengths = [int(v) for v in segmentation.macro_length[0].tolist() if v > 0]
        assert observed_lengths == [length for (_p, length, _k) in specs]
        taus = segmentation.macro_tau[0, segmentation.macro_next_index[0] >= 0]
        assert len(set(int(v) for v in taus.tolist())) > 1
        _assert_macro_invariants(segmentation, rollout)
        oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=1.0, gae_lambda=0.95)
        assert oracle.max_abs_diff < tolerance


def test_pass_and_commit_kind_decoding():
    specs = [
        (0, 1, KIND_PASS), (1, 3, KIND_COMMIT), (2, 1, KIND_PASS),
        (3, 2, KIND_COMMIT), (0, 1, KIND_PASS), (1, 1, KIND_PASS),
    ]
    tokens = _macro_spec_tokens(specs)
    rollout, values = _make_rollout(tokens, terminal_rewards=[1, -1, 1, -1])
    segmentation = segment_rollout_macros(rollout)
    expected = [COMMIT_KIND_PASS if kind == KIND_PASS else COMMIT_KIND_COMMIT for (_p, _l, kind) in specs]
    assert [int(v) for v in segmentation.macro_commit_kind[0].tolist() if v > 0] == expected
    diagnostics = macro_diagnostics(segmentation)
    assert diagnostics["macro/pass_count"] == 4
    assert diagnostics["macro/commit_count"] == 2
    assert diagnostics["macro/committed_count"] == 6

    # Without avail_tokens the fallback classifies length-1 macros as PASS.
    stripped = {key: rollout[key] for key in rollout.keys()}
    del stripped["avail_tokens"]
    fallback = segment_rollout_macros(TensorDict(stripped, batch_size=[1]))
    fallback_kinds = [int(v) for v in fallback.macro_commit_kind[0].tolist() if v > 0]
    assert fallback_kinds == expected

    # A commit-flag mismatch between avail_is_commit and the decoded row is rejected.
    tampered = rollout["avail_is_commit"].clone()
    tampered[0, 0, 0] = not tampered[0, 0, 0]
    with pytest.raises(ValueError, match="disagrees"):
        segment_rollout_macros(
            TensorDict({**{k: rollout[k] for k in rollout.keys()}, "avail_is_commit": tampered}, batch_size=[1])
        )


def test_finished_player_relay_and_exchange_phase():
    # Seat 1 finishes early (no more tokens), seat 2 relays (two commits in a
    # row), tribute/return commits are two-token CARD+COMMIT macros.
    specs = [
        (0, 2, KIND_COMMIT), (1, 1, KIND_PASS), (2, 1, KIND_PASS),
        (3, 1, KIND_PASS), (0, 1, KIND_PASS), (2, 2, KIND_COMMIT),
        (2, 1, KIND_PASS),  # relay: same player commits again immediately
        (3, 2, KIND_COMMIT), (0, 1, KIND_PASS), (2, 3, KIND_COMMIT),
        (3, 1, KIND_PASS), (0, 1, KIND_PASS),
    ]
    tokens = _macro_spec_tokens(specs)
    rollout, values = _make_rollout(
        tokens, terminal_rewards=[1, -1, 1, -1],
        reward_overrides=[(4, 1, 0.25)],
    )
    segmentation = segment_rollout_macros(rollout)
    _assert_macro_invariants(segmentation, rollout)
    players = [int(v) for v in segmentation.macro_player[0].tolist() if v >= 0]
    assert players == [p for (p, _l, _k) in specs]
    assert players.count(1) == 1  # seat 1 finished early and never acts again
    diagnostics = macro_diagnostics(segmentation)
    assert diagnostics["macro/incomplete_count"] == 0
    assert diagnostics["macro/player_switch_violation_count"] == 0
    oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=1.0, gae_lambda=0.95)
    assert oracle.max_abs_diff < 1e-10

    for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 1e-6)):
        rollout_d, values_d = _make_rollout(
            tokens, dtype=dtype, terminal_rewards=[1, -1, 1, -1], values_seed=5,
        )
        oracle_d = legacy_gae_macro_oracle(rollout_d, values_d, discount_gamma=1.0, gae_lambda=0.95)
        assert oracle_d.max_abs_diff < tolerance


def test_intermediate_and_terminal_rewards():
    specs = [(0, 3, KIND_COMMIT), (1, 1, KIND_PASS), (2, 2, KIND_COMMIT), (3, 4, KIND_COMMIT)]
    tokens = _macro_spec_tokens(specs)
    overrides = [(1, 0, 0.5), (2, 1, -0.25), (3, 2, 0.75), (5, 3, -0.5), (6, 0, 0.1)]
    for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 1e-6)):
        rollout, values = _make_rollout(
            tokens, dtype=dtype, terminal_rewards=[1, -1, 1, -1],
            reward_overrides=overrides, values_seed=7,
        )
        segmentation = segment_rollout_macros(rollout)
        oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=1.0, gae_lambda=0.95)
        assert oracle.max_abs_diff < tolerance

    # Terminal-only rewards.
    rollout, values = _make_rollout(tokens, terminal_rewards=[-1, 1, -1, 1], values_seed=8)
    oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=1.0, gae_lambda=0.95)
    assert oracle.max_abs_diff < 1e-10

    # The oracle also holds when non-acting players hold nonzero rewards.
    rollout, values = _make_rollout(
        tokens, terminal_rewards=[1, -1, 1, -1],
        reward_overrides=[(2, 3, 0.9), (4, 0, -0.4)], values_seed=9,
    )
    oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=1.0, gae_lambda=0.95)
    assert oracle.max_abs_diff < 1e-10


def test_gamma_and_lambda_endpoints():
    specs = [(index % P, 1 + (index * 5) % 8, KIND_COMMIT) for index in range(24)]
    tokens = _macro_spec_tokens(specs)
    for gamma in (1.0, 0.97):
        for lam in (0.0, 0.5, 0.95, 1.0):
            rollout, values = _make_rollout(
                tokens, terminal_rewards=[1, -1, 1, -1], values_seed=11,
                reward_overrides=[(2, 1, 0.3)],
            )
            oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=gamma, gae_lambda=lam)
            assert oracle.max_abs_diff < 1e-10, (gamma, lam)

    # lambda = 0 collapses the expansion to the delta of the macro start token.
    rollout, values = _make_rollout(tokens, terminal_rewards=[1, -1, 1, -1], values_seed=12)
    segmentation = segment_rollout_macros(rollout)
    delta = compute_legacy_token_deltas(rollout, values, discount_gamma=1.0)
    expanded = expand_macro_start_advantages(delta, segmentation, discount_gamma=1.0, gae_lambda=0.0)
    start_delta = delta.gather(1, segmentation.macro_start_index.clamp(min=0))
    assert torch.equal(expanded, start_delta)


def test_chance_environment_prefix_and_padding():
    tokens = [
        (-2, None), (0, 2), (-1, None), (0, KIND_COMMIT),
        (1, KIND_COMMIT), (0, KIND_COMMIT),
        (-2, None), (3, 2), (3, KIND_COMMIT),
    ]
    horizon = len(tokens) + 6
    rollout, values = _make_rollout(
        tokens, terminal_rewards=[1, -1, 1, -1], horizon=horizon,
        reward_overrides=[(4, 0, 0.5), (7, 1, -0.5)], values_seed=13,
    )
    segmentation = segment_rollout_macros(rollout)
    assert [int(v) for v in segmentation.token_to_macro_id[0, : len(tokens)].tolist()] == [-1, 0, -1, 0, 1, 2, -1, 3, 3]
    assert bool((segmentation.token_to_macro_id[0, len(tokens):] == -1).all())
    assert not bool(segmentation.chosen_is_commit[0, 0] or segmentation.chosen_is_commit[0, 2] or segmentation.chosen_is_commit[0, 6])
    assert int(segmentation.num_macros[0]) == 4
    assert segmentation.macro_start_index[0, :4].tolist() == [1, 4, 5, 7]
    assert segmentation.macro_end_index[0, :4].tolist() == [3, 4, 5, 8]
    # tau counts global token distance including chance steps: player 0's
    # second macro starts at 5, four tokens after its first macro start at 1.
    assert int(segmentation.macro_next_index[0, 0]) == 2
    assert int(segmentation.macro_tau[0, 0]) == 4
    assert int(segmentation.macro_length[0, 0]) == 2
    assert int(segmentation.macro_length[0, 3]) == 2
    assert segmentation.incomplete_count == 0
    assert segmentation.player_switch_violation_count == 0
    oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=1.0, gae_lambda=0.95)
    assert oracle.max_abs_diff < 1e-10

    # Together with an environment prefix inside the macro span the token gap
    # inside the macro is two, and the expansion still matches the token GAE.
    nested = [(-2, None), (0, 2), (-1, None), (0, KIND_COMMIT), (2, KIND_COMMIT)]
    rollout_n, values_n = _make_rollout(nested, terminal_rewards=[1, -1, 1, -1], values_seed=14)
    oracle_n = legacy_gae_macro_oracle(rollout_n, values_n, discount_gamma=1.0, gae_lambda=0.95)
    assert oracle_n.max_abs_diff < 1e-10


def test_player_switch_violation_and_incomplete_count():
    tokens = [(0, 2), (1, 2), (1, KIND_COMMIT)]
    rollout, _values = _make_rollout(tokens, terminal_rewards=[1, -1, 1, -1])
    segmentation = segment_rollout_macros(rollout)
    assert segmentation.incomplete_count == 1
    assert segmentation.player_switch_violation_count == 1
    diagnostics = macro_diagnostics(segmentation)
    assert diagnostics["macro/incomplete_count"] == 1
    assert diagnostics["macro/player_switch_violation_count"] == 1


GAMMA_GRID = (0.0, 0.5, 0.97, 1.0)
LAMBDA_GRID = (0.0, 0.5, 0.95, 1.0)


def test_gamma_counterexample_exact_reward_windows():
    """The legacy delta must discount *forward* from the decision token.

    Four-player rotation, player 0 acts at t=0 and t=4, all values are zero.
    A single unit reward for player 0 at position k gives delta(0) = gamma^k
    for k in {0,1,2,3}; the reward at t=4 belongs to player 0's *next* decision
    window and must not enter delta(0).  Production advantage at the macro
    start equals the same gamma^k with lambda=1 (k=4 enters via A(4)).
    """
    tokens = [(0, KIND_COMMIT), (1, KIND_COMMIT), (2, KIND_COMMIT),
              (3, KIND_COMMIT), (0, KIND_COMMIT)]
    for gamma in GAMMA_GRID:
        for k in range(5):
            for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 1e-6)):
                rollout, values = _make_batch_rollout(
                    [tokens], dtype=dtype, values_seed=None,
                    rewards_overrides=[(0, k, 0, 1.0)],
                )
                delta = compute_legacy_token_deltas(rollout, values, discount_gamma=gamma)
                expected_delta0 = gamma ** k if k < 4 else 0.0
                assert abs(float(delta[0, 0]) - expected_delta0) < tolerance, (gamma, k, dtype)
                advantage = compute_legacy_token_advantages(
                    rollout, values, discount_gamma=gamma, gae_lambda=1.0,
                )
                assert abs(float(advantage[0, 0]) - gamma ** k) < tolerance, (gamma, k, dtype)
                oracle = legacy_gae_macro_oracle(
                    rollout, values, discount_gamma=gamma, gae_lambda=1.0,
                )
                assert oracle.max_abs_diff < tolerance, (gamma, k, dtype)


def test_oracle_gamma_lambda_matrix():
    specs = [
        (0, 3, KIND_COMMIT), (1, 1, KIND_PASS), (2, 2, KIND_COMMIT),
        (3, 4, KIND_COMMIT), (0, 2, KIND_COMMIT), (2, 1, KIND_PASS),
    ]
    tokens = _macro_spec_tokens(specs)
    overrides = [
        (0, 1, 0, 0.5), (0, 3, 2, -0.25), (0, 5, 3, 0.75), (0, 6, 0, 0.4), (0, 9, 1, -0.6),
    ]
    for gamma in GAMMA_GRID:
        for lam in LAMBDA_GRID:
            rollout, values = _make_batch_rollout(
                [tokens], values_seed=21, rewards_overrides=overrides,
                terminal_rewards_per_traj=[[1, -1, 1, -1]],
            )
            oracle = legacy_gae_macro_oracle(
                rollout, values, discount_gamma=gamma, gae_lambda=lam,
            )
            assert oracle.max_abs_diff < 1e-10, (gamma, lam)


def _random_trajectory(rng, *, max_macros=10, max_length=10, chance_prob=0.25):
    tokens = []
    for _ in range(int(rng.integers(1, max_macros + 1))):
        if rng.random() < chance_prob:
            tokens.append((-1 if rng.random() < 0.5 else -2, None))
        player = int(rng.integers(0, P))
        length = int(rng.integers(1, max_length + 1))
        kind = KIND_PASS if length == 1 and rng.random() < 0.5 else KIND_COMMIT
        for step in range(length):
            tokens.append((player, kind if step == length - 1 else 2))
        if rng.random() < chance_prob:
            tokens.append((-2, None))
    return tokens


def test_oracle_random_synthetic_property():
    rng = np.random.default_rng(20261018)
    oracle_checks = 0
    for seed in range(120):
        tokens = _random_trajectory(rng)
        overrides = []
        for _ in range(int(rng.integers(0, 5))):
            overrides.append((
                0,
                int(rng.integers(0, len(tokens))),
                int(rng.integers(0, P)),
                float(rng.uniform(-1.0, 1.0)),
            ))
        gamma = GAMMA_GRID[seed % len(GAMMA_GRID)]
        lam = LAMBDA_GRID[(seed // len(GAMMA_GRID)) % len(LAMBDA_GRID)]
        dtype = torch.float32 if seed % 8 == 7 else torch.float64
        tolerance = 1e-6 if dtype == torch.float32 else 1e-10
        rollout, values = _make_batch_rollout(
            [tokens], dtype=dtype, values_seed=seed + 1, rewards_overrides=overrides,
            terminal_rewards_per_traj=[[1, -1, 1, -1]],
        )
        oracle = legacy_gae_macro_oracle(
            rollout, values, discount_gamma=gamma, gae_lambda=lam,
        )
        assert oracle.max_abs_diff < tolerance, (seed, gamma, lam, dtype, oracle.max_abs_diff)
        oracle_checks += 1
        if seed % 8 == 0:
            rollout_f64, values_f64 = _make_batch_rollout(
                [tokens], dtype=torch.float64, values_seed=seed + 1,
                rewards_overrides=overrides, terminal_rewards_per_traj=[[1, -1, 1, -1]],
            )
            oracle_f64 = legacy_gae_macro_oracle(
                rollout_f64, values_f64, discount_gamma=gamma, gae_lambda=lam,
            )
            assert oracle_f64.max_abs_diff < 1e-10, (seed, gamma, lam)
            oracle_checks += 1
    assert oracle_checks >= 120


def test_oracle_batch_different_num_steps_and_garbage_padding():
    traj_a = [(0, 2), (0, KIND_COMMIT), (1, KIND_PASS)]
    traj_b = [(0, 2), (0, KIND_COMMIT), (1, 2), (2, 2), (1, KIND_COMMIT), (3, KIND_COMMIT)]
    overrides = [(1, 1, 1, 0.3), (0, 3, 0, -0.2)]
    rollout_garbage, values_garbage = _make_batch_rollout(
        [traj_a, traj_b], values_seed=7, garbage_padding=True, rewards_overrides=overrides,
        terminal_rewards_per_traj=[[1, -1, 1, -1], [-1, 1, -1, 1]],
    )
    rollout_clean, values_clean = _make_batch_rollout(
        [traj_a, traj_b], values_seed=7, garbage_padding=False, rewards_overrides=overrides,
        terminal_rewards_per_traj=[[1, -1, 1, -1], [-1, 1, -1, 1]],
    )
    for gamma in GAMMA_GRID:
        delta_garbage = compute_legacy_token_deltas(rollout_garbage, values_garbage, discount_gamma=gamma)
        delta_clean = compute_legacy_token_deltas(rollout_clean, values_clean, discount_gamma=gamma)
        assert torch.equal(delta_garbage, delta_clean), gamma
        oracle = legacy_gae_macro_oracle(
            rollout_garbage, values_garbage, discount_gamma=gamma, gae_lambda=0.95,
        )
        assert oracle.max_abs_diff < 1e-10, gamma
    segmentation_garbage = segment_rollout_macros(rollout_garbage)
    segmentation_clean = segment_rollout_macros(rollout_clean)
    assert torch.equal(segmentation_garbage.token_to_macro_id, segmentation_clean.token_to_macro_id)
    assert torch.equal(segmentation_garbage.macro_valid, segmentation_clean.macro_valid)
    assert segmentation_garbage.num_macros.tolist() == segmentation_clean.num_macros.tolist() == [2, 4]


def test_oracle_chance_steps_and_macro_span_gaps():
    tokens = [
        (-2, None), (0, 2), (-1, None), (0, KIND_COMMIT),
        (1, KIND_COMMIT), (-2, None), (0, KIND_COMMIT),
    ]
    overrides = [(0, 2, 0, 0.6), (0, 4, 0, -0.4), (0, 5, 1, 0.2)]
    for gamma in (0.5, 0.97, 1.0):
        for lam in (0.0, 0.95):
            rollout, values = _make_batch_rollout(
                [tokens], values_seed=17, rewards_overrides=overrides,
                terminal_rewards_per_traj=[[1, -1, 1, -1]],
            )
            oracle = legacy_gae_macro_oracle(
                rollout, values, discount_gamma=gamma, gae_lambda=lam,
            )
            assert oracle.max_abs_diff < 1e-10, (gamma, lam)


def test_padding_and_chance_action_index_ignored():
    tokens = [(0, 2), (0, KIND_COMMIT), (1, KIND_PASS)]
    for with_rows in (True, False):
        garbage, _ = _make_batch_rollout([tokens], values_seed=3, with_rows=with_rows)
        clean, _ = _make_batch_rollout(
            [tokens], values_seed=3, with_rows=with_rows, garbage_padding=False,
        )
        segmentation_garbage = segment_rollout_macros(garbage)
        segmentation_clean = segment_rollout_macros(clean)
        assert torch.equal(segmentation_garbage.token_to_macro_id, segmentation_clean.token_to_macro_id)
        assert torch.equal(segmentation_garbage.chosen_is_commit, segmentation_clean.chosen_is_commit)
        assert torch.equal(segmentation_garbage.macro_is_last_for_player, segmentation_clean.macro_is_last_for_player)

        # chance positions inside num_steps with negative / oversized indices
        for bad_index in (-5, 999):
            chance, _ = _make_batch_rollout(
                [[(0, 2), (-1, None), (0, KIND_COMMIT)]], values_seed=3,
                with_rows=with_rows, garbage_padding=False,
            )
            chance["action_taken"][0, 1] = bad_index
            segmentation = segment_rollout_macros(chance)
            assert segmentation.num_macros[0] == 1
            assert segmentation.incomplete_count == 0


def test_active_action_index_out_of_range_is_rejected():
    for bad_index in (-1, 8, 999):
        for with_rows in (True, False):
            rollout, _ = _make_batch_rollout(
                [[(0, 2), (0, KIND_COMMIT)]], values_seed=3,
                with_rows=with_rows, garbage_padding=False,
            )
            rollout["action_taken"][0, 0] = bad_index
            with pytest.raises(ValueError, match="available-action range"):
                segment_rollout_macros(rollout)


def test_padding_macro_masks_are_false():
    traj_a = [(0, KIND_COMMIT)]
    traj_b = [(0, 2), (0, KIND_COMMIT), (1, KIND_PASS), (0, 2), (0, KIND_COMMIT)]
    rollout, _values = _make_batch_rollout([traj_a, traj_b], values_seed=5, pad=4)
    segmentation = segment_rollout_macros(rollout)
    assert segmentation.num_macros.tolist() == [1, 3]
    assert int(segmentation.max_macros) == 3

    assert bool(segmentation.macro_valid[0, 0])
    assert bool(segmentation.macro_is_last_for_player[0, 0])
    assert bool(segmentation.macro_is_terminal[0, 0])
    for macro in range(1, segmentation.max_macros):
        assert not bool(segmentation.macro_valid[0, macro])
        assert not bool(segmentation.macro_is_last_for_player[0, macro])
        assert not bool(segmentation.macro_is_terminal[0, macro])
        assert int(segmentation.macro_start_index[0, macro]) == -1
        assert int(segmentation.macro_next_index[0, macro]) == -1
        assert int(segmentation.macro_tau[0, macro]) == 0
        assert int(segmentation.macro_length[0, macro]) == 0
        assert int(segmentation.macro_player[0, macro]) == -1

    # trajectory B: player 0 owns macros 0 and 2, player 1 owns macro 1.
    last_for_player = [
        macro for macro in range(segmentation.max_macros)
        if bool(segmentation.macro_is_last_for_player[1, macro])
    ]
    assert last_for_player == [1, 2]
    terminal_flags = [
        macro for macro in range(segmentation.max_macros)
        if bool(segmentation.macro_is_terminal[1, macro])
    ]
    assert terminal_flags == [2]
    assert not any(
        bool(segmentation.macro_is_last_for_player[1, macro])
        for macro in range(segmentation.max_macros)
        if not bool(segmentation.macro_valid[1, macro])
    )


def test_long_macro_segmentation():
    lengths = [1, 8, 10, 16, 24]
    specs = [(index % P, lengths[index], KIND_COMMIT) for index in range(len(lengths))]
    tokens = _macro_spec_tokens(specs)
    rollout, values = _make_batch_rollout(
        [tokens], values_seed=31, terminal_rewards_per_traj=[[1, -1, 1, -1]],
    )
    segmentation = segment_rollout_macros(rollout)
    observed = [int(v) for v in segmentation.macro_length[0].tolist() if v > 0]
    assert observed == lengths
    _assert_macro_invariants(segmentation, rollout)
    oracle = legacy_gae_macro_oracle(rollout, values, discount_gamma=1.0, gae_lambda=0.95)
    assert oracle.max_abs_diff < 1e-10


def _set_padding(rollout, specs):
    """Overwrite padding positions with explicit (player_id, action_taken) pairs.

    ``specs[b]`` lists the values for positions ``num_steps[b]``, ``num_steps[b]+1``
    ...  Positions beyond the list keep whatever hostile garbage the builder
    already installed.
    """
    B, T = rollout["player_ids"].shape
    for b, padding in enumerate(specs):
        start = int(rollout["num_steps"][b])
        for offset, (player_id, action) in enumerate(padding):
            position = start + offset
            assert position < T, f"padding position {position} exceeds horizon {T}"
            rollout["player_ids"][b, position] = player_id
            rollout["action_taken"][b, position] = action


def _without_field(rollout, field):
    fields = {key: rollout[key] for key in rollout.keys()}
    del fields[field]
    return TensorDict(fields, batch_size=rollout.batch_size)


PADDING_EQUIVALENCE_FIELDS = (
    "num_macros",
    "token_to_macro_id",
    "macro_start_index",
    "macro_end_index",
    "macro_last_index",
    "macro_length",
    "macro_player",
    "macro_valid",
    "macro_next_index",
    "macro_tau",
    "macro_is_terminal",
    "macro_is_last_for_player",
    "macro_commit_kind",
)


def test_positive_padding_player_id_does_not_overflow_lookup():
    """Regression: padding player_id=3 while only players 0..2 are active.

    The old code sized the ``(player, ordinal)`` lookup from active ids but then
    indexed it with ``player_ids.clamp(min=0)`` for every ``[B, T]`` position,
    raising ``IndexError: index 3 is out of bounds for dimension 1 with size 3``.
    """
    for with_rows in (True, False):
        for with_behavior in (True, False):
            rollout, _values = _make_batch_rollout(
                [[(0, KIND_COMMIT), (1, KIND_COMMIT), (2, KIND_COMMIT)]],
                values_seed=41, with_rows=with_rows, garbage_padding=True,
            )
            _set_padding(rollout, [[(3, 999), (3, 999)]])
            if not with_behavior:
                rollout = _without_field(rollout, "behavior_action_prob")
            segmentation = segment_rollout_macros(rollout)
            assert segmentation.num_players == 3
            assert segmentation.num_macros.tolist() == [3]
            assert segmentation.macro_player[0].tolist() == [0, 1, 2]
            assert segmentation.token_to_macro_id[0].tolist()[:3] == [0, 1, 2]
            assert bool((segmentation.token_to_macro_id[0, 3:] == -1).all())
            assert segmentation.incomplete_count == 0
            assert segmentation.player_switch_violation_count == 0
            assert bool(segmentation.macro_valid[0].all())


def test_active_subset_with_larger_positive_padding_ids():
    """Active players only {0}; padding carries 3/99/999 and huge garbage ids."""
    for with_rows in (True, False):
        for with_behavior in (True, False):
            rollout, _values = _make_batch_rollout(
                [[(0, 2), (0, KIND_COMMIT)]], values_seed=43,
                with_rows=with_rows, garbage_padding=True,
            )
            _set_padding(rollout, [[(3, 111), (99, -222), (999, 2 ** 40 + 7)]])
            if not with_behavior:
                rollout = _without_field(rollout, "behavior_action_prob")
            segmentation = segment_rollout_macros(rollout)
            assert segmentation.num_players == 1
            assert segmentation.num_macros.tolist() == [1]
            assert segmentation.macro_player[0].tolist() == [0]
            assert segmentation.macro_length[0].tolist() == [2]
            assert segmentation.macro_start_index[0].tolist() == [0]
            assert segmentation.macro_end_index[0].tolist() == [1]
            assert bool((segmentation.token_to_macro_id[0, 2:] == -1).all())


def test_batch_mixed_active_coverage_positive_padding():
    """Batch A covers player 0 only; batch B covers players 0..3; both with 999 padding."""
    traj_a = [(0, KIND_COMMIT), (0, 2), (0, KIND_COMMIT)]
    traj_b = [(0, KIND_COMMIT), (1, KIND_COMMIT), (2, KIND_COMMIT), (3, KIND_COMMIT)]
    for with_rows in (True, False):
        rollout, _values = _make_batch_rollout(
            [traj_a, traj_b], values_seed=47, with_rows=with_rows, garbage_padding=True,
        )
        _set_padding(rollout, [[(999, 777)] * 4, [(999, 777), (99, 3)]])
        segmentation = segment_rollout_macros(rollout)
        assert segmentation.num_players == 4
        assert segmentation.num_macros.tolist() == [2, 4]
        assert segmentation.macro_player[0, :2].tolist() == [0, 0]
        assert segmentation.macro_player[1, :4].tolist() == [0, 1, 2, 3]
        assert bool((segmentation.token_to_macro_id[0, 3:] == -1).all())
        assert bool((segmentation.token_to_macro_id[1, 4:] == -1).all())
        assert not bool(segmentation.macro_valid[0, 2:].any())
        assert not bool(segmentation.macro_valid[1, 4:].any())
        assert segmentation.player_switch_violation_count == 0


def test_padding_player_id_logical_equivalence():
    """Padding pid=-2/action=0 and pid=999/action=999 must be bit-identical."""
    tokens = [
        (-2, None), (0, 2), (-1, None), (0, KIND_COMMIT),
        (1, KIND_PASS), (0, 2), (0, KIND_COMMIT),
    ]
    for with_rows in (True, False):
        for with_behavior in (True, False):
            rollout_a, _values = _make_batch_rollout(
                [tokens], values_seed=53, with_rows=with_rows, garbage_padding=True,
            )
            rollout_b, _values = _make_batch_rollout(
                [tokens], values_seed=53, with_rows=with_rows, garbage_padding=True,
            )
            _set_padding(rollout_a, [[(-2, 0), (-2, 0), (-2, 0)]])
            _set_padding(rollout_b, [[(999, 999), (99, -123456789), (3, 2 ** 33)]])
            if not with_behavior:
                rollout_a = _without_field(rollout_a, "behavior_action_prob")
                rollout_b = _without_field(rollout_b, "behavior_action_prob")
            seg_a = segment_rollout_macros(rollout_a)
            seg_b = segment_rollout_macros(rollout_b)
            assert seg_a.num_players == seg_b.num_players
            for field in PADDING_EQUIVALENCE_FIELDS:
                value_a = getattr(seg_a, field)
                value_b = getattr(seg_b, field)
                assert value_a is not None and value_b is not None, field
                assert torch.equal(value_a, value_b), field
            if with_behavior:
                assert seg_a.macro_old_logprob is not None
                assert torch.equal(seg_a.macro_old_logprob, seg_b.macro_old_logprob)
            else:
                assert seg_a.macro_old_logprob is None
                assert seg_b.macro_old_logprob is None
            assert seg_a.num_macros.tolist() == [3]
            assert seg_a.token_to_macro_id[0, :7].tolist() == [-1, 0, -1, 0, 1, 2, 2]


def test_active_illegal_action_index_rejected_with_positive_padding():
    for bad_index in (-1, 8, 999):
        for with_rows in (True, False):
            rollout, _values = _make_batch_rollout(
                [[(0, 2), (0, KIND_COMMIT)]], values_seed=3,
                with_rows=with_rows, garbage_padding=False,
            )
            _set_padding(rollout, [[(999, 123456)]])
            rollout["action_taken"][0, 0] = bad_index
            with pytest.raises(ValueError, match="available-action range"):
                segment_rollout_macros(rollout)


def test_inactive_garbage_action_index_ignored_with_positive_padding_ids():
    tokens = [(0, 2), (-1, None), (0, KIND_COMMIT)]
    for with_rows in (True, False):
        rollout, _values = _make_batch_rollout(
            [tokens], values_seed=59, with_rows=with_rows, garbage_padding=True,
        )
        _set_padding(rollout, [[(999, 10 ** 15), (3, -10 ** 15), (99, 123456789)]])
        for bad_index in (-123456789, 999, 2 ** 40):
            rollout["action_taken"][0, 1] = bad_index
            segmentation = segment_rollout_macros(rollout)
            assert segmentation.num_macros.tolist() == [1]
            assert segmentation.macro_player[0].tolist() == [0]
            assert segmentation.token_to_macro_id[0, :3].tolist() == [0, -1, 0]
            assert bool((segmentation.token_to_macro_id[0, 3:] == -1).all())
            assert segmentation.incomplete_count == 0


def test_active_player_id_is_not_clamped_toward_zero():
    """An active top-seat id must keep its real value, not be masked to 0..n-1."""
    rollout, _values = _make_batch_rollout(
        [[(3, KIND_COMMIT)]], values_seed=61, garbage_padding=True,
    )
    _set_padding(rollout, [[(0, 555), (0, 555)]])
    segmentation = segment_rollout_macros(rollout)
    assert segmentation.num_players == 4
    assert segmentation.macro_player[0].tolist() == [3]
    assert segmentation.token_to_macro_id[0].tolist()[0] == 0


def test_macro_old_logprob_and_engine_commit_match():
    specs = [(0, 3, KIND_COMMIT), (1, 1, KIND_PASS), (2, 4, KIND_COMMIT)]
    tokens = _macro_spec_tokens(specs)
    rollout, values = _make_rollout(tokens, terminal_rewards=[1, -1, 1, -1])
    segmentation = segment_rollout_macros(rollout)
    behavior = rollout["behavior_action_prob"][0].to(torch.float64)
    macro_index = 0
    offset = 0
    for (player, length, _kind) in specs:
        expected = behavior[offset:offset + length].log().sum().item()
        assert abs(segmentation.macro_old_logprob[0, macro_index].item() - expected) < 1e-12
        offset += length
        macro_index += 1
    diagnostics = macro_diagnostics(segmentation, engine_committed_steps=3)
    assert diagnostics["macro/engine_committed_steps"] == 3
    assert diagnostics["macro/commit_step_match"] == 1
    mismatch = macro_diagnostics(segmentation, engine_committed_steps=2)
    assert mismatch["macro/commit_step_match"] == 0
    assert diagnostics["macro/old_logprob_mean"] == pytest.approx(
        float(segmentation.macro_old_logprob[0, :3].mean().item()), abs=1e-12
    )
