"""Macro boundary invariants and legacy oracle on real C++ environments."""
from __future__ import annotations

import numpy as np
import torch

from guandan_marvel.training.legacy_gae import legacy_gae_macro_oracle
from guandan_marvel.training.macro_actions import (
    KIND_COMMIT,
    KIND_PASS,
    macro_diagnostics,
    segment_macro_actions,
)
from nanopokerlib import guandan as cpp

PAD = 7


def _play_random_game(seed: int, *, rng_seed: int):
    batch = cpp.EnvBatch(1, auto_reset=False)
    step = batch.reset(seed)
    rng = np.random.default_rng(rng_seed)
    player_ids, action_taken, avail_is_commit, avail_tokens = [], [], [], []
    rewards, terminal = [], []
    while True:
        weights = step.available_actions.weights[0]
        legal = np.flatnonzero(weights > 0)
        choice = int(rng.choice(legal))
        player_ids.append(int(step.player_ids[0]))
        action_taken.append(choice)
        avail_is_commit.append(step.available_actions.is_commit[0].copy())
        avail_tokens.append(step.available_actions.tokens[0].copy())
        rows = step.available_actions.tokens[0, choice][None, :].astype(np.int64)
        next_step = batch.step(rows)
        rewards.append(next_step.rewards[0].copy())
        terminal.append(bool(next_step.is_terminal[0]))
        if next_step.is_terminal[0]:
            info = dict(next_step.infos[0])
            break
        step = next_step
    horizon = len(player_ids)
    trajectory = {
        "num_steps": horizon,
        "player_ids": np.asarray(player_ids, dtype=np.int64),
        "action_taken": np.asarray(action_taken, dtype=np.int64),
        "avail_is_commit": np.stack(avail_is_commit, axis=0),
        "avail_tokens": np.stack(avail_tokens, axis=0),
        "rewards": np.stack(rewards, axis=0).astype(np.float32),
        "is_terminal": np.asarray(terminal, dtype=np.bool_),
    }
    return trajectory, info


def _to_tensors(trajectory, *, seed: int):
    steps = trajectory["num_steps"]
    T = steps + PAD
    A = trajectory["avail_is_commit"].shape[1]
    device = "cpu"
    num_steps = torch.tensor([steps], dtype=torch.long, device=device)
    player_ids = torch.zeros((1, T), dtype=torch.long, device=device)
    action_taken = torch.zeros((1, T), dtype=torch.long, device=device)
    avail_is_commit = torch.zeros((1, T, A), dtype=torch.bool, device=device)
    avail_tokens = torch.zeros((1, T, A, 8), dtype=torch.long, device=device)
    rewards = torch.zeros((1, T, 4), dtype=torch.float32, device=device)
    is_terminal = torch.zeros((1, T), dtype=torch.bool, device=device)
    player_ids[0, :steps] = torch.as_tensor(trajectory["player_ids"], dtype=torch.long)
    action_taken[0, :steps] = torch.as_tensor(trajectory["action_taken"], dtype=torch.long)
    avail_is_commit[0, :steps] = torch.as_tensor(trajectory["avail_is_commit"], dtype=torch.bool)
    avail_tokens[0, :steps] = torch.as_tensor(trajectory["avail_tokens"], dtype=torch.long)
    rewards[0, :steps] = torch.as_tensor(trajectory["rewards"], dtype=torch.float32)
    is_terminal[0, :steps] = torch.as_tensor(trajectory["is_terminal"], dtype=torch.bool)
    # Garbage padding outside num_steps must never be interpreted as a macro.
    player_ids[0, steps:] = 2
    action_taken[0, steps:] = A - 1
    avail_is_commit[0, steps:, 0] = False
    behavior = torch.full((1, T), 0.25, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((1, T, 4), generator=generator, dtype=torch.float32) * 0.1
    for t in range(T):
        player = int(player_ids[0, t])
        for q in range(4):
            if t >= steps or player < 0 or q != player:
                values[0, t, q] = 0.0
    return {
        "num_steps": num_steps,
        "player_ids": player_ids,
        "action_taken": action_taken,
        "avail_is_commit": avail_is_commit,
        "avail_tokens": avail_tokens,
        "behavior_action_prob": behavior,
        "is_terminal": is_terminal,
        "rewards": rewards,
    }, values


def _segmentation_kwargs(fields):
    return {key: value for key, value in fields.items() if key != "rewards"}


def _rollout_tensordict(fields, device="cpu"):
    from guandan_marvel.model.tensordict import TensorDict

    return TensorDict(
        {
            "num_steps": fields["num_steps"],
            "player_ids": fields["player_ids"],
            "action_taken": fields["action_taken"],
            "avail_is_commit": fields["avail_is_commit"],
            "rewards": fields["rewards"],
        },
        batch_size=[1],
        device=device,
    )


def test_real_cpp_games_macro_boundaries_and_commit_counts():
    for game_index, seed in enumerate((11, 2202, 33033, 440044)):
        trajectory, info = _play_random_game(seed, rng_seed=seed + 7)
        fields, values = _to_tensors(trajectory, seed=seed)
        segmentation = segment_macro_actions(**_segmentation_kwargs(fields))
        steps = trajectory["num_steps"]

        assert segmentation.incomplete_count == 0
        assert segmentation.player_switch_violation_count == 0
        assert not bool(segmentation.token_to_macro_id[0, steps:].ge(0).any())
        assert int(segmentation.num_macros[0]) > 0
        committed = int((segmentation.macro_end_index >= 0).sum().item())
        assert committed == int(info["finish_committed_step"])
        selected_commits = int(segmentation.chosen_is_commit.sum().item())
        assert selected_commits == committed
        assert int(segmentation.macro_is_terminal[0].sum().item()) == 1

        player_ids = fields["player_ids"]
        for macro in range(int(segmentation.num_macros[0])):
            start = int(segmentation.macro_start_index[0, macro])
            end = int(segmentation.macro_end_index[0, macro])
            player = int(segmentation.macro_player[0, macro])
            tokens = segmentation.token_to_macro_id[0] == macro
            assert bool((player_ids[0][tokens] == player).all())
            assert bool(segmentation.chosen_is_commit[0, end])
            assert int(segmentation.macro_length[0, macro]) == int(tokens.sum().item())
            kind = int(segmentation.macro_commit_kind[0, macro])
            assert kind in (1, 2)
            if kind == 1:
                assert start == end
        diagnostics = macro_diagnostics(
            segmentation, engine_committed_steps=int(info["finish_committed_step"]),
        )
        assert diagnostics["macro/commit_step_match"] == 1
        assert diagnostics["macro/pass_count"] + diagnostics["macro/commit_count"] == committed

        oracle = legacy_gae_macro_oracle(
            _rollout_tensordict(fields), values, discount_gamma=1.0, gae_lambda=0.95,
        )
        assert oracle.max_abs_diff < 1e-6, (game_index, oracle.max_abs_diff)
        assert diagnostics["macro/player_switch_violation_count"] == 0
        assert diagnostics["macro/incomplete_count"] == 0


def test_real_cpp_macro_kind_decode_uses_real_rows():
    trajectory, _info = _play_random_game(12321, rng_seed=5)
    fields, _values = _to_tensors(trajectory, seed=12321)
    segmentation = segment_macro_actions(**_segmentation_kwargs(fields))
    kinds = set(int(v) for v in segmentation.macro_commit_kind[0].tolist() if v > 0)
    # A random game contains both PASS and COMMIT kinds; the decoder must not
    # collapse them to the length-1 fallback.
    assert kinds == {1, 2}
    assert KIND_PASS == 0 and KIND_COMMIT == 1
    for macro in range(int(segmentation.num_macros[0])):
        end = int(segmentation.macro_end_index[0, macro])
        action = int(fields["action_taken"][0, end])
        row = fields["avail_tokens"][0, end, action]
        kind_code = int(row[3]) - 296
        expected = 1 if kind_code == KIND_PASS else 2
        assert int(segmentation.macro_commit_kind[0, macro]) == expected
