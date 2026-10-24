"""Independent fixture builders and scalar references for phase-2 macro tests.

The builders only construct rollout tensors.  Every expected value used by the
tests is derived here with plain Python scalars over explicit token indices -
never by calling the production macro modules.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.training.macro_actions import MacroSegmentation, segment_macro_actions

P = 4
A = 4
DEFAULT_LEGAL = (0, 1)


def make_case(
    players: Sequence[int],
    commits: Sequence[bool],
    *,
    horizon: int | None = None,
    num_steps: int | None = None,
    rewards: Mapping[tuple[int, int], float] | None = None,
    values: Mapping[tuple[int, int], float] | None = None,
    behavior: Sequence[float] | None = None,
    policy_probs_override: Sequence[float] | None = None,
    on_policy: Sequence[bool] | None = None,
    legal: Sequence[Sequence[int]] | None = None,
    action_taken: Sequence[int] | None = None,
    avail_probs_rows: Sequence[Sequence[float]] | None = None,
    dtype: torch.dtype = torch.float64,
    garbage_padding: bool = True,
    batch: int = 1,
) -> tuple[TensorDict, torch.Tensor, MacroSegmentation, torch.Tensor]:
    """Build a rollout, action values, phase-1 segmentation and padded garbage.

    ``players``/``commits`` describe the real steps; padding after
    ``num_steps`` carries hostile garbage (positive player ids, large action
    index, non-zero rewards and values) that must never influence a macro.
    Returns ``(rollout, values, segmentation, avail_probs_new)``.
    """
    n = len(players)
    steps = n if num_steps is None else int(num_steps)
    assert steps <= n
    t_horizon = max(n, horizon if horizon is not None else n)
    length = t_horizon
    player_ids = torch.full((batch, length), 3, dtype=torch.long)
    action = torch.zeros((batch, length), dtype=torch.long)
    avail_weights = torch.zeros((batch, length, A), dtype=torch.float32)
    avail_is_commit = torch.zeros((batch, length, A), dtype=torch.bool)
    behavior_prob = torch.full((batch, length), 0.125, dtype=dtype)
    policy_probs = torch.zeros((batch, length, A), dtype=dtype)
    on_policy_valid = torch.zeros((batch, length), dtype=torch.bool)
    rew = torch.zeros((batch, length, P), dtype=dtype)
    val = torch.zeros((batch, length, P), dtype=dtype)
    terminal = torch.zeros((batch, length), dtype=torch.bool)
    new_probs = torch.zeros((batch, length, A), dtype=torch.float32)

    for b in range(batch):
        for t in range(steps):
            player = int(players[t])
            player_ids[b, t] = player
            legal_actions = tuple(DEFAULT_LEGAL if legal is None else legal[t])
            chosen = int(action_taken[t]) if action_taken is not None else (0 if commits[t] else 1)
            if chosen not in legal_actions:
                raise ValueError("chosen action must be legal in the fixture")
            for a in legal_actions:
                avail_weights[b, t, a] = 1.0
                avail_is_commit[b, t, a] = bool(commits[t]) if a == chosen else not bool(commits[t])
            action[b, t] = chosen
            prob = float(behavior[t]) if behavior is not None else 0.5
            behavior_prob[b, t] = prob
            selected_reference = (
                float(policy_probs_override[t]) if policy_probs_override is not None else prob
            )
            policy_probs[b, t, chosen] = selected_reference
            others = [a for a in legal_actions if a != chosen]
            if others:
                share = (1.0 - selected_reference) / len(others)
                for a in others:
                    policy_probs[b, t, a] = share
            on_policy_valid[b, t] = True if on_policy is None else bool(on_policy[t])
            row = [0.0] * A
            row[chosen] = prob
            if others:
                share = (1.0 - prob) / len(others)
                for a in others:
                    row[a] = share
            if avail_probs_rows is not None:
                row = [float(v) for v in avail_probs_rows[t]]
            new_probs[b, t] = torch.tensor(row, dtype=torch.float32)
        terminal[b, max(steps - 1, 0)] = True
        if garbage_padding:
            for t in range(steps, length):
                player_ids[b, t] = 2 + (t % 2)
                action[b, t] = A - 1
                behavior_prob[b, t] = 0.25
                policy_probs[b, t, 0] = 1.0
                rew[b, t, :] = 5.0
                val[b, t, :] = 9.0
    for (t, p), value in (rewards or {}).items():
        rew[:, t, p] = value
    for (t, p), value in (values or {}).items():
        val[:, t, p] = value

    rollout = TensorDict(
        {
            "num_steps": torch.tensor([steps] * batch, dtype=torch.long),
            "player_ids": player_ids,
            "action_taken": action,
            "avail_weights": avail_weights,
            "avail_is_commit": avail_is_commit,
            "behavior_action_prob": behavior_prob,
            "policy_probs": policy_probs,
            "on_policy_valid": on_policy_valid,
            "rewards": rew,
            "is_terminal": terminal,
        },
        batch_size=[batch],
    )
    segmentation = segment_macro_actions(
        num_steps=rollout["num_steps"],
        player_ids=player_ids,
        action_taken=action,
        avail_is_commit=avail_is_commit,
        behavior_action_prob=behavior_prob,
        is_terminal=terminal,
    )
    return rollout, val, segmentation, new_probs


def macro_trainable(segmentation: MacroSegmentation) -> torch.Tensor:
    return segmentation.macro_valid & (segmentation.macro_end_index >= 0)


def macro_slots_by_start(segmentation: MacroSegmentation) -> dict[int, int]:
    """Player-agnostic map ``start token -> macro slot id``."""
    return {
        int(segmentation.macro_start_index[0, m]): m
        for m in range(segmentation.max_macros)
        if bool(segmentation.macro_valid[0, m])
    }


# ---------------------------------------------------------------------------
# Independent scalar references (plain Python, explicit token loops)
# ---------------------------------------------------------------------------


def scalar_token_gae(
    players: Sequence[int],
    num_steps: int,
    rewards: Mapping[tuple[int, int], float],
    values: Mapping[tuple[int, int], float],
    *,
    gamma: float,
    lam: float,
) -> dict[tuple[int, int], float]:
    """Legacy token GAE for every acting-player token: ``(token, player) -> A``."""
    per_player: dict[int, list[int]] = {}
    for t in range(num_steps):
        per_player.setdefault(int(players[t]), []).append(t)
    advantage: dict[tuple[int, int], float] = {}
    for player, tokens in per_player.items():
        for index in range(len(tokens) - 1, -1, -1):
            start = tokens[index]
            if index + 1 < len(tokens):
                stop = tokens[index + 1]
            else:
                stop = num_steps
            reward_sum = sum(
                (gamma ** (j - start)) * rewards.get((j, player), 0.0)
                for j in range(start, stop)
            )
            bootstrap = (
                (gamma ** (stop - start)) * values.get((stop, player), 0.0)
                if index + 1 < len(tokens)
                else 0.0
            )
            delta = reward_sum + bootstrap - values.get((start, player), 0.0)
            successor = advantage.get((stop, player), 0.0) if index + 1 < len(tokens) else 0.0
            advantage[(start, player)] = delta + (gamma ** (stop - start)) * lam * successor
    return advantage


def scalar_macro_gae(
    players: Sequence[int],
    num_steps: int,
    rewards: Mapping[tuple[int, int], float],
    values: Mapping[tuple[int, int], float],
    *,
    gamma: float,
    lam: float,
    macro_starts: Sequence[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], float]:
    """Player macro-decision GAE: ``(macro start, player) -> A``.

    ``macro_starts`` pins the explicit ``(start token, player)`` macro decisions
    (from the accepted phase-1 segmentation).  When omitted every acting token
    is treated as a length-1 macro.
    """
    per_player: dict[int, list[int]] = {}
    if macro_starts is None:
        for t in range(num_steps):
            per_player.setdefault(int(players[t]), []).append(t)
    else:
        for start, player in macro_starts:
            per_player.setdefault(int(player), []).append(int(start))
        for player in per_player:
            per_player[player] = sorted(per_player[player])
    advantage: dict[tuple[int, int], float] = {}
    for player, tokens in per_player.items():
        for index in range(len(tokens) - 1, -1, -1):
            start = tokens[index]
            has_successor = index + 1 < len(tokens)
            stop = tokens[index + 1] if has_successor else num_steps
            duration = stop - start
            reward_sum = sum(
                (gamma ** (j - start)) * rewards.get((j, player), 0.0)
                for j in range(start, stop)
            )
            continuation = 1.0 if has_successor else 0.0
            delta = (
                reward_sum
                + (gamma ** duration) * continuation * values.get((stop, player), 0.0)
                - values.get((start, player), 0.0)
            )
            successor = advantage.get((stop, player), 0.0) if has_successor else 0.0
            advantage[(start, player)] = delta + (gamma ** duration) * lam * continuation * successor
    return advantage


def scalar_macro_pg_loss(
    macro_log_ratios: Sequence[float],
    raw_advantages: Sequence[float],
    *,
    clip_coef: float,
    valid: Sequence[bool] | None = None,
) -> float:
    """Textbook macro PPO mean: normalize (population), clip once, mean per macro."""
    mask = list(valid) if valid is not None else [True] * len(raw_advantages)
    values = [a for a, keep in zip(raw_advantages, mask) if keep]
    ratios = [math.exp(lr) for lr, keep in zip(macro_log_ratios, mask) if keep]
    if not values:
        return float("nan")
    count = len(values)
    mean = sum(values) / count
    variance = sum((a - mean) ** 2 for a in values) / count
    std = math.sqrt(variance)
    normalized = [(a - mean) / (std + 1e-8) if std > 0 else 0.0 for a in values]
    total = 0.0
    for advantage, ratio in zip(normalized, ratios):
        clipped = min(max(ratio, 1.0 - clip_coef), 1.0 + clip_coef)
        total += max(-advantage * ratio, -advantage * clipped)
    return total / count


def scalar_token_pg_loss(
    token_log_ratios: Sequence[float],
    token_advantages: Sequence[float],
    *,
    clip_coef: float,
) -> float:
    """Textbook token PPO mean with population normalization on active tokens."""
    count = len(token_advantages)
    mean = sum(token_advantages) / count
    variance = sum((a - mean) ** 2 for a in token_advantages) / count
    std = math.sqrt(variance)
    normalized = [(a - mean) / (std + 1e-8) if std > 0 else 0.0 for a in token_advantages]
    total = 0.0
    for advantage, log_ratio in zip(normalized, token_log_ratios):
        ratio = math.exp(log_ratio)
        clipped = min(max(ratio, 1.0 - clip_coef), 1.0 + clip_coef)
        total += max(-advantage * ratio, -advantage * clipped)
    return total / count
