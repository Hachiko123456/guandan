"""CPU protocol doubles isolate auto-reset indexing without neural training."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from guandan_marvel.phase5 import belief_collect as collect


def install_protocol_doubles(monkeypatch, slots, *, slow_steps=8):
    from nanopokerlib import guandan as cpp
    from guandan_marvel import model

    state = SimpleNamespace(env=None)

    class FakeCache:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, mask):
            pass

    class FakeEnv:
        def __init__(self, batch_size, *, auto_reset):
            assert batch_size == slots and auto_reset
            self.counter = 0
            self.remaining = [1 if slot == 0 else slow_steps for slot in range(slots)]
            self.contexts = [None] * slots
            state.env = self

        def configure_context_schedule(self, schedule, offset):
            self.schedule = schedule
            assert offset == 0

        def new_deal(self, slot):
            # Native next_context_index wraps the finite schedule. Out-of-range
            # slots can reset many times while another requested deal is slow.
            self.contexts[slot] = self.schedule[self.counter % len(self.schedule)]
            self.counter += 1

        def view(self, terminal=None):
            return SimpleNamespace(
                player_ids=np.zeros(slots, dtype=np.int64),
                observations=SimpleNamespace(
                    tokens=np.zeros((slots, 4, 1), dtype=np.int64),
                    channels=np.zeros((slots, 4, 512), dtype=np.uint8),
                ),
                available_actions=SimpleNamespace(
                    tokens=np.zeros((slots, 2, 8), dtype=np.int64),
                    weights=np.ones((slots, 2), dtype=np.float32),
                ),
                is_terminal=np.zeros(slots, dtype=bool) if terminal is None else terminal,
                rewards=np.zeros((slots, 4), dtype=np.float32),
                infos=[{} for _ in range(slots)],
            )

        def reset(self, seed):
            for slot in range(slots):
                self.new_deal(slot)
            return self.view()

        def debug_state(self, slot):
            ctx = self.contexts[slot]
            return {
                "prefix_tokens": [],
                "config": {"level_rank": ctx["level_rank"]},
                "state": {"hands": [list(range(i * 27, (i + 1) * 27)) for i in range(4)]},
            }

        def step(self, rows):
            terminal = np.zeros(slots, dtype=bool)
            for slot in range(slots):
                self.remaining[slot] -= 1
                if self.remaining[slot] == 0:
                    terminal[slot] = True
                    self.remaining[slot] = 1 if slot == 0 else slow_steps
                    self.new_deal(slot)
            return self.view(terminal)

    class FakePolicy:
        decoder = SimpleNamespace(config=SimpleNamespace(
            hidden_size=4, num_attention_heads=1, num_key_value_heads=1,
        ))
        decoder_n_layers = 1
        max_game_horizon = 32

        def __call__(self, *, rollout, **kwargs):
            return SimpleNamespace(avail_probs=torch.full((slots, 1, 2), 0.5))

    def sample(slot, step, debug, meta, maximum, *, ledger):
        if meta["samples"] >= maximum:
            return None
        meta["samples"] += 1
        return SimpleNamespace(deal_index=meta["deal_index"], group_id=meta["group_id"])

    monkeypatch.setattr(cpp, "EnvBatch", FakeEnv)
    monkeypatch.setattr(model, "KVCache", FakeCache)
    monkeypatch.setattr(collect, "_record_sample", sample)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    return FakePolicy(), state


@pytest.mark.parametrize("start,deals,slots", [(0, 2, 2), (100, 5, 2), (411, 8, 3)])
def test_absolute_indices_and_scheduled_contexts_survive_staggered_auto_resets(
    monkeypatch, start, deals, slots,
):
    policy, state = install_protocol_doubles(monkeypatch, slots, slow_steps=15)
    samples, metadata, info = collect.collect_belief_dataset(
        policy=policy, device=torch.device("cpu"), deals=deals, base_seed=7,
        max_macro_starts_per_deal=2, max_wall_seconds=10,
        start_index=start, batch_slots=slots,
    )
    assert sorted(row["deal_index"] for row in metadata) == list(range(start, start + deals))
    assert len(metadata) == info["finished_deals"] == deals
    assert not info["truncated"] and all(row["finished"] for row in metadata)
    assert {sample.deal_index for sample in samples} == set(range(start, start + deals))
    assert all(row["context"] == collect.context_for(row["deal_index"]) for row in metadata)
    assert all(start <= row.deal_index < start + deals for row in samples)
    if slots < deals:
        assert state.env.counter > deals + slots  # actually exercised wrapped padding


@pytest.mark.parametrize("updates", [
    {"deals": 0}, {"deals": -1}, {"deals": 1.5}, {"deals": True},
    {"batch_slots": 0}, {"batch_slots": -1}, {"batch_slots": 1.5},
    {"start_index": -1}, {"max_macro_starts_per_deal": 0},
    {"max_wall_seconds": 0}, {"max_wall_seconds": float("nan")},
    {"max_wall_seconds": float("inf")},
])
def test_invalid_collection_budget_fails_before_policy_or_engine_initialization(updates):
    arguments = dict(policy=None, device=torch.device("cpu"), deals=1, base_seed=7,
                     max_macro_starts_per_deal=1, max_wall_seconds=1,
                     start_index=0, batch_slots=1)
    arguments.update(updates)
    with pytest.raises(ValueError):
        collect.collect_belief_dataset(**arguments)


def test_timeout_reports_only_requested_partial_deals_without_ghost_samples(monkeypatch):
    policy, state = install_protocol_doubles(monkeypatch, 2, slow_steps=15)
    clock = iter([0., 0., 100., 100.])
    monkeypatch.setattr(collect, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    samples, metadata, info = collect.collect_belief_dataset(
        policy=policy, device=torch.device("cpu"), deals=2, base_seed=7,
        max_macro_starts_per_deal=2, max_wall_seconds=1, start_index=0, batch_slots=2,
    )
    assert info["truncated"] is True and info["finished_deals"] == 1
    assert sorted(row["deal_index"] for row in metadata) == [0, 1]
    assert {sample.deal_index for sample in samples} == {0, 1}
    assert [row["deal_index"] for row in metadata if not row["finished"]] == [1]
    assert all(row["samples"] == 1 for row in metadata)
