"""Phase-5 symmetry consistency: actor/observer alignment counterexamples.

Every test in this module fails on the pre-fix implementation, which
transformed the four observer rows but left ``player_ids`` (and every step of
the causal prefix) untouched:

* ``Agent.forward`` gathers the acting state/action row with ``player_ids``, so
  the transformed branch was scored for the wrong observer;
* a per-flat-cell transform spliced different symmetries into one row history;
* only loss-bearing cells were transformed, so a non-loss step inside the
  visible prefix stayed in the original frame.

The expected values below are hand-computed from the documented V2 layout
(``guandan_marvel/phase5/symmetry.py``): seat ``s -> (s + k) mod 4`` writes
observer ``o`` into row ``(o + k) mod 4``, permutes the seat-indexed channel
slices, rotates the event seat fields and keeps observer-relative hand bits in
the observer's own row.  The tests never call a production helper to decide
what the expected value is.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from guandan_marvel import python_game
from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.model import Agent, TensorDict
from guandan_marvel.model.base import BaseMarvelModule
from guandan_marvel.phase5.design import phase5_design_hash
from guandan_marvel.phase5.symmetry import (
    ACTIVE_ONE_HOT, BOS_TOKEN, CARD_TOKEN_BASE, EVENT_END, EVENT_RESET,
    LEADER_ONE_HOT, LENGTH_TOKEN_BASE, LEVEL_ONE_HOT, OBSERVER_ONE_HOT,
    PHASE_ONE_HOT, RANK_TOKEN_BASE, REMAINING_COUNTS, SELF_HAND, SELF_PREFIX,
    SINGLE_BITS, TEAM_OF, TEAM_ONE_HOT,
)
from guandan_marvel.phase5.symmetry_training import (
    SymmetryConsistency, canonical_hash, load_sealed_symmetry_artifact,
)

B, T, P, H, C, A, L = 2, 4, 4, 48, 512, 3, 8


# ---------------------------------------------------------------------------
# Sealed contract plus a tiny real model
# ---------------------------------------------------------------------------


def _tiny_agent() -> Agent:
    cfg = TrainConfig(
        game=python_game.GAME_SPEC, device="cpu",
        rollout_batch_size=2, collect_batch_size=1,
        max_sampler_bytes_per_rank=512 * 2**20,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.optimizer.use_muon = False
    torch.manual_seed(7)
    return Agent(cfg, enable_value_head=True, enable_ntp_head=False)


def _sealed_contract(tmp_path: Path, agent: Agent, transforms: list[str]):
    checkpoint = tmp_path / "init.pt"
    checkpoint.write_bytes(b"phase5-symmetry-actor-alignment-checkpoint")
    document = {
        "schema": "guandan-phase5-symmetry-diagnostic-run-v1",
        "status": "COMPLETED",
        "training": False,
        "truncated": False,
        "deals_requested": 16,
        "design_hash": phase5_design_hash(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_policy_identity": {
            "architecture": "v1.3",
            "algorithm": "ippo",
            "history_dim": 16,
            "feature_dim": 24,
            "parameter_count": sum(
                value.numel() for value in agent.state_dict().values()
            ),
            "state_dict_keys": len(agent.state_dict()),
        },
        "decision": {
            "status": "SIGNIFICANT",
            "augmentation_recommended": True,
            "sample_count": 1536,
            "deal_count": 16,
            "non_finite_kl_count": 0,
            "mean_kl": 0.02857,
            "argmax_disagreement_rate": 0.11263,
            "thresholds": {
                "min_macro_samples": 256,
                "min_deals": 8,
                "mean_restricted_kl_threshold": 0.02,
                "argmax_disagreement_threshold": 0.10,
            },
        },
        "diagnostic": {
            "transforms": sorted(transforms),
            "aggregate": {
                "mean_kl": 0.02857, "median_kl": 0.02, "max_kl": 0.2,
                "non_finite_kl_count": 0,
            },
        },
    }
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(document), encoding="utf-8")
    seal = {
        "schema": "guandan-marvel-phase5-symmetry-artifact-seal-v1",
        "artifact_path": str(artifact),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "content_hash": canonical_hash(document),
        "design_hash": document["design_hash"],
        "checkpoint_sha256": document["checkpoint_sha256"],
        "source_policy_identity": document["source_policy_identity"],
        "transforms": list(transforms),
    }
    seal["seal_hash"] = canonical_hash(seal)
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    return load_sealed_symmetry_artifact(seal_path)


# ---------------------------------------------------------------------------
# Hand-built observations
# ---------------------------------------------------------------------------

HAND_CARD = (0, 11, 22, 33)  # unique physical card per observer


def _observation(active_seat: int, leader_seat: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """One V2 observation for all four observers.

    ``SELF_HAND``/``SELF_PREFIX`` are observer-relative and stay in the
    observer's own row; ``REMAINING_COUNTS``, active/leader one-hots and the
    reset event's seat field are absolute seats and must be rotated.
    """
    tokens = np.zeros((P, H), dtype=np.int64)
    channels = np.zeros((P, C), dtype=np.uint8)
    for observer in range(P):
        tokens[observer, 0] = BOS_TOKEN
        tokens[observer, 1] = EVENT_RESET
        tokens[observer, 2] = 2
        tokens[observer, 3] = RANK_TOKEN_BASE + 12
        tokens[observer, 4] = LENGTH_TOKEN_BASE + observer
        tokens[observer, 5] = EVENT_END
        channels[observer, OBSERVER_ONE_HOT.start + observer] = 1
        channels[observer, ACTIVE_ONE_HOT.start + active_seat] = 1
        channels[observer, LEADER_ONE_HOT.start + leader_seat] = 1
        channels[observer, PHASE_ONE_HOT.start] = 1
        channels[observer, LEVEL_ONE_HOT.start + 4] = 1
        channels[observer, TEAM_ONE_HOT.start + TEAM_OF[observer]] = 1
        channels[observer, REMAINING_COUNTS.start + observer] = 10 + observer
        channels[observer, SELF_HAND.start + HAND_CARD[observer]] = 1
        channels[observer, SELF_PREFIX.start] = CARD_TOKEN_BASE + 20 + observer
        channels[observer, SINGLE_BITS["sentinel"]] = 1
        # Per-step marker (a prefix token is observer-relative and untouched by
        # the seat rotation); used to prove the whole prefix moved together.
        channels[observer, SELF_PREFIX.start + 1] = 40 + 4 * step + observer
    return tokens, channels


def _candidate_rows() -> np.ndarray:
    rows = np.zeros((A, L), dtype=np.int64)
    for index, token in enumerate((6, CARD_TOKEN_BASE + 3, CARD_TOKEN_BASE + 57)):
        rows[index] = (262, 2, 263, 296, 304, token, 306, 307)
    return rows


def _rollout(
    player_ids: list[list[int]],
    num_steps: list[int],
    legal_counts: list[list[int]] | None = None,
) -> TensorDict:
    b = len(player_ids)
    assert b > 0
    steps = len(player_ids[0])
    assert all(len(row) == steps for row in player_ids)
    assert len(num_steps) == b
    assert all(0 <= int(value) <= steps for value in num_steps)
    if legal_counts is None:
        legal_counts = [[3] * steps for _ in range(b)]
    assert len(legal_counts) == b
    assert all(len(row) == steps for row in legal_counts)
    obs_tokens = np.zeros((b, steps, P, H), dtype=np.int64)
    obs_channels = np.zeros((b, steps, P, C), dtype=np.uint8)
    for row in range(b):
        for step in range(steps):
            active_seat = int(player_ids[row][step])
            if active_seat < 0:
                continue
            tokens, channels = _observation(active_seat, (active_seat + 2) % P, step)
            obs_tokens[row, step] = tokens
            obs_channels[row, step] = channels
    avail = np.zeros((b, steps, A, L), dtype=np.int64)
    weights = np.zeros((b, steps, A), dtype=np.float32)
    rows = _candidate_rows()
    for row in range(b):
        for step in range(steps):
            avail[row, step] = rows
            weights[row, step, : int(legal_counts[row][step])] = 1.0
    return TensorDict({
        "num_steps": torch.tensor(num_steps, dtype=torch.long),
        "player_ids": torch.tensor(player_ids, dtype=torch.long),
        "obs_tokens": torch.from_numpy(obs_tokens),
        "obs_channels": torch.from_numpy(obs_channels),
        "avail_tokens": torch.from_numpy(avail),
        "avail_weights": torch.from_numpy(weights),
    }, batch_size=[b])


class _SpyForward:
    """Capture the TensorDict the consistency branch actually forwards."""

    def __init__(self, log_probs: torch.Tensor) -> None:
        self.log_probs = log_probs.detach().clone()
        self.batch = None
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1
        self.batch = batch
        # A shape-compatible placeholder: the loss only needs a finite
        # ``avail_log_probs`` here; the gather tests re-run the real model.
        return SimpleNamespace(avail_log_probs=self.log_probs.clone())


def _run_loss(agent, consistency, rollout):
    base = agent(rollout)
    spy = _SpyForward(base.avail_log_probs)
    result = consistency.loss(rollout, base_log_probs=base.avail_log_probs, forward=spy)
    assert spy.calls == 1
    return spy.batch, result


# ---------------------------------------------------------------------------
# 1. actor/gather alignment (fails on the old implementation)
# ---------------------------------------------------------------------------


def test_transformed_player_ids_and_actor_gather_follow_seat_rotation(tmp_path, monkeypatch):
    agent = _tiny_agent()
    contract = _sealed_contract(tmp_path, agent, ["seat_rotation_1"])
    consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
    # Step 0 is decided by seat 0, step 1 by seat 2.
    rollout = _rollout(player_ids=[[0, 2, -1, -1]], num_steps=[2])
    transformed, _ = _run_loss(agent, consistency, rollout)

    # T(seat) = (seat + 1) mod 4 on the active cells only; padding stays -1.
    assert transformed["player_ids"].tolist() == [[1, 3, -1, -1]]
    # Rows moved: observer o now lives in row o + 1.
    for observer in range(P):
        target = (observer + 1) % P
        # Event seat field rotated by the same T.
        assert int(transformed["obs_tokens"][0, 0, target, 4]) == LENGTH_TOKEN_BASE + target
        # Absolute seat slices rotated.
        assert int(transformed["obs_channels"][0, 0, target, REMAINING_COUNTS.start + target]) == 10 + observer
        assert int(transformed["obs_channels"][0, 0, target, OBSERVER_ONE_HOT.start + target]) == 1
        assert int(transformed["obs_channels"][0, 0, target, ACTIVE_ONE_HOT.start + 1]) == 1
        # Observer-relative hand stays in the observer's own (moved) row.
        assert int(transformed["obs_channels"][0, 0, target, SELF_HAND.start + HAND_CARD[observer]]) == 1
        # Teams are recomputed: row target now plays team TEAM_OF[target].
        assert int(transformed["obs_channels"][0, 0, target, TEAM_ONE_HOT.start + TEAM_OF[target]]) == 1
        assert int(transformed["obs_channels"][0, 0, target, TEAM_ONE_HOT.start + TEAM_OF[observer]]) == 0

    # The real model must gather the *transformed* actor row.  Capture the
    # channel row the model actually consumes.
    captured: dict[str, torch.Tensor] = {}
    original = BaseMarvelModule._encode_state_features

    def spy(self, decoded_feats, obs_channels, time_mask, **kwargs):
        captured["channels"] = obs_channels.detach().clone()
        return original(self, decoded_feats, obs_channels, time_mask, **kwargs)

    monkeypatch.setattr(BaseMarvelModule, "_encode_state_features", spy)
    agent(transformed)

    acting = captured["channels"]
    # t=0: actor 0 -> gathered row 1, so REMAINING[1] == 10 and hand card 0.
    assert int(acting[0, 0, REMAINING_COUNTS.start + 1]) == 10
    assert int(acting[0, 0, SELF_HAND.start + HAND_CARD[0]]) == 1
    assert int(acting[0, 0, SELF_HAND.start + HAND_CARD[3]]) == 0
    # t=1: actor 2 -> gathered row 3, so REMAINING[3] == 12 and hand card 2.
    # The old gather (player_ids unchanged) would instead read row 3 as
    # observer 2's row, i.e. REMAINING[3] == 11 / hand card 1.
    assert int(acting[0, 1, REMAINING_COUNTS.start + 3]) == 12
    assert int(acting[0, 1, SELF_HAND.start + HAND_CARD[2]]) == 1
    assert int(acting[0, 1, SELF_HAND.start + HAND_CARD[1]]) == 0


# ---------------------------------------------------------------------------
# 2. one transform per row, whole causal prefix
# ---------------------------------------------------------------------------


def test_each_row_uses_one_transform_for_its_whole_prefix(tmp_path):
    agent = _tiny_agent()
    contract = _sealed_contract(
        tmp_path, agent, ["seat_rotation_1", "seat_rotation_2"],
    )
    consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
    # Row 0 selects transforms[0] = +1; row 1 selects transforms[1] = +2.
    rollout = _rollout(
        player_ids=[[0, 1, 2, -1], [2, 3, 0, -1]],
        num_steps=[3, 3],
    )
    transformed, _ = _run_loss(agent, consistency, rollout)

    assert transformed["player_ids"].tolist() == [
        [(0 + 1) % 4, (1 + 1) % 4, (2 + 1) % 4, -1],
        [(2 + 2) % 4, (3 + 2) % 4, (0 + 2) % 4, -1],
    ]
    # Every step of a row uses that row's single rotation, including the
    # event seat fields and the per-step observer-relative marker.
    for row, shift in ((0, 1), (1, 2)):
        for step in range(3):
            for observer in range(P):
                target = (observer + shift) % P
                assert int(transformed["obs_tokens"][row, step, target, 4]) == (
                    LENGTH_TOKEN_BASE + target
                )
                assert int(
                    transformed["obs_channels"][row, step, target, REMAINING_COUNTS.start + target]
                ) == 10 + observer
                assert int(
                    transformed["obs_channels"][row, step, target, SELF_PREFIX.start + 1]
                ) == 40 + 4 * step + observer


def test_non_loss_step_inside_the_visible_prefix_is_transformed(tmp_path):
    agent = _tiny_agent()
    contract = _sealed_contract(tmp_path, agent, ["seat_rotation_1"])
    consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
    # Step 0 has a single legal candidate (no distribution, no loss token) but
    # it is still inside the causal prefix of steps 1 and 2.
    rollout = _rollout(
        player_ids=[[0, 1, 2, -1]], num_steps=[3],
        legal_counts=[[1, 3, 3, 0]],
    )
    transformed, _ = _run_loss(agent, consistency, rollout)
    assert transformed["player_ids"].tolist() == [[1, 2, 3, -1]]
    for observer in range(P):
        target = (observer + 1) % P
        assert int(transformed["obs_tokens"][0, 0, target, 4]) == LENGTH_TOKEN_BASE + target
        assert int(
            transformed["obs_channels"][0, 0, target, SELF_PREFIX.start + 1]
        ) == 40 + observer


# ---------------------------------------------------------------------------
# 3. four fields from one T, byte round-trip
# ---------------------------------------------------------------------------


def test_four_fields_share_the_same_transform_and_round_trip(tmp_path):
    agent = _tiny_agent()
    contract = _sealed_contract(tmp_path, agent, ["copy_swap"])
    consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
    # copy_swap is seat-free: player_ids must be unchanged, cards must swap.
    rollout = _rollout(player_ids=[[1, 3, -1, -1]], num_steps=[2])
    transformed, _ = _run_loss(agent, consistency, rollout)
    # T(seat) = seat under copy_swap.
    assert transformed["player_ids"].tolist() == [[1, 3, -1, -1]]
    # Hand bits moved by +54 mod 108 ...
    for observer in range(P):
        assert int(
            transformed["obs_channels"][0, 0, observer, SELF_HAND.start + HAND_CARD[observer]]
        ) == 0
        assert int(
            transformed["obs_channels"][0, 0, observer,
                                         SELF_HAND.start + (HAND_CARD[observer] + 54) % 108]
        ) == 1
    # ... and candidate card tokens are the physical image (same index).
    source_rows = _candidate_rows()
    image = transformed["avail_tokens"][0, 0].numpy()
    assert int(image[1, 5]) == CARD_TOKEN_BASE + (3 + 54) % 108
    assert int(image[2, 5]) == CARD_TOKEN_BASE + (57 + 54) % 108
    assert int(image[0, 5]) == int(source_rows[0, 5])  # PASS is seat/card free
    # Byte round-trip of the captured observation with the explicit inverse
    # (copy_swap is its own inverse); the manual expectations above are the
    # definition, this only proves no information was lost.
    captured = transformed["obs_channels"][0, 0].numpy().copy()
    tokens = transformed["obs_tokens"][0, 0].numpy().copy()
    # Invert the hand bit manually for the hand cards only.
    for observer in range(P):
        moved = (HAND_CARD[observer] + 54) % 108
        captured[observer, SELF_HAND.start + moved] = 0
        captured[observer, SELF_HAND.start + HAND_CARD[observer]] = 1
    assert int(captured[0, SELF_HAND.start + HAND_CARD[0]]) == 1
    np.testing.assert_array_equal(tokens, rollout["obs_tokens"][0, 0].numpy())


# ---------------------------------------------------------------------------
# 4. padding / inactive / num_steps / cache boundaries
# ---------------------------------------------------------------------------


def test_padding_steps_and_inactive_seats_are_never_remapped(tmp_path):
    agent = _tiny_agent()
    contract = _sealed_contract(
        tmp_path, agent, ["seat_rotation_1", "seat_rotation_2"],
    )
    consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
    rollout = _rollout(
        player_ids=[[0, 1, 2, -1], [3, -1, -1, -1]],
        num_steps=[3, 1],
    )
    transformed, _ = _run_loss(agent, consistency, rollout)
    assert transformed["player_ids"].tolist() == [
        [1, 2, 3, -1],   # row 0: +1
        [1, -1, -1, -1],  # row 1: +2 (3 -> 1); inactive stays -1
    ]
    # ``num_steps`` is untouched.
    assert transformed["num_steps"].tolist() == [3, 1]
    # Row 1 has one active step, so only prefix step 0 is transformed; the
    # padding cells keep their exact bytes.
    for step in range(1, T):
        np.testing.assert_array_equal(
            transformed["obs_channels"][1, step].numpy(),
            rollout["obs_channels"][1, step].numpy(),
        )
        np.testing.assert_array_equal(
            transformed["obs_tokens"][1, step].numpy(),
            rollout["obs_tokens"][1, step].numpy(),
        )
    # A finished/inactive seat is never mapped through T.
    assert int(transformed["player_ids"][1, 1]) == -1
    assert int(transformed["player_ids"][0, 3]) == -1


def test_transformed_forward_gets_no_kv_cache_and_does_not_mutate_base(tmp_path):
    agent = _tiny_agent()
    contract = _sealed_contract(tmp_path, agent, ["seat_rotation_1"])
    consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
    rollout = _rollout(player_ids=[[0, 2, -1, -1]], num_steps=[2])
    base_snapshot = {key: value.clone() for key, value in rollout.items()}

    base = agent(rollout)
    received = {}

    def forward(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return SimpleNamespace(avail_log_probs=base.avail_log_probs.detach().clone())

    consistency.loss(rollout, base_log_probs=base.avail_log_probs, forward=forward)
    # Exactly one positional argument, no ``past_key_values``/``use_cache``:
    # the transformed observation must never reuse the base KV cache.
    assert len(received["args"]) == 1 and received["kwargs"] == {}
    transformed = received["args"][0]
    assert transformed is not rollout
    for key in ("obs_tokens", "obs_channels", "avail_tokens", "player_ids"):
        assert transformed[key].data_ptr() != rollout[key].data_ptr()
    # The source mini-batch is not mutated by the branch.
    for key, value in base_snapshot.items():
        assert torch.equal(rollout[key], value)


def test_transform_counts_are_reported_per_row_transform(tmp_path):
    agent = _tiny_agent()
    contract = _sealed_contract(
        tmp_path, agent, ["seat_rotation_1", "seat_rotation_2"],
    )
    consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
    rollout = _rollout(player_ids=[[0, 1, -1, -1], [2, 3, -1, -1]], num_steps=[2, 2])
    base = agent(rollout)
    spy = _SpyForward(base.avail_log_probs)
    result = consistency.loss(rollout, base_log_probs=base.avail_log_probs, forward=spy)
    assert result.token_count_global == 4
    assert result.transform_counts == {"seat_rotation_1": 2, "seat_rotation_2": 2}


# ---------------------------------------------------------------------------
# 5. more than MAX_SYMMETRY_TOKENS_PER_BATCH valid cells
# ---------------------------------------------------------------------------


def test_more_than_64_valid_cells_limits_only_the_loss_token_set(tmp_path):
    """The 64-token cap bounds the loss set, never the causal history."""
    from guandan_marvel.phase5.symmetry_training import MAX_SYMMETRY_TOKENS_PER_BATCH

    agent = _tiny_agent()
    contract = _sealed_contract(
        tmp_path, agent, ["seat_rotation_1", "seat_rotation_2"],
    )
    consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
    steps = 40
    player_ids = [
        [step % 4 for step in range(steps)],
        [(step + 2) % 4 for step in range(steps)],
    ]
    rollout = _rollout(player_ids=player_ids, num_steps=[steps, steps])
    base = agent(rollout)

    weights = rollout["avail_weights"]
    time = torch.arange(steps).unsqueeze(0)
    num_steps = torch.tensor([steps, steps]).unsqueeze(1)
    valid = (time < num_steps) & (torch.tensor(player_ids) >= 0) & ((weights > 0).sum(-1) >= 2)
    flat = valid.reshape(-1).nonzero(as_tuple=False).reshape(-1)
    assert int(flat.numel()) == 2 * steps
    assert int(flat.numel()) > MAX_SYMMETRY_TOKENS_PER_BATCH
    stride = int(math.ceil(flat.numel() / MAX_SYMMETRY_TOKENS_PER_BATCH))
    selected_flat = flat[::stride][:MAX_SYMMETRY_TOKENS_PER_BATCH]
    selected = torch.zeros_like(valid)
    selected.reshape(-1)[selected_flat] = True
    assert int(selected.sum()) == 40

    # Perturb every valid-but-unselected cell: a KL that summed over ``valid``
    # instead of the selected sample would be large, while the fixed branch is
    # exactly zero on the selected cells.
    other = base.avail_log_probs.detach().float().clone()
    perturbed = valid & ~selected
    other[perturbed] = other[perturbed] - 50.0

    spy = _SpyForward(other)
    result = consistency.loss(rollout, base_log_probs=base.avail_log_probs, forward=spy)
    assert result.token_count_local == 40
    assert result.token_count_global == 40
    assert result.candidate_count_local == 40 * 3
    assert result.transform_counts == {"seat_rotation_1": 20, "seat_rotation_2": 20}
    assert result.mean_kl_global == 0.0
    assert result.max_kl_local == 0.0
    assert float(result.loss.detach()) == 0.0

    # The cap did not truncate the prefix: step 1 of row 0 is not a selected
    # loss token, yet it is transformed with row 0's single T.
    transformed = spy.batch
    assert int(transformed["player_ids"][0, 1]) == (1 + 1) % 4
    assert int(transformed["obs_tokens"][0, 1, 2, 4]) == LENGTH_TOKEN_BASE + 2


def test_row_without_a_loss_token_is_left_untouched(tmp_path):
    agent = _tiny_agent()
    contract = _sealed_contract(
        tmp_path, agent, ["seat_rotation_1", "seat_rotation_2"],
    )
    consistency = SymmetryConsistency(contract, device=torch.device("cpu"))
    rollout = _rollout(
        player_ids=[[0, 1, -1, -1], [2, 3, -1, -1]], num_steps=[2, 2],
        legal_counts=[[3, 3, 0, 0], [1, 1, 0, 0]],
    )
    transformed, result = _run_loss(agent, consistency, rollout)
    # Row 1 has no cell with two legal candidates: no T is drawn for it, so its
    # actor seats and observation bytes are untouched.
    assert transformed["player_ids"].tolist() == [[1, 2, -1, -1], [2, 3, -1, -1]]
    for step in range(2):
        np.testing.assert_array_equal(
            transformed["obs_tokens"][1, step].numpy(),
            rollout["obs_tokens"][1, step].numpy(),
        )
        np.testing.assert_array_equal(
            transformed["obs_channels"][1, step].numpy(),
            rollout["obs_channels"][1, step].numpy(),
        )
    assert result.token_count_local == 2
    assert result.transform_counts == {"seat_rotation_1": 2, "seat_rotation_2": 0}
