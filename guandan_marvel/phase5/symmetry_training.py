# SPDX-License-Identifier: MIT
"""Fail-closed artifact gate and aligned consistency loss for phase-5 symmetry.

The diagnostic artifact (``symmetry_diagnostic_r3.json``) is a frozen, read-only
evidence file.  It carries no internal seal, so enabling the feature requires a
*seal wrapper* (``guandan-marvel-phase5-symmetry-artifact-seal-v1``) that binds

* the exact bytes of the original artifact (SHA256 and canonical content hash),
* the frozen design hash and the pre-registered significance thresholds,
* the diagnostic decision (``status == SIGNIFICANT``),
* the transform vocabulary and the initial policy checkpoint SHA256.

Nothing is read while ``symmetry_weight == 0`` and ``augmentation is False``.

Training consumption
--------------------
The only supported training branch is an *extra consistency loss*.  For a
mini-batch of active macro/token decisions the model produces

    p_i = pi(a_i | s)                     (current state, detached target)
    q_i = pi(P_T(a_i) | T(s))             (transformed state)

and the loss is the aligned, legal-candidate KL

    L_sym(s) = sum_i p_i * (log p_i - log q_i),     i over legal candidates

The action unit is the **restricted next-token candidate** set (the same
candidate rows the policy already scores).  It is never reported as a full
macro-action KL: complete macro enumeration is not available in the learner.
Gradient flows through ``log q`` only; ``p`` and its weights are detached, so
the on-policy behaviour probability and the PPO ratio are untouched.

Seat identity travels with the observation.  ``transform_observation`` writes
observer ``o`` into row ``T(o)``; the decoder embeds the row index as the seat
embedding, and the model gathers the acting seat's state/action row by
``player_ids``.  The transformed mini-batch therefore stores
``player_ids[b, t] = T(actor)`` on every active cell (``-1`` padding is left
untouched).  One transform is drawn per *sample row* and applied to that row's
whole visible causal prefix (all steps up to its last active decision), because
the decoder attends over earlier steps of the same row.  Per-cell transforms
would mix different symmetries inside a single history and are not supported.

DDP: every rank contributes its local token sum; the differentiable loss is
``world_size / global_token_count * local_sum`` so that DDP's gradient average
equals the global per-token mean.  A rank with no local tokens contributes a
model-connected exact zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .belief_data import canonical_records_json, canonical_records_value
from .design import (
    PHASE5_SYMMETRY_SCHEMA, SYMMETRY_SIGNIFICANCE, SYMMETRY_TRANSFORMS,
    phase5_design_hash,
)
from .symmetry import (
    CopySwap, SeatRotation, SymmetryDomainError, Transform,
    transform_candidate_rows, transform_observation,
)

SYMMETRY_ARTIFACT_SEAL_SCHEMA = "guandan-marvel-phase5-symmetry-artifact-seal-v1"
SYMMETRY_DIAGNOSTIC_RUN_SCHEMA = "guandan-phase5-symmetry-diagnostic-run-v1"
SYMMETRY_TRAINING_SCHEMA = "guandan-marvel-phase5-symmetry-training-v1"
# A mini-batch is a deterministic subsample of the active decisions; the loss
# formula is unchanged and every selected token contributes exactly once.
MAX_SYMMETRY_TOKENS_PER_BATCH = 64
SYMMETRY_ACTION_UNIT = "restricted_next_token_candidate"


class SymmetryArtifactError(ValueError):
    """The symmetry artifact or its declaration is not trustworthy."""


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        canonical_records_json([value]).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform_from_name(name: str) -> Transform:
    text = str(name)
    if text == "copy_swap":
        return CopySwap()
    if text.startswith("seat_rotation_"):
        suffix = text[len("seat_rotation_"):]
        if suffix.isdigit():
            return SeatRotation(int(suffix))
    raise SymmetryArtifactError(f"unknown symmetry transform {name!r}")


@dataclass(frozen=True)
class SymmetryContract:
    """Validated identity of one sealed diagnostic artifact."""

    seal_path: str
    seal_sha256: str
    seal_hash: str
    artifact_path: str
    artifact_sha256: str
    content_hash: str
    design_hash: str
    transforms: tuple[str, ...]
    checkpoint_sha256: str
    source_policy_identity: Mapping[str, Any]
    mean_kl: float
    argmax_disagreement_rate: float
    sample_count: int
    deal_count: int
    decision: Mapping[str, Any]

    def audit(self) -> dict[str, Any]:
        return {
            "schema": SYMMETRY_TRAINING_SCHEMA,
            "seal_path": self.seal_path,
            "seal_sha256": self.seal_sha256,
            "seal_hash": self.seal_hash,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "content_hash": self.content_hash,
            "design_hash": self.design_hash,
            "transforms": list(self.transforms),
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_policy_identity": dict(self.source_policy_identity),
            "mean_kl": self.mean_kl,
            "argmax_disagreement_rate": self.argmax_disagreement_rate,
            "sample_count": self.sample_count,
            "deal_count": self.deal_count,
            "decision": dict(self.decision),
            "action_unit": SYMMETRY_ACTION_UNIT,
            "max_tokens_per_batch": MAX_SYMMETRY_TOKENS_PER_BATCH,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SymmetryArtifactError(message)


def _finite(value: object, name: str) -> float:
    number = float(value)
    _require(math.isfinite(number), f"{name} must be finite")
    return number


def load_sealed_symmetry_artifact(path: str | Path) -> SymmetryContract:
    """Load and fully validate a sealed diagnostic artifact.

    Every failure mode (missing file, wrong schema, tampered bytes, stale
    design, non-significant decision, insufficient sample size, non-finite
    statistics, unknown transforms) raises :class:`SymmetryArtifactError`.
    """
    seal_path = Path(path)
    _require(seal_path.is_file(), f"symmetry seal artifact is missing: {seal_path}")
    seal_bytes = seal_path.read_bytes()
    try:
        seal = json.loads(seal_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SymmetryArtifactError(f"symmetry seal is not valid JSON: {error}") from error
    _require(isinstance(seal, Mapping), "symmetry seal must be a JSON object")
    _require(
        seal.get("schema") == SYMMETRY_ARTIFACT_SEAL_SCHEMA,
        f"unexpected symmetry seal schema {seal.get('schema')!r}",
    )
    declared = seal.get("seal_hash")
    _require(isinstance(declared, str) and declared, "symmetry seal is missing seal_hash")
    recomputed = canonical_hash({key: value for key, value in seal.items() if key != "seal_hash"})
    _require(declared == recomputed, "symmetry seal_hash is inconsistent (tampered seal)")

    artifact_path = seal.get("artifact_path")
    _require(
        isinstance(artifact_path, str) and artifact_path,
        "symmetry seal is missing artifact_path",
    )
    artifact_file = Path(artifact_path)
    _require(artifact_file.is_file(), f"sealed symmetry artifact is missing: {artifact_file}")
    artifact_bytes = artifact_file.read_bytes()
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
    _require(
        artifact_sha == seal.get("artifact_sha256"),
        "sealed symmetry artifact bytes do not match the declared SHA256",
    )
    try:
        artifact = json.loads(artifact_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SymmetryArtifactError(f"sealed symmetry artifact is not valid JSON: {error}") from error
    _require(isinstance(artifact, Mapping), "sealed symmetry artifact must be a JSON object")
    content_hash = canonical_hash(artifact)
    _require(
        content_hash == seal.get("content_hash"),
        "sealed symmetry artifact content hash does not match the seal",
    )

    _require(
        artifact.get("schema") == SYMMETRY_DIAGNOSTIC_RUN_SCHEMA,
        f"unexpected diagnostic artifact schema {artifact.get('schema')!r}",
    )
    _require(artifact.get("status") == "COMPLETED", "diagnostic artifact is not COMPLETED")
    _require(artifact.get("training") is False, "diagnostic artifact unexpectedly trained")
    _require(artifact.get("truncated") is False, "truncated diagnostic is not admissible")
    _require(
        int(artifact.get("deals_requested", 0)) <= int(SYMMETRY_SIGNIFICANCE["deals"]),
        "diagnostic artifact exceeds the pre-registered 16-deal budget",
    )
    design_hash = artifact.get("design_hash")
    _require(
        design_hash == phase5_design_hash(),
        "diagnostic artifact design hash does not match the frozen phase-5 design",
    )
    _require(
        seal.get("design_hash") == design_hash,
        "symmetry seal design hash disagrees with the artifact",
    )

    decision = artifact.get("decision")
    _require(isinstance(decision, Mapping), "diagnostic artifact has no decision block")
    _require(
        decision.get("status") == "SIGNIFICANT",
        f"symmetry feature requires SIGNIFICANT, got {decision.get('status')!r}",
    )
    _require(
        decision.get("augmentation_recommended") is True,
        "SIGNIFICANT diagnostic must recommend the consistency feature",
    )
    _require(
        int(decision.get("non_finite_kl_count", 0)) == 0,
        "diagnostic decision contains non-finite KL samples",
    )
    thresholds = decision.get("thresholds")
    _require(isinstance(thresholds, Mapping), "diagnostic decision has no thresholds")
    expected_thresholds = {
        "min_macro_samples": int(SYMMETRY_SIGNIFICANCE["min_macro_samples"]),
        "min_deals": int(SYMMETRY_SIGNIFICANCE["min_deals"]),
        "mean_restricted_kl_threshold": float(
            SYMMETRY_SIGNIFICANCE["mean_restricted_kl_threshold"]
        ),
        "argmax_disagreement_threshold": float(
            SYMMETRY_SIGNIFICANCE["argmax_disagreement_threshold"]
        ),
    }
    for key, expected in expected_thresholds.items():
        _require(
            key in thresholds and float(thresholds[key]) == expected,
            f"diagnostic threshold {key!r} is not the pre-registered value",
        )
    sample_count = int(decision.get("sample_count", 0))
    deal_count = int(decision.get("deal_count", 0))
    _require(
        sample_count >= expected_thresholds["min_macro_samples"],
        "diagnostic sample count is below the pre-registered minimum",
    )
    _require(
        deal_count >= expected_thresholds["min_deals"],
        "diagnostic deal count is below the pre-registered minimum",
    )
    mean_kl = _finite(decision.get("mean_kl"), "decision.mean_kl")
    disagreement = _finite(
        decision.get("argmax_disagreement_rate"), "decision.argmax_disagreement_rate",
    )
    _require(mean_kl >= 0.0, "mean KL must be non-negative")
    _require(0.0 <= disagreement <= 1.0, "argmax disagreement must lie in [0, 1]")

    diagnostic_block = artifact.get("diagnostic")
    _require(
        isinstance(diagnostic_block, Mapping),
        "diagnostic artifact has no diagnostic block",
    )
    _require(
        isinstance(diagnostic_block.get("transforms"), list),
        "diagnostic artifact has no transform list",
    )
    aggregate = diagnostic_block.get("aggregate")
    _require(isinstance(aggregate, Mapping), "diagnostic artifact has no aggregate block")
    _finite(aggregate.get("mean_kl"), "aggregate.mean_kl")
    _finite(aggregate.get("median_kl"), "aggregate.median_kl")
    _finite(aggregate.get("max_kl"), "aggregate.max_kl")
    _require(
        int(aggregate.get("non_finite_kl_count", 0)) == 0,
        "diagnostic aggregate contains non-finite KL samples",
    )

    raw_transforms = seal.get("transforms")
    _require(
        isinstance(raw_transforms, list) and raw_transforms,
        "symmetry seal declares no transforms",
    )
    transforms = tuple(str(name) for name in raw_transforms)
    _require(len(set(transforms)) == len(transforms), "symmetry transforms are not unique")
    for name in transforms:
        _require(
            name in SYMMETRY_TRANSFORMS,
            f"transform {name!r} is outside the frozen design vocabulary",
        )
        transform_from_name(name)
    _require(
        sorted(transforms) == sorted(str(name) for name in artifact["diagnostic"]["transforms"]),
        "symmetry seal transforms disagree with the artifact",
    )

    checkpoint_sha = seal.get("checkpoint_sha256")
    _require(
        isinstance(checkpoint_sha, str) and len(checkpoint_sha) == 64,
        "symmetry seal is missing the initial checkpoint SHA256",
    )
    _require(
        checkpoint_sha == artifact.get("checkpoint_sha256"),
        "symmetry seal checkpoint SHA256 disagrees with the artifact",
    )
    identity = seal.get("source_policy_identity")
    _require(isinstance(identity, Mapping), "symmetry seal is missing the policy identity")
    for key in ("architecture", "algorithm", "history_dim", "feature_dim",
                "parameter_count", "state_dict_keys"):
        _require(key in identity, f"symmetry policy identity is missing {key!r}")

    return SymmetryContract(
        seal_path=str(seal_path.resolve()),
        seal_sha256=hashlib.sha256(seal_bytes).hexdigest(),
        seal_hash=declared,
        artifact_path=str(artifact_file.resolve()),
        artifact_sha256=artifact_sha,
        content_hash=content_hash,
        design_hash=str(design_hash),
        transforms=transforms,
        checkpoint_sha256=checkpoint_sha,
        source_policy_identity=dict(identity),
        mean_kl=mean_kl,
        argmax_disagreement_rate=disagreement,
        sample_count=sample_count,
        deal_count=deal_count,
        decision=dict(decision),
    )


def validate_config_artifact(
    *, artifact_path: str | None, checkpoint_path: str | None,
) -> SymmetryContract:
    """Config-time gate: sealed artifact plus the exact initial checkpoint bytes."""
    _require(bool(artifact_path and str(artifact_path).strip()),
             "symmetry.symmetry_weight > 0 requires symmetry.diagnostic_artifact")
    _require(bool(checkpoint_path and str(checkpoint_path).strip()),
             "symmetry.symmetry_weight > 0 requires symmetry.compatibility_checkpoint")
    contract = load_sealed_symmetry_artifact(str(artifact_path))
    checkpoint = Path(str(checkpoint_path))
    _require(checkpoint.is_file(), f"symmetry compatibility checkpoint is missing: {checkpoint}")
    _require(
        file_sha256(checkpoint) == contract.checkpoint_sha256,
        "symmetry compatibility checkpoint SHA256 does not match the sealed artifact",
    )
    return contract


def validate_contract_against_model(
    contract: SymmetryContract,
    *,
    agent: torch.nn.Module,
    architecture: str,
    algorithm: str,
    history_dim: int,
    feature_dim: int,
) -> None:
    """Bind the sealed diagnostic to the model that will actually be trained."""
    identity = contract.source_policy_identity
    _require(str(identity["architecture"]) == str(architecture),
             "sealed artifact architecture does not match the training model")
    _require(str(identity["algorithm"]) == str(algorithm),
             "sealed artifact algorithm does not match the trainer")
    _require(int(identity["history_dim"]) == int(history_dim),
             "sealed artifact history_dim does not match the training model")
    _require(int(identity["feature_dim"]) == int(feature_dim),
             "sealed artifact feature_dim does not match the training model")
    state = agent.state_dict()
    _require(int(identity["state_dict_keys"]) == len(state),
             "sealed artifact state_dict key count does not match the training model")
    _require(int(identity["parameter_count"]) == sum(value.numel() for value in state.values()),
             "sealed artifact parameter count does not match the training model")


@dataclass
class SymmetryLossOutput:
    loss: torch.Tensor
    token_count_local: int
    token_count_global: int
    mean_kl_global: float
    max_kl_local: float
    candidate_count_local: int
    transform_counts: dict[str, int] = field(default_factory=dict)
    schema: str = SYMMETRY_TRAINING_SCHEMA
    action_unit: str = SYMMETRY_ACTION_UNIT

    def audit(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "action_unit": self.action_unit,
            "token_count_local": int(self.token_count_local),
            "token_count_global": int(self.token_count_global),
            "mean_kl_global": float(self.mean_kl_global),
            "max_kl_local": float(self.max_kl_local),
            "candidate_count_local": int(self.candidate_count_local),
            "transform_counts": dict(self.transform_counts),
        }


def _all_reduce_scalar(value: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return value


class SymmetryConsistency:
    """Aligned ``pi(P_T(a)|T(s))`` vs ``pi(a|s)`` consistency term."""

    def __init__(self, contract: SymmetryContract, *, device: torch.device) -> None:
        self.contract = contract
        self.device = torch.device(device)
        self.transforms: tuple[Transform, ...] = tuple(
            transform_from_name(name) for name in contract.transforms
        )

    @classmethod
    def from_config(cls, config, *, agent, architecture, algorithm, history_dim,
                    feature_dim, device) -> "SymmetryConsistency":
        contract = validate_config_artifact(
            artifact_path=getattr(config, "diagnostic_artifact", None),
            checkpoint_path=getattr(config, "compatibility_checkpoint", None),
        )
        validate_contract_against_model(
            contract, agent=agent, architecture=architecture, algorithm=algorithm,
            history_dim=history_dim, feature_dim=feature_dim,
        )
        return cls(contract, device=device)

    def select(self, flat_index: int) -> Transform:
        return self.transforms[int(flat_index) % len(self.transforms)]

    def loss(
        self,
        rollout,
        *,
        base_log_probs: torch.Tensor,
        forward: Callable[[Any], Any],
        world_size: int = 1,
    ) -> SymmetryLossOutput:
        from guandan_marvel.model.tensordict import TensorDict
        from guandan_marvel.training.macro_ppo import model_connected_zero

        if not isinstance(rollout, TensorDict):
            raise TypeError("symmetry consistency requires a TensorDict mini-batch")
        num_steps = rollout["num_steps"].to(torch.long)
        player_ids = rollout["player_ids"].to(torch.long)
        obs_tokens = rollout["obs_tokens"]
        obs_channels = rollout["obs_channels"]
        avail_tokens = rollout["avail_tokens"]
        avail_weights = rollout["avail_weights"]
        b, t = player_ids.shape
        if base_log_probs.shape[:2] != (b, t):
            raise SymmetryArtifactError("base log-probabilities do not match the mini-batch")
        time = torch.arange(t, device=player_ids.device).unsqueeze(0)
        active = (time < num_steps.unsqueeze(1)) & (player_ids >= 0)
        # A single legal candidate carries no distributional information, so it
        # is excluded from the token count instead of contributing an undefined
        # (or trivially zero) comparison.
        legal_counts = (avail_weights > 0).sum(dim=-1)
        valid = active & (legal_counts >= 2)
        flat_valid = valid.reshape(-1).nonzero(as_tuple=False).reshape(-1)
        count_local_all = int(flat_valid.numel())
        # Deterministic bounded subsample (no RNG draw, identical on any replay).
        flat_active = flat_valid
        if count_local_all > MAX_SYMMETRY_TOKENS_PER_BATCH:
            stride = int(math.ceil(count_local_all / MAX_SYMMETRY_TOKENS_PER_BATCH))
            flat_active = flat_active[::stride][:MAX_SYMMETRY_TOKENS_PER_BATCH]
        local_tokens = int(flat_active.numel())
        # The *one* token set that every statistic of this call must agree on:
        # count, transform audit, candidate audit and the KL numerator.
        selected = torch.zeros((b, t), dtype=torch.bool, device=player_ids.device)
        if local_tokens:
            selected_rows_flat = torch.div(flat_active, t, rounding_mode="floor")
            selected_steps_flat = flat_active % t
            selected[selected_rows_flat, selected_steps_flat] = True

        count_tensor = torch.as_tensor(
            [float(local_tokens)], dtype=torch.float64, device=player_ids.device,
        )
        count_tensor = _all_reduce_scalar(count_tensor)
        global_tokens = int(round(float(count_tensor.item())))
        if global_tokens == 0:
            raise SymmetryArtifactError("no active decision occurred on any rank")
        if local_tokens == 0:
            zero = model_connected_zero(base_log_probs)
            # Every rank must execute the same collective sequence: a rank with
            # no local decisions still participates in the mean-KL reduction
            # with an exact zero instead of returning before the all-reduce
            # (which would deadlock the ranks that do have decisions).
            detached_zero = torch.zeros((), dtype=torch.float64, device=player_ids.device)
            global_sum = _all_reduce_scalar(detached_zero)
            return SymmetryLossOutput(
                loss=zero, token_count_local=0, token_count_global=global_tokens,
                mean_kl_global=float(global_sum.item()) / float(global_tokens),
                max_kl_local=0.0, candidate_count_local=0,
            )

        transformed_tokens = obs_tokens.detach().clone()
        transformed_channels = obs_channels.detach().clone()
        transformed_avail = avail_tokens.detach().clone()
        transformed_player_ids = player_ids.detach().clone()
        transform_counts: dict[str, int] = {name: 0 for name in self.contract.transforms}
        weights_cpu = avail_weights.detach().to("cpu")
        candidate_total = 0
        # One transform per *sample row*.  A row is one causal sequence: the
        # decoder attends to its own earlier steps, so the whole visible prefix
        # (and the acting seat of every step) must be transported by the same T.
        # Independently chosen per-cell transforms would splice observations
        # from different symmetries into one history and are not equivariant.
        # Only rows that carry at least one *selected* loss token are transported;
        # a row without a loss token cannot affect the loss.
        selected_rows = selected.any(dim=1).nonzero(as_tuple=False).reshape(-1).tolist()
        row_transforms: dict[int, Transform] = {}
        row_transform_names: dict[int, str] = {}
        for row in selected_rows:
            index = int(row) % len(self.transforms)
            row_transforms[int(row)] = self.transforms[index]
            # The audit key is the sealed contract name of the transform that was
            # actually drawn for this row (``SeatRotation.name`` is class-level
            # and cannot distinguish ``seat_rotation_1`` from ``_2``).
            row_transform_names[int(row)] = self.contract.transforms[index]
        transform_counts: dict[str, int] = {name: 0 for name in self.contract.transforms}
        for row in selected_rows:
            transform_counts[row_transform_names[int(row)]] += int(selected[row].sum())
        transformed_player_tensor = transformed_player_ids
        for row in selected_rows:
            transform = row_transforms[int(row)]
            # Prefix needed for every loss target in this row: from the first
            # step up to (and including) the last decision that is either a loss
            # token or part of its causal prefix.  The 64-token cap limits the
            # loss token set only; it never truncates this history.
            active_steps = active[row].nonzero(as_tuple=False).reshape(-1).tolist()
            last_step = int(max(active_steps))
            for step in range(last_step + 1):
                tokens_np = np.asarray(
                    obs_tokens[row, step].detach().to("cpu"), dtype=np.int64,
                )
                channels_np = np.asarray(
                    obs_channels[row, step].detach().to("cpu"), dtype=np.uint8,
                )
                new_tokens, new_channels = transform_observation(
                    tokens_np, channels_np, transform,
                )
                transformed_tokens[row, step] = torch.as_tensor(
                    new_tokens, dtype=obs_tokens.dtype, device=obs_tokens.device,
                )
                transformed_channels[row, step] = torch.as_tensor(
                    new_channels, dtype=obs_channels.dtype, device=obs_channels.device,
                )
            # The acting seat rides the same T; padding/inactive cells keep -1.
            for step in active_steps:
                actor = int(player_ids[row, step])
                transformed_player_tensor[row, step] = int(transform.apply_seat(actor))
            # Candidate rows are the induced action image P_T under the same T.
            # Every active step is transported so the row is one coherent
            # symmetry image; only *selected* loss tokens are counted/validated.
            for step in active_steps:
                rows_np = np.asarray(
                    avail_tokens[row, step].detach().to("cpu"), dtype=np.int64,
                )
                legal = np.asarray(weights_cpu[row, step] > 0)
                image_rows = transform_candidate_rows(rows_np, transform)
                if bool(selected[row, step]):
                    image_legal = image_rows[legal]
                    keys = {tuple(int(value) for value in item) for item in image_legal}
                    if len(keys) != int(image_legal.shape[0]):
                        raise SymmetryDomainError(
                            "legal candidate images are not one-to-one under the transform"
                        )
                    candidate_total += int(legal.sum())
                transformed_avail[row, step] = torch.as_tensor(
                    image_rows, dtype=avail_tokens.dtype, device=avail_tokens.device,
                )

        data = {key: value for key, value in rollout.items()}
        data["obs_tokens"] = transformed_tokens
        data["obs_channels"] = transformed_channels
        data["avail_tokens"] = transformed_avail
        # The observer rows moved to T(o); the model gathers the actor's row by
        # ``player_ids``, so it must be T(actor) or the branch compares the base
        # policy against another seat's observation.
        data["player_ids"] = transformed_player_tensor
        transformed = TensorDict(data, batch_size=rollout.batch_size)

        output = forward(transformed)
        transformed_log_probs = getattr(output, "avail_log_probs", None)
        if transformed_log_probs is None:
            raise SymmetryArtifactError("the policy did not return FP32 avail_log_probs")
        if transformed_log_probs.shape != avail_weights.shape:
            raise SymmetryArtifactError("transformed log-probabilities have the wrong shape")

        base = torch.where(active.unsqueeze(-1), base_log_probs.float(), torch.zeros_like(base_log_probs, dtype=torch.float32))
        other = torch.where(active.unsqueeze(-1), transformed_log_probs.float(), torch.zeros_like(transformed_log_probs, dtype=torch.float32))
        base64 = base.detach().to(torch.float64)
        other64 = other.to(torch.float64)
        if not bool(torch.isfinite(base64[active]).all()):
            raise SymmetryArtifactError("base log-probabilities are non-finite on active tokens")
        if not bool(torch.isfinite(other64[active]).all()):
            raise SymmetryArtifactError("transformed log-probabilities are non-finite")
        probabilities = torch.exp(base64)
        # Exactly the deterministic subsample that produced ``local_tokens`` and
        # ``transform_counts``; the numerator and the normaliser agree.
        legal_masked = (avail_weights > 0) & selected.unsqueeze(-1)
        weighted = torch.where(legal_masked, probabilities * (base64 - other64), torch.zeros_like(base64))
        per_token = weighted.sum(dim=-1)
        summed = per_token.sum()
        if not bool(torch.isfinite(summed)):
            raise SymmetryArtifactError("aligned symmetry KL is non-finite")
        scaled = summed * (float(world_size) / float(global_tokens))
        with torch.no_grad():
            detached_sum = per_token.detach().to(torch.float64).sum()
            global_sum = _all_reduce_scalar(detached_sum)
            mean_kl = float(global_sum.item()) / float(global_tokens)
        return SymmetryLossOutput(
            loss=scaled,
            token_count_local=local_tokens,
            token_count_global=global_tokens,
            mean_kl_global=mean_kl,
            max_kl_local=float(per_token.detach().max().item()),
            candidate_count_local=candidate_total,
            transform_counts=transform_counts,
        )


__all__ = [
    "MAX_SYMMETRY_TOKENS_PER_BATCH", "SYMMETRY_ACTION_UNIT",
    "SYMMETRY_ARTIFACT_SEAL_SCHEMA", "SYMMETRY_DIAGNOSTIC_RUN_SCHEMA",
    "SYMMETRY_TRAINING_SCHEMA", "SymmetryArtifactError", "SymmetryConsistency",
    "SymmetryContract", "SymmetryLossOutput", "canonical_hash", "file_sha256",
    "load_sealed_symmetry_artifact", "transform_from_name",
    "validate_config_artifact", "validate_contract_against_model",
]
