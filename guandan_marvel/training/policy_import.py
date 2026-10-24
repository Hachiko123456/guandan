# SPDX-License-Identifier: MIT
"""Explicit, versioned "import an old policy into a *new* run" entry point.

The historical ``LocalTrainerBase.load_warm_start_policy`` only supports a
pool-free policy warm start and deliberately refuses to run with an opponent
pool ("policy-only warm start cannot reconstruct opponent-pool assignments").
That guard stays in place: this module is a *separate*, explicitly versioned
migration contract for a controlled fresh-initialization run, and it never
claims to be an exact resume.

What is preserved
-----------------
* the shared backbone / state / action / actor-fuse / logit / value tensors are
  copied bitwise into both the trainable and the merged model;
* only an explicit, closed whitelist of source tensors may be *dropped*
  (``ntp_head.*`` for the candidate arm whose NTP auxiliary head is disabled).

What is freshly initialized (and recorded)
------------------------------------------
* optimizer state, AMP scaler state, merged-policy/EMA state;
* sampler state and counters, opponent-pool assignment state;
* RNG continuation, iteration/durable/learner counters;
* the context curriculum logical origin (``curriculum_offset``) so a fresh run
  does not silently replay the early schedule.

The original source identity (checkpoint hash, resolved-config hash, source
hash, backend identity) is recorded *separately* from the new target identity,
together with the migration version and the whitelist that was applied.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping

import torch

from guandan_marvel.evaluation.checkpoints import EVALUATION_CHECKPOINT_SCHEMA
from guandan_marvel.training.checkpoints import canonical_json_hash, capture_rng_state, restore_rng_state

POLICY_IMPORT_SCHEMA = "guandan-marvel-policy-import-v1"
POLICY_IMPORT_MANIFEST_SCHEMA = "guandan-marvel-policy-import-manifest-v1"
# The only tensor family this contract may ever drop.  Extending this tuple is
# a new migration version, not a configuration option.
ALLOWED_DROP_PREFIXES: tuple[str, ...] = ("ntp_head.",)


class PolicyImportError(ValueError):
    """Raised when an imported policy is not a permitted fresh-run source."""


def _trainable_parameter_ids(optimizer) -> list[int]:
    targets = getattr(optimizer, "optimizers", None)
    groups = []
    if isinstance(targets, Mapping):
        for target in targets.values():
            groups.extend(target.param_groups)
    else:
        groups.extend(optimizer.param_groups)
    ids: list[int] = []
    for group in groups:
        for parameter in group["params"]:
            ids.append(id(parameter))
    return ids


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_identity(agent) -> dict[str, object]:
    return {
        "history_dim": int(agent.history_dim),
        "feature_dim": int(agent.feature_dim),
        "decoder_layers": int(agent.model_cfg.decoder_layers),
        "decoder_heads": int(agent.model_cfg.decoder_heads),
        "state_layers": int(agent.model_cfg.state_layers),
        "action_layers": int(agent.model_cfg.action_layers),
        "fuse_layers": int(agent.model_cfg.fuse_layers),
        "ffn_ratio": int(agent.model_cfg.ffn_ratio),
    }


def _source_identity(payload: Mapping[str, object]) -> dict[str, object]:
    backend_identity = payload.get("backend_identity")
    return {
        "schema": "guandan-marvel-policy-import-source-identity-v1",
        "checkpoint_schema": str(payload.get("schema")),
        "algorithm": str(payload.get("algorithm")),
        "architecture": str(payload.get("architecture")),
        "protocol_version": str(payload.get("protocol_version")),
        "backend": str(payload.get("backend")),
        "backend_identity_hash": (
            None if not isinstance(backend_identity, Mapping) else backend_identity.get("hash")
        ),
        "resolved_config_hash": str(payload.get("resolved_config_hash")),
        "source_hash": None if payload.get("source_hash") is None else str(payload.get("source_hash")),
        "source_parent_commit": payload.get("source_parent_commit"),
        "origin_iteration": int(payload.get("iteration", -1)),
        "training_resume_allowed": payload.get("training_resume_allowed"),
    }


def _target_identity(trainer, *, model_identity: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "guandan-marvel-policy-import-target-identity-v1",
        "algorithm": str(trainer.algorithm),
        "architecture": str(trainer.cfg.model.architecture),
        "protocol_version": str(trainer.cfg.game.protocol_version),
        "backend": str(trainer.backend_name),
        "backend_identity_hash": str(trainer.backend_identity.get("hash")),
        "resolved_config_hash": str(trainer.resolved_config_hash),
        "source_hash": str(trainer.source_hash),
        "source_parent_commit": str(trainer.source_parent_commit),
        "model": dict(model_identity),
        "training_profile": str(trainer.training_semantics_profile),
        "policy_regularizer_mode": str(trainer.policy_regularizer_mode),
        "new_iteration_start": 0,
    }


def _validate_source_payload(trainer, payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise PolicyImportError("policy import requires a mapping checkpoint payload")
    schema = payload.get("schema")
    if schema != EVALUATION_CHECKPOINT_SCHEMA:
        # A training checkpoint would make this look like an exact resume of a
        # different run; only the frozen evaluation-only artifact is a legal
        # fresh-run source.
        raise PolicyImportError(
            "policy import accepts only an evaluation-only checkpoint "
            f"({EVALUATION_CHECKPOINT_SCHEMA}); got {schema!r}"
        )
    if payload.get("training_resume_allowed") is not False:
        raise PolicyImportError(
            "policy import requires training_resume_allowed=False; an artifact that claims "
            "training resume must use the exact-resume path instead"
        )
    if str(payload.get("algorithm")) != str(trainer.algorithm):
        raise PolicyImportError(
            f"policy import algorithm mismatch: {payload.get('algorithm')!r} != {trainer.algorithm!r}"
        )
    if str(payload.get("architecture")) != str(trainer.cfg.model.architecture):
        raise PolicyImportError(
            "policy import architecture mismatch: "
            f"{payload.get('architecture')!r} != {trainer.cfg.model.architecture!r}"
        )
    if str(payload.get("protocol_version")) != str(trainer.cfg.game.protocol_version):
        raise PolicyImportError(
            "policy import protocol mismatch: "
            f"{payload.get('protocol_version')!r} != {trainer.cfg.game.protocol_version!r}"
        )
    backend_identity = payload.get("backend_identity")
    if not isinstance(backend_identity, Mapping) or not backend_identity.get("hash"):
        raise PolicyImportError("policy import source lacks a backend identity hash")
    if canonical_json_hash({k: v for k, v in backend_identity.items() if k != "hash"}) != backend_identity.get("hash"):
        raise PolicyImportError("policy import source backend identity hash is inconsistent")

    resolved = payload.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise PolicyImportError("policy import source lacks resolved_config")
    if str(payload.get("resolved_config_hash")) != canonical_json_hash(resolved):
        raise PolicyImportError("policy import source resolved_config hash is inconsistent")

    source_model = resolved.get("model")
    if not isinstance(source_model, Mapping):
        raise PolicyImportError("policy import source resolved_config lacks the model block")
    source_features = _model_identity(trainer.agent_resume)
    for name, expected in source_features.items():
        actual = source_model.get(name)
        if actual is None:
            raise PolicyImportError(f"policy import source model config is missing {name!r}")
        if int(actual) != int(expected):
            raise PolicyImportError(
                f"policy import model feature mismatch for {name!r}: {actual!r} != {expected!r}"
            )

    source_ppo = resolved.get("ppo")
    if not isinstance(source_ppo, Mapping):
        raise PolicyImportError("policy import source resolved_config lacks the ppo block")
    # Read the *real* nesting rather than assuming a flat schema.
    if str(source_ppo.get("training_semantics_profile")) != str(trainer.training_semantics_profile):
        raise PolicyImportError(
            "policy import training semantics mismatch: "
            f"{source_ppo.get('training_semantics_profile')!r} != "
            f"{trainer.training_semantics_profile!r}"
        )
    if str(source_ppo.get("policy_regularizer_mode")) != str(trainer.policy_regularizer_mode):
        raise PolicyImportError(
            "policy import policy regularizer mismatch: "
            f"{source_ppo.get('policy_regularizer_mode')!r} != "
            f"{trainer.policy_regularizer_mode!r}"
        )
    return {
        "model": dict(source_model),
        "training_semantics_profile": str(source_ppo.get("training_semantics_profile")),
        "policy_regularizer_mode": str(source_ppo.get("policy_regularizer_mode")),
        "reward_semantics": resolved.get("reward_semantics"),
        "reward_scale": resolved.get("reward_scale"),
    }


def _split_state(
    trainer, source_state: Mapping[str, object], *, allowed_drop_prefixes: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not isinstance(source_state, Mapping) or not source_state:
        raise PolicyImportError("policy import source lacks a non-empty agent_state_dict")
    if any(not isinstance(value, torch.Tensor) for value in source_state.values()):
        raise PolicyImportError("policy import source state dict contains non-tensor entries")
    target_state = trainer.agent_resume.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    dropped = [key for key in unexpected if key.startswith(tuple(allowed_drop_prefixes))] if allowed_drop_prefixes else []
    disallowed_unexpected = sorted(set(unexpected) - set(dropped))
    if missing or disallowed_unexpected:
        raise PolicyImportError(
            "policy import state dict is incompatible: "
            f"missing={missing}, unexpected={disallowed_unexpected}"
        )
    kept: dict[str, torch.Tensor] = {}
    for key, target_value in target_state.items():
        source_value = source_state[key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise PolicyImportError(
                f"policy import tensor shape mismatch for {key!r}: "
                f"{tuple(source_value.shape)} != {tuple(target_value.shape)}"
            )
        if source_value.dtype != target_value.dtype:
            raise PolicyImportError(
                f"policy import tensor dtype mismatch for {key!r}: "
                f"{source_value.dtype} != {target_value.dtype}"
            )
        kept[key] = source_value
    return kept, dropped


def import_policy_as_fresh_run(
    trainer,
    payload: Mapping[str, object],
    *,
    source_role: str,
    allowed_drop_prefixes: tuple[str, ...] = (),
    curriculum_offset: int | None = None,
    equivalence_rollout=None,
) -> dict[str, object]:
    """Import ``payload`` as the *initial* policy of a brand-new training run.

    Returns a versioned migration manifest.  The trainer must be unstarted
    (iteration/durable/learner counters all zero); anything else would silently
    mix a historical training line with a new one.
    """
    if not str(source_role).strip():
        raise PolicyImportError("policy import requires a non-empty source_role")
    require_healthy = getattr(trainer, "_require_healthy", None)
    if callable(require_healthy):
        require_healthy()
    if trainer.iteration_status in {"running", "loading"} or getattr(trainer, "active_iteration", None) is not None:
        raise PolicyImportError("policy import is forbidden while an iteration/load is active")
    counters = (
        int(trainer.iteration), int(trainer.durable_updates),
        int(trainer.learner_updates), int(trainer.completed_games_total),
    )
    if any(counters):
        raise PolicyImportError(
            "policy import requires an unstarted trainer (iteration/durable/learner/games all "
            f"zero); got {counters}"
        )
    if getattr(trainer, "policy_router", None) is not None:
        pool = trainer.opponent_pool
        if pool is not None and (int(pool._assignment_counter) != 0 or int(pool._history_count) != 0):
            raise PolicyImportError(
                "policy import requires a freshly constructed opponent pool with no assignments"
            )

    unknown = tuple(prefix for prefix in allowed_drop_prefixes if prefix not in ALLOWED_DROP_PREFIXES)
    if unknown:
        raise PolicyImportError(
            f"policy import drop whitelist may only contain {ALLOWED_DROP_PREFIXES}; got {unknown}"
        )

    semantics = _validate_source_payload(trainer, payload)
    source_state = payload.get("agent_state_dict")
    kept, dropped = _split_state(trainer, source_state, allowed_drop_prefixes=allowed_drop_prefixes)

    rng_before = capture_rng_state()
    rng_hash_before = canonical_json_hash(rng_before)
    try:
        filtered = dict(kept)
        trainer.agent_resume.load_state_dict(filtered, strict=True)
        trainer.agent_merged.load_state_dict(filtered, strict=True)
    finally:
        restore_rng_state(rng_before)
    rng_hash_after = canonical_json_hash(capture_rng_state())
    if rng_hash_after != rng_hash_before:
        raise PolicyImportError("policy import consumed the learner RNG stream")

    # Bitwise proof: every kept tensor is exactly the source tensor, in both the
    # trainable and the merged model.
    kept_records = []
    for key in sorted(kept):
        source_value = kept[key].detach().to(trainer.device)
        resume_value = trainer.agent_resume.state_dict()[key].detach()
        merged_value = trainer.agent_merged.state_dict()[key].detach()
        if not torch.equal(resume_value, source_value):
            raise PolicyImportError(f"imported tensor {key!r} is not bitwise equal to the source")
        if not torch.equal(merged_value, source_value):
            raise PolicyImportError(f"merged tensor {key!r} is not bitwise equal to the source")
        kept_records.append({
            "name": key,
            "shape": list(kept[key].shape),
            "dtype": str(kept[key].dtype),
            "sha256": _tensor_digest(source_value),
        })

    # Freshly initialized state, recorded explicitly.
    trainer.iteration = 0
    trainer.durable_updates = 0
    trainer.learner_updates = 0
    trainer.durable_learner_updates = 0
    trainer.completed_games_total = 0
    trainer.iteration_status = "completed"
    trainer.active_iteration = None
    trainer.failed = False
    trainer.failure = None
    trainer.amp_skipped_steps = 0
    if hasattr(trainer, "grad_scaler"):
        # A fresh GradScaler is equivalent to resetting the scale/state.
        trainer.grad_scaler = torch.amp.GradScaler(
            trainer.amp_device_type,
            enabled=bool(trainer.amp_scaler_enabled),
            **(
                {"init_scale": float(trainer.cfg.amp_init_scale)}
                if getattr(trainer.cfg, "amp_init_scale", None) is not None else {}
            ),
        )
    trainer.merge_updater.set_reference_iteration(0)
    trainer._frozen_hashes = None
    if curriculum_offset is not None:
        offset = int(curriculum_offset)
        if offset < 0:
            raise PolicyImportError("curriculum_offset must be non-negative")
        trainer.curriculum_offset = offset
    trainer.sampler.set_curriculum_iteration(int(trainer.curriculum_offset))
    restore_rng_state(rng_before)

    # Optimizer coverage: every trainable parameter exactly once, no frozen pool
    # parameter anywhere.
    parameter_ids = _trainable_parameter_ids(trainer.agent_optimizer)
    trainable_ids = [id(parameter) for parameter in trainer.agent_resume.parameters()]
    if sorted(parameter_ids) != sorted(trainable_ids) or len(set(parameter_ids)) != len(parameter_ids):
        raise PolicyImportError(
            "policy import optimizer does not cover every trainable parameter exactly once"
        )
    verify_frozen = getattr(trainer, "_verify_frozen_excluded_from_optimizer", None)
    if callable(verify_frozen):
        verify_frozen(trainer.agent_optimizer)

    equivalence = None
    if equivalence_rollout is not None:
        equivalence = assert_ntp_removal_preserves_inference(
            trainer, source_state=kept, rollout=equivalence_rollout,
        )

    manifest: dict[str, object] = {
        "schema": POLICY_IMPORT_MANIFEST_SCHEMA,
        "migration_version": POLICY_IMPORT_SCHEMA,
        "source_role": str(source_role),
        "source_identity": _source_identity(payload),
        "target_identity": _target_identity(
            trainer, model_identity=_model_identity(trainer.agent_resume),
        ),
        "source_semantics": semantics,
        "logical_origin": {
            "origin_iteration": int(payload.get("iteration", -1)),
            "new_iterations": 0,
            "curriculum_offset": int(trainer.curriculum_offset),
            "note": (
                "the historical origin_iteration is provenance only; this run's counters start "
                "at zero and must never be reported as an iteration-"
                f"{int(payload.get('iteration', -1))} resume"
            ),
        },
        "migration": {
            "version": POLICY_IMPORT_SCHEMA,
            "allowed_drop_prefixes": list(ALLOWED_DROP_PREFIXES),
            "applied_drop_prefixes": list(allowed_drop_prefixes),
            "dropped_keys": list(dropped),
            "drop_reason": (
                "NTP auxiliary training head is disabled for this arm; it is not part of the "
                "policy/value inference graph"
            ) if dropped else None,
        },
        "kept_keys": kept_records,
        "kept_key_count": len(kept_records),
        "dropped_key_count": len(dropped),
        "bitwise_verified": True,
        "rng_stream_untouched": True,
        "rng_state_hash": rng_hash_before,
        "optimizer": {
            "trainable_parameter_count": len(trainable_ids),
            "covered_exactly_once": True,
            "frozen_pool_parameters_excluded": True,
        },
        "resets": {
            "optimizer_state": True,
            "amp_scaler_state": True,
            "merged_policy_state": True,
            "sampler_state": True,
            "opponent_pool_assignment": True,
            "counters": True,
            "curriculum_offset": int(trainer.curriculum_offset),
        },
        "inference_equivalence": equivalence,
    }
    manifest["hash"] = canonical_json_hash(manifest)
    trainer.policy_import_manifest = manifest
    trainer.import_origin_iteration = int(payload.get("iteration", -1))
    return manifest


def assert_ntp_removal_preserves_inference(trainer, *, source_state, rollout) -> dict[str, object]:
    """Prove dropping ``ntp_head.*`` leaves policy/value outputs unchanged.

    A reference ``Agent`` with the NTP head *enabled* receives the same kept
    tensors (plus the original NTP tensors when the source carries them) and
    both models run under ``torch.no_grad()`` in eval mode on the same rollout.
    The comparison is exact (``torch.equal``) because both sides are FP32 with
    no autocast and share every parameter on the policy/value path.
    """
    from guandan_marvel.model import Agent

    cfg = trainer.cfg
    reference = Agent(cfg, enable_value_head=True, enable_ntp_head=True).to(trainer.device)
    reference_state = reference.state_dict()
    for key, value in source_state.items():
        if key in reference_state:
            reference_state[key] = value.detach().to(reference_state[key].dtype)
    reference.load_state_dict(reference_state, strict=True)
    reference.eval()
    was_training = bool(trainer.agent_resume.training)
    trainer.agent_resume.eval()
    with torch.no_grad():
        left = reference(rollout=rollout)
        right = trainer.agent_resume(rollout=rollout)
    differences = {}
    for name in ("avail_probs", "avail_log_probs", "player_state_value", "state_values"):
        a = getattr(left, name, None)
        b = getattr(right, name, None)
        if (a is None) != (b is None):
            raise PolicyImportError(f"NTP removal changed whether {name!r} is produced")
        if a is None:
            continue
        if not torch.equal(a.detach().float(), b.detach().float()):
            raise PolicyImportError(
                f"NTP-head removal changed the {name!r} inference output; the auxiliary head is "
                "not allowed to be on the policy/value path"
            )
        differences[name] = "exact"
    # Restore the mode we found, not an unconditional ``train()``: the import
    # contract promises not to perturb the fresh trainer's module state.
    trainer.agent_resume.train(was_training)
    return {
        "schema": "guandan-marvel-policy-import-inference-equivalence-v1",
        "compared_fields": differences,
        "tolerance": "exact (torch.equal, FP32, no autocast)",
    }


def initial_state_fingerprint(trainer, *, include_ntp: bool = True) -> dict[str, object]:
    """Hash the current initial policy so two arms can be proven identical.

    ``include_ntp=False`` compares only the shared (non-dropped) tensors, which
    is exactly the intersection a control (NTP kept) and a candidate (NTP
    dropped) arm must share.
    """
    state = trainer.agent_resume.state_dict()
    records = {
        key: _tensor_digest(value)
        for key, value in state.items()
        if include_ntp or not key.startswith(ALLOWED_DROP_PREFIXES)
    }
    return {
        "schema": "guandan-marvel-policy-import-initial-fingerprint-v1",
        "tensor_count": len(records),
        "tensors": records,
        "hash": canonical_json_hash(records),
    }


__all__ = [
    "ALLOWED_DROP_PREFIXES", "POLICY_IMPORT_MANIFEST_SCHEMA", "POLICY_IMPORT_SCHEMA",
    "PolicyImportError", "assert_ntp_removal_preserves_inference",
    "import_policy_as_fresh_run", "initial_state_fingerprint",
]
