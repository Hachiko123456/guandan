# SPDX-License-Identifier: MIT
from .checkpoints import (
    CheckpointCompatibilityError, PRE_REPAIR_TRAINING_CHECKPOINT_SCHEMA,
    PRE_TRAINING_PROVENANCE_CHECKPOINT_SCHEMA,
    ResumeDecision, TRAINING_CHECKPOINT_SCHEMA, TRAINING_SEMANTIC_CONTRACT_SCHEMA,
    TrainingCheckpointStore, build_execution_fingerprint,
    build_semantic_configuration, build_semantic_contract, classify_resume,
    compute_source_fingerprint,
)
from .gae import GAEFromQValueOutput, GAEFromValueOutput, compute_gae_from_qvalue, compute_gae_from_value
from .ippo import IPPOTrainer
from .losses import ClippedPGLossOutput, QValueLossOutput, ValueLossOutput, compute_clipped_pg_loss, compute_qvalue_loss, compute_value_loss
from .metrics import rollout_metrics
from .schedules import ScheduleState, compute_agent_schedule, compute_stable_decay_scale
from .smoke import PythonReferenceSmokeGame
from .validation import TrajectoryValidation, validate_complete_trajectories
from .vrpo import VRPOTrainer, VRPO_Q_TARGET_CONTRACT

__all__ = [
    "CheckpointCompatibilityError", "ClippedPGLossOutput", "GAEFromQValueOutput",
    "GAEFromValueOutput", "IPPOTrainer", "PythonReferenceSmokeGame",
    "PRE_REPAIR_TRAINING_CHECKPOINT_SCHEMA", "PRE_TRAINING_PROVENANCE_CHECKPOINT_SCHEMA",
    "QValueLossOutput", "ResumeDecision", "ScheduleState",
    "TRAINING_CHECKPOINT_SCHEMA", "TRAINING_SEMANTIC_CONTRACT_SCHEMA",
    "TrainingCheckpointStore", "TrajectoryValidation", "VRPOTrainer",
    "build_execution_fingerprint", "build_semantic_configuration",
    "build_semantic_contract", "compute_source_fingerprint",
    "VRPO_Q_TARGET_CONTRACT", "ValueLossOutput", "classify_resume",
    "compute_agent_schedule", "compute_clipped_pg_loss", "compute_gae_from_qvalue",
    "compute_gae_from_value", "compute_qvalue_loss", "compute_stable_decay_scale",
    "compute_value_loss", "rollout_metrics", "validate_complete_trajectories",
]
