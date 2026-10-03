# A04 Observation and Batch Environment

Command: `python scripts/run_acceptance.py --stage A04`.

Implement policy-safe observation encoders, `GuandanEnv`, and `GuandanEnvBatch`. Test fixed shapes/dtypes/masks, no hidden-hand leakage, single-vs-batch equivalence, independent RNG/state, terminal/reset semantics, serialization, and overflow errors.
