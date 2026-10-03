# Training protocol

## Initial order

1. Validate A00-A04 with deterministic and property tests.
2. Run policy forward/rollout smoke tests with compile disabled.
3. Run IPPO smoke training.
4. Run VRPO smoke training.
5. Validate checkpoint save/load and resume.
6. Benchmark environment, model and learner separately.
7. Package the same commands for Kaggle.

## Step accounting

The environment distinguishes `token_step` from `committed_game_step`. Prefix extension has zero immediate game reward and does not advance the public turn. A commit may advance the trick/turn and may deliver terminal team rewards. The training adapter must document how GAE and masks treat both kinds of rows.

## Reproducibility

Every smoke/acceptance report records the Python executable, package versions, game/rules/action/observation protocol versions, seed, Git commit and command line.
