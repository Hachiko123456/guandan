# Architecture

## Runtime layers

1. `guandan.cards` and `guandan.combos` own card identity, counts, recognition and comparison.
2. `guandan.state` owns one complete omniscient game state.
3. `guandan.action_state` exposes only the active player's stepwise action-prefix state and commits complete actions.
4. `guandan.environment` provides one environment; `guandan.env_batch` provides a fixed-size batch adapter.
5. `guandan.encoding` converts observations and available actions into fixed-size integer/float arrays.
6. `training` consumes the environment contract and owns policy/value learning.

The policy observation is a projection of the omniscient state. The omniscient state is never passed to the policy. Training-only labels are kept in a separate path.

## Backend boundary

The first backend is pure Python. The environment interface must remain backend-neutral so that a future optimized backend can replace only the rules/environment layer.

## Stepwise action boundary

A token step may extend an action prefix. Only a committed action mutates public game state, consumes cards, and advances turn/trick state. Training code must record both token steps and committed game steps.
