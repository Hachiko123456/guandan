# Completion criteria

## Global acceptance rule

A stage is accepted only when all of the following are true:

1. Its implementation is present in the current Git commit.
2. Its required tests pass in the `guandan_train` environment.
3. Its acceptance command emits a passing JSON report.
4. Its documented invariants and information-boundary checks pass.
5. The supervising agent has inspected the diff and compared it with the original requirements.

`implemented` and `tested` are not equivalent to `accepted`.

## A00 Specification

- Rules, public-information boundaries, action protocol, observation protocol, and reward semantics are documented.
- Open rule decisions are explicitly listed; none are silently guessed.

## A01 Cards and combinations

- Two-deck card identity/counting is lossless.
- All approved GuanDan combination families have deterministic recognition and comparison.
- Invalid combinations are rejected.
- Independent reference enumeration agrees with the production combination layer on exhaustive small-card fixtures.

## A02 Round rules engine

- Deal, play, pass, lead, follow, trick reset, tribute/return, team ranking, and terminal rewards are implemented.
- Card conservation and deterministic replay pass.
- At least 100 directed rule cases and 10,000 seeded legal random rounds pass without invariant violations.

## A03 Stepwise action protocol

- Every accepted action prefix can reach at least one legal committed action.
- No legal committed action is omitted or duplicated by canonical encoding.
- Prefix actions do not mutate public state until commit.
- Token-step versus committed-game-step accounting is explicit.

## A04 Observation and batch environment

- Single environment and batch environment agree under identical seeds/actions.
- Policy observations never expose opponents' private cards.
- Shapes, dtypes, masks, and terminal reset semantics are validated.

## A05 Training integration

- IPPO and VRPO smoke runs complete at least 100 updates with finite loss/gradients.
- Checkpoint save/load and resume are tested.
- Invalid-action probability is zero and legal probability distributions normalize.

## A06 Evaluation

- Fixed-seed, seat-swapped evaluation reports team metrics and confidence metadata.
- Training and evaluation deals are separated.
- Evaluation errors do not silently fall back to an alternate agent.

## A07 Kaggle

- Clean-environment dependency check, rules smoke test, short training, timed checkpoint save, and resume all work.
- The script detects actual Python/Torch/GPU versions rather than assuming hardware.

## A08 Belief/search (later)

- Belief labels/samples obey information constraints and card conservation.
- Search returns legal actions, obeys budget, and has a direct-policy comparison.
