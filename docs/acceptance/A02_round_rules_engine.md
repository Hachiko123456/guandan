# A02 round rules engine — partial milestone

## Current implemented slice

- Deterministic two-deck deal and full physical-card partition.
- Four seats and fixed teams.
- Atomic committed single-card/combination play validation.
- Active-seat turn progression.
- Pass and trick reset.
- Invalid-action state preservation.
- Card-conservation invariant checks.
- Deterministic replay via seeded deal and serialized core state.

## Not yet accepted

The full A02 gate is intentionally blocked until the engine implements and tests:

- complete legal atomic-action enumeration;
- all lead/follow combination generation, including wild alternatives;
- finished-seat skipping and complete 接风 behavior;
- tribute, anti-tribute and return phases;
- complete ranking, double-down and terminal reward transitions;
- 100+ directed rule fixtures and bounded random legal rounds;
- a demand-driven `tests/acceptance/test_a02_rules.py`.

The current files are an implementation slice, not an A02 acceptance result.
