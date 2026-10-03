# A02 round rules engine — acceptance record

## Scope

A02 implements one deterministic four-seat, two-team, two-deck GuanDan hand. It
is an atomic committed-action layer only; Scheme-B prefix/token construction
starts at A03 and is not imported into this engine.

## Implemented requirements

- deterministic 108-card deal, physical-card conservation, transactional clone/
  validate/commit updates, and deterministic serialization/replay;
- complete canonical non-pass play enumeration for all A01 families, including
  all physical selections, wild assignment declarations, lead/follow filtering,
  bomb ladder, four-kings, and PASS in follow states;
- active-seat progression, finished-seat skipping, trick reset, and 接风 to the
  emptied player's partner (or the next active seat when the partner is empty);
- prior-hand `HEAD_THIRD`, `HEAD_LAST`, and `DOUBLE_DOWN` tribute setup;
  donor-side anti-tribute, mandatory highest eligible natural tribute, lower-seat
  deterministic tie break, explicit return choices, and resolved public
  transfers;
- immediate finish ranking, cyclic deterministic assignment of implicit losing
  ranks under double-down/three-finish termination, terminal outcome class,
  winner team, player/team +/-1 reward, and finish-step metadata;
- forged declarations, duplicate/missing cards, wrong seats, invalid phases,
  illegal passes, and post-terminal actions fail without mutating the state.

## Demand-driven evidence

- `tests/unit/test_round.py` covers deal, turn, pass/reset, conservation,
  invalid-action atomicity, finished-seat skipping, 接风, ranking, terminal
  reward, tribute, anti-tribute, return, and phase guards.
- `tests/property/test_round_invariants.py` covers seeded conservation and
  10,000 seeded bounded legal one-trick trajectories. The property loop uses
  the complete physical deck and checks invariants after every committed step.
- `tests/acceptance/test_a02_rules.py` contains more than 100 independently
  named v1 rule-case IDs and executes demand-driven family, follow, ranking,
  reward, tribute, return, conservation, invalid-action, and replay checks.
- A01 acceptance remains green after the A02 sequence-window comparison fix:
  sequences compare by natural window order, while rank groups retain level
  strength ordering.

## Explicit v1 policy applied

The implementation follows the final policy in `docs/rules.md`: exact fixed
sequence windows, no JQKA2/KA234 wrap, heart-level wilds only, no wildcard
jokers, the documented bomb ladder, donor-side anti-tribute, natural return
cards in printed ranks 2..10 with heart-level wilds excluded, deterministic
seat tie breaks, and shared zero-sum terminal reward.

## Acceptance commands

```powershell
C:\Users\yhx\.conda\envs\guandan_train\python.exe -m pytest -q
C:\Users\yhx\.conda\envs\guandan_train\python.exe scripts/run_acceptance.py --stage A02
```

The automated report's `accepted` field remains false by design. The supervising
agent sets the formal stage status only after inspecting the final diff,
requirements, tests, and this record.
