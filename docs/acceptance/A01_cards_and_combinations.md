# A01 cards and combinations acceptance

## Scope

Validate the independent two-deck physical-card model and the ten v1 combination families before any round engine is built.

## Required command

```powershell
python scripts/run_acceptance.py --stage A01
```

## Required evidence

- 108 unique physical cards and 27-card partition conservation.
- Exact-five straight, exact-three pair sequence and exact-two triple sequence.
- A-low sequence windows and rejection of JQKA2/KA2 wraps.
- Same-joker pairs, four kings, and rejection of the two-joker rocket.
- Straight flush and bomb hierarchy.
- Level/wild declarations, non-joker substitution, and physical assignment preservation.
- Level-context mismatch and forged-declaration rejection.
- Unit, property and acceptance tests pass with no skipped/xfail tests.

## Manual review gate

The supervising agent must inspect the combination API and diff, run focused and full tests, and confirm that tests are demand-driven rather than merely enumerating the implementation's current enum members.
