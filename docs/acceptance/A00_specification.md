# A00 specification acceptance

## Scope

Freeze the single-hand v1 rule, public/private information boundary, Scheme B action protocol, observation dimensions and training-time step semantics.

## Required command

```powershell
python scripts/run_acceptance.py --stage A00
```

## Required evidence

- `docs/rules.md` contains the explicit v1 project policy and source-status section.
- `docs/environment_contract.md` contains the fixed v1 dimensions and token ranges.
- `docs/action_protocol.md` fixes scalar-token, COMMIT, overflow, reset and token-level PPO semantics.
- A00 tests pass with no skipped/xfail tests.
- The report records Git commit, dirty files and input fingerprints.

## Manual review gate

The supervising agent must inspect the documents and confirm that open match-level features are explicitly deferred rather than silently omitted. A passing test report alone never sets `accepted: true`.
