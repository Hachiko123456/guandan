# Master Execution Plan

## Mission

Complete the independent pure-Python GuanDan training project in this repository. The rules and model pipeline must not import, copy, or depend on FableDan. The final target is reproducible Kaggle training with optional belief/search extensions.

## Non-negotiable constraints

- Python/PyTorch only; no C++ backend unless the user explicitly changes this requirement.
- Keep `D:\project\doudizhu` and `D:\project\FableDan` untouched.
- Use `C:\Users\yhx\.conda\envs\guandan_train\python.exe` for local validation.
- Never accept a subagent conclusion without diff review, focused tests, full tests, and requirements review.
- Automated reports never set `accepted: true`; only the supervising main agent may do that.
- Never silently resolve a rule ambiguity.

## Dependency graph

```text
A00 -> A01 -> A02 -> A03 -> A04 -> A05 -> A06 -> A07 -> A08
```

Only the first stage whose `accepted` field is false should be actively implemented.

## Stage map

| Stage | Scope | Depends on | Gate |
|---|---|---|---|
| A00 | Rules, protocols, governance, evidence runner | none | Versioned specifications and governance tests |
| A01 | Cards, levels, wilds, ten combinations | A00 | Conservation, declaration, comparison and wild tests |
| A02 | Atomic round engine, trick, 接风, tribute/return, ranking/reward | A01 | Directed rules, random rounds, invariants, terminal tests |
| A03 | Scheme B prefix state and COMMIT | A02 | Complete prefixes, canonical actions, counters |
| A04 | Policy observations and batch environment | A02/A03 | Shapes, masks, batch equivalence, no information leak |
| A05 | IPPO/VRPO integration | A04 | Smoke updates, finite gradients, checkpoint resume |
| A06 | Fixed-deal evaluation | A05 | Seat-swapped metrics and baselines |
| A07 | Kaggle install/train/save/resume | A05/A06 | Clean Kaggle session evidence |
| A08 | Belief sampling and search | A06/A07 | Information-safe samples and legal budgeted search |

## Universal stage gate

Each stage requires implementation, focused tests, property/integration tests where relevant, a demand-driven acceptance test, a report from `scripts/run_acceptance.py`, main-agent diff review, a full test run, and a Git commit recorded in `project_status/STATUS.yaml`. A passing subset is not acceptance.

## Continuation rule

At every new turn, read `project_status/STATUS.yaml`, locate the first unaccepted stage, read its acceptance document, and continue that stage. Do not restart accepted stages or require a new user prompt for each stage. Stop only for a genuine unresolved product/rule decision, external permission, or repeated blocker with evidence.
