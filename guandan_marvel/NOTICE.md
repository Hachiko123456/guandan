# guandan_marvel source notice

`guandan_marvel` is a game-agnostic derivative of selected MARVEL modules from:

- project: AtaraxosAI `doudizhu`
- local audited source: `D:\project\doudizhu`
- source commit: `acdcf199cf422e2b683cb22142b8e7cb33b14460`
- license: MIT, Copyright (c) 2026 AtaraxosAI

The complete source license is distributed as `guandan_marvel/LICENSE`.

This derivative deliberately uses the independent top-level package name
`guandan_marvel`. It does not import the source project's `marvel` package or
`nanopokerlib`, and it contains no Dou Dizhu game rules or evaluator agents.

Detailed file-by-file provenance and rewrite classification is tracked in
`docs/marvel_migration/M01_SOURCE_AUDIT.md` and
`docs/marvel_migration/M02_SOURCE_AUDIT.md`. The M02 derivative additions
include `guandan_marvel/env_batch.py` and `guandan_marvel/runtime/sampler.py`;
the GuanDan-specific `guandan_marvel/game.py` adapter is newly written over
the accepted local Python rule oracle.
