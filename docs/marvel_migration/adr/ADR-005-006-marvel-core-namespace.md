# ADR-005/006: MARVEL core package namespace and source integration

- Date: 2026-10-08
- Status: accepted
- Source baseline: `D:\project\doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460`

## Context

The reference project installs a top-level `marvel` package. The Guandan project may be
installed in the same interpreter environment, so reusing that name can silently import
the wrong implementation depending on `sys.path` order. The migrated core must also run
without a dependency on the absolute reference-project path.

## Options considered

1. Keep top-level `marvel` and require environment isolation.
2. Vendor a renamed package inside this repository.
3. Depend on a local/shared checkout of the reference project.

## Decision

- **D005:** use the independent top-level package `guandan_marvel`.
- **D006:** vendor a license-preserving derivative in this repository, with explicit
  per-file provenance and no runtime import/path dependency on `D:\project\doudizhu`.

## Evidence

- `tests/marvel_core/test_core.py::test_independent_namespace_keeps_legacy_guandan_importable`
  imports a separate `marvel` sentinel beside `guandan_marvel` and `guandan`.
- `guandan_marvel/NOTICE.md` and `guandan_marvel/LICENSE` retain source and MIT terms.
- `docs/marvel_migration/M01_SOURCE_AUDIT.md` records file-level source/rewrite status.

## Consequences

- Imports differ from the reference project, but collision risk is removed.
- Kaggle packaging remains a single repository build.
- Upstream changes require a reviewed rebase/port rather than an implicit shared checkout.
- M02 must inject a versioned Guandan `GameSpec`; no game is auto-discovered.

## Rollback

A future standalone distribution could expose a compatibility facade named `marvel`, but
only in an isolated artifact. The repository-internal canonical package remains
`guandan_marvel` to protect mixed environments.
