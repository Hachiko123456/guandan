# STATUS.yaml Schema

Each stage records `status`, `implementation`, `tests`, `accepted`, `commit`, `evidence`, and `blockers`.

Rules:

- `accepted: true` requires `status: accepted`, a full Git SHA, and evidence paths.
- Automated reports always keep `accepted: false`.
- Partial code remains `accepted: false` even when focused tests pass.
- Blockers must be concrete and reproducible.
