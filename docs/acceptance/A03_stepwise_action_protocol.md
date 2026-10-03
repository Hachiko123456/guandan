# A03 stepwise action protocol — acceptance matrix

**Status:** draft requirements only; not implemented, tested, or accepted. This document does not change A02.

## Scope boundary

A03 is the Scheme-B stepwise layer over the accepted A02 atomic rules engine. Its public entry is `guandan.action_state.StepwiseActionState`; it owns private prefix construction, canonical token paths, trial validation/rollback, and commit accounting. It does **not** introduce or require `guandan.environment`, `GameConfig`, observation tensors, batch execution, or training integration. Those are A04/later scope.

## Public A03 API

From `guandan.action_state`:

- `StepwiseActionState(round_state, *, max_action_tokens=32, max_legal_next_tokens=256, max_token_steps=4096)`; `deal(...)`; `prefix`; `done`; `committed_step`; `token_step`; `truncated`.
- `step(token_id) -> ProtocolStepResult`; `clone()`; `serialize()`/`deserialize()`.
- `legal_next_tokens()`/`legal_tokens()`; `legal_token_rows()`; `legal_token_arrays()`.
- `ActionPrefix` with `phase`, `family`, `declared_ranks`, `declared_length`, `declared_suit`, `selected_card_ids`, `wild_assignments`, `next_expected_kind`, `complete`, and token history.
- `TokenCodec` for family, rank, suit, length, physical-card, and wild-token encode/decode.
- `ProtocolStepResult` with `observation: ActionPrefix`, `rewards: float32[4]`, `done`, `truncated`, `is_commit`, protocol `CommittedAction | None`, `token_step`, `committed_step`, and `info`.
- Protocol `CommittedAction` with actor/phase/kind, family, physical `card_ids`, declared ranks/suit, wild assignments, public index, and its A02 atomic action reference.
- `LegalTokenRow(token_id, kind, value, is_commit)` plus padded legal-token arrays with a mask; invalid rows are masked, non-commit, and zero-valued.
- `ProtocolError` for capacity/configuration failures; A02 `IllegalActionError` remains for illegal submitted tokens; post-terminal/truncated steps retain the existing terminal error contract.

## V1 token contract

| Tokens | Normative meaning |
|---|---|
| `0`, `1` | `PAD`, `BOS`; contextual only; neither is an action decision |
| `2..5` | phase context (`LEAD_PLAY`, `FOLLOW_PLAY`, `TRIBUTE`, `RETURN`); contextual only, not emitted by the normative FAMILY-first action grammar |
| `6`, `7` | `PASS`, `COMMIT` |
| `8..17` | ten family ordinals: SINGLE, PAIR, TRIPLE, FULL_HOUSE, STRAIGHT, PAIR_SEQUENCE, TRIPLE_SEQUENCE, RANK_BOMB, STRAIGHT_FLUSH, FOUR_KINGS |
| `18..32` | rank ordinal mapping `3..A, 2, SMALL_JOKER, BIG_JOKER` |
| `33..37` | `Suit` enum values |
| `38..48` | length values `0..10` |
| `49..63`, `224..255` | reserved; never legal |
| `64..171` | physical card IDs `0..107` |
| `172..223` | non-joker wild target rank/suit tokens |

`FAMILY` is the first action token for every non-pass play. `PASS` is a one-token commit only in `FOLLOW_PLAY` with a current winner. `TRIBUTE` and `RETURN` are `CARD, COMMIT`; no wildcard assignment is allowed. `CANCEL` is not exposed in v1.

## Acceptance matrix against `docs/action_protocol.md`

| Clause | A03 requirement | Evidence/test obligation |
|---|---|---|
| §§1–2 lifecycle | Every exposed prefix has a legal completion; prefix is private; only `COMMIT`/`PASS` mutates A02 state. | `test_no_dead_prefix_is_exposed`; public-state snapshot before/after prefix. |
| §§3–4 vocabulary/grammar | Implement the four phases and all ten families with exact v1 ranges above; `COMMIT` only at an exact legal completion. | `test_public_api_and_v1_token_ranges`; phase/family fixture coverage. |
| §4 canonical order | Full house emits triple group then pair; sequence groups increase by natural printed rank `2..A`; cards within each physical group ascend by `card_id`. | Canonical path/set comparison and permutation fixtures. |
| §4 wild encoding | Every selected wild physical card emits exactly one `WILD` token, including its natural target, in ascending selected wild `card_id`; two wilds therefore have unambiguous positions. Same rank/suit natural-vs-substitute semantic declarations canonicalize to the natural declaration. | Wild/natural fixtures; no duplicate canonical paths; two-wild case. |
| §5 prefix state | `ActionPrefix` mirrors the private token construction and reports the next expected kind/completeness. | Prefix field and serialization assertions. |
| §6 commit result | Every successful commit returns one protocol `CommittedAction` linked to exactly one A02 atomic action. | `test_each_exposed_commit_is_one_legal_canonical_action`. |
| §7 accounting/reward | Accepted token: `token_step += 1`; prefix: `committed_step` unchanged, zero reward. Commit/PASS: both advance once; only terminal commit may carry A02 reward. | Counter, PASS, prefix-reward, terminal-reward tests. |
| §8 legal rows | `legal_token_rows()` is complete; `legal_token_arrays()` is fixed-capacity and padded; no top-N selection or silent truncation. | Row/array mask and set-completeness tests. |
| §9 equality | Reachable committed-action keys equal an independent complete-action enumerator; no card-order/group-order duplicates. | Reduced-deck exhaustive plus sampled full-deck equality. |
| §10 failure safety | Wrong phase/token, incomplete commit, duplicate card, joker target, non-beating play, and illegal exchange card fail atomically. | Exception and byte-for-byte state rollback tests. |
| §10 overflow | `max_action_tokens=32` and `max_legal_next_tokens=256` overflow is `ProtocolError`, never truncation. | Deliberate boundary/overflow fixtures and diagnostic assertions. |
| §§7/10 truncation | `max_token_steps` sets `truncated`, preserves zero win/loss reward, records `info["termination_reason"]`, and rejects later steps. | Truncation test. |
| §§11–12 | Serialization/clone preserves prefix, counters, legal rows/arrays, and next committed result; token-level PPO/GAE policy remains deferred. | Round-trip continuation test; no training tests in A03. |

## Trial validation and rollback

`step(token_id)` is transactional. Snapshot prefix, A02 public state, counters, `done`, and `truncated`; validate the token against the current legal set; trial-extend the prefix or trial-apply the commit on a clone. Recompute the next phase/active player and its legal rows/arrays **after the transition**. If validation fails, the candidate is incomplete/ambiguous, or either capacity overflows before or after the next transition, raise `IllegalActionError`/`ProtocolError` and restore the entire snapshot. No counter, prefix, public history, card ownership, reward, or terminal flag may remain changed.

A successful non-commit publishes only the new private prefix. A successful commit publishes exactly one A02 transition, clears the prefix, advances `committed_step` exactly once, and returns the next protocol state. This trial boundary must not modify A02 implementation or semantics.

## Required test path and parent-owned retest

The runner path is exactly `tests/acceptance/test_a03_actions.py`; `scripts/run_acceptance.py --stage A03` must collect it. That suite must cover reachability/no-dead-prefix, exact independent set equality, canonical ordering, wild/natural equivalence, all phases/families, atomic rollback, counters/rewards, serialization, information boundary (no opponent hands in protocol observation), and both overflow points.

The parent owns the retest command and evidence review. Required evidence is the focused A03 test result plus the runner JSON/report; this document must not claim a pass or acceptance. A04 separately owns `guandan.environment`, observation tensors, batch APIs/equivalence, padding/dtypes/channels, and batch information-boundary acceptance.
