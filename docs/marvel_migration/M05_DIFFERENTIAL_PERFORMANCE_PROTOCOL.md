# M05 Python/C++ 差分与性能协议

- Stage: `M05_differential_performance`
- Correctness evidence schema: `guandan-marvel-m05-differential-summary-v1`
- Corpus schema: `guandan-marvel-m05-state-action-corpus-v1`
- Mismatch fixture schema: `guandan-marvel-m05-mismatch-fixture-v1`
- Benchmark schema: `guandan-marvel-m05-benchmark-v1`
- Protocols under test: `GD-ACTION-0.2-MARVEL`, `GD-ENV-0.2-MARVEL`, `GD-ENCODING-0.2-MARVEL`

## 1. Oracle and cutover boundary

`guandan_marvel.game` remains the Python reference oracle. `nanopokerlib.guandan` / `guandan_marvel.cpp_game` is compared against it and never rewrites oracle output or A01/A02 semantics.

After all M05 gates pass, package-level `guandan_marvel.game` selects the C++ training backend. The oracle remains explicitly available as `guandan_marvel.python_game` and `get_game_backend("python_reference")`. `get_game_backend("cpp")` does not silently fall back when the extension is unavailable.

## 2. Matched state/action corpus

Each corpus game stores:

- deterministic corpus `game_index` and `seed`;
- one of all 13 printed level ranks;
- leader seat;
- complete previous-result object or `null`;
- all four physical-card deals;
- exact step records encoded as `[action_token_id, *prefix_tokens]`;
- a restore checkpoint index;
- terminal result and complete trajectory digest.

The complete V2 row is uniquely regenerated from the live legal candidate set. Replay rejects a missing/duplicate action token or a prefix mismatch. The accepted 10,000-game corpus is gzip-sharded under `runs/M05_differential_performance/20261009_acceptance/corpus`; Git tracks only its manifest, sizes and SHA-256 identities.

## 3. Per-state comparison

Every initial, intermediate and terminal state compares exact shape, dtype and content for:

- legal candidate rows;
- legal weights and `is_commit`;
- active player id;
- four-seat incremental observation tokens;
- four-seat actor-information channels;
- privileged ownership, escrow and played-card tensors;
- rewards and terminal flag.

The serialized summary digest includes all fields above. The action row is appended to independent Python and C++ trajectory hashers. Terminal ranking, winner team, outcome class, team reward and finish counters are compared separately.

Every complete game performs one backend-local byte-stable serialize/deserialize roundtrip and then continues the same differential trajectory. A deterministic random subset of exactly 100,000 nonterminal states is identified by digest from the states already compared.

## 4. Coverage requirements

Acceptance requires all of the following with zero mismatch:

- 10,000 complete games and 100,000 sampled nonterminal states;
- all 13 level ranks;
- all ten `CombinationKind` families;
- natural and substituted wild declarations;
- rank bombs, straight flushes and four kings;
- tribute, return, anti-tribute and contact;
- all three terminal outcome classes;
- four observer-specific hidden-hand reorder checks;
- directed complex-action cases from the accepted M02 corpus;
- 10,000 serialization/resume continuations;
- matching complete trajectory digests.

Any mismatch writes a minimal fixture containing the initial deal/configuration, exact executed prefix/action records, restore point, mismatch field and compact value/hash details. Mismatch fixtures are never filtered from the result.

## 5. Performance workloads

All timed lanes run on one machine and use matched deals, action rows or an identical frozen policy. The report records wall time, process CPU time, peak RSS and segmented time.

The hard `>=2x` backend gate applies to all three backend-sensitive matched workloads:

1. Python single-env versus C++ single-env over the same complete games and rows;
2. Python batch versus C++ batch over the same batch deals, rows and global transition count;
3. end-to-end observation conversion + identical frozen policy + environment rollout over the same complete games.

The target is `5x`, but it is not required for acceptance.

A complete unchanged IPPO iteration is also run with the same deal, model state, seed, trajectory length and learner configuration. It is segmented into rollout and learner time and is diagnostic rather than substituted for the backend gate: identical neural inference and learner work is backend-independent and can dominate wall time. Its lower aggregate ratio must be reported, not hidden or described as an environment speedup.

No GPU utilization workload is synthesized. When the local PyTorch build cannot use CUDA, the physical GPU is inventoried and GPU use is reported as unavailable.

## 6. Acceptance and claims

M05 is accepted only when correctness is valid with zero mismatch and the minimum speedup across the three gated workloads is at least `2.0`. Acceptance authorizes the C++ default training backend; it does not claim model strength, Kaggle verification, dual-T4 utilization or practical playing strength.
