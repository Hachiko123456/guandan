# GuanDan Scheme B Action Protocol

- **Document ID:** GD-ACTION-0.1
- **Compatible rules:** GD-RULES-0.1
- **Compatible environment:** GD-ENV-0.1
- **Scheme:** stepwise token construction with an explicit COMMIT token.

## 1. Goal

The policy does not choose a complete combination from a flat action list. It chooses one legal token at a time. The environment maintains a private unfinished construction for the active player. Only COMMIT turns that construction into an atomic game action.

The protocol applies to four decision phases:

- LEAD_PLAY: choose any legal non-pass play;
- FOLLOW_PLAY: choose a legal beating play or PASS;
- TRIBUTE: choose exactly one eligible tribute card;
- RETURN: choose exactly one eligible return card.

## 2. Action lifecycle

~~~text
phase starts
  -> PREFIX token 0
  -> PREFIX token 1 ...
  -> optional PREFIX tokens
  -> COMMIT
  -> atomic rule transition
  -> next phase/active player
~~~

A prefix is valid only if at least one complete legal committed action can be completed from it. The environment must not expose a dead prefix.

The incomplete prefix is private to the acting player and is not appended to public history. A rejected token leaves the state unchanged.

## 3. Token kinds

The first vocabulary must support these semantic kinds. Numeric IDs are implementation-defined but versioned.

### 3.1 Control tokens

- PASS: valid only in FOLLOW_PLAY; immediately committed and has no additional prefix.
- COMMIT: valid only when the current prefix describes exactly one complete legal action.
- CANCEL: not a game action and not exposed in v1. To abandon construction, restore a clone/pre-prefix state.

### 3.2 Play-construction tokens

Semantic kinds include:

- FAMILY: SINGLE, PAIR, TRIPLE, FULL_HOUSE, STRAIGHT, PAIR_SEQUENCE, TRIPLE_SEQUENCE, RANK_BOMB, STRAIGHT_FLUSH, FOUR_KINGS;
- RANK: printed or declared rank;
- LENGTH: group/sequence length where required;
- SUIT: required for STRAIGHT_FLUSH;
- CARD: physical card_id from the acting player's hand;
- WILD_ASSIGNMENT: declared non-joker target for one selected wild;
- GROUP_BOUNDARY: separates triple/pair or repeated groups where canonical grammar requires it.

### 3.3 Tribute/return tokens

- CARD selects one physical card.
- COMMIT finalizes it.
- No wildcard assignment is legal in TRIBUTE or RETURN.

## 4. Canonical grammar

The implementation must choose one canonical grammar and expose only prefixes of that grammar. The following grammar is normative for v1; an equivalent deterministic encoding is allowed only if it produces the same committed-action set and is recorded as a protocol revision.

### 4.1 LEAD_PLAY and FOLLOW_PLAY

FOLLOW_PLAY additionally exposes PASS.

For a non-pass play:

~~~text
FAMILY
  -> family-specific declaration tokens
  -> CARD selections and WILD_ASSIGNMENT tokens as required
  -> COMMIT
~~~

The environment must not accept card orders that create duplicate canonical actions. Canonicalization:

- physical cards are sorted by card_id inside each semantic group;
- groups are ordered by increasing declared rank, except FULL_HOUSE emits its primary triple group first;
- wild assignments are sorted by physical card_id;
- no unused token appears before COMMIT.

Permitted family-specific shapes:

- SINGLE: FAMILY, CARD, optional WILD_ASSIGNMENT, COMMIT;
- PAIR/TRIPLE: FAMILY, RANK, CARD..., optional WILD_ASSIGNMENT..., COMMIT;
- FULL_HOUSE: FAMILY, TRIPLE_RANK, PAIR_RANK, CARD..., optional WILD_ASSIGNMENT..., COMMIT;
- STRAIGHT: FAMILY, START_RANK, CARD..., optional WILD_ASSIGNMENT..., COMMIT;
- PAIR_SEQUENCE: FAMILY, START_RANK, LENGTH, CARD..., optional WILD_ASSIGNMENT..., COMMIT;
- TRIPLE_SEQUENCE: FAMILY, START_RANK, LENGTH, CARD..., optional WILD_ASSIGNMENT..., COMMIT;
- RANK_BOMB: FAMILY, RANK, LENGTH, CARD..., optional WILD_ASSIGNMENT..., COMMIT;
- STRAIGHT_FLUSH: FAMILY, SUIT, START_RANK, CARD..., optional WILD_ASSIGNMENT..., COMMIT;
- FOUR_KINGS: FAMILY, CARD, CARD, CARD, CARD, COMMIT.

The number of CARD and WILD_ASSIGNMENT tokens is derived from family, declaration, and length. COMMIT must not be legal until the exact required structure is complete.

### 4.2 TRIBUTE

~~~text
CARD
COMMIT
~~~

Only the eligible physical card set from GD-RULES-0.1 may be exposed.

### 4.3 RETURN

~~~text
CARD
COMMIT
~~~

Only a natural non-joker printed rank 2..10, excluding the heart-level wild physical card, may be exposed.

## 5. Prefix state

The private prefix record contains at least:

~~~python
@dataclass
class ActionPrefix:
    phase: str
    family: str | None
    declared_ranks: tuple[int, ...]
    declared_length: int | None
    declared_suit: int | None
    selected_card_ids: tuple[int, ...]
    wild_assignments: tuple[tuple[int, int, int | None], ...]
    next_expected_kind: str
    complete: bool
~~~

The physical hand is not modified while complete is false. The prefix is discarded after successful COMMIT or reset/clone restore.

## 6. COMMIT semantics

COMMIT is the only token that authorizes a rules transition.

On successful COMMIT:

1. validate the complete prefix against the rules engine;
2. construct one canonical CommittedAction;
3. remove/transfer selected physical cards for play, tribute, or return;
4. update public history and current phase;
5. calculate terminal result/reward if the hand ends;
6. clear the prefix;
7. expose the next active player's observation.

If validation fails, raise IllegalActionError and leave all fields unchanged, including counters.

CommittedAction must include:

~~~python
@dataclass(frozen=True)
class CommittedAction:
    actor: int
    phase: str
    kind: str                 # PLAY, PASS, TRIBUTE, RETURN
    family: str | None
    card_ids: tuple[int, ...]
    declared_ranks: tuple[int, ...]
    declared_suit: int | None
    wild_assignments: tuple[tuple[int, int, int | None], ...]
    public_index: int
~~~

## 7. Token-step versus committed-game-step accounting

Both counters are mandatory in observations, step results, rollout records, and diagnostics.

### 7.1 Token step

- token_step starts at zero at reset.
- Every accepted token, including prefix tokens and COMMIT, increments it by one.
- A rejected token does not increment it.
- It measures neural decision granularity and is the time axis for the first MARVEL adapter.

### 7.2 Committed game step

- committed_step starts at zero at reset.
- It increments once for each successful COMMIT that produces one atomic rule action.
- It does not increment for non-commit prefix tokens.
- PASS, PLAY, TRIBUTE, and RETURN each count as one committed game step.
- A one-token PASS still counts as one committed game step.

### 7.3 Required flags and rewards

Every step result contains:

~~~text
is_commit: bool
committed_action: CommittedAction | None
reward: float[4]
done: bool
truncated: bool
token_step: int
committed_step: int
~~~

Non-commit steps have is_commit false, no committed action, and zero reward. Successful commits have is_commit true; only a terminal commit may have non-zero reward.

## 8. Legal-next-token set

Every non-terminal observation returns a padded list of legal next-token rows. Each row contains token_id, semantic kind, semantic value, is_commit, and valid mask.

The set must be complete for the current prefix and phase. It must not be made by selecting only the top-N complete actions. Capacity overflow is an environment configuration error and must raise a diagnostic rather than silently truncate.

## 9. Action equality and de-duplication

Two token sequences are the same committed action if they select the same physical cards, family, declaration, and wild assignments. The environment exposes one canonical sequence only.

- Reordering cards in a pair is not a second action.
- Swapping full-house groups is not a second action.
- Assigning a wild to a different target is a different action when the final declaration differs.
- A natural heart-level card and its wild assignment are different declarations only when the committed action differs; identical canonical actions expose only the canonical natural form.

## 10. Failure and safety behavior

- A token from another phase is illegal.
- A token not in legal-next-token is illegal.
- An incomplete COMMIT is illegal.
- Duplicate physical card selection is illegal.
- A wild assignment to a joker is illegal.
- A play that does not beat the current trick winner is illegal, except PASS.
- PASS while leading or with no current winner is illegal.
- Exceeding max_action_tokens is a protocol error, not truncation.
- Reaching max_token_steps makes the episode truncated with a diagnostic reason and no win/loss reward.

## 11. Required protocol tests before acceptance

The implementation must show:

1. every legal complete action has an exposed prefix path;
2. every exposed COMMIT produces exactly one legal committed action;
3. no dead prefix is exposed;
4. no illegal card or wild assignment is exposed;
5. canonicalization removes permutation duplicates;
6. an independent complete-action reference enumerator and the stepwise protocol produce identical committed-action sets on reduced decks and sampled full-deck states;
7. non-commit steps do not mutate public game state or committed-step count;
8. commit steps mutate exactly once and increment committed-step count exactly once;
9. serialization preserves prefix and next legal-token set;
10. token-level and terminal rewards match GD-ENV-0.1.

## 12. V1 fixed decisions and deferred scope

1. V1 uses the global token vocabulary and numeric ranges in `GD-ENV-0.1`.
2. `max_action_tokens=32` and `max_legal_next_tokens=256`; overflow is a protocol error, never silent truncation.
3. One policy forward pass chooses one scalar next token.
4. The first training adapter uses token-level PPO/GAE. Prefix extension rows have zero reward; only a terminal commit may have a non-zero hand reward. Committed-action log-probability aggregation is deferred and must not be mixed into v1 checkpoints.
5. A future protocol version may add a different objective or vocabulary, but it must reject incompatible v1 checkpoints.
