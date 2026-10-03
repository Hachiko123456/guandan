# GuanDan First-Version Game Specification

- **Document ID:** GD-RULES-0.1
- **Status:** proposed specification for implementation
- **Scope:** one four-player, two-team GuanDan hand; a match wrapper is explicitly out of scope for the first engine milestone.
- **Implementation constraint:** this document is a standalone specification. It does not reuse or depend on any other GuanDan implementation.

## 1. Purpose and rule baseline

The project needs a deterministic, testable rule set rather than an ambiguous collection of regional conventions. This first version uses the common competitive GuanDan baseline associated with the *竞技掼蛋竞赛规则（试行）* published by the General Administration of Sport of China, Chess and Card Sports Administrative Center. Regional and event supplements differ; therefore every point below is an explicit project rule for GD-RULES-0.1.

The implementation must not silently add a local variant. Any behavior not specified here is an open decision and must block acceptance until resolved.

## 2. Players, seats, teams, and direction

- There are exactly four seats: 0, 1, 2, 3.
- Seats form a ring. The next seat is (seat + 1) mod 4; this is the engine's clockwise/action order.
- Teams are fixed: team 0 is seats 0 and 2; team 1 is seats 1 and 3.
- A player's partner is the seat two steps away. The other two seats are opponents.
- A hand has a designated leader_seat. In a standalone first hand the default leader is seat 0; tests may override it.
- Seat numbering and direction are engine conventions.

## 3. Cards, two decks, and physical identity

### 3.1 Physical deck

The game uses two standard 54-card decks, 108 physical cards total:

- Suits: clubs, diamonds, hearts, spades.
- Printed ranks: 3,4,5,6,7,8,9,10,J,Q,K,A,2.
- There are two small jokers and two big jokers in total.
- Every physical card has a stable card_id in 0..107. Duplicate printed cards remain distinct in serialization.
- A hand starts with 27 cards per player after a uniform deal of all 108 cards.

### 3.2 Current level

- level_rank is one of 3,4,5,6,7,8,9,10,J,Q,K,A; the special starting value is rank 2.
- Ordinary single-card order from low to high is: 2 < 3 < 4 < ... < A < level_rank < small joker < big joker.
- If level_rank == 2, the printed 2 is the level position above A; there is no second effective level rank.
- The level remains a printed rank and a special high ordinary rank. The engine stores printed_rank and effective_rank where needed.
- Suits do not break ordinary or same-size rank-bomb ties.

### 3.3 Wild card (逢人配)

- Every heart card whose printed rank equals level_rank is a wild card; there are two physical wild cards.
- A wild may represent any non-joker printed card, including chosen suit and rank, when a declared combination needs it.
- A wild may be used naturally as its printed heart-level card.
- A wild may not represent a small or big joker.
- A declaration records whether each wild is natural or its non-joker substitution.
- Physical identity never changes; substitution belongs to the committed action.

## 4. Approved combination families

Only these committed play families are legal. Other families, including four-with-two, airplanes with wings, arbitrary mixed groups, and unlisted local patterns, are illegal.

| Family ID | Name | Length/shape | Comparison key |
|---|---|---|---|
| SINGLE | 单牌 | exactly 1 card | effective rank |
| PAIR | 对子 | exactly 2 cards of one declared rank | declared rank |
| TRIPLE | 三同张 | exactly 3 cards of one declared rank | declared rank |
| FULL_HOUSE | 三带二 | one triple plus one pair, 5 cards | triple rank only |
| STRAIGHT | 顺子 | exactly 5 consecutive natural ranks | highest natural rank of window |
| PAIR_SEQUENCE | 三连对 | exactly 3 consecutive rank pairs (6 cards) | highest natural rank of window |
| TRIPLE_SEQUENCE | 二连三/钢板 | exactly 2 consecutive rank triples (6 cards) | highest natural rank of window |
| RANK_BOMB | 炸弹 | 4 to 10 cards of one declared rank | bomb length, then rank |
| STRAIGHT_FLUSH | 同花顺 | exactly 5 consecutive ranks of one suit | highest natural rank of window |
| FOUR_KINGS | 天王炸 | exactly two small + two big jokers | highest bomb class |

### 4.1 Natural rank windows

- A window is a sequence of distinct printed non-joker ranks.
- A straight, pair sequence, or triple sequence uses exactly its fixed v1 length; it is not an arbitrary-length run.
- Normal windows use consecutive natural ranks 2,3,4,5,6,7,8,9,10,J,Q,K,A.
- The low-A window A,2,3,4,5 is also allowed.
- J,Q,K,A,2 and other wrap-around windows are not allowed.
- A level rank does not create an extra rank; it participates through its printed rank.
- Wilds may fill missing ranks, but the final window has no duplicate rank.
- In a straight flush, a wild must be assigned to the same declared suit.

### 4.2 Family-specific legality

- PAIR, TRIPLE, and repeated groups inside other families compare after wild substitution. Two small jokers or two big jokers may form a pair; jokers cannot form a triple or rank sequence.
- FULL_HOUSE may use wilds in either group, but resolves to one triple and one pair with different ranks.
- PAIR_SEQUENCE is exactly three consecutive rank pairs; it contains six cards.
- TRIPLE_SEQUENCE is exactly two consecutive rank triples; it contains six cards.
- RANK_BOMB resolves four or more cards to one rank; maximum length is ten.
- STRAIGHT_FLUSH has exactly five cards and one declared suit; it is bomb-class.
- FOUR_KINGS / 天王炸 requires exactly two small jokers and two big jokers. It cannot use wilds or any non-joker card.

## 5. Comparison and following

### 5.1 Ordinary families

- An ordinary action beats only the same family with the same card count.
- FULL_HOUSE compares only its triple rank.
- STRAIGHT, PAIR_SEQUENCE, TRIPLE_SEQUENCE, and STRAIGHT_FLUSH compare the highest rank of their declared window using window order.
- Ordinary actions cannot beat bomb-class actions.

### 5.2 Bomb classes

From low to high: rank bomb length 4; rank bomb length 5; five-card straight flush; rank bombs length 6, 7, 8, 9, 10; four kings.

Same-length rank bombs compare by declared rank. Straight flushes compare by their highest declared rank against other straight flushes; suit does not break a tie. Four kings always wins.

### 5.3 Legal response set

When a current winning play exists, the active player may pass or commit a play that beats it. A bomb-class play may beat any ordinary family; a stronger bomb-class play may beat a weaker bomb-class play. Passing never changes the winning play.

## 6. Trick, pass, reset, and 接风

- A trick begins when a player leads any legal non-pass play.
- Following players act in seat order, skipping players who have emptied their hands.
- Pass is legal only when a current winning play exists; it is committed and consumes no cards.
- A trick resets when every other active player has passed since the last non-pass play. The last player who committed a non-pass play becomes the next leader.
- If that last play emptied a player and all other active players pass, the emptied player's partner receives the next lead. This is 接风. The emptied player leaves active rotation.
- If the partner is already empty, the next active seat after the emptied player becomes leader.
- A player's finishing rank is recorded immediately when their hand empties.

## 7. Tribute, return, and hand start

Tribute applies only when a previous hand result exists. A standalone first hand has no tribute.

### 7.1 Ranking and outcome class

- First player to empty receives rank 1; the second distinct finisher receives rank 2; then ranks 3 and 4.
- If ranks 1 and 2 are the same team, outcome_class is DOUBLE_DOWN. The remaining two seats are assigned ranks 3 and 4 by next-seat order after the rank-2 finisher, without further play. This serialization tie-break does not change the outcome class.
- Otherwise outcome_class is HEAD_THIRD when the winner's team has ranks 1 and 3, or HEAD_LAST when it has ranks 1 and 4.
- The team of rank 1 is the hand winner.

### 7.2 Tribute obligations

- For HEAD_THIRD or HEAD_LAST, rank 4 gives one tribute card to rank 1.
- For DOUBLE_DOWN, ranks 3 and 4 each give one tribute card. The higher tribute goes to rank 1 and the lower to rank 2.
- A tribute card is the donor's highest eligible non-wild physical card under single-card order. A heart-level wild is never eligible. Equal ranks use lowest card_id as deterministic tie-break.
- In DOUBLE_DOWN, equal tribute ranks pair to rank 1 for the donor closer to rank 1 in action order; the other pairs to rank 2.
- Anti-tribute: if any required donor holds both big jokers in the new hand, the entire tribute/return stage is canceled. This is the only anti-tribute condition in v1.
- If canceled, the previous rank-1 player leads the new hand.

### 7.3 Return obligations

- A recipient returns one card to each donor from whom they received tribute.
- A return card is a natural non-joker with printed rank 2..10; heart-level wild is excluded and no substitution is allowed.
- The recipient chooses the return card; it need not be the lowest.
- In a non-double-down outcome, rank 1 returns to rank 4. In DOUBLE_DOWN, rank 1 returns to the higher-tribute donor and rank 2 to the lower-tribute donor.
- After returns, the donor associated with the higher tribute leads. In the non-double-down case this is the sole donor. If canceled, rank 1 leads.
- Tribute/return card identities become public after the whole exchange resolves; unfinished choices are private.

## 8. Terminal reward for one hand

The environment uses shared zero-sum terminal reward:

- each player on the hand-winning team receives +1.0;
- each player on the other team receives -1.0;
- all non-terminal steps receive 0.0.

Terminal output includes ranking[4], winner_team, outcome_class, team_reward[2], finish_token_step, and finish_committed_step. No shaped reward is part of the first-version contract.

## 9. Public and private information

### 9.1 Public

Seat/team mapping; level; phase and active seat; all committed play/pass actions; current trick lead and winner; remaining-card counts; finished status/ranks; resolved tribute/return transfers; public history and indices.

### 9.2 Acting player's private information

The acting player's physical hand; unfinished prefix; their uncommitted tribute/return choice; explicitly protected debugging state. Opponent hands are never policy observations.

### 9.3 Training-only full information

A centralized critic, oracle test harness, or belief-label generator may receive the complete deal through a separate full-state channel. It must never be concatenated into the policy observation.

## 10. Single-hand boundary and open match decisions

The first engine milestone is a single-hand environment with caller-supplied level_rank and optional previous-hand outcome for tribute tests. A multi-hand match manager is not required for the first engine milestone.

Open for a later match specification: exact level advancement; whether level A has a special win condition; match length/time limit and ties; whether a completed match carries the leader; event-specific conversion to match points.

## 11. V1 decisions and non-blocking future scope

The following are fixed for the first single-hand training milestone:

1. V1 omits human-table reporting of a hand of ten or fewer cards.
2. V1 has no timeout, forfeit, or match-level termination.
3. V1 uses the explicit project rules in this document; regional/event supplements are not silently mixed in.
4. Diagnostic tie displays use card_id only for serialization; no game result depends on that ordering.

The following are explicitly deferred and do not block the single-hand milestone:

5. Multi-hand level advancement, match length, level-A match termination, and match-point conversion.
6. Platform-specific protocol fields and tournament-only supplements.

## 12. Reference note

This is the project's explicit GD-RULES-0.1 profile. It is not a claim that every regional or event supplement is identical. The explicit project rules above control implementation; deferred match-level variants do not block the single-hand milestone.

## 12. Source status and project policy

The following publicly accessible rule texts were used as research references, not as a claim that all regional/event variants are identical:

- Nanjing Sport University rule summary: https://www.nsi.edu.cn/jgzj/09/70/c1954a67952/page.htm?PageSpeed=noscript
- Changzhou Institute of Industry and Vocational Technology event rule summary: https://gh.ciit.edu.cn/2022/0923/c4494a105163/page.htm

The project must keep a source/clause table in the acceptance report. Where the two public summaries differ, GD-RULES-0.1 uses the following explicit v1 policy so the engine is deterministic:

1. The first milestone is a single hand. Match progression and the special multi-hand “过 A” victory condition are separate future work.
2. A straight is exactly five cards; a pair sequence is exactly three consecutive pairs; a triple sequence/steel plate is exactly two consecutive triples.
3. Natural windows are 2-3-4-5-6 through 9-10-J-Q-K, 10-J-Q-K-A, and A-2-3-4-5. J-Q-K-A-2 is invalid. The same window policy applies to pair sequences and triple sequences after expanding each rank group.
4. A pair may be two small jokers or two big jokers. A mixed small-plus-big pair is invalid. Four kings is exactly two small plus two big jokers.
5. A heart level card is wild only when it is used inside a multi-card declaration; as a standalone card it is its natural printed card. A wild cannot represent a joker. A declaration records the physical wild assignment.
6. Bomb order is four-rank bomb, five-rank bomb, five-card straight flush, six-rank bomb through ten-rank bomb, then four kings. Wilds may complete a rank bomb up to ten cards, subject to physical cards and non-joker substitution.
7. For tribute, the v1 anti-tribute condition is evaluated over the required losing donor side: one losing player in a non-double-down hand, or both losing players together in a double-down hand. The condition is possession of two big jokers in that donor side before tribute.
8. A return card is a chosen natural card with printed rank 2 through 10, excluding the current level card and A. The v1 engine does not use a fixed “smallest return” heuristic; return is a player action.
9. If tribute sizes tie, the lower seat number among the eligible donor/recipient pair is used as the deterministic tie-break for assignment and lead. This is an engine serialization policy, not a claim about all tables.
10. Reporting a hand of ten or fewer cards, clocks, forfeits, and platform protocol fields are out of scope for the single-hand engine and are not silently simulated.

The v1 profile is accepted only after the source/clause table and the rules tests agree with this section. Changing any item requires a new rules version and invalidates incompatible checkpoints.
