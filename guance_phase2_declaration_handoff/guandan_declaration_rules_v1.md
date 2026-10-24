# Guandan complex declaration contract v1

> Guance Phase 2 / GMV2-S2-DECL-003 权威输入候选。所有规则证据来自当前 Python reference 与源码匹配的 native C++ MARVEL backend；未访问 Guance。

## 身份与 durability

- Ruleset: `GD-RULES-0.1`
- Protocol: `GD-ACTION-0.2-MARVEL|GD-ENV-0.2-MARVEL|GD-ENCODING-0.2-MARVEL`
- Authority source commit: `8ea264c1bde4b8faac24560afa27d37955232028`
- Handoff artifact commit: `null`
- Handoff artifact git status: `untracked`
- Exporter SHA256: `61cce080052ec4208fd86fac78e847b9a93fab88f9a7ae09a18d2d7ef05823d6`
- Generated at: `2026-10-12T11:41:42+08:00`
- CWD: `D:\project\guandan`
- Rules JSON SHA256: `29772a3ab8a61d9f603e8d57f2432abdec6f5e591927b94a7beb87fc74a77287`
- Rules semantic SHA256: `96b1fdccf795af4e51361e8bd155d5e46c663d50ea1d1dda22a3a6021ce663dc`
- Golden vectors JSON SHA256: `e183a6e4a9cbf01dffe424d4054ebab433f6107704680eab6e783607c812c7ae`
- Vectors semantic SHA256: `8573a4def03f9bd72d0bfddbca136ec3bec5ff0255fb7b2b578cffb012f6c739`
- Native extension SHA256: `55803483a3271ddfc53a04930ad7b40c13a8ce7da6d02981a426f4b08602a9dd`
- Native source matches loaded binary: `true`

`authority_source_commit` 只绑定权威规则源码；当前 handoff/exporter 仍是 untracked，`handoff_artifact_commit=null`，未把源码 commit 冒充 artifact commit。

## Authority source hash scope

每个 authority source record 同时固定三种身份：

1. `working_tree`: 当前 checkout 原始字节、换行类型、字节数和 SHA256，仅表示本机执行环境；
2. `normalized_lf`: 将 CRLF 和 bare CR 转换为 LF 后的字节数和 SHA256；
3. `git_blob`: `authority_source_commit:path` 的 blob OID、blob bytes 和内容 SHA256，作为跨平台 authority identity。

Exporter 要求每个文件 `normalized_lf == git_blob bytes` 且 authority paths 相对 HEAD clean，否则 fail closed / exit 1。

## 权威路径

- Recognizer: `guandan/combos.py::recognize_combinations`, `nanopokerlib/cpp/guandan/guandan.cpp::recognize_combinations`
- Legal generator: `guandan/round.py::enumerate_plays`, `guandan/round.py::enumerate_legal_actions`, `nanopokerlib/cpp/guandan/guandan.cpp::enumerate_plays`
- Serializer: `guandan/action_state.py::_declaration_fields`, `guandan/action_state.py::_ordered_action_cards`, `guandan/action_state.py::canonical_tokens`, `nanopokerlib/cpp/guandan/guandan.cpp::declaration_ranks`, `nanopokerlib/cpp/guandan/guandan.cpp::declaration_length`, `nanopokerlib/cpp/guandan/guandan.cpp::declaration_suit`, `nanopokerlib/cpp/guandan/guandan.cpp::ordered_action_cards`, `nanopokerlib/cpp/guandan/guandan.cpp::canonical_tokens`
- Validator: `guandan/round.py::_valid`, `guandan/round.py::_play`, `guandan/action_state.py::StepwiseActionState.step`, `guandan/environment.py::GuandanEnv.step`, `nanopokerlib/cpp/guandan/guandan.cpp::Game::step`
- MARVEL encoder/decoder: `guandan_marvel/game.py::_action_row`, `guandan_marvel/game.py::_parse_action_row`, `nanopokerlib/cpp/guandan/guandan.cpp::action_row`, `nanopokerlib/cpp/bindings.cpp::actions_from_np`

## 13-level parity matrix

- Levels: `THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE, TWO`
- Invalid levels rejected: `SMALL_JOKER, BIG_JOKER`
- Matrix cases: `338` (`26` per level)
- 每个 level：6 个 family natural baseline；5 个 wildcard family × natural/substituted-other-rank/substituted-level-rank/max-two；FOUR_KINGS wildcard rejection 与无 rank/suit/length/wild token。

| Level | cases | natural | natural wild | substituted-other | substituted-level | max-two wild | result |
|---|---:|---:|---:|---:|---:|---:|---|
| THREE | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| FOUR | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| FIVE | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| SIX | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| SEVEN | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| EIGHT | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| NINE | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| TEN | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| JACK | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| QUEEN | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| KING | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| ACE | 26 | 6 | 5 | 5 | 5 | 5 | PASS |
| TWO | 26 | 6 | 5 | 5 | 5 | 5 | PASS |

聚合 parity 值全部由逐 case 实际执行结果计算，不从已有 JSON 读取，也不直接硬编码。`--check` 会重新运行完整 matrix。

## THREE / ACE / TWO 固定边界 vectors

- THREE：level-heart natural/substituted、缺 THREE 的 sequence、FULL_HOUSE 中 THREE 作为 triple 与 pair。
- ACE：heart ACE natural/substituted、A2345、10JQKA、非法 JQKA2、ACE 作为 full-house triple/pair、hearts straight flush。
- TWO：heart TWO natural/substituted、A2345、pair-sequence A23、triple-sequence A2、非法 KA2、TWO 作为 full-house triple/pair、hearts straight flush。
- Detailed special vectors: `43`。

## 关键序列语义

序列基础顺序是 `2,3,4,5,6,7,8,9,10,J,Q,K,A`。只额外允许 A-low 窗口：`A2345`、`A23`、`A2`。不允许 `JQKA2`、`KA2` 或其他绕回；大小王不参加序列；比较使用窗口末位在 WINDOW_RANKS 中的索引。

## Family 摘要

| Family | physical length | multiplicity | declared_ranks | comparison_rank | LENGTH | suit | wild |
|---|---:|---|---|---|---|---|---|
| FULL_HOUSE | 5 | triple×3; pair×2 | triple_rank,pair_rank | triple_rank | none | physical suits are ignored; declared_suit must be null | yes / max 2 |
| STRAIGHT | 5 | each rank in window×1 | window_first_rank | window_last_rank | none | unrestricted; same-suit cards also admit a distinct STRAIGHT_FLUSH declaration | yes / max 2 |
| STRAIGHT_FLUSH | 5 | each rank in window×1 | window_first_rank | window_last_rank | none | all non-wild physical cards and represented wildcard targets share exactly one declared standard suit | yes / max 2 |
| PAIR_SEQUENCE | 6 | each rank in 3-rank window×2 | window_first_rank | window_last_rank | 3 | unrestricted; declared_suit must be null | yes / max 2 |
| TRIPLE_SEQUENCE | 6 | each rank in 2-rank window×3 | window_first_rank | window_last_rank | 2 | unrestricted; declared_suit must be null | yes / max 2 |
| FOUR_KINGS | 4 | SMALL_JOKER×2; BIG_JOKER×2 | [] | BIG_JOKER | none | physical joker suit marker only; declared_suit must be null | no / max 0 |

## Encoding 要点

- FULL_HOUSE: `declared_ranks=[triple_rank,pair_rank]`; card token 先 triple 后 pair。
- STRAIGHT/STRAIGHT_FLUSH: `declared_ranks=[window_first_rank]`; comparison rank 是 `window_last_rank`。
- STRAIGHT_FLUSH 字段顺序为 `FAMILY,SUIT,RANK`。
- PAIR_SEQUENCE: physical length 6，`LENGTH=3`；TRIPLE_SEQUENCE: physical length 6，`LENGTH=2`。
- FOUR_KINGS: `declared_ranks=[]`, `comparison_rank=BIG_JOKER`, `comparison_key=[9]`，无 SUIT/LENGTH/WILD；action order `[52,53,106,107]`，storage order `[52,106,53,107]`。
- 无 suit family 的 substituted wildcard 使用 `CLUBS` 作为权威占位 suit；`target_rank=level_rank` 时 natural HEARTS 与 substituted CLUBS 是两个可区分的合法 declaration。
- STRAIGHT_FLUSH 的 target suit 必须等于 declared suit；level rank 在 HEARTS 中 canonicalize 为 natural，在 CLUBS/DIAMONDS/SPADES 中为 substituted。

## Golden vectors

- Positive: `73`（base `32`）
- Negative: `50`（base `48`）
- Special boundary: `43`
- Total: `123`

## 生成和验证

```powershell
cd D:\project\guandan
py -3.12 -B tools/export_guance_declaration_contract.py --output-dir guance_phase2_declaration_handoff --generated-at 2026-10-12T11:41:42+08:00
py -3.12 -B tools/export_guance_declaration_contract.py --output-dir guance_phase2_declaration_handoff --check
py -3.12 -B -m pytest -q -p no:cacheprovider tests/unit/test_combos.py tests/unit/test_action_protocol.py tests/acceptance/test_a02_rules.py tests/marvel_cpp/test_round_protocol.py tests/acceptance/test_guance_declaration_handoff.py
git diff --check
```

## 停止点 / 未验证

- Handoff final acceptance 候选可由本仓独立验收；不等于 Guance DECL-003 或 Guance Phase 2 已通过。
- 未访问或修改 `D:\project\guance`；未进入 Phase 3。
- 未读取 checkpoint、真实 capture、完整真实牌局或隐藏牌；未执行训练或模型推理。
- 未 commit，未 push；artifact durability 尚未由 Git commit 提供。

