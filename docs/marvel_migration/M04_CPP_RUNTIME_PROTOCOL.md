# M04 掼蛋 C++/pybind 运行时协议

- 实现日期：2026-10-09
- backend：`cpp`
- game identifier：`GuandanMARVELV2Cpp`
- action：`GD-ACTION-0.2-MARVEL`
- environment：`GD-ENV-0.2-MARVEL`
- encoding：`GD-ENCODING-0.2-MARVEL`
- serialization：`guandan-marvel-cpp-runtime-v1`

## 1. 边界

M04 在 `nanopokerlib/cpp/guandan/` 中重新实现掼蛋规则和逐 token 运行时。`guandan/cards.py`、`combos.py`、`round.py`、`state.py` 继续作为只读语义 oracle；本阶段没有修改这些文件，也没有复制/改名斗地主规则实现。

`guandan_marvel.game` 仍是默认 `python_reference`。C++ 后端只能通过 `nanopokerlib.guandan` 或显式的 `guandan_marvel.cpp_game` 导入；扩展不存在时导入直接失败，不进行 silent fallback。正式训练默认后端只有在 M05 完整差分和性能门槛通过后才允许切换。

## 2. 来源和许可证

- 通用类型/批量环境形状派生自 MIT 许可的 `D:\project\doudizhu\nanopokerlib\cpp\core\types\*` 和 `env_batch.hpp`；参考 commit `acdcf199cf422e2b683cb22142b8e7cb33b14460`。
- 目标包保留 `nanopokerlib/LICENSE`、`NOTICE.md` 和头文件内 SPDX/来源注释。
- Guandan 牌、组合、轮次、贡还牌、协议和 observation 代码为本项目新写实现；未复制 `cpp/games/doudizhu/*`。

## 3. 规则实现

C++ core 覆盖：

- 两副牌、108 个稳定物理 card ID；
- 13 个非 joker 级牌上下文和有效牌力；
- 两张红桃级牌的自然/替代声明；
- single、pair、triple、full house、straight、pair sequence、triple sequence、rank bomb、straight flush、four kings；
- 普通牌型可比边界和 `4炸 < 5炸 < 同花顺 < 6..10炸 < 四王`；
- 发牌、轮转、PASS、trick reset、接风、完牌名次、`DOUBLE_DOWN`/`HEAD_THIRD`/`HEAD_LAST` 和团队奖励；
- 单贡/双贡、最高合法贡牌、还牌限制、抗贡、未完成交换隐私和完成后公开。

非法动作、伪造 declaration、stale action row、容量溢出、horizon overflow 和 108 张牌守恒错误均抛显式异常。批量 `reset`/`step` 在输出构造成功前使用 trial copy，失败不提交部分状态。

## 4. A03/V2 action

V1 canonical token 语义与 Python oracle 相同。V2 固定 8-token row：

```text
[ROW_BEGIN, PHASE, PREFIX_STAGE, KIND,
 PAYLOAD_MARKER, V1_TOKEN_ID, IS_COMMIT, ROW_END]
```

`actions_from_np()` 验证 dtype、shape、framing、phase、stage、kind/payload 和 COMMIT flag。`EnvBatch.step()` 再要求完整 row 与当前 live candidate 唯一匹配；只提交相同 payload 或旧 prefix stage 不可执行。不同语义动作不得共享 canonical sequence，候选容量不得静默截断。

## 5. Observation 和隐私

输出保持 M02 schema：

```text
rewards                       float32 [B,4]
is_terminal                   bool    [B]
player_ids                    int64   [B]
observations.tokens           int64   [B,4,48]
observations.channels         uint8   [B,4,512]
available_actions.tokens      int64   [B,256,8]
available_actions.weights     float32 [B,256]
available_actions.is_commit   bool    [B,256]
```

Actor channel 只包含 observer 手牌、公开 current winning、自己的 prefix/未公开交换选择和公共状态。RETURN 阶段已进入 donor 手牌但尚未公开的 incoming return card 对 donor 隐藏。未完成贡还牌 commit event 只发给 actor；全部完成后一次公开完整 transfer records。

完整 ownership/escrow/played 仅由独立 `critic_state()` 返回，不附着到 `EnvStep`，并逐 card 检查守恒。

## 6. Batch、序列化和调试

`EnvBatch` 支持：

- `reset(seed)`；
- `reset_from_deals(...)`，供固定发牌测试和 M05 差分；
- `observe()`、`step(rows)`；
- `critic_state()`；
- `debug_state(slot)`；
- `serialize()` / `EnvBatch.deserialize()`；
- `auto_reset=true|false`。

序列化保存版本、容量、每 slot config、完整 round/protocol state、prefix、token/commit counters、事件、trajectory counter 和 RNG 状态。反序列化重新执行协议、牌守恒和 live-frame 验证；序列化 roundtrip 必须 byte-stable，后续 transition 必须一致。

## 7. Build

`pyproject.toml` 的 build isolation 安装 `pybind11>=2.11,<3`。`setup.py` 直接调用 C++20 编译器：

- Windows：MinGW `g++` + `gendef` + `dlltool`，本地生成 Python import library，并静态链接 GCC/libstdc++/winpthread runtime；
- Linux：host `c++/g++ -std=c++20 -shared -fPIC`。

源码 wheel 包含平台扩展；`MANIFEST.in` 保留 C++ source/header 供 source build。

## 8. M04 验证边界

本阶段已执行全部十类 directed action、贡还牌/抗贡/接风/终局、隐私、序列化、batch 独立性和真实批量终局专项测试；另执行 3 个小样本完整 Python/C++ 同路径 smoke。初步本机 batch smoke 仅作为功能/吞吐观测。

这些结果不是 M05 验收。尚未执行 10,000 完整对局、100,000 中间状态、全规模统计差分或固定工作负载 >=2x 门槛，不得据此切换正式训练默认后端。
