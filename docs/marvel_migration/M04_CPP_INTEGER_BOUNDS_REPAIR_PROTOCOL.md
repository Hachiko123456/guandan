# M04 C++ Action Row 整数边界补修协议

- 日期：2026-10-09
- 原始 M04 实现：`75735031721a44304ffe4d7be7fc1532fadd0ba6`
- 前次补修实现：`ff0d1d097d8989ff9037c303493002a2a4c1388a`
- 前次验收登记：`1703456d0de1b0643513a37fcc7fd8f45482a7ef`
- action/environment/encoding/serialization version：不变

## 1. 缺陷

原 `parse_action_row()` 把 NumPy integer row 强制转换为 `int64_t`，随后立即：

```cpp
result.tokens[index] = static_cast<int>(view(index));
```

词表范围检查发生在 C++ `int` 窄化之后。Windows/MinGW 的 32-bit `int` 使 `token + 2**32` 和 `token - 2**32` 截断回原合法 token，导致 raw rows 和 `actions_from_np(rows)` 两条入口都接受并执行非法输入。

## 2. 修复

`parse_action_row()` 先根据原 NumPy dtype kind 分支：

- signed integer：无损读取到 `std::int64_t[8]`；
- unsigned integer：无损读取到 `std::uint64_t[8]`。

对原始八个值全部执行：

```text
0 <= value < NUM_TOKENS
```

只有所有原始值通过后，才转换为 C++ `int`。unsigned 路径不会先转成 `int64_t`，因此 `INT64_MAX+1` 和 `UINT64_MAX` 不会在检查前溢出。词表大小使用协议常量 `NUM_TOKENS`，不写死数值。

范围检查之后继续执行既有完整语义校验：framing、phase、prefix stage、kind、payload、COMMIT flag、decoded metadata 和 live-candidate 唯一匹配。

## 3. 输入与失败语义

保留：

- `step(np.ndarray[B,8])`；
- `step(actions_from_np(rows))`。

越界 row 必须在解码阶段抛 `ProtocolError`。不允许裁剪、取模、饱和转换或 fallback。Batch 继续使用 trial-copy；任一 slot 越界时所有 slot 的 state、prefix、RNG 和 counter 保持原序列化 bytes。

## 4. 兼容性

合法 `int32`、`int64`、`uint32`、`uint64` row 的解码和 transition 保持一致。实现继续支持由 NumPy 正确转换的非连续和非本机字节序 integer arrays。

本补修不改变 action row wire、候选排序、合法动作集合、RNG、序列化 schema、Sampler、Python oracle 或训练目标。

## 5. 验证范围

- 八个 row 字段 × signed `+2**32`、signed `-2**32`、unsigned `+2**32`；
- `-1`、`NUM_TOKENS`、int32 min/max、int64 min/max、`INT64_MAX+1`、`UINT64_MAX`；
- raw/decoded 两条入口；
- 两 slot atomic failure；
- Python/C++ 同一越界输入拒绝对照；
- 四种合法 integer dtype；
- 非连续和非本机字节序合法 rows；
- 前次 sampler、RNG、trajectory counter 和 serialization 补修回归；
- 强制 rebuild 和隔离 wheel。

这是范围严格限定的 M04 补修，不启动或替代 M05。
