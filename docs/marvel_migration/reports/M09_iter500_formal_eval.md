# M09 iter500 Formal Selection / Champion 结果

- 日期：2026-10-14
- evaluation id：`08dab22e0d4b4474a58be715ea5d8ab9`
- 状态：`accepted_complete_no_promotion`
- M09 accepted：**否**
- final holdout：**未使用**

## 远程运行与完整性

- Kaggle T4×2 runner exit：0；
- 评估：640/640 complete；
- 32 clusters × 4 rotations × 5-lane fixed matrix；
- 每个 lane：128/128；
- illegal：0；
- truncated：0；
- ZIP CRC、路径安全、artifact hashes、JSON self-hashes：通过；
- ZIP SHA-256：`204f6fa0b7b118b771d37805ea42867e6127c6aa98a584fe8e964acaaf3a0fe3`；
- 日志 SHA-256：`bff973724b71c76c7a293be0188b5ec032d8406f5796b2d3f32fd08ae0605615`；
- 评估耗时：2695.61 秒，约 44.9 分钟；
- heartbeat：135 条，最大 stalled_seconds=0.0。

## Candidate 与 Champion

- candidate：IPPO iter500，SHA-256 `f4d1d5872fa80ad1be2804ee91620046256dc0d481b78a25e4b0311f80355cea`；
- current champion：IPPO iter100，SHA-256 `b00a2c178e8b3e99d80418ae283a785285639b8172a0d96232beef7233b9d41c`；
- source/backend identity：一致；
- candidate/champion training provenance：complete。

## 结果

| 对手 | 胜局 | 对局 | 胜率 | clustered CI95 low | high |
|---|---:|---:|---:|---:|---:|
| Champion iter100 | 74 | 128 | 57.81% | 1.13% | 100.00% |
| Rule | 72 | 128 | 56.25% | 0.00% | 100.00% |
| Random | 107 | 128 | 83.59% | 36.35% | 100.00% |
| Legacy V1 | 107 | 128 | 83.59% | 35.78% | 100.00% |
| Opponent pool | 74 | 128 | 57.81% | 1.13% | 100.00% |
| Overall | 434 | 640 | 67.81% | 不跨相关对手汇总 | 不跨相关对手汇总 |

Rule 点估计达到 40% gate，Random 点估计达到 70% gate，且 health 全部为 0。但 candidate-vs-champion clustered CI95 lower bound 仅 0.0113，未达到 formal promotion threshold 0.52。

## Promotion 决定

- status：`inconclusive`；
- promote：`false`；
- reason：`candidate-vs-champion CI lower bound does not reach the promotion threshold`；
- decision hash：`ad5a759ef2e7d52887328418abe5fe0bd4ee92eabd9a7f2e463502825d7faa1f`；
- evaluation hash：`9a0606ff7aabb6476fbbc411ac68477cb91eb3a38e46c221f9dbc87dc3be5fcc`。

该结果表示 iter500 相对 iter100 的点估计更高，但当前 32-cluster 证据不足以通过预先冻结的保守 CI gate。不得将其解释为 iter500 已显著优于 champion。没有 promotion transaction；current champion 保持 generation 1 / iter100。

## 下一步

iter500 formal review 已完成，segment1000 现可按原分段计划授权运行：从已验证 iter500 recovery exact resume 到 iteration1000/4,000 completed games。训练继续使用 iter500 training state，而评估 champion 仍保持 iter100。segment1000 完成后必须再次独立运行 formal selection/champion review。不得运行 final holdout，也不得声明 M09 accepted 或模型达到强度门槛。
