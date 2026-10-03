# A05 训练滚动与证据契约

A05 的 counted local_fast 运行必须使用真实 108 张牌、4 个独立 A04 环境、正常
A03 token steps，不能用 reduced fixture 代替目标计数。`trainer.py` 只负责 profile
解析、模块/优化器建立、checkpoint/resume 和证据；实际采样与更新由
`training.collector.RolloutCollector` 和 `training.trainer_core.TrainingEngine`
完成。

## 每次实际更新

- `token_steps_per_update=256` 是四个环境聚合总量，即每个环境 64 ticks；每次
  `batch.step()` 同时推进四个 slot，实际计数加 4；不能把一次 batch call 记为 1。
- 每一行记录 active player/team、token_step 前后、committed_step 前后、token、
  legal mask、old log probability、`rewards[:,4]`、acting-player reward、
  terminated、truncated 和 reset 后的下一观察。
- terminal 先捕获真实排名/reward/finish steps 再 reset；truncated 使用截断前
  observation 做 bootstrap，不支付 terminal reward；两个边界不得混淆。
- IPPO 使用 `[T,B,4]` 固定座位奖励视角的完整 GAE/V target；VRPO 使用独立
  688 维 privileged critic、masked Expected-SARSA Q、Q-boost advantage/target。
  actor 只接收 A04 policy observation，critic 的牌权 one-hot 不能进入 actor。
- 合法 token 概率和非法 token 零概率在每次更新前验证；梯度、loss、entropy、
  ratio、critic 输出必须 finite。

## local_fast 实际目标

每个算法（ippo、vrpo）各完成：

- 5 updates；每次 256 aggregate token steps；合计 1280 real transitions；
- 4 envs、每 env 每 update 64 ticks；
- update 5 保存 checkpoint；加载该文件后再实际完成 1 update；
- checkpoint 记录 actor/critic model config、profile SHA、rules/action/env/encoding/
  model/training versions、optimizer、环境 bytes、采样 RNG、计数器和更新摘要；
- A05 acceptance 报告的 `training_counts.json` 与每算法 execution JSON 是证据源，
  不以计划数替代实际数。

`remote_full` 本轮不执行。A05 本机通过只设置 `local_ready=true`，保持
`accepted=false`，不能被解释为远端完整稳定性或模型能力证明。
