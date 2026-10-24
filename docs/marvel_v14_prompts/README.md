# MARVEL v1.4 分阶段实施 Prompt 索引

本目录存放可直接复制给 DeepSeek 的实施 Prompt。一次只执行一个阶段；DeepSeek 完成后，把其完整回复交给 Codex 独立验收，验收通过后再执行下一阶段。


## 当前执行入口

- 阶段 1 修复已完成并保留 [`01A_phase1_repair.md`](./01A_phase1_repair.md) 作为审计记录；不要重复执行。
- 下一步只执行修订后的 [`02_macro_sequence_ppo_and_gae.md`](./02_macro_sequence_ppo_and_gae.md)。该 Prompt 明确区分 legacy、只替换 sequence ratio 的消融模式，以及完整宏决策模式；Reviewer PASS 前不得进入阶段 3。

## 核心阶段

1. [`01_baseline_and_macro_infrastructure.md`](./01_baseline_and_macro_infrastructure.md)  
   冻结基线、宏动作分段、旧 token GAE 等价 Oracle。**不改变训练语义。**

2. [`02_macro_sequence_ppo_and_gae.md`](./02_macro_sequence_ppo_and_gae.md)  
   完整宏动作 Sequence PPO、legacy 对照模式、玩家宏决策 GAE。

3. [`03_actor_fuse_amp_checkpoint.md`](./03_actor_fuse_amp_checkpoint.md)  
   零初始化 Actor Residual Fuse、真实 AMP、版本化 checkpoint/warm-start 迁移。

4. [`04_reward_context_delivery.md`](./04_reward_context_delivery.md)  
   分差奖励、13 等级 × 4 leader、贡还上下文审计、4 小时安全分段 Kaggle 交付。

## 条件阶段

5. [`05_symmetry_and_belief_conditional.md`](./05_symmetry_and_belief_conditional.md)  
   座位/重复牌对称性；belief 必须先通过按 deal 切分的信息量预检。

6. [`06_opponent_pool_search_final_conditional.md`](./06_opponent_pool_search_final_conditional.md)  
   Checkpoint opponent pool；value 资格通过后才允许搜索；最终分级训练与盲测工具。

## 执行规则

- 不要一次连续执行多个 Prompt。
- 不要把六个文件合并成一次任务。
- 在 Codex 验收前不要执行下一阶段。
- 不允许执行者使用 `git reset`、`git clean`、`git checkout` 覆盖当前工作区。
- 不允许删除、回滚或覆盖与当前阶段无关的现有修改。
- 不允许未经用户明确授权启动正式长程 Kaggle 训练。
- FableDan 只能作为最终外部评估对手，不得用于蒸馏、监督标签或训练数据。

## 每次返回给 Codex 的材料

请保留工作区原状，并发送 DeepSeek 的完整回复，尤其包括：

1. 修改文件列表；
2. `git diff --stat` 与关键 diff；
3. 新增测试名称；
4. 实际测试命令和完整结果；
5. smoke/recovery 输出；
6. 新产物路径与 SHA256；
7. 每项验收结论；
8. 明确列出未实施的后续阶段内容。

Codex 会直接检查真实文件、数学公式、反例测试、checkpoint/recovery 和是否越界实施。


