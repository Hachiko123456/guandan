# 阶段 4 实施 Prompt：分差奖励＋完整训练分布＋Kaggle 交付

> 将本文件正文完整复制给 DeepSeek。阶段 1–3 必须已通过独立验收。

## 任务

在 `D:/project/guandan` 实施掼蛋分差奖励、完整训练上下文、评估覆盖修正和可在约4小时额度内安全续跑的Kaggle分段交付。不得启动正式长训。

## 约束

- 不得 reset/clean/checkout 覆盖现有工作区。
- 不得修改已验收的 macro PPO/GAE/fuse 数学语义。
- 完成后停止，不得实施 belief、opponent pool 或 search。

## 一、分差奖励

环境 raw reward：

[
R_{raw}=egin{cases}pm3&DOUBLE_DOWN\pm2&HEAD_THIRD\pm1&HEAD_LASTend{cases}
]

每名玩家取所属队伍 reward。训练使用：

[
R_{train}=R_{raw}/3
]

即双下±1、头三±2/3、头末±1/3。这样保留真实相对目标，同时降低旧value尺度冲击。

审计并修改：C++终局赋值和反序列化不变量、Python参考引擎/state、candidate_win派生、comparison/evaluation汇总、测试、metrics、checkpoint semantic contract。candidate_win必须按符号，不得依赖 `==1.0`。

记录：raw team/player reward、normalized training reward、outcome class、mean rank-point reward；配置/检查点记录 `reward_semantics=guandan_rank_points`、raw范围和scale=1/3。

## 二、奖励验收

三类 outcome各构造至少一局定向牌谱，Python/C++逐项比较：ranking、winner_team、outcome_class、raw rewards逐seat、team rewards、serialized round trip。必须完全一致。

## 三、上下文 7A：13等级×4 leader、无进贡

训练schedule覆盖 rank 3..15、leader 0..3，共52 contexts，初始均匀权重。`training_data_manifest_hash` 必须覆盖schedule顺序和权重。每局provenance记录实际level/leader/previous_result/deal/context hash，不能只记声明coverage。

## 四、上下文 7B：previous_result/贡还

扩展versioned schedule schema支持：`previous_result=None`，或包含ranking、winner_team、outcome_class、tribute_cancelled。覆盖none、single tribute、double tribute、cancelled、winner team和合法ranking变化。所有payload必须通过规则对象验证。

## 五、贡还观察审计

当前event内部token会求和；贡还resolved event可能包含多条transfer。启用7B前必须验证：

1. policy observation含所有规则允许公开信息；
2. 不泄漏隐藏牌；
3. 不同贡还关系在求和表示后是否产生语义碰撞；
4. 若发现碰撞，必须给出具体反例，把7B标为blocked/deferred；仍可交付7A，但不得伪称贡还覆盖生效。

## 六、Warm-start课程

建议版本化课程：

- iter 0–20：100%旧context（level15, leader0, previous=None）；
- iter 20–100：(p_t=(1-alpha_t)p_{old}+alpha_t p_{target})，(alpha_t=(t-20)/80)；
- iter 100后：100%目标分布。

课程必须由iteration确定，能exact resume，进入semantic contract和provenance，不依赖墙钟。

## 七、评估修正

不得用8副牌却通过请求参数metadata声称实际覆盖13级。manifest coverage必须根据实际生成deal rows计算。

区分：

- 快速anchor：只作健康/趋势，不作强度声明；
- selection：实际覆盖全部等级、leader及已启用贡还模式，四座位轮换，按deal聚类；
- final holdout：独立seed/hash，不用于调参，候选冻结后一次性运行。

## 八、Kaggle分段交付

用户单次可用额度约4小时；不能继续只有11小时主动停止配置。

构建：

- start交付：policy-only warm start；
- repeatable resume交付：可重复运行直到绝对最终iteration；
- 每段 `max_wall_seconds` 留足收尾，例如约3.5小时，具体值版本化；
- 到点在迭代边界保存checkpoint、原子构建recovery、完成output复制并正常返回 `budget_stopped`；
- 使用完全相同source package identity；
- 不依赖平台强杀自动保存。

## 九、重建交付

重建并校验 source ZIP/manifest、start Dataset、repeatable resume Dataset、notebook、plan、README、delivery manifest、warm-start identity、semantic hash、native backend hash、upload ZIP。文档hash必须与实际文件一致。

## 十、本地验证

至少：

1. full warm start 2 iter；
2. 构建recovery；
3. resume 2 iter；
4. macro指标连续；
5. curriculum/context provenance连续；
6. raw/train reward连续；
7. optimizer/GradScaler/RNG/sampler恢复；
8. completed-game预算精确；
9. source identity严格匹配；
10. 不兼容source recovery拒绝。

## 十一、禁止事项

不加belief、pool、search，不做FableDan蒸馏，不启动正式长训，不改macro公式，不伪造coverage，不削弱recovery fail-closed。

## 十二、交付报告

包含修改文件、奖励公式/scale、三类双引擎实测、schedule schema、52 context实际smoke分布、贡还审计与是否blocked、curriculum恢复、评估隔离、start/resume说明、2+2 iter实际输出、全量回归、所有sha256和风险，并声明未实施条件阶段。完成后停止。
