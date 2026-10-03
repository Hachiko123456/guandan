# A06 评估

## 范围

实现固定随机种子、交换座位的队伍评估，以及规则/随机策略/策略快照基线。

## Profile

评估规模来自 `configs/acceptance_profiles.json`：

- `local_fast`：每个对手配置 4 局，并启用座位交换；用于检查评估管线、奖励、排名和报告格式；
- `remote_full`：每个对手配置 250 局，并启用座位交换；用于统计模型表现，不把少量 smoke 局解释为能力结论。

## 最小正确性门槛

- 固定 seed 可复现；
- 四个座位/队伍交换逻辑生效；
- 训练发牌与评估发牌分离；
- 队伍胜负、奖励、排名和置信度元数据正确；
- rule/random/snapshot 基线可以运行；
- 模型或环境错误不能静默回退到备用代理；
- 报告包含模型、规则、动作、环境、seed、profile 和 Git 元数据。

## 命令

```powershell
python scripts/run_acceptance.py --stage A06 --profile local_fast
python scripts/run_acceptance.py --stage A06 --profile remote_full
```

本机 `local_fast` 只能证明评估管线正确；模型能力和胜率结论必须使用 `remote_full` 的样本量。
