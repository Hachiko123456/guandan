# 总体执行计划

## 任务目标

完成本仓库中独立实现的纯 Python 掼蛋训练项目。规则和模型流水线不得导入、复制或依赖 FableDan。最终目标是在 Kaggle 上实现可复现的训练，并可选扩展信念（猜牌）建模/搜索功能。

## 不可妥协的约束

- 仅使用 Python/PyTorch；除非用户明确更改此要求，否则不得使用 C++ 后端。
- 不得修改 `D:\project\doudizhu` 和 `D:\project\FableDan`。
- 使用 `C:\Users\yhx\.conda\envs\guandan_train\python.exe` 进行本地验证。
- 未经差异审查、针对性测试、全量测试和需求审查，不得采纳子代理的结论。
- 自动生成的报告不得设置 `accepted: true`；只有负责监督的主代理可以设置该值。
- 不得在未说明的情况下自行消解规则歧义。

## 依赖关系图

```text
A00 -> A01 -> A02 -> A03 -> A04 -> A05 -> A06 -> A07 -> A08
```

只能推进第一个 `accepted` 字段为 false 的阶段的实现。

## 阶段一览

| 阶段 | 范围 | 依赖阶段 | 验收门槛 |
|---|---|---|---|
| A00 | 规则、协议、治理与证据运行器 | 无 | 纳入版本管理的规范与治理测试 |
| A01 | 牌、级别、逢人牌与十种牌型 | A00 | 牌张守恒、声明、比较与逢人牌测试 |
| A02 | 原子化单局引擎、出牌轮次、接风、进贡/还贡、排名/奖励 | A01 | 定向规则测试、随机对局测试、不变量测试与终局测试 |
| A03 | 方案 B 前缀状态与 COMMIT | A02 | 完整前缀、规范动作与计数器 |
| A04 | 策略观测与批量环境 | A02/A03 | 形状、掩码、批量等价性与无信息泄露 |
| A05 | IPPO/VRPO 集成 | A04 | 训练更新冒烟测试、梯度值有限与检查点恢复 |
| A06 | 固定发牌评估 | A05 | 换座指标与基线 |
| A07 | Kaggle 安装/训练/保存/恢复 | A05/A06 | 全新 Kaggle 会话中的证据 |
| A08 | 信念（猜牌）采样与搜索 | A06/A07 | 不泄露信息的样本与预算约束内的合法搜索 |

## 执行 profile：本机快速验证与远端完整运行

`configs/acceptance_profiles.json` 定义两个执行 profile：

- `local_fast`：本机开发使用。只验证接线、状态安全、梯度、检查点和少量评估，不用于判断模型能力。
- `remote_full`：Kaggle/远程 GPU 使用。执行完整训练更新、大规模评估、保存和恢复。

协议常量（词表、观测长度、动作长度、合法行容量、通道数）不因 profile 改变。profile 只控制更新次数、rollout 规模、评估局数、检查点间隔、运行时限和信念/搜索预算。

A05/A06 命令格式：

```powershell
python scripts/run_acceptance.py --stage A05 --profile local_fast
python scripts/run_acceptance.py --stage A05 --profile remote_full
python scripts/run_acceptance.py --stage A06 --profile local_fast
python scripts/run_acceptance.py --stage A06 --profile remote_full
```

本机通过 `local_fast` 不等于远端 `remote_full` 通过。报告必须记录实际 profile。

## 通用阶段验收门槛

每个阶段都需要完成实现、针对性测试、适用的性质测试/集成测试、由需求驱动的验收测试、由 `scripts/run_acceptance.py` 生成的报告、主代理差异审查、一次全量测试运行，以及记录在 `project_status/STATUS.yaml` 中的 Git 提交。仅有部分测试通过不等于通过验收。

## 续接规则

每个新轮次都应读取 `project_status/STATUS.yaml`，找到第一个尚未验收通过的阶段，阅读其验收文档，并继续推进该阶段。不得重新开始已验收通过的阶段，也不得要求用户为每个阶段重新提供提示。只有在确实存在尚未解决的产品/规则决策、需要外部权限，或有证据表明阻塞问题反复出现时，才可停止。
