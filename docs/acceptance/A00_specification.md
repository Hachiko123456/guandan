# A00 规范验收

## 范围

冻结单副牌局 v1 规则、公开/私有信息边界、方案 B 动作协议、观测维度，以及训练期间的步进语义。

## 必需命令

```powershell
python scripts/run_acceptance.py --stage A00
```

## 必需证据

- `docs/rules.md` 包含明确的 v1 项目规则约定和来源状态章节。
- `docs/environment_contract.md` 包含固定的 v1 维度和 token 范围。
- `docs/action_protocol.md` 明确规定标量 token、COMMIT、溢出、重置以及 token 级 PPO 的语义。
- A00 测试全部通过，且没有跳过（skipped）或预期失败（xfail）的测试。
- 报告记录 Git 提交、存在未提交修改的文件（脏文件）和输入指纹。

## 人工审查门槛

监督代理必须检查这些文档，并确认仍待处理的整场比赛级功能已明确推迟到后续阶段，而不是被悄然省略。仅凭测试通过的报告，绝不能设置 `accepted: true`。
