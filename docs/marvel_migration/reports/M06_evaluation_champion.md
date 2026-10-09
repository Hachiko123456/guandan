# M06 评估、Champion 与 Opponent Pool 阶段报告

- 日期：2026-10-10
- 分支：`codex/marvel-guandan-migration`
- 开始 commit：`1ab5d03602866eda600277353ee1b9c65ab58ae4`
- 实现 commit：`10cb3452f69cd23a3bed848498e4353dcfbe3b9c`
- 执行环境：Windows；Python 3.12；PyTorch 2.13.0+cpu；C++ pybind backend；CPU local
- 阶段状态：`accepted`

## 范围

本阶段在 C++ backend 和 MARVEL V2 checkpoint 上实现独立 selection 系统：versioned training/selection/final deals、固定五 lane 对手矩阵、四座位轮换、按 deal cluster 的 95% CI、promotion 三态决策、champion 原子更新、历史 opponent pool、legacy V1 opponent-only adapter、training evaluator hook，以及与 selection 隔离的 final audit。

没有修改 A01/A02 规则语义，没有删除 legacy V1 训练/模型/只读评估能力，没有比较 FableDan，没有开始 Kaggle 正式训练。

## 实际修改文件

### 新评估运行时

- `guandan_marvel/evaluation/__init__.py`
- `guandan_marvel/evaluation/common.py`
- `guandan_marvel/evaluation/manifests.py`
- `guandan_marvel/evaluation/checkpoints.py`
- `guandan_marvel/evaluation/agents.py`
- `guandan_marvel/evaluation/host.py`
- `guandan_marvel/evaluation/stats.py`
- `guandan_marvel/evaluation/promotion.py`
- `guandan_marvel/evaluation/pool.py`
- `guandan_marvel/evaluation/evaluator.py`
- `guandan_marvel/evaluation/hooks.py`
- `guandan_marvel/training/local.py`
- `guandan_marvel/NOTICE.md`

### 脚本和测试

- `scripts/run_marvel_m06_evaluation.py`
- `tests/marvel_evaluation/conftest.py`
- `tests/marvel_evaluation/test_manifests_checkpoints.py`
- `tests/marvel_evaluation/test_stats.py`
- `tests/marvel_evaluation/test_promotion_pool_hook.py`
- `tests/marvel_evaluation/test_host_integration.py`

### 规范、报告和证据

- `docs/marvel_migration/M06_EVALUATION_PROTOCOL.md`
- `docs/marvel_migration/M06_SOURCE_AUDIT.md`
- `docs/marvel_migration/00_README.md`
- `docs/marvel_migration/02_SOURCE_MAPPING.md`
- `docs/marvel_migration/05_TRAINING_SPEC.md`
- `docs/marvel_migration/06_EVALUATION_CHAMPION_SPEC.md`
- `docs/marvel_migration/09_CHECKPOINT_COMPATIBILITY.md`
- `docs/marvel_migration/10_RISK_AND_DECISIONS.md`
- `docs/marvel_migration/STATUS.yaml`
- `docs/marvel_migration/reports/M06_evaluation_champion.{md,json}`
- `docs/marvel_migration/evidence/M06_evaluation_champion/*`

## 未修改的受保护范围

- `guandan/cards.py`
- `guandan/combos.py`
- `guandan/round.py`
- `guandan/state.py`
- `docs/rules.md`

只读参考仓库保持 clean；M06 只读取 `D:\project\doudizhu` 的通用 evaluation/hook 组织代码。

## 测试命令与原始结果

1. `python -m pytest tests/marvel_evaluation tests/marvel_training -q --junitxml=...`
   - exit 0；43 passed；0 failed；0 skipped；39.39 s。
2. `python -m pytest tests/marvel_core tests/marvel_adapter tests/marvel_cpp tests/marvel_differential -q --junitxml=...`
   - exit 0；180 passed；0 failed；0 skipped；100.32 s。
3. A01-A04 acceptance regression
   - exit 0；152 passed；0 failed；0 skipped；10.49 s。
4. legacy A06 evaluation regression
   - exit 0；155 passed；0 failed；0 skipped；0.06 s。
5. A00 specification
   - exit 0；5 passed；0 failed；0 skipped；0.06 s。

非重叠正式集合合计：535 passed，0 failed，0 errors，0 skipped。

保留的开发诊断：首次直接执行 M06 脚本时，脚本没有把仓库根加入 `sys.path`，因此 exit 1；修复入口后使用最终输出目录完整重跑并 exit 0。该失败不被隐藏，也不计作正式验收通过命令。

## 小规模真实 checkpoint 全流程演练

命令：

`python scripts/run_marvel_m06_evaluation.py --output runs/M06_evaluation_champion/20261010_acceptance_final2 --device cpu`

结果：

- exit 0；
- 使用真实 C++ backend MARVEL V2 IPPO checkpoint；
- 固定矩阵 5 lanes × 1 base deal × 4 rotations = 20/20 完整终局；
- illegal=0，truncated=0；
- seat rotation coverage 完整；
- 逐局 JSONL、plan、cluster summary、promotion decision、champion/pool/final audit manifests 全部落盘；
- promotion=`reject`，原因是 candidate 与 bootstrap champion byte-identical；
- champion manifest hash 前后相同，明确 no-promotion；
- opponent pool 从 manifest 恢复并按记录权重重采样；
- final holdout `uses=[]`，未用于选择。

本地总体吞吐约 `0.583` games/s。该值只描述小规模 CPU evaluation 管线，不是训练吞吐或强度结论。

## 机器可读证据

- `docs/marvel_migration/evidence/M06_evaluation_champion/protocol_schema.json`
- `docs/marvel_migration/evidence/M06_evaluation_champion/deals_manifest.json`
- `docs/marvel_migration/evidence/M06_evaluation_champion/e2e_summary.json`
- `docs/marvel_migration/evidence/M06_evaluation_champion/checkpoint_identity.json`
- `docs/marvel_migration/evidence/M06_evaluation_champion/promotion_recovery.json`
- `docs/marvel_migration/evidence/M06_evaluation_champion/test_results.json`
- `docs/marvel_migration/evidence/M06_evaluation_champion/source_manifest.json`
- `docs/marvel_migration/evidence/M06_evaluation_champion/environment.json`
- `docs/marvel_migration/evidence/M06_evaluation_champion/verification.json`
- `docs/marvel_migration/evidence/M06_evaluation_champion/artifact_manifest.json`

大型日志和 20 局逐局记录位于 `runs/M06_evaluation_champion/20261010_acceptance_final2`，未提交 Git。

## 正确性与隐私审查

- V2 agent 只接收 C++ `EnvStep` 的 actor-facing observations 和 legal rows；未调用 critic ownership；
- legacy V1 通过独立 mirror observation 选择 token，token 必须唯一映射到当前 C++ legal row；终局 ranking/outcome 交叉检查；
- algorithm、architecture、protocol、resolved config hash 和 C++ binary/build identity mismatch fail-closed；
- legacy V1 loader 明确 opponent-only，不能进入 V2 resume/champion；
- promotion atomic replace 注入失败测试证明旧 champion 不变；
- incomplete/CI 不足不会晋升；
- final runner 与 selection/promotion API 分离，holdout candidate lock 和 one-shot use 有专项测试；
- training hook 只传 evaluation-only snapshot，模型/RNG/模拟 optimizer/sampler 状态不变。

## 与参考仓库的来源映射

参考通用 plan/recording/stats/hook 边界；Guandan host、controller 行为、agents、summary、cluster CI 和 manifests 全部重写。详见 `M06_SOURCE_AUDIT.md`。

## 已知限制

1. 每个 lane 仅 1 个 base deal，因此 per-opponent clustered CI 为 inconclusive；这是 E0 管线演练，不是强度样本。
2. Candidate 与 bootstrap champion 相同，因此本轮只验证明确 no-promotion 和旧 champion 保持；成功 promotion 路径由合成统计和原子性单测覆盖。
3. Opponent pool 当前只有一个 champion generation，加 rule/legacy anchors；历史深度将在后续真实晋升中增长。
4. 本地仅 CPU；Kaggle T4×2 实际吞吐、显存和双卡效率仍由 M08 验证。
5. 未使用 final holdout，没有最终强度结论。
6. 未比较 FableDan。

## Blockers

无新的正确性、安全、隐私、恢复或协议 blocker。

## 是否允许启动下一阶段

允许启动 M07 Kaggle DDP 交付阶段；本结论不授权跳过 M07/M08，也不代表已开始或通过 Kaggle 正式训练。
