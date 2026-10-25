# DMC 主线当前状态与接手指南（权威入口）

> 日期：2026-10-26 · 分支：`dmc-v1`（推送至 origin）· 本文件是**任何新会话/新模型接手 DMC 主线的第一读物**，10 分钟读完即可工作。任务范围与授权永远以当前人类请求为准。
> 历史背景（为什么弃 PPO 走 DMC、论文依据、完整计划）见 [00_DMC_MIGRATION_PLAN.md](00_DMC_MIGRATION_PLAN.md)。

## 1. 一段话现状

DMC（Deep Monte-Carlo，DanZero/DouZero/FableDan 路线）已在 `guandan_marvel/dmc/` 完整实现并验证：**Gate L（Kaggle 4h 小规模验证）于 2026-10-26 判定 PASS**——vs rule 65.2%（≥65% 门槛）、vs random 89.1%、qz_r2 0.193 上升、12,289 局 @ 1.024 局/s、健康全 0。旧 PPO 线（m09）8000 局对 rule 仅 55~66% 且平台化，已冻结为历史。**下一步是 Gate A（阶段 6）**：9h session、启用并行拓扑、5~10 万局、正面 h2h m09-iter2000，详见 [06_gate_a_kaggle.md](06_gate_a_kaggle.md)（2026-10-26 修订版）。

## 2. 阶段状态（详表见 00 计划）

| 阶段 | 状态 |
|---|---|
| 1-4（配置/枚举/Q头/采样器） | ✅ 已合入 |
| 5 Gate L 首跑 | FAIL 已归档（rule 59.8%、吞吐 0.711 局/s） |
| 07A/07B/07C（重叠/窗口/快速驱动） | ✅ 已合入 |
| 5' Gate L 重跑 | ✅ **PASS**（2026-10-26） |
| **6 Gate A** | ⬜ **当前任务**（交付构建待做） |
| 7 规模化+增强 | 大纲在 [07_scale_and_enhancements.md](07_scale_and_enhancements.md) |

## 3. 环境事实（接手必读）

| 环境 | 用途 | 备注 |
|---|---|---|
| `.venv-gpu/` | torch 2.13.0+cu126（CUDA 可用）；跑 `tests/dmc`、交付构建/校验 | 默认 `python` 是 CPU torch，别用 |
| `.venv-kaggle/` | kaggle SDK 2.2.5；跑 `tests/unit/test_kaggle_pipeline*` 与一切上传/监控命令 | |
| 代理 | **上传/下载必须 `https_proxy`**（本机 `http://127.0.0.1:7897`；blob 走 googleapis 直连会挂死） | |

**已知既有失败（不是回归，别去"修"）**：`test_memory_budget`（期望常数对应 C++ 4096 horizon，python oracle 是 1024）；`test_kaggle_pipeline_cli/token_config` 各 1 项在 `.venv-gpu` 下假失败（缺 SDK，`.venv-kaggle` 下通过）。

## 4. 验证协议（本项目的工作制度）

1. **计划/复核会话**（用户的一个长期会话，即"协调者"）：写分阶段 prompt（本目录格式）→ 实现会话执行 → 报告贴回复核者 → **复核者独立重跑验收命令 + 读 diff + 查证据文件** → 通过才 commit + push → 下一阶段。
2. **实现会话**（Claude Code/DeepSeek/GLM Flash）：一个阶段 = 一个全新会话；`.claude/settings.json` deny 了 `git commit`——提交由复核者或用户执行，实现会话别尝试。
3. 每阶段产物：代码 + 测试 + `runs/dmc_stageN_<date>/REPORT.md`（含 NOT_RUN 清单）；证据放 `runs/`（gitignore，仅本地）。
4. 预声明门槛不许事后改；失败按处置分支走（回查实现/修吞吐），不烧额外配额硬试。

## 5. 地雷与教训（全部实测踩过）

1. **Kaggle API 对不存在的 dataset/kernel，详情端点返回 403 不是 404**——backend 的存在性探测已改走账号列表端点（`dataset_slug_exists`/`kernel_slug_exists`）。
2. SDK 断点上传状态存于 `%TEMP%\.kaggle-uploads`，失败后残留会让下次 `resume=True` 死锁——backend `create_dataset` 已做预清理；交付路径必须用绝对原生分隔符。
3. `fetch` 对 DMC run **必然显示 FETCHED_INCOMPLETE**（管线结果校验器绑旧 serial_compare 格式）——输出仍会完整下载，权威判定在包内 `artifacts/RESULT.json`。
4. **一包一资源**：每个内容不同的交付 = 1 个新私有 dataset + 1 个新 notebook；相同内容复用同一 run_id（内容哈希派生）。用完删旧远端资源。
5. 吞吐瓶颈地图（07A 剖析）：瓶颈曾是 `EnvBatch.step` 的逐步验证契约（占采集 53~66%），07C 用差分等价的 `dmc/env_fast.py` 绕开（200 局逐字节等价、无容差）；**下一个吞吐杠杆 = Python 环境步进的 C++ 化**（阶段 7 立项）。
6. 观测窗口 = head 128 + tail 896（总数 1024 不变）；`SAMPLER_STATE_SCHEMA` 已升 `-v2`，旧恢复包会被拒绝（新训练身份）。
7. 07A 并行拓扑（双进程 collector/learner）已实现且测试完备，但 **Gate L 重跑实际跑的是串行**——Gate A 交付必须显式启用并行（预计 ~1.4 局/s）。
8. 上传命令统一走 `scripts/kaggle.ps1`（内部用 `.venv-kaggle`）；远端作业与本地 PC 无关，`--no-wait` 提交后可关机。

## 6. 命令速查

```powershell
# 测试（tests/dmc，~15 分钟）
& .venv-gpu\Scripts\python.exe -m pytest tests/dmc -q
# 交付构建 + 校验（Gate A 用新 builder，参考 05 阶段的 build_dmc_gate_l_delivery.py 模式）
& .venv-gpu\Scripts\python.exe scripts/verify_dmc_delivery.py --delivery <交付目录>
# 管线 dry-run（.venv-kaggle）
& .venv-kaggle\Scripts\python.exe scripts/kaggle_pipeline.py prepare --delivery <交付目录> --dry-run
# 上传（需 https_proxy；--no-wait 提交后可关机）
& .venv-kaggle\Scripts\python.exe scripts/kaggle_pipeline.py run --delivery <交付目录> --no-wait
# 状态/监控/下载
& .venv-kaggle\Scripts\python.exe scripts/kaggle_pipeline.py status|watch|fetch
```

## 7. 不在 git 里的资产

`runs/`（全部阶段证据、checkpoint、结果 zip——按仓库约定 gitignore，仅本地）；`configs/kaggle_account.local.json`（Kaggle token，永不入库）；两次 Gate L 结果 zip 归档在 `runs/kaggle_pipeline/dmc-gate-l-*/`。

## 8. 里程碑（对 FableDan ~400 万局/2.5 亿样本的追赶线）

Gate A（~1-2 周，含 m09 h2h + FableDan 基线可选）→ 25M 样本（~3-4 周，预期 15-30%）→ 150M（~2 月，30-40%）→ 250-400M（3-6 月，对拼 50%+）。长期目标：Botzone 天梯验证"职业邻近"水平。
