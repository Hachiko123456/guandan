# MARVEL v1.4 阶段 1 基线冻结报告（Baseline Report）

- 生成时间（Created）: 2026-10-18T00:00:00+00:00
- Manifest: `phase1_baseline_manifest.json`（hash `38704bb57238950e79f362f94790519467dda21c8300821a1e954fb0be8a007a`）
- snapshot_inputs hash: `a47d92b3fa3b4037a8734ef9ab3843377074434b0b7e8ad2126a88b6b1467972`

## Git 基线

- commit: `fa4a7af3227dd8d76663bd9cf2b0e3a659d70373`（branch `codex/marvel-guandan-migration`，dirty=True）
- tracked diff sha256: `f670a8dc68b253e72059e6c3e95c34fb130bced33b250d98467152c4e6cfd555`（36093 bytes）
- untracked source files: 51

## 环境（Environment）

- Python 3.12.0（CPython），torch 2.13.0+cpu，numpy 2.3.5
- C++ backend identity hash（checkpoint loader identity）: `538476e420e48081a9fd3bd580d4906f8e2cf42923d4180edc3a5dc721cf439b`
- C++ native identity hash（raw extension identity）: `09907f0527c89ba4773b0d8cb0630ec0435ec4898015407c1d2f87fde09e33ec`
- C++ source manifest hash: `66c6621dd4cd6c48521d2588e9a55e487b926206b1776e837d04625dcdb0305c`
- extension sha256: `927a06c837bbf110a23369c3fe2a5693316f9cb966a9e0ff0c5870b606e93930`
- source matches loaded binary: True

## 模型基线（Model baseline）

- M09 checkpoint: `runs/comparison_delivery/dataset/m09_candidate_iter002000.pt`（8588181 bytes）
- sha256: `af0df1eaeb67c88e467ee78bf6b0494c49d000f5733d655ee81b40351d4ca9c8`（frozen candidate match: True）
- architecture/algorithm/iteration: v1.3 / ippo / 2000
- resolved config hash（declared/recomputed）: `9dc92b6910ca900038724df31829377ab64e52ed615c89c6fc1d4ff0aaedc1bb` / `9dc92b6910ca900038724df31829377ab64e52ed615c89c6fc1d4ff0aaedc1bb`
- checkpoint native backend identity: `5b8b7300f8bb32f0892959990ff9414dce67b51bd371e1063a8d9e78894c30de`

## 正式 strict loader（未削弱）

- status: **rejected**
- error: `ValueError: checkpoint C++ backend/build identity mismatch`
- 旧 M09 checkpoint 在 Kaggle（Linux/CPython 3.13）构建，其 native backend identity 与本地扩展不一致，正式 strict 校验如实拒绝且未被削弱。下方模型输出来自单独标注的 diagnostic-only 手工加载，不构成评估证据。

## 固定 probe（Diagnostic-only model probe）

- probe inputs: `../../runs/marvel_v14_phase1/probe_inputs.pt` sha256 `0f5f10aed4b6a5ea1725b0785eb53b6e52fca474de91c80797e7eac292b9f78e`
- probe outputs: `../../runs/marvel_v14_phase1/probe_outputs.pt` sha256 `9a153eef13e6d5f7cd557ec04727487cbfe13c30fc815a740a774add1f740a0b`
- macro probe: `../../runs/marvel_v14_phase1/macro_probe.json` sha256 `1ab6c8aa4650d3348259dae7a9e704ecd6359bc227d84b972df85fd76dcbea71`
- model output content sha256（timestamp-free）: `d9812f4e9e6289f9b15ed02ec40a7be511cbf77943a7c390a50b33ac4df6d06b`
- state_dict sha256: `b7c5954ba6383ddf55df6611fdd4513c2ab59ae2f387a9e2dcd49fe064acf39a`
- parameter counts: total 2061314

## 真实环境宏分段 probe（Real-environment macro probe）

- seed 90001: tokens=310, engine commits=127, macro count=127, commit match=1, length_mean=2.441, tau_mean=9.041, pass=80, commit=47, incomplete=0, violations=0
- seed 97920: tokens=320, engine commits=132, macro count=132, commit match=1, length_mean=2.424, tau_mean=9.406, pass=84, commit=48, incomplete=0, violations=0
- seed 105839: tokens=347, engine commits=151, macro count=151, commit match=1, length_mean=2.298, tau_mean=9.020, pass=94, commit=57, incomplete=0, violations=0
- seed 113758: tokens=335, engine commits=141, macro count=141, commit match=1, length_mean=2.376, tau_mean=9.226, pass=84, commit=57, incomplete=0, violations=0

## 交付物（Delivery artifacts）

- `runs/bootcamp_week1_delivery/datasets/session1_start/guandan-marvel-source.zip`: 754929 bytes, sha256 `691d2633d84f95800412a162a340c903539202c520b7eecb2c409f13f92f3f1f`
- `runs/bootcamp_week1_delivery/datasets/session1_start/marvel_bootcamp_week1_plan.json`: 3418 bytes, sha256 `5602240e44b8c3ca40e25e198b70510c1c7b1094ac036112415cf95501c16952`
- `runs/bootcamp_week1_delivery/datasets/session1_start/warm_start_policy.pt`: 8588181 bytes, sha256 `af0df1eaeb67c88e467ee78bf6b0494c49d000f5733d655ee81b40351d4ca9c8`
- `runs/bootcamp_week1_delivery/delivery_manifest.json`: 7602 bytes, sha256 `dd8fc1c4490370b763f9a84156528cf76687ad6d0d999f8d11f866df5a5ab00e`
- `runs/comparison_delivery/dataset/guandan-marvel-source.zip`: 670971 bytes, sha256 `1a1ddf71642624d66524ff9b7696817dd8830ba29e6eb42f24c76bbd827f6b12`
- `runs/comparison_delivery/dataset/comparison_control.json`: 8317 bytes, sha256 `dec919fb176b191d2ec59ed5c61d11fb53da6c8fe74b4cf93a65ec3c2c89dd67`
- `runs/comparison_delivery/dataset/m09_champion_iter000100.pt`: 8316529 bytes, sha256 `b00a2c178e8b3e99d80418ae283a785285639b8172a0d96232beef7233b9d41c`
- `runs/comparison_delivery/delivery_manifest.json`: 1080 bytes, sha256 `676afbb41a9ea031f6ee9d6223649c95dba215c77ab8bf10a0e92f8c381251ee`
- `configs/marvel_m09_full_plan.json`: 5627 bytes, sha256 `163fc07b4415bb39253c8639d2a02d0147c2a8993ffdcc9d32465afdc613031b`
- `configs/marvel_m09_main_training_plan.json`: 4869 bytes, sha256 `04e4e9ee332481829b9d2d8c6fe510f3fdc4d444c44ad7ba5ee5dd30fa426c22`
- `configs/marvel_m09_two_stage_continuation_plan.json`: 1998 bytes, sha256 `d21e3b4f552ed9c7ce37a831bd7f27108e6029104d778dfe3e2710edf6cbf697`

## 生成物（Generated outputs）

以下文件由本次构建生成；其 sha256 记录在外部 build result 中，manifest 不写入自身哈希以避免自引用失效。

- macro_probe: `../../runs/marvel_v14_phase1/macro_probe.json`
- manifest: `phase1_baseline_manifest.json`
- probe_inputs: `../../runs/marvel_v14_phase1/probe_inputs.pt`
- probe_outputs: `../../runs/marvel_v14_phase1/probe_outputs.pt`
- report: `phase1_baseline_report.md`
- result: `../../runs/marvel_v14_phase1/baseline_build_result.json`

## 范围声明（Scope statement）

阶段 1 仅冻结本基线，并新增宏动作分段、legacy GAE 等价 Oracle、诊断指标与测试。训练语义未改变：token PPO、token GAE、flat ±1 reward、actor、context 与 AMP 状态均未改动，且未提前实施阶段 2 及以后内容。
