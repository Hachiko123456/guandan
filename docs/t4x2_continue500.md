# Kaggle T4×2：从 v4 的 iter250 完整续训到 iter500

## 先看这一页

本包是 **2026-10-22 的独立续训包**。起点是你已完成的 v4 iter250，保留模型、优化器、采样器、随机数与 AMP scaler；不是从 iter50 重新训练，也不是只加载权重。

- 绝对目标：**iter500**。
- 每轮全局收集40局：起点累计10,000局，终点累计20,000局，**最多新增10,000局**。
- 中途约 iter375 比较胜率；iter500做更充分的开发集比较。评估独立进程、冻结快照，不更新训练状态。
- 保留 v4 训练配方：clip=0.2、reg=0.01、lr=1e-4 恒定，macro PPO、ActorFuse、NTP=0.02、原贡还课程，rollout48/collect40，native horizon1024。**本次不额外改变学习变量。**
- 双T4分工：训练2个DDP rank/NCCL，元信息走Gloo，AMP、完成轨迹裁剪、每rank2个CPU线程；评估也分配到两张卡。不会为了显示100%占用率而做无用计算；以有效更新和实际局/秒衡量。

## 需要上传什么

只需要交付目录内两个文件，**不要上传整个项目，也不要混用旧 v4 Notebook**：

1. 新建Kaggle Dataset：上传 `guandan_continue500_payload_v1.bin`。它内含旧源码、原Linux native、iter250完整恢复状态、固定对照模型、新驱动和评估牌局。
2. 导入Notebook：`guandan_continue500_v1.ipynb`。
3. Notebook挂载上面的Dataset，Accelerator选择 **GPU T4×2**，Internet可关闭，然后 Run All。首次运行不用改代码。
4. Payload不是ZIP：保持 `.bin` 文件名，不手动解压。Notebook与payload做SHA256配对，错包会开训前停止。

平台必须兼容原恢复环境（原native为 **Linux / CPython 3.13.15**，严格身份绑定此补丁版本）。如果Kaggle更换Python/硬件、资源不足、输入损坏，会在预检停止；不会静默重编译、CPU回退或丢弃优化器重训。

## 为什么不会再把已知评估错误留到训练结束

旧错误是评估请求4096步，而本次native只支持1024。新外置评估器显式传入计划和native共同确认的1024；不是简单把评估异常忽略掉。

顺序是：

1. **身份与资源检查**：输入、旧源码、checkpoint全成员、rank一致性、真实计数、native、双T4、NCCL/Gloo、磁盘。
2. **任何新增训练之前**：当前模型与旧control真实完成48局冒烟（4副牌×4轮换×3臂），覆盖原贡还上下文。失败则零新增训练。
3. **更新/落盘/恢复门槛**：250→252，保存；252→253恢复，再保存；新模型再完成48局评估。均计入500目标内。
4. 固定iter250对照先完成终点512局rule基线及监控128局基线，避免长期训练后才发现对照评估失败。
5. 训练到375，做256局候选快评；与128局缓存基线构成384局三臂监控报告。
6. 训练到500，做1024局候选终点评估；与512局缓存基线构成1536局三臂终点报告。

三臂：candidate vs iter250、candidate vs rule、iter250 vs rule；同牌局四座位轮换，按独立牌局聚类估计95%区间，同时比较胜率、配对rule胜率增量、原始队伍得分。375是监控，不用来挑最佳模型；500是预先固定的开发集终点。FableDan未运行，最终holdout未使用，**不能把这个结果冒称为已超过FableDan**。

技术错误立即停止后续训练，保存最近通过完整性校验的恢复包和错误日志；统计上的 `INCONCLUSIVE` 不是程序错误，也不等于模型无进步。

## 时间、恢复与边界

总安全预算 **4小时**（包含输入、训练、评估与收尾），训练阶段为终点评估保留90分钟；较早门槛阶段保留更大余量。不能保证每台Kaggle都在4小时内达到500：如果吞吐/评估速度不足，主动在可恢复边界保存实际进度，不虚报500、不超500。

- 每50轮及阶段结束保存完整双rank状态，最近恢复ZIP经过逐文件hash检查；模型快照与训练状态分别保存。
- 续跑：先下载本次结果中的 `latest_recovery.zip`，上传为另一个Dataset；使用**同一份Notebook及payload**，在全新kernel/新的RUN_ROOT下，把代码开头 `RECOVERY_ZIP` 改成该文件的Kaggle绝对路径。
- 不能把旧结果总ZIP填入RECOVERY_ZIP；应使用其中的 `latest_recovery.zip`。
- 新恢复必须匹配本包plan hash；不允许拿另一套配方的checkpoint混用。
- 已到500但评估没完：同样恢复，**只补评估，零新增更新**。
- 不允许在已有RUN_ROOT上重复追加。先下载旧输出再用新kernel/新路径，避免历史日志和模型身份混杂。
- 平台断电/强制杀进程/磁盘损坏无法由程序绝对保证，但既有技术错误均设有前置门槛和可见失败状态。

## 输出与怎么判断是否有正向趋势

下载 `guandan_continue500_results.zip`（Notebook最后提供链接）。至少保留：

- `continuation_result.json`：训练状态、实际迭代/局数、评估状态、错误阶段，分别标记。
- `REPORT.md` / `report.html`：摘要、学习曲线、评估节点。
- `learning_curve.json`：375/500三臂胜率、聚类CI、得分与配对差异。
- `training_history.jsonl`：继承的1..250及新增训练记录；不重置计数、不写死每轮20局。
- `performance_summary.json` / `telemetry.jsonl`：阶段耗时、有效吞吐、每卡利用率/显存/功耗；原始逐rank optimizer更新/AMP skip/分阶段耗时。
- `model_iter*.pt`：模型快照，仅用于评估，不能代替完整恢复状态。
- `latest_recovery.zip`：最近完整可恢复训练状态；技术失败后仍应优先下载它。
- `evaluation/`：逐局身份、座位、牌局hash、终局名次/胜负/奖励校验、worker结果及三臂summary。

判断优先级：技术健康通过 → candidate相对固定iter250的棋力结果 → 配对rule增量及区间 → 辅助NTP/value曲线。不要单凭loss下降、权重变化或GPU忙碌宣布棋力提升；小样本CI覆盖0时，结论只能是证据不足。评估使用开发集，不用于最终排名宣称。

## 源码和恢复身份为何这样安排

旧checkpoint严格绑定原训练代码/native。为了不通过篡改source hash来“骗过”恢复：

- 原 v4 source ZIP **字节不变**，原Linux二进制从已校验的旧recovery恢复。
- 旧native身份包含绝对路径：源码固定部署到 `/kaggle/working/guandan_t4x2_combo_v1/source/guandan-marvel-source`。这是有意保留的旧身份路径，不是误用旧训练流程；本次日志/状态仍放在新RUN_ROOT。已有同路径仅在源码与二进制全部相同时复用，绝不覆盖冲突旧文件。Python补丁版本或后端hash不匹配会在新增训练前拒绝。
- 新版调度/评估在独立 `continuation500` 驱动中；只替换观察器以允许500和补充日志，原trainer/sampler/optimizer数学不改。
- 目标/日志/壁钟属于兼容恢复变更，明确记录新driver/plan身份。它不是宣称整个执行器完全未变的exact-resume。
- 本地Windows单卡测试只证明接线/数值/恢复兼容性，不冒充Kaggle双T4验收。Kaggle真实预检门槛仍必须运行通过。

本地测试详情与范围见随包 `LOCAL_VALIDATION.json`；打包身份见 `DELIVERY_VERIFICATION.json`。
