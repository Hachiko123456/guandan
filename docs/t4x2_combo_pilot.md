# 新三参数组合 · T4×2 有界训练与提速实验

日期：2026-10-21。本轮是用户选定的新配方实验，不是复跑旧50轮，也不是直接8万局长训。

## 1. 用户需要上传什么

最终交付位置：`D:/project/guandan/runs/t4x2_combo_delivery_20261021/delivery_v1/`。

1. 创建一个新的 Kaggle Dataset，**只上传 `guandan_t4x2_combo_v1.bin`**。不要上传旧ZIP、旧v3的bin、整个源码文件夹或结果ZIP。
2. 创建Notebook，导入 `guandan_t4x2_combo_v1.ipynb`，Add Input 选择刚建的Dataset。
3. Accelerator选择 **GPU T4 ×2**。Internet可关闭，不执行pip联网安装。
4. Run All。Notebook先校验payload、每个内层文件和源码身份；它与bin成对绑定，不能混用不同构建的文件。

`.bin`采用上次已验证的header+Base64(ZIP)传输格式；它本身不是ZIP，避免Kaggle自动展开输入。数据损坏或布局不符时，下载 `guandan_t4x2_combo_v1_bootstrap/bootstrap_diagnostics.json`，不要删校验。

## 2. 这轮到底训练什么

| 项目 | 冻结值 |
|---|---|
| 起点 | 用户回传训练后 candidate_iter000050.pt，SHA256 `1f4de9ad180398c3041e6737588a4f92f59e2e97d07d5868e9066efbee725eda` |
| 继承方式 | v1.4→v1.4 policy-only warm start，保留包括ActorFuse在内的全部90个权重键；新optimizer/sampler/RNG/计数 |
| clip / reg / lr | **0.2 / 0.01 / 1e-4**，三者在本轮都恒定，衰减power显式为0 |
| 算法/模型 | 宏决策IPPO、macro-normalized KL正则、v1.4-macro-fuse，生产模型2783234参数 |
| 分布 | 原52个级牌×leader无贡还上下文；curriculum offset=50，接续原课程进度而不是重新锁回旧级牌 |
| 新功能 | 贡还/NTP/Belief/Symmetry/Pool/Search均关闭 |
| 本轮上限 | **250次新迭代 ×20局=最多新增5000局**，预检2次+恢复到3+主训练到250，不是额外再跑250次 |
| 预算 | 单次会话总12600秒/3.5小时，含准备；预留5400秒/90分钟开发评估与额外保存余量 |

父模型已完成旧pilot的1000局，本轮5000是**新增**，不表示模型整个训练史只有5000局。所有计数以实际完成为准。

这是**单个新配方分支对冻结父模型**的学习筛查，不是同预算旧配方继续训练的严格A/B；不能从结果单独归因clip、reg或lr哪个更好。不同配方不冒充exact resume，不修改旧recovery的哈希。

## 3. 如何让双卡更快，而不制造假忙碌

### 增大采样并发，不改变学习batch

- 两个CUDA/NCCL rank，每张T4承担模型前向、反向与梯度同步。
- global rollout从16提高到24，即**每卡12个环境**；global collect仍20，即每卡10局；epochs=4、mini_batches=8不变。
- 固定数据结构下每条dense轨迹91381768字节；sampler每卡需要 `2×12+10=34` 条，共 **3106980112字节，约2.894 GiB**，低于既有4GiB守卫。
- 这**不包含**模型、optimizer、KV cache、learner激活与临时副本。预检必须记录全尺寸实际allocated/reserved峰值；OOM立即停，不暗中换CPU、减horizon或扩大守卫。
- horizon安全上限仍4096，保留已完成轨迹尾部padding裁剪；learner仍FP16 AMP，采样仍FP32。不用更低采样精度换取隐蔽的策略变化。

### 降低存档开销

- 主训练每50轮做一次完整可靠保存，而非每10轮；最终/预算停止边界也保存。一次意外最多可能损失约50轮进度，需定期下载最近恢复包。
- recovery ZIP使用DEFLATE level=1快速压缩，仍保留分片、optimizer/sampler/RNG/scaler/native、完整历史与哈希检查。
- 避免同一个保存边界的重复manifest校验调用；不取消完整性验证。
- 下一阶段的恢复副本已经完整校验且压缩ZIP仍保留后，提前清理上一阶段冗余raw分片，避免更大采样并发下同时留下四代原始checkpoint挤爆Kaggle工作盘；不会删除唯一恢复包，也不处理用户外部路径。
- 梯度仍走NCCL；Python字典、预算决策、发布状态等元数据改用**独立Gloo CPU侧组**。这不是CPU训练回退；用来避免第二张GPU在等主卡压缩时显示无意义的高busy。

### 用日志判断是否真的变快

新增/保留：

- 每rank有效更新、参数变化探针、AMP scale/跳步、allocated与reserved峰值、rollout/learner/iteration时间。
- `checkpoint_io.jsonl`：保存分片、校验/导出、索引/保留、恢复包、发布等待、总存档耗时。当前存档完成的最终计时会保留在结果日志中；恢复ZIP快照不能自包含自身最终压缩耗时。
- `pilot_iterations.jsonl`：cache预填充/分配、模型前向、采样/验证、native step、观察转换、轨迹写入的host分段累计时间及调用数。包含已有同步/dispatch，不冒充GPU kernel profiler。
- 新的 `macro/approx_kl_old_new`：基于rollout样本的 `(r-1)-log(r)` 估计，和 `macro/macro_kl=KL(pi||U)` 分开命名，后者不能表示策略移动了多少。
- 遥测每5秒保留两卡busy/显存、CPU/RAM/磁盘；报告将近似区分迭代窗口与迭代外窗口，并展示实际端到端耗时。

**本地不是双T4，不承诺始终100% busy或固定加速倍数。** 以相同口径的实测局/秒、每轮墙钟、存档比例、显存与更新有效性评估提速。仅显存占满或NCCL等待显示100%都不算提速。

## 4. 实际执行与停止规则

```text
验证输入与两张T4
 → 当前源码构建native
 → 新配方预检2轮（真实full batch/horizon4096）
 → 保存/恢复到第3轮
 → 预检/恢复全部通过后，继续到本轮绝对250轮
 → 双GPU分片开发评估
 → 保存报告、逐局结果、完整恢复包
```

任何输入、NCCL/Gloo侧组、更新、OOM、非有限值、非法动作、恢复或评估完整性错误立即停止并留证据。不会自动绕过失败，也不会继续超过250或启动另一组实验。预算不足会报告budget_stopped和实际完成计数，不能补写计划数当成完成数。

## 5. 评估如何判断效果

使用**新的**128副独立开发发牌，seed从1810210000开始，与旧32副开发牌不同。每副4个座位轮换、三个arm：

1. 本轮优化前冻结的父策略 vs rule；
2. 本轮candidate vs rule；
3. 本轮candidate vs 本轮冻结父策略。

共1536局，两个GPU各768局。核对task/rotation/物理deal hash/温度/权重身份，检查当前训练和父模型训练的实际deal hash不与开发发牌重叠。final=0，FableDan不运行。

统计继续按独立deal聚类，rule前后配对，保留原Bernstein主判定；没有新增“85%才算学到”的事后门槛。样本不足仍可INCONCLUSIVE。128副牌改善精度，但不保证检出1–2个百分点的小增益，更不能证明超过FableDan。

## 6. 下载和恢复

下载 `guandan_t4x2_combo_results.zip`，主要包括报告、`pilot_result.json`、`performance_summary.json`、训练历史、遥测、分rank日志、逐局评估、初始control、`candidate_iter000250.pt`（完成时）和`latest_recovery.zip`。

中断后：

1. 同时挂载**同一份新combo Dataset**与该新实验产生的`latest_recovery.zip`。
2. 使用同一Notebook，设置`RECOVERY_ZIP`，选一个新的RUN_ROOT，禁止覆盖原输出。
3. 恢复会核对新实验manifest/source/config/原Linux native和world_size=2；不能拿旧50轮的recovery替代warm start。
4. 尚未到250时只补足本轮绝对250；已到250则只恢复策略补评估，不新增训练更新。新会话仍受3.5小时上限。

## 7. 本地与远端验证边界

本地规则/统计/单位和显式CPU/Gloo兼容测试不等于双T4认证。真实模型CUDA smoke使用 `D:/project/guandan/.venv-gpu/Scripts/python.exe` 和显式CUDA，但本机只有RTX3050Ti 4GiB，只能缩小采样批次/horizon做接线验证；不得称其通过正式全尺寸显存验证。

最终本地测试数量、命令、源码hash和实际CUDA smoke结果写入交付的 `LOCAL_VALIDATION.json`。没有自动上传、启动Kaggle或运行FableDan/final holdout。
