# A10 评估预跑、进度与预算验收

本轮只修改评估执行和交付，不修改训练 checkpoint、游戏规则、策略参数或训练目标。无需重新训练。

## 本地交付门槛

1. 默认每个候选/对手组合预跑 10 局，不计入正式胜率。两算法各 random、rule、旧快照、跨算法快照，共 8 个组合，默认合计 80 局预跑。
2. 正式目标由 `TARGET_GAMES_PER_OPPONENT` 指定，必须是 4 的倍数。100 表示每个组合 25 个基础发牌 × 4 个座位轮换，不是整套只跑 100 局。
3. random/rule 不因更换 snapshot 重复执行；默认正式目标100时，全套为800局。两个 worker 都结束预跑后才做整个套件的预算判定。
4. 默认 `AUTO_CONTINUE_AFTER_WARMUP=False`：预跑结束后以 `warmup_complete` 正常退出，不调用 input、不无限等待，也不宣称正式评估完成。
5. 预估超过预算：`stopped_by_budget`，不开始正式评估。估计值包含单位、已测局数、速度和不确定性提示；无完成样本时 ETA 为 unknown/null，不能为0或虚构速度。
6. 真进度至少包含：algorithm、matchup、opponent、phase、completed/target、elapsed、games/s、token steps、ETA。首次对局未结束时仍输出已推进 token / 活跃局数；不把线程心跳说成对局进度。
7. 每局完成及时保存，不等待一个完整 batch/report；异常时已完成记录留存，正式统计不得混入预跑或未终局记录。
8. 错误与结束有结构化状态，父进程检查子进程 exitcode；子进程崩溃、无结果退出不能无限 Queue.get。
9. ZIP先验证路径、重复项、解压上限，再写入独立目录。checkpoint与manifest的SHA256、算法和update核对。接受ZIP和包含workers的展开目录。
10. 日志实时刷出到Notebook控制台和文件。GPU采样来自NVML或nvidia-smi，失败明确unavailable，不把缺失当0。标明设备总体利用率不等于本worker核函数耗时。
11. Notebook `EXECUTE=False`只生成计划；每次新输出目录，不能覆盖前次评估。打包在正常结束、预算退出、异常退出时均尝试执行；平台硬杀不能承诺finally执行。
12. 出包验证包括：源码hash逐文件核对、Notebook各cell语法和CLI参数联调、当前真实checkpoint本地小规模双worker预跑，以及强制预算退出的可重复测试。不能只靠AST或两个mock测试宣布完成。

## GPU远端门槛

本机没有双T4时必须标注未执行远端性能验收。用户首次上传只跑预跑，查看实测ETA和每张卡采样后再决定规模；不承诺倍数提速或指定利用率。

## 统计边界

预跑用于估时而不是能力结论。正式统计按完整四座位轮换基础发牌聚类；95%区间不是模型强度保证，小样本试验不能替代完整泛化评估。不同级牌和连续多手不在本轮范围。

## 本次本地真实 checkpoint 验证（2026-10-07 审查）

输入为用户的 `guandan-dual-gpu-progress.zip`，没有训练或改写任何输入权重。以下全部为本机 CPU 双进程，**不是 T4 x2 性能验收**。

| 验证 | 预跑 | 正式 | 结果 |
|---|---:|---:|---|
| 最小预跑 | 8 | 0 | warmup_complete |
| 默认每对手预跑10局 | 80 | 0 | warmup_complete |
| 自动继续的小规模全矩阵 | 8 | 32 | complete |
| 超过估时预算 | 8 | 0 | stopped_by_budget |

核验逐局文件数量与 result 计数一致；正式每基础发牌四座位齐全；预跑/正式 seed 集合无交集；所有保存游戏均为真实终局且未截断。默认80局预跑本机约355秒，仅为功能验证时间，不据此预测Kaggle。

机器可读证据：`runs/a10_real_validation.json`。原始日志：`runs/a10-default-warmup-console.log`、`runs/a10-real-formal-console.log`、`runs/a10-real-budget-console.log`。这些本地产物不纳入源码ZIP。

独立子代理审查的进度路径疑点由主代理用实际文件及专门测试复核：SuiteProgress 的根为 `<output>/live`，Notebook 读取 `<output>/live/progress.json`；worker telemetry 在 `<output>/<algorithm>/live/system.json`。二者已一致，没有为了消除疑点而错误移动路径。
