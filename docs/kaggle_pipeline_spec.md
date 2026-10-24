# Kaggle 自动交付实施规范

## 2026-10-24 临时新版 Token 认证验证补充

用户明确提供临时 KGAT token 并授权测试，原“禁止账号网络测试”仅对上轮配置任务有效。本轮允许该账号的认证与只读 Dataset/Notebook 列表验证，不允许上传、改写、删除资源或启动任何训练，仍不使用 v10 测试。

发现本地账号文件仍为空 username/key，截图提供的是新版 token，原只支持 Legacy 的文件格式不能直接使用。扩展同一个 local.json 支持 `api_token`；保留 Legacy 兼容，禁止双模式混填。临时 token 仅写入已忽略的账号文件，不写入代码、测试、报告或命令行。用户测试后撤销并替换该字段即可。

验收：真实 introspection 活跃/账号确认、只读列表可访问；新旧账号格式与 SDK Bearer 身份的本地回归测试；不得输出 token 或原始异常正文。主代理负责 core/CLI/文档与真实只读验证；SDK子代理只允许修改 backend.py 和对应适配器测试，不得读取真实账号文件或连接网络。

2026-10-24。用户最新约束：只配置供下次使用；v10 已上传，禁止用 v10 测试，禁止本次上传/启动远端作业。旧方案中的 v10 实验验证安排由本规范替代。

## Spec

- 提供独立 `.venv-kaggle`，不改训练环境。使用官方 kaggle Python 客户端，版本以本地实际安装验证后锁定。
- 用户唯一填写位置 `configs/kaggle_account.local.json`，字段 `username`、`key`，兼容 Kaggle Legacy API Credentials 下载的 kaggle.json。该文件被 Git 忽略；提供无密钥 example。不存在时创建空模板，不覆盖已有账号。
- 公共配置 `configs/kaggle_pipeline.json` 保存 T4×2、私有、离线、预算及本地产物位置。不默认选择 v10 或自动搜索最新历史交付。
- `prepare --delivery <新交付目录>` 离线验证新包，生成唯一 run_id、仅含 payload 的 Dataset staging、绑定原 Notebook 的 kernel staging、run.json。可以接 `--build` 使用既有 serial_compare builder 从明确配置的构建输入生成新包；不修改已封存交付或训练语义。
- `run` 复用最新 prepare 的显式记录，自动认证、上传私有 Dataset、等待处理、提交 GPU Notebook、监控并回收；默认等待，`--no-wait` 只提交。支持显式 run_id。重复 run 不重复上传/开训。
- `status`、`fetch`、`watch` 恢复已有任务；下载严格指定 Notebook 版本。未知提交结果只对账，不自动二次提交。
- `doctor` 检查解释器、依赖、文件配置和本机账号是否填写；默认离线、不打印凭据、不调用 Kaggle。`--dry-run` 使用占位账号且不导入或调用远端客户端、不写 staging。
- 所有远端资源默认私有；staging 只允许指定输入，账号不进 Notebook、manifest、日志。CLI 输出不包含原始服务错误正文（可能含凭据）。
- 平台完成与业务完成分开；下载结果绑定 plan hash、结果清单及文件哈希，缺失/不完整时不得声明 VERIFIED。
- 本地中断可恢复状态；每 run 排他锁，提交之前持久化 SUBMITTING，成功后记录 kernel version，异常保留不确定状态。

## 验收条件与命令

1. 只填账号文件即可使用准备好的 CLI/PowerShell 入口；无需安装依赖或再配 owner。
2. 默认无历史交付；空账号可完成 doctor/dry-run，真实 run 给出准确本地填写路径。
3. 临时小样例 + 假远端验证：私有 staging、Dataset 未就绪不提交、重复 run、提交响应丢失对账、跨版本输出隔离、错误/不完整结果、凭据脱敏、锁。
4. 真实 Kaggle SDK 的签名与返回对象在本地核实；本次不发起账号/网络作业测试，不使用 v10 做测试。
5. 主代理独立审查新增 diff、测试断言并亲自运行测试，记录实际结果和未验证的远端边界。

拟执行：`.venv-kaggle/Scripts/python.exe -m pytest tests/unit/test_kaggle_pipeline*.py -q`、CLI doctor / dry-run、PowerShell 入口帮助检查、git diff --check。

## 分工与允许修改文件

- 主代理：`tools/kaggle_pipeline/core.py`、`results.py`、`__init__.py`、`scripts/kaggle_pipeline.py`、公共配置、账号模板、PowerShell 入口、.gitignore、用户文档与验收报告。
- deepseek-coder 子代理：只写 `tools/kaggle_pipeline/backend.py` 及 `tests/unit/test_kaggle_pipeline_backend.py`，实现可注入的 Kaggle SDK 适配器；不得写 core/config/docs、不得读取账号值、不得调用真实 Kaggle API、不得使用 v10。
- 后续独立测试子代理：只写 `tests/unit/test_kaggle_pipeline.py`（接口稳定后），不得改生产实现；禁止远端或 v10 测试。
- 禁止修改模型、训练器、实验计划、既有交付与恢复协议。已有大量工作区变更保留。

## 本轮交付范围

完成本地工具和配置及独立验收；账号由用户填写，真实上传/双卡运行验收留至下次新交付。不得把本地模拟测试表述为 Kaggle 实测通过。
