# M07 Kaggle 自动解包输入兼容性补修协议

- 日期：2026-10-10
- 基线实现：`77a3674993d14d8c87fcab3a5084b1cf0c399b67`
- 基线验收登记：`33e06507041f702e49f05e6ef47b60d7a8909d8f`
- 范围：只修复 Kaggle Dataset 自动解包后的 source/recovery 输入布局
- 非范围：A–F 训练/恢复补修、IPPO/VRPO 数学、DDP 算法、规则、性能、M08、正式长训练和 final holdout

## 1. 单一解析实现

`guandan_marvel/kaggle/bootstrap.py` 是 stdlib-only 输入解析真值：

- 作为源码模块进入 source ZIP；
- 生成器同时把同一字节复制为 Dataset 外部 `guandan_marvel_bootstrap.py`；
- Notebook 在源码可导入前验证 bootstrap SHA-256 并加载它；
- runtime `discovery.py` 复用同一 resolver；
- runner 使用同一 recovery materializer/validator。

不得在 Notebook 中维护第二套 ZIP/目录判断逻辑。

## 2. Source 双形态

Dataset schema 升级为 `guandan-marvel-kaggle-dataset-v2`。Source identity 同时保存：

- 原始 archive 相对路径和原始 ZIP 字节 SHA-256；
- 外部 source manifest 相对路径和 SHA-256；
- source commit；
- source content manifest hash。

支持：

1. ZIP 保留：验证实际 ZIP SHA-256、CRC、无路径越界/符号链接、成员集合及逐文件 hash；
2. ZIP 不存在：递归发现与外部 manifest 身份一致的解包目录，验证 manifest hash、逐文件 bytes/hash、无 symlink、无缺失/未声明文件；
3. 额外包装目录；
4. ZIP 根目录被保留，或内容被平铺到 Dataset 目录；
5. ZIP 和目录并存且内容 identity 相同：视为一个逻辑输入，验证 ZIP 后优先使用只读目录；
6. ZIP 和目录 identity 不同、多个不同目录或冲突 manifest：fail closed。

目录形态不得声称验证了不存在的 ZIP。`archive_sha256_verified=false`，但保留 package 声明的原始 archive identity；不得重新压缩硬凑 hash。

## 3. Recovery 双形态

Recovery schema 升级为 `guandan-marvel-recovery-manifest-v3`，新增 `source_content_manifest_hash`。恢复必须同时匹配：

- source commit；
- 原始 source archive identity；
- source content manifest identity；
- recovery manifest hash 和逐文件 hash；
- 后续 DDP world-size/schema/shard/source/backend/semantic identity。

支持 recovery ZIP 和包含 `recovery_manifest.json` 的解包目录。两者并存且 manifest identity 相同则去重并优先目录；不同 identity 或多个冲突 recovery 必须拒绝。

## 4. 路径和只读边界

所有 manifest 路径必须是安全相对路径，resolve 后保持在声明 root 内。ZIP 路径越界和 symlink 条目、目录 symlink、缺失/额外/篡改文件均拒绝。

`/kaggle/input` 或模拟 input 始终只读。Source/recovery 目录只复制到 working：

- source -> `working/guandan-marvel-source`；
- recovery -> `working/guandan-marvel-recovery`。

编译、native binary 恢复、checkpoint 和输出只写 working/output。

## 5. Notebook 和交付

每个模式 Dataset 目录包含四个文件：

- `guandan_marvel_dataset.json`；
- `guandan_marvel_bootstrap.py`；
- `source_manifest.json`；
- `guandan-marvel-source.zip`。

用户上传这四个文件。Kaggle 保留 ZIP 或自动解包均无需修改 Notebook。不得合并多个模式目录；旧 source/recovery 不得与新包混用。

## 6. 本地验收边界

临时目录模拟 ZIP 保留、自动解包、包装层级、平铺、ZIP+目录并存、篡改、缺失、错误 identity、路径越界、symlink 和冲突输入。必须直接执行生成 Notebook 的初始化 cell。

解包 source + 解包 recovery 还必须完成一次实际 CPU/gloo checkpoint resume。该结果只证明本地输入布局和恢复链路，不构成 Kaggle/T4/NCCL 证据。
