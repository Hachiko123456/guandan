# 自动交付独立审查问题与修复证据

2026-10-24。全部验证只用临时样例和假服务，不使用 v10，不连接真实 Kaggle。

## R1：重复结果包

证据：`tools/serial_compare/pipeline.py` 的 summarize 将同一个结果 ZIP 同时写在 artifacts 和父目录。初版下载验证要求恰好一个 ZIP，会把正常下载判为不完整。

复现：临时输出目录放两份字节相同的 `guandan_serial_compare_results.zip`，调用 verify_results。

修复：允许多份哈希相同的副本；哈希不同则拒绝猜测。由独立测试覆盖，最终结果汇总在验收报告。

## R2：SDK 在本机 token 文件上的隐式认证优先级

证据：已安装 `kagglesdk==0.1.38` 的 `kaggle_env.py` 第106行仍直接读取 `~/.kaggle/access_token`，不受 `KAGGLE_CONFIG_DIR` 隔离；`kaggle` 包首次 import 自动 authenticate。仅清除环境变量并设置临时 config dir 不能保证只使用 local.json 中明确填写的账号，也可能在 import 阶段发起 token introspection。

复现设计：用假 token-reader 代替真实本机文件读取，令默认位置返回一个临时 token，拦截 introspection/网络。初始化应不读取默认 token，且最终凭据必须为显式 username/key。不得为了测试读取或改写用户真实 token。

修复：在首次 import、认证和每次 SDK 调用期间屏蔽默认 token 读取辅助函数，结束后恢复；用户名/key 由调用方明确提供。回归测试在干净子进程中覆盖首次/已有 import、成功/异常退出、懒加载 HTTP 会话身份，不读取真实 home 凭据。主代理独立重跑适配器及 CLI 测试，共79项通过（含其余用例）。

## R3：Dataset 处理中不应提前读取文件列表

证据：适配器初版在状态未 ready 时仍读取 Dataset 文件列表；处理中的文件列表可能尚不可用。复现采用 pending 状态与抛404的假列表接口，应继续等待而不误判上传失败。

修复：适配器仅在 ready 状态读取/分页文件列表，未 ready 时返回已知状态、隐私及版本。已补充假列表404且不得调用的测试，主代理独立运行通过。
