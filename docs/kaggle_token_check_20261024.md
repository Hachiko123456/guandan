# 新版 Kaggle API Token 配置与真实只读验收

2026-10-24。用户明确提供临时 token 授权测试，测试后由用户撤销。**最终验收：账号认证与只读访问 PASS；本地回归 PASS。**

## 配置检查与修正

初查 `configs/kaggle_account.local.json` 仍为空 username/key；截图是新版 KGAT token，不能直接当 Legacy key 使用。已扩展同一文件支持 `api_token`，保留旧 username/key 兼容，拒绝混填。临时 token 仅写入此 Git 忽略文件；认证成功后在同一文件缓存账号名，不将 token 写入代码、测试、运行记录或本报告。

现已确认账号：`hachiko123`。用户撤销临时 token 后，在原文件 `api_token` 字段换成正式 token 即可。账号不变时可保留自动识别的 username；如果更换账号，应删除旧 username 或改成新账号后重新 check。

## 真实验证

用户授权范围内完成两次只读验证：

1. 直接使用官方 kagglesdk 的 token introspection、Dataset 列表与 Notebook 列表接口，认证 active 并返回账号名，全部成功。只保存脱敏结论 `runs/kaggle_pipeline_auth_20261024/readonly_probe.json`，未保存资源明细。
2. 主代理审查新版适配器后，从正式 PowerShell 入口执行：

```powershell
.\scripts\kaggle.ps1 check
```

实际退出码0，返回：

```json
{
  "status": "CONNECTED",
  "authenticated": true,
  "username": "hachiko123",
  "auth_method": "api_token",
  "dataset_read": true,
  "kernel_read": true,
  "remote_mutations": false,
  "training_started": false
}
```

以上请求仅认证和读取，不上传、不改写、不删除资源、不提交作业，不使用 v10 测试。未创建本工具的任务指针，原有上传和运行保持原状态。

## 本地独立验收

SDK实现/对应测试由 deepseek-coder 子代理负责，主代理独立审查关键代码与测试断言，并亲自执行全套：

```powershell
& '.\.venv-kaggle\Scripts\python.exe' -m pytest tests/unit/test_kaggle_pipeline_backend.py tests/unit/test_kaggle_pipeline_token_config.py tests/unit/test_kaggle_pipeline_cli.py tests/unit/test_kaggle_pipeline.py -q
```

**实际结果：169 passed in 18.66s，退出0。** 测试仅使用假 token 与假远端，覆盖原74项SDK测试、新版Bearer身份、首次/已有SDK导入、异常恢复和环境隔离、token失效/身份不符/混填拒绝、临时用户名缓存和并发配置变更保护，以及原有上传流程的本地回归。

其他实际检查：`doctor` 为 READY/api_token/remote_calls=false；`pip check` 无依赖冲突；语法及修改文件空白检查通过；local账号文件仍命中Git忽略；无默认历史交付或latest任务指针。

## 关键变更文件

- `tools/kaggle_pipeline/backend.py`：显式Bearer客户端、账号识别、只读连接检查。
- `tools/kaggle_pipeline/core.py`：双认证格式、离线检查、用户名缓存与owner边界。
- `scripts/kaggle_pipeline.py`、`scripts/kaggle.ps1`：新增 `check` 命令。
- `configs/kaggle_account.example.json`、`docs/kaggle_pipeline.md`：新版token模板与说明。
- `tests/unit/test_kaggle_pipeline_token_config.py`、适配器测试：回归与脱敏断言。

真实上传权限、GPU调度和训练未在本轮验证；本轮结论不扩展为这些能力已实测通过。用户现在可以撤销临时token，换上正式token后执行一次check即可确认新凭据。
