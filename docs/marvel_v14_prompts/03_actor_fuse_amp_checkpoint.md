# 阶段 3 实施 Prompt：Actor Residual Fuse＋AMP＋Checkpoint 迁移

> 将本文件正文完整复制给 DeepSeek。阶段 1–2 必须已通过独立验收。

## 任务

在 `D:/project/guandan` 实施 actor residual fuse、真实 AMP 和版本化 checkpoint 迁移。不得改变已验收的 macro PPO/GAE 数学语义。

## 约束

- 不得 reset/clean/checkout 覆盖当前工作区。
- 完成后停止，不得改奖励、全分布、belief、pool 或 search。

## 一、Actor Fuse

保留旧路径：

[
z=sodot a
]

增加：

[
u=[s,a,sodot a,|s-a|]
]

[
h=operatorname{ResGatedMLP}(W_{in}u)
]

[
z_{new}=z+W_{out}h
]

[
logit=old_logit_head(z_{new})
]

要求：

- feature_dim=256。
- `fuse_layers` 真正用于 actor，默认2。
- `output_proj=Linear(256,256,bias=False)`。
- `output_proj.weight` 严格零初始化。
- output_proj 标记 no weight decay。
- input projection/trunk 正常初始化。
- 只 gather 合法候选运行 MLP，再 scatter 回去。
- 旧乘法路径必须保留。
- 不得直接用 concat 大 MLP 替换旧路径。

不得把 ResGatedMLP 最后一个 SwiGLU/down_proj 置零冒充零残差；其 residual identity path 仍会输出非零。必须使用独立零初始化 output projection。

## 二、初始化等价

加载旧 v1.3 权重到新模型后、任何 optimizer step 前：

[
W_{out}=0Rightarrow z_{new}=z
]

FP32、eval、autocast off、相同输入下必须检查：

- `torch.equal(old_logits,new_logits)`
- `torch.equal(old_probs,new_probs)`
- `torch.equal(old_argmax,new_argmax)`

AMP容差测试不能替代FP32强验收。

## 三、梯度门控

第一次 backward：

- output_proj.weight grad 应非零；
- trunk/input_proj grad 应为零或数值零，这是预期。

第一次有效 optimizer step 后：

- output_proj 不再全零；
- 新旧 logits 可开始不同。

第二次 backward：

- output_proj、trunk、input_proj grad 都应非零。

必须测试这三个时点，不得错误要求所有新参数第一次 backward 就有梯度。

## 四、Optimizer

- output_proj 不做 weight decay。
- norm/bias/embed 遵守现有分组。
- trunk大矩阵正常 weight decay。
- 报告所有新参数所在optimizer组。
- 验证 AdamW 和可选 Muon 路径。
- 新参数不得遗漏在optimizer外。

## 五、版本与 checkpoint

建议旧 `v1.3`，新 `v1.4-macro-fuse`（或仓库一致的明确版本名）。不得把含 fuse 的模型仍标为 v1.3。

### 正式评估

- v1.3 checkpoint按v1.3模型构造并 strict=True。
- v1.4 checkpoint按v1.4构造并 strict=True。
- 声称v1.4但缺actor_fuse key必须失败。
- 通用评估loader不得白名单缺失actor_fuse。

### Policy-only warm start

允许 v1.3 evaluation checkpoint → v1.4 policy-only warm start，使用 strict=False 但只允许明确前缀：`actor_fuse.*`及当前配置批准的训练期辅助头。任何旧主干/state/action/logit/value key缺失或unexpected key必须失败。

加载后验证：output_proj全零、固定probe新旧逐位一致，并记录迁移schema/source/target architecture。

### Exact resume

拒绝 v1.3 training recovery → v1.4 exact resume。v1.4 recovery只能恢复到完全兼容的v1.4语义。

## 六、AMP

T4使用fp16，不假定bf16。

实现：

- Transformer/Linear/MLP进入autocast。
- softmax、selected logprob、macro logratio、ratio、KL、entropy、GAE、value target、loss accumulation保持FP32。
- RMSNorm方差继续FP32。
- GradScaler。
- `scale(loss).backward()` → `unscale_(optimizer)` → gradient clip → `step/update`。
- scaler state进入checkpoint/recovery并在resume恢复。
- `delta=delta.to(z.dtype)` 后再 residual add，避免意外promotion。

## 七、AMP验收

固定同一数据、不重新采样，比较 AMP off/on：

- logits/probs/macro logprob误差；
- policy/value loss误差；
- gradient norm误差；
- argmax一致率；
- scaler scale和skipped steps；
- NaN/Inf；
- learner/完整iter耗时。

FP32 fuse初始化必须bitwise兼容；AMP只允许预先定义的小容差。

## 八、DDP与恢复

验证：

1. 所有rank新参数初始化hash一致。
2. 第一次backward W_out梯度同步。
3. 第一次step后参数hash一致。
4. 第二次backward trunk梯度非零且同步。
5. `find_unused_parameters=False` 不报错。
6. full → recovery → resume后模型、optimizer、GradScaler、iteration、sampler/RNG正确恢复。

## 九、A/B

Control：macro PPO + macro GAE + fuse off。
Treatment：相同但 fuse on。

必须从同一 M09 v1.3 warm start、相同seed/发牌/局数开始；先在 AMP off 下隔离架构效果。不得从已训练Control切换fuse冒充A/B。

## 十、禁止事项

不得改reward/context，不加belief/pool/search，不删除旧乘法路径，不放宽strict loader，不把v1.3 recovery当v1.4 exact resume。

## 十一、交付报告

包含修改文件、模块结构、参数量、稀疏计算、optimizer分组、FP32逐位一致、第一次/第二次梯度数字、AMP结果、checkpoint迁移、DDP/recovery、A/B（若执行）、全量回归和风险，并声明未实施阶段4以后内容。完成后停止。
