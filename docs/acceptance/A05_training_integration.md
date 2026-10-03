# A05 训练集成

命令：`python scripts/run_acceptance.py --stage A05`。

将环境接入 IPPO/VRPO。要求进行至少 100 次更新的冒烟运行；损失和梯度必须为有限值；使用合法动作掩码；遵循 token 步级别的 GAE 语义；完成 CUDA/CPU 检查；支持检查点保存/加载和恢复训练，并进行协议/规则版本检查。
