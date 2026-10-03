# A04 观测和批量环境

命令：`python scripts/run_acceptance.py --stage A04`。

实现供策略安全使用的观测编码器、`GuandanEnv` 和 `GuandanEnvBatch`。测试固定的形状/数据类型/掩码、不泄漏隐藏手牌、单实例与批量执行的等价性、独立的随机数生成器（RNG）/状态、终局/重置语义、序列化，以及溢出错误。
