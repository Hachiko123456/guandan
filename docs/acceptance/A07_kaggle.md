# A07 Kaggle 部署验收

命令：`python scripts/run_acceptance.py --stage A07`；最终验收门槛是在干净的会话中运行 Kaggle 笔记本/脚本。

要求不得使用 Windows 专用路径，也不得进行 C++ 构建；在运行时检测硬件；进行规则冒烟测试和短时 GPU 训练；定时保存检查点；在新会话中恢复；正确处理 Kaggle 输出。
