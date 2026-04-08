# Qwen Overfit 诊断归档

本文档不再维护原始的逐轮诊断过程。

当前与 `Qwen overfit` 相关、仍然有效的结论已经合并进：

- `docs/research_history_zh.md`

建议以后只把这份文档当作归档入口，而不是当前状态文档。

## 保留下来的核心工程结论

1. `Qwen` 路线的 overfit 失败，主因曾是工程链路错误，而不是模型本身完全不会学。
2. 三个最关键的问题是：
   - 训练 / 验证 token 来源不一致
   - `lengths` / padding 处理错误
   - overfit 诊断默认还带随机 token-drop augmentation
3. 修复这些问题后，`m2t / t2m / mc` 的少量样本 overfit 已经可以成功。
4. 但这些修复只证明 pipeline 打通，并没有解决 full train 的泛化失败。

## 当前建议

如果需要理解当前分支的研究状态，请优先阅读：

1. `docs/research_history_zh.md`
2. `mGPT/archs/mgpt_qwen.py`
3. `mGPT/models/mgpt.py`

原始长篇诊断过程已经过时，因此不再在此处保留。
