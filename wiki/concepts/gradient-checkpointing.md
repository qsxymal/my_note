---
tags: [memory, training]
related:
  - gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism
  - reducing-activation-recomputation-in-large-transformer-models
---

# Gradient Checkpointing (Re-materialization)

在前向传播时丢弃中间激活值、反向传播时重新计算的技术，用于减少训练时的峰值显存占用。

## 原理

标准训练：前向保存所有中间激活 $f_i(x)$，反向时直接读取 → 显存 $O(N \times L)$

Re-materialization：前向只保存分区边界的激活，反向时**重新计算**分区内的中间激活 → 显存 $O(N + \frac{L}{K} \times \frac{N}{M})$

## Trade-off

- **节省显存**：可训练更大的模型或使用更大的 batch size
- **增加计算**：反向时需要一次额外前向计算（因此也叫 compute-for-memory trade-off）
- 在 GPipe 中，由于 pipeline 天然有 bubble，重算可以**提前调度**，不会额外增加延迟

## 适用场景

- 大模型训练（GPT-3、LLaMA 等）
- Pipeline parallelism 中天然使用
- 单卡训练大 batch 时的 memory saving 技术

## Selective Activation Recomputation

完整 checkpointing 引入 30-40% 计算开销。选择性重算只对**内存密集但计算便宜**的操作重算（如 attention 中的 $QK^T$、softmax、$AV$），而正常存储 MLP 和线性层激活。

- 内存节省：5×（与 [[sequence-parallelism]] 结合使用）
- 计算开销：从 36% 降至 2%（大模型上）

详见 [[reducing-activation-recomputation-in-large-transformer-models]]。
