---
tags: [distributed-training, activation-memory, megatron]
related:
  - reducing-activation-recomputation-in-large-transformer-models
  - tensor-parallelism
---

# Sequence Parallelism

在非 tensor parallel 区域（layer-norm, dropout）沿序列维度分布激活的并行技术，由 Megatron-LM 团队提出。

## 动机

Tensor parallelism 能拆分 MLP 和 Attention 中的矩阵乘法，但 layer-norm、dropout 和 residual 连接的激活**仍然在各 TP rank 上复制**——对应 $10sbh$ 的激活内存未分布化。

序列维度的操作（layer-norm, dropout）与位置互相独立，天然的并行维度。

## 实现

重定 Megatron-LM 的 `f`/`ḟ` 通信算子为 `g`/`ḡ`：

| 算子 | 前向 | 反向 |
|------|------|------|
| `g` | all-gather（序列维度） | reduce-scatter |
| `ḡ` | reduce-scatter（序列维度） | all-gather |

通信量与原有的 all-reduce 完全相同（reduce-scatter + all-gather = 1 次 all-reduce），**零额外通信开销**。

激活内存从 $sbh(10 + 24/t + 5as/ht)$ 降至 $sbh(34/t + 5as/ht)$——除以 `t`（tensor parallel size）。

## 限制

- 仅适用于序列维度独立的操作，不适用于所有模型架构
- 需要 TP 配合使用
