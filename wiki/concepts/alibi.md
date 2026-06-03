---
tags: [position-encoding, attention, extrapolation, architecture]
related:
  - transformer
  - rope
  - positional-interpolation
---

# ALiBi (Attention with Linear Biases)

一种位置编码方法，不将位置信息添加到 word embedding 中，而是在 attention score 上添加与 query-key 距离成正比的线性偏置（bias），从而使模型在推理时可以外推到比训练序列更长的输入。

## 核心公式

对于第 i 个 query qi 和前 i 个 key K：

$$\text{softmax}(q_i K^\top + m \cdot [-(i-1), \ldots, -2, -1, 0])$$

其中 m 是每个 head 独有的固定斜率（non-learned），构成几何序列：

- 8 heads: $\frac{1}{2^1}, \frac{1}{2^2}, \ldots, \frac{1}{2^8}$
- n heads: 从 $2^{-\frac{8}{n}}$ 开始，以相同值为公比的几何序列

## 关键特性

- **长度外推（Length Extrapolation）**：训练时用短序列，推理时可处理长序列。例如 L=512 训练的模型在 3072 tokens 时 perplexity 持续改善
- **零运行时开销**：只需修改 attention mask 矩阵，不新增网络操作
- **接近零内存增加**：仅需存储 n×L×L 的 bias 矩阵（n=heads）
- **偏向近邻（Recency Bias）**：对近距离 query-key 的惩罚小，远距离惩罚大；不同 head 以不同速率增加惩罚
- **无需位置 embedding**：不向 word embedding 或 value 注入位置信息

## 与其他位置编码的关系

- 与 [[rope|RoPE]] 同属"在 attention 内部注入位置信息"路线（而非加在 input embedding）
- 与 T5 Bias 类似但 ALiBi 是静态非学习 bias，T5 是学习型 bias
- 与 [[positional-interpolation]] 互补：ALiBi 天然支持 extrapolation，PI 用于扩展 RoPE 窗口

## 效果

1.3B 参数模型，训练 L=1024、外推到 L=2048，达到与 sinusoidal 训练 L=2048 相同的 perplexity，且训练快 11%、内存少 11%。
