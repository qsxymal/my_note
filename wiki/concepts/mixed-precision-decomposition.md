---
tags: [quantization, llm]
related: [[llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale]]
---

# Mixed-precision Decomposition

LLM.int8() 的核心技术之一，用于处理大模型中的极端 outlier 特征。

## 原理

将 outlier 特征维度（幅度 > 6.0）从矩阵乘法中分离，用 16-bit 精度计算，其余 99.9% 的值用 8-bit 计算。Outlier 维度数量不超过 7 个（13B 模型以下），仅增加约 0.1% 的额外内存。

## 公式

给定权重矩阵 $W \in \mathbb{R}^{h \times o}$，混合精度分解为：

$$C \approx \sum_{h \in O} X_h W_h + S \cdot \sum_{h \notin O} X_{i8} W_{i8}$$

其中 $O$ 是 outlier 特征维度集合，$S$ 是 Int8 的反归一化项。
