---
tags: [llm, scaling, quantization]
related: [[llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale]]
---

# Emergent Outliers in Transformers

大语言模型在规模扩大到 6.7B 参数时**相变式涌现**的极端 outlier 特征。

## 特征

- 幅度高达其他维度的 **20 倍**
- 集中在 attention projection 和 FFN expansion 层
- 所有 Transformer 层和约 75% 的序列维度受影响
- 但仅集中在 **~7 个特征维度**（13B 模型以下）
- 每个 2048 token 序列约 150,000 个 outlier，但只有 7 个 unique 维度

## 影响

将 outlier 特征维度置零：
- Top-1 attention softmax 概率下降 >20%
- Validation perplexity 退化 600-1000%
- 随机删除等量特征的影响 <0.3%

## 与模型规模的关系

- < 2.7B: 约 25% 的层有零星 outlier
- 6.7B: **相变点**，所有层和 75% 序列维度受系统性影响
- 这种涌现现象解释了为什么小模型（<350M）可以无损量化，而大模型需要特殊处理
