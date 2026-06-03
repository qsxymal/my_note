---
tags: [normalization, transformer, training-stability, deep-network]
related:
  - deepnet-scaling-transformers-to-1000-layers
  - attention-is-all-you-need
---

# DeepNorm

一种修改残差连接的 normalization 函数，配合理论推导的初始化将 Transformer 稳定扩展到 1,000 层。

## 核心公式

$$x^{l+1} = \text{LN}(\alpha \cdot x^l + G^l(x^l, \theta^l))$$

其中 α > 1 是残差权重超参数。相比 Post-LN（α=1），DeepNorm 通过 α 放大残差连接分支，缩小每个 sublayer 的更新贡献。

## 核心洞察

论文证明 Post-LN 的不稳定性根源**不是梯度爆炸，而是模型更新（model update）幅度失控**——LN 输入 x 的幅度 ||x|| >> √d 导致 LN 梯度极小 → 梯度消失 → 模型无法继续更新。

DeepNorm 的 α > 1 等效于将每个 sublayer 的学习率隐式降低 1/α，同时保持 Post-LN 的梯度结构。

## 初始化策略

基于 Theorem 4.2（模型更新量 ||ΔF|| 被 bound 在 O(η)，与深度无关）推导出各层权重缩放因子：
- Encoder：FFN/value/output 投影缩放 0.87·(N⁴M)^(-1/16)
- Decoder：FFN/value/output 投影缩放 (12M)^(-1/4)

其中 N = encoder 层数，M = decoder 层数。

## 与常见 Norm 对比

| 方案 | 训练稳定性 | 深层精度 | 可扩展深度 |
|------|:---------:|:--------:|:---------:|
| Post-LN | ✗ | ✓ | ~20 层 |
| Pre-LN | ✓ | ✗ | ~100 层 |
| **DeepNorm** | ✓ | ✓ | **1,000 层** |

## 局限性

- 仅在 NMT 任务上大规模验证，NLU 任务未充分探索
- 1,000 层 vs 200 层的边际收益递减
- 理论分析假设 SGD，实际使用 Adam，存在 gap
- "窄而深"（hidden dim 512）的设计可能不如"宽而深"效果好
