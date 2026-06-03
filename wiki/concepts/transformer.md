---
tags: [architecture, attention, encoder-decoder, nlp]
related:
  - attention-is-all-you-need
  - bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding
---

# Transformer

完全基于 self-attention 机制的序列转录模型架构，由编码器和解码器堆叠组成，替代了 RNN/CNN 的序列建模方式，成为现代 LLM 的基础架构。

## 架构概览

**编码器**：N=6 层，每层包含 multi-head self-attention + position-wise FFN，均带 residual connection + layer norm。

**解码器**：N=6 层，每层包含 masked multi-head self-attention + cross-attention（关注编码器输出）+ FFN。

## Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) = \text{softmax}\left(\frac{QW_i^Q (KW_i^K)^T}{\sqrt{d_k}}\right) VW_i^V$$

缩放因子 $\sqrt{d_k}$ 防止 softmax 进入梯度极小区。

## Position-wise FFN

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

隐藏层维度 $d_{ff}=2048$（输入/输出 $d_{model}=512$），ReLU 激活。

## 位置编码

使用正弦函数：
$$\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$\text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

让模型通过相对位置学习注意力模式，后被 [[rope]] 取代为事实标准。

## 影响

Transformer 衍生出三大路线：
- **Encoder-only**：BERT、RoBERTa（理解型任务）
- **Decoder-only**：GPT 系列、LLaMA（生成型任务）
- **Encoder-decoder**：T5、BART（序列转录任务）
