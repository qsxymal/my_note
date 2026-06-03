---
tags: [training-objective, speculative-decoding, llm]
related:
  - better-and-faster-large-language-models-via-multi-token-prediction
---

# Multi-Token Prediction

一种训练目标，在共享 trunk 上添加 n 个独立 output head，使 LLM 同时预测未来 n 个 token，兼具训练质量提升和推理加速（self-speculative decoding）。

## 核心设计

共享 Transformer trunk → n 个独立 Transformer 层 head → 共享 unembedding 矩阵。每个 head 预测位置 t+i 的 token，损失为 n 个交叉熵之和。

## 内存高效实现

Logits 向量 V >> latent d 是显存瓶颈。标准实现同时物化 n 个 head 的 logits → O(nVd)。采用**顺序 forward/backward**：依次计算每个 head 的 forward + loss + backward 并立即释放，峰值内存降至 O(Vd)（与 n 无关）。

## 推理加速

精确保留 head 1 用于标准自回归。其余 n-1 个 head 用于 self-speculative decoding——同时预测后续 tokens 作为草案，用原始 head 验证并行接受。

| 配置 | 加速比 | 平均接受步长 |
|:---:|:------:|:----------:|
| 4-token (code) | 3.05× | 3.5 |
| 4-token (text) | 2.74× | — |
| 8-byte | 6.39× | — |

加速在 batch size 1-42 范围内保持一致，不同于传统投机解码需小 batch 配合 draft model。

## 最佳 n 值

| 任务 | 最佳 n |
|:----:|:------:|
| 代码生成 | 4 |
| 摘要 | 2 |
| Byte 级建模 | 8 |
| 算法推理 | 2 |

## 效果

- 代码 13B：HumanEval pass@1 +12%，MBPP +17%
- 促进 induction head 形成（2-token 预测 1M 参数即展现 induction 行为）
- 生成式任务显著提升，判别式任务中性或略差
- 训练成本增加约 7-12%（n=4），相对推理加速可接受
