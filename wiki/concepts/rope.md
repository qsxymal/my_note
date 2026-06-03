---
tags: [position-encoding, attention, transformer, llm]
related:
  - roformer-enhanced-transformer-with-rotary-position-embedding
  - attention-is-all-you-need
---

# Rotary Position Embedding (RoPE)

通过旋转矩阵将位置信息注入 query/key，使 self-attention 内积自动仅依赖相对位置的编码方案。

## 核心思想

将 d 维空间分为 d/2 个 2D 子空间，每个子空间对 query/key 向量施加旋转角度 mθᵢ（m 为位置，θᵢ = 10000^{-2(i-1)/d} 为预设频率）：

$$f_{\{q,k\}}(x_m, m) = R^d_{\Theta,m} \cdot W_{\{q,k\}} \cdot x_m$$

旋转矩阵 R^d_{\Theta,m} 是块对角正交矩阵，保证 query-key 内积仅依赖于相对位置 n-m：

$$q_m^\top k_n = x^\top W_q \cdot R^d_{\Theta,n-m} \cdot W_k \cdot x_n$$

## 优良性质

| 性质 | 描述 |
|------|------|
| **相对位置编码** | qₘᵀkₙ 仅依赖于 n-m，而非绝对位置 |
| **长期衰减** | 内积幅度随 \|m-n\| 增大而衰减（近强远弱） |
| **旋转不变性** | 正交矩阵保持向量范数与内积结构 |
| **兼容线性注意力** | 不改变表示范数，可直接用于 Performer 等 |
| **零额外参数** | 仅需数学变换，不引入可学习参数 |

## 与 Sinusoidal 编码的关系

两者共享同一频率公式 10000^{-2i/d}，但 Sinusoidal 是加法注入（位置和 token 内积耦合），RoPE 是乘法注入（位置通过旋转融入 query/key，内积与相对位置解耦）。

## 与 ALiBi 的关系

RoPE 的长期衰减性质与 ALiBi 思路不谋而合——都认为位置相关性应随距离衰减。但 ALiBi 直接在 attention score 上加线性偏置，RoPE 则通过旋转矩阵数学推导自动实现。

## 局限性

- 超出训练长度时旋转角度可能超出模型适应区间 → 被 Positional Interpolation、NTK-aware scaling、YaRN 等解决
- 无法表达位置顺序反转（旋转对称性导致 m 和 -m 不能区分，但在 NLP 中通常不重要）

## 采用 RoPE 的主流模型

LLaMA、Mistral、Gemma、Qwen、Baichuan、GLM、Falcon 等——已成为 LLM 事实标准的位置编码。
