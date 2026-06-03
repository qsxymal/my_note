---
tags: [architecture, memory, linear-attention, recurrent]
related:
  - linear-transformers-are-secretly-fast-weight-programmers
  - gated-delta-networks
  - attention-is-all-you-need
---

# Fast Weight Programmer (FWP)

Fast Weight Programmer（快权重编程器）是一种可微的两网络系统概念，最早由 Schmidhuber 在 1991 年提出。其核心思想是**一个"慢网络"通过梯度下降学会用输入依赖的指令编程另一个网络的"快权重"**，使快权重充当短期记忆。

## 历史

FWPs 的历史可以追溯到 20 世纪 90 年代初。在传统神经网络中，权重在训练后固定不变，仅激活值随输入变化。FWPs 通过让权重也随输入依赖变化来实现动态记忆管理。

- **von der Malsburg (1981)**：突触调制（synaptic modulation）——权重 = 慢权重 × 快权重
- **Hinton & Plaut (1987)**：两种权重的叠加，不同学习率
- **Schmidhuber (1991, 1992, 1993)**：首次提出可微的 FWP 系统——慢网络通过外积（outer product）等可微指令编程快权重存储器
- **Schlag et al. (2021)**：改进 FWP 的编程指令，引入 delta rule 式更新

## 核心公式

FWPs 的基本形式：

$$a(i), b(i) = W_a x(i), W_b x(i)$$
$$W(i) = \sigma(W(i-1) + a(i) \otimes b(i))$$
$$y(i) = W(i) x(i)$$

其中 ⊗ 是外积，W_a、W_b 是训练好的"慢权重"，W(i) 是生成的"快权重"矩阵（充当短期记忆）。

## FWPs 与线性注意力

[[linear-fwp|Schlag et al. (2021)]] 证明了线性化 self-attention 与 FWPs 的等价关系：

去掉 softmax 的 self-attention：
$$W(i) = \sum_{j=1}^{i} v(j) \otimes k(j), \quad y(i) = W(i) q(i)$$

线性化 softmax 后的 attention（归一化后）：
$$W(i) = \sum_{j=1}^{i} v(j) \otimes \phi(k(j)), \quad z(i) = \sum_{j=1}^{i} \phi(k(j))$$
$$y(i) = \frac{W(i) \phi(q(i))}{z(i) \cdot \phi(q(i))}$$

因此，现代**线性注意力模型、状态空间模型（SSM）、linear RNNs** 在本质上都是 FWPs 的具体实现。

## 关键设计维度

| 维度 | 选项 | 例子 |
|------|------|------|
| **编程指令（Update Rule）** | Sum rule | Katharopoulos Linear Attention |
| | Delta rule | DeltaNet (Schlag 2021) |
| | Gated delta rule | Gated DeltaNet |
| | Gated sum rule | Mamba2 |
| **容量控制** | φ 函数增维 | DPFP, FAVOR+ |
| | 门控衰减 | Mamba2 (α_t) |
| | 矩阵值学习率 | Longhorn |
| **归一化** | Attention normalization | Katharopoulos |
| | Sum normalization | DeltaNet |
| | 无归一化 | Gated DeltaNet (依赖 L2 norm) |

## 与深度学习的联系

FWPs 的概念在其他领域中也有独立的形式：
- **Hypernetworks** (Ha et al., 2017)：用神经网络生成另一个网络的权重
- **Dynamic convolution** (Klein et al., 2015)：卷积核在输入上动态变化
- **Meta-learning**：学习如何快速调整权重的学习算法
- **Tensor Product Representation** (Smolensky, 1990)：外积绑定的分布式表示

## 总结

FWPs 提供了一个统一的理论框架来理解线性化 self-attention、状态空间模型和各种高效 Transformer 变体。FWP 视角允许从记忆容量、更新规则、归一化等角度系统分析这些模型的优劣，为设计和改进线性时间复杂度序列模型提供指导。
