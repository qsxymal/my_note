---
tags: [architecture, linear-attention, memory, recurrent]
related:
  - gated-delta-networks
  - linear-transformers-are-secretly-fast-weight-programmers
  - mamba2
---

# Gated Delta Rule

Gated delta rule 是一种将 **gating（门控/衰减）** 与 **delta rule（增量更新规则）** 相结合的线性 RNN memory 管理机制，由 Gated DeltaNet（Yang et al., ICLR 2025）提出。

## 核心公式

$$S_t = S_{t-1} \left( \alpha_t (I - \beta_t k_t k_t^T) \right) + \beta_t v_t k_t^T$$

其中：
- α_t ∈ (0,1)：数据相关的标量门控，控制整体状态衰减（来自 Mamba2）
- β_t ∈ (0,1)：数据相关的标量写入强度（来自 DeltaNet）
- S_t ∈ R^{d_v × d_k}：矩阵值隐状态
- k_t, v_t：输入的 key/value 向量

## 设计动机

两种已有的记忆管理机制具有互补优势：

| 机制 | 代表模型 | 优势 | 劣势 |
|------|---------|------|------|
| Gating | Mamba2 | 快速批量清除陈旧信息（α_t→0） | 无法选择性遗忘——所有关联被同等衰减 |
| Delta Rule | DeltaNet | 精准修改单个 key-value pair | 一次只改一个 pair，缺乏快速批量擦除 |

Gated delta rule 将两者统一：
- α_t→0：几乎完全重置记忆（gating 的快速清除）
- α_t→1：退化为纯 delta rule（精准选择性更新）

## 理论视角

### 在线学习框架

通过在线学习目标（Liu et al., 2024）的统一视角，gated delta rule 对应以下优化问题：

$$\min_{S_t} \|S_t - \alpha_t S_{t-1}\|^2_F - 2\langle S_t k_t, \beta_t(v_t - \alpha_t S_{t-1} k_t) \rangle$$

即：在允许状态以 α_t 比例偏离前一步的前提下，使新状态在当前 key 上的输出接近目标 value。

### Fast Weight Programming 视角

从 [[linear-fwp|Fast Weight Programmers]] 视角，delta rule 可视为以 β_t 为学习率的 SGD 优化在线回归目标：

$$S_{t+1} = S_t - \beta_t \nabla \mathcal{L}(S_t), \quad \mathcal{L}(S_t) = \frac{1}{2}\|S_t k_t - v_t\|^2$$

Gated delta rule 则相当于引入自适应的 weight decay 项 α_t，即：

$$S_{t+1} = \alpha_t S_t - \beta_t \nabla \mathcal{L}(\alpha_t S_t)$$

## 硬件高效实现

Gated delta rule 的并行训练通过 **WY 表示**（Bischof & Loan, 1985）实现 chunkwise 加速：

1. 将序列分割为大小 C 的 chunks
2. 在 WY 表示的前处理中将 gating 累积乘积 γ_t 融合
3. 利用 tensor-core 优化矩阵乘法

## 变体和扩展

- **负特征值**（Grazzi et al., 2024）：β_t∈(0,2) 允许负特征值，增强 state-tracking 能力
- **多 Householder 乘积**（Siems et al., 2025）：使用多个 Householder 矩阵乘积提高秩
- **Diagonal-plus-Low-Rank**（RWKV-7）：将 α_tI 扩展为 diag(d_t) - a_t b_t^T，增加表达力

## 相关概念

- [[transformer|Transformer]] — 标准注意力机制
- [[linear-fwp|Linear Transformers as Fast Weight Programmers]] — Delta Rule 的原始理论基础
- 稀疏注意力 — 另一种高效序列建模技术
- 线性注意力 — gated delta rule 所属的更广泛家族
