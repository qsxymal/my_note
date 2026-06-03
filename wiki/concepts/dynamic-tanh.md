---
tags: [normalization, activation, transformer, architecture]
related:
  - transformer
  - deepnorm
  - alibi
---

# Dynamic Tanh (DyT)

一种替代 normalization layer（LN/RMSNorm）的元素级操作，用于 Transformer 架构。DyT(x) = gamma * tanh(alpha * x) + beta，其灵感来自于对 LN 的 empirical observation：LN 的输入-输出映射呈现 tanh 状的 S 形曲线。

## 核心公式

$$\text{DyT}(x) = \gamma \cdot \tanh(\alpha x) + \beta$$

其中：
- α：可学习标量（控制缩放强度），记为 Dynamic Tanh 的 "Dynamic"
- γ, β：可学习 per-channel 向量（与 normalization layer 的 affine 参数相同）
- tanh：$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

## 关键特性

- **元素级操作**：不计算统计量（mean/std），不跨 token/channel 聚合
- **S 形映射**：接近零的输入线性变换，极端值被 squash——模拟 LN 的核心行为
- **无 normalization 语义**：不是 normalization，是 element-wise activation-like 操作
- **零额外计算开销**：比 LN 更简单（无 mean/std 计算）
- **Drop-in replacement**：直接替换 Transformer 中所有 LN/RMSNorm 层（attention, FFN, final）

## α 的初始化

- 默认 α₀ = 0.5，大多数场景不需要调
- LLM 训练推荐 α₀ 可调高（~1.0），尤其对 attention block 中的 DyT
- α 在训练中与 activation 的 1/std 紧密相关，自动调整

## 效果

| 模型 | LN/RMSNorm | DyT | Change |
|:---|---|---:|---:|
| ViT-B (ImageNet) | 82.3% | 82.5% | +0.2% |
| ViT-L (ImageNet) | 83.1% | 83.6% | +0.5% |
| LLaMA 7B (zero-shot) | 0.513 | 0.513 | — |
| LLaMA 70B (zero-shot) | 0.549 | 0.549 | — |
| DiT-XL (FID) | 19.9 | 20.8 | +0.9 |

## 与其他方法的对比

| 方法 | 类型 | ViT-B | ViT-L |
|:---|:---|---:|---:|
| LN | baseline | 82.3% | 83.1% |
| Fixup | 初始化 | 77.2% | 78.1% |
| SkipInit | 初始化 | 74.1% | 75.6% |
| σReparam | weight-norm | 82.5% | 83.0% |
| **DyT** | element-wise | **82.8%** | **83.6%** |

DyT 显著优于 Fixup/SkipInit，与 σReparam 持平或更好。

## 与 normalization 的关系

- 不像 LN/RMSNorm 那样按 token 计算统计量
- 保留了 LN 的核心功能：squash extreme values + 近似线性变换中间值
- 不是 normalization，但达到了 normalization 的效果
- 挑战了 "normalization layer 在现代深度网络中不可或滅" 的传统认知
