---
tags: [position-encoding, context-window, length-extension, rope]
related:
  - extending-context-window-via-positional-interpolation
  - roformer-enhanced-transformer-with-rotary-position-embedding
---

# Positional Interpolation (PI)

通过线性缩放 RoPE 输入位置索引使所有位置落在预训练区间内，配合少量 fine-tuning 即可扩展 LLM 上下文窗口的技术。

## 核心思想

给定预训练上下文长度 L 和目标长度 L'，将位置索引 m ∈ [0, L') 线性缩放到 [0, L)：

$$f'(x, m) = f(x, m/\lambda), \quad \lambda = L'/L$$

相当于将 RoPE 旋转角度从 mθᵢ 改为 (m/λ)θᵢ，使所有位置的旋转角度都落在模型已适应的范围内。

## 为什么 Interpolation 优于 Extrapolation

RoPE 的 attention score $a(s) = \text{Re}(\sum h_j e^{is\theta_j})$ 在预训练区间 [0, L] 内良好有界，但三角函数族 {e^{isθⱼ}} 是通用函数逼近器，在区间外可能急剧增大。

Interpolation 保证位置差 s 始终在相邻整数 grid 点之间，通过 Taylor 展开可得 bound 仅 ~294.73·max|hⱼ|，而 extrapolation bound 大 ~600×（LLaMA 7B 设定）。

## 与其他扩展方法对比

| 方法 | 核心思路 | 是否需要微调 | 原始质量保持 | 外推能力 |
|:---|:--------|:----------:|:----------:|:-------:|
| **PI** | 线性缩放位置索引 | 1000 steps | ≤2% 退化 | 有限 |
| **NTK-aware** | 改变 RoPE 基频 θᵢ | 可选 | 更好 | 更好 |
| **YaRN** | NTK + PI + temperature | 少量 or 无需 | 优 | 优 |
| **ALiBi** | 线性偏置 attention score | 无需（训练即内置） | N/A | 优 |

## 局限性

- 需要 fine-tuning，非零样本方案
- 推理成本随窗口线性增长（O(n²) attention 不变）
- 扩展到极长窗口（128K+）时质量退化明显
- 被后续 NTK-aware 和 YaRN 在无需微调时超越
