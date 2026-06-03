---
tags: [activation, gating, architecture, convnet]
related:
  - gated-convolutional-language-model
  - transformer
---

# Gated Linear Unit (GLU)

GLU（门控线性单元）是一种由 Gated Convolutional Network（Dauphin et al., ICML 2017）提出的门控激活函数，定义为输入线性变换 A 与经过 sigmoid 门控的线性变换 B 的逐元素乘积：

$$GLU(x) = (x * W + b) \otimes \sigma(x * V + c)$$

其中 `*` 表示卷积或线性变换，σ 为 sigmoid 函数，⊗ 为逐元素乘法。

## 设计动机

与 LSTM 风格的门控机制（Gated Tanh Unit, GTU）相比：

$$\text{GTU}(x) = \tanh(x * W + b) \otimes \sigma(x * V + c)$$

GLU 去掉了输入路径的 tanh 非线性，允许梯度直接通过线性路径传播。这一设计有两个优势：

1. **缓解梯度消失**：线性路径 A 在反向传播中提供了畅通的梯度通道（∂/∂A = σ(B)），不像 tanh 会对梯度进行压缩
2. **加速收敛**：实验显示 GLU 收敛到同等 perplexity 所需的 epoch 数仅为 GTU 的约 1/3

## 实验对比

| 门控/激活 | 公式 | WikiText-103 PPL | 说明 |
|-----------|------|------------------|------|
| Tanh | tanh(A) | ~75 | 无门控，梯度"消失"严重 |
| GTU | tanh(A) ⊗ σ(B) | ~68 | LSTM 风格，梯度经过 tanh 压缩 |
| ReLU | max(0, A) | ~60 | 简单非线性，无门控 |
| **GLU** | **A ⊗ σ(B)** | **~47** | 线性路径 + sigmoid 门控 |

## 变体与扩展

GLU 的设计启发了一系列后续变体：

- **SwiGLU**（PaLM, 2022）：`Swish(xW) ⊗ (xV)`，将 sigmoid 替换为 Swish，已成为 LLM 的标准 FFW 激活
- **GeGLU**（-）：`GELU(xW) ⊗ (xV)`，使用 GELU 作为门控
- **ReGLU**：`ReLU(xW) ⊗ (xV)`
- Gated Linear Attention（Yang et al., 2024）中的 output gating 也继承了 GLU 的设计理念

## 总结

GLU 的核心创新在于认识到门控中的"线性路径"比"非线性路径"对训练效率更重要。这一观察对现代 LLM 的激活函数设计（SwiGLU 成为默认选择）产生了深远影响。
