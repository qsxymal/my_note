---
tags: [quantization, llm, w4a16, on-device]
related:
  - awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration
  - gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers
---

# AWQ (Activation-aware Weight Quantization)

MIT Song Han Lab 提出的 **4-bit weight-only** 量化方法，面向端侧 LLM 部署。

## 核心思想

并非所有权重都同等重要。通过激活分布识别最重要的 ~1% weight channels（salient channels），用 scaling 变换放大这些通道的值，在统一精度下实现对重要权重的"软保护"。

## 与 SmoothQuant 的关系

师出同门，技术一脉相承：

| 方面 | SmoothQuant | AWQ |
|------|------------|-----|
| 量化方案 | W8A8 | W4A16 (weight-only) |
| 目标场景 | 云端推理加速 | 端侧部署 |
| 核心操作 | 缩小激活 outlier | 放大重要权重 |
| 数学工具 | Per-channel scaling (等价变换) | Per-channel scaling (等价变换) |
| Salient 识别依据 | 激活幅度 | 激活幅度 |

## 与 GPTQ 对比

- AWQ 不需要反向传播和重建，**速度快 3x**
- AWQ 泛化性更好，不会过拟合校准集
- GPTQ 在纯语言建模上精度接近 AWQ，但在多模态和 instruction-tuned 模型上退化明显

## 配套框架

**TinyChat** — 端侧推理引擎，支持 kernel fusion 和 platform-aware weight packing。
