---
tags: [quantization, efficiency]
related: [[llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale]]
---

# Quantization

将模型权重或激活值从高精度（FP16/FP32）映射到低精度（Int8）的技术，主要用于减少模型内存占用和加速推理。

## 常见方法

- **Absmax quantization** — 除以张量绝对最大值缩放到 [-127, 127]
- **Zeropoint quantization** — 使用归一化动态范围和零点偏移，利用完整 bit 范围
- **Vector-wise quantization** — 矩阵乘法中每个内积使用独立的归一化常数
- **LLM.int8()** — vector-wise + mixed-precision decomposition，无精度损失的 8-bit 推理
