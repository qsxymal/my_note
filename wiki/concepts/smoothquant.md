---
tags: [quantization, llm, w8a8]
related:
  - smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models
  - llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale
  - awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration
---

# SmoothQuant

W8A8（权重和激活均为 INT8）LLM 后训练量化方案，通过**量化难度迁移**实现纯 INT8 GEMM。

## 核心思想

激活有 outlier 难量化，权重均匀平坦易量化 → 通过数学等价变换将激活的量化难度**迁移**到权重上，使两者都容易量化。

## 对比 LLM.int8()

| 方面 | LLM.int8() | SmoothQuant |
|------|-----------|-------------|
| 量化方案 | W8A8 + FP16 outlier mixed-precision | W8A8 纯 INT8 |
| 硬件效率 | 差（mixed-precision 分解不易高效实现） | 好（纯 INT8 GEMM 内核） |
| 精度 | 无损 | 无损 |
| 加速 | 大模型 ~2x | 最高 1.56x |
| 覆盖模型 | OPT, BLOOM | OPT, BLOOM, GLM, Llama-1/2, Falcon, Mistral, Mixtral |

## 关键参数

**迁移强度 α**: s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
- 0.5 → OPT/BLOOM 通用最优
- 0.75 → GLM-130B（激活 outlier 更严重）
- 0.8 → LLaMA 系列
