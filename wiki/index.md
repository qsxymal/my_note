# Wiki Index

<!-- Auto-generated catalog of wiki pages. Updated on each ingest. -->

## Entities

## Concepts

- [Quantization](concepts/quantization.md) — 模型量化技术概述（absmax, zeropoint, vector-wise, SmoothQuant, LLM.int8(), AWQ, GPTQ）
- [Mixed-precision Decomposition](concepts/mixed-precision-decomposition.md) — LLM.int8() 中分离 outlier 用 16-bit 计算其余用 8-bit 的技术
- [Emergent Outliers](concepts/emergent-outliers.md) — 大模型在 6.7B 参数时相变式涌现的极端 outlier 特征
- [SmoothQuant](concepts/smoothquant.md) — 将激活量化难度通过数学等价变换迁移到权重的 W8A8 量化方案
- [AWQ](concepts/awq.md) — 基于激活分布识别重要权重通道的 4-bit weight-only 量化方法
- [GPTQ](concepts/gptq.md) — 基于 Hessian 二阶近似的 weight-only 量化，与 AWQ 形成两条技术路线对比

## Sources

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](sources/llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale.md) — 首次实现无精度损失的 175B 参数级 Transformer Int8 量化推理，核心发现 outlier 特征在 6.7B 参数时相变式涌现
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](sources/smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models.md) — W8A8 量化方案，通过迁移量化难度到权重实现纯 INT8 GEMM，最高 1.56x 加速
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](sources/awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration.md) — 基于激活分布保护 ~1% salient weight channels 的 4-bit weight-only 量化，无需反向传播，泛化性优于 GPTQ
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](sources/gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers.md) — 基于 Hessian 二阶近似的 weight-only 4-bit 量化，首次实现 175B 模型单卡推理
