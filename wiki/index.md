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
- [Pipeline Parallelism](concepts/pipeline-parallelism.md) — 将模型按层分段放置到不同加速器的流水线并行策略
- [Tensor Parallelism](concepts/tensor-parallelism.md) — 将单一矩阵乘法按行/列拆分到多 GPU 的 intra-layer 模型并行
- [Sequence Parallelism](concepts/sequence-parallelism.md) — 沿序列维度分布激活，与 tensor parallelism 正交，零额外通信
- [3D Parallelism (PTD-P)](concepts/3d-parallelism.md) — Pipeline + Tensor + Data parallelism 的组合方案，训练万亿参数模型的标准配置
- [Gradient Checkpointing](concepts/gradient-checkpointing.md) — 前向丢弃中间激活、反向重算的 memory-saving 技术

## Sources

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](sources/llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale.md) — 首次实现无精度损失的 175B 参数级 Transformer Int8 量化推理，核心发现 outlier 特征在 6.7B 参数时相变式涌现
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](sources/smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models.md) — W8A8 量化方案，通过迁移量化难度到权重实现纯 INT8 GEMM，最高 1.56x 加速
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](sources/awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration.md) — 基于激活分布保护 ~1% salient weight channels 的 4-bit weight-only 量化，无需反向传播，泛化性优于 GPTQ
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](sources/gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers.md) — 基于 Hessian 二阶近似的 weight-only 4-bit 量化，首次实现 175B 模型单卡推理
- [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](sources/gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism.md) — 通过 batch-splitting pipeline parallelism 实现接近线性加速比的模型并行方案，训练 6B 多语言 NMT 和 557M AmoebaNet
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](sources/megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism.md) — 通过 intra-layer tensor parallelism 在 512 GPU 上训练 8.3B GPT-2 和 3.9B BERT，76% 扩展效率
- [Optimus: An Efficient 2D Method for Training Super-Large Deep Learning Models](sources/an-efficient-2d-method-for-training-super-large-deep-learning-models.md) — 基于 SUMMA 的 2D 模型并行，同时分布参数和激活，1.48× 加速和 8× batch size vs Megatron
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](sources/efficient-large-scale-language-model-training-on-gpu-clusters-using-megatron-lm.md) — PTD-P（3D 并行）+ interleaved 流水线调度，3072 GPU 上训练 1 万亿参数模型达到 502 petaFLOP/s
- [Reducing Activation Recomputation in Large Transformer Models](sources/reducing-activation-recomputation-in-large-transformer-models.md) — sequence parallelism + selective activation recomputation，激活内存降 5×，recomputation 开销从 36% 降至 2%
