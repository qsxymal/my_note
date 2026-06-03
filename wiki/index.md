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
- [RoPE (Rotary Position Embedding)](concepts/rope.md) — 通过旋转矩阵编码相对位置的 LLM 事实标准位置编码方案
- [DeepNorm](concepts/deepnorm.md) — 修改残差连接的 normalization + 理论初始化，将 Transformer 扩展到 1,000 层
- [Block-Attention](concepts/block-attention.md) — 将 RAG 文档划分为独立 block 各算 KV cache、仅 query 关注全局的 attention 变体
- [Sparse Attention](concepts/sparse-attention.md) — 通过稀疏注意力矩阵将 self-attention 从 O(n²) 降至 O(n) 的技术集合
- [Positional Interpolation](concepts/positional-interpolation.md) — 线性缩放 RoPE 位置索引以扩展 LLM 上下文窗口的技术
- [Multi-Token Prediction](concepts/multi-token-prediction.md) — 共享 trunk + n 个独立 head 同时预测未来 n 个 token，兼具训练质量提升和推理自加速
- [PagedAttention](concepts/pagedattention.md) — 借鉴虚拟内存分页管理 KV cache，消除碎片、支持跨请求 memory sharing
- [ELMo](concepts/elmo.md) — 将预训练 biLM 所有层表示加权组合为上下文词向量，开创预训练+微调范式
- [Big Bird](concepts/big-bird.md) — Random + Window + Global 三种稀疏 attention 组件组合，O(n) 复杂度逼近 full attention
- [2D Parallelism](concepts/2d-parallelism.md) — 基于 SUMMA 的 2D 分区模型并行，参数和激活同时分布化，消除激活冗余瓶颈
- [Transformer](concepts/transformer.md) — 完全基于 self-attention 的序列建模架构，现代 LLM 的基础
- [BERT](concepts/bert.md) — 深度双向 Transformer 预训练模型，通过 MLM + NSP 学习双向表示

## Sources

### Architecture — 模型架构

- [Attention Is All You Need (Transformer)](sources/architecture/attention-is-all-you-need.md) — 完全基于注意力机制的序列模型架构
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](sources/architecture/roformer-enhanced-transformer-with-rotary-position-embedding.md) — 用旋转矩阵编码位置信息，使 attention 内积仅依赖于相对位置，已成为 LLM 事实标准的位置编码方案
- [DeepNet: Scaling Transformers to 1,000 Layers](sources/architecture/deepnet-scaling-transformers-to-1000-layers.md) — 提出 DeepNorm + 理论初始化将 Transformer 扩展到 1,000 层
- [Big Bird: Transformers for Longer Sequences](sources/architecture/big-bird-transformers-for-longer-sequences.md) — 通过 random + window + global 三种稀疏 attention 组件以 O(n) 复杂度逼近 full attention
- [Block-Attention for Efficient Prefilling](sources/architecture/block-attention-for-efficient-prefilling.md) — 将 RAG 文档划分为独立 block 各算各的 KV cache，仅 query block 关注全局，TTFT 降低 98.7%

### Parallelism — 并行训练

- [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](sources/parallelism/gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism.md) — 通过 batch-splitting pipeline parallelism 实现接近线性加速比的模型并行方案
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](sources/parallelism/megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism.md) — 通过 intra-layer tensor parallelism 在 512 GPU 上训练 8.3B GPT-2 和 3.9B BERT
- [Optimus: An Efficient 2D Method for Training Super-Large Deep Learning Models](sources/parallelism/an-efficient-2d-method-for-training-super-large-deep-learning-models.md) — 基于 SUMMA 的 2D 模型并行，同时分布参数和激活，1.48× 加速比
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](sources/parallelism/efficient-large-scale-language-model-training-on-gpu-clusters-using-megatron-lm.md) — PTD-P（3D 并行）+ interleaved 流水线调度，3072 GPU 上训练 1 万亿参数模型
- [Reducing Activation Recomputation in Large Transformer Models](sources/parallelism/reducing-activation-recomputation-in-large-transformer-models.md) — sequence parallelism + selective activation recomputation，激活内存降 5×

### Quantization — 模型量化

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](sources/quantization/llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale.md) — 首次实现无精度损失的 175B 参数级 Transformer Int8 量化推理
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](sources/quantization/smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models.md) — W8A8 量化方案，通过迁移量化难度到权重实现纯 INT8 GEMM
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](sources/quantization/awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration.md) — 基于激活分布保护 ~1% salient weight channels 的 4-bit weight-only 量化
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](sources/quantization/gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers.md) — 基于 Hessian 二阶近似的 weight-only 4-bit 量化，首次实现 175B 模型单卡推理

### Representation — 表示学习

- [BERT: Pre-training of Deep Bidirectional Transformers](sources/representation/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding.md) — 通过 Masked LM + NSP 实现深度双向预训练，在 11 项 NLP 任务上取得 SOTA
- [Deep contextualized word representations (ELMo)](sources/representation/deep-contextualized-word-representations.md) — 将 biLM 所有层的表示加权组合作为上下文词向量，开创预训练范式
- [BERT Rediscovers the Classical NLP Pipeline](sources/representation/bert-rediscovers-the-classical-nlp-pipeline.md) — 探测 BERT 各层发现其内部按经典 NLP pipeline 顺序编码语言信息

### Inference — 推理优化

- [PagedAttention: Efficient Memory Management for LLM Serving with vLLM](sources/inference/efficient-memory-management-for-large-language-model-serving-with-pagedattention.md) — 将 OS 分页/虚拟内存引入 KV cache 管理，吞吐量 2-4× 提升

### Training — 训练方法

- [Better & Faster LLMs via Multi-token Prediction](sources/training/better-and-faster-large-language-models-via-multi-token-prediction.md) — 用 n 个独立 head 同时预测多 token，提升采样效率 + 推理 3× 自加速
- [Extending Context Window via Positional Interpolation](sources/training/extending-context-window-via-positional-interpolation.md) — 线性缩放 RoPE 位置索引 + 1000 steps 微调将 LLaMA 窗口从 2048 扩展到 32768
