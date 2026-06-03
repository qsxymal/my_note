# Wiki Log

## [2026-06-03] ingest | Better & Faster LLMs via Multi-token Prediction

- 添加 source 页面: `sources/better-and-faster-large-language-models-via-multi-token-prediction.md`
- 提取关键图表至 `images/multi-token-prediction/`
- 更新 `index.md`, `images/index.md`

## [2026-06-03] ingest | BERT Rediscovers the Classical NLP Pipeline

- 添加 source 页面: `sources/bert-rediscovers-the-classical-nlp-pipeline.md`
- 提取关键图表至 `images/bert-pipeline/`
- 更新 `index.md`, `images/index.md`

## [2026-06-03] ingest | BERT: Pre-training of Deep Bidirectional Transformers

- 添加 source 页面: `sources/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding.md`
- 提取关键图表至 `images/bert/`
- 更新 `index.md`, `images/index.md`

## [2026-06-03] ingest | Attention Is All You Need (Transformer)

- 添加 source 页面: `sources/attention-is-all-you-need.md`
- 提取关键图表至 `images/transformer/`
- 更新 `index.md`, `images/index.md`

## [2026-06-03] ingest | RoFormer: Rotary Position Embedding (RoPE)

- 添加 source 页面: `sources/roformer-enhanced-transformer-with-rotary-position-embedding.md`
- 提取关键图表至 `images/rope/`
- 更新 `index.md`, `images/index.md`

## [2026-06-03] ingest | DeepNet: Scaling Transformers to 1,000 Layers

- 添加 source 页面: `sources/deepnet-scaling-transformers-to-1000-layers.md`
- 提取关键图表至 `images/deepnet/`
- 更新 `index.md`, `images/index.md`

## [2026-06-03] ingest | Deep contextualized word representations (ELMo)

- 添加 source 页面: `sources/deep-contextualized-word-representations.md`
- 提取关键图表至 `images/elmo/`
- 更新 `index.md`, `images/index.md`

## [2026-06-03] ingest | Block-Attention for Efficient Prefilling

- 添加 source 页面: `sources/block-attention-for-efficient-prefilling.md`
- 提取关键图表至 `images/block-attention/`
- 更新 `index.md`, `images/index.md`

## [2026-06-03] ingest | Big Bird: Transformers for Longer Sequences

- 添加 source 页面: `sources/big-bird-transformers-for-longer-sequences.md`
- 提取关键图表至 `images/big-bird/`
- 更新 `index.md`, `images/index.md`

## [2026-06-03] ingest | PagedAttention: Efficient Memory Management for LLM Serving with vLLM

- 添加 source 页面: `sources/efficient-memory-management-for-large-language-model-serving-with-pagedattention.md`
- 提取关键图表至 `images/pagedattention/`
- 更新 `index.md`, `images/index.md`

## [2026-06-03] concepts | Add concept pages for new papers

- 添加 concept 页面: `concepts/rope.md` — Rotary Position Embedding
- 添加 concept 页面: `concepts/deepnorm.md` — DeepNorm normalization
- 添加 concept 页面: `concepts/block-attention.md` — Block-Attention for RAG
- 添加 concept 页面: `concepts/sparse-attention.md` — Sparse attention patterns (Big Bird)
- 更新 `index.md`

## [2026-06-03] ingest | Positional Interpolation for Context Extension

- 添加 source 页面: `sources/extending-context-window-via-positional-interpolation.md`
- 添加 concept 页面: `concepts/positional-interpolation.md`
- 提取关键图表至 `images/positional-interpolation/`
- 添加 wikilinks 至 [[RoPE]] 和 [[positional-interpolation]]
- 更新 `index.md`, `images/index.md`

## [2026-06-03] concepts | Add missing concept pages

- 添加 concept 页面: `concepts/multi-token-prediction.md` — Multi-Token Prediction
- 添加 concept 页面: `concepts/pagedattention.md` — PagedAttention
- 添加 concept 页面: `concepts/elmo.md` — ELMo
- 添加 concept 页面: `concepts/big-bird.md` — Big Bird
- 添加 concept 页面: `concepts/2d-parallelism.md` — 2D Parallelism (Optimus)
- 添加 concept 页面: `concepts/transformer.md` — Transformer 架构
- 添加 concept 页面: `concepts/bert.md` — BERT
- 更新 `index.md`

## [2026-06-03] enhancement | arXiv source figure extraction

- 新增 `bin/arxiv-figure-dl.py` — 从 arXiv e-print 源码包提取原始图表
- 已为 18 篇论文下载源码并替换高分辨率图（600 DPI 矢量渲染）
- 更新 `CLAUDE.md`（figure extraction priority）
- 图片索引全覆盖 119 张图，无缺失引用
- 主要升级：SmoothQuant（~28MB 高质量图）、PagedAttention、BERT、DeepNet、Big Bird 等
- Transformer 架构图回退到 600 DPI 直裁（源码 ModalNet-19 分辨率不足）


## [2026-06-02] ingest | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

- 添加 source 页面: `sources/llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale.md`
- 更新 `index.md`

## [2026-06-02] ingest | SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs

- 添加 source 页面: `sources/smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models.md`
- 更新 `index.md`

## [2026-06-02] re-analysis | LLM.int8() + SmoothQuant — structured format

- 重构 source 页面（结构化 6 步分析格式）
- 修复 index.md 重复条目
- 添加跨文档 `[[page-name]]` 链接

## [2026-06-02] ingest | AWQ: Activation-aware Weight Quantization

- 添加 source 页面: `sources/awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration.md`
- 添加 concept 页面: `concepts/awq.md`
- 更新 `index.md`

## [2026-06-02] ingest | GPTQ: Accurate Post-Training Quantization

- 添加 source 页面: `sources/gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers.md`
- 添加 concept 页面: `concepts/gptq.md`
- 更新 `index.md`

## [2026-06-02] re-analysis | All papers — figures + rewrite

- 重写 3 篇 source 页面（嵌入图片，重新组织叙事流）
- 用 PyMuPDF4LLM 提取各论文关键图表
- 新建 `wiki/images/` 目录及 per-paper 子目录
- 新建 `wiki/images/index.md` 统一图片索引
- 更新 `CLAUDE.md`（images 目录和命名规范）
- 更新 `paper-analysis` 技能（figure extraction + markdown 嵌入流程）

## [2026-06-02] ingest | GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism

- 添加 source 页面: `sources/gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism.md`
- 添加 concept 页面: `concepts/pipeline-parallelism.md`, `concepts/gradient-checkpointing.md`
- 提取关键图表至 `images/gpipe/`
- 更新 `index.md`, `images/index.md`

## [2026-06-02] ingest | Optimus: An Efficient 2D Method for Training Super-Large Deep Learning Models

- 添加 source 页面: `sources/an-efficient-2d-method-for-training-super-large-deep-learning-models.md`
- 更新 `index.md`

## [2026-06-02] ingest | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

- 添加 source 页面: `sources/efficient-large-scale-language-model-training-on-gpu-clusters-using-megatron-lm.md`
- 提出 PTD-P（Pipeline + Tensor + Data Parallelism）方案和 interleaved pipeline schedule
- 3072 A100 GPU 上 1 万亿参数模型达 502 petaFLOP/s、52% 峰值利用率
- 更新 `index.md`

## [2026-06-02] ingest | Reducing Activation Recomputation in Large Transformer Models

- 添加 source 页面: `sources/reducing-activation-recomputation-in-large-transformer-models.md`
- 提出 sequence parallelism + selective activation recomputation 两个技术
- 激活内存降低 5×，recomputation 开销从 30-40% 降至 2-4%
- 更新 `index.md`
