# Wiki Log

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
