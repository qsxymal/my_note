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
