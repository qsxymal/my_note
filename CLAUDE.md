# LLM Wiki Schema

## Directory Structure

- `raw/` — immutable source documents (articles, papers, notes), categorized:
  - `raw/architecture/` — 模型架构 (Transformer, RoPE, DeepNet, Big Bird, Block-Attention, etc.)
  - `raw/parallelism/` — 并行训练 (GPipe, Megatron-LM, Optimus, etc.)
  - `raw/quantization/` — 模型量化 (LLM.int8, SmoothQuant, AWQ, GPTQ, KVQuant)
  - `raw/representation/` — 表示学习 (BERT, ELMo, etc.)
  - `raw/inference/` — 推理优化 (PagedAttention, FlashInfer)
  - `raw/training/` — 训练方法 (Multi-token Prediction, Positional Interpolation)
- `wiki/` — LLM-maintained markdown files
  - `wiki/images/` — extracted figures, organized by paper
  - `wiki/entities/` — entity/人物/事物页面
  - `wiki/concepts/` — 概念/术语页面
  - `wiki/sources/` — 来源摘要页面, categorized same as `raw/`:
    - `wiki/sources/architecture/`
    - `wiki/sources/parallelism/`
    - `wiki/sources/quantization/`
    - `wiki/sources/representation/`
    - `wiki/sources/inference/`
    - `wiki/sources/training/`
  - `wiki/index.md` — 内容目录
  - `wiki/log.md` — 操作日志

## Conventions

- All wiki pages use Markdown with YAML frontmatter
- Cross-reference using `[[page-name]]` links
- On every ingest: update `index.md`, `log.md`, and `images/index.md`
- Extracted paper figures go in `wiki/images/[paper-short-name]/`
- Naming: `[abbr]_[figN]_[keyword].png` (e.g., `smoothquant_fig2_migration_intuition.png`)
- `wiki/images/index.md` maintains the centralized image index with description table
- Embed figures in source pages with standard Markdown: `![描述](../images/paper-name/filename.png)`
- Place images after the paragraph that introduces them, not before
- **Figure extraction priority**: ① `bin/arxiv-figure-dl.py <arxiv_id> --outdir wiki/images/<paper>/` (arXiv e-print source, highest quality), ② PyMuPDF4LLM, ③ PDF page rendering + clipping
