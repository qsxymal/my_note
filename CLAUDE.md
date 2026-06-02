# LLM Wiki Schema

## Directory Structure

- `raw/` — immutable source documents (articles, papers, notes)
- `wiki/` — LLM-maintained markdown files
  - `wiki/entities/` — entity/人物/事物页面
  - `wiki/concepts/` — 概念/术语页面
  - `wiki/sources/` — 来源摘要页面
  - `wiki/index.md` — 内容目录
  - `wiki/log.md` — 操作日志

## Conventions

- All wiki pages use Markdown with YAML frontmatter
- Cross-reference using `[[page-name]]` links
- Update `index.md` and `log.md` on every ingest
