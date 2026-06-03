---
tags: [attention, inference, kv-cache, efficiency, rag]
related:
  - block-attention-for-efficient-prefilling
  - big-bird-transformers-for-longer-sequences
---

# Block-Attention

一种将 RAG 输入序列划分为独立 block 各算各的 KV cache、仅 query block 关注全局的 attention 变体。

## 核心思想

RAG 中检索到的文档彼此**语义独立**，文档之间无需相互 attention——只有 query 需要关注文档。Block-Attention 将这一观察转化为 attention mask 的结构性稀疏：

1. 输入序列分为 k 个 block（每篇文档为一个 block）
2. 各 block 独立计算内部 self-attention 得到 KV 状态
3. 仅最后一个 block（用户 query）关注所有前序 block
4. 当某个文档更新时，只重算该 block 和 query block 的 KV

## 位置重编码

利用 [[rope]] 旋转位置编码的可逆性实现 KV cache 复用：
1. 将缓存 token 的 RoPE **逆时针旋转**复位到 0 位置
2. 再**顺时针旋转**到新上下文中的位置
3. 使缓存 KV 的位置编码适配新上下文

## 效果

| 序列长度 | TTFT 降低 | FLOPs 降低 |
|:-------:|:---------:|:----------:|
| 512 | 48% | 90.1% |
| 4K | 91% | 98.7% |
| 8K | 95% | 99.3% |
| 32K | **98.7%** | **99.8%** |

## Block Fine-tuning

通过少量训练（~200 steps）让 LLM 适应 Block-attention 模式。同一模型可在 block 和 full attention 间无缝切换——通用任务（IFEval, MMLU 等）自动 fallback 到 full attention。

## 与 Sparse Attention 的关系

与 [[big-bird-transformers-for-longer-sequences|Big Bird]] 的稀疏 attention 思路不同：Block-Attention 不是 sparsify attention 本身，而是通过 block-level 的独立性约束消除跨文档 attention。Game AI 场景天然适合此模式（JSON 结构 + >99.5% 内容重复）。

## 局限性

- 需要 block fine-tuning（约 23% 通用 SFT 数据可自然分块）
- Block 划分依赖规则分隔符，对非结构化文本不友好
- 仅在 8B 模型上验证，更大模型效果未知
