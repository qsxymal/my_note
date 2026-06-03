---
tags: [sparse-attention, long-context, transformer]
related:
  - big-bird-transformers-for-longer-sequences
  - sparse-attention
---

# Big Bird

将 self-attention 视为图稀疏化问题，提出 random + window + global 三种稀疏注意力组件组合，以 O(n) 复杂度逼近 full attention 的 O(n²) 表达能力。

## 三种 Attention 组件

| 组件 | 作用 | 理论依据 |
|:---|:---|:--------|
| **Random** | 每个 query 关注 r 个随机 key，信息快速传播 | 随机图 O(log n) 路径长度 |
| **Window** | 每个 query 关注宽度 w 的局部邻居 | NLP 的 locality of reference |
| **Global** | 少量 token 关注整个序列，也被全序列关注 | 保证通用逼近能力（Theorem 1） |

三者缺一不可：Random-only 或 Window-only 都远低于 full attention 表现。

## 复杂度

O(n(g+w+r)bd) = O(n)，g/w/r/b 为常数。通过 **blockification**（将序列分块，稀疏运算转为密集块运算）实现高效 GPU 执行。

## 效果

- QA：Natural Questions、TriviaQA、WikiHop 上 SOTA（2020）
- 长文档摘要：Arxiv R-1 41.22，BigPatent +8.39 分
- 基因组学：DNA MLM BPC 1.12，启动子预测 99.9% F1
