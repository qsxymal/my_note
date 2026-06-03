---
title: "Big Bird: Transformers for Longer Sequences"
authors: Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed
venue: NeurIPS 2020
tags: [sparse-attention, long-context, transformer, nlp, genomics]
---

# Big Bird: Transformers for Longer Sequences

| 项目 | 内容 |
|------|------|
| **Authors** | Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed (Google Research) |
| **Published** | 2020-07 (NeurIPS 2020) |
| **Link** | [arXiv 2007.14062](https://arxiv.org/abs/2007.14062) |

**一句话总结:**
- 将 self-attention 视为图稀疏化问题，提出 random + window + global 三种 [[sparse-attention]] 组件组合，以 O(n) 复杂度逼近 full attention 的 O(n²) 表达能力，支持 8× 更长序列（4096+ tokens）。

**核心贡献:**
- **稀疏注意力机制**：将 full attention 的二次复杂度降低到线性，结合随机注意力 + 滑动窗口注意力 + 全局注意力
- **理论保证**：证明 BIGBIRD 是通用函数逼近器（universal approximator）和 Turing complete，保留 full attention 的表达能力
- **高效 GPU 实现**：通过 blockification 将稀疏操作转换为密集张量运算，避免 GPU 不擅长的稀疏矩阵乘法
- **NLP SOTA**：在 Natural Questions (LA)、TriviaQA、WikiHop 上实现 SOTA；长文档摘要 Arxiv +5%、BigPatent 大幅提升
- **基因组学新应用**：首次将 Transformer 预训练（MLM + NSP）应用于 DNA 序列，BPC 1.12，启动子预测 99.9% F1

---

### 1. Background & Motivation

Transformer 的核心限制是 self-attention 的 O(n²) 计算和内存复杂度，这限制了输入序列长度（BERT 为 512 tokens）。

已有稀疏化方法（Longformer 的 sliding window、Reformer 的 LSH）多基于启发式设计，**缺乏理论保证**且在不同任务上的通用性不足。

本文的核心洞察：从**图稀疏化**视角重新理解 self-attention——full attention 等价于完全图，稀疏注意力就是图稀疏化问题。好的稀疏图应满足两个性质：
- **短平均路径长度**（信息快速传递 → 随机图）
- **高聚类系数**（局部信息密集 → 滑动窗口）

### 2. High-Level Method

![Figure 1: Attention building blocks](../images/big-bird/bigbird_fig1_building_blocks.png)

**三种 attention 组件：**

1. **Random attention**（随机）：每个 query 随机关注 r 个 key。基于随机图的谱性质——只需 Θ̃(n) 条边就能使任意两节点间路径长度为 O(log n)，信息快速传播。
2. **Window attention**（滑动窗口）：每个 query 关注宽度 w 的局部邻居。利用 NLP 数据的 locality of reference（Clark et al. 发现相邻 token 的 attention 权重最高）。
3. **Global attention**（全局）：少量全局 token 关注整个序列，也被整个序列关注。理论分析（Theorem 1）证明这是保证通用逼近能力的关键。

**两种全局 token 配置：**
- **BIGBIRD-ITC**（Internal Transformer Construction）：从现有 token 中选择 g 个作为全局 token
- **BIGBIRD-ETC**（Extended Transformer Construction）：新增额外全局 token（类似 CLS），效果更好

最终 BIGBIRD 的注意力矩阵 A 满足：每行有 r 个随机位置、w 个窗口位置、以及来自 g 个全局 token 的全连接。

### 3. Key Implementation Details

**Blockification（块化）**：GPU 不擅长稀疏矩阵乘法，BIGBIRD 将序列分为大小为 b 的块，在块级别定义稀疏模式，将稀疏运算转化为密集张量乘法和 roll/gather 操作。

核心运算流程：
1. 对 Q 和 K 做 blockify 得到 Q′, K′（尺寸 ⌈n/b⌉ × b × d）
2. Window 部分：复制 w 份 K′，roll 后与 Q′ 密集相乘
3. Global 部分：取前 g 个 block 直接计算
4. Random 部分（r 很小，通常 r=3）：用 gather 操作
5. 拼接 K"（尺寸 ⌈n/b⌉ × (g+w+r)b × d），与 Q′ 密集相乘

**复杂度**：O(n(g+w+r)bd) = O(n)，其中 g, w, r, b 均为常数。

**参数量控制**：预训练从 RoBERTa checkpoint warm-start，保持与 RoBERTa-base/large 相同的参数量和架构。

### 4. Experiments & Results

**消融实验——三种组件缺一不可：**

![Table 1: Building block comparison](../images/big-bird/bigbird_table1_ablations.png)

Random-only（60.1 MLM）或 Window-only（58.3）都远低于 BERT-base（64.2），R+W 组合（62.7）仍差 1.5 分，仅三者组合（BIGBIRD）才能追上 full attention。

**QA——利用长上下文：**

![Table 2: QA Dev results](../images/big-bird/bigbird_table2_qa_dev.png)

BIGBIRD-ETC 在所有四个 QA 数据集上优于 Longformer 和 RoBERTa（后者只能处理 512 tokens）。

![Table 3: QA Test results](../images/big-bird/bigbird_table3_qa_test.png)

在 Natural Questions (LA)、TriviaQA、WikiHop 上 **SOTA**（2020 年水平），与 top-3 leaderboard 条目对比。

**长文档摘要：**

![Table 4: Summarization results part 1](../images/big-bird/bigbird_table4_summarization_p1.png)
![Table 4: Summarization results part 2](../images/big-bird/bigbird_table4_summarization_p2.png)

Base 级：BIGBIRD-RoBERTa 在 Arxiv（R-1 41.22）、PubMed（43.70）、BigPatent（55.69）上显著超过 Pegasus。
Large 级：BIGBIRD-Pegasus 在 BigPatent 上 R-1 60.64（vs Pegasus 52.25），+8.39 分。

**分类——长文档优势明显：**

| 数据集 | RoBERTa | BIGBIRD | SoTA | 超 512 比例 |
|--------|---------|---------|------|-----------|
| Arxiv | 87.42 | **92.31** | 87.96 | 100% |
| Hyperpartisan | 87.8 | **92.2** | 90.6 | 53% |
| Patents | 67.07 | **69.30** | 69.01 | 90% |

Arxiv 上比 SoTA 高 5 分，超 512 比例越高的任务收益越大。短文本（IMDb 仅 14% 超 512）无显著提升。

**基因组学——全新领域的探索：**

DNA 序列 MLM 预训练（BPE tokenization，平均 8.78 bp/token）：

![Table 5: MLM BPC for DNA](../images/big-bird/bigbird_table5_mlm_bpc.png)

BIGBIRD（seqlen 4096）取得 1.12 BPC，优于 BERT（seqlen 512）的 1.23。

![Table 7: Chromatin-Profile Prediction](../images/big-bird/bigbird_table7_chromatin_profile.png)

启动子预测 **99.9% F1**（+4.3% over previous best）；染色质谱预测 TF/HM/DHS 均达到或超过 DeepSea。

### 5. Theoretical Guarantees

**通用逼近性（Universal Approximation, Theorem 1）：** 任何包含 star graph S（即至少有一个全局 token 连接到所有其他 token）的稀疏图 D，对应的 Transformer 编码器是序列函数的通用逼近器。证明分三步：连续函数 → 分段常数函数 → modified Transformer → 原始 Transformer。

**Turing Completeness：** 稀疏编码器-解码器 Transformer 仍可模拟任意图灵机，核心是用稀疏 attention 替代 Pérez et al. 的 full attention 寻址方案。

**下界（Limitation）：** 存在自然任务（找最远向量），full attention 只需 1 层，而任何 Õ(n) 边的稀疏 attention 需要 Ω̃(n^{1-o(1)}) 层。基于 Orthogonal Vector Conjecture（OVC）的细粒度复杂度下界。

### 6. Limitations & Reflection

**局限：**
- 块大小 b、窗口 w、随机 r、全局 g 为超参数，需针对硬件调整（本文默认 b=64, g=2b, w=3b, r=3b）
- GLUE 短文本任务上与 RoBERTa 持平而非领先，稀疏 attention 在短序列上的优势无法体现
- 随机注意力采样的不可重复性可能影响调试和可解释性
- GPU 上仍需 gather 操作处理随机部分，不是完全密集计算

**思考：**
- **图稀疏化视角**是本文最优雅的洞察——将 attention 优化问题与图论经典问题建立联系，为后续稀疏 attention 提供理论基础
- 理论部分（通用逼近、Turing complete）虽然不直接指导工程实践，但证明了"稀疏 attention 不会丢失表达能力"，这对实际应用很关键
- 基因组学应用令人印象深刻——用 NLP 方法处理 DNA 序列，BPE tokenization + MLM pretraining + finetuning 的完整 pipeline
- Longformer 同年提出类似方案，但 BIGBIRD 多了随机注意力和理论保证，且验证了更多任务

**Key Concepts:**
- **[[big-bird|Big Bird]]** — Random + Window + Global 三种稀疏 attention 组件组合，O(n) 复杂度逼近 full attention
- **[[sparse-attention|Sparse Attention]]** — 通过稀疏注意力矩阵将 self-attention 从 O(n²) 降至 O(n) 的技术集合

---

**Extracted Figures:**
- `bigbird_fig1_building_blocks.png` — 四种 attention pattern 对比：random、window、global、BIGBIRD 组合
- `bigbird_table1_ablations.png` — 各 building block 在 MLM/SQuAD/MNLI 上的贡献消融
- `bigbird_table2_qa_dev.png` — QA Dev 结果（Base 模型，4 数据集）
- `bigbird_table3_qa_test.png` — QA Test 结果 vs Leaderboard top-3
- `bigbird_table4_summarization_p1.png` — 长文档摘要 ROUGE 分（Base + Large 对比）
- `bigbird_table4_summarization_p2.png` — 长文档摘要 ROUGE 分（续表）
- `bigbird_table5_mlm_bpc.png` — DNA MLM 预训练 BPC
- `bigbird_table7_chromatin_profile.png` — 染色质谱预测结果
