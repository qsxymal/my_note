---
title: "Reducing Activation Recomputation in Large Transformer Models"
authors: Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, Bryan Catanzaro
venue: MLSys 2023
tags: [distributed-training, activation-memory, sequence-parallelism, gradient-checkpointing, megatron]
---

# Reducing Activation Recomputation in Large Transformer Models

| 项目 | 内容 |
|------|------|
| **Authors** | Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, Bryan Catanzaro (NVIDIA) |
| **Published** | 2022-05 (MLSys 2023) |
| **Link** | [arXiv 2205.05198](https://arxiv.org/abs/2205.05198) |

**一句话总结:**
- 提出 sequence parallelism 和 selective activation recomputation，将激活内存降低 5×，recomputation 开销从 36% 降至 2%，成为 Megatron-LM 标准组件。

**核心贡献:**
- Sequence parallelism：在非 TP 区域沿序列维度分布激活，利用 ring all-reduce 自然分解实现零额外通信
- Selective activation recomputation：只重算内存密集但计算便宜的 attention 操作（$5as/h$ 项），仅 1.6-2.7% FLOPs 开销
- 1 万亿参数模型 MFU 达 56.3%，端到端训练比 full recomputation 快 ~30%

---

### 1. Background & Motivation

Activation recomputation（gradient checkpointing）是大模型训练的标配——前向不存中间激活，反向时重算。但这引入了 30-40% 的计算开销。

![Figure 1: 参数、优化器状态与激活显存对比——即使模型并行分布了参数和优化器状态，激活显存仍是瓶颈（红色虚线为 A100 80GB 限）](../images/activation/activation_fig1_memory_breakdown.png)

核心问题：**tensor parallelism 和 pipeline parallelism 都有盲区**——tensor parallelism 能并行化注意力/MLP 块内的激活，但 layer-norm 和 dropout 的激活**仍然在各 tensor parallel rank 上复制**；pipeline parallelism 为了压低 pipeline bubble 需要缓存多个 micro-batch 的激活，第一级 pipeline stage 实际存储了 **全部 L 层**的激活。

### 2. High-Level Method

两个简单技术：

**Sequence Parallelism**：在非 tensor parallel 区域（layer-norm, dropout），这些操作沿序列维度互相独立，因此可以沿 sequence 维度 ($s$) 做 partitions 分布激活。

- 引入两个新的共轭算子 `g` 和 `ḡ` 替代原有的 `f`/`ḟ`
- `g`: 前向 all-gather（序列维度拼接），反向 reduce-scatter
- `ḡ`: 前向 reduce-scatter（序列维度分割），反向 all-gather
- 通信量与原有的 tensor parallel all-reduce 完全相同（all-reduce = reduce-scatter + all-gather）
- **零额外通信开销**

![Figure 3: Self-attention 块——红虚线标注的区域为 selective activation recomputation 的重算范围（QK^T, softmax, attention over V），内存大但计算便宜](../images/activation/activation_fig3_self_attention_selective.png)

**Selective Activation Recomputation**：不是 checkpoint 整个 transformer layer，而是只重算 attention 中**内存密集但计算量小**的部分：

- 被重算的部分：$QK^T$ 矩阵乘、softmax、softmax dropout、$AV$
- 这些操作贡献了 $5as/h$ 项的激活内存（GPT-3 中占比 70%）但仅占 **2.7% FLOPs**
- 其余部分（34 项：MLP 和线性 projection）正常存储中间激活

**组合效果（per-layer 激活内存公式）：**

| 方案 | 每层激活内存 |
|------|------------|
| 无并行 | $sbh(34 + 5as/h)$ |
| Tensor parallel | $sbh(10 + 24/t + 5as/ht)$ |
| TP + Sequence Parallel | $sbh(34/t + 5as/ht)$ = 基线/t |
| Full recomputation | $2sbh$ |
| **TP + SP + Selective** | **$34sbh/t$** |

**实现细节：**

![Figure 5: Transformer 层引入 tensor + sequence parallelism——g/ḡ 算子将序列维度的 splitting 整合进已有的 all-reduce 通信中，零额外开销](../images/activation/activation_fig5_tp_with_sequence.png)

- `g` 和 `ḡ` 的实现利用了 ring all-reduce 的自然分解——不需要新的通信原语
- 第一级 pipeline stage 的激活压力最大——论文引入了 **microbatch-level 选择性存储**：只对有限窗口内的 micro-batch 进行选择性重算
- 还引入了 output-tensor-deallocation 优化：forward 后将每个 micro-batch 的 output tensor 释放（下一级已持有数据）

### 3. Experiments & Results

**测试模型配置（见表 3）：**

| 模型 | 层数 | hidden | heads | TP | PP | GPUs |
|------|------|--------|-------|-----|-----|------|
| 22B | 48 | 6144 | 64 | 8 | 1 | 8 |
| 175B (GPT-3) | 96 | 12288 | 96 | 8 | 8 | 64 |
| 530B (MT-NLG) | 105 | 20480 | 128 | 8 | 35 | 280 |
| **1T** | **128** | **25600** | **160** | **8** | **64** | **512** |

**单层执行时间（22B 模型）：**

| 方案 | 前向 | 反向 | 总计 | 开销 |
|------|------|------|------|------|
| Baseline（不重算） | 7.7ms | 11.9ms | 19.6ms | — |
| Full recomputation | 7.7ms | 19.5ms | 27.2ms | **+39%** |
| Selective recompute | 7.7ms | 13.2ms | 20.9ms | +7% |
| **Selective + Sequence** | **7.2ms** | **13.1ms** | **20.3ms** | **+4%** |

**端到端迭代时间对比：**

| 模型 | Full Recompute | Present Work | 提升 |
|------|---------------|-------------|------|
| 22B | 1.42s | 1.10s | **29.0%** |
| 175B | 18.13s | 13.75s | **31.8%** |
| 530B | 49.05s | 37.83s | **29.7%** |
| 1T | 94.42s | 71.49s | **32.1%** |

> 对于 530B 和 1T 模型，recomputation 开销从 36% 降至仅 **2%**。

**MFU 对比：**
- 530B 模型 2240 GPU 上 MFU 达 **54.2%**（full recomputation 为 42.1%），**29% 更快**

![Figure 7: 各技术激活内存占比——sequence parallelism + selective recompute 将内存降至 baseline 的 20% 以下（~5× 降低）](../images/activation/activation_fig7_memory_reduction.png)

**内存优化实际效果：**
- 激活内存 **降低 5×**（baseline 的 20% 以下）
- 结合 TP 和 pipeline parallel 后，内存节省使原来需要 recomputation 的配置可以完全不需要重算

![Figure 8: Per layer 前向/反向/重算时间对比——随着模型增大，selective recompute 的开销从 7% 降至仅 2%](../images/activation/activation_fig8_perf_breakdown.png)

### 4. Limitations & Reflection

**局限性：**

- Sequence parallelism 仅适用于 Transformer（依赖序列维度独立的操作），不适用于 CNN 等架构
- First pipeline stage 的激活内存压力是 pipeline parallelism 的根本性矛盾——虽然 microbatch-level 选择性存储有缓解，但未根本解决
- 没有与 ZeRO/FSDP 等数据并行中的 state sharding 方案做对比分析
- Selective recomputation 的 FLOPs 节省量与 $5as/h$ 相关——模型越大效果越好，小模型收益递减

**我的判断：**

- Sequence parallelism 和 selective recomputation 是这篇论文最有价值的两个 ideas——它们后来成为了 Megatron-LM/NeMo Megatron 的标准组件
- 从公式角度看，将 $34 + 5as/h$ 分解为（存储 $34/t$，重算 $5as/ht$）是极简但有洞察力的分析——**不是每条激活都值得存储**，有些算起来很便宜
- 与 [[efficient-large-scale-language-model-training-on-gpu-clusters-using-megatron-lm|PTD-P（3D 并行）]] 形成完整的大规模训练技术栈
- Selective recomputation 的哲学在后续 FlashAttention 等工作中也有体现——不是所有中间结果都值得存储

### 5. Personal Takeaways

- Activation memory 公式推导（$34+5as/h$）是论文最有价值的部分——这个简洁的公式把一个复杂的内存问题分解为两个有明确物理意义的项
- Sequence parallelism 的巧妙在于利用 ring all-reduce 的自然分解把通信隐藏在已有的通信模式中——设计新原语的思路是 "reuse, not add"
- "不是每条激活都值得存储"的哲学很通用——对于现代框架，了解每层的 compute-to-memory 比率可以指导更精细的 rematerialization 策略
- 整体的 Megatron-LM 技术路线：TP → PP → PTD-P → **Seq Parallel + Selective Recompute**，每一步都是对前一步"盲区"的精确补充

---

**Key Concepts:**
- [[tensor-parallelism|Tensor parallelism]] — 层内按矩阵维度拆分
- Sequence parallelism — 在非 TP 区域沿 sequence 维度分布激活，零额外通信
- Selective activation recomputation — 只重算内存密集但计算便宜的操作
- [[gradient-checkpointing|Activation recomputation (gradient checkpointing)]] — 前向丢激活、反向重算
- g / ḡ operators — sequence ↔ tensor parallel 之间的转换算子

**Extracted Figures:**
- `activation_fig1_memory_breakdown.png` — 参数、优化器状态、激活内存分解（激活是 real bottleneck）
- `activation_fig3_self_attention_selective.png` — Self-attention 块中 selective recomputation 的重算范围
- `activation_fig5_tp_with_sequence.png` — Tensor + Sequence Parallelism 结合的 Transformer 层
- `activation_fig7_memory_reduction.png` — 各技术激活内存占比（5× 降低）
- `activation_fig8_perf_breakdown.png` — 各规模模型的前向/反向/重算时间分解
