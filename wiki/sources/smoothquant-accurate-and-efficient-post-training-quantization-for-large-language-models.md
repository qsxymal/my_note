---
title: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
authors: Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
venue: ICML 2023
tags: [quantization, llm, efficiency, inference, w8a8]
---

# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

### 1. Background & Motivation

[[llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale|LLM.int8()]] 已经证明了大模型可以用 8-bit 推理且精度无损，但它的 mixed-precision decomposition 在 GPU 上实现效率低——每次矩阵乘法都需要分解、分别计算、再拼接，无法充分利用 Tensor Core 的纯 INT8 GEMM 内核。业界需要一种 **纯 W8A8**（权重和激活全部 INT8）的方案，才能真正提升推理吞吐。

核心问题：激活比权重**难量化得多**——outlier 导致 per-tensor 量化时有效 bit 数极低（非 outlier channel 仅 2-3 个有效 quantization level）。而 per-channel 量化虽然精度高，但无法映射到高效的 INT8 GEMM 内核（缩放只能在外维度进行）。

与 LLM.int8() 的关键区别：SmoothQuant 不依赖 mixed-precision，而是通过数学等价变换让**所有值都适合纯 INT8 计算**。

### 2. High-Level Method

核心洞察：权重的分布平坦均匀、易于量化；激活有 outlier、难以量化。但两者之间存在一个**数学等价变换**的空间——通过 per-channel smoothing factor s 将激活中 outlier 的量化难度**迁移**到权重上，使得变换后两者都易于量化。

具体而言，对线性层 Y = X·W，引入平滑变换：
- Ŷ = (X·diag(s)^{-1})·(diag(s)·W) = X̂·Ŵ
- X̂ 的 outlier 被平滑，Ŵ 依然保持平坦

**迁移强度 α** 控制难度分配比例：s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
- α=0 → 全部难度留在激活（精度差）
- α=1 → 全部难度迁移到权重（权重量化误差大）
- **α=0.5** → OPT/BLOOM 最优平衡点
- α=0.75 → GLM-130B（激活 outlier 更严重）
- α=0.8 → LLaMA 系列

### 3. Key Implementation Details

- **三档效率级别**：
  - O1: per-token activation dynamic + per-channel weight static（最高精度）
  - O2: per-tensor activation dynamic + per-channel weight static
  - O3: per-tensor activation static + per-channel weight static（最高效率）
- **校准**：从预训练数据中取 512 条样本，离线计算 smoothing factor 和量化步长
- **融合到前一层参数中**：smoothing factor s 可融合到前一层（LayerNorm 或前一个 Linear）的参数中，推理时零额外开销
- **覆盖范围**：量化所有 Linear 层和 Attention 中的 BMM，保留 Softmax、LayerNorm 等轻量操作为 FP16
- 后端实现支持 PyTorch HuggingFace 和 NVIDIA FasterTransformer，使用 CUTLASS INT8 GEMM 内核

### 4. Experiments & Results

**OPT-175B zero-shot 平均精度：**
| 方法 | 平均精度 | WikiText ↓ |
|------|---------|-----------|
| FP16 | 66.9% | 10.99 |
| Naive W8A8 | 35.5% | 93080 |
| ZeroQuant | 35.8% | 84648 |
| LLM.int8() | 66.7% | 11.10 |
| **SmoothQuant-O3** | **66.8%** | **11.17** |

**多模型验证：**
- OPT-175B: O1/O2/O3 全部匹配 FP16 精度
- BLOOM-176B: 自然更容易量化，SmoothQuant O1/O2 无损
- GLM-130B: 最难量化，SmoothQuant O1 无损，O3 仅降 1%
- LLaMA-7B→65B: 几乎所有模型 W8A8 量化后 perplexity 偏差 < 0.05

**加速效果：**
- PyTorch: 最高 1.51x 加速，1.96x 内存节省
- FasterTransformer: 最高 1.56x 加速，内存减半
- 首次实现 **530B 模型单节点（8-GPU）推理**

### 5. Limitations & Reflection

**作者承认的局限：**
- O3（per-tensor static）在分布偏移大的场景下精度可能下降
- α 需要通过 grid search 选择（虽然只需做一次），不同模型族需要不同 α
- 校准集需要来自预训练分布，可能导致 domain shift 下的泛化问题

**我的判断：**
- 这是对 LLM.int8() 非常漂亮的改进——用简单的数学变换绕过了 hardware 不友好的 mixed-precision 分解
- Smoothing factor 融合到前一层参数是工程上的精巧设计，推理时零额外开销
- 但迁移强度 α 需要手动调这一点不够优雅——后续 AWQ 等工作的改进方向
- 实验覆盖非常全面（OPT/BLOOM/GLM/LLaMA/Falcon/Mistral/Mixtral），泛化性有说服力

### 6. Personal Takeaways

- **"迁移量化难度"** 这个思路非常 elegant——不增加计算量、不改模型结构、纯数学变换
- 与 [[llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale|LLM.int8()]] 对比: LLM.int8() 做了 outlier 分析发现自己打不过 outlier，于是搞 mixed-precision 绕开；SmoothQuant 选择把 outlier 抹平，让所有计算都落在 INT8 里。两种哲学
- 三档效率级别（O1/O2/O3）的设计很好——用户可以根据硬件和精度需求灵活选择
- 后续 AWQ 和 GPTQ 都在此基础上改进，形成了 LLM 量化的一条清晰发展线

---

**Key Concepts:**
- [[quantization|W8A8]] — 权重和激活均为 8-bit 的量化方案
- [[smoothquant|Quantization difficulty migration]] — 通过数学变换将激活的量化难度转移到权重
- [[smoothquant|Smoothing factor]] — per-channel 缩放因子 s，控制量化难度在激活和权重间的分配
- [[quantization|Per-token vs per-tensor vs per-channel quantization]] — 量化粒度，越细精度越高但硬件效率越低
- [[quantization|Static vs dynamic quantization]] — 静态量化（校准集预计算步长）vs 动态量化（推理时实时计算）
