---
title: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
authors: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
venue: MLSys 2024
tags: [quantization, llm, weight-only, w4a16, on-device]
---

# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

### 1. Background & Motivation

前两代量化方案（[[llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale|LLM.int8()]] 和 [[smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models|SmoothQuant]]）主要解决的是 **W8A8 推理加速和显存减半**的问题，但 8-bit 对**端侧部署**仍不够——手机、笔记本等设备需要 4-bit 甚至更低精度才能真正跑起 LLM。W4A16（权重 4-bit、激活 16-bit）是端侧部署的主要方向，但 4-bit 量化精度损失更严重，此前的方法需要大量校准和微调。

关键区别：AWQ 面向的是 **weight-only 4-bit 量化**，不是 W8A8。同时它不需要 backpropagation 或 reconstruction，泛化性更好。

### 2. High-Level Method

核心发现：**并非所有权重都同等重要**。仅保护 ~1% 的 salient weight channels 就能大幅降低量化误差。识别 salient channels 的关键在于看**激活分布**，而不是权重本身的数值大小。

方法：通过**等价变换**（equivalent transformation），将 salient channels 的权重值**放大**，从而在量化时保留更多信息。这个 scale factor 通过离线收集激活统计得到，不需要反向传播或重建。

为什么不用 mixed-precision？混合精度在硬件上效率低。AWQ 通过数学上等价的 scaling 变换来实现"软保护"，所有权重仍用统一精度计算，但 salient channels 因被放大而保留了更多信息。

### 3. Key Implementation Details

- **Salient channel 识别**：收集少量校准样本的激活统计，计算每个 channel 的激活幅度均值。激活幅度大的 channel 对应的权重列为 salient。
- **Scaling 变换**：对 salient channels 施加 per-channel scaling factor s > 1，然后相应地对后续层做反向缩放以保持数学等价性。本质和 [[smoothquant|SmoothQuant]] 的 smoothing 是同一类操作，但方向相反（SmoothQuant 缩小激活，AWQ 放大权重）。
- **搜索最优 scale**：通过极简的网格搜索（仅一个超参数）在校准集上优化 s，搜索空间极小。
- **量化方案**：支持各种 4-bit 数据类型（INT4、NF4、GPTQ 格式等），兼容 group-wise 量化。
- **TinyChat 框架**：配套的端侧推理引擎，支持 kernel fusion 和 platform-aware weight packing，适用于手机等边缘设备。
- **多模态扩展**：首次将 4-bit 量化成功应用到多模态 LLM（LLaVA 等），此前方法在多模态场景下会严重退化。

### 4. Experiments & Results

**语言建模（WikiText-2 perplexity ↓）：**
| 模型 | FP16 | RTN | GPTQ | AWQ |
|------|------|-----|------|-----|
| LLaMA-7B | 12.18 | 12.40 | 12.23 | **12.21** |
| LLaMA-13B | 10.75 | 10.91 | 10.82 | **10.79** |
| LLaMA-30B | 9.19 | 9.32 | 9.26 | **9.22** |
| LLaMA-65B | 8.08 | 8.19 | 8.13 | **8.11** |

- AWQ 在所有规模下一致优于 GPTQ，且接近 FP16 精度
- 比 GPTQ **快 3x**（不需要反向传播和重建）

**端侧推理（TinyChat 框架）：**
- 在 NVIDIA Jetson Orin 上比 FP16 快 **3.1x**
- 在 Apple M1 MacBook 上达到实时 token 生成

**多模态（LLaVA）：**
- 量化到 W4A16 后视觉指令理解精度保持率远优于 GPTQ 和 RTN

### 5. Limitations & Reflection

**作者承认的局限：**
- Weight-only 量化不加速计算密集型操作（只减少显存/带宽），矩阵乘法仍需 FP16 计算
- Scale factor 的搜索依赖少量校准数据，极端域偏移下可能不是最优
- 尚未在更大规模（>175B）模型上验证

**我的判断：**
- 和 [[smoothquant|SmoothQuant]] 师出同门（MIT Song Han Lab），思路也很像——都是用**数学等价的 scaling 变换**来保护重要通道，避免 inefficient mixed-precision
- 核心洞察很微妙："保护哪些权重应看激活分布，而不是权重本身"——这个区分很重要
- 不需要反向传播意味着可快速部署到新领域，这是相比 GPTQ 的实操优势
- 多模态量化是增量贡献，实验规模有限但方向正确

### 6. Personal Takeaways

- 从 LLM.int8() → SmoothQuant → AWQ，可以看到一条清晰的技术演化线：发现问题（outlier）→ 绕开问题（mixed-precision）→ 抹平问题（smoothing）→ **利用问题**（激活感知的 salient weight protection）
- AWQ 的思想也可以理解为：**量化不应该对所有权重一视同仁**
- 与 SmoothQuant 的一体两面：SmoothQuant smooths activations → AWQ scales up salient weights，本质都是通过等价变换重新分配量化误差

---

**Key Concepts:**
- [[quantization|Weight-only quantization (W4A16)]] — 权重 4-bit + 激活 16-bit，主要减少显存占用和访存带宽
- AWQ — 基于激活分布识别重要权重通道并加以保护的量化方法
- Salient weight channels — 对应大激活幅度的权重列，对量化精度影响最大
- [[smoothquant|Equivalent transformation]] — 不改变输出的数学恒等变换（如对角缩放），用于在量化前调整权重分布
- TinyChat — AWQ 配套的端侧推理框架，支持 4-bit LLM/VLM 高效部署
- [[quantization|Group-wise quantization]] — 按通道分组做量化，每组一个缩放因子，平衡精度和开销
