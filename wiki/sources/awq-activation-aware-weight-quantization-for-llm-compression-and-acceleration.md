---
title: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
authors: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
venue: MLSys 2024
tags: [quantization, llm, weight-only, w4a16, on-device]
---

# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

### 1. Background & Motivation

前两代量化方案（[[llm.int8-8-bit-matrix-multiplication-for-transformers-at-scale|LLM.int8()]] 和 [[smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models|SmoothQuant]]）主要解决 **W8A8 推理加速**，但 8-bit 对**端侧部署**仍不够——手机、笔记本等设备需要 4-bit 甚至更低精度才能真正跑起 LLM。

作者对端侧 LLM 推理做了瓶颈分析（Figure 3）：生成阶段（generation stage）受限于显存带宽（arithmetic intensity ≈ 1），而权重的访存量比激活大 1-2 个数量级。因此，**weight-only 4-bit 量化**（W4A16）对端侧部署更为关键——它直接减少权重访存 4 倍，将 arithmetic intensity 提升到 4。

![Figure 3: Llama-2-7B 在 RTX 4090 上的瓶颈分析——生成阶段 memory-bound，权重访存占主导，W4A16 可将 arithmetic intensity 提升 4×](../images/awq/awq_fig3_bottleneck.png)

关键区分：AWQ 是 **weight-only 4-bit 量化**（不是 W8A8）。它不需要 backpropagation 或重建，泛化性更好。

### 2. High-Level Method

**核心发现：** 并非所有权重都同等重要。仅保护 ~1% 的 salient weight channels 就能大幅降低量化误差。识别 salient channels 的关键在于看**激活分布**，而不是权重本身的数值大小（Table 1）：基于激活分布选择 1% 的 FP16 权重可将 OPT-6.7B 的 perplexity 从 23.54 降到 11.39，而基于权重分布或随机选择则几乎没有效果。

![Figure 2: 基于激活分布识别 ~1% salient weights——mixed-precision 硬件效率低，AWQ 通过 scaling 变换实现等效保护](../images/awq/awq_fig2_salient_weights.png)

为什么不用 mixed-precision（Figure 2 中间子图）？混合精度在硬件上效率低——salient 和 non-salient 权重需要不同的量化路径。AWQ 通过数学等价的 scaling 变换来实现"软保护"（Figure 2 右子图）：将 salient channels 的权重值放大 s > 1，使它们在统一精度下保留更多信息，同时对激活做反向缩放以保持数学等价性。

本质上，这和 [[smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models|SmoothQuant]] 的 smoothing 是同一类操作，但方向相反——SmoothQuant 缩小激活的 outlier，AWQ 放大重要权重。

### 3. Key Implementation Details

- **Salient channel 识别**：收集少量校准样本的激活统计，计算每个 channel 的激活幅度均值。激活幅度大的 channel 对应的权重列为 salient。
- **最优 scale 搜索**：通过极简的网格搜索（仅一个超参数 $\alpha$）在校准集上优化 $s$，搜索空间极小（grid size = 20），无需反向传播：

$$s^* = \arg\min_s \| Q(W \cdot \text{diag}(s))(\text{diag}(s)^{-1} \cdot X) - WX \|$$

- **量化方案**：支持 INT4/INT3 group-wise 量化（group size = 128），兼容多种 4-bit 数据类型
- **TinyChat 框架**：配套端侧推理引擎，支持 kernel fusion 和 SIMD-aware weight packing（Figure 4），适用于手机等边缘设备

![Figure 4: SIMD-aware weight packing——通过重排序和位运算实现 ARM NEON 上 4-bit 权重的高效解包](../images/awq/awq_fig4_weight_packing.png)

- **多模态扩展**：首次将 4-bit 量化成功应用到多模态 LLM（LLaVA, OpenFlamingo），此前方法在多模态场景下会严重退化（Table 6, 7）

### 4. Experiments & Results

**语言建模（WikiText-2 perplexity ↓）：**

![Table 4: AWQ 在所有规模下一致优于 GPTQ，INT3-g128 下优势更明显](../images/awq/awq_table4_results.png)

| 模型 | FP16 | RTN | GPTQ | AWQ |
|------|------|-----|------|-----|
| LLaMA-7B | 5.68 | 5.96 | 6.22 | **5.78** |
| LLaMA-13B | 5.09 | 5.25 | 5.23 | **5.19** |
| LLaMA-30B | 4.10 | 4.23 | 4.24 | **4.21** |
| LLaMA-65B | 3.53 | 3.67 | 3.66 | **3.62** |

**校准集鲁棒性（Figure 8）：** AWQ 相比 GPTQ 有两个关键优势：
1. 需要更少的校准数据（10× 更少即可达到同等精度）
2. 对校准集分布偏移更鲁棒（跨域偏移时 AWQ 仅退化 0.5-0.6 PPL，GPTQ 退化 2.3-4.9 PPL）

![Figure 8: AWQ 需要更少的校准数据（左），且对校准集分布偏移更鲁棒（右）——这是相比 GPTQ 的实操优势](../images/awq/awq_fig8_calibration_robustness.png)

**端侧推理（TinyChat）：**
- NVIDIA RTX 4090: 2.7-3.9× 加速（vs Huggingface FP16）
- NVIDIA Jetson Orin: 3.5× 加速
- Raspberry Pi 4: 成功部署 7B 模型

![Figure 10: TinyChat 在 Jetson Orin 和 Raspberry Pi 上的端到端延迟——相比 AutoGPTQ、llama.cpp 等框架有 1.2-3.0× 加速](../images/awq/awq_fig10_latency.png)

### 5. Limitations & Reflection

**作者承认的局限：**
- Weight-only 量化不加速计算密集型操作（只减少显存/带宽），矩阵乘法仍需 FP16 计算
- Scale factor 的搜索依赖少量校准数据，极端域偏移下可能不是最优
- 尚未在更大规模（>175B）模型上验证

**我的判断：**
- 和 [[smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models|SmoothQuant]] 师出同门（MIT Song Han Lab），技术路线一脉相承——都是用**数学等价的 scaling 变换**来保护重要通道，避免 inefficient mixed-precision
- 核心洞察很微妙："保护哪些权重应看激活分布，而不是权重本身"——这个区分很重要
- 不需要反向传播意味着可快速部署到新领域，这是相比 [[gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers|GPTQ]] 的实操优势
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

**Extracted Figures:**
- `awq_fig3_bottleneck.png` — 端侧 LLM 瓶颈分析：生成阶段 memory-bound，权重访存占主导
- `awq_fig2_salient_weights.png` — Salient weights 识别与 AWQ scaling 变换
- `awq_fig4_weight_packing.png` — SIMD-aware 4-bit 权重打包与解包流程
- `awq_table4_results.png` — 各模型在各 bit 精度下的 perplexity 对比
- `awq_fig8_calibration_robustness.png` — AWQ 校准集效率与分布偏移鲁棒性
- `awq_fig10_latency.png` — TinyChat 在边缘设备上的端到端延迟对比
