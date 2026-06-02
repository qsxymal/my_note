# Image Index

<!-- Auto-generated catalog of extracted paper figures. Updated on each ingest. -->

## LLM.int8()

| File | Description |
|------|-------------|
| `llm.int8/llmint8_fig1_outlier_emergence.png` | Outlier 涌现导致常规量化在 6.7B 参数时失效，LLM.int8() 保持 16-bit 精度 |
| `llm.int8/llmint8_fig2_schematic.png` | LLM.int8() 流程：outlier 分解 + vector-wise 量化的两阶段方案 |
| `llm.int8/llmint8_fig3_outlier_analysis.png` | Outlier 特征幅值和数量与 C4 perplexity 退化的相关性 |
| `llm.int8/llmint8_table_results.png` | OPT 模型全量实验结果（zero-shot + perplexity） |

## SmoothQuant

| File | Description |
|------|-------------|
| `smoothquant/smoothquant_fig1_model_scaling.png` | 大模型规模 vs GPU 内存增长趋势——量化弥合供需差距 |
| `smoothquant/smoothquant_fig2_migration_intuition.png` | 核心直觉：激活因 outlier 难量化，平滑后激活和权重都易于量化 |
| `smoothquant/smoothquant_fig3_quant_schemes.png` | Per-tensor vs per-token+per-channel 量化定义与硬件兼容性 |
| `smoothquant/smoothquant_fig4_magnitude_evidence.png` | OPT-13B 激活与权重幅值分布——平滑前后对比 |
| `smoothquant/smoothquant_fig5_smoothing_factor.png` | Smoothing factor 计算与融合到前一层参数的流程 |
| `smoothquant/smoothquant_fig6_precision_mapping.png` | Transformer 块精度映射：INT8 用于计算密集型算子 |
| `smoothquant/smoothquant_table3_results.png` | OPT-175B 零样本精度——SmoothQuant 匹配 FP16 |
| `smoothquant/smoothquant_fig10_alpha_sweetspot.png` | 迁移强度 α 的 sweet spot 分析 |

## AWQ

| File | Description |
|------|-------------|
| `awq/awq_fig3_bottleneck.png` | 端侧 LLM 瓶颈分析：生成阶段 memory-bound，权重访存占主导 |
| `awq/awq_fig2_salient_weights.png` | 基于激活分布识别 salient weights，AWQ 通过 scaling 变换保护重要通道 |
| `awq/awq_fig4_weight_packing.png` | SIMD-aware 4-bit 权重打包与 ARM NEON 解包流程 |
| `awq/awq_table4_results.png` | LLaMA/Llama-2 各模型在各 bit 精度下的 perplexity 对比 |
| `awq/awq_fig8_calibration_robustness.png` | AWQ 校准集效率与分布偏移鲁棒性（vs GPTQ） |
| `awq/awq_fig10_latency.png` | TinyChat 在 Jetson Orin 和 Raspberry Pi 上的延迟对比 |

## GPTQ

| File | Description |
|------|-------------|
| `gptq/gptq_fig1_method.png` | OBQ→GPTQ 方法：逐列量化 + Hessian 误差补偿 |
| `gptq/gptq_fig_ablation.png` | GPTQ 算法流程：Cholesky 预处理 + 懒惰批量更新 |
| `gptq/gptq_fig_speedup.png` | GPTQ 4-bit 推理加速比与 175B 模型单卡部署 |
| `gptq/gptq_table_results.png` | 跨模型量化精度对比 |
