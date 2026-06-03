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
| `smoothquant/smoothquant_fig1_precision_mapping_transformer.png` | 大模型规模 vs GPU 内存增长趋势——量化弥合供需差距 |
| `smoothquant/smoothquant_fig2_suitable_migration_strength.png` | 核心直觉：激活因 outlier 难量化，平滑后激活和权重都易于量化 |
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
| `gptq/gptq_fig1a_quantizing_opt_models.png` | OBQ→GPTQ 方法：逐列量化 + Hessian 误差补偿 |
| `gptq/gptq_fig_ablation.png` | GPTQ 算法流程：Cholesky 预处理 + 懒惰批量更新 |
| `gptq/gptq_fig_speedup.png` | GPTQ 4-bit 推理加速比与 175B 模型单卡部署 |
| `gptq/gptq_table_results.png` | 跨模型量化精度对比 |

## KVQuant

| File | Description |
|------|-------------|
| `kvquant/kvquant_fig1a_model_size_versus.png` | KV cache vs 权重的内存占比分解——短序列 vs 长序列 |
| `kvquant/kvquant_fig1b_model_size_versus.png` | KVQuant 各技术组件的逐步 perplexity 改善 |
| `kvquant/kvquant_fig2_example_distributions_activation.png` | Key pre-RoPE / post-RoPE / Value 激活分布对比 |
| `kvquant/kvquant_fig3_perplexity_results_llama.png` | 不同序列长度下的 perplexity 结果 |
| `kvquant/kvquant_fig4_distribution_magnitude_elements.png` | Key 和 Value activation 各层幅值分布 |
| `kvquant/kvquant_fig5_one_typically_achieves.png` | 离线 vs 在线校准策略分析 |

## Block-Attention

| File | Description |
|------|-------------|
| `block-attention/blockattn_fig1_masks.png` | Block-attention mask 示意——各 block 独立 attention，仅末尾 block 关注全局 |
| `block-attention/blockattn_table1_rag.png` | RAG 4 基准精度对比——差距 ≤1% vs full-attention |
| `block-attention/blockattn_table2_general.png` | 通用/ICL 7 基准——seamless switching 能力 |
| `block-attention/blockattn_table3_efficiency.png` | TTFT/FLOPs 效率——32K 时 TTFT 45ms (98.7%↓) |
| `block-attention/blockattn_fig4_accuracy.png` | Block fine-tuning 收敛速度——~200 steps |

## Multi-token Prediction

| File | Description |
|------|-------------|
| `multi-token-prediction/multitoken_fig1_overview.png` | 架构概览：共享 trunk + n 独立 head，训练+推理流程及 MBPP 结果 |
| `multi-token-prediction/multitoken_fig3_scaling.png` | MBPP pass@1/10/100 随模型规模扩展——大模型收益更大 |
| `multi-token-prediction/multitoken_table1_results.png` | 7B 模型在不同数据量和 n 值下的代码生成结果 |
| `multi-token-prediction/multitoken_fig7_induction.png` | 小模型 induction head 形成——n=2 大幅优于 n=1 |
| `multi-token-prediction/multitoken_fig8_arithmetic.png` | 多项式算法推理——n=2/4 优于 n=1，OOD 泛化显著 |
| `multi-token-prediction/multitoken_fig6_summarization.png` | 8 摘要 ROUGE-L F1 平均——n=2/4 持续优于 baseline |

## BERT Pipeline

| File | Description |
|------|-------------|
| `bert-pipeline/bert-pipeline_fig1_summary.png` | BERT-large 各探测任务的 F1、权重中心和期望层汇总 |
| `bert-pipeline/bert-pipeline_fig2_layer_metrics.png` | 逐层混合权重和差分分数——句法集中、语义分散 |

## BERT

| File | Description |
|------|-------------|
| `bert/bert_fig1_overall_pre_training.png` | 预训练 + 微调整体流程：预训练用 MLM+NSP，微调只需替换输入输出层 |
| `bert/bert_fig2_bert_input_representation.png` | 输入表示：Token Embedding + Segment Embedding + Position Embedding 求和 |
| `bert/bert_fig3_differences_pre_training.png` | BERT (bi-directional) vs OpenAI GPT (left-to-right) vs ELMo (LSTM) 架构差异 |
| `bert/bert_fig4_illustrations_fine_tuning.png` | 不同下游任务的微调方式（分类、QA、NER、句对分类） |
| `bert/bert_table1_glue.png` | GLUE 8 任务结果——BERT_LARGE 平均 82.1，高于 GPT 7% |
| `bert/bert_table2_squad.png` | SQuAD 1.1/2.0 EM 和 F1 分数——BERT_LARGE F1 93.2/83.1 |
| `bert/bert_table5_ablation_pretrain.png` | 预训练任务消融：去掉 NSP 和双向性后各任务精度大幅下降 |
| `bert/bert_table6_model_size.png` | 模型大小消融：更大预训练模型一致提升下游任务性能 |
| `bert/bert_table7_ner.png` | CoNLL-2003 NER 结果：微调 vs 特征提取方法对比 |

## Transformer

| File | Description |
|------|-------------|
| `transformer/transformer_fig1_transformer_model_architecture.png` | Transformer 编码器-解码器架构图——N×6 层 self-attention + FFN，residual + layer norm |
| `transformer/transformer_fig2a_scaled_dot_product.png` | Scaled Dot-Product Attention: QK^T/√d_k → softmax → weighted sum |
| `transformer/transformer_fig2b_scaled_dot_product.png` | Multi-Head Attention: h=8 头的并行 attention 拼接再投影 |
| `transformer/transformer_table1_complexity.png` | Self-Attention vs RNN vs CNN 的复杂度、顺序操作数、最大路径长度对比 |
| `transformer/transformer_table2_translation_results.png` | WMT 2014 翻译 BLEU 分数及训练成本对比——Transformer big 以最低成本达 SOTA |
| `transformer/transformer_table3_model_variations.png` | 架构消融实验：头数、key 维度、模型大小、dropout、位置编码方式 |
| `transformer/transformer_table4_parsing.png` | 英语成分句法分析 WSJ F1 分数——泛化能力验证 |

## PagedAttention (vLLM)

| File | Description |
|------|-------------|
| `pagedattention/pagedattention_fig1_memory_throughput.png` | A100 显存分配及 vLLM 吞吐量对比——参数 65%，KV cache >30%，batch 增大时现有系统迅速饱和 |
| `pagedattention/pagedattention_fig2_memory_waste.png` | 现有系统仅 20.4-38.2% KV cache 内存利用率，其余被碎片化和预留浪费 |
| `pagedattention/pagedattention_fig5_algorithm.png` | PagedAttention 将 KV cache 划分为非连续物理块的核心示意 |
| `pagedattention/pagedattention_fig6_block_table.png` | 逻辑到物理块的 block table 地址翻译与动态分配 |
| `pagedattention/pagedattention_fig8_parallel_sampling_cow.png` | 并行采样中 copy-on-write 机制：引用计数 >1 时复制后修改 |
| `pagedattention/pagedattention_fig12_throughput_latency.png` | 主实验结果：OPT-13B/66B/175B 在 ShareGPT/Alpaca 上的吞吐量-延迟曲线 |
| `pagedattention/pagedattention_fig13_batched_requests.png` | vLLM batch 请求数 2.2×-4.3× 高于 Orca（ShareGPT） |
| `pagedattention/pagedattention_fig11_length_distribution.png` | ShareGPT 和 Alpaca 数据集的输入/输出长度分布 |
| `pagedattention/pagedattention_table1_model_sizes.png` | 实验模型和服务器的详细配置（13B/66B/175B） |

## GPipe

| File | Description |
|------|-------------|
| `gpipe/gpipe_fig1_model_scaling.png` | ImageNet accuracy 和 BLEU vs model size 的强相关 |
| `gpipe/gpipe_fig2_pipeline_mechanism.png` | GPipe batch-splitting pipeline parallelism 核心机制 |
| `gpipe/gpipe_fig2c_figure.png` | Pipeline timing diagram：K=4 个加速器上的 bubble overhead |
| `gpipe/gpipe_table1_model_capacity.png` | GPipe 支持的最大模型容量（AmoebaNet + Transformer） |
| `gpipe/gpipe_fig3_translation_quality_across.png` | 6B 多语言 NMT 在各语言上的翻译质量提升 |

## Megatron-LM

| File | Description |
|------|-------------|
| `megatron-lm/megatron_fig1_scaling.png` | 模型并行 + 数据并行在 512 GPU 上的弱扩展 FLOPS |
| `megatron-lm/megatron_fig3a_mlp_parallel.png` | Transformer MLP 块的行/列拆分模型并行 |
| `megatron-lm/megatron_fig4_communication.png` | 单个 Transformer 层中的 4 次通信操作 |
| `megatron-lm/megatron_fig7_bert_layernorm.png` | BERT Pre-LN vs Post-LN 架构对比 |
| `megatron-lm/megatron_fig8_hybrid_parallel.png` | 混合模型并行 + 数据并行的 GPU 分组方案 |

## Megatron-LM GPU Clusters (PTD-P)

| File | Description |
|------|-------------|
| `megatron-cluster/meg_cluster_fig2_ptdp_combination.png` | PTD-P（Pipeline + Tensor + Data）并行组合 |
| `megatron-cluster/meg_cluster_fig3_gpipe_schedule.png` | GPipe 的 naive pipeline schedule 示意 |
| `megatron-cluster/meg_cluster_fig4_interleaved_schedule.png` | 1F1B vs Interleaved 流水线调度对比 |
| `megatron-cluster/meg_cluster_fig5_tensor_parallel.png` | Transformer 层内的 tensor parallelism 拆分 |
| `megatron-cluster/meg_cluster_fig11_perf_comparison.png` | 多种并行配置吞吐性能对比 |

## Optimus (Efficient 2D Method)

| File | Description |
|------|-------------|
| `optimus/optimus_fig4_optimus_architecture.png` | Optimus MLP + Self-Attention 的 2D SUMMA 架构 |
| `optimus/optimus_fig6_memory_management.png` | Optimus 分层显存管理与通信隐藏策略 |
| `optimus/optimus_fig7_scaling.png` | Weak/Strong scaling 效率（Optimus vs Megatron） |
| `optimus/optimus_fig9_memory_limits.png` | Memory limits 对比（Optimus 8× batch size） |

## RoPE (Rotary Position Embedding)

| File | Description |
|------|-------------|
| `rope/rope_fig1_illustration.png` | RoPE 旋转位置编码图形化示意 |
| `rope/rope_fig2_long_term_decay.png` | 内积上界随相对距离衰减曲线 |
| `rope/rope_table1_bleu.png` | WMT 2014 En→De BLEU 对比 |
| `rope/rope_fig3a_evaluation_rope_language.png` | MLM 和 PerFormer 训练 loss 曲线 |

## DeepNet

| File | Description |
|------|-------------|
| `deepnet/deepnet_fig1_trend_transformer_depths.png` | NLP 模型深度趋势——DeepNet 突破 1,000 层 |
| `deepnet/deepnet_table1_bleu_comparison.png` | WMT-17 各方法+深度 BLEU 对比——DeepNet 全程最优 |
| `deepnet/deepnet_table2_opus100.png` | OPUS-100 多语言翻译——200 层 31.1 BLEU |
| `deepnet/deepnet_table3_m2m_comparison.png` | M2M-100 12B vs DeepNet 3.2B——全面超越 |
| `deepnet/deepnet_fig6_iwslt_results.png` | IWSLT-14 深度 vs BLEU——含 8 种方法对比 |
| `deepnet/deepnet_fig8_average_bleu_scores.png` | 深度 vs BLEU 散点——持续增长无饱和 |

## ELMo

| File | Description |
|------|-------------|
| `elmo/elmo_table1_main_results.png` | 6 个 NLP 任务主结果——全面 SOTA，SQuAD +24.9% 相对提升 |
| `elmo/elmo_table2_layer_ablation.png` | 层权重消融——全部层优于仅顶层 |
| `elmo/elmo_fig_ablations.png` | ELMo 消融实验：层权重正则化与不同层组合效果 |
| `elmo/elmo_table4_nearest_neighbors.png` | "play" 最近邻——biLM 上下文表示 vs GloVe 静态向量 |
| `elmo/elmo_table5_wsd.png` | WSD 词义消歧——biLM 第二层 > 第一层 |

## Big Bird

| File | Description |
|------|-------------|
| `big-bird/bigbird_fig1_building_blocks.png` | 四种 attention pattern：random、window、global、BIGBIRD 组合——核心架构图 |
| `big-bird/bigbird_table1_ablations.png` | Building block 消融：仅三者组合追平 BERT-base (64.2 MLM) |
| `big-bird/bigbird_table2_qa_dev.png` | QA Dev 结果——BIGBIRD-ETC 在 4 数据集上优于 Longformer |
| `big-bird/bigbird_table3_qa_test.png` | QA Test 结果 vs Leaderboard top-3——NQ(LA)/TriviaQA/WikiHop SOTA |
| `big-bird/bigbird_table4_summarization_p1.png` | 长文档摘要 ROUGE——BigPatent 上 BIGBIRD-Pegasus R-1 60.64 |
| `big-bird/bigbird_table4_summarization_p2.png` | 长文档摘要 ROUGE 续表 |
| `big-bird/bigbird_table5_mlm_bpc.png` | DNA 序列 MLM BPC：BIGBIRD 1.12 < BERT 1.23 < SRILM 1.57 |
| `big-bird/bigbird_fig7_dna_mlm.png` | DNA MLM 预训练 BPC 与已有方法对比 |
| `big-bird/bigbird_fig8_accuracy.png` | 长文档分类 accuracy——Arxiv +5% over SoTA |
| `big-bird/bigbird_table7_chromatin_profile.png` | 基因组染色质谱预测——启动子检测 99.9% F1 |

## Reducing Activation Recomputation

| File | Description |
|------|-------------|
| `activation/activation_fig1_memory_breakdown.png` | 参数、优化器状态、激活显存分解——激活是最大瓶颈 |
| `activation/activation_fig2_transformer_arch.png` | Transformer 标准架构示意——alternating self-attention + FFN |
| `activation/activation_fig3_self_attention_selective.png` | Self-attention 中 selective recomputation 的重算范围 |
| `activation/activation_fig4_tensor_parallelism.png` | Tensor parallelism 跨 GPU 拆分示意 |
| `activation/activation_fig5_tp_with_sequence.png` | Tensor + Sequence Parallelism 结合的 Transformer 层 |
| `activation/activation_fig7_memory_reduction.png` | 各技术激活内存降低效果（5× 降低） |
| `activation/activation_fig8_perf_breakdown.png` | 各规模模型前向/反向/重算时间分解 |
| `activation/activation_fig9_pipeline_memory_opt.png` | Pipeline parallelism 下各 stage 的激活内存优化

## FlashInfer

| File | Description |
|------|-------------|
| `flashinfer/flashinfer_fig1_overview_system_design.png` | FlashInfer 系统设计总览：JIT 编译 + 运行时调度 |
| `flashinfer/flashinfer_fig2_representation_page_bsr.png` | Page Table 在 BSR 格式下的统一表示 |
| `flashinfer/flashinfer_fig3_composable_formats_shared.png` | Composable formats 共享前缀分解示意图 |
| `flashinfer/flashinfer_fig4_data_transfer_global.png` | Global 到 shared memory 的稀疏/稠密数据传输 |
| `flashinfer/flashinfer_fig5_jit_compiler_attention.png` | JIT 编译器架构与 attention 变体模板 |
| `flashinfer/flashinfer_fig6_load_balanced_runtime.png` | 负载均衡运行时调度器工作流 |
| `flashinfer/flashinfer_fig7a_medium_inter_token.png` | SGLang ITL 主实验结果 |
| `flashinfer/flashinfer_fig7b_medium_inter_token.png` | SGLang TTFT 主实验结果 |
| `flashinfer/flashinfer_fig8a_achieved_bandwidth_flops.png` | Decode 带宽利用率对比 |
| `flashinfer/flashinfer_fig8b_achieved_bandwidth_flops.png` | Prefill FLOPs 利用率对比 |
| `flashinfer/flashinfer_fig9a_end_end_latency.png` | StreamingLLM 端到端延迟 |
| `flashinfer/flashinfer_fig9b_end_end_latency.png` | Fused RoPE kernel 带宽利用率 |
| `flashinfer/flashinfer_fig10a_itl_ttft_mlc.png` | Composable formats 在 MLC-Engine 上的 ITL 加速 |
| `flashinfer/flashinfer_fig10b_itl_ttft_mlc.png` | Composable formats 在 MLC-Engine 上的 TTFT 加速 |
| `flashinfer/flashinfer_fig11_head_group_fusion.png` | GQA head-group fusion 融合策略 |
| `flashinfer/flashinfer_fig12a_achieved_tflops_prefill.png` | Sparse/dense prefill TFLOPs 对比 |
| `flashinfer/flashinfer_fig12b_achieved_tflops_prefill.png` | Sparse/dense decode 带宽利用率 |

## Gated Convolutional (GCNN)

| File | Description |
|------|-------------|
| `gated-convolutional/gated-conv_fig1_architecture_gated_convolutional.png` | GCNN 架构图——embedding + 堆叠门控卷积 + adaptive softmax |
| `gated-convolutional/gated-conv_fig2_comparison_state_art.png` | 计算量 vs 困惑度——GCNN + AdaSoftmax ops/ppl 效率远高于 LSTM+Full Softmax |
| `gated-convolutional/gated-conv_fig3a_learning_curves_wikitext.png` | WikiText-103 学习曲线——GLU 收敛最快、困惑度最低 |
| `gated-convolutional/gated-conv_fig3b_learning_curves_wikitext.png` | Google Billion Word 100 小时学习曲线——GLU 比 GTU/ReLU 低 ~5 ppl |
| `gated-convolutional/gated-conv_fig4a_test_perplexity_function.png` | Google Billion Word 困惑度 vs 上下文长度——>20 token 后收益递减 |
| `gated-convolutional/gated-conv_fig4b_test_perplexity_function.png` | WikiText-103 困惑度 vs 上下文长度——~30 token 已足够 |
| `gated-convolutional/gated-conv_fig5_learning_curves_google.png` | 不同非线性程度模型的 Google Billion Word 学习曲线 |
| `gated-convolutional/gated-conv_fig6_effect_weight_normalization.png` | WeightNorm + Gradient Clipping 的收敛加速效果——学习率可从 0.01 提升至 1.0 |

## Gated DeltaNet

| File | Description |
|------|-------------|
| `gated-delta-networks/gated-delta_fig1_architecture.png` | Gated DeltaNet 架构和 block 设计——Hybrid 层间布局 + Gated Delta Rule 内部流程（Linear/Conv/SiLU/L2 Norm）|
| `gated-delta-networks/gated-delta_fig2_length_extrapolation.png` | 6 个长文本基准上的长度外推 perplexity——Gated DeltaNet 在纯循环模型中最低 |
| `gated-delta-networks/gated-delta_fig3_throughput.png` | 1.3B 模型在单 H100 上的训练吞吐量对比——Gated DeltaNet 与 DeltaNet 几乎相同 |

## Linear Fast Weight (DeltaNet)

| File | Description |
|------|-------------|
| `linear-fast-weight/linear-fwp_fig1_visualisation_dpfp_space.png` | DPFP 可视化——2d 平面 → 4d 颜色曲面，四个象限各激活一个正交分量 |
| `linear-fast-weight/linear-fwp_fig2_final_evaluation_loss.png` | 容量测试——超过 d_dot 后线性注意力检索误差骤升，softmax 不受限 |
| `linear-fast-weight/linear-fwp_fig3_learning_curves_different.png` | Delta rule vs sum rule 在关联检索任务上的学习曲线——delta ~20 epoch 收敛 |
| `linear-fast-weight/linear-fwp_fig4_training_curves_setting.png` | Setting 1 中 600 个关联的训练曲线 |
| `linear-fast-weight/linear-fwp_fig5_final_evaluation_loss.png` | Setting 2（有替换采样）不同更新规则最终损失对比——delta rule 最优 |

## Positional Interpolation

| File | Description |
|------|-------------|
| `positional-interpolation/pi_fig1_effective_context_window.png` | PI 方法示意——线性缩放位置索引使所有位置落在预训练范围内 |
| `positional-interpolation/pi_fig2_illustration_position_interpolation.png` | Extrapolation 出界 vs Interpolation 有界的 attention score 对比 |
| `positional-interpolation/pi_fig3_passkey_prompt.png` | Passkey retrieval 的 prompt 格式 |
| `positional-interpolation/pi_table1_pg19_perplexity.png` | PG-19/Proof-pile 的 PI vs FT perplexity |
| `positional-interpolation/pi_table4_effective_window.png` | Passkey retrieval 有效上下文窗口对比 |
| `positional-interpolation/pi_table5_zeroshot_benchmarks.png` | 原始窗口内零样本基准任务结果 |
| `positional-interpolation/pi_table6_govreport_rouge.png` | GovReport 长文档摘要 ROUGE 分数 |

## Dynamic Tanh (DyT) — Transformers without Normalization

| File | Description |
|------|-------------|
| `transformers-without-normalization/no-norm_fig1_original_transformer_block.png` | DyT 替换 LN 的架构对比——左：原版 Transformer block，右：DyT block |
| `transformers-without-normalization/no-norm_fig2a_output_input_selected.png` | ViT LN 层的输入-输出 S 形映射——与 tanh 高度相似 |
| `transformers-without-normalization/no-norm_fig2b_output_input_selected.png` | wav2vec 2.0 LN 层的 S 形映射 |
| `transformers-without-normalization/no-norm_fig2c_output_input_selected.png` | DiT LN 层的 S 形映射 |
| `transformers-without-normalization/no-norm_fig3_three_different_values.png` | tanh(αx) 在不同 α 下的三种曲线形态 |
| `transformers-without-normalization/no-norm_fig4_output_input_two.png` | token/channel 分组下 LN 映射的可视化——per-token 线性 + collective S 形 |
| `transformers-without-normalization/no-norm_fig6a_training_loss_curves.png` | ViT-B 训练 loss——LN vs DyT 几乎重合 |
| `transformers-without-normalization/no-norm_fig6b_training_loss_curves.png` | ConvNeXt-B 训练 loss——LN vs DyT 几乎重合 |
| `transformers-without-normalization/no-norm_fig7a_llama_pretraining_loss.png` | LLaMA 7B 预训练 loss——DyT vs RMSNorm 一致 |
| `transformers-without-normalization/no-norm_fig7b_llama_pretraining_loss.png` | LLaMA 13B 预训练 loss——DyT vs RMSNorm 一致 |
| `transformers-without-normalization/no-norm_fig7c_llama_pretraining_loss.png` | LLaMA 34B 预训练 loss——DyT vs RMSNorm 一致 |
| `transformers-without-normalization/no-norm_fig7d_llama_pretraining_loss.png` | LLaMA 70B 预训练 loss——DyT vs RMSNorm 一致 |
| `transformers-without-normalization/no-norm_fig8b_curves_three_squashing.png` | α 训练中追踪 1/std——自动学习缩放范围 |
| `transformers-without-normalization/no-norm_fig9_performance_different_tasks.png` | 不同 α₀ 对各任务性能的影响——大范围稳定 |

## ALiBi (Attention with Linear Biases)

| File | Description |
|------|-------------|
| `alibi-attention/alibi_fig1a_extrapolation_input_sequence.png` | L=512 训练各位置编码外推 perplexity——ALiBi 持续改善，sinusoidal/rotary 爆炸 |
| `alibi-attention/alibi_fig1b_extrapolation_input_sequence.png` | L=1024 训练相同对比 |
| `alibi-attention/alibi_fig2a_comparison_batched_training.png` | 训练速度对比——ALiBi 与 Sinusoidal 接近，T5 Bias 显著更慢 |
| `alibi-attention/alibi_fig2b_comparison_batched_training.png` | 推理速度对比——ALiBi 与 Sinusoidal 几乎相同 |
| `alibi-attention/alibi_fig2c_comparison_batched_training.png` | 训练内存对比——ALiBi 仅增加 ~100MB（n×L×L bias） |
| `alibi-attention/alibi_fig3_when_computing_attention.png` | ALiBi 核心机制：query-key dot product 后加线性 bias（右图），softmax 前 |

## Attention Visualization

| File | Description |
|------|-------------|
| `attention-visualization/attn-viz_fig1a_attention_head_view.png` | GPT-2 注意力头视图（三种 attention 模式） |
| `attention-visualization/attn-viz_fig1b_attention_head_view.png` | BERT 注意力头视图（sentence-pair 过滤） |
| `attention-visualization/attn-viz_fig2_examples_attention_heads.png` | GPT-2 中特定词法模式 head 示例 |
| `attention-visualization/attn-viz_fig3_attention_pattern_gpt.png` | 指代消解 attention 中的性别偏差 |
| `attention-visualization/attn-viz_fig4_model_view_gpt.png` | GPT-2 模型视图（small multiples） |
| `attention-visualization/attn-viz_fig5_neuron_view_gpt.png` | 第 8 层第 6 头神经元视图 |
| `attention-visualization/attn-viz_fig6_neuron_view_gpt.png` | 第 1 层第 10 头距离衰减 attention 模式 |
| `alibi-attention/alibi_fig4_models_trained_evaluated.png` | WikiText-103 主结果——所有训练长度的 ALiBi 均优于 Sinusoidal |
| `alibi-attention/alibi_fig5a_1_parameter_alibi.png` | 1.3B ALiBi L=512 训练 vs Sinusoidal L=1024 训练的外推对比 |
| `alibi-attention/alibi_fig5b_1_parameter_alibi.png` | 1.3B ALiBi L=1024 训练 vs Sinusoidal L=2048 训练的外推对比 |

## Layer Normalization in Transformer (Post-LN / Pre-LN)

| File | Description |
|------|-------------|
| `layer-normalization-transformer/layernorm-transformer_fig1_post_transformer_layer.png` | Post-LN vs Pre-LN 架构对比——LN 放在 residual block 内/外的位置差异 |
| `layer-normalization-transformer/layernorm-transformer_fig2a_performances_models_optimized.png` | Adam 优化下 Post-LN/Pre-LN 在不同 warmup 策略的 loss/BLEU 对比 |
| `layer-normalization-transformer/layernorm-transformer_fig2b_performances_models_optimized.png` | SGD 优化下 Post-LN/Pre-LN 在不同 warmup 策略的 loss/BLEU 对比 |
| `layer-normalization-transformer/layernorm-transformer_fig3a_norm_gradients_1.png` | Post-LN 各层梯度范数——输出层极大（~10²），随层 index 衰减 |
| `layer-normalization-transformer/layernorm-transformer_fig3b_norm_gradients_1.png` | Pre-LN 各层梯度范数——小且均匀（~10⁻¹），不随深度变化 |
| `layer-normalization-transformer/layernorm-transformer_fig4a_performances_models_iwslt14.png` | IWSLT14 De-En 翻译 loss 曲线——Pre-LN 收敛更快 |
| `layer-normalization-transformer/layernorm-transformer_fig5a_performances_models_unsupervised.png` | BERT 预训练 MLM accuracy——Pre-LN 更快且最终更好 |
