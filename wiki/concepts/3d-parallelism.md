---
tags: [distributed-training, parallelism, megatron]
related:
  - efficient-large-scale-language-model-training-on-gpu-clusters-using-megatron-lm
  - tensor-parallelism
  - pipeline-parallelism
---

# 3D Parallelism (PTD-P)

Pipeline + Tensor + Data parallelism 的组合方案，由 Megatron-LM 团队系统化。训练万亿参数模型的事实标准。

## 组合策略

| 并行维度 | 放置位置 | 原理 |
|---------|---------|------|
| **Tensor Parallelism** | 单节点内（8 GPU） | 利用 NVLink 高带宽，按矩阵维度拆分 |
| **Pipeline Parallelism** | 跨节点 | 按层拆分，点对点通信（仅传激活），支持 NVLink/IB |
| **Data Parallelism** | 跨 pipeline 副本 | 梯度 all-reduce，仅 per-batch 一次 |

## 关键尺度关系

- 模型并行度 $M = t \cdot p$ 需足够容纳模型参数和中间状态
- 剩余 $n/M$ 个副本用于 data parallelism
- Tensor parallel size $t$ 不应超过单节点 GPU 数（跨节点 all-reduce 太慢）

## Interleaved Pipeline Schedule

将每个 pipeline stage 分割为 $v$ 个更小的 chunks，交替分配：

- Bubble 占比从 $\frac{p-1}{m}$ 降至 $\frac{1}{v} \cdot \frac{p-1}{m}$
- 代价：通信量增加 $v$ 倍


## 大规模结果

3072 A100 GPU 上训练 1 万亿参数模型：502 petaFLOP/s，单 GPU 利用率 52% 理论峰值。
