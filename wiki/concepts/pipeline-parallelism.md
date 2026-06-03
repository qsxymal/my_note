---
tags: [distributed-training, parallelism]
related:
  - gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism
  - megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism
---

# Pipeline Parallelism

将模型按层分段（vertical slicing），每段放置在不同加速器上，通过流水线方式执行 micro-batch 的并行策略。

## 核心概念

- **Partition / Cell** — 连续层的组合，放置在一个加速器上
- **Micro-batch** — 将 mini-batch 进一步拆分的小批量，用于填充流水线
- **Bubble overhead** — 流水线启动和排空产生的设备空闲比例
- **Stage** — pipeline 中的每个设备/分区

## Bubble Overhead 分析

$K$ 个分区、$M$ 个 micro-batch 时，bubble 占比为：

$$\text{bubble} = \frac{K-1}{M+K-1}$$

当 $M \ge 4K$ 时，bubble overhead 可忽略。

## 对比 Tensor Parallelism

| 维度 | Pipeline Parallelism | Tensor Parallelism |
|------|---------------------|-------------------|
| 切分方式 | 按层切分（vertical） | 按矩阵维度切分（horizontal） |
| 通信粒度 | 每 micro-batch 一次激活传输 | 每层多次 AllReduce |
| 是否需高速互联 | 否 | 是 |
| 内存均衡 | 需分区算法最小化差异 | 天然均衡 |
| 单层过大 | 不支持 | 支持拆分单层 |

## 主要方案演变

- **GPipe** — batch-splitting + 同步梯度，$M \ge 4K$ 时近线性加速
- **PipeDream** — 1F1B schedule + 异步梯度，存在权重陈旧
- **Megatron-LM** — PP + TP + DP 组合，interleaved schedule
- **1F1B (One-Forward-One-Backward)** — 前向后向交替调度，减少 bubble
