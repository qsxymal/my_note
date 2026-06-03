---
tags: [distributed-training, parallelism]
related:
  - megatron-lm-training-multi-billion-parameter-language-models-using-model-parallelism
  - gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism
---

# Tensor Parallelism (Intra-Layer Model Parallelism)

将单一矩阵乘法或张量运算按行/列拆分到多个 GPU 执行的并行策略。与 pipeline parallelism（按层切分）正交互补。

## 核心思路（Megatron-LM）

Transformer MLP 块：

- 第一个 GEMM 按**列拆分**：$A = [A_1, A_2]$ → 各 GPU 独立计算 $Y_i = \text{GeLU}(XA_i)$，无需同步
- 第二个 GEMM 按**行拆分**：直接从 $Y_i$ 取输入，输出前做一次 all-reduce

Self-Attention：将 QKV GEMM 按列拆分，每个 GPU 处理部分 attention head。

## f 和 g 算子

两个共轭算子通过 PyTorch autograd Function 实现，各添加一次 all-reduce：

| 算子 | 前向 | 反向 |
|------|------|------|
| `f` | 恒等 | all-reduce（梯度同步） |
| `g` | all-reduce（激活同步） | 恒等 |

## 对比 Pipeline Parallelism

| 维度 | Tensor Parallelism | Pipeline Parallelism |
|------|-------------------|---------------------|
| 切分粒度 | 层内（intra-layer） | 层间（inter-layer） |
| 通信频率 | 每层 2 次 all-reduce | 每 micro-batch 1 次激活传输 |
| 通信量 | 大（每次传输完整 hidden size） | 小（仅分区边界） |
| 是否需要高速互联 | **是**（NVLink/NVSwitch） | 否（PCI-E 即可） |
| 单层过大 | 支持 | 不支持 |
| 典型应用 | Megatron-LM | GPipe, 1F1B |
