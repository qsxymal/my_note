---
tags: [distributed-training, model-parallelism, summa]
related:
  - an-efficient-2d-method-for-training-super-large-deep-learning-models
  - tensor-parallelism
  - 3d-parallelism
---

# 2D Parallelism

基于 SUMMA 的 2D 分区模型并行方案，将参数和激活同时分布在 $q \times q$ 网格上，消除 [[tensor-parallelism|Megatron 式 1D 并行]]的激活冗余瓶颈。

## 核心思想

Megatron 1D 并行：参数拆分，但**每个设备持有完整激活**。当 GPU 数 $p > N/3$ 时，激活成为瓶颈。

Optimus 的 2D 并行：将矩阵乘法在 $q \times q$ 网格上分块执行（SUMMA 算法），**参数和激活全部冗余消除**。

## SUMMA 原理

```
C = AB, 设备(i,j)持有 Aᵢⱼ, Bᵢⱼ
每轮 l:
  - Aᵢₗ 沿行 broadcast
  - Bₗⱼ 沿列 broadcast
  - Cᵢⱼ += Aᵢₗ × Bₗⱼ
```

## 对比 1D (Megatron)

| 方面 | Megatron (1D) | Optimus (2D) |
|:----|:-------------|:-------------|
| 激活存储 | 每个设备完整复制 | 分布化，无冗余 |
| 最大 batch (64 GPU) | 60 | 480 (8×) |
| Isoefficiency | $p^3$ | $(\sqrt{p}\log p)^3$ |
| 通信量 | $\Theta(bsh)$ | $\Theta(bsh/\sqrt{p})$ |
| GPU 数约束 | 任意 | 必须为完全平方数 $p=q^2$ |

## 局限

- GPU 数必须为完全平方数，部署灵活性受限
- 工程复杂度高于 Megatron 的 1D 方案，限制了实际落地
