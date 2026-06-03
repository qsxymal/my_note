---
tags: [attention, efficiency, long-context, transformer]
related:
  - big-bird-transformers-for-longer-sequences
  - block-attention-for-efficient-prefilling
  - attention-is-all-you-need
---

# Sparse Attention

通过限制每个 query 可关注的 key 范围（稀疏注意力矩阵），将 self-attention 从 O(n²) 降至 O(n) 计算复杂度的技术集合。

## 常见稀疏模式

| 模式 | 核心思路 | 代表工作 |
|------|---------|---------|
| **Sliding Window** | 每个 token 仅关注 w 个相邻 token | Longformer, [[big-bird-transformers-for-longer-sequences]] |
| **Random** | 每个 query 关注 r 个随机 key（保证信息传播） | [[big-bird-transformers-for-longer-sequences]] |
| **Global** | 少量全局 token 关注全序列 | [[big-bird-transformers-for-longer-sequences]], Beltagy |
| **Stride/Dilated** | key 按固定间隔采样 | Sparse Transformer |
| **LSH** | 通过哈希聚类实现近似 attention | Reformer |
| **Block-level** | 按 block 粒度定义稀疏模式 | [[big-bird-transformers-for-longer-sequences]] (blockification) |

## 图稀疏化视角

Big Bird 将 self-attention 视为图稀疏化问题——full attention 等价于完全图。好的稀疏图应同时满足：
- **短平均路径长度** → 随机图（O(log n) 路径）
- **高聚类系数** → 滑动窗口（局部密集）

## 理论保证

Big Bird 证明了组合 random + window + global 三种模式的稀疏 attention 仍然是通用函数逼近器和 Turing complete，不会丢失表达能力。但存在理论下界：某些任务（找最远向量）在 Õ(n) 边稀疏 attention 下需要 Ω̃(n^{1-o(1)}) 层。

## GPU 高效率实现

GPU 不擅长稀疏矩阵乘法。Big Bird 引入 **blockification**（块化）：将序列按块大小 b 分组，用密集张量乘法和 roll/gather 操作替代稀疏运算。

## 与 Full Attention 的对比

| 方面 | Full Attention | Sparse Attention |
|:----|:-------------:|:---------------:|
| 计算复杂度 | O(n²) | O(n) |
| 长序列能力 | 受限（~512） | 强（4K-32K+） |
| 通用逼近性 | 是 | 是（需含全局 token） |
| 短文本精度 | 基准 | 持平或略降 |
| 实现复杂度 | 简单 | 复杂（需定制 kernel） |
