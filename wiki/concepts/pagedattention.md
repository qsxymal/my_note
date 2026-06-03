---
tags: [attention, kv-cache, inference, memory-management]
related:
  - efficient-memory-management-for-large-language-model-serving-with-pagedattention
---

# PagedAttention

借鉴操作系统虚拟内存分页思想管理 KV cache——将 KV cache 划分为固定大小的 block，按需分配、非连续存储，消除内部碎片和外部碎片，同时支持跨请求的 memory sharing（copy-on-write）。

## 核心思想

KV cache 存储在连续显存中导致严重碎片化：预分配最大长度浪费（内部碎片），请求间独立分配无法共享（外部碎片）。PagedAttention 将 KV cache 分页，每个 block 存储固定数量 token 的 KV 向量，物理上可不连续。

## Block Table

类似虚拟内存的 page table，维护逻辑 block → 物理 block 的映射：

- **Logical blocks**：请求的连续 token 位置
- **Physical blocks**：显存中任意位置的固定大小块
- **Block table**：记录每个 logical block 对应的 physical block 及引用计数

## Memory Sharing

相同前缀的请求可共享 KV block（如 system prompt、few-shot examples），引用计数为 0 时才释放。Copy-on-write 策略处理需要修改的场景。将共享 KV cache 的内存开销从 O(请求数 × 前缀长度) 降至 O(唯一前缀长度)。

## 效果

在 LLaMA-13B 上：
- 推理吞吐提升 2-4×
- 显存利用率从 20-40% 提升至 90%+
- 共享前缀场景下吞吐提升高达 5×
