---
tags: [quantization, llm, w4a16, hessian]
related:
  - gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers
  - awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration
---

# GPTQ (Accurate Post-Training Quantization for Generative Pre-trained Transformers)

算法基于 OBQ（Optimal Brain Quantizer）在大模型上的高效重写。

## 与 AWQ 的对比

| 方面 | GPTQ | AWQ |
|------|------|-----|
| 原理 | Hessian-guided 误差补偿 | 激活感知 scaling 变换 |
| 校准开销 | ~4 GPU 小时（175B） | 分钟级 |
| 精度（4-bit） | 无损 | 无损 |
| 是否需要反向传播 | 否（解析形式补偿） | 否 |
| 极端量化（2-bit） | 支持且有合理精度 | 未验证 |
| 理论基础 | 二阶近似，严格的误差界 | 启发式 |
| 社区采纳 | Hugging Face 集成，应用更广 | 端侧部署为主 |

## 数学基础

GPTQ 使用 Hessian 矩阵 $H = 2XX^T$（$X$ 为校准数据的激活）来量化每列权重。量化第 $i$ 列时：
1. 找到最优量化值 $q_i = \text{quant}(w_i)$
2. 计算误差 $\delta_i = (q_i - w_i) / H_{ii}^{-1}$
3. 用 $\delta_i$ 更新剩余未量化列：$w_{j>i} \mathrel{-}= \delta_i \cdot H_{:,j}^{-1}$
