---
tags: [word-embeddings, contextualized, nlp, pre-training]
related:
  - deep-contextualized-word-representations
---

# ELMo

Embeddings from Language Models——将预训练双向 LSTM 语言模型所有层的表示加权组合作为词向量，开创"预训练 + 微调"范式在 NLP 的全面应用。

## 核心思想

每个 token 的表示是所有 biLM 层的线性组合，而非仅用顶层：

$$\text{ELMo}_k = \gamma \cdot \sum_{j} s_j \cdot \mathbf{h}_{k,j}^{LM}$$

其中 $s_j$ 是 softmax 归一化的层权重，$\gamma$ 是任务特定的缩放系数。不同层编码不同信息：**低层编码句法**（POS 97.22%），**高层编码语义**（WSD 69.0 F1）。

## 架构

2 层 biLSTM（4096 units + 512 dim 投影），char CNN 输入（2048 个 n-gram 滤波器 + 2 层 highway），层间 residual connection。

## 效果

| 任务 | 相对提升 |
|:---:|:--------:|
| SQuAD | +24.9% |
| SRL | +17.2% |
| Coref | +9.8% |
| NER | +21% |
| SST-5 | +6.8% |
| SNLI | +5.8% |

## 与 BERT/GPT 的关系

ELMo 是 feature-based（预训练表示作为特征拼接），GPT 和 BERT 是 fine-tuning（在下游任务上微调全部参数）。BERT 同样使用所有层的表示做下游任务，继承了 ELMo 对不同层信息差异的洞察。
