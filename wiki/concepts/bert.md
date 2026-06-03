---
tags: [pre-training, language-representation, bidirectional, transformer-encoder]
related:
  - bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding
  - transformer
  - elmo
---

# BERT

Bidirectional Encoder Representations from Transformers——首个深度双向预训练 [[transformer]] 编码器模型，通过 Masked LM + Next Sentence Prediction 从无标注文本学习双向表示，开创 NLP 的"预训练+微调"范式。

## 预训练任务

**Masked LM (MLM)**：随机 mask 15% token 并预测，实现真正深度双向条件化。为解决 [MASK] 在微调时不出现的问题，采用 80/10/10 混合策略：
- 80% → 替换为 [MASK]
- 10% → 替换为随机词
- 10% → 保持不变

**Next Sentence Prediction (NSP)**：50% 真实下一句 / 50% 随机句子，二分类，学习句子间关系。

## 架构

| 模型 | Layers | Hidden | Heads | 参数量 |
|:----|:------:|:------:|:----:|:-----:|
| BERT_BASE | 12 | 768 | 12 | 110M |
| BERT_LARGE | 24 | 1024 | 16 | 340M |

输入 = Token Embeddings + Segment Embeddings + Position Embeddings，[CLS] 用于分类，[SEP] 分隔句子对。

## 效果

GLUE 平均 82.1（+7.0%），SQuAD 1.1 F1 93.2，SQuAD 2.0 F1 83.1（+5.1），11 项任务全部 SOTA。消融实验证明 MLM + NSP 和双向性缺一不可。

## 影响

BERT 与 GPT 形成 NLP 两大路线：encoder-only（理解型）vs decoder-only（生成型）。后续几乎全部 NLU 模型（RoBERTa、ALBERT、DistilBERT、DeBERTa）均在其基础上改进。
