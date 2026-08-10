# Vanilla Transformer

这个目录提供一个最小 decoder-only Transformer block，用可打印的小矩阵观察完整数据流。数学背景统一放在 [Attention 学习笔记 00](../../notes/attention/00-from-vanilla-attention-to-kda.md)。

## 实验范围

脚本只使用单头 attention，依次执行：

```text
token embedding
-> Q/K/V projection
-> QK^T / sqrt(d)
-> causal mask
-> softmax
-> probabilities @ V
-> output projection
-> residual + layernorm
-> FFN
-> residual + layernorm
```

为了让全部数值可以直接打印，实验固定使用：

| 配置 | 数值 |
|---|---:|
| sequence length | 4 |
| hidden size | 4 |
| FFN hidden size | 8 |
| attention heads | 1 |

脚本没有引入 RoPE、GQA、KV cache、真实模型权重或高性能 kernel。

## 运行

```bash
python 01_model_basics/vanilla_transformer/vanilla_decoder_block.py
```

## 观察点

输出会依次打印：

- embedding 与 Q/K/V 的 shape 和数值
- raw scores、causal mask、masked scores
- softmax probabilities 与加权后的 V
- attention residual、layernorm、FFN 和最终 block 输出
- 第 4 个 token 的 query 如何形成一整行 scores，并完成 mask、softmax 和 value 聚合

完成这个实验后，再进入 [`attention_variants`](../attention_variants/) 查看 multi-head、KV head 共享和 MLA latent cache。
