# Attention Variants

这个目录用同一组 toy 输入对照 MHA、MQA、GQA 和 MLA。完整定义、真实模型配置与逐层 shape 统一放在 [Attention 学习笔记 00](../../notes/attention/00-from-vanilla-attention-to-kda.md)。

## 脚本覆盖范围

| 变体 | 教学脚本中的设置 | 重点观察 |
|---|---|---|
| MHA | 2 个 Q heads，2 个 KV heads | 每个 Q head 使用独立 KV head |
| MQA | 2 个 Q heads，1 个 KV head | 所有 Q heads 共享一组 K/V |
| GQA | 4 个 Q heads，2 个 KV heads | 每组 Q heads 共享一个 KV head |
| MLA | KV latent dimension 为 3 | 缓存 latent，再展开 K/V |

所有变体共享：

```text
sequence length = 4
hidden size = 8
```

MLA 部分只实现 latent KV 压缩与展开。DeepSeek MLA 的 Query/KV LoRA、NoPE/RoPE 分支、projection absorption 和真实 shape 请直接阅读笔记 00 的第 7 节。

## 运行

```bash
python 01_model_basics/attention_variants/compare_attention_variants.py
```

## 观察点

运行输出重点包括：

- 各变体的 Q head 与 KV head 数量
- Q/K/V projection 和拆 head 后的 shape
- 某个 Q head 映射到哪个 KV head
- 第 4 个 token 在指定 head 上的 score row
- MLA 的完整 K/V 逻辑 shape 与 latent cache shape

这个脚本使用 dummy weight，目标是验证数据流和 head-sharing 关系。真实模型行为与部署性能需要结合官方权重和对应 kernel 分析。
