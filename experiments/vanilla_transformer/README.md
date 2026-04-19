# Vanilla Transformer

这个目录现在只保留最朴素的一版 `decoder-only transformer block`：

- 不用 Hugging Face
- 不用真实模型
- 不用 multi-head
- 不用 RoPE
- 不用 GQA
- 不用 KV cache

也就是只看最基础的数据流：

1. token embedding
2. `Q / K / V` 线性投影
3. `QK^T / sqrt(d)`
4. causal mask
5. softmax
6. `attention_probs @ V`
7. output projection
8. residual
9. layernorm
10. FFN
11. residual
12. layernorm

## 为什么继续收敛到单头

前一版已经能讲 attention，但 `multi-head` 仍然会把注意力分散掉。

如果当前目标只是：

- 看清第 4 个 token 到底怎么进入 attention
- 看清 `attention mask` 到底作用在哪一步
- 看清一整行 score 是怎么来的

那单头版本更合适。

因为这时所有张量都会直接是：

- `X : [seq_len, hidden_size]`
- `Q : [seq_len, hidden_size]`
- `K : [seq_len, hidden_size]`
- `V : [seq_len, hidden_size]`
- `scores : [seq_len, seq_len]`

不需要再解释：

- head split
- head concat
- 不同 head 看不同子空间

## 这个实验的设置

这里故意把维度压得非常小，方便直接 print 和手推：

- `seq_len = 4`
- `hidden_size = 4`
- `ffn_hidden_size = 8`

也就是：

- 一共有 4 个 token
- 每个 token 的 hidden state 长度只有 4
- attention 就只有一个头

## 运行

```bash
python experiments/vanilla_transformer/vanilla_decoder_block.py
```

## 这个脚本会打印什么

脚本会打印：

- 输入 token id
- embedding 后的 `X`
- `Q / K / V` 的 shape 和数值
- raw attention score 矩阵
- causal mask
- mask 后的 score
- softmax 后的 attention probability
- `attention_probs @ V` 的输出
- output projection 之后的结果
- residual + layernorm 之后的结果
- FFN 中间结果和最终 block 输出

同时会专门单独解释：

- 第 4 个 token `t3`
- 它对应的 `q3`
- 它怎样和全部 `K` 做点积，得到一整行 score
- 怎样被 causal mask 保留 / 屏蔽
- 怎样对全部 `V` 做加权求和

## 现在最该记住什么

1. attention mask 不是独立模块，它只是作用在 `QK^T` 之后、softmax 之前。
2. 每一行 attention score 都对应“当前 query token 去看所有 key token”。
3. decoder-only causal mask 的本质就是：不允许当前位置看未来列。
4. 把这版最小单头 attention 看清楚之后，再去加 multi-head、KV cache、RoPE 才不会混。
