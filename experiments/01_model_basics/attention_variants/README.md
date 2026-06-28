# Attention Variants

这个目录接在 `experiments/01_model_basics/vanilla_transformer/` 后面。

上一节的目标是：

- 先把最基础的单头 `Q / K / V -> score -> mask -> softmax -> PV` 看明白

这一节开始才引入 attention 变体，但仍然坚持两个原则：

- 只做教学版 dummy weight
- 只要求逻辑正确，不追求真实模型细节完整

所以这里不会去实现：

- HF wrapper
- 真实模型权重
- 高性能 kernel
- 完整工程版 cache / rope / fused path

## 这里实现哪些变体

当前脚本会对照跑四种形式：

1. `MHA`
   - `num_q_heads = num_kv_heads`
   - 每个 query head 都有自己独立的 K/V head

2. `MQA`
   - `num_q_heads > num_kv_heads = 1`
   - 多个 query heads 共享同一组 K/V

3. `GQA`
   - `num_q_heads > num_kv_heads > 1`
   - 每几个 query heads 共享一组 K/V

4. `MLA`
   - 这里实现的是教学版 MLA
   - 核心只保留：
     - 先把 `KV` 压到更小的 latent
     - 再从 latent 重建 K/V 参与 attention
   - 不实现 DeepSeek 论文里的完整 decoupled RoPE 等细节

## 这份脚本最想让你看到什么

核心不是背定义，而是建立最基本的系统直觉：

1. `MHA / MQA / GQA` 的主要差别，不在 attention 公式本身，而在：
   - `Q` 有多少个 heads
   - `K/V` 有多少个 heads
   - `K/V` 会被多少个 query heads 共享

2. `MLA` 的主要差别，不在于 softmax 公式换了，而在于：
   - 不直接缓存完整 K/V
   - 先缓存更小的 latent KV
   - 需要时再从 latent 恢复出参与 attention 的 K/V

3. 也就是说：
   - `MHA / MQA / GQA` 更像 “head sharing pattern 不同”
   - `MLA` 更像 “KV state representation 不同”

## 先把两大类分清

这一节里其实有两类完全不同的思路。

第一类是：

- `MHA`
- `MQA`
- `GQA`

它们的共同点是：

- attention 公式没有本质变化
- 还是显式产生 `Q / K / V`
- 还是显式缓存 `K / V`

它们之间主要只差一件事：

- `Q heads` 和 `KV heads` 的共享关系

可以直接记成：

- `MHA = 不共享 KV`
- `GQA = 部分共享 KV`
- `MQA = 全部共享 KV`

第二类是：

- `MLA`

它和前面三者不是同一层面的微调。

它改的不是：

- “一个 KV head 被几个 Q heads 共享”

而是：

- “KV 到底以什么形式被缓存”

也就是：

- 前三者改的是 `head sharing`
- `MLA` 改的是 `state representation`

## MQA 和 GQA 到底差在哪里

`MQA` 和 `GQA` 很容易混，因为它们都不是标准 `MHA`。

最短的区分方式是：

- `GQA` 是部分共享
- `MQA` 是把共享推到极限

举例：

如果：

- `num_q_heads = 8`

那么：

- `MHA`
  - `num_kv_heads = 8`
  - 每个 `q head` 都有自己独立的 `kv head`

- `GQA`
  - `num_kv_heads = 2`
  - 例如 `4` 个 `q heads` 共享一个 `kv head`

- `MQA`
  - `num_kv_heads = 1`
  - 所有 `q heads` 共用同一个 `kv head`

所以从关系上看：

- `MQA` 可以看成 `GQA` 的一个极端特例

从系统角度看，最重要的差别是：

- `num_kv_heads` 越少
- KV cache 越小
- decode 时读 KV 的带宽压力越小

所以：

- `MHA` 最贵
- `GQA` 折中
- `MQA` 最省

## MLA 和 sparse attention 不是一回事

这是这一节里最容易混的点。

很多人会觉得：

- 长 context 下真正重要的历史 token 只有一部分
- 所以 `MLA` 也是在利用这种 attention 稀疏性

这个说法不准确。

更准确地说：

- `sparse attention` 利用的是 `连接稀疏性`
- `MLA` 利用的是 `表示冗余`

### 什么叫 sparse attention

`sparse attention` 改的是：

- 当前 query 到底看哪些 key

它的典型做法是：

- 只看局部窗口
- 只看某些 block
- 只看 top-k 的连接

所以 sparse 路线的本质是：

- attention matrix 里的很多位置根本不参与计算
- 也就是 `q -> k` 的连接图变稀了

它更像：

- 少看一些 token

### 什么叫 MLA

`MLA` 不主要改：

- 当前 query 看哪些 key

它主要改的是：

- 历史 KV 用什么方式存下来

也就是：

- 不是直接缓存完整 `K / V`
- 而是先压成更小的 latent 表示
- 需要 attention 时，再从 latent 恢复出参与 attention 的 `K / V`

所以 `MLA` 更像：

- 同样这些 token 还都在
- 但它们不是用原始大 KV 存
- 而是用更紧凑的 latent state 存

### 一句话区分

可以直接记成：

- `sparse attention`：改 “看谁”
- `MLA`：改 “怎么存”

或者：

- `sparse attention`：`prune connections`
- `MLA`：`compress state`

### 为什么这很重要

因为它们优化的系统瓶颈也不一样。

`sparse attention` 更偏：

- attention FLOPs
- long-context attention 访问量
- attention matrix 规模

`MLA` 更偏：

- KV cache 大小
- decode memory bandwidth
- 长期状态存储成本

所以不能简单说：

- “MLA 也是 sparse attention”

更准确的说法应该是：

- `MLA` 是 `latent KV compression` 路线
- 不是 `sparse attention` 路线

## 可以用哪张表快速记

| variant | 改了什么 | K/V 怎么处理 | 更像哪类问题 |
| --- | --- | --- | --- |
| `MHA` | 不共享 KV | 每个 q head 都有独立 KV | 标准基线 |
| `GQA` | 部分共享 KV | 多个 q heads 共享一个 KV group | KV cache 折中 |
| `MQA` | 全部共享 KV | 所有 q heads 共用一组 KV | 极限省 KV cache |
| `MLA` | 改 KV 表示 | 先压 latent，再恢复 K/V | latent KV compression |
| `sparse attention` | 改 attention 连接图 | 不是所有 q 都看所有 k | sparse connectivity |

## 运行

```bash
python experiments/01_model_basics/attention_variants/compare_attention_variants.py
```

## 当前设置

为了可打印，这里统一固定一个很小的输入：

- `seq_len = 4`
- `hidden_size = 8`

并且所有变体都共用同一组输入 token ids 和 embedding table。

## 现在最该记住什么

1. 如果你的问题是“不同 attention 变体到底改了哪里”，先看 `Q heads` 和 `KV heads` 的关系。
2. `MHA / MQA / GQA` 是一组，它们主要在改 `KV sharing`。
3. `MLA` 是另一组，它主要在改 `KV representation`。
4. `MLA` 不是 sparse attention；前者是 `compress state`，后者是 `prune connections`。
5. 这份脚本的定位是建立概念边界，不是模拟真实大模型的全部实现细节。
