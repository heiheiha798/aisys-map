# Chunked Prefill

这份说明对应：

- [`chunked_prefill.py`](chunked_prefill.py)

它只解释一件事：

- `chunked prefill` 到底是怎么实现的

## 一句话定义

`chunked prefill` 不是新的 attention 公式，而是：

- 不把整段 prompt 一次性送进模型
- 而是把长 prompt 切成多个连续 chunk
- 按顺序多次 forward
- 每次都复用前面 chunk 产生的 `past_key_values`

所以它本质上是：

- `prefill` 的一种分段执行方式

不是新的 attention 变体。

## 先看普通 full prefill

假设一条 prompt 的总长度是：

```text
46 tokens
```

如果做普通 `full prefill`，就是一次直接送进去：

```text
input_ids shape      = [1, 46]
attention_mask shape = [1, 46]
past_key_values      = None
```

一次 forward 结束后：

- 这 46 个 token 的前向都做完了
- 第一层 KV cache 长度直接变成 `46`

也就是：

```text
K cache: [1, num_kv_heads, 46, head_dim]
V cache: [1, num_kv_heads, 46, head_dim]
```

## chunked prefill 的基本做法

如果：

```text
chunk_size = 8
```

那长度 `46` 的 prompt 会被切成：

```text
[0:8)
[8:16)
[16:24)
[24:32)
[32:40)
[40:46)
```

然后这些 chunk 会按顺序一个个送进模型。

关键点是：

- 每次只输入“新的那一段 token”
- 但 `past_key_values` 会保留前面所有 chunk 的历史
- `attention_mask` 会扩成“当前总长度”

## 第 1 个 chunk 是怎么跑的

第一段：

```text
[0:8)
```

这时没有任何历史 cache，所以它和普通小 prefill 没区别：

```text
chunk_input_ids shape      = [1, 8]
chunk_attention_mask shape = [1, 8]
past_key_values            = None
```

跑完以后：

```text
K cache: [1, num_kv_heads, 8, head_dim]
V cache: [1, num_kv_heads, 8, head_dim]
```

也就是：

- 第一个 chunk 的 8 个 token 已经变成历史 KV

## 第 2 个 chunk 是怎么跑的

第二段：

```text
[8:16)
```

这里最容易误解的地方是：

- 不是把前 16 个 token 全部重新送进去

我们先把 token 编号写死。

假设整条 prompt 的 token 是：

```text
t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15
```

现在：

- 第 1 个 chunk 已经处理完了
- 所以 `t0 ... t7` 的 K/V 已经在 cache 里

这时第 2 个 chunk 要处理的是：

```text
t8  t9  t10 t11 t12 t13 t14 t15
```

所以这一轮真正送进模型的 `input_ids` 只有这 8 个新 token：

```text
chunk_input_ids = [t8, t9, t10, t11, t12, t13, t14, t15]
shape = [1, 8]
```

这一步非常关键：

- 这轮 forward 的输入只有新 chunk
- 不是把 `t0 ... t15` 这 16 个 token 重新算一遍

但是这 8 个新 token 在做 attention 时，并不是只能看彼此。

例如：

- `t8` 应该能看到：
  - `t0 ... t7`
  - 以及它自己 `t8`
- `t15` 应该能看到：
  - `t0 ... t7`
  - `t8 ... t14`
  - 以及它自己 `t15`

也就是说，这一轮虽然只输入了：

```text
8 个新 token
```

但这 8 个新 token 的真实可见上下文其实已经是：

```text
16 个 token
```

这就是为什么此时必须同时满足两件事：

```text
chunk_input_ids shape      = [1, 8]
past_key_values length     = 8
chunk_attention_mask shape = [1, 16]
```

三者分别在表达不同的东西：

- `chunk_input_ids = [1, 8]`
  - 这一轮新进入模型的 token 只有 8 个

- `past_key_values length = 8`
  - 前 8 个 token 的 K/V 已经作为历史缓存存在

- `chunk_attention_mask = [1, 16]`
  - 当前这轮 attention 的总有效上下文长度已经是 16

如果你把这一步想成一个更具体的 attention 子问题，其实是在算：

```text
新的 query: t8 ... t15
去看：
  历史 cache 里的 key/value: t0 ... t7
  再加上当前 chunk 自己的 key/value: t8 ... t15
```

所以这轮 forward 并不是：

- “重新做一遍前 16 个 token 的 full prefill”

而是：

- “只对后 8 个 token 做新前向”
- “但让它们能看到前 16 个 token 的完整上下文”

这时就能更清楚地理解为什么：

- `input_ids` 只需要 `[1, 8]`
- 可是 `attention_mask` 必须已经是 `[1, 16]`

跑完这一轮之后，新的 `t8 ... t15` 的 K/V 也会被 append 到旧 cache 后面。

于是：

```text
K cache length: 8 -> 16
V cache length: 8 -> 16
```

所以第 2 个 chunk 跑完之后，cache 的时间维才真正补齐到 `16`。

## 后续 chunk 都是同一个模式

第 3 段：

```text
[16:24)
```

对应：

```text
chunk_input_ids shape      = [1, 8]
chunk_attention_mask shape = [1, 24]
past_key_values length     = 16
```

跑完后：

```text
cache length = 24
```

第 4 段：

```text
[24:32)
```

对应：

```text
chunk_input_ids shape      = [1, 8]
chunk_attention_mask shape = [1, 32]
past_key_values length     = 24
```

跑完后：

```text
cache length = 32
```

一直到最后一段：

```text
[40:46)
```

对应：

```text
chunk_input_ids shape      = [1, 6]
chunk_attention_mask shape = [1, 46]
past_key_values length     = 40
```

跑完后：

```text
cache length = 46
```

## 为什么 attention_mask 要按“总长度”增长

这是 chunked prefill 里最关键的实现点之一。

很多人第一反应会写成：

- 当前 chunk 输入 8 个 token
- 所以 `attention_mask` 也写 `[1, 8]`

这是不对的。

因为当前 chunk 里的 token 不是只看当前 chunk。

例如第 2 个 chunk 里的 token，应该能看到：

- 第 1 个 chunk 的历史 token
- 当前 chunk 内它前面的 token

所以对第 2 个 chunk 来说，真正的有效上下文已经是：

```text
16 tokens
```

因此 `attention_mask` 必须是：

```text
[1, 16]
```

同理，第 3 个 chunk 必须是 `[1, 24]`，不是 `[1, 8]`。

一句话说：

- `chunk_input_ids` 只放“新 token”
- `attention_mask` 表示“当前总上下文长度”

## 为什么 chunked prefill 和 full prefill 应该对齐

只要实现正确，`chunked prefill` 理论上应该和 `full prefill` 对齐。

原因是：

- 对任意一个 token，它该看到的历史并没有变
- 只是这些历史不是在同一轮 forward 一次性处理完
- 而是前面已经通过 `past_key_values` 存起来了

所以：

- `full prefill` 是一次性处理全部上下文
- `chunked prefill` 是分多次处理，但每次都带上历史 cache

两者的逻辑可见范围应该一致。

所以这里真正该记住的是：

- `chunked prefill` 虽然拆成了多轮 forward
- 但每个新 token 看到的历史范围不应该变
- 因此它在逻辑上应该和 `full prefill` 对齐

当前的 [`chunked_prefill.py`](chunked_prefill.py) 只保留最小实现，
重点是把这个分块执行过程写清楚，而不是在脚本里做额外对照。

## 为什么需要 chunked prefill

它的动机不是：

- attention 数学不够好

而是：

- 长 prompt 一次 full prefill 会占住 GPU 很久
- 这会拖慢系统里正在等待 decode 的请求

所以系统上会想：

- 不要让一条长 prompt 一次性独占太久
- 把它拆成多个 chunk
- 在 chunk 和 chunk 之间，调度器就有机会插入别的工作

例如：

- 先做长 prompt 的 chunk 1
- 中间插几步 decode
- 再回来做 chunk 2

所以 `chunked prefill` 的系统意义是：

- 给调度器制造更细粒度的切分点

它更像：

- 一种 runtime / scheduling 友好的 prefill 执行策略

而不是新的模型结构。

## 这一页最该记住什么

1. `chunked prefill` 不是新 attention，而是把长 prefill 拆成多次 forward。
2. 每次只输入当前 chunk 的新 token，但 `past_key_values` 会保留前面 chunk 的历史。
3. `chunk_input_ids` 是“新 token 长度”，`attention_mask` 是“当前总长度”，这两者不能混。
4. 跑完最后一个 chunk 后，最终 KV cache 长度应该和 full prefill 一样。
5. 它的系统价值在于：给长 prompt 的 prefill 提供更细粒度的调度切分点。
