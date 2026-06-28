# 09 Attention: A Minimal Triton Teaching Kernel

这个目录对应：

- `triton_attention.py`

默认你已经看过 `01` 到 `08`。
这里开始第一次出现“在 kernel 内显式保留一整行中间结果”的写法。

## 这个 kernel 在算什么

最基础的单头 attention：

```text
scores = Q K^T / sqrt(head_dim)
probs  = softmax(scores)
out    = probs V
```

这版代码故意写得非常直白：

- 一个 program 负责一个 query row
- program 内自己算完整一行 `scores`
- program 内自己做 softmax
- program 内自己聚合 `V`

## 结合代码看执行流程

先看中间结果：

```python
scores = tl.zeros([BLOCK_SEQ], dtype=tl.float32)
```

然后第一段循环是：

```python
for key_row in range(0, BLOCK_SEQ):
    ...
    dot = tl.sum(q * k, axis=0)
    scores = tl.where(offs_seq == key_row, dot * scale, scores)
```

也就是：

- 逐个 key row 计算点积
- 把结果写进 `scores` 这一整行

后面 softmax：

```python
row_max = tl.max(scores, axis=0)
probs = tl.exp(scores - row_max)
row_sum = tl.sum(probs, axis=0)
probs = probs / row_sum
```

最后再做加权求和：

```python
acc = tl.zeros([BLOCK_D], dtype=tl.float32)
for key_row in range(0, BLOCK_SEQ):
    ...
    weight = tl.sum(tl.where(offs_seq == key_row, probs, 0.0), axis=0)
    acc += weight * v
```

## 新增语法 1：`tl.where`

这里第一次出现：

```python
scores = tl.where(offs_seq == key_row, dot * scale, scores)
```

`tl.where(cond, a, b)` 的意思和你熟悉的条件选择类似：

```text
如果 cond 为真，就取 a，否则取 b
```

这里的用途是：

- 只把当前 `key_row` 对应的位置更新掉
- 其他位置保持原来的 `scores`

后面这一句也是同样的思路：

```python
weight = tl.sum(tl.where(offs_seq == key_row, probs, 0.0), axis=0)
```

## 新增语法 2：在 kernel 内物化整行中间结果

前面虽然也用过 `tl.zeros`，但这里第一次明显拿它做：

```python
scores = tl.zeros([BLOCK_SEQ], dtype=tl.float32)
```

也就是：

- 在当前 program 内保留完整一行分数

这和前面那些“读进来就立刻 reduction / 立刻写回”的 kernel 很不一样。

它说明：

- Triton kernel 也可以显式维护比较长的中间向量

当然，这也意味着：

- 更高的局部状态开销

## 新增语法 3：compile-time 尺寸驱动的局部向量

这里同时出现了两个局部向量：

```python
scores = tl.zeros([BLOCK_SEQ], dtype=tl.float32)
acc = tl.zeros([BLOCK_D], dtype=tl.float32)
```

它们的长度都由编译期常量控制：

- `BLOCK_SEQ`
- `BLOCK_D`

也就是说，这里已经开始进入：

- “kernel 内局部状态尺寸由 tile 大小决定”

的风格了。

## 这份代码里新增的 Triton 语法

相对前面目录，这里新增的是：

- `tl.where(cond, a, b)`
  - 条件选择
- 在 kernel 内显式保留一整行中间结果
- 用编译期常量决定局部向量长度

## 运行

```bash
python triton_attention.py
```

## 现在最值得记住的点

1. 这版 attention 的重点不是高性能，而是把 `QK^T -> softmax -> PV` 明确地直译成 Triton 代码。
2. `tl.where` 是这里最值得新记住的语法。
3. 从这个目录开始，kernel 里的局部中间状态明显变复杂了。
