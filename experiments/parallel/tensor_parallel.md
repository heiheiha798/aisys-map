# Tensor Parallel

这份文档只解释推理里最常见、最值得先理解的 `TP`。

## `TP` 到底在解决什么问题

`TP` 解决的是：

- 一个 layer 太大，单卡放不下
- 或者单卡虽然放得下，但单卡算太慢，希望把一个 layer 的计算拆到多卡

它的核心思想不是“把不同请求分给不同卡”，而是：

- **把同一个 layer 的张量切开，让多张卡一起算同一个请求**

这就是它和 data parallel 的根本区别。

## 最常见的两种切法

### 1. Column Parallel

把权重按输出维切开。

如果一个 linear 层是：

- 输入 `x` shape = `[batch, in_features]`
- 权重 `W` shape = `[out_features, in_features]`

那么 column parallel 会把 `W` 切成：

- `W0` shape = `[out_features/2, in_features]`
- `W1` shape = `[out_features/2, in_features]`

两个 rank 分别算：

- `y0 = x @ W0^T`
- `y1 = x @ W1^T`

最后把结果在输出维拼起来：

- `y = concat([y0, y1], dim=-1)`

所以 column parallel 的特点是：

- 每个 rank 都拿完整输入 `x`
- 每个 rank 只负责输出的一部分
- 最后通常需要 `all-gather` 或逻辑拼接

### 2. Row Parallel

把权重按输入维切开。

还是同一个 linear 层：

- 权重 `W` shape = `[out_features, in_features]`

row parallel 会把 `W` 切成：

- `W0` shape = `[out_features, in_features/2]`
- `W1` shape = `[out_features, in_features/2]`

同时输入也要跟着切：

- `x0` shape = `[batch, in_features/2]`
- `x1` shape = `[batch, in_features/2]`

两个 rank 分别算部分和：

- `p0 = x0 @ W0^T`
- `p1 = x1 @ W1^T`

最后求和：

- `y = p0 + p1`

所以 row parallel 的特点是：

- 每个 rank 只看输入的一部分
- 每个 rank 都产出完整输出 shape 的“部分和”
- 最后通常需要 `all-reduce`

## 为什么 transformer 里两种切法都会出现

因为不同 linear 层适合的通信形态不同。

你之后看真实框架时，经常会看到：

- 某些 projection 用 column parallel
- 某些 projection 用 row parallel

这是为了让：

- 上一层的输出形态
- 下一层的输入形态
- 中间的通信次数

尽量协调。

## 这个实验脚本在做什么

[`tensor_parallel.py`](tensor_parallel.py) 做了两个最小整数实验：

1. `column_parallel_demo()`
   用一个小矩阵展示“切输出维，再拼起来”
2. `row_parallel_demo()`
   用一个小矩阵展示“切输入维，再把部分和加起来”

这两个实验最重要的不是代码，而是你要亲眼看到：

- 切开之后每个 rank 各自算了什么
- 为什么合并结果还能和全量 linear 完全一致

## 你读代码时最该问的问题

1. 为什么 column parallel 里输入不用切，但 row parallel 里输入必须跟着切？
2. 为什么 column parallel 最后更像拼接，row parallel 最后更像求和？
3. 在真实多卡系统里，这两者分别对应什么通信原语？
4. 为什么 dense model inference 里 `TP` 是最应该优先理解的并行方式？
