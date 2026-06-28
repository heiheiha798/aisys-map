# 12 Flash Attention in Triton

这个目录对应：

- `flash_attention.py`

默认你已经看过前面的所有目录，尤其是：

- `07_online_softmax`
- `09_attention`
- `11_gemm`

因为这个文件把那三条线组合到了一起：

- tiled scan
- online softmax 状态
- tile 级矩阵乘加

## 这个 kernel 在算什么

它仍然是在算 attention：

```text
scores = Q K^T / sqrt(d)
probs  = softmax(scores)
out    = probs V
```

但和 `09_attention` 不同，这里不再显式保存整行 `scores`，而是：

- 一边扫 `K/V` tile
- 一边在线更新 softmax 状态
- 一边累计输出

## 结合代码看执行流程

先看局部状态初始化：

```python
m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
```

这三个量分别表示：

- `m_i`
  - 每个 query row 当前看到的最大分数
- `l_i`
  - 在当前坐标系下的指数和
- `acc`
  - 当前累计出来的输出部分和

然后每次扫一个 `K/V` tile：

```python
scores = tl.dot(q, tl.trans(k)) * scale
m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
p = tl.exp(scores - m_ij[:, None])
alpha = tl.exp(m_i - m_ij)

acc = acc * alpha[:, None] + tl.dot(p, v)
l_i = l_i * alpha + tl.sum(p, axis=1)
m_i = m_ij
```

最后再统一归一化：

```python
out = acc / l_i[:, None]
```

## 新增语法 1：`tl.full`

这里第一次出现：

```python
m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
```

`tl.full(shape, value, dtype=...)` 的意思是：

- 创建一个指定形状的局部张量
- 每个位置都初始化成同一个值

和 `tl.zeros` 相比，它的区别只是：

- 初值不是 0
- 而是任意你指定的常量

这里用 `-inf` 是因为：

- online softmax 一开始还没看到任何分数
- 初始最大值就应该是负无穷

## 新增语法 2：`tl.trans`

这里第一次出现：

```python
tl.trans(k)
```

因为 `q` 的 shape 是：

```text
[BLOCK_M, BLOCK_D]
```

而 `k` 当前读进来时是：

```text
[BLOCK_N, BLOCK_D]
```

要做：

```text
q @ k^T
```

就需要先转置：

```python
scores = tl.dot(q, tl.trans(k))
```

所以 `tl.trans` 可以理解成：

- 对 tile 做转置

## 新增语法 3：`axis=1` 的 reduction

前面大多数 reduction 都写成：

```python
tl.max(..., axis=0)
tl.sum(..., axis=0)
```

这里第一次比较明确地出现：

```python
tl.max(scores, axis=1)
tl.sum(p, axis=1)
```

因为现在 `scores` 和 `p` 都已经是二维矩阵了。

所以：

- `axis=1`
  - 表示沿每一行做 reduction

这和前面一维向量的 `axis=0` 是不同层次的事情。

## 新增语法 4：向量和矩阵状态一起维护

这个文件最值得注意的不是某一个 API，而是 program 内同时维护了三种不同形状的状态：

- 一维向量 `m_i`
- 一维向量 `l_i`
- 二维矩阵 `acc`

这说明 FlashAttention 这类 kernel 的代码复杂度明显高于前面的例子：

- 它不是只有一个简单 reduction
- 也不是只有一个简单 accumulator
- 而是多种状态一起更新

## 这份代码里新增的 Triton 语法

相对前面目录，这里新增的是：

- `tl.full(shape, value, dtype=...)`
- `tl.trans(x)`
- 二维矩阵上的 `axis=1` reduction
- 在同一个 program 内同时维护多个不同形状的状态张量

## 运行

```bash
python flash_attention.py
```

## 现在最值得记住的点

1. `flash_attention.py` 本质上是在把 `07_online_softmax`、`09_attention`、`11_gemm` 三个目录里的东西组合起来。
2. `tl.full`、`tl.trans`、`axis=1` reduction 是这个文件最重要的新语法。
3. 到这里你应该能明显感觉到：Triton 代码的难点已经不只是单个 API，而是怎么组织 program 内的多种状态。
