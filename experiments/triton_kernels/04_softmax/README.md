# 04 Triton Row-wise Softmax

这个目录对应：

- `row_softmax.py`

默认你已经看过 `01` 到 `03`。
这里开始第一次出现真正的 row-wise reduction 语法。

## 这个 kernel 在算什么

对一行输入：

```text
x = [x_1, x_2, ..., x_N]
```

softmax 的稳定写法是：

```text
m = max_i x_i
e_i = exp(x_i - m)
s = sum_i e_i
y_i = e_i / s
```

所以它不是简单 elementwise，而是：

- 先做行内 reduction
- 再做 elementwise normalize

## 结合代码看执行流程

kernel 的核心是：

```python
x = tl.load(x_row_ptr + cols_offsets, mask=mask, other=-float("inf"))
row_max = tl.max(x, axis=0)
exp_x = tl.exp(x - row_max)
row_sum = tl.sum(exp_x, axis=0)
y = exp_x / row_sum
tl.store(y_row_ptr + cols_offsets, y, mask=mask)
```

这里第一次出现了 Triton 里非常核心的一组 reduction 语法。

## 新增语法 1：`tl.max(..., axis=0)`

```python
row_max = tl.max(x, axis=0)
```

`x` 是当前这一行被读进来的一个向量。

`tl.max(x, axis=0)` 的意思是：

- 沿这个向量做 reduction
- 最后得到一个标量最大值

你可以把它理解成：

```text
max(x[0], x[1], ..., x[N-1])
```

## 新增语法 2：`tl.sum(..., axis=0)`

```python
row_sum = tl.sum(exp_x, axis=0)
```

这和 `tl.max` 类似，也是沿当前向量做 reduction。

只是这次不是求最大值，而是求和。

## 新增语法 3：`tl.exp`

```python
exp_x = tl.exp(x - row_max)
```

这表示逐元素指数函数。

后面只要看到：

- `tl.exp`
- `tl.log`
- `tl.sin`
- `tl.cos`

都可以先理解成 Triton 提供的逐元素数学函数。

## 新增语法 4：`other=-float("inf")`

这里的 load 写法有个很重要的小细节：

```python
tl.load(..., mask=mask, other=-float("inf"))
```

为什么不是 `other=0.0`？

因为这里后面立刻要做：

```python
tl.max(x, axis=0)
```

如果越界位置填 `0.0`，那它可能错误地参与最大值比较。

填成 `-inf` 才能保证：

- 越界位置不会影响 max reduction

这是数值稳定里一个很常见的技巧。

## 这份代码里新增的 Triton 语法

相对前面目录，这里新增的是：

- `tl.max(..., axis=0)`
  - 做 reduction 最大值
- `tl.sum(..., axis=0)`
  - 做 reduction 求和
- `tl.exp(...)`
  - 逐元素指数函数
- `other=-float("inf")`
  - 为 max reduction 提供合适的无效填充值

## 运行

```bash
python row_softmax.py
```

## 现在最值得记住的点

1. softmax 是第一次让你看到 Triton 的 row-wise reduction 写法。
2. `tl.max` 和 `tl.sum` 是后面很多 kernel 的基础。
3. 越界位置填什么值，不只是边界问题，也是数值正确性问题。
