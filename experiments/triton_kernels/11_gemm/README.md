# 11 GEMM in Triton

这个目录对应：

- `triton_gemm.py`

默认你已经看过 `01` 到 `10`。
这里第一次正式进入二维 tile matmul 的 Triton 语法。

## 这个 kernel 在算什么

标准矩阵乘法：

```text
C[m, n] = A[m, k] @ B[k, n]
```

这版实现是一个最小 tiled matmul：

- 一个 program 负责一个 `BLOCK_M x BLOCK_N` 输出 tile
- 沿着 `K` 维分块累加

## 结合代码看执行流程

先看 program 定位：

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
```

说明 grid 是二维：

- 一个维度枚举输出 tile 的行块
- 一个维度枚举输出 tile 的列块

然后是三组索引：

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)
```

接着初始化 accumulator：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

然后沿 `K` 维循环：

```python
for k_start in range(0, k, BLOCK_K):
    ...
    a = tl.load(...)
    b = tl.load(...)
    acc = tl.dot(a, b, acc=acc, input_precision="ieee", out_dtype=tl.float32)
```

最后把整个 tile 写回：

```python
tl.store(c_ptrs, acc, mask=c_mask)
```

## 新增语法 1：二维 tile 的二维索引张量

这里第一次明显出现这种写法：

```python
offs_m[:, None]
offs_n[None, :]
```

例如：

```python
a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
```

它的意义是：

- `offs_m[:, None]`
  - 把一维行索引变成列向量
- `offs_k[None, :]`
  - 把一维列索引变成行向量

这样两者组合起来，就能得到一个二维 tile 的坐标网格。

这是 Triton matmul 代码里最核心的索引写法之一。

## 新增语法 2：二维 `tl.zeros`

这里第一次出现二维局部 accumulator：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

前面虽然已经见过一维 `tl.zeros`，但这里的重点是：

- 局部状态不再是一个向量
- 而是一个二维 tile

也就是：

- 当前 program 在寄存器里维护整个输出 tile 的部分和

## 新增语法 3：`tl.dot`

这一行是本目录最重要的新语法：

```python
acc = tl.dot(a, b, acc=acc, input_precision="ieee", out_dtype=tl.float32)
```

`tl.dot(a, b, ...)` 可以理解成：

- 对两个 tile 做矩阵乘加

这里的：

- `acc=acc`
  - 表示把结果累加到已有 accumulator 上
- `input_precision="ieee"`
  - 用更保守的输入精度路径
- `out_dtype=tl.float32`
  - 输出累加保持在 `float32`

你可以把这一步理解成：

```text
acc += A_tile @ B_tile
```

## 新增语法 4：二维 mask

最后写回时的 mask 也是二维的：

```python
c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
```

前面虽然见过复合 mask，但这里第一次出现：

- 行方向一组条件
- 列方向一组条件
- 组合成二维 tile 的有效区域

## 这份代码里新增的 Triton 语法

相对前面目录，这里新增的是：

- `offs_m[:, None]` / `offs_n[None, :]`
  - 用广播方式构造二维 tile 坐标
- 二维 `tl.zeros((M, N), ...)`
- `tl.dot(...)`
  - tile 级矩阵乘加
- 二维 mask

## 运行

```bash
python triton_gemm.py
```

## 现在最值得记住的点

1. 看到 `[:, None]` 和 `[None, :]`，就要意识到代码开始在构造二维 tile 坐标网格了。
2. `tl.dot` 是 Triton matmul 代码里最核心的语法之一。
3. 这个目录最重要的不是性能，而是看懂“一个 program 负责一个输出 tile”到底如何落在代码里。
