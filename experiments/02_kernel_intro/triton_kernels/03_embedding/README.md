# 03 Embedding and Gather in Triton

这个目录对应：

- `row_gather.py`

默认你已经看过 `01_elementwise` 和 `02_scatter`。
这里继续只讲新增语法，不重复前面讲过的基础部分。

## 这个 kernel 在算什么

embedding lookup 本质上就是：

```text
out[row, :] = table[ids[row], :]
```

也就是：

- 先读出一个 token id
- 再把 embedding table 里对应那一整行搬出来

所以它不是 matmul，而是 gather。

## 结合代码看执行流程

先看 kernel：

```python
row = tl.program_id(0)
token_id = tl.load(ids_ptr + row)
offsets = tl.arange(0, BLOCK_SIZE)
mask = offsets < dim

src_ptr = table_ptr + token_id * table_stride + offsets
dst_ptr = out_ptr + row * out_stride + offsets

values = tl.load(src_ptr, mask=mask, other=0.0)
tl.store(dst_ptr, values, mask=mask)
```

这个结构和 `01` 很像，都是：

- 一维 grid
- 一段连续列 tile
- `load -> compute -> store`

区别在于这里的源地址不是固定连续的，而是先由：

```python
token_id = tl.load(ids_ptr + row)
```

决定读哪一行。

## 新增语法 1：stride 参数

这里第一次显式把 stride 传进 kernel：

```python
def row_gather_kernel(table_ptr, ids_ptr, out_ptr, batch, dim, table_stride, out_stride, BLOCK_SIZE: tl.constexpr):
```

然后地址计算写成：

```python
src_ptr = table_ptr + token_id * table_stride + offsets
dst_ptr = out_ptr + row * out_stride + offsets
```

这和直接写：

```text
base + row * dim + col
```

的区别是：

- 这里不假设张量一定是某种固定布局
- 而是显式告诉 kernel：
  - 一行和下一行之间隔多少元素

以后只要看到 `stride_*` 参数，就应该意识到：

- 这是在手动做张量地址映射

## 新增语法 2：host 侧 `triton.next_power_of_2`

这里第一次出现：

```python
block_size = triton.next_power_of_2(dim)
```

意思是：

- 找到不小于 `dim` 的最小 2 的幂

例如：

```text
dim = 256 -> block_size = 256
dim = 300 -> block_size = 512
```

这样做的目的通常是：

- 让 `BLOCK_SIZE` 更适合 Triton 的向量化和 reduction 写法

后面的 softmax、layernorm 里你也会再次看到这种 host 侧准备方式。

## 这份代码里新增的 Triton 语法

相对前面目录，这里新增的是：

- 显式 stride 参数
  - `table_stride`
  - `out_stride`
- 用 stride 做地址计算
  - `base + row * stride + offsets`
- `triton.next_power_of_2`
  - host 侧准备合适的 `BLOCK_SIZE`

## 运行

```bash
python row_gather.py
GATHER_ID_MODE=repeated python row_gather.py
```

## 现在最值得记住的点

1. gather 的关键不是算术，而是“先读索引，再决定去哪里取数据”。
2. stride 参数意味着 kernel 在自己做张量地址映射。
3. `triton.next_power_of_2` 是 Triton 里很常见的 host 侧参数准备方式。
