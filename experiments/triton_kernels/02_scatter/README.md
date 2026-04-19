# 02 Triton Scatter / Index-Add

这个目录对应：

- `index_add_rows.py`

默认你已经看过 `01_elementwise`，所以下面不再重复解释：

- `@triton.jit`
- `tl.program_id`
- `tl.arange`
- `tl.load`
- `tl.store`
- `mask`
- `kernel[grid](...)`

这一节只讲相对 `01` 新出现的 Triton 语法和写法。

## 这个 kernel 在算什么

最小行级 scatter add 可以写成：

```text
dst[ids[i], :] += src[i, :]
```

也就是：

- 输入有很多行 `src[i, :]`
- `ids[i]` 告诉我们每一行要加到 `dst` 的哪一行

如果很多个 `ids[i]` 指向同一个目标行，就会发生写冲突。

## 结合代码看执行流程

先看 launch：

```python
grid = (src_rows, triton.cdiv(dim, block_d))
```

这里和 `01` 的一维 grid 不同，它变成了二维：

- 第 0 维：处理第几行 `src`
- 第 1 维：处理这一行里的第几个列块

再看 kernel 开头：

```python
src_row = tl.program_id(0)
d_block = tl.program_id(1)
```

这说明一个 program 现在不再只对应“一段连续的一维元素”，而是对应：

- 固定一行 `src_row`
- 固定一个列块 `d_block`

## 新增语法 1：二维 `program_id`

这是第一次出现：

```python
tl.program_id(0)
tl.program_id(1)
```

它表示 grid 有多个维度。

在这个例子里可以理解成：

- `program_id(0)`
  - 第几行
- `program_id(1)`
  - 这一行里的第几个列 tile

这也是后面很多二维 kernel 的共同写法。

## 新增语法 2：类型转换 `.to(...)`

这行是新的：

```python
dst_row = tl.load(ids_ptr + src_row, mask=valid_src_row, other=0).to(tl.int64)
```

`tl.load` 读出来之后，可以继续：

```python
.to(tl.int64)
```

意思是把这个值转换成 Triton 里的 `int64`。

为什么这里要转？

因为后面要拿它参与地址计算：

```python
dst_offsets = dst_row * dim + d_offsets
```

这种时候显式转成整数类型更稳妥，也更容易读懂。

## 新增语法 3：复合布尔 mask

这里第一次出现了更复杂的 mask：

```python
valid_write = (valid_src_row & (dst_row >= 0) & (dst_row < dst_rows) &
               (d_offsets < dim))
```

这说明 Triton 里的 mask 不一定只是一个简单的“有没有越界”。

它也可以同时表达多个条件：

- 当前行是不是有效
- `dst_row` 有没有越界
- 当前列 offset 有没有越界

后面很多真实 kernel 都会用这种“复合 mask”。

## 新增语法 4：`tl.atomic_add`

这一行是这个目录最重要的新语法：

```python
tl.atomic_add(dst_ptr + dst_offsets, src_vals, mask=valid_write)
```

它表示：

- 对同一个地址做原子加

为什么必须用原子加？

因为多个 program 可能同时写同一个 `dst_row`。

如果不用原子加，就会有竞争，结果不对。

所以这个目录第一次让你看到：

- Triton 不只是能做普通读写
- 也能表达带冲突的原子更新

## 这份代码里新增的 Triton 语法

相对 `01`，这个目录新增了：

- 二维 `program_id`
  - `tl.program_id(0)` / `tl.program_id(1)`
- `.to(tl.int64)`
  - 显式类型转换
- 复合布尔 mask
  - 把多种边界条件合并到一起
- `tl.atomic_add`
  - 原子加

## 运行

```bash
python index_add_rows.py
```

## 现在最值得记住的点

1. 当一个 kernel 需要同时处理“行”和“列块”时，grid 往往会变成二维。
2. scatter 和 gather 的本质区别之一，就是 scatter 会碰到写冲突。
3. `tl.atomic_add` 是 Triton 里处理这类写冲突的最直接方法。
