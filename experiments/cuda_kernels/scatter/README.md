# Scatter and Index-Add

这个目录放一个最基础的 `scatter / index_add` CUDA kernel 实验：

- `index_add_rows.cu`

它的重点是：

- 理解 `scatter` 和 `gather` 的差别
- 理解不规则写为什么通常比不规则读更麻烦
- 理解原子操作 `atomicAdd` 为什么会把“正确性”和“性能”绑在一起

## 先说清楚：scatter 是什么

如果 `gather` 是：

- 给我一组索引
- 我去源表里把这些位置读出来

那 `scatter` 就更像反过来：

- 给我一组索引
- 我把数据写到这些索引对应的位置

最简单的行级版本可以写成：

```text
dst[ids[i], :] += src[i, :]
```

这里：

- `src[i, :]`
  - 是第 `i` 行输入
- `ids[i]`
  - 决定这一行应该加到 `dst` 的哪一行

所以这类操作也常叫：

- `index_add`
- `scatter_add`

## 为什么 scatter 比 gather 更麻烦

`gather` 的主要问题是：

- 不规则读
- cache 命中不稳定

但 `scatter` 除了不规则写，还会多一个大问题：

- **多个线程可能同时写同一个目标位置**

比如：

```text
ids[0] = 17
ids[1] = 17
ids[2] = 17
```

那三行输入都会加到：

```text
dst[17, :]
```

这时候如果你直接普通写：

```text
dst[...] += src[...]
```

结果就可能错，因为多个线程会互相覆盖。

所以这类 kernel 最常见的第一步实现就是：

- 使用 `atomicAdd`

## 当前这个教学版 kernel 在做什么

当前 `index_add_rows.cu` 的结构是：

- 一个 block 负责一行 `src`
- block 先读取 `ids[src_row]`
- 再把 `src[src_row, :]` 原子累加到 `dst[ids[src_row], :]`

也就是：

```text
dst_row = ids[src_row]
dst[dst_row, :] += src[src_row, :]
```

因为这里是加法归并，所以用：

- `atomicAdd`

来保证多个 block 同时写同一行时仍然正确。

## 编译

```bash
make
```

## 运行

```bash
./index_add_rows
```

## 现在最值得记住的点

1. `gather` 主要是“不规则读”问题。
2. `scatter / index_add` 主要是“不规则写 + 写冲突”问题。
3. 这类 kernel 的第一性问题通常不是算力，而是：
   - 原子冲突
   - memory contention
   - cache / 写回行为
