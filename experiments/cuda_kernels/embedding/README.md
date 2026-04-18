# Embedding and Gather

这个目录放一个最基础的 `embedding / gather` CUDA kernel 实验：

- `row_gather.cu`

它的重点不是复杂数学，而是：

- 为什么 `embedding lookup` 本质上更像 gather，而不是 GEMM
- 为什么这类 kernel 往往更容易暴露不规则访存问题
- 为什么它们通常是 memory-bound，而不是 compute-bound

## 先说清楚：embedding 到底在做什么

可以先把 embedding table 理解成一个大矩阵：

```text
table[vocab, dim]
```

这里：

- `vocab`
  - 一共有多少个 token / id
- `dim`
  - 每个 token 对应多长的向量

如果你给定一个 id：

```text
id = 17
```

那 embedding lookup 做的事情非常简单：

```text
out = table[17]
```

也就是：

- 直接把第 17 行向量取出来

如果你有一批 id：

```text
ids = [17, 5, 302, 17, ...]
```

那输出就是：

```text
out[0] = table[17]
out[1] = table[5]
out[2] = table[302]
...
```

所以它本质上不是“算很多乘加”，而是：

- **按一组离散索引，把对应的行从大表里 gather 出来**

## 为什么它和 GEMM 很不一样

`gemm` 的特点是：

- 访存规则
- 计算密集
- 数据复用很强

而 `embedding / gather` 的特点通常正相反：

- 访存位置由 `ids` 决定
- 相邻线程读到的行不一定相邻
- 算法本身几乎没有复杂乘加
- 主要压力经常落在 memory system 上

所以这类 kernel 很适合用来理解：

- irregular memory access
- coalescing 为什么会变差
- cache 命中为什么不稳定

## 当前这个教学版 kernel 在做什么

当前 `row_gather.cu` 采用最简单的结构：

- 一个 `block` 负责一个输出行
- block 先读当前这行对应的 `token_id`
- block 内线程协作，把 `table[token_id, :]` 拷到 `out[row, :]`

也就是：

```text
ids[row] -> token_id
table[token_id, :] -> out[row, :]
```

所以这版更像：

- 一个最小可运行的 embedding lookup 样例
- 一个专门用来观察 gather 访存模式的基线

## 编译

```bash
make
```

## 运行

```bash
./row_gather
```

## 现在最值得记住的点

1. `embedding lookup` 本质上是 gather，不是 dense matrix multiply。
2. 这类 kernel 的主要难点通常不是算力，而是不规则访存。
3. 它是理解很多推荐系统、LLM token embedding、cache 读写模式的一个很好入口。
