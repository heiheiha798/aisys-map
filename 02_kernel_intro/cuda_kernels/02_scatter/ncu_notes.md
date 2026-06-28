# NCU Notes: Scatter / Index Add

这份笔记记录当前目录里的 `index_add` kernel：

- `index_add_rows_kernel`

它和 `embedding gather` 正好形成一组对照：

- `gather`
  - 从不规则位置读
- `scatter / index_add`
  - 向不规则位置写，而且这里还是原子加

## Profiling 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-count 1 \
  --kernel-name index_add_rows_kernel \
  ./index_add_rows
```

## 当前样本

当前程序使用：

- `src_rows = 4096`
- `dst_rows = 512`
- `dim = 256`
- `threads_per_block = 256`

也就是：

- 一个 block 处理一行 `src`
- 先读取 `indices[src_row]`
- 再把 `src[src_row, :]` 原子加到 `dst[dst_row, :]`

## NCU 关键指标

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `index_add_rows` | `index_add_rows_kernel` | `8.29` | `58.24` | `13.03` | `16.08` | `34.25` | `74.30` |

## 最直接的结论

- 这版 scatter 明显偏 memory-bound
- 并且比 `embedding gather` 更难受，因为这里不是普通 store，而是原子加

先看最关键的数字：

- `memory throughput = 58.24%`
- `compute throughput = 13.03%`
- `achieved occupancy = 74.30%`

这说明：

- block 数量够多，不是小 grid 问题
- occupancy 也不低
- 真正的问题是访存和原子写相关等待

## 这次 profile 最重要的信号

`ncu` 里最值得记住的一句是：

- `L1TEX scoreboard dependency ≈ 47.0 cycles`

它占到了 issue 间隔的：

- `79.6%`

这基本就是一个很典型的 scatter / atomic 写入信号：

- 线程并不是在忙着算
- 而是在等目标地址相关的 load/store / atomic 路径完成

另外还有一个很关键的提示：

- DRAM miss 到达时，每个 sector 平均只利用了 `26.4 / 32 bytes`

说明：

- 当前这版写入也带有不规则访问特征
- 地址由 `indices` 决定
- 即使 block 内部沿 `dim` 方向是连续的，不同 block 之间仍然会打散到不同 `dst_row`

## 为什么 scatter 比 gather 更值得警惕

这次 profile 虽然没有直接把“原子冲突次数”单独做成一列，但从 kernel 语义就能知道：

- gather 主要是随机读
- scatter / index_add 是随机写，而且要做原子加

这意味着：

- 目标地址上的竞争会让等待更明显
- cache 和带宽问题之外，还要承受原子更新带来的序列化风险

所以这类 kernel 往往比看上去更难优化。

## 这份实验最该记住的结论

1. `scatter / index_add` 和 `embedding gather` 一样，都很容易受输入分布影响，但 scatter 还额外带着原子写的代价。
2. 当前 profile 已经很清楚地说明它不是算力瓶颈，而是访存和依赖链主导。
3. 如果后面继续做性能版，最自然的方向通常是：
   - 减少原子冲突
   - 研究更好的索引分桶或重排
   - 观察不同重复度的 `indices` 分布对 profile 的影响

