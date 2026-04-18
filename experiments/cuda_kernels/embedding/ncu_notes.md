# NCU Notes: Embedding Gather

这份笔记记录当前目录里的一个最小 gather kernel：

- `row_gather`

它的价值不在复杂公式，而在于：

- 观察不规则访存
- 观察 gather 和规整 dense kernel 的区别
- 观察**同一个 kernel**在不同 `id` 分布下的 profile 差异

## 建议的 profiling 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-count 1 \
  --kernel-name row_gather_kernel \
  ./row_gather
```

## 当前样本和输入模式

当前程序使用：

- `vocab = 8192`
- `batch = 4096`
- `dim = 256`
- `threads_per_block = 256`

也就是：

- 一个 block 负责一个输出行
- 先读一个 `token_id`
- 再把对应 embedding row 拷到输出

当前支持两种 `id` 分布：

- `random`
  - 默认模式，`id` 比较分散
- `repeated`
  - 用 `GATHER_ID_MODE=repeated` 打开，让很多相邻输出行反复访问很少几条 embedding row

## NCU 关键指标

| mode | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random` | `row_gather_kernel` | `7.30` | `51.91` | `12.72` | `16.10` | `32.28` | `74.34` |
| `repeated` | `row_gather_kernel` | `6.08` | `34.77` | `15.26` | `20.98` | `34.77` | `69.82` |

## 这类 kernel 该重点看什么

和 GEMM、softmax 不一样，`embedding gather` 更值得看的通常不是：

- Tensor pipeline
- FLOPS 利用率

而是：

- global memory access pattern
- coalescing
- L1/TEX / L2 行为
- warp stall 是否主要卡在访存依赖

## `random` 模式：最有价值的信息

`random` 模式下的画像很典型，说明这版 kernel 的主要问题不是算力，而是访存：

- `Memory Throughput ≈ 51.91%`
- `Compute (SM) Throughput ≈ 12.72%`
- `L1/TEX Hit Rate ≈ 10.20%`
- `L2 Hit Rate ≈ 56.71%`

这基本就是一个标准的 gather 信号：

- 没有多少重计算
- 线程大部分时间在等数据回来
- 近端 cache 命中并不高

`ncu` 里最值得记住的一句是：

- `L1TEX scoreboard dependency ≈ 56.0 cycles`

它占了：

- `≈ 79.7%` 的 warp issue 间隔

这说明：

- warp 主要在等 load 完成
- 不是在等某个计算 pipeline

另外还有一个非常贴切的提示：

- DRAM miss 到达时，每个 sector 平均只利用了 `26.4 / 32 bytes`

这说明：

- global load 还不够理想
- 虽然 block 内拷一整行时局部访问是连续的
- 但不同 block 指向的 embedding row 由 `ids` 决定，整体仍然会呈现 gather 式的不规则行为

## `repeated` 模式：同一个 kernel 为什么会变

`repeated` 模式下，最显眼的变化是：

- `duration: 7.30 us -> 6.08 us`
- `DRAM Throughput: 51.91% -> 0.88%`
- `L1/TEX Hit Rate: 10.20% -> 36.80%`
- `L2 Hit Rate: 56.71% -> 97.96%`
- `L1TEX scoreboard dependency: 56.0 cycles -> 36.9 cycles`

这说明：

- kernel 本身没有变
- 线程映射也没有变
- 只是 `ids` 分布从分散访问变成了高重复访问

但 profile 已经明显改变了。

最核心的解释是：

- 相同 embedding row 被反复访问
- 这些访问更容易留在 cache 里
- 所以真正落到 DRAM 的流量大幅下降
- warp 等数据回来的时间也跟着下降

这里最值得记住的是：

- `repeated` 模式并不是“算法更会算”
- 它只是更容易命中 cache
- gather kernel 的表现会被输入分布强烈影响

## 当前这版教学 kernel 的结构

- 一个 block 负责一个输出行
- 先读取 `ids[row]`
- 再把 `table[token_id, :]` 拷到 `out[row, :]`

所以这版更像：

- 一个最小可运行的 embedding lookup 基线

## 当前这类 kernel 应该怎么理解

如果把这次 `random vs repeated` 对照一起看，可以得出三个更强的结论：

1. `embedding gather` 和 GEMM 完全不是一类问题。
2. 对这种 kernel，继续盯着 `compute %` 没什么意义，真正该盯的是：
   - cache hit
   - coalescing
   - L1TEX scoreboard stall
3. 这类 kernel 很适合用来研究：
   - `ids` 分布对 cache 的影响
   - 随机访问和重复访问的差别
   - gather 为什么天然更接近 memory-bound

如果后面继续优化，最自然的方向通常是：

1. 调整线程映射和向量化 load/store
2. 观察不同 `id` 分布下 cache 行为怎么变化
3. 对比“重复 id 很多”和“完全随机 id”时的 profile 差异
