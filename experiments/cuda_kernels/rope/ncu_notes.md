# NCU Notes: RoPE

这份笔记记录当前目录里的最小 `RoPE` kernel：

- `rope_forward_kernel`

重点是看这版“按 pair 做二维旋转”的教学实现，在 GPU 上更像哪一类 kernel。

## Profiling 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-count 1 \
  --kernel-name rope_forward_kernel \
  ./rope_forward
```

## 当前样本

当前程序使用：

- `seq_len = 128`
- `num_heads = 8`
- `head_dim = 64`
- `threads_per_block = 128`

因此：

- 总共有 `seq_len * num_heads = 1024` 个 block
- 一个 block 处理一个 `(token, head)` 行
- 每个线程负责若干个二维 pair 的旋转

## NCU 关键指标

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rope_forward` | `rope_forward_kernel` | `3.04` | `9.27` | `9.10` | `4.17` | `5.20` | `17.89` |

## 最直接的结论

- 这版 RoPE 很轻，单次 kernel 只有 `3.04 us`
- `memory throughput` 和 `compute throughput` 都不高
- 它更像一个短小、规则明确、但吞吐没有被充分拉起来的轻量逐元素 kernel

## 这版 kernel 的主要限制

先看几个直接信号：

- `grid size = 1024`
- `achieved occupancy = 17.89%`
- `No Eligible ≈ 77.70%`

这说明：

- 比 `attention`、`kv_cache` 这种极小 grid 要好得多
- 但它仍然没有把 GPU 喂得很满
- warp 大量时间在等待，而不是连续发射指令

`ncu` 里最显眼的 stall 不是普通 global load，而是：

- `immediate constant cache miss ≈ 3.5 cycles`

这和当前 RoPE 的写法是对得上的：

- kernel 里会频繁用到和旋转相关的常量路径
- 如果 warp 内访问模式不够理想，这类常量读取会被放大成 stall

## 访存画像怎么理解

这版 RoPE 的访存不是最糟，但也不算完全规整：

- `L1/TEX Hit Rate = 50.00%`
- `L2 Hit Rate = 46.42%`
- excessive sectors 占比约 `50%`

也就是说：

- 这不是纯 cache-friendly 的理想向量核
- 当前线程映射下，load/store 仍有明显不规整部分

`ncu` 还明确提示：

- global load 和 global store 到 DRAM 时，平均每个 sector 只用了 `16 / 32 bytes`

这基本说明：

- 当前写法在 pair 级旋转上是“好理解”的
- 但从 coalescing 角度看，还远不是最优布局

## 这份实验最该记住的结论

1. 教学版 RoPE 的核心价值是把“二维旋转”映射成最小可运行 kernel，不是把 GPU 跑满。
2. 当前 profile 说明它是一个轻量 kernel，compute 和 memory 都不重，但 warp 仍然经常因为依赖和常量访问而等待。
3. 如果后面继续做性能版，最自然的方向通常是：
   - 改善 pair 方向的 load/store 规整性
   - 研究更好的向量化读写
   - 减少常量/三角函数路径带来的 stall

