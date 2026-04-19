# NCU Notes: Attention

这份笔记记录当前目录里的教学版 `scaled dot-product attention` kernel：

- `attention_kernel`

重点不是追求高性能 attention，而是看这份最小实现的 profile 长什么样，以及它为什么离真正高性能 attention 还很远。

## Profiling 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-count 1 \
  --kernel-name attention_kernel \
  ./attention
```

## 当前样本

当前程序使用：

- `seq_len = 64`
- `head_dim = 32`
- `threads_per_block = 128`

并且当前实现是：

- 一个 block 处理一个 query row
- block 内先算 `QK^T`
- 再在 shared memory 里做 row-wise softmax
- 最后聚合 `V`

## NCU 关键指标

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention` | `attention_kernel` | `6.34` | `8.78` | `2.89` | `22.35` | `5.43` | `5.80` |

## 最直接的结论

- 这份教学版 attention 现在最明显的问题不是算力不够，而是 kernel 太小。
- `grid size = 64`，但机器上有 `128` 个 SM，这意味着连“一块一个 SM”都铺不满。
- `achieved occupancy = 5.80%`，说明这次 profile 更像在看一个过小 kernel 的启动和同步画像，而不是在看 attention 的极限性能。

## 这个画像在说明什么

先看几个最关键的数字：

- `duration = 6.34 us`
- `memory throughput = 8.78%`
- `compute throughput = 2.89%`
- `achieved occupancy = 5.80%`

这几项一起看时，结论很清楚：

- 它不是 compute-bound
- 也不是典型的大带宽 memory-bound
- 更像一个因为 grid 太小、同步点很多、活跃 warp 太少而表现很差的小 kernel

`ncu` 里最值得记住的两个信号是：

- `No Eligible ≈ 94.81%`
- `barrier stall ≈ 6.7 cycles`，约占 issue 间隔的 `34.9%`

这说明：

- 很多时候 scheduler 根本没有可发射 warp
- block 内 softmax reduction 和同步，把这个小 kernel 的可并行性进一步压低了

## 访存不是零问题，但不是第一问题

虽然这个 kernel 很小，但访存也不是完全规整：

- `L1/TEX Hit Rate ≈ 78.08%`
- `L2 Hit Rate ≈ 87.04%`
- 全局访问里有 `114688` 个 excessive sectors，约占总量的 `76%`

所以当前这版 attention 不能简单说成“访存很好，只是算得慢”。

更准确的说法是：

- cache 命中率不差
- 但访问仍然不够规整
- 再加上 grid 很小、block 内 barrier 多，最后整体表现还是非常弱

## 这份教学版最该记住什么

1. 这次 profile 的主结论不是某条数学路径慢，而是当前问题规模太小，导致整个 kernel 根本没有把 GPU 跑起来。
2. 这类“单 block 处理一行”的教学版 attention，适合解释 `QK^T -> softmax -> PV` 的闭环，但不适合代表真实 attention 的性能画像。
3. 如果后面真的要做性能版 attention，第一步通常不是微调公式，而是：
   - 扩大问题规模
   - 改线程映射
   - 减少 barrier
   - 减少中间 `scores` 的显式存取

