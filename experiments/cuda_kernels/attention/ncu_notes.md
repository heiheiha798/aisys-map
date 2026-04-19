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

- `seq_len = 512`
- `head_dim = 32`
- `threads_per_block = 32`

并且当前实现是：

- 一个 block 处理一个 query row
- block 内先算 `QK^T`
- 再在 shared memory 里做 row-wise softmax
- 最后聚合 `V`

## NCU 关键指标

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention` | `attention_kernel` | `49.92` | `64.10` | `12.91` | `66.23` | `9.33` | `8.34` |

## 最直接的结论

- 同样在 `seq_len = 512, head_dim = 32` 这组参数下，这份普通 attention 明显比当前教学版 `flash_attention` 更快。
- 当前 `attention_kernel` 的 `duration = 49.92 us`，而 `flash_attention_kernel` 是 `1.06 ms`，差了大约 `21x`。
- 这不代表普通 attention 比 FlashAttention 思路更优，只代表当前这个最小 `flash_attention` 实现还非常原始。

## 这个画像在说明什么

先看几个最关键的数字：

- `duration = 49.92 us`
- `memory throughput = 64.10%`
- `compute throughput = 12.91%`
- `achieved occupancy = 8.34%`
- `grid size = 512`

这几项一起看时，结论很清楚：

- 它已经不再是之前那种“过小 kernel 完全没铺开”的状态
- `grid size = 512` 至少能把机器上的 `128` 个 SM 铺起来
- 但它仍然不是一个高性能 attention，只是一个相对更像样的 vanilla baseline

`ncu` 里最值得记住的两个信号是：

- `No Eligible ≈ 94.07%`
- `L1/TEX Hit Rate ≈ 94.54%`
- `L2 Hit Rate ≈ 98.06%`

这说明：

- scheduler 仍然经常拿不到 ready warp
- 但至少访存命中率不差，问题不再只是 “kernel 太小”
- 真正限制它的更像是：
  - 显式 materialize 整行 `scores`
  - block 内 softmax reduction
  - 以及后续 `PV` 聚合的重复读写

## 访存不是零问题，但不是第一问题

虽然这个 kernel 很小，但访存也不是完全规整：

- `L1/TEX Hit Rate ≈ 92.69%`
- `L2 Hit Rate ≈ 98.20%`
- 全局访问里有 `7340032` 个 excessive sectors，约占总量的 `76%`

所以当前这版 attention 不能简单说成“访存很好，只是算得慢”。

更准确的说法是：

- cache 命中率不差
- 但访问仍然不够规整
- 仍然有明显的 uncoalesced global access
- 同时 softmax 这一段的同步和中间结果存取也很重

## 和 `flash_attention` 的同参对比

同样是 `seq_len = 512, head_dim = 32`：

| kernel | grid | block | dur | mem % | compute % | achieved occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention_kernel` | `512` | `32` | `49.92 us` | `64.10` | `12.91` | `8.34` |
| `flash_attention_kernel` | `32` | `32` | `1.06 ms` | `3.44` | `1.02` | `2.08` |

这里最值得记住的不是“普通 attention 赢了”，而是：

1. 当前普通 attention 至少把 grid 铺开了，所以 GPU 利用率看起来像一个正常得多的 baseline。
2. 当前教学版 flash attention 的主要问题不是 FlashAttention 思路，而是实现还太粗糙：
   - `grid size = 32`
   - `Active Warps Per Scheduler = 1.00`
   - `Avg. Active Threads Per Warp = 17.14`
   - shared memory bank conflict 非常重
3. 因此现在这个对比更适合解释“为什么一个不成熟的 FlashAttention 教学实现会比普通 attention 更慢”，而不适合拿来判断算法优劣。
