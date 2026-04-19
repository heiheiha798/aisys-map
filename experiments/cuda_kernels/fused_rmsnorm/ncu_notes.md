# NCU Notes: Fused Residual + RMSNorm

这份笔记记录当前目录里的 fused kernel：

- `fused_residual_rmsnorm_kernel`

它把两步逻辑合在一次 launch 里：

- residual add
- row-wise RMSNorm

## Profiling 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-count 1 \
  --kernel-name fused_residual_rmsnorm_kernel \
  ./fused_residual_rmsnorm
```

## 当前样本

当前程序使用：

- `rows = 1024`
- `cols = 256`
- `threads_per_block = 256`

也就是：

- 一个 block 处理一整行
- block 内先做 `x + residual`
- 再做 row-wise reduction，求 `mean_sq`
- 最后写出 `gamma * s * inv_rms`

## NCU 关键指标

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fused_residual_rmsnorm` | `fused_residual_rmsnorm_kernel` | `5.63` | `38.38` | `34.06` | `48.90` | `16.81` | `80.37` |

## 最直接的结论

- 这版 fused kernel 既不是纯 compute-bound，也不是纯 bandwidth-bound
- 更准确地说，它是一个同时带有：
  - row-wise reduction
  - 多次全局访存
  - elementwise normalize
  的 latency-sensitive 小 kernel

和单独的 `rmsnorm` 类似，它更接近很多模型里高频出现的小归一化算子，而不是 GEMM 那类大算力 kernel。

## 这次 profile 最值得看的信号

先看几个关键数字：

- `duration = 5.63 us`
- `memory throughput = 38.38%`
- `compute throughput = 34.06%`
- `achieved occupancy = 80.37%`

这组数字说明：

- occupancy 已经足够高
- 但 compute 和 memory 都没有真正接近峰值
- 主要问题仍然是 latency 和依赖链

`ncu` 里最明确的 stall 原因是：

- `L1TEX scoreboard dependency ≈ 15.8 cycles`

它占 issue 间隔的：

- `45.4%`

这和当前实现很一致：

- 第一次遍历读 `x` 和 `residual`
- reduction 之后第二次再读一遍
- 再乘上 `gamma`

所以虽然逻辑已经 fused 了，但全局访存和依赖链仍然很重。

## 为什么它比“单纯 elementwise”更复杂

这版 fused kernel 不能简单当作一个普通逐元素核来看，因为它同时有两层结构：

- 行内 reduction
- reduction 之后的逐元素 normalize

这会带来两个后果：

- block 内同步不可避免
- 谓词和尾部路径会拉低 warp 的有效线程数

`ncu` 里能看到：

- `Avg. Active Threads Per Warp = 32`
- `Avg. Not Predicated Off Threads Per Warp = 22.13`

说明：

- 线程都在 warp 里被调度了
- 但不是所有线程都始终在有效执行同一条路径

## 这份实验最该记住的结论

1. 把 residual add 和 RMSNorm 融合到一个 kernel 里，首先减少的是 launch 和中间张量写回，不会自动消除 reduction 的代价。
2. 当前这版 fused kernel 已经比“两个完全分开的教学 kernel”更接近真实模型里的小算子形态。
3. 如果后面继续做性能版，最自然的方向通常是：
   - 减少二次读 `x + residual`
   - 研究更好的中间值缓存方式
   - 优化 reduction 和尾部谓词路径

