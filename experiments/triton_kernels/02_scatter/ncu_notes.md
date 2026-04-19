# NCU Notes: Triton Scatter / Index Add

这份笔记记录当前目录里的 Triton kernel：

- `index_add_rows_kernel`

并和 `experiments/cuda_kernels/02_scatter/ncu_notes.md` 里的 CUDA 版本对照。

## Profiling 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --csv \
  --page raw \
  --target-processes all \
  --kernel-name-base demangled \
  --launch-count 1 \
  --metrics \
    gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,\
launch__block_size \
  --kernel-name regex:index_add_rows_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/triton_kernels/02_scatter/index_add_rows.py
```

## NCU 关键指标

| impl | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `index_add_rows_kernel` | `8192` | `128` | `9.18` | `52.72` | `14.95` | `14.95` | `34.56` | `75.69` |
| CUDA | `index_add_rows_kernel` | - | - | `8.29` | `58.24` | `13.03` | `16.08` | `34.25` | `74.30` |

## 最直接的结论

- Triton 和 CUDA 这两版的 profile 非常接近。
- 二者都明显偏 memory-bound，而不是 compute-bound。
- 这和 scatter / index_add 的本质完全一致：
  - 不规则写
  - 原子加
  - 写冲突

## 和 CUDA 版对比怎么看

最值得记住的是下面这组并排数字：

- Triton: `mem % = 52.72`, `compute % = 14.95`, `occ % = 75.69`
- CUDA: `mem % = 58.24`, `compute % = 13.03`, `occ % = 74.30`

也就是说：

- occupancy 基本同一档
- memory throughput 基本同一档
- compute throughput 也都远低于 memory throughput

这说明 Triton 版并没有把这个算子“改造成另一种问题”。

它仍然是那个熟悉的 scatter：

- 地址由 `ids` 决定
- 目标位置可能冲突
- 线程不是在忙着算，而是在等 memory / atomic 路径

## 这份对比最有价值的地方

这里最重要的不是 Triton 比 CUDA 慢了 `0.89 us`。

真正有价值的是：

- 两种实现都给出了几乎一样的硬件画像
- 这说明你对这个算子的第一直觉应该很稳定

也就是：

1. scatter 和 gather 都偏访存，但 scatter 更麻烦，因为这里还有 atomic add。
2. 这种 kernel 的性能讨论，重点通常不在算术，而在索引分布和写冲突。
3. Triton 在这里更像是换了一种表达方式，而不是换了一种算子本质。
