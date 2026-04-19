# NCU Notes: Triton Row-wise Softmax

这份笔记记录当前目录里的 Triton kernel：

- `row_softmax_kernel`

并和 `experiments/cuda_kernels/04_softmax/ncu_notes.md` 的 CUDA 版本对照。

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
  --kernel-name regex:row_softmax_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/triton_kernels/04_softmax/row_softmax.py
```

## NCU 关键指标

| impl | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `row_softmax_kernel` | `128` | `128` | `2.40` | `5.78` | `3.59` | `3.59` | `3.29` | `8.31` |
| CUDA | `row_softmax_kernel` | `128` | `256` | `3.14` | `13.56` | `13.56` | - | - | `16.07` |

## 最直接的结论

- Triton 和 CUDA 在这个目录里都不是“真实 softmax 性能样例”。
- 两者共同的问题都不是 compute vs memory，而是：
  - grid 太小
  - occupancy 太低
  - GPU 没有真正被喂饱

## 和 CUDA 版对比怎么看

这组数据里最重要的并不是 `2.40 us` 和 `3.14 us`。

真正重要的是：

- Triton: `grid = 128`, `occ = 8.31%`
- CUDA: `grid = 128`, `occ = 16.07%`

也就是说，两边都还是教学规模。

这正对应 CUDA 笔记里的核心判断：

- 这个 softmax 更像一个 under-filled 的 reduction kernel
- 不是一个已经进入真实性能区间的 softmax

Triton 版甚至比 CUDA 版更轻：

- memory throughput 更低
- compute throughput 更低
- occupancy 也更低

所以它更像一个：

- 语义很清楚
- correctness 已打通
- 但不适合拿来判断“softmax 本质上更偏 memory 还是 compute”

## 这份对比最该记住的点

1. 当前这两版 softmax 都首先暴露的是“并发不够”，而不是单纯的 compute-bound 或 memory-bound。
2. 这种教学版 kernel 更适合建立 reduction 结构直觉，不适合直接拿来做性能结论。
3. 如果后面真的要研究 softmax 性能，第一步通常不是继续改公式，而是先把问题规模和并发度拉起来。
