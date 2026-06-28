# NCU Notes: Triton LayerNorm and RMSNorm

这份笔记记录当前目录里的两条 Triton kernel：

- `row_layernorm_kernel`
- `row_rmsnorm_kernel`

并和 `experiments/02_kernel_intro/cuda_kernels/05_layernorm/ncu_notes.md` 的 CUDA 版本对照。

## Profiling 命令

LayerNorm：

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
  --kernel-name regex:row_layernorm_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/02_kernel_intro/triton_kernels/05_layernorm/row_layernorm.py
```

RMSNorm：

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
  --kernel-name regex:row_rmsnorm_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/02_kernel_intro/triton_kernels/05_layernorm/row_rmsnorm.py
```

## NCU 关键指标

| impl | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `row_layernorm_kernel` | `1024` | `128` | `3.55` | `30.59` | `20.94` | `20.94` | `15.68` | `44.31` |
| Triton | `row_rmsnorm_kernel` | `1024` | `128` | `3.20` | `33.88` | `11.89` | `11.89` | `17.22` | `42.37` |
| CUDA | `row_layernorm_kernel` | - | - | `6.53` | `50.06` | `50.06` | `67.94` | `9.27` | `79.40` |
| CUDA | `row_rmsnorm_kernel` | - | - | `4.61` | `38.44` | `38.44` | `58.06` | `13.23` | `76.33` |

## 最直接的结论

- Triton 和 CUDA 两边都保留了同一个核心关系：
  - `rmsnorm` 比 `layernorm` 更轻
  - `rmsnorm` 的时延更短

## 和 CUDA 版对比怎么看

CUDA 笔记里最关键的结论是：

- `layernorm` 要做 `sum(x)` 和 `sum(x^2)`
- `rmsnorm` 只需要 `sum(x^2)`

这个逻辑在 Triton 版里完全没有变。

从 Triton 数据看：

- `row_layernorm_kernel`: `3.55 us`
- `row_rmsnorm_kernel`: `3.20 us`

虽然两者的 profile 百分比和 CUDA 不完全一样，但趋势非常稳定：

- `rmsnorm` 路径更短
- `layernorm` 统计更多

## Triton 画像和 CUDA 画像的差别

Triton 这两版的 occupancy 比 CUDA 低很多：

- Triton `44.31% / 42.37%`
- CUDA `79.40% / 76.33%`

这说明 Triton 当前实现更像：

- 依赖 `tl.sum` 的单 program row-wise reduction
- 简洁优先
- 教学优先

而 CUDA 版更接近：

- 一个 block 处理一整行
- block 内并行归约
- 硬件利用率更高一些

但共同点仍然成立：

- 它们都不是 GEMM 那种大算力 kernel
- 都更接近高频的小型 normalization kernel

## 这份对比最该记住的点

1. Triton 没有改变 `layernorm` 和 `rmsnorm` 的本质区别，真正的差别仍然来自 reduction 次数。
2. 即使硬件利用率画像不同，`rmsnorm` 更轻、更快这件事在两种实现里都成立。
3. 这类 kernel 的学习重点仍然是 row-wise reduction，而不是极限吞吐。
