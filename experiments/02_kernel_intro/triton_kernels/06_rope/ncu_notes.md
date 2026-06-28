# NCU Notes: Triton RoPE

这份笔记记录当前目录里的 Triton kernel：

- `rope_forward_kernel`

并和 `experiments/02_kernel_intro/cuda_kernels/06_rope/ncu_notes.md` 的 CUDA 版本对照。

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
  --kernel-name regex:rope_forward_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/02_kernel_intro/triton_kernels/06_rope/triton_rope.py
```

## NCU 关键指标

| impl | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `rope_forward_kernel` | `1024` | `128` | `2.75` | `10.04` | `23.65` | `4.92` | `6.01` | `39.09` |
| CUDA | `rope_forward_kernel` | - | - | `3.04` | `9.27` | `9.10` | `4.17` | `5.20` | `17.89` |

## 最直接的结论

- Triton 和 CUDA 都把 RoPE 呈现成一个很轻的 kernel。
- 两边的共同特征是：
  - duration 很短
  - throughput 不高
  - 都不是“重型算子”

## 和 CUDA 版对比怎么看

这组数据最稳定的共同点是：

- `mem %` 都不高
- `l1tex %` 和 `l2 %` 都不高
- 这说明当前 RoPE 更像轻量旋转 kernel，而不是吞吐打满的主力算子

Triton 和 CUDA 的差别在于：

- Triton 的 `compute % = 23.65`
- CUDA 的 `compute % = 9.10`

但这不应该被解读成“RoPE suddenly 变成 compute-bound 了”。

更合理的理解是：

- Triton 当前实现把较多工作留在单个 program 里
- occupancy 也比 CUDA 更高一些
- 所以 profile 里看到的算术占比会更显眼

可即便如此，它仍然是一个：

- 很轻
- 很短
- 重点在数据流而不是吞吐极限

的教学 kernel。

## 这份对比最该记住的点

1. RoPE 在 Triton 和 CUDA 里都没有变成“大算力问题”，它仍然是一个轻量旋转算子。
2. 当前 profile 的主要价值是说明它的轻量性质，而不是争论谁更快。
3. 如果后面真要做性能版，重点通常还是读写规整性和向量化，而不是公式本身。
