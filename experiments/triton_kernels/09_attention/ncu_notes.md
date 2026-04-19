# NCU Notes: Triton Attention

这份笔记记录当前目录里的 Triton kernel：

- `attention_kernel`

并和 `experiments/cuda_kernels/09_attention/ncu_notes.md` 的 CUDA 版本对照。

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
  --kernel-name regex:attention_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/triton_kernels/09_attention/triton_attention.py
```

## NCU 关键指标

| impl | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `attention_kernel` | `64` | `128` | `38.62` | `5.19` | `5.19` | `5.19` | `0.67` | `8.33` |
| CUDA | `attention_kernel` | `512` | `32` | `49.92` | `64.10` | `12.91` | `66.23` | `9.33` | `8.34` |

## 先说清楚：这里不能直接拿绝对时延下结论

当前 Triton 脚本的样本是：

- `seq_len = 64`
- `head_dim = 32`

而 CUDA 笔记里的样本是：

- `seq_len = 512`
- `head_dim = 32`

所以：

- `38.62 us` 和 `49.92 us` 不能直接比快慢

## 真正有价值的对比

更值得看的其实是 launch 规模：

- Triton: `grid = 64`
- CUDA: `grid = 512`

这说明当前 Triton attention 更像一个：

- 更小的教学样本
- 更明显 under-filled 的 demo

从 profile 也能看出来：

- `mem %` 很低
- `compute %` 很低
- `occ %` 只有 `8.33`

这说明当前 Triton 版的第一问题不是 attention 算法本身，而是：

- 样本太小
- grid 太小
- GPU 根本没被真正铺开

## 和 CUDA 版的关系

CUDA 笔记里最关键的点是：

- 那版 vanilla attention 虽然远不是高性能实现
- 但至少 `grid = 512`
- 已经比较像一个正常 baseline

Triton 版则更像：

- 一个最小 correctness 用例
- 重点是把 `QK^T -> softmax -> PV` 数据流写出来
- 不是一个适合看性能画像的 attention baseline

## 这份对比最该记住的点

1. 当前 Triton attention 的 profile 主要反映的是“小样本教学实现”，而不是 attention 的一般性能画像。
2. CUDA 版至少在更大的 `seq_len` 下提供了一个更像 baseline 的 profile。
3. 如果后面真要做 Triton attention 的性能对比，首先要统一样本规模，否则绝对时延没有比较意义。
