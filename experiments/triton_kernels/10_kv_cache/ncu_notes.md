# NCU Notes: Triton KV Cache Append / Update

这份笔记记录当前目录里的 Triton kernel：

- `kv_cache_append_update_kernel`

并和 `experiments/cuda_kernels/10_kv_cache/ncu_notes.md` 的 CUDA 版本对照。

## Profiling 命令

append 第一次 launch：

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
  --kernel-name regex:kv_cache_append_update_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/triton_kernels/10_kv_cache/kv_cache_append_update.py
```

update 第二次 launch：

```bash
/usr/local/cuda-12.4/bin/ncu \
  --csv \
  --page raw \
  --target-processes all \
  --kernel-name-base demangled \
  --launch-skip 1 \
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
  --kernel-name regex:kv_cache_append_update_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/triton_kernels/10_kv_cache/kv_cache_append_update.py
```

## NCU 关键指标

| impl | phase | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `append` | `kv_cache_append_update_kernel` | `6` | `128` | `2.18` | `0.58` | `0.08` | `0.18` | `0.58` | `8.25` |
| Triton | `update` | `kv_cache_append_update_kernel` | `2` | `128` | `2.18` | `0.53` | `0.01` | `0.05` | `0.53` | `8.26` |
| CUDA | `append` | `kv_cache_append_update_kernel` | `6` | - | `2.62` | `0.47` | `0.03` | `6.42` | `0.47` | `5.67` |
| CUDA | `update` | `kv_cache_append_update_kernel` | `2` | - | `2.56` | `0.45` | `0.01` | `18.98` | `0.45` | `6.05` |

## 最直接的结论

- Triton 和 CUDA 这两版的结论几乎完全一致。
- 这份 profile 的主导因素不是算法，而是：
  - grid 极小
  - 每次工作量极轻
  - launch 开销和空转占比很高

## 和 CUDA 版对比怎么看

最重要的共同点是：

- append 都只有 `6` 个 block
- update 都只有 `2` 个 block

所以无论是 Triton 还是 CUDA，这次 NCU 都更像：

- 在 profile 一个非常轻的小写入 kernel
- 不是在 profile“真实 KV cache 写入上限”

这也是为什么：

- `mem %` 极低
- `compute %` 极低
- `occ %` 极低

## 这份对比最该记住的点

1. Triton 和 CUDA 在这个目录里都给出了同样的信号：样本太小，profile 主要被 launch 规模主导。
2. append 和 update 在这份最小实现里共享同一个 kernel，所以画像非常相似。
3. 如果后面真要研究 KV cache 性能，首先要扩大操作批量，否则 NCU 的信息密度很低。
