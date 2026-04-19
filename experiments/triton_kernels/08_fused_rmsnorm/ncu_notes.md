# NCU Notes: Triton Fused Residual + RMSNorm

这份笔记记录当前目录里的 Triton 两段实现：

- `row_mean_sq_kernel`
- `fused_residual_rmsnorm_kernel`

并和 `experiments/cuda_kernels/08_fused_rmsnorm/ncu_notes.md` 的 CUDA 单 kernel fused 实现对照。

## Profiling 命令

第一段 `row_mean_sq_kernel`：

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
  --kernel-name regex:row_mean_sq_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/triton_kernels/08_fused_rmsnorm/fused_residual_rmsnorm.py
```

第二段 `fused_residual_rmsnorm_kernel`：

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
  --kernel-name regex:fused_residual_rmsnorm_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/triton_kernels/08_fused_rmsnorm/fused_residual_rmsnorm.py
```

## NCU 关键指标

| impl | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `row_mean_sq_kernel` | `1024` | `128` | `4.70` | `45.95` | `8.74` | `8.74` | `19.47` | `44.13` |
| Triton | `fused_residual_rmsnorm_kernel` | `1024` | `128` | `4.48` | `48.51` | `6.65` | `6.73` | `20.62` | `39.61` |
| Triton total | end-to-end | - | - | `9.18` | - | - | - | - | - |
| CUDA | `fused_residual_rmsnorm_kernel` | - | - | `5.63` | `38.38` | `34.06` | `48.90` | `16.81` | `80.37` |

## 最直接的结论

- Triton 当前这版不是一个真正单 kernel fused 实现。
- 它为了教学性拆成了两段，所以 end-to-end 时延大约是：
  - `4.70 + 4.48 = 9.18 us`
- CUDA 版则是一个单 kernel fused 实现：
  - `5.63 us`

## 和 CUDA 版对比怎么看

这里最重要的不是“为什么 Triton 更慢”。

真正重要的是：

- Triton 这版是两次 launch
- CUDA 这版是一次 launch

所以它们的对比本来就不是 apples-to-apples。

更准确的理解是：

1. Triton 版优先保留了可读性：
   - 第一段只做 row-wise reduction
   - 第二段只做 normalize
2. CUDA 版优先保留了 fused 形态：
   - residual add
   - reduction
   - normalize
   全部在一次 kernel 里完成

## Triton 这两段各自说明了什么

第一段 `row_mean_sq_kernel`：

- 明显偏 memory-bound
- 几乎就是一条 row-wise reduction 路径

第二段 `fused_residual_rmsnorm_kernel`：

- 仍然偏 memory-bound
- 本质上是读取 `x`、`residual`、`gamma` 和 `mean_sq`
- 然后做一段 elementwise normalize

所以当前 Triton 版更像：

- 用两个最小 kernel 把 fused 数据流拆开看清楚

而不是：

- 用一个 kernel 去追极限性能

## 这份对比最该记住的点

1. Triton 当前实现故意为教学拆成两段，所以不能直接拿总时延和 CUDA 单 kernel fused 实现做性能输赢结论。
2. 这组 profile 更适合说明“为什么单 kernel fusion 会少一次 launch 和中间读写”。
3. 如果后面真的要做 Triton 性能版，最自然的方向就是把这两段重新合并。
