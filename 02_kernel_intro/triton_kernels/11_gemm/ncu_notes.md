# NCU Notes: Triton GEMM

这份笔记记录当前目录里的 Triton kernel：

- `matmul_kernel`

并和 `02_kernel_intro/cuda_kernels/11_gemm/ncu_notes.md` 的 CUDA 目录对照。

这里需要先把限制说清楚：

- Triton 这版是一个最小 `fp32` 教学 matmul
- 当前样本是 `m = 192, n = 160, k = 224`
- CUDA 目录里的重点则是：
  - `bf16_gemm_cuda_core`
  - `bf16_gemm_tensor_core`
  - `bf16_gemm_cublas`
  - 并且主样本是 `1024 x 1024 x 1024`

所以两边不能直接比较绝对时延。

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
  --kernel-name regex:matmul_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  02_kernel_intro/triton_kernels/11_gemm/triton_gemm.py
```

## NCU 关键指标

| impl | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `matmul_kernel` | `9` | `128` | `10.24` | `3.43` | `2.93` | `2.66` | `2.97` | `8.32` |
| CUDA | `tiled_gemm_kernel` | - | - | `109.76` | `49.94` | `50.45` | `50.72` | `18.77` | `32.91` |
| CUDA | `wmma_gemm_16x16x16_kernel` | - | - | `67.01` | `73.08` | `31.04` | `63.25` | `73.08` | `64.43` |
| CUDA | `ampere_s16816gemm_bf16_128x64_ldg8_stages_32x6_nn` | - | - | `20.64` | `60.67` | `36.93` | `30.27` | `60.67` | `8.28` |

## 最直接的结论

- 当前 Triton GEMM 的 profile 主要说明了一件事：
  - 这是一个很小的 teaching sample
  - 不是一个可以和 CUDA / cuBLAS 主样本正面比性能的 case

## 和 CUDA 版对比怎么看

最关键的差别不是编程语言，而是样本规模和定位：

- Triton: `grid = 9`
- CUDA 主样本: `1024 x 1024 x 1024`

所以 Triton 这里看到的是：

- `mem %` 很低
- `compute %` 很低
- `occ %` 只有 `8.32`

这说明当前 Triton 版首先暴露的是：

- 工作量太小
- 并发太少
- 还没进入 GEMM 真正该比较 feeding path / Tensor Core 的阶段

而 CUDA 那边的三条线回答的是另外一个问题：

- CUDA core GEMM 长什么样
- Tensor Core GEMM 长什么样
- cuBLAS 为何又是另一档

## 这份对比最该记住的点

1. 当前 Triton GEMM 是一个教学版最小 kernel，不是和 `cuBLAS` 抢大矩阵乘法的实现。
2. 这个 profile 反而支持了根 README 里的定位：大 GEMM 更应该继续交给 `cuBLAS`。
3. Triton 在这个仓库里的更合理角色，仍然是围绕 GEMM 周围的小算子和 fusion，而不是自己接管主力 GEMM。
