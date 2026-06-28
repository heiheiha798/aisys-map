# NCU Notes: BF16 GEMM

这里只保留 `bf16` 三条线：

- `bf16_gemm_cuda_core`
- `bf16_gemm_tensor_core`
- `bf16_gemm_cublas`

关注点只有两个：

- `CUDA core` 和 `Tensor Core` 的画像差在哪
- 我们自己的 `bf16` Tensor Core kernel 和 `cuBLAS` 还有多大差距

## Profiling 命令

统一使用：

```bash
GEMM_PROFILE_ONCE=1 /usr/local/cuda-12.4/bin/ncu \
  --target-processes all \
  --set full \
  --launch-skip 1 \
  --launch-count 1 \
  --kernel-name <kernel_name> \
  ./<binary>
```

这里：

- `GEMM_PROFILE_ONCE=1`
  - 程序只跑一次大矩阵 kernel，不进入 benchmark 循环
- `--launch-skip 1 --launch-count 1`
  - 跳过前面的 correctness 小矩阵，只抓后面的 `1024 x 1024 x 1024` 主样本

## Benchmark

| binary | avg_ms | tflops |
| --- | ---: | ---: |
| `bf16_gemm_cuda_core` | `0.0844` | `25.43` |
| `bf16_gemm_tensor_core` | `0.0572` | `37.56` |
| `bf16_gemm_cublas` | `0.0181` | `118.93` |

最直接的结论：

- `bf16_gemm_tensor_core` 明显快于 `bf16_gemm_cuda_core`
- 但 `cuBLAS` 依然快很多

## NCU 关键指标

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16_gemm_cuda_core` | `tiled_gemm_kernel` | `109.76` | `49.94` | `50.45` | `50.72` | `18.77` | `32.91` |
| `bf16_gemm_tensor_core` | `wmma_gemm_16x16x16_kernel` | `67.01` | `73.08` | `31.04` | `63.25` | `73.08` | `64.43` |
| `bf16_gemm_cublas` | `ampere_s16816gemm_bf16_128x64_ldg8_stages_32x6_nn` | `20.64` | `60.67` | `36.93` | `30.27` | `60.67` | `8.28` |

## BF16 CUDA Core

`bf16_gemm_cuda_core` 的画像很典型：

- `Memory Throughput ≈ 50.80%`
- `Compute (SM) Throughput ≈ 50.45%`
- `L1/TEX ≈ 50.72%`
- `Occupancy ≈ 32.91%`

这说明它更像一个传统 tiled GEMM：

- shared memory 和寄存器累加都在工作
- 计算和访存压力比较接近
- 主要不是 Tensor Core feeding path 的问题

## BF16 Tensor Core

当前 `bf16_gemm_tensor_core` 是一个保留了最小优化的教学型 WMMA kernel：

- 一个 warp 负责一个 `16 x 16` 输出 tile
- 一个 block 放 `8` 个 warp
- block 内共享同一个 `B` tile
- `A` 直接从 global memory 进 fragment
- `B` 先进入 shared memory，再喂给 `wmma::load_matrix_sync`

对应 profile：

- `duration = 67.01 us`
- `mem % = 73.08`
- `compute % = 31.04`
- `l1/tex % = 63.25`
- `occ % = 64.43`

这说明：

- Tensor Core 本身已经在工作
- 但主要瓶颈仍然在 feeding path
- `L1TEX scoreboard dependency` 仍然是很重要的 stall 来源

和最初完全不 staging `B` 的版本相比，这版最大的意义是：

- block 内对同一个 `B` tile` 的重复 global load 减少了
- `L1/TEX` 压力明显下降
- benchmark 也比最初基线略好

但它依然只是教学实现，不是库级实现。

## BF16 Tensor Core vs cuBLAS

`cuBLAS` 这一条最值得看的不是“它更快”这句废话，而是画像完全不同：

- `duration = 20.64 us`
- `l1/tex % = 30.27`
- `compute % = 36.93`

这说明 `cuBLAS` 并没有像我们的 kernel 那样，被近端 feeding path 压得这么重。

简单说，差距主要不在“会不会用 Tensor Core”，而在：

- block tiling 更大
- warp / CTA 分工更复杂
- pipeline 更深
- global load / shared load 组织更好
- Tensor Core feeding 更接近持续饱和

## 这份实验最该记住的结论

1. `bf16_gemm_cuda_core` 和 `bf16_gemm_tensor_core` 的主要区别，不只是数据类型，而是计算主路径从标量 `FMA` 变成了 `mma_sync`。
2. 自写 WMMA kernel 的主要难点，不是把 `mma_sync` 调起来，而是把它持续喂饱。
3. 当前这版 `bf16` Tensor Core kernel 已经能作为学习样例，但离 `cuBLAS` 还有明显差距。
4. 如果后面继续优化，重点应该继续放在 `bf16` 这条主线里的 feeding path，而不是把目录再扩展回更多数据类型。
