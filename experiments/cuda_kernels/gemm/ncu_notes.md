# NCU Notes: GEMM CUDA Core vs Tensor Core

这份笔记记录当前 `gemm/` 目录里 9 个 kernel 的 profile 结果，重点回答两件事：

- `CUDA core` tiled GEMM 和 `Tensor Core / WMMA` GEMM 的画像到底差在哪
- `fp16 / bf16 / int8 / int4` 换精度之后，瓶颈有没有真的变

## Profiling 命令

所有 profile 都用同一套方式跑：

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

## 运行结果

先看正常运行时的 benchmark：

| binary | avg_ms | tflops |
| --- | ---: | ---: |
| `fp32_gemm_cuda_core` | `0.0853` | `25.17` |
| `fp16_gemm_cuda_core` | `0.0847` | `25.37` |
| `fp16_gemm_tensor_core` | `0.0599` | `35.87` |
| `bf16_gemm_cuda_core` | `0.0851` | `25.24` |
| `bf16_gemm_tensor_core` | `0.0589` | `36.47` |
| `int8_gemm_cuda_core` | `0.0848` | `25.32` |
| `int8_gemm_tensor_core` | `0.0325` | `66.09` |
| `int4_gemm_cuda_core` | `0.0923` | `23.28` |
| `int4_gemm_tensor_core` | `0.0339` | `63.29` |

最直接的结论：

- `cuda_core` 组整体都集中在 `23-25 TFLOPS`
- `tensor_core` 组整体明显更快
- `int8/int4` Tensor Core 的提速最大

## NCU 关键指标总表

下面只保留最关键的一组指标：

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp32_gemm_cuda_core` | `tiled_gemm_kernel` | `112.19` | `48.92` | `49.43` | `49.75` | `21.18` | `32.85` |
| `fp16_gemm_cuda_core` | `tiled_gemm_kernel` | `108.86` | `50.52` | `51.91` | `51.44` | `18.85` | `31.97` |
| `bf16_gemm_cuda_core` | `tiled_gemm_kernel` | `108.77` | `50.43` | `51.81` | `51.26` | `18.91` | `33.02` |
| `int8_gemm_cuda_core` | `tiled_gemm_kernel` | `105.47` | `52.07` | `53.39` | `52.95` | `12.67` | `33.08` |
| `int4_gemm_cuda_core` | `tiled_gemm_int4_kernel` | `103.71` | `53.13` | `62.59` | `54.38` | `6.20` | `33.07` |
| `fp16_gemm_tensor_core` | `wmma_gemm_16x16x16_kernel` | `67.71` | `89.41` | `22.44` | `94.51` | `69.07` | `56.87` |
| `bf16_gemm_tensor_core` | `wmma_gemm_16x16x16_kernel` | `67.20` | `88.99` | `22.33` | `94.33` | `73.90` | `56.03` |
| `int8_gemm_tensor_core` | `wmma_gemm_int8_kernel` | `35.84` | `85.77` | `22.86` | `91.87` | `71.75` | `56.69` |
| `int4_gemm_tensor_core` | `wmma_gemm_int4_kernel` | `36.61` | `84.77` | `28.49` | `93.51` | `63.59` | `90.30` |

## 先看两大类的本质区别

### 1. `CUDA core` 版是典型的 tiled FMA GEMM

`fp32/fp16/bf16/int8/int4` 的 `cuda_core` 版本都很像：

- `Memory Throughput` 大约 `49-53%`
- `Compute (SM) Throughput` 大约 `49-63%`
- occupancy 大约 `32-33%`
- NCU 直接把它们描述成更偏 `balanced` 或 `compute + memory` 都重要

这和代码结构是完全一致的：

- shared-memory tiling
- register accumulator
- 标量 `acc += a * b`

也就是说：

- 这些 kernel 虽然输入精度不同
- 但计算主路径仍然是普通 CUDA core 的 `FMA`
- 所以吞吐大体还停留在同一档

### 2. `Tensor Core` 版已经明显变成 feeding-path 问题

`fp16/bf16/int8/int4` 的 `tensor_core` 版本整体呈现另一种画像：

- `Memory Throughput` 已经到 `84-89%`
- `L1/TEX Cache Throughput` 已经到 `91-95%`
- `Compute (SM) Throughput` 反而只有 `22-28%`
- occupancy 明显更高，尤其 `int4` 到了 `90.30%`

这说明：

- Tensor Core 的乘加路径已经切对了
- 当前瓶颈不再是“不会算”
- 而是“怎么把数据更顺地喂进去”

## 各组怎么看

### `fp32/fp16/bf16` CUDA core

这三组 profile 很接近：

- duration 都在 `108-112 us`
- `FMA` 是最高利用的 pipeline
- `global store` 到 L1TEX 的访问模式不理想
- 还有一部分 `uncoalesced global accesses`

更准确地说：

- 这是比较标准的共享内存 tiled GEMM
- 受寄存器、occupancy、store pattern 一起约束

### `fp16/bf16` Tensor Core

这两组几乎是同一张图：

- duration 大约 `67 us`
- `Memory Throughput` 约 `89%`
- `L1/TEX Cache Throughput` 约 `94%`
- occupancy 大约 `56%`
- top stall 是 `waiting for a scoreboard dependency on a L1TEX operation`
- NCU 明确提示 global load 只有 `16 / 32 bytes per sector` 被有效利用

这说明：

- 现在主要不是 Tensor Core 算得不够快
- 而是 `load_matrix_sync` 前后的 global-memory feeding path 还不够理想

### `int8` CUDA core vs Tensor Core

`int8_gemm_cuda_core`：

- `duration = 105.47 us`
- `mem % = 52.07`
- `compute % = 53.39`

`int8_gemm_tensor_core`：

- `duration = 35.84 us`
- `mem % = 85.77`
- `compute % = 22.86`

这组最能说明：

- 低精度存储本身不会自然带来质变
- 真正带来质变的是计算路径从标量 `FMA` 切到 `mma_sync`

不过 `int8` Tensor Core 这版还有一个额外问题：

- shared stores 出现了平均 `4-way bank conflict`
- NCU 也报告了 `uncoalesced shared accesses`

所以这版虽然已经很快，但还不是最终形态。

### `int4` CUDA core vs Tensor Core

`int4_gemm_cuda_core`：

- `duration = 103.71 us`
- `mem % = 53.13`
- `compute % = 62.59`
- 仍然是比较均衡的 ALU + memory 内核

`int4_gemm_tensor_core`：

- `duration = 36.61 us`
- `mem % = 84.77`
- `compute % = 28.49`
- occupancy 达到 `90.30%`
- top stall 仍然是 `L1TEX scoreboard dependency`

这说明：

- `4090 / sm_89` 上 `int4` Tensor Core 路径确实能跑起来
- 而且确实比 `int4` 的 CUDA core 对照版快很多
- 但它也同样没有摆脱 feeding-path 限制

## 这个实验最该记住的结论

1. 低精度数据类型本身，不等于 Tensor Core GEMM。
2. 如果 kernel 还是普通标量 `FMA`，即使读的是 `fp16/bf16/int8/int4`，整体吞吐仍然可能停在同一量级。
3. 一旦切到 `WMMA` / Tensor Core，吞吐会明显上一个台阶。
4. 但 Tensor Core 版当前最主要的限制不是乘加单元，而是 global load、L1TEX、layout、shared-memory feeding path。
5. `int8/int4` Tensor Core 已经跑通，但如果继续优化，最值得做的不是改数学公式，而是改：
   - tile mapping
   - global load coalescing
   - shared-memory layout
   - bank conflict
   - fragment feeding path
