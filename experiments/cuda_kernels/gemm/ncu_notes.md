# NCU Notes: GEMM CUDA Core vs Tensor Core

这份笔记记录当前 `gemm/` 目录里自写 kernel 加上一个 `cuBLAS` 基线的 profile 结果，重点回答三件事：

- `CUDA core` tiled GEMM 和 `Tensor Core / WMMA` GEMM 的画像到底差在哪
- `bf16 / int8 / int4` 换精度之后，瓶颈有没有真的变
- 我们当前自写的 `bf16` Tensor Core kernel，和成熟库还有多大差距

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
| `bf16_gemm_cuda_core` | `0.0849` | `25.29` |
| `bf16_gemm_tensor_core` | `0.0571` | `37.63` |
| `bf16_gemm_cublas` | `0.0181` | `118.71` |
| `int8_gemm_tensor_core` | `0.0325` | `66.02` |
| `int4_gemm_tensor_core` | `0.0336` | `63.87` |

最直接的结论：

- `cuda_core` 组整体都集中在 `23-25 TFLOPS`
- `tensor_core` 组整体明显更快
- `int8/int4` Tensor Core 的提速最大

## NCU 关键指标总表

下面只保留最关键的一组指标：

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16_gemm_cuda_core` | `tiled_gemm_kernel` | `107.30` | `50.80` | `52.19` | `51.80` | `19.12` | `33.96` |
| `bf16_gemm_tensor_core` | `wmma_gemm_16x16x16_kernel` | `67.55` | `72.50` | `30.79` | `63.80` | `72.50` | `64.83` |
| `bf16_gemm_cublas` | `ampere_s16816gemm_bf16_128x64_ldg8_stages_32x6_nn` | `20.83` | `60.45` | `36.87` | `30.29` | `60.45` | `8.29` |
| `int8_gemm_tensor_core` | `wmma_gemm_int8_kernel` | `35.84` | `85.90` | `22.89` | `91.47` | `71.70` | `56.76` |
| `int4_gemm_tensor_core` | `wmma_gemm_int4_kernel` | `37.82` | `82.71` | `27.80` | `93.97` | `62.36` | `90.70` |

## BF16 Tensor Core: commit / version 对比

这里单独记录 `bf16_gemm_tensor_core` 的一次最小优化迭代，避免后面又新开很多 notes 文件。

### 基线：commit `e70208b`

这是最初的 `bf16` Tensor Core 版本：

- 直接 `global -> fragment -> mma_sync`
- 没有 shared-memory staging
- 同一个 block 内 4 个 warp 会重复加载同一个 `B` tile

实测 benchmark：

```text
bf16_gemm_tensor_core   avg_ms=0.0589  tflops=36.47
```

对应的大问题规模 `ncu` 指标：

| version | dur(us) | mem % | compute % | l1/tex % | occ % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `e70208b` | `67.20` | `88.99` | `22.33` | `94.33` | `56.03` |

基线版本的关键信号是：

- global load 只有 `16 / 32 bytes per sector` 被有效利用
- `L1TEX scoreboard dependency` stall 很重，平均约 `84.9 cycles`
- `L1/TEX` 非常忙
- 但 block 内没有 shared-memory bank conflict

这说明：

- 当前瓶颈主要在近端 load/cache 路径
- 不只是“必须搬数据”
- 而是有明显的 global-load 组织浪费

### 中间尝试：shared-memory staged `B` tile

这一步只做了一个最简单的优化：

- 对 `bf16/fp16` 的 `wmma_gemm_16x16x16_kernel`
- 把 block 内共享的 `B` tile 先搬到 shared memory
- 再让各个 warp 从 shared memory `load_matrix_sync`

动机很直接：

- 同一个 block 的 4 个 warp 在每个 `k0` 上共用同一个 `B` tile
- 这部分原来被重复从 global memory 读了 4 次

中间版本 benchmark：

```text
bf16_gemm_tensor_core   avg_ms≈0.06216  tflops≈34.55
```

多次运行结果很稳定，基本都在：

- `avg_ms ≈ 0.0621 ~ 0.0623`
- `tflops ≈ 34.49 ~ 34.57`

对应 `ncu` 指标：

| version | dur(us) | mem % | compute % | l1/tex % | occ % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `working tree` | `72.03` | `64.75` | `34.03` | `68.68` | `66.12` |

和基线对比，出现了很明显的变化：

- `Memory Throughput`: `88.99% -> 64.75%`
- `L1/TEX Cache Throughput`: `94.33% -> 68.68%`
- `Compute (SM) Throughput`: `22.33% -> 34.03%`
- `Achieved Occupancy`: `56.03% -> 66.12%`
- `Registers / thread`: `40 -> 30`
- `L1TEX scoreboard dependency` stall: `84.9 cycles -> 22.6 cycles`

这说明这一步优化**确实打到了原来的问题**：

- 近端 `L1/TEX` 压力下降了
- warp 等待 `L1TEX` 数据的时间大幅下降了
- block 内重复加载 `B` tile 这个方向是对的

### 为什么反而变慢了

因为这一步虽然缓解了原来的 global/L1 路径压力，但又引入了新的 shared-memory 问题：

- shared loads 平均出现 `2-way bank conflict`
- `ncu` 报告了约 `1,049,039` 次 bank conflicts
- `uncoalesced shared accesses` 也很明显

同时还残留：

- `uncoalesced global accesses`

也就是说，当前 working tree 版本的画像变成了：

- 旧问题：`L1/TEX` 太忙，global load 组织差
- 新问题：shared-memory staging 太粗糙，shared load 访问模式不友好

所以这一步的结论不是“优化失败”，而是：

- **方向是对的**
- **实现还不够好**
- **我们已经把瓶颈从 global/L1 路径的一部分，转移到了 shared-memory 访问模式上**

### 这次迭代最该记住的结论

1. `B` tile 的 block 内重复加载，确实是当前 kernel 的真实浪费，不是不可避免的搬运下界。
2. 最简单的 shared-memory staging 已经能显著降低 `L1/TEX` 压力和 `L1TEX scoreboard` stall。
3. 但如果 shared-memory 布局和访问模式没处理好，就会马上引入新的 bank conflict，抵消收益。
4. 所以下一步优化重点，不是再讨论“要不要 staging”，而是：
   - 怎样让 staged `B` tile 的 shared load 更适合 `wmma::load_matrix_sync`
   - 怎样减少 shared bank conflict
   - 是否要进一步调整 block / warp tile 组织

### 当前版本：8 warps 复用同一个 `B` tile

这一步保留了“只 staging `B`”这个方向，但把结构重新收紧成更简单的版本：

- 恢复成单列输出组织
  - 一个 warp 仍然只负责一个 `16 x 16` 输出 tile
- 让一个 block 放 `8` 个 warp
  - 这 `8` 个 warp 在同一个 `k0` 上共享同一个 `B` tile
- `B` tile 继续放到 shared memory
  - 但只对 shared-memory leading dimension 加很小的 `skew=8`
- `A` 仍然直接从 global memory 喂给 `wmma::load_matrix_sync`
  - 不再把 `A` 也先搬进 shared memory

这样做的直觉是：

- 之前第一次 working tree 版本的问题，不是 “`B` staging 这件事错了”
- 而是 “复用收益还不够大，但 shared-memory 访问副作用已经进来了”
- 于是这一步直接把 block 内复用度从 `4 warp` 提高到 `8 warp`
  - 让同一个 `B` tile 的搬运成本被更多 warp 摊薄

实测 benchmark：

```text
bf16_gemm_tensor_core   avg_ms≈0.0571 ~ 0.0573  tflops≈37.45 ~ 37.61
```

多次运行都比较稳定，已经略好于最初基线 `36.47 TFLOPS`。

对应 `ncu` 指标：

| version | dur(us) | mem % | compute % | l1/tex % | occ % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `working tree v2` | `67.55` | `72.50` | `30.79` | `63.80` | `64.83` |

相对基线 `e70208b`：

- `Duration`: `67.20 -> 67.55 us`
  - `ncu` 单次 profile 下几乎持平
  - 但 benchmark 循环里整体吞吐更高
- `Memory Throughput`: `88.99% -> 72.50%`
- `L1/TEX Cache Throughput`: `94.33% -> 63.80%`
- `Compute (SM) Throughput`: `22.33% -> 30.79%`
- `Achieved Occupancy`: `56.03% -> 64.83%`
- `L1TEX scoreboard dependency`: `84.9 cycles -> 18.6 cycles`

这说明这次版本的主要收益很明确：

- block 内 `B` tile 复用度更高了
- warp 等待 `L1TEX` 数据的时间继续显著下降
- Tensor Core 路径不再像最初版本那样被近端 load/cache 压得太死

更重要的是，这次没有再像第一次 working tree 版本那样，把收益换成大量 shared-memory 冲突：

- `shared load bank conflicts ≈ 45`
- `shared store bank conflicts ≈ 10,753`

这里和上一个 working tree 版本相比已经不是一个量级；上次 shared load conflict 是大问题，这次基本可以认为 shared load 冲突已经被压下去了。

### 这一步之后的判断

到这里，问题画像已经更清楚了：

- 之前的 `L1/TEX` 压力里，确实有相当一部分来自 block 内对同一 `B` tile 的重复 global load
- 只要复用度足够高，而且 shared-memory 布局别做得太激进，这部分浪费是可以实实在在消掉的

但当前 `ncu` 仍然明确提示：

- 还有明显的 `uncoalesced global accesses`
- 仍然存在以 `L1TEX scoreboard` 为主的等待
- `L2` 利用已经比较高，说明 feeding path 依然是主要矛盾

所以当前版本可以视为一个有效的“最简单优化”：

- 它已经证明这条路能带来正收益
- 但它还没有触到更深层的上限

下一步如果继续优化，重点应该放在：

1. `A` 的 global load 访问模式还能不能进一步整理
2. 是否要做更系统的 block tiling，而不只是让更多 warp 共享一个 `B` tile
3. 是否要改成更接近真实高性能 GEMM 的多阶段 pipeline，而不是当前这种 `load -> sync -> mma -> sync` 的单阶段循环

## BF16: 自写 Tensor Core vs cuBLAS

这一节专门回答一个更现实的问题：

- 我们现在这个 `bf16_gemm_tensor_core`
- 和成熟第三方实现到底差多少

这里用 `bf16_gemm_cublas` 作为基线。它不是 PyTorch 包装层，而是更接近 PyTorch eager 在 CUDA 上常见的底层 GEMM 路径。

### Benchmark 对比

同样是 `1024 x 1024 x 1024`：

| binary | avg_ms | tflops |
| --- | ---: | ---: |
| `bf16_gemm_tensor_core` | `0.0571` | `37.63` |
| `bf16_gemm_cublas` | `0.0181` | `118.71` |

也就是说：

- `cuBLAS` 大约快 `3.2x`
- 这已经足以说明当前自写 kernel 仍然是教学型 WMMA kernel，不是接近库级优化上限的实现

### NCU 对比

| binary | kernel | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16_gemm_tensor_core` | `wmma_gemm_16x16x16_kernel` | `67.55` | `72.50` | `30.79` | `63.80` | `72.50` | `64.83` |
| `bf16_gemm_cublas` | `ampere_s16816gemm_bf16_128x64_ldg8_stages_32x6_nn` | `20.83` | `60.45` | `36.87` | `30.29` | `60.45` | `8.29` |

从画像上看，最重要的不是某一个百分比谁更高，而是整个结构已经完全不同了。

### 自写 kernel 现在卡在哪

`bf16_gemm_tensor_core` 的画像还是很典型的 feeding-path 受限：

- `L1/TEX` 仍然偏高
- 主 stall 仍然是 `L1TEX scoreboard dependency`
- 还有 `uncoalesced global accesses`

换句话说：

- 现在不是 Tensor Core 不够快
- 而是我们还在用比较朴素的方式给 Tensor Core 喂数

### cuBLAS 为什么能快这么多

`cuBLAS` 这次抓到的主 kernel 名是：

- `ampere_s16816gemm_bf16_128x64_ldg8_stages_32x6_nn`

光从名字就已经能读出一些信号：

- `128x64`
  - tile 明显比我们现在的 `16x16` warp 级小块更大
- `ldg8`
  - load 组织更激进
- `stages_32x6`
  - 有更深的多阶段 pipeline

从 `ncu` 指标也能看出来：

- `Duration` 直接压到 `20.83 us`
- `L1/TEX` 只剩 `30.29%`
- `Tensor` pipeline 已经成为主利用管线
- 即使 occupancy 只有 `8.29%`，它仍然远快于我们

这一点很关键：

- 高性能 GEMM 不是简单追求 occupancy 高
- 成熟库很多时候反而会用更大的 tile、更多寄存器、更深的 shared-memory pipeline
- 最终把 occupancy 压低，但把单个 block 的有效工作量和 Tensor Core 利用率做得更高

### 这组对比最该记住什么

1. 我们当前版本已经证明：
   - `B` tile block 内复用
   - 减轻 `L1/TEX` 压力
   - 确实能带来正收益
2. 但和成熟库相比，差距已经不在“会不会用 WMMA”这一层。
3. 真正的差距在：
   - tile 设计
   - 多阶段 pipeline
   - load/store 组织
   - 对 shared memory / register / Tensor Core feeding 的整体协同
4. 所以后面如果还想继续逼近库级性能，方向应该是：
   - 从“教学型 WMMA kernel”走向更分层的 tiled GEMM 和更深的 pipeline
   - 而不是继续只在当前这版单阶段循环上做小修小补

## 先看两大类的本质区别

### 1. `CUDA core` 版是典型的 tiled FMA GEMM

当前保留的 `bf16_gemm_cuda_core`，足够代表传统 `cuda_core` tiled GEMM 的画像：

- `Memory Throughput` 大约 `50%`
- `Compute (SM) Throughput` 大约 `52%`
- occupancy 大约 `33%`
- NCU 会把它描述成更偏 `balanced`

这和代码结构是完全一致的：

- shared-memory tiling
- register accumulator
- 标量 `acc += a * b`

也就是说：

- 这个对照组的计算主路径仍然是普通 CUDA core 的 `FMA`
- 可以直接用来和真正的 Tensor Core 路径做本质对照

### 2. `Tensor Core` 版已经明显变成 feeding-path 问题

`bf16/int8/int4` 的 `tensor_core` 版本整体呈现另一种画像：

- `Memory Throughput` 已经到 `84-89%`
- `L1/TEX Cache Throughput` 已经到 `91-95%`
- `Compute (SM) Throughput` 反而只有 `22-28%`
- occupancy 明显更高，尤其 `int4` 到了 `90.70%`

这里的 `L1/TEX` 口径按 [../../../notes/gpu_components.md](../../../notes/gpu_components.md) 里的定义理解。

这说明：

- Tensor Core 的乘加路径已经切对了
- 当前瓶颈不再是“不会算”
- 而是“怎么把数据更顺地喂进去”

## 各组怎么看

### `bf16` CUDA core

- duration 都在 `108-112 us`
- `FMA` 是最高利用的 pipeline
- `global store` 到 L1TEX 的访问模式不理想
- 还有一部分 `uncoalesced global accesses`

更准确地说：

- 这是比较标准的共享内存 tiled GEMM
- 受寄存器、occupancy、store pattern 一起约束

### `bf16` Tensor Core

这组画像很典型：

- duration 大约 `67 us`
- `Memory Throughput` 约 `89%`
- `L1/TEX Cache Throughput` 约 `94%`
- occupancy 大约 `56%`
- top stall 是 `waiting for a scoreboard dependency on a L1TEX operation`
- NCU 明确提示 global load 只有 `16 / 32 bytes per sector` 被有效利用

这说明：

- 现在主要不是 Tensor Core 算得不够快
- 而是 `load_matrix_sync` 前后的 global-memory feeding path 还不够理想

### `int8` Tensor Core

- `duration = 35.84 us`
- `mem % = 85.90`
- `compute % = 22.89`

这组最能说明：

- 低精度存储本身不会自然带来质变
- 真正带来质变的是计算路径从标量 `FMA` 切到 `mma_sync`

不过 `int8` Tensor Core 这版还有一个额外问题：

- shared stores 出现了平均 `4-way bank conflict`
- NCU 也报告了 `uncoalesced shared accesses`

所以这版虽然已经很快，但还不是最终形态。

### `int4` Tensor Core

- `duration = 37.82 us`
- `mem % = 82.71`
- `compute % = 27.80`
- occupancy 达到 `90.70%`
- top stall 仍然是 `L1TEX scoreboard dependency`

这说明：

- `4090 / sm_89` 上 `int4` Tensor Core 路径确实能跑起来
- 而且确实比 `int4` 的 CUDA core 对照版快很多
- 但它也同样没有摆脱 feeding-path 限制

## 这个实验最该记住的结论

1. 低精度数据类型本身，不等于 Tensor Core GEMM。
2. `bf16_gemm_cuda_core` 这个对照组已经足够说明：如果 kernel 还是普通标量 `FMA`，它和 Tensor Core 路线的瓶颈画像会很不一样。
3. 一旦切到 `WMMA` / Tensor Core，吞吐会明显上一个台阶。
4. 但 Tensor Core 版当前最主要的限制不是乘加单元，而是 global load、L1TEX、layout、shared-memory feeding path。
5. `int8/int4` Tensor Core 已经跑通，但如果继续优化，最值得做的不是改数学公式，而是改：
   - tile mapping
   - global load coalescing
   - shared-memory layout
   - bank conflict
   - fragment feeding path
