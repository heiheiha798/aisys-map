# GPU Components

这份笔记只保留写 CUDA / Triton kernel 和读 profiler 时最常用的 GPU 层级边界。

## 最小地图

```text
thread values -> register
block shared data -> shared memory / L1-TEX path
SM cluster reuse -> L2
large tensors -> VRAM / global memory
```

| 对象 | 谁管理 | 关键边界 |
|---|---|---|
| `SM` | 硬件 | 执行 warp 的计算核心簇，不是单个算术单元 |
| `register` | 编译器/线程 | 线程私有、最快、不是 cache；用多了会压 occupancy |
| `local memory` | 编译器/线程语义 | 逻辑线程私有，但通常走慢内存路径；常见于 spill 或大局部数组 |
| `shared memory` | 程序显式管理 | block 内共享、快、小、需要同步；不是 cache |
| `L1/TEX` | 硬件 | 近 SM 的 cache / load-store 通路；不是 shared memory |
| `L2` | 硬件 | 更大、更远、多个 SM 共享的 cache |
| `VRAM / global memory` | 程序分配，硬件访问 | 容量最大，访问最远；权重、大张量、KV cache 主要在这里 |

## 最容易混淆的边界

- `register` 和 `shared memory` 都不是 cache。
- `local memory` 不是“local 所以快”。
- `L1/TEX throughput` 高不等于 shared memory 很忙。
- `global memory` 在性能讨论里通常近似理解成显存路径。
- 靠近计算单元不等于免费；寄存器和 shared memory 用多了都会挤压并发。

## 为什么这对实验重要

| 实验位置 | 主要用到的直觉 |
|---|---|
| `02_kernel_intro/cuda_kernels` | thread/block/SM、shared memory、L1/L2/VRAM、NCU 指标 |
| `02_kernel_intro/triton_kernels` | memory-bound vs compute-bound、访存路径、block/tile 粒度 |
| `04_inference_system/quantization` | weight-only quantization 先影响存储和带宽，不自动线性提升吞吐 |
| `05_case_studies/flash-deepseek-v2-lite` | decode path 常被 memory traffic、kernel granularity 和 graph 内执行路径限制 |

## 读 profiler 时先问

1. 数据主要从哪一层来？
2. 热点值有没有留在寄存器或 shared memory？
3. global memory 访问是否连续、规整、可复用？
4. 当前瓶颈更像 bandwidth、latency、occupancy，还是 kernel launch/runtime overhead？
