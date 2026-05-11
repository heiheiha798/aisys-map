# Experiments

这个目录放两类内容：

1. 最小实验
   用小脚本验证一个系统机制或一个 kernel 写法。
2. 开源 case study
   通过完整开源项目观察真实 inference runtime / kernel optimization / serving engine 如何组织。

这两类内容不要混在同一条编号体系里。

## Kernel Learning Tracks

这些目录服务于 kernel 和 operator 的基础学习：

- `triton_kernels/`
  自己写的教学版 Triton kernel，从 elementwise、reduction、RoPE、online softmax 到 attention / flash attention。
- `cuda_kernels/`
  自己写的教学版 CUDA kernel，用来理解 CUDA 编程对象、memory hierarchy、Tensor Core 和 NCU 指标。
- `triton-tutorials/`
  Triton 官方 tutorial 的本地注释版镜像，用来和自己的教学 kernel 对照。
- `tilelang-puzzles/`
  TileLang 官方 puzzles 的本地 fork 和学习记录，用来补齐 `T.Kernel`、`T.Parallel`、`T.Serial`、`T.Pipelined`、`alloc_shared`、`alloc_fragment`、`T.reduce_*`、`T.gemm` 这一套 DSL 抽象。
  它更偏 kernel DSL 入门，不直接承担 serving runtime 的系统学习任务。
- `KDT-DSL/`
  HPCGame Kernel Design Trial 的 DSL 和题目仓库，用来学习 tile-based kernel DSL、software pipeline、SPM / fragment / load-store / compute overlap，以及“带硬件模型的教学型 kernel 编程”。
  它更偏 `kernel / compiler runtime` 抽象和数据流训练，不直接承担 serving engine 学习任务。
- `SGEMM_CUDA/`
  一个相对独立的 CUDA SGEMM 优化仓库，用来集中学习 block tiling、warp tiling、vectorized load/store、shared memory bank conflict、double buffering 和 `ncu` 指标分析。
  它比 `cuda_kernels/11_gemm` 更偏“完整 case study + profiler 驱动优化”，适合作为 GEMM 优化专项训练。

当前边界是：

- `Triton` 仍然需要系统学习，因为它是 inference infra 中很重要的 kernel 抽象。
- 手写 CUDA 保留为理解硬件和 profiler 的基础，不继续作为 repo 主线追极限性能。

## Inference System Mini Experiments

这些目录用最小脚本理解 inference 系统机制：

- `vanilla_transformer/`
- `attention_variants/`
- `attention_patterns/`
- `hf_inference/`
- `continuous_batching/`
- `quantization/`
- `parallel/`

这些实验的目标不是追真实性能，而是把数据流、状态流转和系统边界讲清楚。

## Open Source Case Studies

这些目录是开源项目 submodule，和最小教学实验并列，但定位不同：

- `nano-vllm/`
  轻量 vLLM 风格 inference engine，用来学习 serving engine、scheduler、KV cache 和 runtime 组织。
- `nanoPD/`
  轻量 prefill/decode disaggregation 项目，用来学习 PD 分离和推理系统拆分。
- `flash-deepseek-v2-lite/`
  DeepSeek-V2-Lite decode path 的 Triton optimization case study，用来学习真实模型上的 kernel fusion、small GEMV、grouped batching、MoE route grouping、CUDA graph 和 `nsys` / `ncu` profiling workflow。

`flash-deepseek-v2-lite/` 不属于 `triton_kernels/` 的教学编号体系。它是学完基础 Triton kernel 后，用来观察真实工程如何把 kernel optimization 接到 inference runtime 的案例。

`KDT-DSL/` 和 `tilelang-puzzles/` 更偏 DSL / kernel 编程抽象训练；`SGEMM_CUDA/` 更偏经典 CUDA GEMM 优化与 profiler 分析；`flash-deepseek-v2-lite/` 更偏真实推理路径上的 Triton case study。它们都放在 `experiments/`，但承担的学习任务不同。

## Submodule Usage

初始化或更新 submodule：

```bash
git submodule update --init --recursive
```

只拉某个 case study：

```bash
git submodule update --init experiments/flash-deepseek-v2-lite
```

或只拉 `KDT-DSL`：

```bash
git submodule update --init experiments/KDT-DSL
```

或只拉 `SGEMM_CUDA`：

```bash
git submodule update --init experiments/SGEMM_CUDA
```
