# Experiments

这个目录按**学习顺序**分成 5 组，从模型基础一路到真实开源 engine。每组内既有自己手写的最小实验，也有作为 case study 的开源 submodule（标 `*`）。

```
01_model_basics/      先把模型本身的数据流看懂（不碰 kernel、不碰真实框架）
02_kernel_intro/      入门 kernel：自己写教学版 + 官方 tutorial 对照
03_kernel_advanced/   深入 kernel：kernel DSL puzzles 与 GEMM 优化专项
04_inference_system/  inference 系统机制：真实模型路径、调度、量化、并行
05_case_studies/      实战演练：完整开源 inference engine 与 PD 分离
```

教学实验和开源 case study 定位不同，不要混进同一条编号体系：前者重逻辑、弱性能，只为讲清机制；后者是观察真实工程如何组织。

## 01_model_basics — 模型入门

不用 Hugging Face、不用真实权重，只看最朴素的数据流。

- `vanilla_transformer/` — 最朴素的 decoder-only block：embedding → QKV → score → mask → softmax → PV → FFN → residual/LN，单头、无 RoPE、无 KV cache。
- `attention_variants/` — 在上面的基础上引入 `MHA / MQA / GQA / MLA` 等变体，仍是 dummy weight、只求逻辑正确。
- `attention_patterns/` — 一条独立主线：`dense / window / sparse / linear` attention，关注 token 间 connectivity 与 execution pattern。

## 02_kernel_intro — Kernel 入门

理解一个 kernel 在做什么、最基本的并行思路与访存模式。

- `cuda_kernels/` — 教学版原生 CUDA C++ kernel（`01`~`12`，重逻辑弱性能），用来理解 CUDA 编程对象、memory hierarchy、Tensor Core 和 NCU 指标。
- `triton_kernels/` — 自己写的教学版 Triton kernel（`01`~`12`），从 elementwise、reduction、RoPE、online softmax 到 attention / flash attention，并和 `cuda_kernels/` 对应编号的 NCU 结果对照。
- `triton-tutorials/` — Triton 官方 tutorial 的本地注释版镜像，用来和自己的教学 kernel 对照。

## 03_kernel_advanced — Kernel 深入

偏 kernel DSL 抽象训练和经典 GEMM 优化，不直接承担 serving 系统学习。

- `tilelang-puzzles/`* — TileLang 官方 puzzles 的 fork，补齐 `T.Kernel / T.Parallel / T.Pipelined / alloc_shared / alloc_fragment / T.gemm` 这套 DSL 抽象。
- `KDT-DSL/`* — HPCGame Kernel Design Trial 的 DSL 与题目仓库，学习 tile-based kernel DSL、software pipeline、SPM / fragment / load-store / compute overlap，以及“带硬件模型的教学型 kernel 编程”。
- `SGEMM_CUDA/`* — 独立的 CUDA SGEMM 优化仓库，集中练 block/warp tiling、vectorized load/store、shared memory bank conflict、double buffering 与 `ncu` 指标分析。比 `cuda_kernels/11_gemm` 更偏“完整 case study + profiler 驱动优化”。

## 04_inference_system — Inference 系统机制

用最小脚本把真实推理路径上的状态流转和系统边界讲清楚，不追真实性能。

- `hf_inference/` — 基于 `transformers` 和真实模型的最小推理实验，作为前面教学概念的“真实 backend 对照物”（single request decode、batching、chunked prefill、CUDA graph decode）。
- `continuous_batching/` — 手搓 toy scheduler，把“request 如何动态进入、batch 每轮如何重组”建立成稳定心智模型。
- `quantization/` — 在 HF backend 上对真实 `Qwen3-0.6B` 做 4-bit weight-only quantization 接 CUDA Graph，对照 `bf16` 看 decode 吞吐（含 nsys 结果）。
- `parallel/` — 推理侧最值得优先理解的两类并行：`TP`（column/row parallel）与 `EP`（expert parallel），用最小整数例子讲清“张量怎么切、结果怎么合”。

## 05_case_studies — 实战演练

学完基础后，观察完整开源工程如何把这些机制接到一起。

- `nano-vllm/`* — 轻量 vLLM 风格 inference engine，学习 serving engine、scheduler、paged KV cache、CUDA Graph、continuous batching 如何组织成一个能顺着读下去的实现。
- `nanoPD/`* — 从零实现的 prefill/decode disaggregation engine，学习 PD 分离与推理系统拆分。
- `flash-deepseek-v2-lite/`* — DeepSeek-V2-Lite decode path 的 Triton optimization case study：真实模型上的 kernel fusion、small GEMV、grouped batching、MoE route grouping、CUDA graph 和 `nsys` / `ncu` profiling workflow。

## Submodule Usage

标 `*` 的为 git submodule。初始化或更新全部：

```bash
git submodule update --init --recursive
```

只拉某一个 case study（注意路径已带分组前缀）：

```bash
git submodule update --init experiments/05_case_studies/flash-deepseek-v2-lite
git submodule update --init experiments/03_kernel_advanced/SGEMM_CUDA
```
