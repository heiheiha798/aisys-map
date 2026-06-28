# Roadmap

这份 roadmap 只约束当前仓库已经放进去的学习材料：`notes/` 下的 5 篇概念笔记，以及根目录下的 5 组 experiment / case study。

它不追求覆盖完整 AI systems 版图，也不把暂时没有材料支撑的方向写成计划。

## 当前主线

| 顺序 | 材料 | 目标 |
|---|---|---|
| 0 | [notes](notes) | 建立读 kernel、profiler 和 inference 实验所需的共同词表 |
| 1 | [01_model_basics](01_model_basics) | 先看懂 decoder-only 模型和 attention 变体的数据流 |
| 2 | [02_kernel_intro](02_kernel_intro) | 用教学 CUDA / Triton kernel 建立并行、访存和基本性能直觉 |
| 3 | [03_kernel_advanced](03_kernel_advanced) | 通过 DSL puzzles 和 SGEMM case study 深入 tile / pipeline / profiler 驱动优化 |
| 4 | [04_inference_system](04_inference_system) | 把真实推理路径里的 batching、KV cache、量化和并行机制拆开 |
| 5 | [05_case_studies](05_case_studies) | 读完整开源 engine / PD 分离 / 真实 decode kernel 优化案例 |

## Notes 范围

`notes/` 只放跨实验复用的基础概念，不放一次性实验结论。

| 笔记 | 解决的问题 | 对应实验 |
|---|---|---|
| [gpu_components.md](notes/gpu_components.md) | `SM / register / shared memory / local memory / L1 / L2 / VRAM` 的边界 | `02_kernel_intro`、`03_kernel_advanced`、`05_case_studies/flash-deepseek-v2-lite` |
| [cuda_programming_objects.md](notes/cuda_programming_objects.md) | `kernel / grid / block / warp / thread / SM` 如何对应 | `02_kernel_intro/cuda_kernels` |
| [cuda_kernel_advanced.md](notes/cuda_kernel_advanced.md) | 索引、launch 维度、occupancy、register、shared memory、divergence 的性能直觉 | `02_kernel_intro`、`03_kernel_advanced` |
| [cuda_tensor_core_wmma.md](notes/cuda_tensor_core_wmma.md) | `CUDA core / Tensor Core / FMA / MMA / WMMA / fragment` 的计算路径边界 | `02_kernel_intro/*/11_gemm`、`03_kernel_advanced/SGEMM_CUDA` |
| [basic_kernel_categories.md](notes/basic_kernel_categories.md) | elementwise、reduction、GEMM、indexed、fused attention 的瓶颈差异 | `02_kernel_intro`、`05_case_studies/flash-deepseek-v2-lite` |

## 01 Model Basics

目标：在不引入真实框架和真实权重的前提下，先把模型结构和 attention 数据流看清楚。

对应材料：

- `vanilla_transformer/`
- `attention_variants/`
- `attention_patterns/`

完成标准：

- 能说清 decoder-only block 的数据流。
- 能区分 `MHA / MQA / GQA / MLA` 改变了什么。
- 能把 dense、window、sparse、linear attention 放到 token connectivity 视角下比较。

## 02 Kernel Intro

目标：用教学 kernel 建立最小 GPU 编程和 profiling 直觉。

对应材料：

- `cuda_kernels/`
- `triton_kernels/`
- `triton-tutorials/`
- `notes/gpu_components.md`
- `notes/cuda_programming_objects.md`
- `notes/cuda_kernel_advanced.md`
- `notes/cuda_tensor_core_wmma.md`
- `notes/basic_kernel_categories.md`

完成标准：

- 能读懂一个 kernel 的 thread/block 映射。
- 能初步判断 kernel 更像 compute-bound、memory-bound、sync-bound 还是 runtime overhead。
- 能把 CUDA 教学 kernel、Triton 教学 kernel 和官方 tutorial 对照起来。

## 03 Kernel Advanced

目标：在更真实的 kernel 优化语境下理解 tile、pipeline、shared memory、fragment 和 profiler 指标。

对应材料：

- `tilelang-puzzles/`
- `KDT-DSL/`
- `SGEMM_CUDA/`
- `notes/cuda_kernel_advanced.md`
- `notes/cuda_tensor_core_wmma.md`

完成标准：

- 能区分教学 kernel 和优化 case study 的目标差异。
- 能解释 SGEMM 优化里 block tile、warp tile、register tile、shared memory 和 Tensor Core 路径各自负责什么。
- 能用 profiler 结果提出下一步假设，而不是只改参数。

## 04 Inference System

目标：把真实推理路径中的状态、调度和 backend 行为拆开看。

对应材料：

- `hf_inference/`
- `continuous_batching/`
- `quantization/`
- `parallel/`

完成标准：

- 能区分 prefill 和 decode 的工作负载特征。
- 能解释 continuous batching 如何动态重组 request。
- 能说明 weight-only quantization 主要改变存储、带宽和 kernel path，而不是自动带来线性加速。
- 能用最小例子说明推理侧 `TP / EP` 在切什么、合什么。

## 05 Case Studies

目标：观察完整工程如何把模型、scheduler、KV cache、backend kernel 和 profiler workflow 接到一起。

对应材料：

- `nano-vllm/`
- `nanoPD/`
- `flash-deepseek-v2-lite/`
- `notes/basic_kernel_categories.md`
- `notes/gpu_components.md`

完成标准：

- 能顺着 nano-vLLM 读出 engine、scheduler、paged KV cache、CUDA Graph 和 continuous batching 的关系。
- 能解释 PD 分离为什么会把状态和调度边界暴露出来。
- 能把 flash-deepseek-v2-lite 里的 decode kernel 优化和前面的 kernel 分类、GPU 存储层级对应起来。

## 暂不展开

这些方向只保留位置感，除非后续进入新的 experiment 或 case study，否则不写成当前 roadmap 主线：

- 训练系统和训练并行
- 集群调度、HTTP service、可靠性工程
- 通用分布式通信专题
- speculative decoding、MoE 系统、edge inference 的完整展开
- 权重格式和部署产物的专门路线

## 通用判断模板

读任何一个实验或 case study 时，先回答：

1. 它对应上面哪一个 experiment 分组？
2. 它依赖哪一篇 `notes/` 概念笔记？
3. 它主要优化 compute、memory、sync、runtime、scheduler、state，还是 deployment？
4. 它是在教学最小机制，还是在观察真实工程取舍？
5. 当前结论能否从单个 kernel 推到端到端系统？如果不能，边界在哪里？
