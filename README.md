# AISys Map

一张持续维护的 `AI infrastructure` 知识地图，外加一组最小实验和开源 case study。

它的目的不是罗列名词，也不是追热门系统，而是建立一张足够稳定的系统分层图：把 `hardware`、`runtime`、`compiler`、`serving`、`cache`、`scheduler`、`orchestration`、`observability` 放回同一张图里，从而能够判断——

- 任意一篇 inference / memory / scheduling 论文处在系统栈的哪一层
- 一个工作到底在优化 `compute`、`memory`、`interconnect`、`runtime`、`scheduler` 还是 `state management`
- 一个方向是 infra 核心创新、工程集成，还是 deployment 假设变化下的重新组合

它既是我的学习轨迹，也希望逐步长成一个可公开阅读、可复用的开源知识仓库。

## 当前边界

repo 已主动收束到 **inference systems** 主线，关注：单机到多卡 inference runtime、serving engine 与 scheduler、KV cache / paged KV、prefill / decode / continuous batching / chunked prefill、weight-only quantization、backend / engine / kernel 的边界、profiling 与系统判断。

暂不展开（保留地图位置，不作为主学习目标）：

- 训练系统全景与训练并行策略细节
- HTTP server / web serving 设计
- 手写 CUDA 的极限 kernel 优化

`Triton` 是例外：它仍作为重点系统学习，因为它是理解 inference execution path、operator fusion 和 profiling 的关键 kernel 抽象。手写 CUDA 只保留理解硬件和 profiler 所需的基础实验。

## 目录结构

```
notes/                  分主题技术笔记：GPU 组织、CUDA 编程对象、Tensor Core/WMMA、kernel 分类
roadmap/                问题驱动的 foundations checklist
experiments/            最小实验 + 开源 case study（见下）
```

`experiments/` 按**学习顺序**分成 5 组（详见 [experiments/README.md](experiments/README.md)）：

| 分组 | 目录 | 用途 |
|------|------|------|
| 模型入门 | `01_model_basics/` | `vanilla_transformer` `attention_variants` `attention_patterns`：只看最朴素的数据流 |
| Kernel 入门 | `02_kernel_intro/` | `cuda_kernels` `triton_kernels` `triton-tutorials`：教学版 kernel 与官方 tutorial 对照 |
| Kernel 深入 | `03_kernel_advanced/` | `tilelang-puzzles`* `KDT-DSL`* `SGEMM_CUDA`*：kernel DSL puzzles 与 GEMM 优化专项 |
| Inference 机制 | `04_inference_system/` | `hf_inference` `continuous_batching` `quantization` `parallel`：真实推理路径、调度、量化、并行 |
| 实战演练 | `05_case_studies/` | `nano-vllm`* `nanoPD`* `flash-deepseek-v2-lite`*：完整开源 engine / PD 分离 / 真实 decode path Triton 优化 |

`*` 为 git submodule。初始化：`git submodule update --init --recursive`，或只拉某一个：`git submodule update --init experiments/<group>/<name>`。

## Knowledge Map：12 个模块

把 `AI infra` 粗分成 12 个模块。带 ★ 的是当前 repo 的主线，其余保留地图位置。

| | 模块 | 核心问题 | 关键词 |
|---|------|----------|--------|
| A | Hardware / Memory / Interconnect | GPU 强在哪；HBM/L2/shared/register 层级意味着什么；compute/memory/interconnect-bound 如何区分 | roofline, HBM, SM, occupancy, NVLink, RDMA |
| B | Kernel / Operator / Compiler Runtime ★ | kernel 优化到底优化什么；CUDA runtime/PTX/SASS 关系；Triton/CUTLASS/手写 CUDA 各处什么抽象层；graph capture / fusion / allocator | CUDA, Triton, CUTLASS, CUDA Graph, fusion, allocator |
| C | Distributed Execution / Collective | 为什么绕不开通信；all-reduce/all-gather/reduce-scatter 的角色；NCCL 负责什么 | NCCL, all-reduce, all-gather, ring, tree |
| D | Training Parallelism | DP/TP/PP/ZeRO/EP/FSDP 各切什么；activation/optimizer/gradient/param 在哪占资源 | tensor parallel, pipeline parallel, FSDP, ZeRO, Megatron |
| E | Inference Fundamentals ★ | prefill 与 decode 为何本质不同；decode 为何 memory-bound；KV cache 的逻辑作用与物理成本 | prefill, decode, KV cache, paged KV, MLA |
| F | Serving Runtime / Scheduler ★ | request 如何进 engine；continuous batching / chunked prefill 为何出现；scheduler 的目标是 TTFT/ITL/throughput 还是 goodput | continuous batching, chunked prefill, goodput, SLO-aware scheduling |
| G | Serving Primitive / Backend / Engine ★ | engine 为何不自己重写所有 kernel；backend、engine、kernel 为何不能混成一层 | FlashInfer, vLLM, SGLang, TensorRT-LLM |
| H | Memory System / Cache / State Externalization ★ | KV/activation/param/optimizer 各是什么性质的状态；prefix cache / KV externalization / offload 的边界 | prefix caching, LMCache, KV transport, disaggregation |
| I | Model Artifact / Weight Format / Loading | checkpoint/shard/weight format 如何影响部署与冷启动 | safetensors, GGUF, engine build, cold start |
| J | Cluster Scheduling / Orchestration / Reliability | job/serving/cluster scheduler 的边界；multi-tenant/quota/preemption | Kubernetes, Ray, Slurm, autoscaling, preemption |
| K | Observability / Profiling / Evaluation ★ | 该看哪些指标；单机 profiling / 分布式 trace / 线上 observability 各解决什么 | Nsight, PyTorch profiler, TTFT, ITL, P99, goodput |
| L | Sparse / MoE / Speculation / Edge | sparse 的瓶颈是 compute 还是 routing/transfer；speculative decoding 真正省什么；edge 与 datacenter 瓶颈差异 | MoE, expert cache, speculative decoding, edge inference |

## 四层视角

12 个模块可压成 4 层：

1. **物理资源层** — hardware、memory hierarchy、interconnect (A)
2. **单机执行层** — kernel/operator、compiler runtime、inference fundamentals、weight loading (B, E, I)
3. **多卡与系统抽象层** — communication、training system、serving runtime、backend/engine、cache/state (C, D, F, G, H)
4. **生产基础设施层** — cluster scheduling、reliability、observability、方向统一 (J, K, L)

## 推荐学习顺序

1. Hardware / memory / interconnect fundamentals
2. CUDA / Triton / compiler runtime / graph execution
3. Attention / KV / prefill-decode fundamentals
4. PagedAttention / vLLM / continuous batching / chunked prefill
5. FlashAttention / backend primitive / engine layering
6. Disaggregation / KV transport / state externalization
7. Profiling / evaluation / system judgement
8. Speculation / quantization / edge / MoE 回看与统一

先补硬件与 runtime，否则看 engine 时容易误判瓶颈；先看系统共性，再看具体 engine 的设计差异。

## 使用方式

这是一个长期迭代的知识工程，不是一次性看完的教程。建议每个模块至少做四件事，并回答三个固定问题：

**四件事**：写 1 份技术笔记 · 画 1 张系统图 · 做 1 个最小实验 · 回答下面 3 个问题。

**三个问题**：
1. 这个模块优化的是哪一层？
2. 它和其他模块的边界是什么？
3. 它给未来研究方向带来什么约束？

并强制给每个模块贴一个主瓶颈标签：`compute-bound` / `memory-bound` / `interconnect-bound` / `runtime-overhead-bound` / `scheduler-bound` / `state-management-bound` / `deployment-bound`。

## 这个项目不希望做成什么

- 不是热门名词清单，不是系统论文摘要仓库
- 不只堆源码链接而不做抽象，也不只做抽象而不做最小验证
