# Roadmap

这份 roadmap 用来约束 `aisys-map` 的学习范围和优先级。它不是完整问题库，也不是待办清单；它只回答三件事：

1. 当前主线是什么
2. 哪些模块只保留位置感
3. 每个模块先问哪些关键问题

当前 repo 已经收束到 **inference systems**：从硬件和 kernel 执行模型出发，理解 prefill/decode、KV cache、serving runtime、backend primitive、state management 和 profiling。

## 优先级

| 优先级 | 模块 | 处理方式 |
|---|---|---|
| 主线 | A Hardware / Memory、B Kernel / Runtime、E Inference Fundamentals、F Serving Runtime、G Backend / Engine、H State / Cache、K Profiling / Evaluation | 需要笔记、实验或 case study 支撑 |
| 辅助地图 | C Collective Communication、I Model Artifact / Loading、J Cluster / Reliability、L MoE / Speculation / Edge | 知道位置和边界，按需要补 |
| 只保留位置感 | D Training Parallelism | 不作为当前实验主线 |

`Triton` 仍然保留在主线里，因为它连接 kernel、operator fusion、backend path 和 profiling。手写 CUDA 保留为理解硬件和 profiler 的基础实验，不继续追求极限优化。

## A. Hardware / Memory / Interconnect

目标：建立性能判断的物理直觉。

关键问题：

- GPU 和 CPU 的根本差别是什么？
- `SM / warp / block / thread` 如何对应执行行为？
- `register / shared memory / L1-TEX / L2 / VRAM` 的边界是什么？
- 什么情况下优先怀疑 compute-bound、memory-bound 或 interconnect-bound？
- 为什么 peak FLOPS / peak bandwidth 不能直接代表真实性能？

对应材料：

- [notes/gpu_components.md](notes/gpu_components.md)
- [notes/cuda_programming_objects.md](notes/cuda_programming_objects.md)

## B. Kernel / Operator / Compiler Runtime

目标：理解一个 operator 如何落到 kernel、runtime 和 backend path。

关键问题：

- operator 和 kernel 的关系是什么？
- kernel launch、CUDA Graph、allocator、fusion 分别解决什么 overhead？
- Triton、CUTLASS、手写 CUDA 各处在哪个抽象层？
- 单个 kernel 更快，为什么不一定带来端到端更快？
- Tensor Core / WMMA / MMA 到底是哪条计算路径？

对应材料：

- [notes/basic_kernel_categories.md](notes/basic_kernel_categories.md)
- [notes/cuda_kernel_advanced.md](notes/cuda_kernel_advanced.md)
- [notes/cuda_tensor_core_wmma.md](notes/cuda_tensor_core_wmma.md)
- [02_kernel_intro](02_kernel_intro)
- [03_kernel_advanced](03_kernel_advanced)

## C. Collective Communication

目标：保留多卡系统的位置感，避免把通信问题误判成 kernel 问题。

关键问题：

- all-reduce、all-gather、reduce-scatter 分别在交换什么？
- latency 和 bandwidth 对小消息/大消息的影响有什么不同？
- PCIe、NVLink、RDMA 拓扑如何影响并行策略？
- 通信和计算 overlap 为什么难？

当前只作为辅助地图，不单独展开实验。

## D. Training Parallelism

目标：只保留训练系统的概念边界。

关键问题：

- DP、TP、PP、EP、ZeRO、FSDP 分别切什么？
- parameter、gradient、optimizer state、activation 的状态性质有什么不同？
- 训练和推理在状态管理上的核心差别是什么？

当前不深入训练系统，也不把训练并行作为 repo 主线。

## E. Inference Fundamentals

目标：理解 prefill/decode、attention 和 KV cache 的系统含义。

关键问题：

- prefill 和 decode 为什么本质不同？
- decode 为什么常常更像 memory traffic 问题？
- KV cache 的大小怎么估算，为什么它会成为 serving 的中心状态？
- paged KV、block table、prefix cache 分别解决什么？
- MLA、GQA、MQA 这类结构变化会怎样影响 serving 系统？

对应材料：

- [01_model_basics](01_model_basics)
- [04_inference_system/hf_inference](04_inference_system/hf_inference)

## F. Serving Runtime / Scheduler

目标：理解 request 如何进入 engine，以及 scheduler 在优化什么。

关键问题：

- static batching 和 continuous batching 的差别是什么？
- prefill 和 decode 为什么会竞争 GPU 资源？
- chunked prefill 在平衡什么？
- TTFT、ITL、throughput、goodput、P99 为什么会互相拉扯？
- SLO-aware scheduling 为什么不能只看平均吞吐？

对应材料：

- [04_inference_system/continuous_batching](04_inference_system/continuous_batching)
- [05_case_studies/nano-vllm](05_case_studies/nano-vllm)

## G. Backend Primitive / Serving Engine

目标：把 engine、backend primitive 和 kernel library 分清楚。

关键问题：

- serving engine 为什么不自己重写所有 kernel？
- backend、engine、kernel 的边界是什么？
- vLLM 的 PagedAttention 更像 kernel 优化还是 memory abstraction？
- FlashInfer / TensorRT-LLM / SGLang 分别更靠近哪一层？
- prefill 和 decode 为什么常常走不同 backend path？

对应材料：

- [05_case_studies/nano-vllm](05_case_studies/nano-vllm)
- [05_case_studies/flash-deepseek-v2-lite](05_case_studies/flash-deepseek-v2-lite)

## H. Memory System / Cache / State Externalization

目标：把 KV cache、prefix cache、parameter/offload、PD 分离放到统一状态视角里。

关键问题：

- parameter、activation、optimizer state、KV cache 各是什么性质的状态？
- 哪些状态适合缓存，哪些状态适合 externalization？
- prefix caching 和 decode acceleration 有什么不同？
- prefill-decode disaggregation 为什么不是简单拆成两台机器？
- 状态从进程私有变成系统资源后，会引入哪些调度问题？

对应材料：

- [04_inference_system/quantization](04_inference_system/quantization)
- [05_case_studies/nanoPD](05_case_studies/nanoPD)

## I. Model Artifact / Loading Path

目标：知道权重格式和加载路径会影响部署，而不是只影响文件后缀。

关键问题：

- checkpoint、shard、safetensors、GGUF、engine build 的边界是什么？
- cold start 慢在加载、layout 转换、graph capture，还是 engine build？
- weight-only quantization 改变的是 storage、bandwidth、kernel path，还是全部？

当前作为辅助地图，需要时再展开。

## J. Cluster / Orchestration / Reliability

目标：知道 serving engine scheduler 和 cluster scheduler 不是同一层。

关键问题：

- job scheduler、serving scheduler、cluster scheduler 的边界是什么？
- placement、quota、priority、preemption 各解决什么问题？
- LLM serving 的 autoscaling 为什么比普通 web service 更难？
- retry、rolling update、fault tolerance 在有状态 serving 下有什么额外成本？

当前保留系统位置，不展开 HTTP service 细节。

## K. Observability / Profiling / Evaluation

目标：能用指标和 profiler 判断系统瓶颈，而不是只看平均吞吐。

关键问题：

- Nsight Compute、Nsight Systems、PyTorch profiler 各适合看什么？
- kernel-level profiling 和 system-level tracing 的区别是什么？
- TTFT、ITL、P99、goodput、GPU utilization、bandwidth utilization 应该如何一起看？
- synthetic workload 和 real workload 会怎样改变结论？
- benchmark 什么时候会误导？

对应材料：

- [02_kernel_intro](02_kernel_intro)
- [05_case_studies/flash-deepseek-v2-lite](05_case_studies/flash-deepseek-v2-lite)

## L. Sparse / MoE / Speculation / Edge

目标：保留下一阶段方向的位置感。

关键问题：

- speculative decoding 真正省的是哪部分成本？
- MoE 的瓶颈更像 compute、memory、communication，还是 orchestration？
- expert parallel 和 expert cache 在系统上意味着什么？
- edge inference 与 datacenter inference 的最大瓶颈差异是什么？
- quantization 在 edge 场景为什么经常变成 memory bandwidth 问题？

当前按 case study 或论文需要补，不作为全部展开的主线。

## 通用判断模板

学任何一个新机制时，都先回答：

1. 它属于哪一层？
2. 它主要优化 compute、memory、communication、runtime、scheduler、state，还是 deployment？
3. 它依赖什么前提？
4. 它不解决什么问题？
5. 换成单卡、多卡、多机，结论会不会变？
6. 换成离线推理、在线 serving、训练，结论会不会变？
