# AISys Map

`AISys Map` 是一个同时具备两层定位的 repo：

- 个人学习记录
- 开源知识地图项目

一方面，它会持续记录我对 `AI infrastructure / AI systems` 的学习、理解、笔记、图和最小实验。  
另一方面，我也希望把这些内容整理成一份可以公开演进、能帮助别人建立系统地图的开源 repo。

它的目标不是罗列零散名词，也不是只追热门系统，而是建立一张足够稳定的 `AI infra knowledge map`，帮助你把 `hardware`、`runtime`、`compiler`、`distributed training`、`serving`、`cache`、`scheduler`、`cluster orchestration`、`observability` 放回同一张系统图里。

如果你在读系统论文、看 serving engine 源码、分析训练/推理 bottleneck、判断新方向是否成立时，经常遇到这些问题：

- 知道一些局部点，但缺少总地图
- 看过很多实现细节，但不知道它处在系统栈哪一层
- 容易把 `kernel`、`backend`、`engine`、`scheduler`、`orchestrator` 混成一层
- 会做局部优化，但不容易判断一个方向的真实系统价值

那么这个项目就是为这些问题服务的。

## Repo Positioning

这个 repo 会同时保留两种属性。

### 1. 个人学习记录

这里会保留比较真实的学习过程，包括：

- 阶段性知识地图
- 分主题技术笔记
- 阅读后的理解重写
- 最小验证实验
- 对研究方向的判断和修正

它不追求一开始就“像教材一样完整”，而是强调：

- 把学习过程结构化
- 把系统理解显式化
- 把模糊认识逐步压成稳定地图

### 2. 开源知识地图项目

它也不只是私人笔记仓库。  
这个 repo 的另一个目标，是逐步变成一个适合公开阅读、持续扩展、可被他人复用的 `AI systems knowledge map`。

因此它会尽量朝下面这个方向演进：

- 模块边界清晰
- 结构可维护
- 图和文字能相互印证
- 学习路线、阅读清单、实验验证可以复用

## 项目目标

这个项目希望逐步产出：

1. 一张可持续维护的 `AI infra knowledge map`
2. 一套围绕 `AI systems` 的结构化中文技术笔记
3. 一组最小实验，用来验证关键系统机制
4. 一套判断系统工作的固定框架：先看瓶颈层，再看抽象层，再看 deployment 假设

最终目标不是“学会某个库”，而是形成下面这些能力：

- 能把任何一篇 training / inference / memory / scheduling / distributed paper 放回地图里
- 能说清一个工作到底是在优化 `compute`、`memory`、`interconnect`、`runtime`、`compiler`、`scheduler`、还是 `cluster control plane`
- 能区分一个方向到底是 infra 核心创新、工程集成、还是 deployment 假设变化下的重新组合

## Knowledge Map

目前把 `AI infra` 粗分成下面 12 个模块。

### A. Hardware Architecture / Memory Hierarchy / Interconnect

核心问题：

- GPU 为什么强，强在什么地方
- `HBM / L2 / shared memory / register` 的层级意味着什么
- 什么是 `compute-bound / memory-bound / interconnect-bound`
- `PCIe / NVLink / NVSwitch / RDMA / InfiniBand` 影响哪些系统设计

关键词：

- `roofline`
- `HBM`
- `SM`
- `occupancy`
- `PCIe`
- `NVLink`
- `RDMA`

### B. Kernel / Operator / Compiler Runtime

核心问题：

- kernel 优化到底在优化什么
- `CUDA runtime / driver / PTX / SASS` 之间是什么关系
- `Triton`、`CUTLASS`、手写 CUDA 分别处在什么抽象层
- graph capture / fusion / codegen / allocator 为什么重要

关键词：

- `CUDA`
- `Triton`
- `CUTLASS`
- `PTX`
- `CUDA Graph`
- `fusion`
- `allocator`

### C. Distributed Execution / Collective Communication

核心问题：

- 为什么训练和推理都绕不开通信
- `all-reduce / all-gather / reduce-scatter / broadcast` 的角色分别是什么
- `NCCL` 在系统里到底负责什么
- 通信模式为什么决定并行策略的扩展性

关键词：

- `NCCL`
- `all-reduce`
- `all-gather`
- `reduce-scatter`
- `ring`
- `tree`

### D. Training Parallelism 与训练系统

核心问题：

- `DP / TP / PP / ZeRO / EP / FSDP` 分别切的是什么
- activation、optimizer state、gradient、parameter 分别在哪一层占资源
- 训练系统为什么常常先被 memory 和 communication 卡住

关键词：

- `data parallel`
- `tensor parallel`
- `pipeline parallel`
- `FSDP`
- `ZeRO`
- `Megatron-LM`
- `DeepSpeed`

### E. Inference Fundamentals

核心问题：

- prefill 和 decode 为什么本质不同
- attention 在两者下为什么瓶颈不同
- decode 为什么更容易 memory-bound
- KV cache 的逻辑作用和物理成本是什么

关键词：

- `prefill`
- `decode`
- `KV cache`
- `paged KV`
- `long context`
- `MLA`

### F. Serving Runtime 与 Scheduler

核心问题：

- request 如何进入 engine
- continuous batching 为什么重要
- chunked prefill 为什么出现
- prefill/decode 如何竞争 GPU 资源
- scheduler 的目标到底是 TTFT、ITL、throughput 还是 goodput

关键词：

- `continuous batching`
- `chunked prefill`
- `goodput`
- `prefill/decode mixing`
- `SLO/QoS-aware scheduling`

### G. Serving Primitive / Backend Library / Engine

核心问题：

- engine 为什么不自己重写所有 kernel
- backend primitive library 提供哪些能力
- engine 如何把 scheduler、KV memory、backend primitive 连接起来
- backend、engine、kernel 为什么不能混成一层

关键词：

- `FlashInfer`
- `vLLM`
- `SGLang`
- `TensorRT-LLM`
- `llama.cpp`
- `runtime abstraction`

### H. Memory System / Cache / State Externalization

核心问题：

- KV、activation、parameter、optimizer state 分别是什么性质的系统状态
- prefix cache、expert cache、parameter offload、KV externalization 各自的边界是什么
- state 从单进程私有状态变成系统资源后，调度问题会如何变化

关键词：

- `prefix caching`
- `LMCache`
- `KV transport`
- `parameter offload`
- `state externalization`
- `disaggregation`

### I. Model Artifact / Weight Format / Loading Path

核心问题：

- checkpoint、shard、weight format 为什么直接影响部署和启动时间
- model loading path 为什么经常是线上系统的冷启动瓶颈
- `safetensors / GGUF / TensorRT engine` 分别服务于什么场景

关键词：

- `checkpoint sharding`
- `safetensors`
- `GGUF`
- `engine build`
- `cold start`
- `weight layout`

### J. Cluster Scheduling / Orchestration / Reliability

核心问题：

- job scheduler、serving scheduler、cluster scheduler 的边界是什么
- multi-tenant、quota、priority、preemption 如何影响 AI workload
- `Kubernetes / Ray / Slurm` 各自适合什么场景

关键词：

- `Kubernetes`
- `Ray`
- `Slurm`
- `autoscaling`
- `preemption`
- `fault tolerance`

### K. Observability / Profiling / Evaluation

核心问题：

- 一个 AI 系统到底该看哪些指标
- 单机 profiling、分布式 trace、线上 observability 分别解决什么问题
- `TTFT / ITL / P99 / goodput / GPU utilization / memory BW / link BW` 应该如何一起看

关键词：

- `Nsight`
- `PyTorch profiler`
- `trace`
- `TTFT`
- `ITL`
- `P99`
- `goodput`

### L. Sparse / MoE / Speculation / Edge 路线统一

核心问题：

- sparse model 的瓶颈是 compute 还是 routing / memory / transfer / orchestration
- speculative decoding 真正省的是哪一部分
- edge inference 和 datacenter serving 的瓶颈有什么本质不同
- MoE / speculation / offload / quantization / disaggregation 如何放回同一张图

关键词：

- `MoE`
- `expert cache`
- `offloaded MoE`
- `speculative decoding`
- `draft/target`
- `edge inference`

## 四层视角

这 12 个模块可以进一步压缩成 4 层。

### 第 1 层：物理资源层

- hardware
- memory hierarchy
- interconnect

### 第 2 层：单机执行层

- kernel / operator
- compiler runtime
- inference fundamentals
- weight loading path

### 第 3 层：多卡与系统抽象层

- communication
- training system
- serving runtime
- backend / engine
- cache / state management

### 第 4 层：生产基础设施层

- cluster scheduling / orchestration
- reliability
- observability / evaluation
- research/product direction unification

## 推荐学习顺序

推荐按下面的顺序建立地图：

1. `Hardware / memory / interconnect fundamentals`
2. `CUDA / Triton / compiler runtime / graph execution`
3. `Distributed communication basics`
4. `Training parallelism and training system map`
5. `Attention / KV / prefill-decode fundamentals`
6. `PagedAttention / vLLM / continuous batching / chunked prefill`
7. `FlashAttention / FlashInfer / SGLang / TensorRT-LLM`
8. `Prefix caching / state externalization / disaggregation`
9. `Cluster scheduling / observability / production concerns`
10. `Speculation / llama.cpp / quantization / edge / MoE` 回看与统一

理由：

- 先补硬件与通信，否则看 runtime 时容易误判瓶颈
- 先补训练和推理的共同基础，再进入 serving 主线
- 先看系统共性，再看具体 engine 的设计差异

## 使用方式

这个 repo 更适合作为一个长期迭代的知识工程，而不是一次性看完的教程。

如果你把它当成个人学习记录来看，重点会是：

- 我是如何搭这张地图的
- 每个模块是如何逐步补齐的
- 哪些理解是阶段性的，哪些已经相对稳定

如果你把它当成开源 repo 来看，重点会是：

- 如何快速进入 `AI infra` 的系统分层
- 每个模块该先理解什么、再理解什么
- 可以复用哪些图、笔记和最小实验

建议每个模块都至少做下面几件事：

1. 写 1 份自己的技术笔记
2. 画 1 张系统图
3. 做 1 个最小实验
4. 回答 3 个问题

固定回答的问题是：

1. 这个模块优化的是哪一层
2. 它和其他模块的边界是什么
3. 它给未来研究方向带来的约束是什么

另外，建议强制给每个模块贴一个主瓶颈标签：

- `compute-bound`
- `memory-bound`
- `interconnect-bound`
- `runtime-overhead-bound`
- `scheduler-bound`
- `state-management-bound`
- `deployment-bound`

## 这个项目不希望做成什么

- 不希望变成热门名词清单
- 不希望变成“系统论文摘要仓库”
- 不希望只堆源码链接而不做抽象
- 不希望只做抽象而不做最小验证

## Roadmap

接下来可以逐步补齐：

- `notes/`: 各模块技术笔记
- `diagrams/`: 知识图、路径图、系统分层图
- `experiments/`: 最小 benchmark 和验证脚本
- `reading/`: 分主题阅读清单
- `roadmap/`: 分周学习路线和阶段性总结

## Why This Repo

很多人会做一些局部优化，也会看一些系统源码，但很难把它们放回同一张图里。  
这个 repo 想解决的正是这个问题：让 `AI systems` 的学习和研究不再只是局部点积累，而变成一张可维护、可扩展、可复用的知识地图。

它既是我的学习轨迹，也是一个希望能逐步长成公开项目的知识仓库。
