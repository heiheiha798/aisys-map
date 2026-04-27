# Foundations Checklist

这是一份面向 `AI infra / AI systems` 入门的“问题驱动 checklist”。

使用方式：

- 每一条都是一个问题
- 你能用自己的话回答，就算基本过关
- 不要求一开始很深，但要尽量覆盖全图
- 如果某个问题完全答不上来，就说明这一块需要补

默认标准：

- 先求覆盖，不先求深入
- 先能区分概念边界，再追实现细节
- 先能回答“它是什么/不是什么/为什么重要”，再看源码和论文

## 当前适用范围

这份 checklist 现在保留为一张大地图，但当前 repo 的实际学习边界已经收束。

当前阶段明确：

- 不深入训练系统
- 不把 HTTP server 作为主线
- 不继续把精力投入手写 CUDA 的极限 kernel 优化
- `Triton` 仍然需要系统学习，因为它是理解 inference execution path、operator fusion 和 profiling 的关键抽象

所以这份 checklist 更适合这样使用：

- A、B、E、F、G、H、K 作为当前主线
- C、I、J、L 作为辅助地图或下一阶段参考
- D 只保留位置感，不作为当前主学习目标

在当前主线里，quantization 更适合归到：

- `E. Inference Fundamentals`
- `B. Kernel / Operator / Compiler Runtime`

之间的交叉主题来理解：

- 一方面它改变 weight storage、memory footprint 和 execution path
- 另一方面它并不自动等于吞吐按位宽比例线性提升

也就是说，这份 checklist 不再表示“这些都要同等深度地学完”，而是表示：

- 你要知道它们在地图里的位置
- 但当前 repo 的主线已经明确收束到 inference systems

---

## A. Hardware Architecture / Memory Hierarchy / Interconnect

### A1. GPU 基础

- 什么是 GPU？
- GPU 和 CPU 的根本差别是什么？
- 为什么 GPU 更适合大规模并行计算？
- 什么是 SM？
- warp 是什么？
- thread、warp、block、grid 分别是什么？
- 为什么 warp 是理解 GPU 执行行为的基础单位？
- 什么叫 SIMT？
- GPU 的“高吞吐”是怎么来的？
- 为什么 GPU 不擅长低并发、强分支、低算量任务？
- occupancy 是什么？
- occupancy 高就一定性能好吗？
- latency hiding 是什么？
- GPU 为什么常常通过更多并发线程去隐藏 latency？

### A2. Memory Hierarchy

- 什么是 register？
- 什么是 shared memory？
- 什么是 L2 cache？
- 什么是 HBM？
- global memory 指的是什么？
- local memory 真的是“本地且快”的内存吗？
- 为什么 shared memory 往往比 global memory 更适合数据复用？
- 为什么 register pressure 会影响性能？
- 什么是 memory hierarchy？
- 为什么同样是“读数据”，不同层级的成本差别很大？
- 什么是 memory access pattern？
- coalesced memory access 是什么？
- 为什么 uncoalesced access 会浪费带宽？
- bank conflict 是什么？
- 为什么 shared memory 也可能成为瓶颈？
- 什么是 cache hit / miss？
- 为什么很多系统问题最后会落到 memory system 上？

### A3. 瓶颈分析

- 什么是 compute-bound？
- 什么是 memory-bound？
- 什么是 interconnect-bound？
- 什么是 arithmetic intensity？
- roofline model 是什么？
- roofline model 解决的核心问题是什么？
- 为什么 peak FLOPS 不能直接代表真实系统性能？
- 为什么 peak bandwidth 也不等于有效带宽？
- effective bandwidth 是什么？
- 为什么 bandwidth utilization 很关键？
- 什么情况下应该优先怀疑 memory-bound？
- 什么情况下应该优先怀疑 interconnect-bound？

### A4. Interconnect

- PCIe 是什么？
- NVLink 是什么？
- NVSwitch 是什么？
- RDMA 是什么？
- InfiniBand 是什么？
- 为什么 GPU 间通信不能只看算力？
- scale-up 和 scale-out 分别是什么？
- 单机多卡和多机多卡的系统问题有什么不同？
- 什么是 GPU peer-to-peer？
- GPUDirect RDMA 在解决什么问题？
- 为什么不同 interconnect 会影响并行策略选择？
- 为什么多卡系统的瓶颈可能从计算跳到通信？

---

## B. Kernel / Operator / Compiler Runtime

### B1. Kernel 基础

- 什么是 kernel？
- operator 和 kernel 有什么关系？
- 为什么“一个算子”在底层可能对应多个 kernel？
- kernel launch 是什么？
- launch overhead 是什么？
- 为什么小 kernel 会被 launch overhead 吃掉？
- 为什么单个 kernel 最优不等于端到端最优？
- 什么是 kernel fusion？
- fusion 为什么可能提升性能？
- fusion 为什么也可能带来复杂度和副作用？

### B2. CUDA 执行栈

- CUDA runtime 是什么？
- CUDA driver 是什么？
- CUDA runtime 和 driver 的关系是什么？
- PTX 是什么？
- SASS 是什么？
- PTX 和 SASS 的差别是什么？
- JIT compilation 在 CUDA 里扮演什么角色？
- 什么是 stream？
- stream 和并发执行有什么关系？
- event 有什么用？
- synchronization 在 GPU 程序里为什么昂贵？

### B3. Triton / CUTLASS / 手写 CUDA

- Triton 是什么？
- CUTLASS 是什么？
- Triton 和手写 CUDA 的关系是什么？
- CUTLASS 更像模板库还是 runtime？
- Triton 适合解决什么问题？
- 什么情况下更适合直接写 CUDA？
- 什么情况下更适合用现成库？
- 为什么 AI infra 里经常同时出现 Triton、CUDA、CUTLASS？

### B4. Graph / Allocator / Runtime Overhead

- CUDA Graph 是什么？
- graph capture 为什么能减少 overhead？
- 什么是 eager execution？
- eager 和 graph execution 的差别是什么？
- allocator 是什么？
- memory allocator 为什么会影响性能？
- 什么是 fragmentation？
- 为什么 runtime overhead 也是系统瓶颈？
- 为什么“内核快了”但系统不一定快？

---

## C. Distributed Execution / Collective Communication

### C1. 通信基础

- 为什么训练和推理都绕不开通信？
- 什么是 collective communication？
- all-reduce 是什么？
- all-gather 是什么？
- reduce-scatter 是什么？
- broadcast 是什么？
- gather 和 all-gather 的差别是什么？
- reduce 和 all-reduce 的差别是什么？
- 为什么不同 collective 对应不同系统代价？
- ring 和 tree 是什么？
- ring all-reduce 为什么经典？
- tree 为什么在某些场景更优？

### C2. NCCL 与拓扑

- NCCL 是什么？
- NCCL 解决的是哪一层的问题？
- 为什么深度学习系统喜欢用 NCCL？
- 拓扑感知通信是什么意思？
- PCIe topology 为什么会影响通信性能？
- NVLink topology 为什么会影响通信性能？
- 为什么同样的并行策略，在不同机器上表现可能完全不同？

### C3. 系统视角

- 通信 latency 和 bandwidth 的区别是什么？
- 为什么小消息和大消息优化方式不同？
- 为什么多机训练/推理常常先被通信卡住？
- 为什么“算法切分”必须和“通信模式”一起看？
- 通信重叠 computation 是什么意思？
- overlap 为什么难？

这一模块当前保留为辅助地图，不作为主实验方向。

---

## D. Training Parallelism 与训练系统

### D1. 训练状态

- 参数 parameter 是什么？
- gradient 是什么？
- optimizer state 是什么？
- activation 是什么？
- 为什么 activation 会占很多显存？
- 为什么 optimizer state 会占很多显存？
- 训练和推理在状态管理上最关键的差别是什么？

### D2. 并行策略

- Data Parallel 是什么？
- Tensor Parallel 是什么？
- Pipeline Parallel 是什么？
- Expert Parallel 是什么？
- ZeRO 是什么？
- FSDP 是什么？
- DP、TP、PP、EP 各自在切什么？
- 这些并行策略分别在交换什么资源？
- 为什么没有一种并行策略是“总是最优”？
- sequence parallel 是什么？
- context parallel 是什么？
- 为什么训练常常要组合多种并行策略？

### D3. 训练系统

- Megatron-LM 的核心贡献是什么？
- DeepSpeed 的核心价值是什么？
- FSDP 的核心思路是什么？
- ZeRO 在本质上优化了什么？
- checkpointing 在训练里为什么重要？
- activation checkpointing 解决什么问题？
- 为什么训练系统常常先被 memory 和 communication 联合限制？

这一整块当前只保留地图位置。
如果不服务于 inference 主线，就不再继续展开。

---

## E. Inference Fundamentals

### E1. Prefill / Decode

- 什么是 prefill？
- 什么是 decode？
- 为什么 prefill 和 decode 本质不同？
- 为什么 prefill 更像大矩阵计算？
- 为什么 decode 更像 memory traffic 问题？
- 为什么 decode 往往 batch 更难做大？
- TTFT 是什么？
- ITL 是什么？
- 为什么 TTFT 和 ITL 常常互相拉扯？

### E2. Attention 基础

- 什么是 self-attention？
- 什么是 causal attention？
- attention mask 是什么？
- causal mask 在推理里起什么作用？
- 什么是 softmax attention？
- 为什么 attention 很容易变成 memory-intensive？
- 什么是 FlashAttention？
- FlashAttention 在解决什么层次的问题？
- online softmax 为什么重要？
- FlashAttention 和普通 fused attention 有什么差别？

### E3. KV Cache

- KV cache 是什么？
- 为什么推理时需要 KV cache？
- 为什么 decode 每一步都要读历史 KV？
- KV cache 的大小怎么估算？
- KV cache 为什么会成为 serving 的中心状态？
- 长 context 为什么会把问题推向 memory system？
- paged KV 是什么？
- block table 是什么？
- prefix cache 和 KV cache 是一回事吗？
- KV cache quantization 在试图优化什么？

### E4. 结构变化

- 什么是 MLA？
- MLA 和普通 MHA 的系统意义有什么不同？
- 为什么 attention 结构变化会改变 serving 系统设计？

---

## F. Serving Runtime 与 Scheduler

### F1. 请求与批处理

- 一个 request 是怎么进入 serving engine 的？
- 静态 batching 和 continuous batching 的差别是什么？
- continuous batching 为什么重要？
- 为什么 serving 里的 batch 不像训练里的 batch 那么简单？
- 为什么请求长度差异会影响调度？
- 为什么长 prompt 会拖慢系统？
- 为什么 decode 请求会互相影响？

### F2. Prefill / Decode 调度

- 为什么 prefill 和 decode 会竞争 GPU 资源？
- 什么是 prefill-decode mixing？
- 什么是 chunked prefill？
- chunked prefill 为什么会出现？
- chunked prefill 在平衡什么？
- 为什么长 prefill 会伤害 decode latency？
- 什么情况下要优先保 decode？
- 什么情况下可以牺牲一点 TTFT 换吞吐？

### F3. 调度目标

- throughput 是什么？
- latency 是什么？
- P50 / P95 / P99 分别是什么？
- goodput 是什么？
- 为什么 goodput 比 throughput 更接近真实服务目标？
- SLO 是什么？
- QoS-aware scheduling 是什么？
- fairness 在 serving 里是什么意思？
- 为什么调度目标不可能只有一个？

这是当前 repo 的核心主线之一。

---

## G. Serving Primitive / Backend Library / Engine

### G1. 分层边界

- 什么是 backend primitive library？
- 什么是 serving engine？
- 什么是 kernel library？
- backend、engine、kernel 的边界是什么？
- 为什么它们不能混成一层？
- 为什么 engine 不自己重写所有 kernel？

### G2. 代表性系统

- vLLM 的核心贡献是什么？
- PagedAttention 是 vLLM 的 kernel 优化还是 memory abstraction？
- SGLang 的核心设计点是什么？
- RadixAttention 是什么？
- FlashInfer 在系统栈里属于哪一层？
- TensorRT-LLM 更偏 engine、backend 还是 deployment stack？
- llama.cpp 更偏哪条路线？
- 为什么 llama.cpp 和 vLLM 的设计哲学差别很大？

### G3. Backend Path

- 为什么 prefill 和 decode 常常走不同 backend path？
- ragged batch 是什么？
- sampling backend 在做什么？
- 为什么 structured output 也会影响 runtime 设计？

这是当前 repo 的核心主线之一。

---

## H. Memory System / Cache / State Externalization

### H1. 状态视角

- 什么是系统状态？
- parameter、activation、optimizer state、KV cache 各自是什么性质的状态？
- 哪些状态更适合缓存？
- 哪些状态更适合 externalization？
- 为什么“状态边界”是系统设计问题？

### H2. Cache / Reuse

- prefix caching 是什么？
- prefix reuse 在改善什么指标？
- prefix reuse 和 decode acceleration 有什么不同？
- expert cache 是什么？
- parameter offload 在优化什么？
- cache hit rate 为什么关键？
- cache 不是总能带来收益，为什么？

### H3. Externalization / Disaggregation

- KV externalization 是什么？
- LMCache 在系统里扮演什么角色？
- state externalization 的好处是什么？
- state externalization 的代价是什么？
- prefill-decode disaggregation 是什么？
- disaggregation 为什么不是“简单拆成两台机器”？
- 当状态从进程私有变成系统资源后，会引入哪些新的调度问题？

---

## I. Model Artifact / Weight Format / Loading Path

### I1. 模型工件

- checkpoint 是什么？
- shard 是什么？
- checkpoint sharding 为什么重要？
- safetensors 是什么？
- GGUF 是什么？
- TensorRT engine 是什么？
- 为什么权重格式会影响部署方式？

### I2. 加载路径

- model loading path 是什么？
- 冷启动 cold start 为什么常常很慢？
- 权重加载和真正开始推理之间有哪些步骤？
- weight layout 为什么会影响 runtime/backend 选择？
- quantization 为什么不只是省内存？
- engine build 和 runtime load 的区别是什么？

---

## J. Cluster Scheduling / Orchestration / Reliability

### J1. 调度层次

- job scheduler 是什么？
- serving scheduler 是什么？
- cluster scheduler 是什么？
- 它们三者的边界是什么？
- 为什么不能把 engine scheduler 和 cluster scheduler 混成一层？

### J2. 资源编排

- Kubernetes 在 AI infra 里扮演什么角色？
- Ray 在 AI infra 里扮演什么角色？
- Slurm 更适合什么场景？
- pod、node、service 这些基础概念分别是什么？
- placement 为什么重要？
- GPU resource allocation 为什么难？
- autoscaling 是什么？
- 为什么 LLM serving 的 autoscaling 比普通 web service 更难？

### J3. 多租户与可靠性

- multi-tenant 是什么？
- quota 是什么？
- priority 是什么？
- preemption 是什么？
- fault tolerance 是什么？
- retry 为什么不是总是安全？
- checkpoint 在训练和 serving 中分别如何帮助可靠性？
- rolling update 为什么在 AI 服务中更复杂？

这一模块当前保留为地图层，不展开 HTTP service 细节。

---

## K. Observability / Profiling / Evaluation

### K1. 指标

- GPU utilization 是什么？
- memory bandwidth utilization 是什么？
- link bandwidth utilization 是什么？
- throughput、latency、TTFT、ITL、P99、goodput 之间是什么关系？
- 为什么只看 throughput 很危险？
- 为什么 P99 在在线系统中重要？
- 什么是 tail latency？

### K2. Profiling

- profiler 是什么？
- Nsight Compute 更适合看什么？
- Nsight Systems 更适合看什么？
- PyTorch profiler 更适合看什么？
- trace 是什么？
- timeline 能回答什么问题？
- flame graph 是什么？
- kernel-level profiling 和 system-level tracing 的区别是什么？

### K3. Evaluation

- benchmark 在 AI systems 里为什么容易误导？
- synthetic workload 和 real workload 的差别是什么？
- serving evaluation 为什么不能只测平均 latency？
- BurstGPT 这种真实工作负载数据集为什么重要？
- 为什么 workload shape 会改变系统结论？

---

## L. Sparse / MoE / Speculation / Edge

### L1. Speculative Decoding

- speculative decoding 是什么？
- draft model 和 target model 分别做什么？
- verify 是什么？
- acceptance rate 是什么？
- speculative decoding 真正省的是哪一部分？
- verifier cost 为什么会侵蚀收益？
- speculative decoding 和 prefix caching 有什么本质不同？
- speculative decoding 的 attention mask 有什么特殊性？
- 为什么 speculative decoding 的收益高度依赖 workload？

### L2. MoE

- MoE 是什么？
- router 是什么？
- expert 是什么？
- top-k routing 是什么？
- 为什么 MoE 不等于“天然更快”？
- MoE 的瓶颈更像 compute、memory、communication，还是 orchestration？
- expert parallel 在系统上意味着什么？
- expert cache 为什么会出现？
- dense serving 和 MoE serving 的 runtime 问题有什么相同和不同？

### L3. Edge / Offload / Quantization

- edge inference 和 datacenter inference 的最大差别是什么？
- offload 是什么？
- layer offload 是什么？
- CPU/GPU hybrid execution 在试图解决什么问题？
- quantization 在系统上到底省了什么？
- quantization 会带来哪些新的系统代价？
- 为什么 edge 场景更容易被 memory bandwidth 卡住？

---

## 跨模块通用问题

- 这个知识点属于哪一层？
- 它主要优化的是 compute、memory、communication、runtime、scheduler 还是 deployment？
- 它依赖什么前提？
- 它不解决什么问题？
- 它和相邻模块的边界是什么？
- 它在真实系统里对应哪些项目或组件？
- 它更像基础机制、系统抽象、工程实现，还是部署策略？
- 如果换成单卡、多卡、多机，结论会变吗？
- 如果换成训练、离线推理、在线 serving，结论会变吗？

---

## 建议使用方法

你不需要按顺序一条条学完。  
更高效的方式是：

1. 先扫一遍，标出完全不会的问题
2. 先补“不会但高频出现”的问题
3. 每学一个系统，再回到 checklist 找它对应的知识点
4. 每补完一个模块，就把相关问题勾掉

当前阶段还要再加一条：

5. 如果某个问题明显属于训练系统主线、HTTP server 主线、或者手写 CUDA 极限优化主线，可以先不展开；`Triton` kernel optimization 仍保留在 inference runtime 的学习主线里

这份 checklist 的目标不是证明你已经很懂，  
而是让你快速知道：`自己到底还缺什么最基础的东西`。
