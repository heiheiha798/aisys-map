# Basic Kernel Categories

这份笔记建立最基础的 kernel 分类直觉。目标不是给所有 kernel 做严格 taxonomy，而是先判断：

- 数据怎么并行切分
- 访存是否规则
- 是否需要线程协作和同步
- 更可能是 compute-bound、memory-bound，还是被不规则访存拖住

这些问题会直接影响后面看 `FlashAttention`、`layernorm`、`softmax`、`matmul`、`MoE routing` 时的判断。

---

## 1. 为什么先分类型

不同 kernel 类型通常对应不同的核心矛盾：

| 类型 | 核心问题 |
|---|---|
| elementwise | 每个元素独立处理，但算得少、搬得多 |
| reduction | 多个线程协作，把很多元素汇总成更少结果 |
| GEMM | 大量乘加和数据复用，尽量喂满计算单元 |
| stencil / neighborhood | 一个输出依赖邻域数据，需要局部复用和边界处理 |
| gather / scatter / indexed | 访问位置由索引决定，难点是不规则访存 |
| prefix / scan | 既要汇总前缀，又要保留每个位置的结构 |

所以分类不是为了背名字，而是为了先猜瓶颈和优化方向。

---

## 2. 常见类型

| 类型 | 典型形式 | 系统特点 | 常见优化方向 |
|---|---|---|---|
| `elementwise` | `y[i] = f(x[i])`，bias add，activation，residual add | 线程映射直接，协作少，arithmetic intensity 低，容易 memory-bound | 连续访存、向量化、fusion、减少 launch overhead |
| `reduction` | `sum/max/mean/norm`，softmax 的 `max`/`sum`，layernorm 的 mean/variance | 多线程共同产生一个或少量结果，需要分阶段合并 | warp/block reduction、减少同步、减少 global memory 往返 |
| `GEMM` | `C = A x B`，linear、projection、MLP、attention matmul | 高复用、高吞吐，通常最值得精细优化 | tiling、shared memory 复用、register tiling、Tensor Core、数据布局 |
| `stencil / neighborhood` | convolution、图像滤波、局部窗口操作 | 输出依赖邻域数据，有边界处理和局部复用 | shared memory tile、halo 处理、减少边界分支 |
| `gather / scatter / indexed` | embedding lookup、token dispatch、MoE routing | 索引驱动访问，coalescing 和 cache 利用都可能很差 | 重排数据布局、批量化索引、让访问尽量规整 |
| `prefix / scan` | prefix sum、分桶、压缩中间步骤 | 类似 reduction，但每个位置都要保留前缀结果 | 分层 scan、block 内协作、减少同步和中间写回 |

---

## 3. 按瓶颈重新看

比数学形式更实用的判断方式，是问它主要卡在哪里。

| 更容易卡住的地方 | 常见类型 |
|---|---|
| memory bandwidth | elementwise、gather/scatter、简单 reduction |
| compute throughput | GEMM、高复用 tensor contraction |
| synchronization / cooperation | reduction、scan、block-level cooperative kernel |
| irregular memory access | gather/scatter、sparse kernel、MoE routing |
| launch/runtime overhead | 大量小 elementwise kernel、细碎的 unfused op |

这个表只是初始判断。真实瓶颈还要结合数据规模、layout、cache 命中、fusion、kernel launch 和端到端调度看。

---

## 4. 在 AI 里怎么出现

| AI 操作 | 更像哪类 kernel |
|---|---|
| ReLU / sigmoid / residual add / bias add | elementwise |
| softmax | reduction + elementwise |
| layernorm / rmsnorm | reduction + elementwise |
| linear / projection / MLP | GEMM |
| attention score / value aggregation | GEMM + reduction + memory layout 问题 |
| embedding lookup | gather / indexed |
| MoE token dispatch | indexed + communication / scheduler 问题 |
| convolution / local attention | stencil / neighborhood |

`FlashAttention` 是一个典型例子：它不是发明了新的数学操作，而是在重构 attention 里的 memory access、tiling 和 reduction 路径。

---

## 5. 读系统优化时怎么用

看到一个 kernel 或 operator，先问三件事：

1. 它更像哪一类？
2. 它主要是在算，还是在搬？
3. 它的核心矛盾是复用、同步、launch overhead，还是不规则访存？

如果这三个问题答不清，直接讨论“怎么优化”通常会跑偏。

---

## 6. 最小结论

1. `elementwise`：并行容易，但常常 memory-bound。
2. `reduction`：重点是协作、同步和分阶段汇总。
3. `GEMM`：高复用、高吞吐，最值得做深度优化。
4. `indexed kernel`：难点通常不是算，而是不规则访存。
5. kernel 类型决定第一轮瓶颈假设，但端到端性能还要看 runtime 和调度。

---

## 7. 后续主题

- `memory_access_patterns.md`：coalescing、stride、cache locality、bank conflict
- `reduction_patterns.md`：warp reduction、block reduction、softmax / layernorm reduction
- `gemm_tiling_basics.md`：tile、blocking、data reuse、Tensor Core
