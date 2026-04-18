# Basic Kernel Categories

这份笔记用来建立一个最基础的 kernel 分类直觉。

目标不是把所有 kernel 都严格分类，而是先回答：

- `elementwise` 是什么
- `reduction` 是什么
- `GEMM` 是什么
- 它们在并行方式、访存方式、优化重点上有什么不同

很多 CUDA / AI infra 讨论里，大家默认你知道这些词。  
如果这些基本类型没有概念，后面看 `FlashAttention`、`layernorm`、`softmax`、`matmul`、`MoE routing` 都会很乱。

---

## 1. 为什么要先分 kernel 类型

因为不同 kernel 类型，通常意味着不同的核心问题：

- 有的主要是“每个元素独立处理”
- 有的主要是“很多线程一起汇总一个结果”
- 有的主要是“大规模复用数据做矩阵乘”
- 有的主要是“不规则访存和索引”

所以 kernel 类型通常决定：

- 线程怎么映射数据
- 访存是否规则
- 是否需要同步
- 更容易 compute-bound 还是 memory-bound

---

## 2. Elementwise Kernel

### 它是什么

`elementwise` kernel 的意思是：

- 输出中的每个元素，通常只依赖输入中对应位置的少量元素
- 每个位置基本可以独立计算

最典型的例子：

- `y[i] = x[i] * 2`
- `z[i] = x[i] + y[i]`
- `relu(x[i])`
- `sigmoid(x[i])`

### 它的特点

- 并行方式最直接
- 每个 thread 很适合处理一个或几个元素
- 线程之间通常不需要复杂协作
- 通常不需要太多同步

### 它的系统特点

elementwise kernel 往往：

- 算法逻辑简单
- arithmetic intensity 不高
- 很容易被 memory bandwidth 限制

也就是说，很多 elementwise kernel 更像是在“搬数据顺便做一点计算”。

### 优化重点

- 让访存连续
- 减少 launch overhead
- 尽量 fusion，避免很多小 kernel

### 一句话直觉

elementwise kernel 是最基础的一类：

- 逻辑简单
- 并行容易
- 但很容易 memory-bound

---

## 3. Reduction Kernel

### 它是什么

`reduction` kernel 的意思是：

- 把很多输入元素汇总成更少的输出

最典型的例子：

- 求和 `sum`
- 最大值 `max`
- 均值 `mean`
- norm

例如：

- 一个长度为 `N` 的向量，最后汇总成一个值
- 或者一个矩阵的每一行/每一列汇总成一个值

### 它的特点

和 elementwise 不一样，reduction 不是“大家各算各的”。

它通常需要：

- 多个线程共同参与一个结果
- 分阶段合并部分结果
- 经常使用 shared memory 或 warp-level primitive

### 它为什么更难

因为它天然涉及：

- 同步
- 局部汇总
- 树形归约
- 线程协作

### 常见例子

- softmax 里的 `max` 和 `sum`
- layernorm 里的 mean/variance
- loss 汇总

### 优化重点

- 减少同步开销
- 让 reduction 结构高效
- 用 warp-level / block-level reduction
- 尽量减少不必要的 global memory 往返

### 一句话直觉

reduction kernel 的核心不是“每个线程算一点”，而是：

- “很多线程一起把很多数压成少数结果”

---

## 4. GEMM Kernel

### 它是什么

`GEMM` 是 `General Matrix Multiply`，一般就是矩阵乘法。

最常见形式：

- `C = A x B`

它是深度学习里最重要的一类 kernel 之一。

### 为什么它重要

因为很多大模型计算最后都会落到矩阵乘：

- linear layer
- projection
- MLP
- attention 中的大量张量乘法

### 它的特点

GEMM 和 elementwise / reduction 都不一样。

它的核心是：

- 大量乘加计算
- 大量数据复用
- 非常适合 tile 化

### 为什么 GEMM 往往更“吃算力”

因为同一块输入数据会被重复使用很多次。  
这意味着：

- 每次从内存搬来的数据，可以参与很多次计算

所以 GEMM 通常有更高的 arithmetic intensity，  
更容易逼近 compute-bound。

### 优化重点

- tile/blocking
- shared memory 复用
- register tiling
- Tensor Core 利用
- 合理的数据布局

### 一句话直觉

GEMM 是最典型的“高复用、高吞吐、值得精细优化”的 kernel 类型。

---

## 5. Stencil / Neighborhood Kernel

### 它是什么

这类 kernel 的特点是：

- 一个输出元素依赖附近一小片邻域的数据

典型例子：

- convolution
- 图像滤波
- stencil computation

### 它的系统特点

它既不像纯 elementwise 那么独立，  
也不像 GEMM 那样有那么规则的大规模复用。

它常常需要：

- 邻域访问
- 边界处理
- 局部数据重用

### 优化重点

- shared memory tile
- halo 区域处理
- 边界分支控制

### 一句话直觉

这类 kernel 的重点是“局部邻域复用”。

---

## 6. Gather / Scatter / Indexed Kernel

### 它是什么

这类 kernel 的特点是：

- 数据访问由索引决定
- 不是简单的连续位置一一对应

例如：

- embedding lookup
- gather
- scatter
- 某些 MoE routing / token dispatch

### 它为什么麻烦

因为它通常意味着：

- 访存不规则
- 很难 coalesced
- cache 利用可能差
- 容易出现 load imbalance

### 优化重点

- 改善数据布局
- 改善索引组织
- 尽量批量化和规整化访问

### 一句话直觉

indexed kernel 的核心难点通常不是算，而是不规则访存。

---

## 7. Prefix / Scan Kernel

### 它是什么

`scan` 或 `prefix sum` 的意思是：

- 输出位置 `i` 依赖前面一段元素的累计结果

例如：

- prefix sum
- 某些排序、分桶、压缩中的中间步骤

### 它的特点

它和 reduction 有点像，但不是把很多元素压成一个值，  
而是要保留每个位置的“前缀累计结果”。

### 为什么重要

很多看起来不显眼的系统操作底层都会用到 scan。

### 一句话直觉

scan 是“既要汇总，又不能丢掉位置结构”的 kernel。

---

## 8. 这些类型在 AI 里怎么出现

### elementwise

- activation
- bias add
- dropout
- residual add

### reduction

- softmax
- layernorm
- loss aggregation

### GEMM

- linear
- projection
- MLP
- attention 里的矩阵乘

### indexed / gather-scatter

- embedding
- MoE routing
- token dispatch

### stencil / neighborhood

- convolution
- 某些局部注意力或局部窗口操作

---

## 9. 一个更实用的分类视角

除了按数学形式分，你也可以按系统瓶颈分：

### 更容易 memory-bound 的

- elementwise
- gather/scatter
- 一些简单 reduction

### 更容易 compute-bound 的

- GEMM
- 高复用的 tensor contraction

### 更容易被同步和协作影响的

- reduction
- scan
- 一些 block-level cooperative kernel

### 更容易被访存不规则影响的

- gather/scatter
- sparse kernel
- MoE routing 相关 kernel

---

## 10. 为什么这个分类对后面有用

因为你后面看任何系统优化时，都可以先问：

1. 这个 kernel 属于哪一类？
2. 它更像是在算，还是在搬？
3. 它的主要矛盾是复用、同步、还是不规则访存？

举几个例子：

- `ReLU`：更像 elementwise
- `layernorm`：带 reduction
- `matmul`：典型 GEMM
- `embedding lookup`：典型 indexed kernel
- `MoE token dispatch`：强 indexed / communication 属性
- `FlashAttention`：本质上是在重构 attention 里的 memory access 和 reduction 路径

---

## 11. 先记住的最小结论

如果现在只记最重要的，可以先记这几句：

1. `elementwise`：每个元素基本独立，容易 memory-bound
2. `reduction`：很多线程一起汇总结果，重点是协作和同步
3. `GEMM`：高复用、高吞吐、最值得精细优化
4. `indexed kernel`：难点通常是不规则访存
5. 不同 kernel 类型，决定了不同的优化重点

---

## 12. 下一步自然会补的内容

- `memory_access_patterns.md`
  - 为什么 elementwise 容易 memory-bound
  - 为什么 indexed kernel 难优化

- `reduction_patterns.md`
  - warp reduction
  - block reduction
  - softmax / layernorm 的 reduction 结构

- `gemm_tiling_basics.md`
  - tile
  - blocking
  - data reuse
  - Tensor Core
