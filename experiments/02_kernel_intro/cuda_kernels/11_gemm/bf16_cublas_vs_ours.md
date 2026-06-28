# BF16 GEMM: cuBLAS vs Our Tensor Core Kernel

这份说明默认你已经看过：

- `elementwise`
- `softmax`
- `online_softmax`
- [bf16_cuda_core_vs_tensor_core.md](bf16_cuda_core_vs_tensor_core.md)

所以这里不再重复解释最基础的：

- thread / warp / block
- shared memory
- `WMMA` / `mma_sync` 的最基本定义

涉及稳定术语时，默认按下面两份 notes 理解：

- GPU 存储层次见 [../../../../notes/gpu_components.md](../../../../notes/gpu_components.md)
- `FMA / MMA / Tensor Core / WMMA` 见 [../../../../notes/cuda_tensor_core_wmma.md](../../../../notes/cuda_tensor_core_wmma.md)

这里要回答的不是“怎么写一个 GEMM”，而是两个更现实的问题：

1. 为什么 `cuBLAS` 这么快
2. 我们现在这个 `bf16_gemm_cublas.cu` 到底在做什么

---

## 1. 先看结论

当前这台 `RTX 4090 / sm_89` 机器上的一组结果是：

```text
bf16_gemm_tensor_core   avg_ms≈0.0572   tflops≈37.6
bf16_gemm_cublas        avg_ms≈0.0181   tflops≈118.9
```

也就是说：

- `cuBLAS` 大约快 `3.2x`

这个差距说明的不是：

- 我们没有用上 Tensor Core

而是：

- 我们虽然已经在走 Tensor Core 路线
- 但还只是一个教学型 WMMA kernel
- 离成熟 GEMM 库的分层 tiled + multistage pipeline 还差很远

---

## 2. 为什么 `cuBLAS` 这么快

先把最容易误解的一点说清楚：

- `cuBLAS` 快，不是因为它“调用了一个神奇 API”
- 而是因为这个 API 后面接的是一套已经针对具体 GPU 架构调过很多轮的 kernel 选择和实现

在我们这次 `ncu` 里，`cuBLAS` 抓到的主 kernel 名字是：

```text
ampere_s16816gemm_bf16_128x64_ldg8_stages_32x6_nn
```

这个名字本身就已经透露出很多信息：

- `bf16`
  - 输入是 `bf16`
- `128x64`
  - block / tile 规模明显比我们当前这版 `16x16` warp 小块更大
- `ldg8`
  - global load 组织更激进
- `stages_32x6`
  - 有更深的多阶段 pipeline

### 2.1 它不是只会算得快，而是会“喂得快”

我们当前自己的 WMMA kernel，最大问题不是 `mma_sync` 慢，而是 feeding path 还太朴素：

- `A` 仍然直接从 global memory 喂给 `load_matrix_sync`
- `B` 虽然放进了 shared memory，但整体还是单阶段循环
- 每一轮都是：
  - 搬一点数据
  - 同步
  - 做一点 MMA
  - 再同步

这意味着：

- Tensor Core 本身已经切对了
- 但 Tensor Core 前面的“数据供给系统”还很粗糙

`cuBLAS` 真正强的地方，在于它把下面这些一起做了：

- 更大的 block / warp / mma 分层 tile
- 更高的 block 内复用
- 更深的 shared-memory / register pipeline
- 更好的 load/store 组织
- 更系统的寄存器预算

所以 `cuBLAS` 的优势不是某一个点，而是整体协同。

### 2.2 为什么 occupancy 很低，反而还更快

这次 `ncu` 的一个很重要信号是：

- 我们的 `bf16_gemm_tensor_core`
  - `Achieved Occupancy ≈ 64.43%`
- `cuBLAS`
  - `Achieved Occupancy ≈ 8.28%`

但 `cuBLAS` 仍然快很多。

这说明：

- 高性能 GEMM 不是简单追求 occupancy 越高越好

成熟 GEMM 往往会主动接受：

- 更大的 tile
- 更多寄存器
- 更多 shared memory
- 更低的 occupancy

换来的是：

- 更高的单个 block 有效工作量
- 更强的 Tensor Core feeding
- 更少的近端 cache / load stall

这和我们当前的教学型实现，是两种完全不同的设计目标。

### 2.3 我们和 `cuBLAS` 的差距到底在哪

当前更准确的说法是：

- 我们已经会“用 WMMA 写出能跑的 Tensor Core GEMM”
- 但还没有写出“库级高性能 GEMM”

差距主要在：

1. tile 设计
   - 我们现在的 warp tile 仍然非常小
   - `cuBLAS` 的 block / warp / mma 分层更成熟
2. pipeline 深度
   - 我们还是单阶段
   - `cuBLAS` 明显已经是多阶段流水
3. global -> shared -> register 的数据搬运
   - 我们只做了最朴素的 `B` tile staging
   - `cuBLAS` 的搬运和计算重叠得更好
4. 资源协同
   - 寄存器、shared memory、Tensor Core feeding 是一起设计的
   - 不是只改一个 block 参数就能抄到

所以不要把 `cuBLAS` 理解成“只是一个更大的 WMMA kernel”。

它本质上已经是一套成熟的 GEMM 系统。

---

## 3. 我们这个 `bf16_gemm_cublas.cu` 到底做了什么

这个文件在目录里的角色很明确：

- 它不是自己实现一个新的 GEMM kernel
- 它只是一个最小包装层
- 负责把输入准备好，然后调用 `cuBLAS`

所以它更像一个：

- benchmark harness
- correctness harness
- API 示例

而不是新的 kernel 代码。

文件见：

- [bf16_gemm_cublas.cu](bf16_gemm_cublas.cu)

### 3.1 文件结构

可以把它分成 5 段来看：

1. 数据准备
   - `fill_input`
   - `convert_from_fp32`
   - `cpu_gemm`
2. 误差比较
   - `compare_outputs`
3. 真正的 `cuBLAS` 调用
   - `launch_bf16_gemm`
4. 一次性运行 / benchmark 封装
   - `run_once`
   - `benchmark`
5. 主入口
   - `run_experiment`
   - `main`

这里真正和 `cuBLAS` 直接相关的核心，其实只有：

- `launch_bf16_gemm`

---

## 4. `launch_bf16_gemm` 在做什么

核心代码是：

```cpp
const float alpha = 1.0f;
const float beta = 0.0f;

CUBLAS_CHECK(cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k,
    &alpha,
    d_b, CUDA_R_16BF, n,
    d_a, CUDA_R_16BF, k,
    &beta,
    d_c, CUDA_R_32F, n,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
```

这看起来最绕的地方有两个：

1. 为什么顺序是 `d_b` 在前、`d_a` 在后
2. 为什么尺寸是 `n, m, k`，而不是直觉上的 `m, n, k`

答案都和同一件事有关：

- `cuBLAS` 默认按 **column-major** 理解矩阵
- 我们手里的数据是 **row-major**

### 4.1 row-major 和 column-major 的关系

假设你手里有一个 row-major 的：

- `A[m, k]`
- `B[k, n]`

在 `cuBLAS` 眼里，如果你不转置内存、直接把这段地址给它，它会把它们看成：

- `A^T[k, m]` 的 column-major 存储
- `B^T[n, k]` 的 column-major 存储

所以如果我们想得到：

```text
C = A x B
```

就可以让 `cuBLAS` 去算：

```text
C^T = B^T x A^T
```

这就是为什么这里调用写成：

- 先传 `d_b`
- 再传 `d_a`
- 尺寸写成 `n, m, k`

它不是算错了，而是在借 column-major 视角，等价实现 row-major GEMM。

### 4.2 为什么这里两个 transpose 参数都是 `CUBLAS_OP_N`

因为我们并没有真的让 `cuBLAS` 再做一次逻辑转置操作。

我们只是：

- 利用 row-major 数据在内存中的布局
- 让它被 `cuBLAS` 按 column-major 的转置矩阵去解释

所以：

- 物理地址不变
- API 层面也不需要再额外指定 `OP_T`

这就是这段代码最值得记住的小技巧。

---

## 5. 这个文件里用到的 `cuBLAS` API 各是什么

这里只讲这份文件里真的用到的几个。

### `cublasCreate`

```cpp
cublasCreate(&handle)
```

作用：

- 创建一个 `cuBLAS handle`

你可以把它理解成：

- 后面所有 `cuBLAS` 调用都挂在这个上下文对象上

没有它，就没法发 GEMM。

### `cublasDestroy`

```cpp
cublasDestroy(handle)
```

作用：

- 销毁 `handle`
- 释放 `cuBLAS` 相关资源

### `cublasSetMathMode`

```cpp
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)
```

作用：

- 告诉 `cuBLAS` 允许使用 Tensor Core 路径

对这份实验来说，这句的意义很直接：

- 我们不是想测“普通库调用”
- 而是想测“成熟 Tensor Core GEMM 库实现”

### `cublasGemmEx`

这是这里最核心的 API。

和传统 `cublasSgemm` 相比，`GemmEx` 更通用，因为它允许分别指定：

- `A / B / C` 的数据类型
- compute type
- 算法选择

在这份代码里，它对应的是：

- 输入 `A`：`CUDA_R_16BF`
- 输入 `B`：`CUDA_R_16BF`
- 输出 `C`：`CUDA_R_32F`
- accumulate / compute：`CUBLAS_COMPUTE_32F`

也就是：

- `bf16` 输入
- `float` accumulate / output

这和我们自己 Tensor Core kernel 的目标是对齐的。

### `CUBLAS_GEMM_DEFAULT_TENSOR_OP`

这是 `cublasGemmEx` 的最后一个算法参数。

这里的意思不是：

- “我自己精确指定了哪一个具体 kernel”

而是：

- 让 `cuBLAS` 在默认策略下，优先选择适合 Tensor Core 的实现

真正最后落到哪个具体 kernel，由库自己根据架构和问题规模决定。

这也是为什么我们最后在 `ncu` 里看到的是：

```text
ampere_s16816gemm_bf16_128x64_ldg8_stages_32x6_nn
```

而不是一个你在代码里手写出来的 kernel 名。

---

## 6. `run_once` 和 `benchmark` 的作用

这两个函数本身没有什么复杂算法，主要是实验框架层面的职责。

### `run_once`

用途：

- correctness 检查

做的事情是：

1. `cudaMalloc`
2. 把 host 上的 `bf16` 输入拷到 device
3. 调一次 `launch_bf16_gemm`
4. 把结果拷回 host
5. 和 CPU 参考做比较

### `benchmark`

用途：

- 性能测试

做的事情是：

1. warmup
2. 用 CUDA event 计时
3. 重复调用 `launch_bf16_gemm`
4. 算平均时间和 TFLOPS

也就是说：

- 这两个函数不是 `cuBLAS` 的重点
- 只是把 `cuBLAS` 包装成和我们自己 kernel 一样的实验接口

这样 benchmark 和对照才公平。

---

## 7. 应该怎么理解这份文件的价值

这份 `bf16_gemm_cublas.cu` 最重要的价值，不是：

- 教你怎么“写 `cuBLAS` 库源码”

它真正的价值是：

1. 给当前目录加了一个可信的第三方基线
2. 让我们知道自己的教学型 WMMA kernel 还差多远
3. 提供了一个最小、可运行、可 profile 的 `cublasGemmEx` 示例
4. 把 row-major 数据如何映射到 column-major `cuBLAS` 调用这件事讲清楚了

所以如果你后面再看我们自己的 `bf16_gemm_tensor_core`，心里应该带着一个更明确的参照：

- 现在这版已经足够用来学习 Tensor Core 路径
- 但它远远不是库级实现
- 真正要逼近 `cuBLAS`，需要进入下一层：
  - 更大的分层 tile
  - 多阶段 pipeline
  - 更成熟的 shared-memory / register / Tensor Core feeding 协同

整个对照也只围绕这一条 `bf16` 主线展开：

- `bf16_gemm_cuda_core`
- `bf16_gemm_tensor_core`
- `bf16_gemm_cublas`

这里不再展开其他数据类型，避免把重点从 `bf16` 路线切散。

---

## 8. 最后记住三句话

1. `cuBLAS` 快，不是因为 API 神奇，而是因为 API 后面接的是成熟的高性能 GEMM 系统。
2. `bf16_gemm_cublas.cu` 本质上只是一个很薄的包装层，真正的重点只有 `cublasGemmEx` 那次调用。
3. row-major 数据调用默认 column-major 的 `cuBLAS` 时，最关键的技巧是：
   - 把 `C = A x B`
   - 改写成 `C^T = B^T x A^T`
   - 然后在 API 里交换参数顺序和维度。
