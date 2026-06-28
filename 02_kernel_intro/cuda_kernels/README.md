# CUDA Kernels

这个目录放最基础的原生 CUDA C++ kernel 实验。

这里的定位需要先说清楚：

- 这是一个参考级、学习导向的目录
- 重点是帮助读者建立对常见 CUDA kernel 的基本定义和直觉
- 不是为了沉淀一组高性能、生产级的 CUDA 实现

所以这里所有 `.cu` 文件都应该理解成：

- **重逻辑、弱性能**

它们的主要作用是：

- 让你知道一个 kernel 在做什么
- 理解它最基本的并行思路、访存模式和常见术语
- 在需要的时候有一个最小可运行的参照物

它们不承担的目标是：

- 追求性能最优
- 对标工业级 CUDA kernel
- 让读者以后直接照着手写生产实现

如果你的目标只是：

- 看过之后知道这些 kernel 各自是什么
- 大概知道哪些问题属于 elementwise、reduction、attention、kv cache、flash attention

那这个目录就已经达成目的了。

根目录只放通用信息。  
具体实验按类型拆到子目录里，例如：

- `01_elementwise/`
- `02_scatter/`
- `03_embedding/`
- `04_softmax/`
- `05_layernorm/`
- `06_rope/`
- `07_online_softmax/`
- `08_fused_rmsnorm/`
- `09_attention/`
- `10_kv_cache/`
- `11_gemm/`
- `12_flash_attention/`

## 编译约定

每个子目录里可以有自己的：

- 源码
- Makefile
- README
- profiling 结果

如果 `nvcc` 不在默认 `PATH` 中，可以在各子目录显式指定：

```bash
make NVCC=/path/to/nvcc
```

当前本机可用的 `nvcc` 路径是：

```bash
/usr/local/cuda-12.4/bin/nvcc
```

例如：

```bash
make NVCC=/usr/local/cuda-12.4/bin/nvcc
```

## Profiling

本机的 `nsys` 放在：

```bash
/data/home/tianjianyang/download/nsys
```

如果后面要做 kernel profiling，可以从这里调用对应的 `nsys` 可执行文件。

本机可用的 `ncu` 路径是：

```bash
/usr/local/cuda-12.4/bin/ncu
```

## 建议学习顺序

如果是第一次系统看这些 kernel，建议顺序不要从 `09_attention/` 开始。  
更合理的路径是先从最简单、最局部的模式往上走。

建议顺序如下：

1. `01_elementwise/`
   - 先建立最基础的线程映射、全局内存读写和 “一个线程处理一个元素” 的直觉
2. `02_scatter/`
   - 看 scatter / atomic update 这种不规则写
3. `03_embedding/`
   - 看 gather 这种不规则读
4. `04_softmax/`
   - 开始接触 reduction、shared memory、block 内协作和同步
5. `05_layernorm/`
   - 巩固 row-wise reduction 和 memory-bound kernel 的直觉
6. `06_rope/`
   - 理解 LLM 里的高频小算子
7. `07_online_softmax/`
   - 理解 `(m, l)` 状态合并，以及为什么它能通向 FlashAttention
8. `08_fused_rmsnorm/`
   - 理解 LLM 小算子 fusion
9. `09_attention/`
   - 把 `gemm + softmax + reduction` 真正串成一个完整算子
10. `10_kv_cache/`
   - 理解 decode 时的 cache append / update
11. `11_gemm/`
   - 理解 tiling、数据复用、shared memory 和矩阵乘法这条主线
12. `12_flash_attention/`
   - 最后再看 FA2，重点理解 `sliced-K -> sliced-Q` 和工作划分，而不是性能

这个顺序的原则是：

- 先看最简单的并行模式
- 再看 reduction 和 shared memory
- 再看 tiling 和矩阵乘法
- 最后再看 attention / flash attention 这类复合算子

---

## 目录说明

- `01_elementwise/`
  - 最基础的逐元素 kernel
- `02_scatter/`
  - `scatter / index_add`
  - 以不规则写和原子冲突为入口理解 scatter、contention 和 atomic updates
- `03_embedding/`
  - `embedding lookup / gather`
  - 以不规则访存为入口理解 gather、cache、coalescing 和 memory-bound kernel
- `04_softmax/`
  - 以 softmax 为入口理解 reduction、shared memory、同步和 block 内协作
- `05_layernorm/`
  - `layernorm / rmsnorm`
  - 以 row-wise normalization 为入口理解 reduction、方差/均方根统计和 memory-bound kernel
- `06_rope/`
  - rotary positional embedding
  - 以旋转位置编码为入口理解 LLM 高频小算子
- `07_online_softmax/`
  - 以 `(m, l)` 状态合并为入口理解 online softmax、warp-level reduction 和通往 FlashAttention 的桥
- `08_fused_rmsnorm/`
  - `residual + rmsnorm`
  - 以小算子 fusion 为入口理解 LLM block 中的高频 memory-bound kernel
- `09_attention/`
  - 最小 attention kernel
  - 把 `gemm / softmax / online softmax` 真正串起来
- `10_kv_cache/`
  - `KV cache append / update`
  - 以 decode 阶段 cache 写入为入口理解 LLM 推理状态更新
- `11_gemm/`
  - matrix multiply / tiling / shared memory reuse
  - 分开对比 `CUDA core` tiled GEMM 和 `Tensor Core / WMMA` GEMM
- `12_flash_attention/`
  - `FlashAttention`
  - 以 `sliced-K -> sliced-Q` 为入口理解 attention kernel 的现代并行切分
  - 重点是 FA2 的工作划分与循环理解，不追求教学版 `.cu` 的性能领先

## 当前范围

从“常见 CUDA / LLM kernel 类型”的覆盖角度看，这个目录目前已经够用了。

这里的原则是：

- 不追求把所有 CUDA 题型做一遍
- 而是给最常见的几类 kernel 各放一个最小学习用例
- 后面如果继续补，也更应该是打磨说明，而不是继续横向扩目录
