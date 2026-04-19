# CUDA Kernels

这个目录放最基础的原生 CUDA C++ kernel 实验。

根目录只放通用信息。  
具体实验按类型拆到子目录里，例如：

- `elementwise/`
- `softmax/`
- `online_softmax/`
- `gemm/`
- `layernorm/`
- `embedding/`
- `scatter/`
- `attention/`
- `rope/`
- `kv_cache/`
- `fused_rmsnorm/`
- `flash_attention/`

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

---

## 当前目录建议

- `elementwise/`
  - 最基础的逐元素 kernel
- `softmax/`
  - 以 softmax 为入口理解 reduction、shared memory、同步和 block 内协作
- `online_softmax/`
  - 以 `(m, l)` 状态合并为入口理解 online softmax、warp-level reduction 和通往 FlashAttention 的桥
- `gemm/`
  - matrix multiply / tiling / shared memory reuse
  - 分开对比 `CUDA core` tiled GEMM 和 `Tensor Core / WMMA` GEMM
- `layernorm/`
  - `layernorm / rmsnorm`
  - 以 row-wise normalization 为入口理解 reduction、方差/均方根统计和 memory-bound kernel
- `embedding/`
  - `embedding lookup / gather`
  - 以不规则访存为入口理解 gather、cache、coalescing 和 memory-bound kernel
- `scatter/`
  - `scatter / index_add`
  - 以不规则写和原子冲突为入口理解 scatter、contention 和 atomic updates
- `attention/`
  - 最小 attention kernel
  - 把 `gemm / softmax / online softmax` 真正串起来
- `rope/`
  - rotary positional embedding
  - 以旋转位置编码为入口理解 LLM 高频小算子
- `kv_cache/`
  - `KV cache append / update`
  - 以 decode 阶段 cache 写入为入口理解 LLM 推理状态更新
- `fused_rmsnorm/`
  - `residual + rmsnorm`
  - 以小算子 fusion 为入口理解 LLM block 中的高频 memory-bound kernel
- `flash_attention/`
  - `FlashAttention`
  - 以 `sliced-K -> sliced-Q` 为入口理解 attention kernel 的现代并行切分
  - 重点是 FA2 的工作划分与循环理解，不追求教学版 `.cu` 的性能领先

## 后续规划

后面如果只按 `LLM 常见 kernel` 往下做，建议顺序固定为：

1. `attention/`
   - 先做最小 attention kernel
   - 把 `gemm / softmax / online softmax` 真正串起来
2. `rope/`
   - 补齐 LLM 里高频出现的旋转位置编码
3. `kv_cache/`
   - 以 append / update 为入口理解 decode 阶段的 cache 写入
4. `fused_rmsnorm/`
   - 以 `residual + rmsnorm` 为入口理解 LLM 小算子的 fusion
5. `flash_attention/`
   - 用一个学习用例收束 attention 优化主题，重点理解 FA2 的 `sliced-Q`

这里的原则是：

- 只做和 LLM block / LLM 推理直接相关的 kernel
- 不再继续扩更多通用 CUDA 题目
