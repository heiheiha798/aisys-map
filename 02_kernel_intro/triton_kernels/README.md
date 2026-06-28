# Triton Kernels

这个 README 只服务于：

- `02_kernel_intro/triton_kernels/`

也就是：

- 介绍 Triton 是什么
- 说明这个目录里的例子在看什么
- 给出这个目录自己的阅读顺序

不讨论别的目录。

## Triton 是什么

Triton 可以先粗略理解成：

- 用 Python 写 GPU kernel

更具体一点，它提供的是一种介于：

- 高层 PyTorch op
- 底层 CUDA C++

之间的写法。

它最适合表达的是：

- 一个 program 处理一个 tile
- tile 怎么映射到张量
- load / store / reduction / matmul 怎么组合

所以看 Triton 代码时，最重要的直觉通常不是：

- 一个线程在干什么

而是：

- 一个 program 在处理哪一块数据

## 这个目录在放什么

这个目录放的是一组教学版 Triton kernel。

这些例子的共同特点是：

- 用 Python + Triton 写
- 都能直接运行
- 都带 correctness check
- 优先把数学、数据流和 program 映射写清楚

所以这里更适合回答：

- Triton 代码一般长什么样
- 一个 program 怎样映射到一行、一块、一个 tile
- 常见 LLM 小算子怎样改写成 Triton kernel

而不是：

- 单个 kernel 的极限性能怎么压
- 所有实现都要逼近库级最优

## 运行方式

这里统一直接用 Python 运行，Triton 会在第一次调用时自动 JIT 编译。

默认使用 `aisys` conda 环境：

```bash
conda run -n aisys python 02_kernel_intro/triton_kernels/01_elementwise/elementwise_add.py
```

如果你已经激活了环境，也可以直接：

```bash
python 02_kernel_intro/triton_kernels/01_elementwise/elementwise_add.py
```

## 这个目录怎么读

这里的编号已经按从简单到复杂排好了：

1. `01_elementwise/`
2. `02_scatter/`
3. `03_embedding/`
4. `04_softmax/`
5. `05_layernorm/`
6. `06_rope/`
7. `07_online_softmax/`
8. `08_fused_rmsnorm/`
9. `09_attention/`
10. `10_kv_cache/`
11. `11_gemm/`
12. `12_flash_attention/`

更推荐的阅读方式是：

1. 先从 `01` 开始，看最基础的 Triton 语法：
   - `@triton.jit`
   - `program_id`
   - `arange`
   - `load/store`
   - `mask`
2. 再看 `04`、`05`：
   - 理解 row-wise reduction
   - 理解 `tl.max`、`tl.sum`、`tl.rsqrt`
3. 再看 `06`、`07`、`08`：
   - 看逐元素数学变换
   - 看 online 状态
   - 看多 kernel 组织方式
4. 最后看 `09`、`11`、`12`：
   - attention
   - matmul tile
   - flash attention

## 这个目录最值得关注什么

如果你把这些例子从前往后看，最值得建立的是三种直觉：

### 1. Program 视角

Triton 最核心的问题通常是：

- 一个 program 负责哪一块数据

例如：

- 一段连续元素
- 一整行
- 一块二维 tile
- 一个 `(row, block_col)` 逻辑块

### 2. 地址映射视角

很多 Triton 代码的核心其实不是数学，而是：

- 怎么从逻辑坐标推到真实地址

比如：

- `base + offsets`
- `row * stride + offsets`
- `offs_m[:, None]` 和 `offs_n[None, :]`
- 多维逻辑坐标压平再恢复

### 3. 局部状态视角

越往后看，kernel 内维护的局部状态会越来越复杂：

- `01` 只有简单向量
- `04` 开始出现 row-wise reduction
- `07` 开始出现跨 tile 状态
- `09` 开始显式保留整行中间结果
- `12` 开始同时维护多个不同形状的状态张量

## 当前目录包含的脚本

- `01_elementwise/elementwise_add.py`
- `02_scatter/index_add_rows.py`
- `03_embedding/row_gather.py`
- `04_softmax/row_softmax.py`
- `05_layernorm/row_layernorm.py`
- `05_layernorm/row_rmsnorm.py`
- `06_rope/triton_rope.py`
- `07_online_softmax/row_softmax_online.py`
- `08_fused_rmsnorm/fused_residual_rmsnorm.py`
- `09_attention/triton_attention.py`
- `10_kv_cache/kv_cache_append_update.py`
- `11_gemm/triton_gemm.py`
- `12_flash_attention/flash_attention.py`

## 这个目录最适合回答什么问题

- Triton 最基础的语法长什么样
- 怎么从最小 kernel 一步步走到 attention / flash attention
- 为什么很多 LLM 小算子适合按 tile / row / block 的方式来理解
- Triton 代码里常见的索引、mask、reduction、局部状态组织是怎样的

如果你只是希望：

- 看过之后知道 Triton kernel 大概怎么写
- 知道常见 LLM 小算子在 Triton 里会长成什么样

那这个目录就已经够用了。

## 下一步看真实项目

这个目录是教学版 kernel，不追真实模型的完整执行路径。

如果已经熟悉这里的基础写法，下一步可以看：

- [`05_case_studies/flash-deepseek-v2-lite/`](../../05_case_studies/flash-deepseek-v2-lite)

它是 DeepSeek-V2-Lite decode path 的 Triton optimization case study，更适合观察：

- small GEMV 和 batched/grouped GEMM-like kernel 的取舍
- MoE route grouping 和 fixed topk reduce
- attention decode path 里的 projection、RoPE、KV cache 和 online softmax
- `nsys` / `ncu` 如何驱动 kernel 改动保留或撤回
