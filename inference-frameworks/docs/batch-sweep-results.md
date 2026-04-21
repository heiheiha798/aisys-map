# Batch Sweep Results

这次只记录 `decode throughput`，不记录 `prefill throughput`。

实验对象：

- `nano-vllm`
- `llama.cpp`

实验设置：

- GPU: `gpu7`
- Model:
  - nano-vllm: `/data/pretrained_models/Qwen3-0.6B`
  - llama.cpp: `/data/home/tianjianyang/models/ggufs/Qwen3-0.6B-f16.gguf`
- prompt length: `10` tokens
- decode tokens: `100`
- batch sizes: `1, 2, 4, 8, 16`
- 口径：`decode-only`
  对 `nano-vllm`，首个 `prefill` step 不计入吞吐；统计的是后续 decode 阶段。

## Decode Throughput

| batch size | nano-vllm tok/s | llama.cpp tok/s |
| --- | ---: | ---: |
| 1 | 423.93 | 466.55 |
| 2 | 751.70 | 801.16 |
| 4 | 1498.80 | 1520.17 |
| 8 | 2977.26 | 2779.08 |
| 16 | 5236.88 | 4551.29 |

## 结果解读

可以先给出结论：

- 小 batch 下，`llama.cpp` 仍然更强。
- batch 增大后，`nano-vllm` 增长更快。
- 在这次修复了 `llama.cpp` 的 GPU-side sampling 之后，`llama.cpp` 比旧结果更快，但 `bsz=8` 和 `bsz=16` 仍然落后于 `nano-vllm`。
- 到 `bsz=16`，`nano-vllm` 仍然领先约 `15.1%`。

## Profile-Based Analysis

这里不做泛泛而谈，只基于 `nsys` 和代码路径给出判断。

使用的 profile 文件：

- `nano-vllm`: `inference-frameworks/nsys-reps/nanovllm_b16_graph.nsys-rep`
- `llama.cpp` 原始版本: `inference-frameworks/nsys-reps/llamacpp_b16_graph.nsys-rep`
- `llama.cpp` 修复后版本: `inference-frameworks/nsys-reps/llamacpp_b16_backend_sampling_greedy.nsys-rep`

### 1. 先修掉了一个真实的 `llama.cpp` 问题：raw logits DtoH

原始 `llama.cpp` 路径并不是 GPU-side sampling，而是把整块 logits 从 device 拷回 host，再在 host 侧继续采样相关逻辑。

`nsys` 证据：

- 原始 `llama.cpp` 的 DtoH 总量是 `982.114 MB`
- 其中有 `101` 次 `CUDA memcpy Device-to-Host`
- 每次拷贝大小是 `9.724 MB`

这正是原来看到的那条大块 DtoH copy。

修复后：

- `llama.cpp` 改成了 backend sampling
- benchmark 路径进一步收敛成 `greedy` backend sampling
- `llama.cpp` graph 里不再把 sampler 输入 logits 误当成 sampler 输出 logits 回传

修复后的 `nsys` 证据：

- 新版 `llama.cpp` 的 DtoH 总量只剩 `0.006 MB`
- 不再存在 `9.724 MB` 级别的 logits 回传
- `bsz=16` decode throughput 从旧结果 `4061.33 tok/s` 提升到 `4551.29 tok/s`

所以，原始 `llama.cpp` 的确有一个明显问题，而且已经被修掉了。

但修掉之后，`llama.cpp` 依然比 `nano-vllm` 慢。这说明当前剩下的差距，主因已经不是那条 DtoH。

### 2. 当前差距不是 `cudaGraphLaunch` overhead

如果差距主要来自 CUDA graph launch，本应看到 `llama.cpp` 的 graph launch 更重；但实际不是。

`nsys` 数据：

- `nano-vllm`:
  - `cudaGraphLaunch_v10000`: `60.23 ms / 103 calls`
  - 平均约 `584.8 us / launch`
- 新版 `llama.cpp`:
  - `cudaGraphLaunch_v10000`: `10.24 ms / 99 calls`
  - 平均约 `103.4 us / launch`

也就是说，`nano-vllm` 的 graph launch API 时间反而更大，但它整体更快。

因此，不能把当前差距归因于 graph launch overhead。

同理，修复后的 `llama.cpp` 也已经没有大块 DtoH：

- `nano-vllm` 的 DtoH 总量是 `0.014 MB`
- 新版 `llama.cpp` 的 DtoH 总量是 `0.006 MB`

两边现在都是 token 级别的小回传，不再是决定性差异。

### 3. 当前差距主要来自 CUDA graph 内部 kernel

在 `bsz=16`、`decode_tokens=100` 时，总 decode token 数是 `1600`。

- `nano-vllm`: `5236.88 tok/s`
- `llama.cpp`: `4551.29 tok/s`

对应总 decode 时间约为：

- `nano-vllm`: `305.5 ms`
- `llama.cpp`: `351.5 ms`

差了约 `46 ms`。摊到 `100` 个 decode step，就是每步约 `0.46 ms` 的差距。

而 graph launch 的方向是反的：`llama.cpp` 的 launch 更轻。因此，这 `0.46 ms / step` 主要只能落在 graph 内部 kernel 执行时间上。

`cudaStreamSynchronize` 也支持这个判断：

- `nano-vllm`: `18.58 ms`
- 新版 `llama.cpp`: `347.32 ms`

这里不能把 `cudaStreamSynchronize` 本身当成根因，它只是症状。它意味着 host 更长时间在等 `llama.cpp` 的 GPU 工作完成，也就是 graph 内部 kernel 路径更慢。

### 4. graph 内部到底慢在哪些 kernel

`nano-vllm` 的主力 kernel 组合：

- CUTLASS BF16 Tensor Core GEMM
- `flash_fwd_splitkv_kernel`
- `flash_fwd_splitkv_combine_kernel`
- Triton fused norm / softmax / argmax

`llama.cpp` 新版的主力 kernel 组合：

- `mul_mat_f<__half2...>` 占比最高
- `flash_attn_ext_f16`
- `argmax_f32`
- `convert_unary<__half, float>`
- `convert_unary<float, __half>`
- `rms_norm_f32`
- `rope_neox`

几个关键观察：

- `argmax_f32` 不是主因
  - 它有成本，但只占新版 `llama.cpp` GPU kernel 时间的约 `6.2%`
  - 即使把 sampling kernel 完全抹掉，也不足以解释全部差距
- `convert_unary` 明显存在
  - `convert_unary<__half, float>` 和 `convert_unary<float, __half>` 合计约 `6.5%`
  - 这说明 `llama.cpp` 图内部存在额外的 dtype 转换负担
- 真正的大头还是 matmul 和 attention 路径本身
  - `nano-vllm` 压在更现代的 CUTLASS BF16 GEMM + FlashAttention + Triton fused kernel 上
  - `llama.cpp` 则主要依赖 GGML CUDA 的 `mul_mat_f` + `flash_attn_ext_f16` 路径

因此，当前结果更像是“graph 内部 kernel 质量和 kernel 组合差异”，而不是框架外围开销差异。

### 5. 为什么 `nano-vllm` 没有 `K/V` 先变 `F32` 再 cast 回 `F16` 的问题

这不是因为 `nano-vllm` 完全不用 `fp32`，而是因为两套系统的 dtype 语义不同。

`nano-vllm`：

- 在模型初始化时，直接把默认 dtype 设成 `hf_config.dtype`
- QKV projection 走 `torch.nn.functional.linear`
- decode attention 直接走 `flash_attn_with_kvcache`
- prefill attention 走 `flash_attn_varlen_func`

它也会在局部用 `fp32` 做数值稳定计算，例如：

- RMSNorm 里 `x.float()`，算完后立刻 `to(orig_dtype)`
- Rotary 里 `x.float()`，算完后立刻 `to(x.dtype)`

也就是说：

- `nano-vllm` 会局部升到 `fp32`
- 但不会把 Q/K/V 这种中间张量长期实体化成 `F32 tensor`
- attention kernel 看到的通常仍然是半精度张量

`llama.cpp / GGML` 则不同：

- `ggml_mul_mat` 的结果类型默认就是 `GGML_TYPE_F32`
- `ggml_get_rows` 的结果类型默认也是 `GGML_TYPE_F32`
- 所以即使权重和输入本身是 `F16`，中间 Q/K/V 结果也可能先落成 `F32 tensor`
- 到 `flash_attn_ext` 前才发现 kernel 更适合半精度，于是补 `ggml_cast(..., F16)`

所以 `convert_unary` 不是“参数加载错成了 fp32”，而是 GGML 图的中间 dtype 规则决定的。

这也是为什么：

- `nano-vllm` 更像“半精度张量流 + 局部 fp32 计算”
- `llama.cpp` 更像“很多中间节点直接是 `F32 tensor`，到需要半精度 kernel 时再 cast 回去”

### 6. 当前可以下的结论

基于修复后的 profile，可以给出更准确的结论：

- 原始 `llama.cpp` 确实有一个明显的 raw logits DtoH 问题，这个问题已经修掉
- 修复后，当前差距不主要来自 host 侧 overhead
- `cudaGraphLaunch` 不是主因
- 当前差距主要来自 CUDA graph 内部 kernel 路径
- 其中最重要的是 `llama.cpp` 的 matmul / attention / convert 这组 kernel 路径不如 `nano-vllm` 的 CUTLASS + Triton + FlashAttention 路径高效

## 可视化建议

这组结果非常适合可视化，而且做图之后会比表格更直观。

推荐保留：

- 一张 `batch size -> decode tok/s` 的折线图

理由：

- 这张图能直接看出两条曲线的交叉点
- 能直观看到 `nano-vllm` 后段斜率更高
- 适合后续继续叠加更多后端或更多 batch 点

对应图文件放在：

- `inference-frameworks/plot/decode_batch_sweep.pdf`

原始数据和脚本放在：

- `inference-frameworks/plot/data/decode_batch_sweep.json`
- `inference-frameworks/plot/scripts/plot_decode_batch_sweep.py`
