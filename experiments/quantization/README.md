# Quantization

这个目录当前只做一件事：

- 用 Hugging Face backend 跑真实 `Qwen3-0.6B`
- 用 `int4` 权重量化
- 用 `CUDA Graph` 跑 decode
- 只测 decode 阶段吞吐

这里不做训练侧量化，也不做复杂论文复现。  
第一步只回答一个最直接的问题：

- 同样是 HF backend，`Qwen3-0.6B` 用 `int4` 权重量化并接上 `CUDA Graph` 之后，decode 阶段吞吐是多少

## 当前测得结果

当前结果统一在 `gpu7` 上测，避免不同 GPU 上的背景负载把数据混在一起。

### HF int4 + CUDA Graph

当前 `int4` 脚本使用的是：

- `bitsandbytes` 4-bit
- `nf4`
- `bnb_4bit_compute_dtype=torch.float16`
- `bnb_4bit_use_double_quant=False`

在 `gpu7` 上连续运行 `hf_int4_cuda_graph_decode.py` 5 次，得到：

- run1: `299.348 tok/s`
- run2: `297.422 tok/s`
- run3: `291.662 tok/s`
- run4: `293.450 tok/s`
- run5: `291.796 tok/s`

5 次平均 decode 吞吐是：

- `294.736 tok/s`

当前可以把 `~295 tok/s` 视为这条最新 `HF + bitsandbytes int4 + CUDA Graph` 路径更稳定的单 batch decode 吞吐。

### HF bf16 + CUDA Graph

为了做同口径对照，也在 `gpu7` 上连续运行 `hf_bf16_cuda_graph_decode.py` 5 次，得到：

- run1: `262.010 tok/s`
- run2: `262.253 tok/s`
- run3: `259.160 tok/s`
- run4: `262.457 tok/s`
- run5: `257.755 tok/s`

5 次平均 decode 吞吐是：

- `260.727 tok/s`

所以在同一张 `gpu7` 上，当前最佳 `int4` 路径已经超过 `bf16`，大约快 `13.1%`。

## 关键发现

这次最重要的发现不是简单一句“`fp16` 比 `bf16` 快”，而是：

- 模型主路径里的 hidden states 本来就是 `torch.float16`
- `bitsandbytes.nn.Linear4bit.forward()` 会先把输入 `x` cast 到 `compute_dtype`
- 4-bit matmul 算完之后，再把结果 cast 回原始 `inp_dtype`

对应代码在 `bitsandbytes/nn/modules.py` 里，核心逻辑是：

```python
inp_dtype = x.dtype
if self.compute_dtype is not None:
    x = x.to(self.compute_dtype)
...
return bnb.matmul_4bit(x, weight, bias=bias, quant_state=quant_state).to(inp_dtype)
```

而我们实际 hook 第一层 `Linear4bit` 时，看到：

- `Linear4bit` 的输入 dtype 是 `torch.float16`
- embedding 和 norm 等周边张量 dtype 也是 `torch.float16`

这就意味着：

- `bnb_4bit_compute_dtype=torch.bfloat16` 时，每层都在做 `fp16 -> bf16 -> fp16`
- `bnb_4bit_compute_dtype=torch.float16` 时，这条往返 cast 基本消失

## 为什么 `bf16 compute` 会多出很多 `unrolled_elementwise_kernel`

`nsys` 对比说明，真正塌下去的不是 4-bit GEMM 本体，而是大量辅助的 elementwise 路径。

从 `double_quant=False + bf16 compute` 到 `double_quant=False + fp16 compute`：

- `cudaGraphLaunch_v10000` 平均从 `4.034869 ms/launch` 降到 `2.829468 ms/launch`
- 图内 kernel node 从 `2158` 降到 `1766`
- 图内 kernel 总时间从 `267.852 ms` 降到 `214.328 ms`

按 kernel 名看，最夸张的变化是：

- `unrolled_elementwise_kernel`: `65.084 ms -> 13.486 ms`
- `kgemm_4bit_inference_naive`: `51.279 ms -> 50.969 ms`

这说明：

- `bf16 compute` 版本里大量额外开销并不主要来自 4-bit GEMM 本体
- 真正的问题是 dtype 往返和由此带来的额外 elementwise / cast 风格 kernel
- `fp16 compute` 路径把这部分大幅压缩了

另外，`bitsandbytes` 的 CUDA backend 本身就对不同 dtype 走不同 kernel：

- `torch.float16` 走 `cgemm_4bit_inference_naive_fp16`
- `torch.bfloat16` 走 `cgemm_4bit_inference_naive_bf16`

所以这里不仅是“少了 cast”，也是 backend 对 `fp16` 这条 4-bit 路径优化得更成熟。

## 当前结论

当前这条实验线里，`int4` 要想在 decode 吞吐上赢 `bf16`，至少要满足两点：

- `bnb_4bit_use_double_quant=False`
- `bnb_4bit_compute_dtype=torch.float16`

在这个配置下，`int4` 的收益不再只是 memory footprint，而是同样能在 decode throughput 上超过 `bf16`。

## 当前文件

- `hf_int4_cuda_graph_decode.py`
  - 用 `bitsandbytes` 4-bit 量化加载 Qwen3-0.6B
  - 用 `StaticCache` 固定 decode 形状
  - 用 `CUDA Graph` replay 跑 decode
  - 只统计 graph 路径下的 decode tok/s
- `hf_bf16_cuda_graph_decode.py`
  - 保持同一套 prompt、decode 步数和 graph 流程
  - 只把模型加载改成 `bf16`
  - 用来和 `int4` 做同口径对比
- `nsys-reps/`
  - 保存 `bf16` 和 `int4` 的 `nsys` profile 结果
  - 记录 node 粒度的 graph launch 和 graph 内 kernel 分析

## 运行

```bash
conda run -n aisys python experiments/quantization/hf_int4_cuda_graph_decode.py
```

## 当前边界

- 只看 HF backend
- 只看 weight-only int4 quantization
- 只看 `CUDA Graph` decode throughput
- 不比较训练侧 quant
- 不深入量化 kernel 实现
