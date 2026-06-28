# Quantization

这个目录只回答一个问题：

- 在 Hugging Face backend 上，真实 `Qwen3-0.6B` 做 4-bit weight-only quantization 并接上 `CUDA Graph` 之后，decode 吞吐能到多少

这里不做训练侧量化，也不做复杂论文复现。

## 结论

当前结果统一在 `gpu7` 上测。

- 最新最佳 `int4 + CUDA Graph` 稳态 decode 吞吐：`294.736 tok/s`
- 对照 `bf16 + CUDA Graph` 稳态 decode 吞吐：`260.727 tok/s`
- 同卡同口径下，当前最佳 `int4` 路径比 `bf16` 快约 `13.1%`

当前最佳 `int4` 配置是：

- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_use_double_quant=False`
- `bnb_4bit_compute_dtype=torch.float16`

## 为什么这版 int4 会翻过 bf16

关键点不只是“`fp16` 比 `bf16` 快”，而是这条 `bitsandbytes` 4-bit 路径本身存在 dtype 路径差异。

我们实际检查到：

- 模型主路径里的 hidden states 本来就是 `torch.float16`
- `bitsandbytes.nn.Linear4bit.forward()` 会先把输入 `x` cast 到 `compute_dtype`
- 4-bit matmul 算完之后，再把结果 cast 回原始 `inp_dtype`

也就是：

- `bnb_4bit_compute_dtype=torch.bfloat16` 时，每层都在做 `fp16 -> bf16 -> fp16`
- `bnb_4bit_compute_dtype=torch.float16` 时，这条往返 cast 基本消失

再加上 backend 本身对两条路径走不同 kernel：

- `torch.float16` 走 `cgemm_4bit_inference_naive_fp16`
- `torch.bfloat16` 走 `cgemm_4bit_inference_naive_bf16`

所以这里不只是“少了 cast”，也是 `bitsandbytes` 对 `fp16` 这条 4-bit 路径优化得更成熟。

## nsys 看到的现象

从 `double_quant=False + bf16 compute` 到 `double_quant=False + fp16 compute`：

- `cudaGraphLaunch_v10000` 平均从 `4.034869 ms/launch` 降到 `2.829468 ms/launch`
- 图内 kernel node 从 `2158` 降到 `1766`
- 图内 kernel 总时间从 `267.852 ms` 降到 `214.328 ms`

最显眼的变化不是 4-bit GEMM 本体，而是大量辅助的 elementwise 路径被压掉了：

- `unrolled_elementwise_kernel`: `65.084 ms -> 13.486 ms`
- `kgemm_4bit_inference_naive`: `51.279 ms -> 50.969 ms`

这说明 `bf16 compute` 版本里额外损耗的主要来源，是 dtype 往返带来的辅助 kernel，而不是 4-bit GEMM 本体突然慢很多。

## 量化收益怎么理解

这里的口径需要收准：

- 量化最确定的收益，是权重显存占用下降
- 量化不等于吞吐一定按位宽比例线性提升

原因是端到端推理不只是读权重，还会碰到：

- activation
- KV cache
- 量化元信息
- 临时 buffer
- dequant / elementwise / dtype cast
- launch overhead 和 execution graph 结构

所以：

- 权重显存可以大体按量化方向缩小
- 吞吐不能按量化比例直接线性换算

更稳的心智模型应该是：

- 量化首先是 memory optimization
- 量化有机会变成 throughput optimization
- 是否真的变快，取决于 backend、kernel 路径、dtype 路径和 execution graph 是否顺

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
conda run -n aisys python experiments/04_inference_system/quantization/hf_int4_cuda_graph_decode.py
```

## 当前边界

- 只看 HF backend
- 只看 weight-only int4 quantization
- 只看 `CUDA Graph` decode throughput
- 不比较训练侧 quant
- 不深入量化 kernel 实现
