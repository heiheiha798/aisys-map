# NCU Notes: Triton Flash Attention

这份笔记记录当前目录里的 Triton kernel：

- `flash_attention_kernel`

并和 `experiments/cuda_kernels/12_flash_attention/ncu_notes.md` 的 CUDA 教学版 FlashAttention 对照。

## 先说清楚：这里不能直接比绝对时延

当前 Triton 脚本的样本是：

- `seq_len = 128`
- `head_dim = 64`

CUDA 目录里的教学样本是：

- `seq_len = 512`
- `head_dim = 32`

所以这里最该比较的是 profile 画像，不是绝对时延。

## Profiling 命令

```bash
/usr/local/cuda-12.4/bin/ncu \
  --csv \
  --page raw \
  --target-processes all \
  --kernel-name-base demangled \
  --launch-count 1 \
  --metrics \
    gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,\
launch__block_size \
  --kernel-name regex:flash_attention_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/triton_kernels/12_flash_attention/flash_attention.py
```

## NCU 关键指标

| impl | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `flash_attention_kernel` | `4` | `128` | `7.04` | `1.65` | `0.78` | `0.85` | `1.56` | `8.32` |
| CUDA | `flash_attention_kernel` | `256` | `64` | `149.54` | `15.66` | `15.66` | `15.80` | `2.48` | `8.33` |

## 最直接的结论

- Triton 和 CUDA 这两版都不适合被当成工程性能样例。
- 它们更像两个不同层次的 teaching kernel：
  - Triton 版更偏最小 online softmax 数据流
  - CUDA 版更偏 FA2 工作划分解释

## 和 CUDA 版对比怎么看

共同点非常明显：

- occupancy 都低
- 都不是高吞吐画像
- 都不应该被误读成“FlashAttention 本来就这样慢”

但 Triton 版还要更极端一些：

- `grid = 4`
- `mem % = 1.65`
- `compute % = 0.78`

这基本就是在说：

- 当前 Triton 样本太小
- launch 规模太小
- NCU 主要看到的是一个非常轻的学习用例

CUDA 版虽然也远不是生产实现，但至少：

- `grid = 256`
- profile 信息量更高一些

所以 CUDA 笔记里能更具体地讨论：

- 为什么教学版 FlashAttention 仍然和真正工程版差很远

## 这份对比最该记住的点

1. Triton 和 CUDA 这里都不是“性能答案”，而是“帮助建立 FlashAttention 直觉的学习样例”。
2. 当前 Triton 版的 NCU 更主要说明它是一个很小的 teaching kernel，而不是说明 FlashAttention 算法本身的性能上限。
3. 这也进一步支持当前仓库的定位：FlashAttention 在这里主要是帮助理解 online softmax 和 tiled attention 数据流。
