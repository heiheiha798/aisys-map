# NCU Notes: Triton Embedding Gather

这份笔记记录当前目录里的 Triton kernel：

- `row_gather_kernel`

并和 `experiments/02_kernel_intro/cuda_kernels/03_embedding/ncu_notes.md` 的 CUDA 版本对照。

这里同样保留两种输入模式：

- `random`
- `repeated`

## Profiling 命令

默认 `random`：

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
  --kernel-name regex:row_gather_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/02_kernel_intro/triton_kernels/03_embedding/row_gather.py
```

重复 id 模式：

```bash
GATHER_ID_MODE=repeated \
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
  --kernel-name regex:row_gather_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/02_kernel_intro/triton_kernels/03_embedding/row_gather.py
```

## NCU 关键指标

| impl | mode | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `random` | `row_gather_kernel` | `4096` | `128` | `6.40` | `59.06` | `7.55` | `17.33` | `33.27` | `74.43` |
| Triton | `repeated` | `row_gather_kernel` | `4096` | `128` | `4.61` | `43.01` | `10.11` | `21.45` | `43.01` | `55.22` |
| CUDA | `random` | `row_gather_kernel` | - | - | `7.30` | `51.91` | `12.72` | `16.10` | `32.28` | `74.34` |
| CUDA | `repeated` | `row_gather_kernel` | - | - | `6.08` | `34.77` | `15.26` | `20.98` | `34.77` | `69.82` |

## 最直接的结论

- Triton 和 CUDA 的结论完全一致：
  - `repeated` 比 `random` 更快
  - 重复访问能明显改善 cache 行为
  - gather 仍然是典型的 irregular read kernel

## 和 CUDA 版对比怎么看

这组数据最值得记住的不是绝对时延，而是变化趋势：

- Triton: `6.40 us -> 4.61 us`
- CUDA: `7.30 us -> 6.08 us`

也就是说：

- kernel 没变
- 映射方式没变
- 只是 `ids` 分布从分散变成重复

profile 就已经明显变了。

这正是 gather 类 kernel 最重要的现实特征：

- 它们很容易被输入分布影响
- cache hit 的变化会直接改 profile
- 算法本身几乎没有“重计算”可言

## Triton 和 CUDA 的共同画像

无论是 Triton 还是 CUDA：

- `random` 模式下都表现出更高的 memory 压力
- `repeated` 模式下都能从 cache 重用中受益

最稳的结论仍然是：

1. embedding lookup 本质上是 gather，不是 GEMM。
2. 这类 kernel 最该关注的是访存模式，而不是 FLOPS。
3. 如果输入分布变了，profile 也会跟着明显变化。
