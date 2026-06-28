# NCU Notes: Triton Online Softmax

这份笔记记录当前目录里的 Triton kernel：

- `row_softmax_online_kernel`

并和 `02_kernel_intro/cuda_kernels/07_online_softmax/ncu_notes.md` 的 CUDA 版本对照。

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
  --kernel-name regex:row_softmax_online_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  02_kernel_intro/triton_kernels/07_online_softmax/row_softmax_online.py
```

## NCU 关键指标

| impl | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `row_softmax_online_kernel` | `4096` | `128` | `9.34` | `64.15` | `64.15` | `64.15` | `24.66` | `81.64` |
| CUDA | `row_softmax_online_kernel` | - | - | `12.83` | `37.34` | `61.58` | - | `49.83` | `87.16` |

## 最直接的结论

- Triton 和 CUDA 都已经不再是“没把 GPU 喂饱”的状态。
- 两边都更接近一个：
  - compute / sync / memory 混合型 kernel
  - occupancy 很高
  - 明显比普通教学版 softmax 更接近真实高频算子

## 和 CUDA 版对比怎么看

CUDA 笔记里最重要的结论是：

- online softmax 不再像最小 softmax 那样首先暴露 grid 太小问题
- 它已经进入真正的 kernel 结构权衡

Triton 版完全符合这个判断：

- `grid = 4096`
- `occ = 81.64%`
- `mem %` 和 `compute %` 都很高

这说明当前 Triton 版也不是一个简单搬运 kernel。

它已经具备 online softmax 应有的特征：

- 有状态维护
- 有多轮 tile 扫描
- 最后还要回写归一化输出

## Triton 和 CUDA 的共同点

1. 都已经走出了“under-filled demo”阶段。
2. 都是高 occupancy 的 row-wise kernel。
3. 都不适合简单贴一个纯 memory-bound 或纯 compute-bound 标签。

## 这份对比最该记住的点

1. online softmax 的价值不是只换了个公式，而是把 softmax 变成了可在线合并的状态更新问题。
2. 从 profile 上看，Triton 和 CUDA 都给出了类似的结论：这已经是一个真正的混合型 kernel。
3. 这也是它比普通 softmax 更接近 FlashAttention 路线的原因。
