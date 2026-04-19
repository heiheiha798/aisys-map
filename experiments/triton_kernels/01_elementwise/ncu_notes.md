# NCU Notes: Triton Elementwise Add

这份笔记只想回答一个问题：

- 对最简单的 elementwise kernel，Triton 相比手写教学版 CUDA，到底说明了什么？

结论先写在前面：

- 这个例子里，Triton 不只是代码更短，实际也更快。
- 对初学者来说，这正说明 Triton 往往是一个很现实的 fast path：
  - 更容易写对
  - 更容易快速拿到可运行结果
  - 对标准小算子，经常也能直接拿到不差的性能
- 代价不是“不能优化”，而是：
  - 你失去了一部分手写 CUDA 的细粒度控制
  - 比如 block/thread 映射、shared memory 组织、warp-level 细节、极限调参空间

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
  --kernel-name regex:elementwise_add_kernel \
  /data/home/tianjianyang/miniconda3/envs/aisys/bin/python \
  experiments/triton_kernels/01_elementwise/elementwise_add.py
```

CUDA 对照见：

- `experiments/cuda_kernels/01_elementwise/ncu_notes.md`

## 关键指标

| impl | kernel | grid | block | dur(us) | mem % | compute % | l1/tex % | l2 % | occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton | `elementwise_add_kernel` | `256` | `128` | `2.50` | `21.90` | `1.11` | `2.84` | `9.30` | `13.23` |
| CUDA | `elementwise_add_kernel` | - | - | `10.75` | `79.78` | `8.66` | - | - | `83.06` |

## 最重要的观察

先看最不容易误读的数字：

- Triton: `2.50 us`
- CUDA: `10.75 us`

也就是：

- 在这个最小逐元素样例里，Triton 明显快于这份教学版手写 CUDA

这件事本身就已经足够说明问题：

- 对于非常标准的、小而清晰的 kernel
- 你并不一定需要先去手搓 CUDA
- Triton 往往就是更合理的起点

## 这个对比真正说明了什么

### 1. Triton 对初学者通常是“简单且有效”的第一条路

这个 kernel 的逻辑只有一句：

```text
out[i] = x[i] + y[i]
```

对应到 Triton 代码里，也几乎就是最直接的翻译：

- 一个 program 处理一段连续元素
- `tl.load`
- 相加
- `tl.store`

这种情况下，Triton 的优势非常直接：

- 写法短
- 索引关系直白
- correctness 好对
- 性能也往往已经够好，甚至比教学版手写 CUDA 更好

所以如果你的目标是：

- 先把一个标准小算子跑起来
- 先得到一版正确、清楚、可 profile 的 kernel

那 Triton 往往比“先上手搓 CUDA”更像合理起点。

### 2. 代价是控制粒度更粗

这里也必须把代价说清楚。

Triton 更快，不代表它把 CUDA 这条路“替代掉了”。

你放弃的东西主要是：

- 对 block / thread 级映射的显式控制
- 对 shared memory 布局的显式控制
- 对 warp-level primitive 的显式控制
- 对极限性能调参空间的完全掌控

换句话说：

- Triton 更像高一层的 kernel 编程
- 手写 CUDA 更像所有底层旋钮都在你手里

所以这个例子的正确理解不是：

- “Triton 永远比 CUDA 好”

而是：

- “对标准小算子，Triton 往往提供了一个更简单、也经常更有效的 fast path”

## 为什么这里不要被 throughput 百分比带偏

如果只看这组百分比：

- Triton `mem % = 21.90`
- CUDA `mem % = 79.78`

很容易误读成：

- “Triton 没有把机器吃满，所以它应该更差”

但这个判断在这里不成立。

因为这次最关键的事实是：

- Triton 的总时延更短

也就是说，真正该先看的顺序应该是：

1. 先看 kernel 完成同样工作花了多久
2. 再看 profile 百分比是在解释这个时延画像

这里的正确理解更接近：

- 两者都在做一个低算术强度、访存主导的 elementwise kernel
- 但这份教学版手写 CUDA，并没有把“底层可控性”真正兑现成更好的结果
- Triton 反而用更短的代码拿到了更好的时延

## 这类 kernel 的本质并没有变

即使 Triton 这次更快，这个算子的本质仍然非常稳定：

- 读两次
- 写一次
- 每个元素只做一次加法

所以它仍然是典型的：

- low arithmetic intensity
- memory-dominated
- 很适合作为 fusion 候选

也就是说，Triton 赢下这个例子，并不是因为它把 elementwise 变成了 compute-heavy kernel。

真正的意思只是：

- 对这种标准模板题，Triton 往往能更快地把“一个不错的答案”交出来

## 这份对比最该记住的点

1. 这个例子里，Triton 相比教学版手写 CUDA，同时赢在了代码复杂度和时延。
2. 这说明对初学者来说，Triton 往往是写标准小 kernel 的更好起点。
3. Triton 的代价不是不能做，而是控制粒度更粗，底层旋钮更少。
4. 即便实现方式不同，这个算子的本质仍然没变：它首先是一个访存主导的 elementwise kernel。
